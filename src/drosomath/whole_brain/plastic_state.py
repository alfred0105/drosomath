from __future__ import annotations

from dataclasses import dataclass, replace


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional large-model path
        raise RuntimeError(
            "Whole-brain plasticity needs NumPy. Install drosomath with the "
            "FlyWire/whole-brain extra first."
        ) from exc
    return np


@dataclass(frozen=True, slots=True)
class PlasticStateConfig:
    """Configuration for array-backed plastic state."""

    initial_multiplier: float = 1.0
    min_multiplier: float = 0.05
    max_multiplier: float = 3.0
    plastic_fraction: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.min_multiplier < 0.0:
            raise ValueError("min_multiplier must be >= 0")
        if self.max_multiplier < self.min_multiplier:
            raise ValueError("max_multiplier must be >= min_multiplier")
        if not self.min_multiplier <= self.initial_multiplier <= self.max_multiplier:
            raise ValueError("initial_multiplier must be inside multiplier bounds")
        if not 0.0 <= self.plastic_fraction <= 1.0:
            raise ValueError("plastic_fraction must be in [0, 1]")


class SparsePlasticityState:
    """Compact mutable state for millions of anatomical edges.

    The anatomical edge strength/sign is deliberately *not* stored here. It
    stays in the source connectome. Learned memory is kept in compact arrays.
    A deterministic per-edge plasticity score allows staged experiments to
    unlock 5% -> 10% -> 20% without changing which earlier edges were plastic.
    """

    def __init__(
        self,
        edge_count: int,
        *,
        config: PlasticStateConfig | None = None,
    ) -> None:
        if edge_count < 0:
            raise ValueError("edge_count must be >= 0")

        np = _require_numpy()
        self.np = np
        self.edge_count = int(edge_count)
        self.config = config or PlasticStateConfig()

        self.multiplier = np.full(
            self.edge_count,
            self.config.initial_multiplier,
            dtype=np.float32,
        )
        self.usage_ema = np.zeros(self.edge_count, dtype=np.float32)
        self.eligibility = np.zeros(self.edge_count, dtype=np.float32)
        self.stability = np.zeros(self.edge_count, dtype=np.float32)

        rng = np.random.default_rng(self.config.seed)
        self._plastic_scores = rng.random(self.edge_count).astype(np.float32, copy=False)
        self.plastic_mask = self._plastic_scores < self.config.plastic_fraction
        # Most MaleCNS runs leave 95% of edges permanently non-plastic.  Keep
        # their compact index list so lifecycle operations do not scan the
        # complete anatomical graph. ``_state_indices`` starts the same, then
        # retains previously plastic edges if a curriculum later freezes them;
        # that preserves their pending decay if they are unlocked again.
        self._plastic_indices = np.flatnonzero(self.plastic_mask).astype(
            np.int32, copy=False
        )
        self._state_indices = self._plastic_indices.copy()
        self._promoted_overrides = np.empty(0, dtype=np.int32)
        self._retired_overrides = np.empty(0, dtype=np.int32)

    @property
    def plastic_edge_count(self) -> int:
        return int(len(self._plastic_indices))

    @property
    def plastic_indices(self):
        """Sorted indices of currently plastic edges.

        The returned array is read-only by convention. It is rebuilt whenever
        ``set_plastic_fraction`` changes the deterministic mask.
        """
        return self._plastic_indices

    @property
    def lifecycle_indices(self):
        """Edges that may hold non-default persistent plastic state."""
        return self._state_indices

    @property
    def plastic_fraction(self) -> float:
        return float(self.config.plastic_fraction)

    def set_plastic_fraction(self, fraction: float) -> dict[str, int | float]:
        """Deterministically unlock/freeze edges while preserving learned values."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        before = self.plastic_edge_count
        self.plastic_mask[:] = self._plastic_scores < float(fraction)
        self._plastic_indices = self.np.flatnonzero(self.plastic_mask).astype(
            self.np.int32, copy=False
        )
        # A formerly plastic edge can retain learned/trace state while frozen.
        # Keep it in the lifecycle set so reducing and later increasing the
        # fraction remains equivalent to dense-array decay semantics.
        self._state_indices = self.np.union1d(
            self._state_indices, self._plastic_indices
        ).astype(self.np.int32, copy=False)
        self._apply_allocation_overrides()
        self.config = replace(self.config, plastic_fraction=float(fraction))
        after = self.plastic_edge_count
        return {
            "fraction": float(fraction),
            "before_edges": before,
            "after_edges": after,
            "newly_unlocked_edges": max(0, after - before),
        }

    def _apply_allocation_overrides(self) -> None:
        """Rebuild the compact current allocation without touching learned arrays."""
        if len(self._retired_overrides):
            self.plastic_mask[self._retired_overrides] = False
        if len(self._promoted_overrides):
            self.plastic_mask[self._promoted_overrides] = True
        self._plastic_indices = self.np.flatnonzero(self.plastic_mask).astype(
            self.np.int32, copy=False
        )
        self._state_indices = self.np.union1d(
            self._state_indices, self._plastic_indices
        ).astype(self.np.int32, copy=False)

    def _validate_edge_indices(self, indices, *, name: str):
        values = self.np.asarray(indices, dtype=self.np.int64)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional index sequence")
        if len(values) and (int(values.min()) < 0 or int(values.max()) >= self.edge_count):
            raise IndexError(f"{name} contains an out-of-range edge index")
        if len(values) != len(self.np.unique(values)):
            raise ValueError(f"{name} contains duplicate edge indices")
        return values.astype(self.np.int32, copy=False)

    def _check_retirement_safety(self, indices, *, protected_stability: float, allow_protected: bool) -> None:
        if not allow_protected and len(indices) and bool(
            (self.stability[indices] >= float(protected_stability)).any()
        ):
            raise ValueError("protected stable edge cannot be retired")

    def exchange_plastic_edges(
        self,
        promote_indices,
        retire_indices,
        *,
        protected_stability: float = 0.75,
        allow_protected: bool = False,
    ) -> dict[str, object]:
        """Atomically exchange equal-sized plastic status on existing edges."""
        promote = self._validate_edge_indices(promote_indices, name="promote_indices")
        retire = self._validate_edge_indices(retire_indices, name="retire_indices")
        if len(promote) != len(retire):
            raise ValueError("budget-preserving exchange requires equal promotion and retirement counts")
        if len(promote) and len(self.np.intersect1d(promote, retire)):
            raise ValueError("an edge cannot be promoted and retired in one exchange")
        if len(promote) and bool(self.plastic_mask[promote].any()):
            raise ValueError("promoted edge is already plastic")
        if len(retire) and bool((~self.plastic_mask[retire]).any()):
            raise ValueError("retired edge is already non-plastic")
        self._check_retirement_safety(
            retire,
            protected_stability=protected_stability,
            allow_protected=allow_protected,
        )
        before = self.plastic_edge_count
        if len(retire):
            self.plastic_mask[retire] = False
        if len(promote):
            self.plastic_mask[promote] = True
        self._retired_overrides = self.np.union1d(self._retired_overrides, retire).astype(self.np.int32, copy=False)
        self._promoted_overrides = self.np.union1d(self._promoted_overrides, promote).astype(self.np.int32, copy=False)
        if len(retire):
            self._promoted_overrides = self.np.setdiff1d(self._promoted_overrides, retire).astype(self.np.int32, copy=False)
        if len(promote):
            self._retired_overrides = self.np.setdiff1d(self._retired_overrides, promote).astype(self.np.int32, copy=False)
        self._plastic_indices = self.np.flatnonzero(self.plastic_mask).astype(self.np.int32, copy=False)
        self._state_indices = self.np.union1d(self._state_indices, self._plastic_indices).astype(self.np.int32, copy=False)
        after = self.plastic_edge_count
        return {
            "plastic_edges_before": before,
            "plastic_edges_after": after,
            "promoted_edges": promote.copy(),
            "retired_edges": retire.copy(),
            "budget_delta": after - before,
        }

    def promote_edges(self, indices):
        """Explicit non-budget-preserving promotion for low-level setup/debug."""
        values = self._validate_edge_indices(indices, name="indices")
        if len(values) and bool(self.plastic_mask[values].any()):
            raise ValueError("promoted edge is already plastic")
        before = self.plastic_edge_count
        self.plastic_mask[values] = True
        self._promoted_overrides = self.np.union1d(self._promoted_overrides, values).astype(self.np.int32, copy=False)
        self._retired_overrides = self.np.setdiff1d(self._retired_overrides, values).astype(self.np.int32, copy=False)
        self._plastic_indices = self.np.flatnonzero(self.plastic_mask).astype(self.np.int32, copy=False)
        self._state_indices = self.np.union1d(self._state_indices, self._plastic_indices).astype(self.np.int32, copy=False)
        return {
            "plastic_edges_before": before,
            "plastic_edges_after": self.plastic_edge_count,
            "promoted_edges": values.copy(),
            "retired_edges": self.np.empty(0, dtype=self.np.int32),
            "budget_delta": self.plastic_edge_count - before,
        }

    def retire_edges(self, indices, *, protected_stability: float = 0.75, allow_protected: bool = False):
        """Explicit non-budget-preserving retirement; anatomy remains active."""
        values = self._validate_edge_indices(indices, name="indices")
        if len(values) and bool((~self.plastic_mask[values]).any()):
            raise ValueError("retired edge is already non-plastic")
        self._check_retirement_safety(values, protected_stability=protected_stability, allow_protected=allow_protected)
        before = self.plastic_edge_count
        self.plastic_mask[values] = False
        self._retired_overrides = self.np.union1d(self._retired_overrides, values).astype(self.np.int32, copy=False)
        self._promoted_overrides = self.np.setdiff1d(self._promoted_overrides, values).astype(self.np.int32, copy=False)
        self._plastic_indices = self.np.flatnonzero(self.plastic_mask).astype(self.np.int32, copy=False)
        self._state_indices = self.np.union1d(self._state_indices, self._plastic_indices).astype(self.np.int32, copy=False)
        return {
            "plastic_edges_before": before,
            "plastic_edges_after": self.plastic_edge_count,
            "promoted_edges": self.np.empty(0, dtype=self.np.int32),
            "retired_edges": values.copy(),
            "budget_delta": self.plastic_edge_count - before,
        }

    def allocation_overrides(self) -> dict[str, object]:
        """Return sparse dynamic allocation overrides for checkpointing."""
        return {
            "promoted_edges": self._promoted_overrides.copy(),
            "retired_edges": self._retired_overrides.copy(),
        }

    def restore_allocation_overrides(self, promoted_edges, retired_edges) -> dict[str, int]:
        """Restore sparse overrides on top of this state's deterministic base mask."""
        promoted = self._validate_edge_indices(promoted_edges, name="promoted_edges")
        retired = self._validate_edge_indices(retired_edges, name="retired_edges")
        if len(promoted) and len(retired) and len(self.np.intersect1d(promoted, retired)):
            raise ValueError("checkpoint allocation overrides overlap")
        base = self._plastic_scores < self.config.plastic_fraction
        self.plastic_mask[:] = base
        self._promoted_overrides = promoted.copy()
        self._retired_overrides = retired.copy()
        self._apply_allocation_overrides()
        return {
            "promoted_edges": int(len(promoted)),
            "retired_edges": int(len(retired)),
            "plastic_edges": self.plastic_edge_count,
        }

    def effective_signed_slice(self, base_signed, start: int, stop: int):
        self._check_slice(start, stop)
        base = base_signed[start:stop]
        if len(base) != stop - start:
            raise ValueError("base_signed does not match plastic-state edge count")
        return base * self.multiplier[start:stop]

    def record_use_slice(
        self,
        start: int,
        stop: int,
        *,
        usage_alpha: float = 0.05,
        eligibility_gain: float = 1.0,
    ) -> int:
        self._check_slice(start, stop)
        if not 0.0 < usage_alpha <= 1.0:
            raise ValueError("usage_alpha must be in (0, 1]")
        if eligibility_gain < 0.0:
            raise ValueError("eligibility_gain must be >= 0")
        if start == stop:
            return 0

        local_mask = self.plastic_mask[start:stop]
        count = int(local_mask.sum())
        if count == 0:
            return 0

        usage = self.usage_ema[start:stop]
        eligibility = self.eligibility[start:stop]
        usage[local_mask] += usage_alpha * (1.0 - usage[local_mask])
        eligibility[local_mask] += eligibility_gain
        return count

    def record_use_indices(
        self,
        edge_indices,
        *,
        usage_alpha: float = 0.05,
        eligibility_gain: float = 1.0,
    ) -> int:
        """Record use for a gathered, non-contiguous edge batch.

        This is equivalent to applying :meth:`record_use_slice` to each
        disjoint source-neuron slice, but lets the sparse simulator update one
        vectorized batch after several neurons fire in the same timestep.
        """
        if not 0.0 < usage_alpha <= 1.0:
            raise ValueError("usage_alpha must be in (0, 1]")
        if eligibility_gain < 0.0:
            raise ValueError("eligibility_gain must be >= 0")
        indices = self.np.asarray(edge_indices, dtype=self.np.int64)
        if len(indices) == 0:
            return 0
        if int(indices.min()) < 0 or int(indices.max()) >= self.edge_count:
            raise IndexError("edge index out of range")
        active = self.plastic_mask[indices]
        if not active.any():
            return 0
        selected = indices[active]
        usage = self.usage_ema[selected]
        self.usage_ema[selected] = usage + usage_alpha * (1.0 - usage)
        self.eligibility[selected] += eligibility_gain
        return int(len(selected))

    def decay_episode(
        self,
        *,
        usage_decay: float = 0.995,
        eligibility_decay: float = 0.90,
        stability_decay: float = 1.0,
    ) -> None:
        for name, value in (
            ("usage_decay", usage_decay),
            ("eligibility_decay", eligibility_decay),
            ("stability_decay", stability_decay),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        indices = self._state_indices
        if len(indices) == 0:
            return
        # Multiplication by one was a 6M-edge no-op in the default setup.
        if usage_decay != 1.0:
            self.usage_ema[indices] *= usage_decay
        if eligibility_decay != 1.0:
            self.eligibility[indices] *= eligibility_decay
        if stability_decay != 1.0:
            self.stability[indices] *= stability_decay

    def clear_eligibility(self) -> None:
        if len(self._state_indices):
            self.eligibility[self._state_indices] = 0.0

    def reset_learning_state(self) -> None:
        self.multiplier.fill(self.config.initial_multiplier)
        self.usage_ema.fill(0.0)
        self.eligibility.fill(0.0)
        self.stability.fill(0.0)
        self._state_indices = self.np.union1d(
            self._plastic_indices,
            self.np.concatenate((self._promoted_overrides, self._retired_overrides)),
        ).astype(self.np.int32, copy=False)

    def summary(self) -> dict[str, float | int]:
        np = self.np
        if self.edge_count == 0:
            return {
                "edge_count": 0,
                "plastic_edge_count": 0,
                "plastic_fraction": float(self.config.plastic_fraction),
                "mean_multiplier": 0.0,
                "mean_usage_ema": 0.0,
                "mean_stability": 0.0,
            }
        return {
            "edge_count": self.edge_count,
            "plastic_edge_count": self.plastic_edge_count,
            "plastic_fraction": float(self.config.plastic_fraction),
            "mean_multiplier": float(np.mean(self.multiplier)),
            "mean_usage_ema": float(np.mean(self.usage_ema)),
            "mean_stability": float(np.mean(self.stability)),
        }

    def _check_slice(self, start: int, stop: int) -> None:
        if start < 0 or stop < start or stop > self.edge_count:
            raise IndexError(
                f"edge slice [{start}:{stop}] is outside 0:{self.edge_count}"
            )
