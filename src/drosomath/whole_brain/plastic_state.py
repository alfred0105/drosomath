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
        self.config = replace(self.config, plastic_fraction=float(fraction))
        after = self.plastic_edge_count
        return {
            "fraction": float(fraction),
            "before_edges": before,
            "after_edges": after,
            "newly_unlocked_edges": max(0, after - before),
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
        self._state_indices = self._plastic_indices.copy()

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
