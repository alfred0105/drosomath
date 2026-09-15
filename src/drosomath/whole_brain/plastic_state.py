from __future__ import annotations

from dataclasses import dataclass


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
    """Configuration for array-backed plastic state.

    ``plastic_fraction`` allows staged experiments such as 5%, 10%, 25%, 50%
    and 100% plastic edges while keeping the anatomical graph fixed.
    """

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

    The anatomical edge strength/sign is deliberately *not* stored here.  It
    stays in the source connectome.  This object only stores learned state, so
    an experiment can always reconstruct the original MaleCNS/FlyWire graph.
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

        if self.config.plastic_fraction >= 1.0:
            self.plastic_mask = np.ones(self.edge_count, dtype=np.bool_)
        elif self.config.plastic_fraction <= 0.0:
            self.plastic_mask = np.zeros(self.edge_count, dtype=np.bool_)
        else:
            rng = np.random.default_rng(self.config.seed)
            self.plastic_mask = (
                rng.random(self.edge_count) < self.config.plastic_fraction
            )

    @property
    def plastic_edge_count(self) -> int:
        return int(self.plastic_mask.sum())

    def effective_signed_slice(self, base_signed, start: int, stop: int):
        """Return sign-preserving effective strengths for ``[start:stop]``."""
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
        """Credit outgoing edges that actually transmitted a presynaptic spike.

        Only plastic edges are updated. ``usage_ema`` approaches one with
        repeated use; ``eligibility`` is a short-lived credit trace consumed by
        a later reward signal.
        """
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

    def decay_episode(
        self,
        *,
        usage_decay: float = 0.995,
        eligibility_decay: float = 0.90,
        stability_decay: float = 1.0,
    ) -> None:
        """Apply slow forgetting once per task/episode instead of every timestep."""
        for name, value in (
            ("usage_decay", usage_decay),
            ("eligibility_decay", eligibility_decay),
            ("stability_decay", stability_decay),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        self.usage_ema *= usage_decay
        self.eligibility *= eligibility_decay
        self.stability *= stability_decay

    def clear_eligibility(self) -> None:
        self.eligibility.fill(0.0)

    def reset_learning_state(self) -> None:
        """Restore an untrained copy without touching the source connectome."""
        self.multiplier.fill(self.config.initial_multiplier)
        self.usage_ema.fill(0.0)
        self.eligibility.fill(0.0)
        self.stability.fill(0.0)

    def summary(self) -> dict[str, float | int]:
        np = self.np
        if self.edge_count == 0:
            return {
                "edge_count": 0,
                "plastic_edge_count": 0,
                "mean_multiplier": 0.0,
                "mean_usage_ema": 0.0,
                "mean_stability": 0.0,
            }
        return {
            "edge_count": self.edge_count,
            "plastic_edge_count": self.plastic_edge_count,
            "mean_multiplier": float(np.mean(self.multiplier)),
            "mean_usage_ema": float(np.mean(self.usage_ema)),
            "mean_stability": float(np.mean(self.stability)),
        }

    def _check_slice(self, start: int, stop: int) -> None:
        if start < 0 or stop < start or stop > self.edge_count:
            raise IndexError(
                f"edge slice [{start}:{stop}] is outside 0:{self.edge_count}"
            )
