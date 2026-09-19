from __future__ import annotations

from dataclasses import dataclass

from .plastic_state import SparsePlasticityState


@dataclass(frozen=True, slots=True)
class BudgetNormalizationStats:
    neurons_touched: int
    edges_scaled: int
    mean_scale: float


@dataclass(slots=True)
class ChannelHomeostasis:
    """Slow task-independent output-rate controller.

    It observes named output channels only; no task labels enter this state.
    The returned gain is bounded and intended to scale directional modulation,
    not to overwrite reward learning.
    """

    target_rate_hz: float = 9.0
    alpha: float = 0.01
    strength: float = 0.05
    ema: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.ema = {} if self.ema is None else dict(self.ema)

    def observe(self, channel_rates: dict[str, float]) -> dict[str, float]:
        gains = {}
        for name, rate in channel_rates.items():
            previous = self.ema.get(name, float(rate))
            value = (1.0 - self.alpha) * previous + self.alpha * float(rate)
            self.ema[name] = value
            error = (self.target_rate_hz - value) / max(self.target_rate_hz, 1e-6)
            gains[name] = max(0.8, min(1.2, 1.0 + self.strength * error))
        return gains


@dataclass(frozen=True, slots=True)
class OutgoingBudgetNormalizer:
    """Keep strengthened pathways from consuming unlimited outgoing strength.

    ``stability_protection`` lets consolidated edges resist normalization. This
    matters for continual learning: otherwise a later task can indirectly erase
    an old memory even when the reward rule itself protects stable edges.
    """

    target_scale: float = 1.0
    strength: float = 0.25
    stability_protection: float = 0.0

    def __post_init__(self) -> None:
        if self.target_scale <= 0.0:
            raise ValueError("target_scale must be > 0")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if not 0.0 <= self.stability_protection <= 1.0:
            raise ValueError("stability_protection must be in [0, 1]")

    def normalize_presynaptic(
        self,
        state: SparsePlasticityState,
        *,
        indptr,
        base_abs=None,
        presynaptic_indices=None,
    ) -> BudgetNormalizationStats:
        """Normalize selected neurons' outgoing plastic multipliers.

        Passing only recently active presynaptic neurons avoids scanning every
        MaleCNS neuron on each reward update. Stable edges can be protected from
        the normalization displacement while newer edges absorb more of it.
        """
        np = state.np
        indptr = np.asarray(indptr)
        if indptr.ndim != 1 or len(indptr) < 1:
            raise ValueError("indptr must be a one-dimensional CSR pointer array")
        neuron_count = len(indptr) - 1
        if int(indptr[-1]) != state.edge_count:
            raise ValueError("indptr edge count does not match plasticity state")

        if base_abs is not None:
            base_abs = np.asarray(base_abs)
            if len(base_abs) != state.edge_count:
                raise ValueError("base_abs edge count does not match plasticity state")

        if presynaptic_indices is None:
            presynaptic_indices = range(neuron_count)

        neurons_touched = 0
        edges_scaled = 0
        scale_sum = 0.0

        for pre in presynaptic_indices:
            pre = int(pre)
            if pre < 0 or pre >= neuron_count:
                raise IndexError(f"presynaptic index {pre} is outside 0:{neuron_count}")
            start = int(indptr[pre])
            stop = int(indptr[pre + 1])
            if start == stop:
                continue

            plastic_mask = state.plastic_mask[start:stop]
            plastic_count = int(plastic_mask.sum())
            if plastic_count == 0:
                continue

            multiplier = state.multiplier[start:stop]
            if base_abs is None:
                baseline_budget = float(stop - start)
                current_budget = float(multiplier.sum())
            else:
                anatomical = base_abs[start:stop]
                baseline_budget = float(anatomical.sum())
                current_budget = float((anatomical * multiplier).sum())

            target_budget = baseline_budget * self.target_scale
            if current_budget <= 0.0:
                continue

            exact_scale = target_budget / current_budget
            applied_scale = 1.0 + self.strength * (exact_scale - 1.0)

            if self.stability_protection <= 0.0:
                multiplier[plastic_mask] *= applied_scale
            else:
                local_stability = state.stability[start:stop]
                # Unconsolidated edges receive the full normalization pressure;
                # stability==1 can resist up to ``stability_protection`` of it.
                pressure = 1.0 - self.stability_protection * local_stability
                edge_scale = 1.0 + (applied_scale - 1.0) * pressure
                multiplier[plastic_mask] *= edge_scale[plastic_mask]

            np.clip(
                multiplier,
                state.config.min_multiplier,
                state.config.max_multiplier,
                out=multiplier,
            )

            neurons_touched += 1
            edges_scaled += plastic_count
            scale_sum += applied_scale

        mean_scale = scale_sum / neurons_touched if neurons_touched else 1.0
        return BudgetNormalizationStats(neurons_touched, edges_scaled, mean_scale)
