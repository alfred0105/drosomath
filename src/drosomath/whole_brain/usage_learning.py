from __future__ import annotations

from dataclasses import dataclass

from .plastic_state import SparsePlasticityState


@dataclass(frozen=True, slots=True)
class LearningUpdateStats:
    edge_updates: int
    mean_abs_delta: float
    max_abs_delta: float


@dataclass(frozen=True, slots=True)
class UsageRewardRule:
    """Strengthen frequently used, reward-relevant anatomical connections.

    The rule changes only a non-negative multiplier layered on top of the
    anatomical connection. Therefore an inhibitory anatomical connection stays
    inhibitory and an excitatory connection stays excitatory.
    """

    learning_rate: float = 0.02
    positive_reward_scale: float = 1.0
    negative_reward_scale: float = 1.0
    stability_gain: float = 0.02
    stability_loss: float = 0.005
    min_credit: float = 1e-8
    chunk_size: int = 1_000_000

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")
        if self.positive_reward_scale < 0.0 or self.negative_reward_scale < 0.0:
            raise ValueError("reward scales must be >= 0")
        if self.stability_gain < 0.0 or self.stability_loss < 0.0:
            raise ValueError("stability rates must be >= 0")
        if self.min_credit < 0.0:
            raise ValueError("min_credit must be >= 0")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def apply(self, state: SparsePlasticityState, *, reward: float) -> LearningUpdateStats:
        """Apply delayed reward to currently eligible edges in bounded chunks.

        Chunking prevents a reward update from allocating several additional
        full-size arrays when the graph has tens of millions of edges.
        """
        if reward == 0.0 or self.learning_rate == 0.0 or state.edge_count == 0:
            return LearningUpdateStats(0, 0.0, 0.0)

        np = state.np
        total_updates = 0
        sum_abs_delta = 0.0
        max_abs_delta = 0.0
        reward_scale = (
            self.positive_reward_scale if reward > 0.0 else self.negative_reward_scale
        )

        for start in range(0, state.edge_count, self.chunk_size):
            stop = min(state.edge_count, start + self.chunk_size)
            plastic = state.plastic_mask[start:stop]
            if not plastic.any():
                continue

            usage = state.usage_ema[start:stop]
            eligibility = state.eligibility[start:stop]
            stability = state.stability[start:stop]
            multiplier = state.multiplier[start:stop]

            credit = usage * eligibility
            active = plastic & (credit > self.min_credit)
            if not active.any():
                continue

            local_credit = credit[active]
            delta = self.learning_rate * reward_scale * reward * local_credit

            # Consolidated memories are harder to erase, but positive reward can
            # still strengthen them. This creates a plasticity/stability tradeoff.
            if reward < 0.0:
                delta *= 1.0 - stability[active]

            old = multiplier[active].copy()
            new = np.clip(
                old + delta,
                state.config.min_multiplier,
                state.config.max_multiplier,
            )
            multiplier[active] = new
            actual_delta = new - old

            if reward > 0.0 and self.stability_gain > 0.0:
                stability[active] += self.stability_gain * local_credit * (
                    1.0 - stability[active]
                )
                np.clip(stability, 0.0, 1.0, out=stability)
            elif reward < 0.0 and self.stability_loss > 0.0:
                stability[active] *= max(0.0, 1.0 - self.stability_loss * abs(reward))

            count = int(active.sum())
            total_updates += count
            local_abs = np.abs(actual_delta)
            sum_abs_delta += float(local_abs.sum())
            if count:
                max_abs_delta = max(max_abs_delta, float(local_abs.max()))

        mean_abs = sum_abs_delta / total_updates if total_updates else 0.0
        return LearningUpdateStats(total_updates, mean_abs, max_abs_delta)

    def apply_recent_presynaptic(
        self,
        state: SparsePlasticityState,
        *,
        reward: float,
        indptr,
        presynaptic_indices,
    ) -> LearningUpdateStats:
        """Reference-equivalent reward update over recently firing CSR rows.

        A normal trial clears eligibility immediately after reward delivery.
        Consequently, only edges belonging to presynaptic neurons that fired
        in that trial can have non-zero credit. This avoids chunk-scanning all
        anatomical edges on every successful keyboard trial.
        """
        if reward == 0.0 or self.learning_rate == 0.0 or state.edge_count == 0:
            return LearningUpdateStats(0, 0.0, 0.0)

        np = state.np
        pointers = np.asarray(indptr)
        if pointers.ndim != 1 or int(pointers[-1]) != state.edge_count:
            raise ValueError("indptr does not match plasticity state")
        pres = np.asarray(presynaptic_indices, dtype=np.int64)
        if len(pres) == 0:
            return LearningUpdateStats(0, 0.0, 0.0)
        if int(pres.min()) < 0 or int(pres.max()) >= len(pointers) - 1:
            raise IndexError("presynaptic index out of range")

        total_updates = 0
        sum_abs_delta = 0.0
        max_abs_delta = 0.0
        reward_scale = (
            self.positive_reward_scale if reward > 0.0 else self.negative_reward_scale
        )

        # ``pres`` is unique at the caller. Work row-by-row rather than build
        # a potentially large gathered edge-index array; updates are edgewise
        # independent, so the resulting learned state is identical to the
        # chunked reference implementation.
        for pre in pres:
            start = int(pointers[pre])
            stop = int(pointers[pre + 1])
            if start == stop:
                continue
            plastic = state.plastic_mask[start:stop]
            if not plastic.any():
                continue
            usage = state.usage_ema[start:stop]
            eligibility = state.eligibility[start:stop]
            stability = state.stability[start:stop]
            multiplier = state.multiplier[start:stop]
            credit = usage * eligibility
            active = plastic & (credit > self.min_credit)
            if not active.any():
                continue

            local_credit = credit[active]
            delta = self.learning_rate * reward_scale * reward * local_credit
            if reward < 0.0:
                delta *= 1.0 - stability[active]
            old = multiplier[active].copy()
            new = np.clip(
                old + delta,
                state.config.min_multiplier,
                state.config.max_multiplier,
            )
            multiplier[active] = new
            actual_delta = new - old

            if reward > 0.0 and self.stability_gain > 0.0:
                stability[active] += self.stability_gain * local_credit * (
                    1.0 - stability[active]
                )
                np.clip(stability, 0.0, 1.0, out=stability)
            elif reward < 0.0 and self.stability_loss > 0.0:
                stability[active] *= max(0.0, 1.0 - self.stability_loss * abs(reward))

            count = int(active.sum())
            total_updates += count
            local_abs = np.abs(actual_delta)
            sum_abs_delta += float(local_abs.sum())
            if count:
                max_abs_delta = max(max_abs_delta, float(local_abs.max()))

        mean_abs = sum_abs_delta / total_updates if total_updates else 0.0
        return LearningUpdateStats(total_updates, mean_abs, max_abs_delta)
