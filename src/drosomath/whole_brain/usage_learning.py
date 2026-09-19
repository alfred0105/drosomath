from dataclasses import dataclass

from .plastic_state import SparsePlasticityState


@dataclass(frozen=True, slots=True)
class RewardCredit:
    """Sparse, task-independent causal credit for one reward event."""

    edge_indices: object
    weights: object
    aligned_one_hop_edges: int = 0
    aligned_two_hop_edges: int = 0
    unaligned_edges_skipped: int = 0
    opposing_path_edges_skipped: int = 0
    ambiguous_path_edges_skipped: int = 0

    def __post_init__(self) -> None:
        if len(self.edge_indices) != len(self.weights):
            raise ValueError("reward credit indices and weights must have equal length")
        if any(float(weight) < 0.0 for weight in self.weights):
            raise ValueError("reward credit weights must be non-negative")

    @property
    def mean_weight(self) -> float:
        return float(self.weights.mean()) if len(self.weights) else 0.0

    @property
    def selected_credit_edges(self) -> int:
        return int(len(self.edge_indices))


@dataclass(frozen=True, slots=True)
class LearningUpdateStats:
    edge_updates: int
    mean_abs_delta: float
    max_abs_delta: float
    eligible_reward_edges_before_localization: int = 0
    credited_reward_edges: int = 0
    actual_reward_updated_edges: int = 0
    aligned_one_hop_edges: int = 0
    aligned_two_hop_edges: int = 0
    unaligned_edges_skipped: int = 0
    uncredited_edges_updated: int = 0
    reward_credit_fraction: float = 0.0
    mean_reward_credit_weight: float = 0.0
    selected_credit_edges: int = 0
    actual_credited_reward_updated_edges: int = 0
    opposing_path_edges_skipped: int = 0
    ambiguous_path_edges_skipped: int = 0
    reward_credit_selection_fraction: float = 0.0
    reward_update_fraction: float = 0.0
    uncredited_reward_updated_edges: int = 0


@dataclass(frozen=True, slots=True)
class UsageRewardRule:
    """Strengthen active, reward-relevant anatomical connections."""

    learning_rate: float = 0.02
    positive_reward_scale: float = 1.0
    negative_reward_scale: float = 1.0
    stability_gain: float = 0.02
    stability_loss: float = 0.005
    min_credit: float = 1e-8
    chunk_size: int = 1_000_000
    uncredited_positive_reward_weight: float = 0.0

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
        if self.uncredited_positive_reward_weight < 0.0:
            raise ValueError("uncredited_positive_reward_weight must be >= 0")

    def _apply_indices(self, state, indices, *, reward, route_weights=None, apply_stability=True):
        """Apply one sparse batch; indices are sorted global edge indices."""
        np = state.np
        if len(indices) == 0:
            return 0, 0.0, 0.0
        indices = np.asarray(indices, dtype=np.int64)
        plastic = state.plastic_mask[indices]
        base_credit = state.usage_ema[indices] * state.eligibility[indices]
        active = plastic & (base_credit > self.min_credit)
        if not active.any():
            return 0, 0.0, 0.0
        local_credit = base_credit[active]
        if route_weights is not None:
            local_credit = local_credit * np.asarray(route_weights, dtype=np.float32)[active]
        scale = self.positive_reward_scale if reward > 0.0 else self.negative_reward_scale
        delta = self.learning_rate * scale * reward * local_credit
        stability = state.stability[indices]
        if reward < 0.0:
            delta *= 1.0 - stability[active]
        active_indices = indices[active]
        old = state.multiplier[active_indices].copy()
        new = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
        state.multiplier[active_indices] = new
        actual = new - old
        if apply_stability and reward > 0.0 and self.stability_gain > 0.0:
            stable = state.stability[active_indices]
            stable += self.stability_gain * local_credit * (1.0 - stable)
            np.clip(stable, 0.0, 1.0, out=stable)
            state.stability[active_indices] = stable
        elif reward < 0.0 and self.stability_loss > 0.0:
            state.stability[active_indices] *= max(0.0, 1.0 - self.stability_loss * abs(reward))
        return int(active.sum()), float(np.abs(actual).sum()), float(np.abs(actual).max())

    def _stats(self, state, *, reward, updates, actual_credited, sum_abs, max_abs, credit, eligible):
        positive = reward > 0.0
        selected = int(len(credit.edge_indices)) if credit is not None and positive else 0
        uncredited = max(0, updates - actual_credited) if positive else 0
        return LearningUpdateStats(
            updates, sum_abs / updates if updates else 0.0, max_abs,
            eligible if positive else 0,
            selected, updates,
            credit.aligned_one_hop_edges if credit is not None and positive else 0,
            credit.aligned_two_hop_edges if credit is not None and positive else 0,
            credit.unaligned_edges_skipped if credit is not None and positive else 0,
            uncredited,
            selected / max(1, eligible) if positive else 0.0,
            credit.mean_weight if credit is not None and positive else 0.0,
            selected,
            actual_credited if positive else 0,
            credit.opposing_path_edges_skipped if credit is not None and positive else 0,
            credit.ambiguous_path_edges_skipped if credit is not None and positive else 0,
            selected / max(1, eligible) if positive else 0.0,
            updates / max(1, eligible) if positive else 0.0,
            uncredited,
        )

    def apply(self, state: SparsePlasticityState, *, reward: float, reward_credit: RewardCredit | None = None) -> LearningUpdateStats:
        """Apply reward, retaining legacy full-scan behavior without credit."""
        if reward == 0.0 or self.learning_rate == 0.0 or state.edge_count == 0:
            return LearningUpdateStats(0, 0.0, 0.0)
        np = state.np
        updates = 0; actual_credited = 0; sum_abs = 0.0; max_abs = 0.0; eligible = 0
        localized = reward > 0.0 and reward_credit is not None
        if localized:
            indices = np.asarray(reward_credit.edge_indices, dtype=np.int64)
            count, abs_delta, max_delta = self._apply_indices(
                state, indices, reward=reward, route_weights=reward_credit.weights, apply_stability=False,
            )
            actual_credited += count; updates += count; sum_abs += abs_delta; max_abs = max(max_abs, max_delta)
            for start in range(0, state.edge_count, self.chunk_size):
                stop = min(state.edge_count, start + self.chunk_size)
                eligible += int((state.plastic_mask[start:stop] & (state.usage_ema[start:stop] * state.eligibility[start:stop] > self.min_credit)).sum())
            if self.uncredited_positive_reward_weight > 0.0:
                route_indices = np.asarray(indices, dtype=np.int64)
                for start in range(0, state.edge_count, self.chunk_size):
                    stop = min(state.edge_count, start + self.chunk_size)
                    chunk = np.arange(start, stop, dtype=np.int64)
                    lo = int(np.searchsorted(route_indices, start, side="left"))
                    hi = int(np.searchsorted(route_indices, stop, side="left"))
                    keep = np.ones(len(chunk), dtype=np.bool_)
                    if hi > lo:
                        keep[route_indices[lo:hi] - start] = False
                    count, abs_delta, max_delta = self._apply_indices(
                        state, chunk[keep], reward=reward,
                        route_weights=np.full(int(keep.sum()), self.uncredited_positive_reward_weight, dtype=np.float32),
                    )
                    updates += count; sum_abs += abs_delta; max_abs = max(max_abs, max_delta)
        else:
            for start in range(0, state.edge_count, self.chunk_size):
                stop = min(state.edge_count, start + self.chunk_size)
                if reward > 0.0:
                    eligible += int((state.plastic_mask[start:stop] & (state.usage_ema[start:stop] * state.eligibility[start:stop] > self.min_credit)).sum())
                indices = np.arange(start, stop, dtype=np.int64)
                count, abs_delta, max_delta = self._apply_indices(state, indices, reward=reward)
                updates += count; sum_abs += abs_delta; max_abs = max(max_abs, max_delta)
        return self._stats(state, reward=reward, updates=updates, actual_credited=actual_credited, sum_abs=sum_abs, max_abs=max_abs, credit=reward_credit if localized else None, eligible=eligible)

    def apply_recent_presynaptic(
        self,
        state: SparsePlasticityState,
        *,
        reward: float,
        indptr,
        presynaptic_indices,
        reward_credit: RewardCredit | None = None,
    ) -> LearningUpdateStats:
        """Apply reward row-by-row, optionally intersecting sparse route credit."""
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
        route_indices = np.asarray(reward_credit.edge_indices, dtype=np.int64) if reward_credit is not None and reward > 0.0 else None
        route_values = np.asarray(reward_credit.weights, dtype=np.float32) if route_indices is not None else None
        updates = 0; actual_credited = 0; sum_abs = 0.0; max_abs = 0.0; eligible = 0
        for pre in pres:
            start, stop = int(pointers[pre]), int(pointers[pre + 1])
            if start == stop:
                continue
            if route_indices is None:
                indices = np.arange(start, stop, dtype=np.int64)
                if reward > 0.0:
                    eligible += int((state.plastic_mask[start:stop] & (state.usage_ema[start:stop] * state.eligibility[start:stop] > self.min_credit)).sum())
                count, abs_delta, max_delta = self._apply_indices(state, indices, reward=reward)
            else:
                lo = int(np.searchsorted(route_indices, start, side="left"))
                hi = int(np.searchsorted(route_indices, stop, side="left"))
                indices = route_indices[lo:hi]
                weights = route_values[lo:hi]
                eligible += int((state.plastic_mask[start:stop] & (state.usage_ema[start:stop] * state.eligibility[start:stop] > self.min_credit)).sum())
                count, abs_delta, max_delta = self._apply_indices(state, indices, reward=reward, route_weights=weights, apply_stability=False)
                actual_credited += count
                if self.uncredited_positive_reward_weight > 0.0:
                    uncredited = np.arange(start, stop, dtype=np.int64)
                    keep = ~np.isin(uncredited, indices, assume_unique=True)
                    count2, abs_delta2, max_delta2 = self._apply_indices(
                        state, uncredited[keep], reward=reward,
                        route_weights=np.full(int(keep.sum()), self.uncredited_positive_reward_weight, dtype=np.float32),
                    )
                    count += count2; abs_delta += abs_delta2; max_delta = max(max_delta, max_delta2)
            updates += count; sum_abs += abs_delta; max_abs = max(max_abs, max_delta)
        return self._stats(state, reward=reward, updates=updates, actual_credited=actual_credited, sum_abs=sum_abs, max_abs=max_abs, credit=reward_credit if route_indices is not None else None, eligible=eligible)
