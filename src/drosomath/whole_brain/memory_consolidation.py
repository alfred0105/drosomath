from __future__ import annotations

from dataclasses import dataclass

from .plastic_state import SparsePlasticityState
from .usage_learning import LearningUpdateStats


@dataclass(frozen=True, slots=True)
class ProtectedRewardRule:
    """Reward-modulated plasticity with stability-dependent protection."""

    learning_rate: float = 0.02
    positive_reward_scale: float = 1.0
    negative_reward_scale: float = 1.0
    positive_protection: float = 0.35
    negative_protection: float = 0.90
    protection_power: float = 1.5
    stability_gain: float = 0.035
    stability_loss: float = 0.002
    min_credit: float = 1e-8
    chunk_size: int = 1_000_000

    def __post_init__(self) -> None:
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")
        for name, value in (
            ("positive_protection", self.positive_protection),
            ("negative_protection", self.negative_protection),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.protection_power <= 0.0:
            raise ValueError("protection_power must be > 0")
        if self.stability_gain < 0.0 or self.stability_loss < 0.0:
            raise ValueError("stability rates must be >= 0")
        if self.min_credit < 0.0:
            raise ValueError("min_credit must be >= 0")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

    def apply(self, state: SparsePlasticityState, *, reward: float) -> LearningUpdateStats:
        if reward == 0.0 or self.learning_rate == 0.0 or state.edge_count == 0:
            return LearningUpdateStats(0, 0.0, 0.0)

        np = state.np
        total_updates = 0
        sum_abs_delta = 0.0
        max_abs_delta = 0.0
        reward_scale = self.positive_reward_scale if reward > 0 else self.negative_reward_scale
        protection_strength = self.positive_protection if reward > 0 else self.negative_protection

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
            local_stability = stability[active]
            protection = 1.0 - protection_strength * np.power(local_stability, self.protection_power)
            protection = np.clip(protection, 0.05, 1.0)
            delta = self.learning_rate * reward_scale * reward * local_credit * protection

            old = multiplier[active].copy()
            new = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
            multiplier[active] = new
            actual_delta = new - old

            if reward > 0.0 and self.stability_gain > 0.0:
                stability[active] += self.stability_gain * local_credit * (1.0 - stability[active])
                np.clip(stability, 0.0, 1.0, out=stability)
            elif reward < 0.0 and self.stability_loss > 0.0:
                stability[active] *= 1.0 - self.stability_loss * abs(reward) * (1.0 - local_stability)

            count = int(active.sum())
            total_updates += count
            local_abs = np.abs(actual_delta)
            sum_abs_delta += float(local_abs.sum())
            if count:
                max_abs_delta = max(max_abs_delta, float(local_abs.max()))

        mean_abs = sum_abs_delta / total_updates if total_updates else 0.0
        return LearningUpdateStats(total_updates, mean_abs, max_abs_delta)


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    top_fraction: float = 0.15
    boost: float = 0.30
    min_stability: float = 1e-5
    min_usage: float = 1e-5
    min_stage_gain: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if not 0.0 <= self.boost <= 1.0:
            raise ValueError("boost must be in [0, 1]")
        if self.min_stability < 0.0 or self.min_usage < 0.0 or self.min_stage_gain < 0.0:
            raise ValueError("minimums must be >= 0")


class MemoryConsolidator:
    """Convert repeatedly rewarded/used edges into harder-to-overwrite memory.

    Passing ``baseline_stability`` switches to stage-local mode. Only edges whose
    stability increased during the current stage are candidates, preventing old
    already-stable memories from repeatedly winning every consolidation round.
    """

    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def consolidate(
        self,
        state: SparsePlasticityState,
        *,
        baseline_stability=None,
    ) -> dict[str, float | int | bool]:
        np = state.np
        stage_local = baseline_stability is not None
        stage_gain = None
        candidate = (
            state.plastic_mask
            & (state.stability >= self.config.min_stability)
            & (state.usage_ema >= self.config.min_usage)
        )

        if stage_local:
            baseline = np.asarray(baseline_stability, dtype=state.stability.dtype)
            if baseline.shape != state.stability.shape:
                raise ValueError("baseline_stability shape must match state.stability")
            stage_gain = state.stability - baseline
            candidate &= stage_gain > self.config.min_stage_gain

        idx = np.flatnonzero(candidate)
        if len(idx) == 0 or self.config.boost == 0.0:
            return {
                "candidate_edges": int(len(idx)),
                "consolidated_edges": 0,
                "mean_stability_gain": 0.0,
                "stage_local": bool(stage_local),
                "mean_preboost_stage_gain": 0.0,
            }

        signal = state.stability[idx] if stage_gain is None else np.maximum(stage_gain[idx], 0.0)
        importance = (
            signal
            * (0.25 + state.usage_ema[idx])
            * (0.5 + np.maximum(state.multiplier[idx] - 1.0, 0.0))
        )
        keep_n = max(1, int(round(len(idx) * self.config.top_fraction)))
        if keep_n < len(idx):
            selected_local = np.argpartition(importance, -keep_n)[-keep_n:]
            selected = idx[selected_local]
        else:
            selected = idx

        old = state.stability[selected].copy()
        state.stability[selected] += self.config.boost * (1.0 - state.stability[selected])
        np.clip(state.stability[selected], 0.0, 1.0, out=state.stability[selected])
        gain = state.stability[selected] - old
        preboost = 0.0 if stage_gain is None else float(np.maximum(stage_gain[selected], 0.0).mean())
        return {
            "candidate_edges": int(len(idx)),
            "consolidated_edges": int(len(selected)),
            "mean_stability_gain": float(gain.mean()) if len(gain) else 0.0,
            "stage_local": bool(stage_local),
            "mean_preboost_stage_gain": preboost,
        }


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    interval: int = 4
    fraction: float = 0.80
    rate_hz: float = 230.0
    max_prior_tasks: int = 3

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("interval must be >= 1")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        if self.max_prior_tasks < 1:
            raise ValueError("max_prior_tasks must be >= 1")


class ReplayScheduler:
    """Uniformly interleave previously learned tasks while a new task is trained."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self.config = config or ReplayConfig()

    def should_replay(self, current_trial: int, prior_task_count: int) -> bool:
        return prior_task_count > 0 and current_trial > 0 and current_trial % self.config.interval == 0

    def choose_prior_index(self, rng, prior_task_count: int) -> int:
        if prior_task_count < 1:
            raise ValueError("prior_task_count must be >= 1")
        first = max(0, prior_task_count - self.config.max_prior_tasks)
        return int(rng.integers(first, prior_task_count))


@dataclass(frozen=True, slots=True)
class AdaptiveReplayConfig(ReplayConfig):
    error_power: float = 2.0
    min_weight: float = 0.03
    ema_decay: float = 0.85
    default_accuracy: float = 0.50

    def __post_init__(self) -> None:
        ReplayConfig.__post_init__(self)
        if self.error_power <= 0.0:
            raise ValueError("error_power must be > 0")
        if self.min_weight <= 0.0:
            raise ValueError("min_weight must be > 0")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0.0 <= self.default_accuracy <= 1.0:
            raise ValueError("default_accuracy must be in [0, 1]")


class AdaptiveReplayScheduler(ReplayScheduler):
    """Replay weak prior tasks more often than already-mastered tasks."""

    def __init__(self, config: AdaptiveReplayConfig | None = None) -> None:
        ReplayScheduler.__init__(self, config or AdaptiveReplayConfig())
        self.config: AdaptiveReplayConfig
        self._accuracy: dict[int, float] = {}
        self._updates: dict[int, int] = {}

    def set_accuracy(self, task_index: int, accuracy: float) -> None:
        self._accuracy[int(task_index)] = min(1.0, max(0.0, float(accuracy)))

    def update(self, task_index: int, *, correct: bool) -> float:
        idx = int(task_index)
        old = self._accuracy.get(idx, self.config.default_accuracy)
        target = 1.0 if correct else 0.0
        new = self.config.ema_decay * old + (1.0 - self.config.ema_decay) * target
        self._accuracy[idx] = float(new)
        self._updates[idx] = self._updates.get(idx, 0) + 1
        return float(new)

    def accuracy(self, task_index: int) -> float:
        return float(self._accuracy.get(int(task_index), self.config.default_accuracy))

    def weight(self, task_index: int) -> float:
        error = max(0.0, 1.0 - self.accuracy(task_index))
        return max(self.config.min_weight, error ** self.config.error_power)

    def choose_prior_index(self, rng, prior_task_count: int) -> int:
        if prior_task_count < 1:
            raise ValueError("prior_task_count must be >= 1")
        np = __import__("numpy")
        first = max(0, prior_task_count - self.config.max_prior_tasks)
        indices = np.arange(first, prior_task_count, dtype=np.int32)
        weights = np.asarray([self.weight(int(i)) for i in indices], dtype=np.float64)
        weights /= weights.sum()
        return int(rng.choice(indices, p=weights))

    def snapshot(self, prior_task_count: int) -> dict[str, object]:
        first = max(0, prior_task_count - self.config.max_prior_tasks)
        rows = []
        for i in range(first, prior_task_count):
            rows.append({
                "task_index": int(i),
                "accuracy_ema": self.accuracy(i),
                "weight": self.weight(i),
                "updates": int(self._updates.get(i, 0)),
            })
        total = sum(float(x["weight"]) for x in rows) or 1.0
        for row in rows:
            row["probability"] = float(row["weight"]) / total
        return {"tasks": rows}
