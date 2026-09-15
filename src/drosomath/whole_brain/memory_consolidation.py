from __future__ import annotations

from dataclasses import dataclass

from .plastic_state import SparsePlasticityState
from .usage_learning import LearningUpdateStats


@dataclass(frozen=True, slots=True)
class ProtectedRewardRule:
    """Reward-modulated plasticity with stability-dependent protection.

    Stability is treated as a task-agnostic estimate of memory importance.
    Consolidated edges can still change, but their learning rate is reduced so
    a later task cannot overwrite them as easily. Positive updates retain more
    plasticity than negative updates to avoid freezing useful reusable routes.
    """

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
                # Strong memories decay much more slowly on a single mistake.
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

    def __post_init__(self) -> None:
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if not 0.0 <= self.boost <= 1.0:
            raise ValueError("boost must be in [0, 1]")
        if self.min_stability < 0.0 or self.min_usage < 0.0:
            raise ValueError("minimums must be >= 0")


class MemoryConsolidator:
    """Convert repeatedly rewarded/used edges into harder-to-overwrite memory."""

    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def consolidate(self, state: SparsePlasticityState) -> dict[str, float | int]:
        np = state.np
        candidate = (
            state.plastic_mask
            & (state.stability >= self.config.min_stability)
            & (state.usage_ema >= self.config.min_usage)
        )
        idx = np.flatnonzero(candidate)
        if len(idx) == 0 or self.config.boost == 0.0:
            return {"candidate_edges": int(len(idx)), "consolidated_edges": 0, "mean_stability_gain": 0.0}

        # Stability is reward-derived; usage favors routes that repeatedly
        # participated. A small multiplier term favors potentiated pathways
        # without making it the sole memory signal.
        importance = (
            state.stability[idx]
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
        return {
            "candidate_edges": int(len(idx)),
            "consolidated_edges": int(len(selected)),
            "mean_stability_gain": float(gain.mean()) if len(gain) else 0.0,
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
    """Interleave previously learned tasks while a new task is trained."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self.config = config or ReplayConfig()

    def should_replay(self, current_trial: int, prior_task_count: int) -> bool:
        return prior_task_count > 0 and current_trial > 0 and current_trial % self.config.interval == 0

    def choose_prior_index(self, rng, prior_task_count: int) -> int:
        if prior_task_count < 1:
            raise ValueError("prior_task_count must be >= 1")
        first = max(0, prior_task_count - self.config.max_prior_tasks)
        return int(rng.integers(first, prior_task_count))
