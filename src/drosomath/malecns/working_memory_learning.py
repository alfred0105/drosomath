"""First learned delayed-cue working-memory experiment (Phase F.2B).

The episode runner is deliberately a thin learning layer over the F.2A
learning-disabled delayed-cue primitive.  The brain receives only real cue and
GO populations; target identity enters only when the completed GO decision is
translated into the existing generic :class:`LearningSignal`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain import (
    DirectionalModulationConfig,
    OutgoingBudgetNormalizer,
    PlasticityController,
    TimingProfiler,
    UsageRewardRule,
)

from .symbol_interface import SYMBOLS
from .symbol_learning import build_symbol_learning_signal, symbol_output_context
from .working_memory import DelayedCueSession, DelayedCueTrialResult, WorkingMemoryInterface


@dataclass(frozen=True, slots=True)
class WorkingMemoryLearningConfig:
    """F.2B protocol; timing and learning constants are intentionally fixed."""

    cue_ms: float = 20.0
    delay_ms: float = 20.0
    go_ms: float = 20.0
    stimulus_rate_hz: float = 205.0
    training_trials: int = 800
    schedule_seed_offset: int = 10_000
    directional_learning_rate: float = 0.02
    reward_learning_rate: float = 0.02
    budget_strength: float = 0.25
    two_hop_credit_mode: str = "prospective_anatomical"
    adaptive_plastic_budget: bool = False
    route_cache_enabled: bool = True
    episode_credit_limit: int = 16

    def __post_init__(self) -> None:
        if (self.cue_ms, self.delay_ms, self.go_ms) != (20.0, 20.0, 20.0):
            raise ValueError("F.2B timing is fixed at 20/20/20 ms")
        if self.stimulus_rate_hz != 205.0:
            raise ValueError("F.2B stimulus rate is fixed at 205 Hz")
        if self.training_trials != 800:
            raise ValueError("F.2B training is fixed at 800 trials")
        if self.directional_learning_rate != 0.02 or self.reward_learning_rate != 0.02:
            raise ValueError("F.2B learning rates are fixed at 0.02")
        if self.two_hop_credit_mode != "prospective_anatomical":
            raise ValueError("F.2B requires prospective_anatomical credit")
        if self.adaptive_plastic_budget:
            raise ValueError("F.2B keeps adaptive plastic budget disabled")
        if not self.route_cache_enabled:
            raise ValueError("F.2B scientific runs require route cache enabled")
        if self.episode_credit_limit < 0:
            raise ValueError("episode_credit_limit must be >= 0")


@dataclass(frozen=True, slots=True)
class WorkingMemoryLearningTrial:
    """Compact post-episode record; no dense state is retained."""

    cue: str
    decision: str
    go_output_rates_hz: dict[str, float]
    go_output_spikes: int
    reward: float
    success: bool
    directional_error: dict[str, float]
    reinforcement: dict[str, float]
    directional_update: dict[str, object] = field(default_factory=dict)
    reward_update: dict[str, object] = field(default_factory=dict)
    phase_credit: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cue": self.cue,
            "decision": self.decision,
            "go_output_rates_hz": dict(self.go_output_rates_hz),
            "go_output_spikes": int(self.go_output_spikes),
            "reward": float(self.reward),
            "success": bool(self.success),
            "directional_error": dict(self.directional_error),
            "reinforcement": dict(self.reinforcement),
            "directional_update": dict(self.directional_update),
            "reward_update": dict(self.reward_update),
            "phase_credit": dict(self.phase_credit) if self.phase_credit is not None else None,
        }


def _compact_directional(update) -> dict[str, object]:
    if update is None:
        return {"edge_updates": 0, "unique_edge_updates": 0, "hop_counts": {}}
    return {
        "edge_updates": int(update.edge_updates),
        "unique_edge_updates": int(update.unique_edge_updates),
        "sum_abs_delta": float(update.sum_abs_delta),
        "hop_counts": {str(key): int(value) for key, value in update.hop_counts.items()},
        "channel_updates": {str(key): int(value) for key, value in update.channel_updates.items()},
        "channel_hop_counts": {
            str(channel): {str(hop): int(count) for hop, count in hops.items()}
            for channel, hops in update.channel_hop_counts.items()
        },
        "updated_edge_indices": [int(edge) for edge in update.updated_edge_indices],
    }


def _compact_reward(update) -> dict[str, object]:
    if update is None:
        return {"edge_updates": 0}
    if isinstance(update, dict):
        value = lambda name, default=0: update.get(name, default)
    else:
        value = lambda name, default=0: getattr(update, name, default)
    return {
        "edge_updates": int(value("edge_updates")),
        "actual_reward_updated_edges": int(value("actual_reward_updated_edges")),
        "credited_reward_edges": int(value("credited_reward_edges")),
        "selected_credit_edges": int(value("selected_credit_edges")),
        "mean_abs_delta": float(value("mean_abs_delta")),
    }


class DelayedCueLearningSession:
    """Apply existing generic local learning after one complete GO episode."""

    def __init__(
        self,
        brain,
        interface: WorkingMemoryInterface,
        *,
        config: WorkingMemoryLearningConfig | None = None,
        timing_profiler: TimingProfiler | None = None,
        route_cache_enabled: bool = True,
    ) -> None:
        if brain.connectome is not interface.connectome:
            raise ValueError("brain and working-memory interface must match")
        self.brain = brain
        self.interface = interface
        self.config = config or WorkingMemoryLearningConfig()
        if route_cache_enabled != self.config.route_cache_enabled:
            raise ValueError("route cache setting must match F.2B config")
        self.timing_profiler = timing_profiler
        self.episode = DelayedCueSession(brain, interface, learning_enabled=False)
        self.output_context = symbol_output_context(interface.symbol_interface)
        self.controller = PlasticityController(
            DirectionalModulationConfig(
                learning_rate=self.config.directional_learning_rate,
                two_hop_credit_mode=self.config.two_hop_credit_mode,
            ),
            route_cache_enabled=route_cache_enabled,
        )
        self.reward_rule = UsageRewardRule(learning_rate=self.config.reward_learning_rate)
        self.normalizer = OutgoingBudgetNormalizer(strength=self.config.budget_strength)
        self.trial_results: list[WorkingMemoryLearningTrial] = []
        self._initial_budget = int(brain.plasticity.plastic_edge_count)

    @property
    def plastic_budget_start(self) -> int:
        return self._initial_budget

    def _active_eligible_edges(self) -> set[int]:
        state = self.brain.plasticity
        indptr = self.brain.connectome.indptr
        result: set[int] = set()
        for pre_raw in sorted(self.brain._recent_presynaptic):
            pre = int(pre_raw)
            start, stop = int(indptr[pre]), int(indptr[pre + 1])
            if stop <= start:
                continue
            local = np.flatnonzero(state.eligibility[start:stop] > 0.0)
            result.update(int(start + value) for value in local)
        return result

    def _phase_observer(self, snapshots: dict[str, set[int]]):
        def observe(phase: str, _brain) -> None:
            snapshots[phase] = self._active_eligible_edges()
        return observe

    def train_trial(self, cue: str, *, capture_phase_credit: bool = False) -> WorkingMemoryLearningTrial:
        previous_tracking = self.brain.set_plasticity_tracking(True)
        snapshots: dict[str, set[int]] = {}
        try:
            context = (
                self.timing_profiler.section("network_simulation_seconds")
                if self.timing_profiler is not None else nullcontext()
            )
            with context:
                result = self.episode.run_trial(
                    cue,
                    track_eligibility=True,
                    retain_tracking=True,
                    phase_observer=self._phase_observer(snapshots) if capture_phase_credit else None,
                )
            signal: LearningSignal = build_symbol_learning_signal(
                target=result.cue,
                decision=result.decision,
                output_rates_hz=result.go_output_rates_hz,
            )
            reward_credit = (
                self.controller.build_reward_credit(self.brain, signal, self.output_context)
                if signal.reward > 0.0 and signal.positive_reinforcements()
                else None
            )
            holder: dict[str, object] = {"update": None}

            def apply_directional(_state):
                holder["update"] = self.controller.apply_learning_signal(
                    self.brain,
                    signal,
                    self.output_context,
                    timing_profiler=self.timing_profiler,
                )

            reward_report = self.brain.learn_from_reward(
                reward=signal.reward,
                rule=self.reward_rule,
                normalizer=self.normalizer,
                include_plasticity_summary=False,
                post_reward_hook=apply_directional,
                reward_credit=reward_credit,
                profile_timing=self.timing_profiler is not None,
            )
            if self.timing_profiler is not None:
                self.timing_profiler.absorb(reward_report.get("timing"))
            update = holder["update"]
            phase_credit = None
            if capture_phase_credit and not signal.success:
                updated = set(int(edge) for edge in getattr(update, "updated_edge_indices", ()))
                cue_edges = snapshots.get("cue", set())
                delay_edges = snapshots.get("delay", set())
                go_edges = snapshots.get("go", set())
                phase_credit = {
                    "updated_edges": int(len(updated)),
                    "updated_edges_eligible_after_cue": int(len(updated & cue_edges)),
                    "updated_edges_eligible_by_end_delay": int(len(updated & delay_edges)),
                    "updated_edges_newly_eligible_during_go": int(len(updated & (go_edges - delay_edges))),
                    "eligible_edges_after_cue": int(len(cue_edges)),
                    "eligible_edges_by_end_delay": int(len(delay_edges)),
                    "eligible_edges_by_end_go": int(len(go_edges)),
                }
            trial = WorkingMemoryLearningTrial(
                cue=result.cue,
                decision=result.decision,
                go_output_rates_hz=dict(result.go_output_rates_hz),
                go_output_spikes=int(result.go_output_spikes),
                reward=float(signal.reward),
                success=bool(signal.success),
                directional_error=dict(signal.directional_error),
                reinforcement=dict(signal.reinforcement),
                directional_update=_compact_directional(update),
                reward_update=_compact_reward(reward_report.get("learning")),
                phase_credit=phase_credit,
            )
            self.trial_results.append(trial)
            return trial
        finally:
            # learn_from_reward has already cleared episode eligibility and
            # recent-presynaptic state. Restore the caller's tracking mode.
            self.brain.set_plasticity_tracking(previous_tracking)

    def train(self, schedule: tuple[str, ...], *, capture_first_incorrect: int | None = None):
        remaining = self.config.episode_credit_limit if capture_first_incorrect is None else int(capture_first_incorrect)
        phase_records: list[dict[str, object]] = []
        for cue in schedule:
            before = len(self.trial_results)
            trial = self.train_trial(cue, capture_phase_credit=remaining > 0)
            if remaining > 0 and not trial.success:
                if trial.phase_credit is not None:
                    phase_records.append(dict(trial.phase_credit))
                remaining -= 1
            if len(self.trial_results) == before:
                raise RuntimeError("working-memory trial was not recorded")
        return list(self.trial_results), phase_records


__all__ = [
    "DelayedCueLearningSession",
    "WorkingMemoryLearningConfig",
    "WorkingMemoryLearningTrial",
]
