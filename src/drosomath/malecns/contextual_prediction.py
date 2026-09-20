"""Second-order contextual next-symbol prediction (Phase F.3A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain import (
    DirectionalModulationConfig,
    OutgoingBudgetNormalizer,
    PlasticityController,
    TimingProfiler,
    UsageRewardRule,
)

from .sequence_memory import (
    ORDERED_PAIRS,
    SequenceMemoryConfig,
    TwoCueSequenceSession,
    balanced_pair_schedule,
)
from .symbol_interface import SYMBOLS
from .symbol_learning import build_symbol_learning_signal, symbol_output_context
from .working_memory import WorkingMemoryInterface
from .working_memory_learning import _compact_directional, _compact_reward


CONTEXTUAL_GRAMMAR: dict[tuple[str, str], str] = {
    ("A", "A"): "A", ("A", "B"): "B", ("A", "C"): "C", ("A", "D"): "D",
    ("B", "A"): "C", ("B", "B"): "D", ("B", "C"): "A", ("B", "D"): "B",
    ("C", "A"): "D", ("C", "B"): "C", ("C", "C"): "B", ("C", "D"): "A",
    ("D", "A"): "B", ("D", "B"): "A", ("D", "C"): "D", ("D", "D"): "C",
}


ContextualPredictionConfig = SequenceMemoryConfig


@dataclass(frozen=True, slots=True)
class ContextualPredictionTrial:
    first: str
    second: str
    target: str
    reset_between_items: bool
    decision: str
    go_output_rates_hz: dict[str, float]
    go_output_spikes: int
    first_output_spikes: int
    second_output_spikes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "first": self.first,
            "second": self.second,
            "target": self.target,
            "reset_between_items": bool(self.reset_between_items),
            "decision": self.decision,
            "go_output_rates_hz": dict(self.go_output_rates_hz),
            "go_output_spikes": int(self.go_output_spikes),
            "first_output_spikes": int(self.first_output_spikes),
            "second_output_spikes": int(self.second_output_spikes),
        }


class ContextualPredictionSession(TwoCueSequenceSession):
    """Predict from the ordered pair; grammar lookup happens after GO."""

    def run_trial(self, first: str, second: str, **kwargs) -> ContextualPredictionTrial:
        sequence_result = super().run_trial(first, second, **kwargs)
        # This lookup is deliberately after the parent session has completed
        # FIRST -> SECOND -> GO and produced the decision.
        target = CONTEXTUAL_GRAMMAR[(first, second)]
        return ContextualPredictionTrial(
            first=first,
            second=second,
            target=target,
            reset_between_items=sequence_result.reset_between_items,
            decision=sequence_result.decision,
            go_output_rates_hz=dict(sequence_result.go_output_rates_hz),
            go_output_spikes=sequence_result.go_output_spikes,
            first_output_spikes=sequence_result.first_output_spikes,
            second_output_spikes=sequence_result.second_output_spikes,
        )


class ContextualPredictionLearningSession:
    """Use generic symbol learning after a grammar-labelled GO decision."""

    def __init__(
        self,
        brain,
        interface: WorkingMemoryInterface,
        *,
        config: ContextualPredictionConfig | None = None,
        timing_profiler: TimingProfiler | None = None,
        route_cache_enabled: bool = True,
        plastic_row_cache_enabled: bool = True,
        prospective_index_enabled: bool = True,
    ) -> None:
        if brain.connectome is not interface.connectome:
            raise ValueError("brain and contextual interface must match")
        self.brain = brain
        self.interface = interface
        self.config = config or ContextualPredictionConfig()
        if route_cache_enabled != self.config.route_cache_enabled:
            raise ValueError("route cache setting must match F.3A config")
        self.timing_profiler = timing_profiler
        self.episode = ContextualPredictionSession(brain, interface)
        self.output_context = symbol_output_context(interface.symbol_interface)
        self.controller = PlasticityController(
            DirectionalModulationConfig(
                learning_rate=self.config.directional_learning_rate,
                two_hop_credit_mode=self.config.two_hop_credit_mode,
            ),
            route_cache_enabled=route_cache_enabled,
            plastic_row_cache_enabled=plastic_row_cache_enabled,
            prospective_index_enabled=prospective_index_enabled,
        )
        self.reward_rule = UsageRewardRule(learning_rate=self.config.reward_learning_rate)
        self.normalizer = OutgoingBudgetNormalizer(strength=self.config.budget_strength)
        self.trial_results: list[dict[str, object]] = []
        self._initial_budget = int(brain.plasticity.plastic_edge_count)

    @property
    def plastic_budget_start(self) -> int:
        return self._initial_budget

    def _active_eligible_edges(self) -> set[int]:
        state = self.brain.plasticity
        result: set[int] = set()
        for pre_raw in sorted(self.brain._recent_presynaptic):
            pre = int(pre_raw)
            start = int(self.brain.connectome.indptr[pre])
            stop = int(self.brain.connectome.indptr[pre + 1])
            if stop > start:
                local = np.flatnonzero(state.eligibility[start:stop] > 0.0)
                result.update(int(start + value) for value in local)
        return result

    def _phase_observer(self, snapshots):
        def observe(phase, _brain):
            snapshots[phase] = self._active_eligible_edges()
        return observe

    def train_trial(self, first: str, second: str, *, capture_phase_credit: bool = False):
        previous_tracking = self.brain.set_plasticity_tracking(True)
        snapshots: dict[str, set[int]] = {}
        try:
            context = (
                self.timing_profiler.section("network_simulation_seconds")
                if self.timing_profiler is not None else _NullContext()
            )
            with context:
                result = self.episode.run_trial(
                    first,
                    second,
                    track_eligibility=True,
                    retain_tracking=True,
                    phase_observer=self._phase_observer(snapshots) if capture_phase_credit else None,
                )
            # The grammar target is already present on the post-GO result; no
            # target information entered the simulation or route selection.
            signal: LearningSignal = build_symbol_learning_signal(
                target=result.target,
                decision=result.decision,
                output_rates_hz=result.go_output_rates_hz,
            )
            reward_credit = (
                self.controller.build_reward_credit(self.brain, signal, self.output_context)
                if signal.reward > 0.0 and signal.positive_reinforcements() else None
            )
            holder = {"update": None}

            def apply_directional(_state):
                holder["update"] = self.controller.apply_learning_signal(
                    self.brain,
                    signal,
                    self.output_context,
                    timing_profiler=self.timing_profiler,
                    telemetry_level=self.config.telemetry_level,
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
                first_edges = snapshots.get("first", set())
                second_edges = snapshots.get("second", set())
                go_edges = snapshots.get("go", set())
                phase_credit = {
                    "updated_edges": len(updated),
                    "updated_edges_eligible_after_first": len(updated & first_edges),
                    "updated_edges_eligible_by_end_second": len(updated & second_edges),
                    "updated_edges_newly_eligible_during_second": len(updated & (second_edges - first_edges)),
                    "updated_edges_newly_eligible_during_go": len(updated & (go_edges - second_edges)),
                    "eligible_edges_after_first": len(first_edges),
                    "eligible_edges_by_end_second": len(second_edges),
                    "eligible_edges_by_end_go": len(go_edges),
                }
            record = {
                "result": result,
                "success": bool(signal.success),
                "reward": float(signal.reward),
                "directional_error": dict(signal.directional_error),
                "reinforcement": dict(signal.reinforcement),
                "directional_update": _compact_directional(update),
                "reward_update": _compact_reward(reward_report.get("learning")),
                "phase_credit": phase_credit,
            }
            self.trial_results.append(record)
            return record
        finally:
            self.brain.set_plasticity_tracking(previous_tracking)

    def train(self, schedule, *, capture_first_incorrect: int | None = None):
        remaining = self.config.episode_credit_limit if capture_first_incorrect is None else int(capture_first_incorrect)
        records = []
        for first, second in schedule:
            record = self.train_trial(first, second, capture_phase_credit=remaining > 0)
            if remaining > 0 and not record["success"]:
                remaining -= 1
                if record["phase_credit"] is not None:
                    records.append(dict(record["phase_credit"]))
        return list(self.trial_results), records


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


__all__ = [
    "CONTEXTUAL_GRAMMAR",
    "ContextualPredictionConfig",
    "ContextualPredictionLearningSession",
    "ContextualPredictionSession",
    "ContextualPredictionTrial",
    "ORDERED_PAIRS",
    "balanced_pair_schedule",
]
