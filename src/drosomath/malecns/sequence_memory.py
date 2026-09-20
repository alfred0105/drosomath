"""Short two-item sequence memory task (Phase F.2D).

The task presents ``FIRST -> SECOND -> GO`` with no silent gaps.  The target
is always FIRST, while SECOND is a balanced distractor.  This module keeps
sequence semantics outside the generic whole-brain learning controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

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
from .working_memory import GO_SYMBOL, WorkingMemoryInterface
from .working_memory_learning import _compact_directional, _compact_reward


ORDERED_PAIRS = tuple((first, second) for first in SYMBOLS for second in SYMBOLS)


@dataclass(frozen=True, slots=True)
class SequenceMemoryConfig:
    first_ms: float = 20.0
    second_ms: float = 20.0
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
    telemetry_level: str = "summary"

    def __post_init__(self) -> None:
        if (self.first_ms, self.second_ms, self.go_ms) != (20.0, 20.0, 20.0):
            raise ValueError("F.2D timing is fixed at 20/20/20 ms")
        if self.stimulus_rate_hz != 205.0:
            raise ValueError("F.2D stimulus rate is fixed at 205 Hz")
        if self.training_trials != 800:
            raise ValueError("F.2D training is fixed at 800 episodes")
        if self.directional_learning_rate != 0.02 or self.reward_learning_rate != 0.02:
            raise ValueError("F.2D learning rates are fixed at 0.02")
        if self.two_hop_credit_mode != "prospective_anatomical":
            raise ValueError("F.2D requires prospective_anatomical credit")
        if self.adaptive_plastic_budget:
            raise ValueError("F.2D keeps adaptive plastic budget disabled")
        if not self.route_cache_enabled:
            raise ValueError("F.2D scientific runs require route cache enabled")
        if self.telemetry_level not in {"full", "summary"}:
            raise ValueError("telemetry_level must be 'full' or 'summary'")


@dataclass(frozen=True, slots=True)
class SequenceTrialResult:
    first: str
    second: str
    reset_between_items: bool
    decision: str
    go_output_rates_hz: dict[str, float]
    go_output_spikes: int
    first_output_spikes: int
    second_output_spikes: int

    @property
    def target(self) -> str:
        return self.first

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


class TwoCueSequenceSession:
    """Run FIRST -> SECOND -> GO episodes without applying learning."""

    def __init__(
        self,
        brain,
        interface: WorkingMemoryInterface,
        *,
        item_stimulus_rate_hz: float = 205.0,
        go_stimulus_rate_hz: float = 205.0,
    ):
        if brain.connectome is not interface.connectome:
            raise ValueError("brain and sequence interface must match")
        for name, rate in (
            ("item_stimulus_rate_hz", item_stimulus_rate_hz),
            ("go_stimulus_rate_hz", go_stimulus_rate_hz),
        ):
            if not math.isfinite(float(rate)) or float(rate) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        self.brain = brain
        self.interface = interface
        self.item_stimulus_rate_hz = float(item_stimulus_rate_hz)
        self.go_stimulus_rate_hz = float(go_stimulus_rate_hz)
        self._output_lookup = np.full(brain.connectome.neuron_count, -1, dtype=np.int8)
        for index, symbol in enumerate(SYMBOLS):
            self._output_lookup[interface.output_populations[symbol]] = index

    def _run_phase(self, stimulus_indices, duration_ms: float, stimulus_rate_hz: float):
        steps = int(math.ceil(duration_ms / self.brain.params.dt_ms))
        counts = np.zeros(len(SYMBOLS), dtype=np.int32)
        total_spikes = 0
        for _ in range(steps):
            fired, _ = self.brain.step(
                stimulus_indices=stimulus_indices,
                stimulus_rate_hz=stimulus_rate_hz,
            )
            total_spikes += int(len(fired))
            if len(fired):
                local = self._output_lookup[fired]
                local = local[local >= 0]
                if len(local):
                    np.add.at(counts, local, 1)
        seconds = duration_ms / 1000.0
        rates = {
            symbol: float(counts[index]) / seconds
            for index, symbol in enumerate(SYMBOLS)
        }
        return rates, int(counts.sum())

    def run_trial(
        self,
        first: str,
        second: str,
        *,
        reset_between_items: bool = False,
        track_eligibility: bool = False,
        retain_tracking: bool = False,
        phase_observer=None,
    ) -> SequenceTrialResult:
        if first not in SYMBOLS or second not in SYMBOLS:
            raise KeyError("first and second must be symbols A/B/C/D")
        previous_tracking = self.brain.set_plasticity_tracking(track_eligibility)
        try:
            self.brain.reset()
            _, first_spikes = self._run_phase(
                self.interface.population_for_input(first),
                20.0,
                self.item_stimulus_rate_hz,
            )
            if phase_observer is not None:
                phase_observer("first", self.brain)
            if reset_between_items:
                self.brain.reset()
            _, second_spikes = self._run_phase(
                self.interface.population_for_input(second),
                20.0,
                self.item_stimulus_rate_hz,
            )
            if phase_observer is not None:
                phase_observer("second", self.brain)
            rates, go_spikes = self._run_phase(
                self.interface.population_for_input(GO_SYMBOL),
                20.0,
                self.go_stimulus_rate_hz,
            )
            if phase_observer is not None:
                phase_observer("go", self.brain)
            decision = self.interface.symbol_interface.decision_surface.decide(
                rates, total_output_spikes=go_spikes
            )
            return SequenceTrialResult(
                first=first,
                second=second,
                reset_between_items=bool(reset_between_items),
                decision=decision,
                go_output_rates_hz=rates,
                go_output_spikes=go_spikes,
                first_output_spikes=first_spikes,
                second_output_spikes=second_spikes,
            )
        finally:
            if not retain_tracking:
                self.brain.set_plasticity_tracking(previous_tracking)


class SequenceLearningSession:
    """Apply the existing generic symbol learner after the GO decision."""

    def __init__(
        self,
        brain,
        interface: WorkingMemoryInterface,
        *,
        config: SequenceMemoryConfig | None = None,
        timing_profiler: TimingProfiler | None = None,
        route_cache_enabled: bool = True,
    ) -> None:
        if brain.connectome is not interface.connectome:
            raise ValueError("brain and sequence interface must match")
        self.brain = brain
        self.interface = interface
        self.config = config or SequenceMemoryConfig()
        if route_cache_enabled != self.config.route_cache_enabled:
            raise ValueError("route cache setting must match F.2D config")
        self.timing_profiler = timing_profiler
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
        self.episode = TwoCueSequenceSession(brain, interface)
        self.trial_results: list[SequenceTrialResult] = []
        self._initial_budget = int(brain.plasticity.plastic_edge_count)

    @property
    def plastic_budget_start(self) -> int:
        return self._initial_budget

    def _active_eligible_edges(self) -> set[int]:
        state = self.brain.plasticity
        result: set[int] = set()
        for pre_raw in sorted(self.brain._recent_presynaptic):
            pre = int(pre_raw)
            start, stop = int(self.brain.connectome.indptr[pre]), int(self.brain.connectome.indptr[pre + 1])
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
        snapshots = {}
        try:
            context = (
                self.timing_profiler.section("network_simulation_seconds")
                if self.timing_profiler is not None else _nullcontext()
            )
            with context:
                result = self.episode.run_trial(
                    first,
                    second,
                    track_eligibility=True,
                    retain_tracking=True,
                    phase_observer=self._phase_observer(snapshots) if capture_phase_credit else None,
                )
            signal: LearningSignal = build_symbol_learning_signal(
                target=result.first,
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


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def balanced_pair_schedule(cycles: int, *, seed: int) -> tuple[tuple[str, str], ...]:
    if cycles < 0:
        raise ValueError("cycles must be >= 0")
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(cycles):
        order = rng.permutation(len(ORDERED_PAIRS))
        result.extend(ORDERED_PAIRS[int(index)] for index in order)
    return tuple(result)


__all__ = [
    "ORDERED_PAIRS",
    "SequenceLearningSession",
    "SequenceMemoryConfig",
    "SequenceTrialResult",
    "TwoCueSequenceSession",
    "balanced_pair_schedule",
]
