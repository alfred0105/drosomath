"""Generic A/B/C/D learning on the frozen F.1A decision surface.

This module is deliberately task-agnostic at the controller boundary.  The
small symbol adapter turns a direct population decision into a
``LearningSignal``; ``PlasticityController`` only receives generic channels
named ``symbol/<letter>`` and real output-neuron populations.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.symbol_interface import NO_DECISION, SYMBOLS, SymbolSession
from drosomath.whole_brain import (
    DirectionalModulationConfig,
    OutgoingBudgetNormalizer,
    PlasticityController,
    UsageRewardRule,
)


CHANNELS = tuple(f"symbol/{symbol}" for symbol in SYMBOLS)


@dataclass(frozen=True, slots=True)
class SymbolLearningConfig:
    """F.1B protocol settings; defaults are intentionally conservative."""

    duration_ms: float = 40.0
    stimulus_rate_hz: float = 205.0
    training_trials: int = 400
    evaluation_repetitions: int = 10
    schedule_seed_offset: int = 10_000
    evaluation_seed_offset: int = 20_000
    directional_learning_rate: float = 0.02
    reward_learning_rate: float = 0.02
    budget_strength: float = 0.25
    plastic_fraction: float = 0.05
    adaptive_plastic_budget: bool = False

    def __post_init__(self) -> None:
        if self.duration_ms <= 0.0 or self.stimulus_rate_hz < 0.0:
            raise ValueError("duration and stimulus rate must be valid")
        if self.training_trials < 4 or self.training_trials % len(SYMBOLS):
            raise ValueError("training_trials must be a positive multiple of four")
        if self.evaluation_repetitions < 1:
            raise ValueError("evaluation_repetitions must be >= 1")
        if not 0.0 <= self.plastic_fraction <= 1.0:
            raise ValueError("plastic_fraction must be in [0, 1]")
        if self.adaptive_plastic_budget:
            raise ValueError("F.1B must run with adaptive_plastic_budget=False")


@dataclass(frozen=True, slots=True)
class SymbolTrialResult:
    target: str
    decision: str
    output_rates_hz: dict[str, float]
    total_output_spikes: int
    directional_error: dict[str, float]
    reward: float
    success: bool
    reinforcement: dict[str, float]
    directional_update: dict[str, object] = field(default_factory=dict)
    reward_update: dict[str, object] = field(default_factory=dict)


def symbol_output_context(interface) -> dict[str, np.ndarray]:
    """Return exactly four generic output channels backed by real neurons."""
    return {
        f"symbol/{symbol}": np.asarray(interface.output_populations[symbol], dtype=np.int32).copy()
        for symbol in SYMBOLS
    }


def _winner_symbols(output_rates_hz: Mapping[str, float]) -> tuple[str, ...]:
    values = {symbol: float(output_rates_hz.get(symbol, 0.0)) for symbol in SYMBOLS}
    highest = max(values.values())
    if highest <= 0.0:
        return ()
    return tuple(symbol for symbol in SYMBOLS if values[symbol] == highest)


def build_symbol_learning_signal(
    *,
    target: str,
    decision: str,
    output_rates_hz: Mapping[str, float],
) -> LearningSignal:
    """Translate a direct decision into bounded generic directional feedback."""
    if target not in SYMBOLS:
        raise KeyError(f"unknown target {target!r}")
    if decision not in (*SYMBOLS, NO_DECISION):
        raise ValueError(f"unknown decision {decision!r}")
    winners = _winner_symbols(output_rates_hz)
    target_channel = f"symbol/{target}"

    if decision == target and len(winners) == 1:
        return LearningSignal(
            reward=1.0,
            success=True,
            reinforcement={target_channel: 1.0},
        )

    directions: dict[str, float] = {target_channel: 1.0}
    if decision == NO_DECISION:
        # A silent trial has no competing output to suppress.  For an exact
        # nonzero tie, distribute one unit of negative correction over the
        # tied non-target winners, never over the target itself.
        tied_non_target = tuple(symbol for symbol in winners if symbol != target)
        if tied_non_target:
            negative = -1.0 / len(tied_non_target)
            directions.update({f"symbol/{symbol}": negative for symbol in tied_non_target})
    else:
        directions[f"symbol/{decision}"] = -1.0
    return LearningSignal(
        reward=0.0,
        directional_error=directions,
        success=False,
    )


def balanced_symbol_schedule(*, cycles: int, seed: int) -> tuple[str, ...]:
    """Shuffle each complete A/B/C/D cycle; never oversample hard symbols."""
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    rng = np.random.default_rng(seed)
    schedule: list[str] = []
    for _ in range(cycles):
        cycle = list(SYMBOLS)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return tuple(schedule)


def _empty_directional() -> dict[str, object]:
    return {
        "edge_updates": 0,
        "channel_updates": {},
        "mean_abs_delta": 0.0,
        "sum_abs_delta": 0.0,
        "hop_counts": {},
        "excitatory_updates": 0,
        "inhibitory_updates": 0,
        "consolidated_edges": 0,
        "unique_edge_updates": 0,
        "reinforced_channels": [],
        "ambiguous_path_edges_skipped": 0,
        "updated_edge_indices": (),
        "updated_edge_hops": {},
        "channel_sum_abs_delta": {},
        "channel_unique_edge_updates": {},
        "channel_hop_counts": {},
    }


class SymbolLearningSession:
    """Run generic F.1B trials on one independently seeded brain."""

    def __init__(self, brain, interface, *, config: SymbolLearningConfig | None = None):
        self.brain = brain
        self.interface = interface
        self.config = config or SymbolLearningConfig()
        self.output_context = symbol_output_context(interface)
        self.controller = PlasticityController(
            DirectionalModulationConfig(learning_rate=self.config.directional_learning_rate)
        )
        self.reward_rule = UsageRewardRule(learning_rate=self.config.reward_learning_rate)
        self.normalizer = OutgoingBudgetNormalizer(strength=self.config.budget_strength)
        self._output_lookup = np.full(brain.connectome.neuron_count, -1, dtype=np.int8)
        for index, symbol in enumerate(SYMBOLS):
            self._output_lookup[interface.output_populations[symbol]] = index
        self._initial_budget = int(brain.plasticity.plastic_edge_count)
        self.trial_results: list[SymbolTrialResult] = []
        self._channel_edges_seen: dict[str, set[int]] = {
            channel: set() for channel in CHANNELS
        }

    @property
    def plastic_budget_start(self) -> int:
        return self._initial_budget

    def _run_network(self, target: str) -> tuple[dict[str, float], str, int, dict[str, object]]:
        steps = max(1, int(math.ceil(self.config.duration_ms / self.brain.params.dt_ms)))
        counts = np.zeros(len(SYMBOLS), dtype=np.int32)
        active: set[int] = set()
        self.brain.reset()
        sensory = self.interface.encoder.indices_for(target)
        for step_index in range(steps):
            fired, _ = self.brain.step(
                stimulus_indices=sensory,
                stimulus_rate_hz=self.config.stimulus_rate_hz,
            )
            if len(fired):
                active.update(int(index) for index in fired)
                local = self._output_lookup[fired]
                local = local[local >= 0]
                if len(local):
                    np.add.at(counts, local, 1)
        seconds = self.config.duration_ms / 1000.0
        rates = {symbol: float(counts[i]) / seconds for i, symbol in enumerate(SYMBOLS)}
        decision = self.interface.decision_surface.decide(
            rates, total_output_spikes=int(counts.sum())
        )
        return rates, decision, int(counts.sum()), {
            "total_spikes": int(sum(1 for _ in active)),
            "unique_neurons": int(len(active)),
            "steps": int(steps),
        }

    def train_trial(self, target: str) -> SymbolTrialResult:
        if target not in SYMBOLS:
            raise KeyError(f"unknown target {target!r}")
        previous_tracking = self.brain.set_plasticity_tracking(True)
        try:
            rates, decision, total_spikes, _activity = self._run_network(target)
            signal = build_symbol_learning_signal(
                target=target, decision=decision, output_rates_hz=rates
            )
            reward_credit = (
                self.controller.build_reward_credit(self.brain, signal, self.output_context)
                if signal.reward > 0.0 and signal.positive_reinforcements()
                else None
            )
            generic_holder: dict[str, object] = {"update": None}

            def apply_directional(_state):
                update = self.controller.apply_learning_signal(
                    self.brain, signal, self.output_context
                )
                generic_holder["update"] = update
                return None

            reward_report = self.brain.learn_from_reward(
                reward=signal.reward,
                rule=self.reward_rule,
                normalizer=self.normalizer,
                include_plasticity_summary=False,
                post_reward_hook=apply_directional,
                reward_credit=reward_credit,
            )
            update = generic_holder["update"]
            directional = asdict(update) if update is not None else _empty_directional()
            if update is not None:
                for channel, edges in update.channel_edge_indices.items():
                    if channel in self._channel_edges_seen:
                        self._channel_edges_seen[channel].update(int(edge) for edge in edges)
            # Per-channel edge IDs are needed only for in-memory cumulative
            # telemetry; do not retain them in every trial result.
            directional.pop("channel_edge_indices", None)
            # Edge arrays are diagnostic only; keep the normal result compact.
            directional["updated_edge_indices"] = tuple(
                int(edge) for edge in directional.get("updated_edge_indices", ())
            )
            result = SymbolTrialResult(
                target=target,
                decision=decision,
                output_rates_hz=rates,
                total_output_spikes=total_spikes,
                directional_error=dict(signal.directional_error),
                reward=float(signal.reward),
                success=bool(signal.success),
                reinforcement=dict(signal.reinforcement),
                directional_update=directional,
                reward_update=dict(reward_report.get("learning", {})),
            )
            self.trial_results.append(result)
            return result
        finally:
            self.brain.set_plasticity_tracking(previous_tracking)

    def channel_unique_edges(self) -> dict[str, int]:
        """Return cumulative distinct generic-channel edge counts."""
        return {
            channel: int(len(edges))
            for channel, edges in self._channel_edges_seen.items()
        }

    def evaluate_trial(self, target: str) -> SymbolTrialResult:
        observation = SymbolSession(self.brain, self.interface).present(
            symbol=target,
            duration_ms=self.config.duration_ms,
            stimulus_rate_hz=self.config.stimulus_rate_hz,
            learn=False,
        )
        signal = build_symbol_learning_signal(
            target=target,
            decision=observation.decision,
            output_rates_hz=observation.output_population_rates_hz,
        )
        return SymbolTrialResult(
            target=target,
            decision=observation.decision,
            output_rates_hz=dict(observation.output_population_rates_hz),
            total_output_spikes=int(observation.total_output_spikes),
            directional_error=dict(signal.directional_error),
            reward=float(signal.reward),
            success=bool(signal.success),
            reinforcement=dict(signal.reinforcement),
        )

    def train(self, *, seed: int) -> list[SymbolTrialResult]:
        cycles = self.config.training_trials // len(SYMBOLS)
        schedule = balanced_symbol_schedule(
            cycles=cycles, seed=seed + self.config.schedule_seed_offset
        )
        for target in schedule:
            self.train_trial(target)
        return list(self.trial_results)

    def evaluate(self, *, seed: int) -> list[SymbolTrialResult]:
        schedule = balanced_symbol_schedule(
            cycles=self.config.evaluation_repetitions,
            seed=seed + self.config.evaluation_seed_offset,
        )
        return [self.evaluate_trial(target) for target in schedule]

    def telemetry(self) -> dict[str, object]:
        state = self.brain.plasticity
        saturation = (state.multiplier <= state.config.min_multiplier) | (
            state.multiplier >= state.config.max_multiplier
        )
        directional_edges: set[int] = set()
        hop_counts: Counter[str] = Counter()
        excitatory = inhibitory = 0
        directional_trials = 0
        reinforcement_trials = 0
        consolidated = 0
        for trial in self.trial_results:
            update = trial.directional_update
            if int(update.get("edge_updates", 0)):
                directional_trials += 1
            directional_edges.update(int(edge) for edge in update.get("updated_edge_indices", ()))
            hop_counts.update({str(k): int(v) for k, v in update.get("hop_counts", {}).items()})
            excitatory += int(update.get("excitatory_updates", 0))
            inhibitory += int(update.get("inhibitory_updates", 0))
            if trial.reinforcement:
                reinforcement_trials += 1
            consolidated += int(update.get("consolidated_edges", 0))
        return {
            "directional_update_trials": int(directional_trials),
            "unique_directional_edges_modified": int(len(directional_edges)),
            "directional_hop_counts": dict(hop_counts),
            "excitatory_updates": int(excitatory),
            "inhibitory_updates": int(inhibitory),
            "positive_reinforcement_trials": int(reinforcement_trials),
            "positive_reinforcement_consolidated_edges": int(consolidated),
            "multiplier_saturation_fraction": float(np.mean(saturation)),
            "mean_multiplier": float(np.mean(state.multiplier)),
            "mean_stability": float(np.mean(state.stability)),
            "plastic_budget_start": int(self._initial_budget),
            "plastic_budget_end": int(state.plastic_edge_count),
            "plastic_budget_drift": int(state.plastic_edge_count - self._initial_budget),
            "adaptive_plastic_budget": bool(self.config.adaptive_plastic_budget),
        }


def _metric_summary(results: Sequence[SymbolTrialResult]) -> dict[str, object]:
    total = len(results)
    correct = sum(result.success for result in results)
    per_symbol: dict[str, dict[str, object]] = {}
    confusion = {target: {decision: 0 for decision in (*SYMBOLS, NO_DECISION)} for target in SYMBOLS}
    target_shares: list[float] = []
    margins: list[float] = []
    for result in results:
        confusion[result.target][result.decision] += 1
        competitor = max(
            (rate for symbol, rate in result.output_rates_hz.items() if symbol != result.target),
            default=0.0,
        )
        target_rate = float(result.output_rates_hz.get(result.target, 0.0))
        total_rate = float(sum(result.output_rates_hz.values()))
        target_shares.append(target_rate / total_rate if total_rate > 0.0 else 0.0)
        margins.append(target_rate - competitor)
    for symbol in SYMBOLS:
        selected = [result for result in results if result.target == symbol]
        per_symbol[symbol] = {
            "trials": len(selected),
            "accuracy": float(sum(result.success for result in selected) / len(selected)) if selected else 0.0,
            "no_decision": int(sum(result.decision == NO_DECISION for result in selected)),
            "target_output_share": float(np.mean([
                (r.output_rates_hz.get(symbol, 0.0) / sum(r.output_rates_hz.values()))
                if sum(r.output_rates_hz.values()) > 0.0 else 0.0
                for r in selected
            ])) if selected else 0.0,
            "target_minus_best_competitor_margin": float(np.mean([
                r.output_rates_hz.get(symbol, 0.0) - max(
                    (rate for other, rate in r.output_rates_hz.items() if other != symbol),
                    default=0.0,
                ) for r in selected
            ])) if selected else 0.0,
        }
    distribution = Counter(result.decision for result in results)
    largest_fraction = max(distribution.values(), default=0) / total if total else 0.0
    return {
        "trials": int(total),
        "accuracy": float(correct / total) if total else 0.0,
        "macro_accuracy": float(np.mean([per_symbol[symbol]["accuracy"] for symbol in SYMBOLS])) if total else 0.0,
        "per_symbol": per_symbol,
        "confusion_matrix": confusion,
        "no_decision": int(sum(result.decision == NO_DECISION for result in results)),
        "no_decision_fraction": float(sum(result.decision == NO_DECISION for result in results) / total) if total else 0.0,
        "prediction_distribution": dict(distribution),
        "largest_prediction_fraction": float(largest_fraction),
        "distinct_predictions": int(len(distribution)),
        "collapsed": bool(largest_fraction >= 0.8),
        "target_output_share": float(np.mean(target_shares)) if target_shares else 0.0,
        "target_minus_best_competitor_margin": float(np.mean(margins)) if margins else 0.0,
    }


def summarize_results(results: Sequence[SymbolTrialResult]) -> dict[str, object]:
    """Public compact metric summary used by the F.1B runner and tests."""
    return _metric_summary(results)


__all__ = [
    "CHANNELS",
    "SymbolLearningConfig",
    "SymbolLearningSession",
    "SymbolTrialResult",
    "balanced_symbol_schedule",
    "build_symbol_learning_signal",
    "summarize_results",
    "symbol_output_context",
]
