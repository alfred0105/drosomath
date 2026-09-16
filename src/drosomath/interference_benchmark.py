from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from math import sqrt
from statistics import fmean, pstdev
from typing import Callable, Iterable

from .core import (
    ActivityBiasedCandidateConfig,
    ActivityBiasedCandidateGenerator,
    AdaptiveBridge,
    BrainCore,
    BridgeActivityCandidateConfig,
    BridgeActivityCandidateGenerator,
    BridgeStructuralConfig,
    BridgeStructuralPlasticityManager,
    BridgeSynapseState,
    ConsolidationConfig,
    DualBrainSystem,
    MemoryConsolidator,
    PlasticityTracker,
    RewardWeightRule,
    SpikingNetwork,
    STDPPlasticity,
    STDPRule,
    StructuralPlasticityConfig,
    StructuralPlasticityManager,
    SynapseState,
)
from .evaluation import ContinualMemoryEvaluator
from .experiments import AssociativeTrial, SequentialMemoryExperiment, SequentialTask


VARIANT_NAMES = (
    "reward_only",
    "reward_plus_stdp",
    "reward_plus_rewiring",
    "reward_consolidation_rewiring",
    "dual_brain_full",
)


@dataclass(frozen=True, slots=True)
class SeedResult:
    seed: int
    mean_retention: float
    mean_forgetting: float
    final_scores: dict[str, float]
    retention: dict[str, float]
    forgetting: dict[str, float]
    synapse_count: int
    bridge_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "mean_retention": self.mean_retention,
            "mean_forgetting": self.mean_forgetting,
            "final_scores": self.final_scores,
            "retention": self.retention,
            "forgetting": self.forgetting,
            "synapse_count": self.synapse_count,
            "bridge_count": self.bridge_count,
        }


def _jitter(rng: random.Random, base: float, spread: float = 0.025) -> float:
    return max(0.01, min(0.99, base + rng.uniform(-spread, spread)))


def _output_ids(seed: int) -> tuple[int, int]:
    # Balance the deterministic lower-neuron-ID tie break across the seed set.
    return (6, 7) if seed % 2 == 0 else (7, 6)


def _tasks(seed: int) -> tuple[SequentialTask, ...]:
    out_a, out_b = _output_ids(seed)
    return (
        SequentialTask(
            name="task_A",
            train_trials=(AssociativeTrial({0: 1.0, 1: 1.0}, out_a),) * 10,
            eval_trials=(AssociativeTrial({0: 1.0, 1: 1.0}, out_a),) * 4,
        ),
        SequentialTask(
            name="task_B",
            train_trials=(AssociativeTrial({0: 1.0, 2: 1.0}, out_b),) * 10,
            eval_trials=(AssociativeTrial({0: 1.0, 2: 1.0}, out_b),) * 4,
        ),
        SequentialTask(
            name="task_C",
            train_trials=(AssociativeTrial({0: 1.0, 3: 1.0}, out_a),) * 10,
            eval_trials=(AssociativeTrial({0: 1.0, 3: 1.0}, out_a),) * 4,
        ),
    )


def _single_synapses(seed: int) -> list[SynapseState]:
    rng = random.Random(seed)
    out_a, out_b = _output_ids(seed)
    return [
        # Shared feature fans into both hidden paths. Reward learning on later
        # tasks can therefore disturb earlier decisions and create interference.
        SynapseState(0, 4, _jitter(rng, 0.255)),
        SynapseState(0, 5, _jitter(rng, 0.255)),
        SynapseState(1, 4, _jitter(rng, 0.265)),
        SynapseState(1, 5, _jitter(rng, 0.055)),
        SynapseState(2, 4, _jitter(rng, 0.055)),
        SynapseState(2, 5, _jitter(rng, 0.265)),
        SynapseState(3, 4, _jitter(rng, 0.245)),
        SynapseState(3, 5, _jitter(rng, 0.075)),
        SynapseState(4, out_a, _jitter(rng, 0.535)),
        SynapseState(4, out_b, _jitter(rng, 0.315)),
        SynapseState(5, out_a, _jitter(rng, 0.315)),
        SynapseState(5, out_b, _jitter(rng, 0.535)),
        # Low-value spare edges give structural plasticity something to replace.
        SynapseState(8, 4, _jitter(rng, 0.045, 0.01)),
        SynapseState(9, 5, _jitter(rng, 0.045, 0.01)),
    ]


def _build_single(
    seed: int,
    *,
    use_stdp: bool,
    use_consolidation: bool,
) -> SequentialMemoryExperiment:
    tracker = PlasticityTracker(
        _single_synapses(seed),
        reward_window=10,
        weight_rule=RewardWeightRule(
            learning_rate=0.012,
            min_weight=0.0,
            max_weight=1.0,
        ),
    )
    stdp = None
    if use_stdp:
        stdp = STDPPlasticity(
            rule=STDPRule(
                potentiation_rate=0.004,
                depression_rate=0.005,
                tau_plus=8.0,
                tau_minus=8.0,
                window=8,
            )
        )
    network = SpikingNetwork.from_ids(
        range(10),
        tracker,
        threshold=0.5,
        decay=0.90,
        stdp=stdp,
    )
    consolidator = None
    if use_consolidation:
        consolidator = MemoryConsolidator(
            config=ConsolidationConfig(
                min_usage=4,
                min_reward=0.04,
                growth_rate=0.10,
                decay_rate=0.001,
            )
        )
    out_a, out_b = _output_ids(seed)
    return SequentialMemoryExperiment(
        network,
        tracker,
        output_neurons=(out_a, out_b),
        response_steps=3,
        correct_reward=1.0,
        wrong_reward=-0.65,
        consolidator=consolidator,
    )


def _run_single(seed: int, variant: str) -> SeedResult:
    use_stdp = variant == "reward_plus_stdp"
    use_consolidation = variant == "reward_consolidation_rewiring"
    use_rewiring = variant in {
        "reward_plus_rewiring",
        "reward_consolidation_rewiring",
    }
    experiment = _build_single(
        seed,
        use_stdp=use_stdp,
        use_consolidation=use_consolidation,
    )
    evaluator = ContinualMemoryEvaluator()
    seen: list[SequentialTask] = []

    generator = ActivityBiasedCandidateGenerator(
        config=ActivityBiasedCandidateConfig(max_candidates=48, pool_size=10)
    )
    structural = StructuralPlasticityManager(
        experiment.tracker,
        config=StructuralPlasticityConfig(
            min_age_cycles=1,
            stale_steps=6,
            reward_threshold=0.0,
            protected_stability=0.65,
            max_rewire_per_cycle=2,
            regrow_weight=0.08,
        ),
    )

    for stage, task in enumerate(_tasks(seed)):
        experiment.train_task(task, epochs=1)
        if use_rewiring:
            candidates = generator.generate(
                experiment.tracker.synapses,
                step=experiment.network.step_index,
                neuron_ids=experiment.network.neurons,
            )
            structural.rewire(
                step=experiment.network.step_index,
                candidate_pairs=candidates,
            )
        seen.append(task)
        evaluator.record(
            stage=stage,
            scores={t.name: experiment.evaluate_task(t) for t in seen},
        )

    summary = evaluator.summary()
    return SeedResult(
        seed=seed,
        mean_retention=float(summary["mean_retention"]),
        mean_forgetting=float(summary["mean_forgetting"]),
        final_scores=dict(evaluator.latest_scores),
        retention={name: float(value) for name, value in summary["retention"].items()},
        forgetting={name: float(value) for name, value in summary["forgetting"].items()},
        synapse_count=experiment.tracker.synapse_count,
    )


def _dual_brain_a(seed: int) -> BrainCore:
    rng = random.Random(seed)
    out_a, out_b = _output_ids(seed)
    synapses = [
        SynapseState(0, 4, _jitter(rng, 0.255)),
        SynapseState(0, 5, _jitter(rng, 0.255)),
        SynapseState(1, 4, _jitter(rng, 0.265)),
        SynapseState(1, 5, _jitter(rng, 0.055)),
        SynapseState(2, 4, _jitter(rng, 0.055)),
        SynapseState(2, 5, _jitter(rng, 0.265)),
        SynapseState(3, 4, _jitter(rng, 0.245)),
        SynapseState(3, 5, _jitter(rng, 0.075)),
        SynapseState(4, out_a, _jitter(rng, 0.535)),
        SynapseState(4, out_b, _jitter(rng, 0.315)),
        SynapseState(5, out_a, _jitter(rng, 0.315)),
        SynapseState(5, out_b, _jitter(rng, 0.535)),
        SynapseState(8, 4, _jitter(rng, 0.045, 0.01)),
        SynapseState(9, 5, _jitter(rng, 0.045, 0.01)),
    ]
    tracker = PlasticityTracker(
        synapses,
        reward_window=16,
        weight_rule=RewardWeightRule(learning_rate=0.010),
    )
    network = SpikingNetwork.from_ids(
        range(10),
        tracker,
        threshold=0.5,
        decay=0.90,
        stdp=STDPPlasticity(rule=STDPRule(potentiation_rate=0.004, depression_rate=0.005, window=8)),
    )
    return BrainCore("A", network, tracker)


def _dual_brain_b(seed: int) -> BrainCore:
    rng = random.Random(seed + 100_000)
    out_a, out_b = _output_ids(seed)
    synapses = [
        SynapseState(0, 4, _jitter(rng, 0.60)),
        SynapseState(1, 5, _jitter(rng, 0.60)),
        SynapseState(0, 5, _jitter(rng, 0.08)),
        SynapseState(1, 4, _jitter(rng, 0.08)),
        SynapseState(4, out_a, _jitter(rng, 0.60)),
        SynapseState(4, out_b, _jitter(rng, 0.10)),
        SynapseState(5, out_a, _jitter(rng, 0.10)),
        SynapseState(5, out_b, _jitter(rng, 0.60)),
        SynapseState(8, 4, _jitter(rng, 0.04, 0.01)),
        SynapseState(9, 5, _jitter(rng, 0.04, 0.01)),
    ]
    tracker = PlasticityTracker(
        synapses,
        reward_window=16,
        weight_rule=RewardWeightRule(learning_rate=0.010),
    )
    network = SpikingNetwork.from_ids(
        range(10),
        tracker,
        threshold=0.5,
        decay=0.90,
        stdp=STDPPlasticity(rule=STDPRule(potentiation_rate=0.004, depression_rate=0.005, window=8)),
    )
    return BrainCore("B", network, tracker)


class _DualExperiment:
    RESPONSE_STEPS = 6

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.brain_a = _dual_brain_a(seed)
        self.brain_b = _dual_brain_b(seed)
        out_a, out_b = _output_ids(seed)
        self.bridge = AdaptiveBridge(
            [
                BridgeSynapseState("A", out_a, "B", 0, 0.70),
                BridgeSynapseState("A", out_b, "B", 1, 0.70),
                BridgeSynapseState("A", 8, "B", 8, 0.05),
                BridgeSynapseState("B", 9, "A", 9, 0.05),
            ],
            reward_window=16,
            reward_alpha=0.08,
            learning_rate=0.012,
        )
        self.system = DualBrainSystem(self.brain_a, self.brain_b, self.bridge)
        self.out_a = out_a
        self.out_b = out_b
        self.consolidators = (
            MemoryConsolidator(config=ConsolidationConfig(min_usage=4, min_reward=0.04, growth_rate=0.10, decay_rate=0.001)),
            MemoryConsolidator(config=ConsolidationConfig(min_usage=4, min_reward=0.04, growth_rate=0.10, decay_rate=0.001)),
        )
        self.internal_generators = (
            ActivityBiasedCandidateGenerator(config=ActivityBiasedCandidateConfig(max_candidates=48, pool_size=10)),
            ActivityBiasedCandidateGenerator(config=ActivityBiasedCandidateConfig(max_candidates=48, pool_size=10)),
        )
        self.internal_structural = (
            StructuralPlasticityManager(self.brain_a.tracker, config=StructuralPlasticityConfig(min_age_cycles=1, stale_steps=8, protected_stability=0.65, max_rewire_per_cycle=1, regrow_weight=0.08)),
            StructuralPlasticityManager(self.brain_b.tracker, config=StructuralPlasticityConfig(min_age_cycles=1, stale_steps=8, protected_stability=0.65, max_rewire_per_cycle=1, regrow_weight=0.08)),
        )
        self.bridge_generator = BridgeActivityCandidateGenerator(
            config=BridgeActivityCandidateConfig(max_candidates=64, pool_size=10)
        )
        self.bridge_structural = BridgeStructuralPlasticityManager(
            self.bridge,
            {"A": range(10), "B": range(10)},
            config=BridgeStructuralConfig(
                min_age_cycles=1,
                stale_steps=8,
                reward_threshold=0.0,
                max_rewire_per_cycle=1,
                regrow_weight=0.08,
            ),
        )

    def _reset_fast(self) -> None:
        for brain in (self.brain_a, self.brain_b):
            brain.network.reset_state(clear_spike_history=True)
            brain.tracker.clear_recent()
        self.bridge.clear_recent()

    def run_trial(self, trial: AssociativeTrial, *, training: bool) -> bool:
        self._reset_fast()
        previous = (
            self.brain_a.network.learning_enabled,
            self.brain_b.network.learning_enabled,
        )
        self.brain_a.network.set_learning_enabled(training)
        self.brain_b.network.set_learning_enabled(training)
        counts = {self.out_a: 0, self.out_b: 0}
        try:
            for index in range(self.RESPONSE_STEPS):
                result = self.system.step(
                    currents_a=trial.stimulus if index == 0 else None
                )
                for neuron_id in result.brain_b.fired:
                    if neuron_id in counts:
                        counts[neuron_id] += 1
            active = [(count, neuron_id) for neuron_id, count in counts.items() if count > 0]
            prediction = max(active, key=lambda item: (item[0], -item[1]))[1] if active else None
            correct = prediction == trial.expected_output
            if training:
                reward = 1.0 if correct else -0.65
                self.system.apply_reward(reward)
                self.consolidators[0].consolidate(self.brain_a.tracker.synapses)
                self.consolidators[1].consolidate(self.brain_b.tracker.synapses)
            return correct
        finally:
            self.brain_a.network.set_learning_enabled(previous[0])
            self.brain_b.network.set_learning_enabled(previous[1])

    def train_task(self, task: SequentialTask) -> None:
        for trial in task.train_trials:
            self.run_trial(trial, training=True)

    def evaluate(self, task: SequentialTask) -> float:
        return sum(self.run_trial(trial, training=False) for trial in task.eval_trials) / len(task.eval_trials)

    def maintain(self) -> None:
        brains = (self.brain_a, self.brain_b)
        for brain, generator, manager in zip(brains, self.internal_generators, self.internal_structural):
            candidates = generator.generate(
                brain.tracker.synapses,
                step=brain.network.step_index,
                neuron_ids=brain.network.neurons,
            )
            manager.rewire(step=brain.network.step_index, candidate_pairs=candidates)
        bridge_candidates = self.bridge_generator.generate(
            {"A": self.brain_a, "B": self.brain_b},
            self.bridge,
            step=self.system.step_index,
        )
        self.bridge_structural.rewire(
            step=self.system.step_index,
            candidate_keys=bridge_candidates,
        )


def _run_dual(seed: int) -> SeedResult:
    experiment = _DualExperiment(seed)
    evaluator = ContinualMemoryEvaluator()
    seen: list[SequentialTask] = []
    for stage, task in enumerate(_tasks(seed)):
        experiment.train_task(task)
        experiment.maintain()
        seen.append(task)
        evaluator.record(
            stage=stage,
            scores={t.name: experiment.evaluate(t) for t in seen},
        )
    summary = evaluator.summary()
    return SeedResult(
        seed=seed,
        mean_retention=float(summary["mean_retention"]),
        mean_forgetting=float(summary["mean_forgetting"]),
        final_scores=dict(evaluator.latest_scores),
        retention={name: float(value) for name, value in summary["retention"].items()},
        forgetting={name: float(value) for name, value in summary["forgetting"].items()},
        synapse_count=experiment.brain_a.tracker.synapse_count + experiment.brain_b.tracker.synapse_count,
        bridge_count=experiment.bridge.synapse_count,
    )


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _aggregate(name: str, runs: list[SeedResult]) -> dict[str, object]:
    retention = [run.mean_retention for run in runs]
    forgetting = [run.mean_forgetting for run in runs]
    final_accuracy = [fmean(run.final_scores.values()) for run in runs]

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "mean": fmean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "sem": (pstdev(values) / sqrt(len(values))) if len(values) > 1 else 0.0,
            "min": min(values),
            "p25": _quantile(values, 0.25),
            "median": _quantile(values, 0.50),
            "p75": _quantile(values, 0.75),
            "max": max(values),
        }

    return {
        "name": name,
        "runs": len(runs),
        "mean_retention": stats(retention),
        "mean_forgetting": stats(forgetting),
        "final_accuracy": stats(final_accuracy),
        "per_task_final_accuracy": {
            task: stats([run.final_scores.get(task, 0.0) for run in runs])
            for task in ("task_A", "task_B", "task_C")
        },
        "raw": [run.as_dict() for run in runs],
    }


def run_benchmark(*, runs: int = 100, seed_start: int = 0) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    results: dict[str, list[SeedResult]] = {name: [] for name in VARIANT_NAMES}
    for offset in range(runs):
        seed = seed_start + offset
        for name in VARIANT_NAMES[:-1]:
            results[name].append(_run_single(seed, name))
        results["dual_brain_full"].append(_run_dual(seed))
    return {
        "experiment": "catastrophic_interference_v1",
        "runs_per_variant": runs,
        "variants": len(VARIANT_NAMES),
        "total_model_runs": runs * len(VARIANT_NAMES),
        "task_order": ["task_A", "task_B", "task_C"],
        "design": {
            "shared_input": 0,
            "task_cues": {"task_A": 1, "task_B": 2, "task_C": 3},
            "note": "Tasks are distinguishable but share a high-traffic input pathway to induce learnable interference.",
        },
        "results": [_aggregate(name, results[name]) for name in VARIANT_NAMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded DrosoMath catastrophic-interference benchmark")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(runs=args.runs, seed_start=args.seed_start), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
