from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from statistics import fmean, pstdev
from math import sqrt

from .evaluation import ContinualMemoryEvaluator
from .experiments import AssociativeTrial, SequentialTask
from . import interference_benchmark as v1


@dataclass(frozen=True, slots=True)
class SeedResultV2:
    seed: int
    acquisition_scores: dict[str, float]
    final_scores: dict[str, float]
    forgetting: dict[str, float]
    learned_retention: dict[str, float]
    synapse_count: int
    bridge_count: int = 0

    @property
    def acquisition_accuracy(self) -> float:
        return fmean(self.acquisition_scores.values())

    @property
    def final_accuracy(self) -> float:
        return fmean(self.final_scores.values())

    @property
    def learned_retention_mean(self) -> float:
        values = list(self.learned_retention.values())
        return fmean(values) if values else 0.0

    @property
    def mean_forgetting(self) -> float:
        return fmean(self.forgetting.values()) if self.forgetting else 0.0

    @property
    def learned_task_count(self) -> int:
        return len(self.learned_retention)

    @property
    def forgotten_learned_tasks(self) -> int:
        return sum(1 for value in self.learned_retention.values() if value < 0.5)

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "acquisition_scores": self.acquisition_scores,
            "final_scores": self.final_scores,
            "forgetting": self.forgetting,
            "learned_retention": self.learned_retention,
            "acquisition_accuracy": self.acquisition_accuracy,
            "final_accuracy": self.final_accuracy,
            "learned_retention_mean": self.learned_retention_mean,
            "mean_forgetting": self.mean_forgetting,
            "learned_task_count": self.learned_task_count,
            "forgotten_learned_tasks": self.forgotten_learned_tasks,
            "synapse_count": self.synapse_count,
            "bridge_count": self.bridge_count,
        }


def _learned_metrics(evaluator: ContinualMemoryEvaluator) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    acquisition = dict(evaluator.initial_scores)
    final = dict(evaluator.latest_scores)
    forgetting = {
        task: max(0.0, evaluator.best_scores[task] - final[task])
        for task in final
    }
    # Only call something "retention" when the task was actually acquired.
    learned_retention = {
        task: final[task] / score
        for task, score in acquisition.items()
        if score > 0.0
    }
    return acquisition, final, forgetting, learned_retention


def _run_single(seed: int, variant: str) -> SeedResultV2:
    use_stdp = variant == "reward_plus_stdp"
    use_consolidation = variant == "reward_consolidation_rewiring"
    use_rewiring = variant in {"reward_plus_rewiring", "reward_consolidation_rewiring"}
    experiment = v1._build_single(seed, use_stdp=use_stdp, use_consolidation=use_consolidation)
    evaluator = ContinualMemoryEvaluator()
    seen: list[SequentialTask] = []

    generator = v1.ActivityBiasedCandidateGenerator(
        config=v1.ActivityBiasedCandidateConfig(max_candidates=48, pool_size=10)
    )
    structural = v1.StructuralPlasticityManager(
        experiment.tracker,
        config=v1.StructuralPlasticityConfig(
            min_age_cycles=1,
            stale_steps=6,
            reward_threshold=0.0,
            protected_stability=0.65,
            max_rewire_per_cycle=2,
            regrow_weight=0.08,
        ),
    )

    for stage, task in enumerate(v1._tasks(seed)):
        experiment.train_task(task, epochs=1)
        if use_rewiring:
            candidates = generator.generate(
                experiment.tracker.synapses,
                step=experiment.network.step_index,
                neuron_ids=experiment.network.neurons,
            )
            structural.rewire(step=experiment.network.step_index, candidate_pairs=candidates)
        seen.append(task)
        evaluator.record(
            stage=stage,
            scores={t.name: experiment.evaluate_task(t) for t in seen},
        )

    acquisition, final, forgetting, learned_retention = _learned_metrics(evaluator)
    return SeedResultV2(
        seed=seed,
        acquisition_scores=acquisition,
        final_scores=final,
        forgetting=forgetting,
        learned_retention=learned_retention,
        synapse_count=experiment.tracker.synapse_count,
    )


class _DualWTAExperiment(v1._DualExperiment):
    def run_trial(self, trial: AssociativeTrial, *, training: bool) -> bool:
        self._reset_fast()
        previous = (
            self.brain_a.network.learning_enabled,
            self.brain_b.network.learning_enabled,
        )
        self.brain_a.network.set_learning_enabled(training)
        self.brain_b.network.set_learning_enabled(training)
        counts = {self.out_a: 0, self.out_b: 0}
        strengths = {self.out_a: 0.0, self.out_b: 0.0}
        try:
            for index in range(self.RESPONSE_STEPS):
                result = self.system.step(currents_a=trial.stimulus if index == 0 else None)
                for neuron_id in result.brain_b.fired:
                    if neuron_id in counts:
                        counts[neuron_id] += 1
                for neuron_id, strength in result.brain_b.firing_strengths:
                    if neuron_id in strengths:
                        strengths[neuron_id] += strength
            active = [
                (counts[neuron_id], strengths[neuron_id], neuron_id)
                for neuron_id in counts
                if counts[neuron_id] > 0
            ]
            prediction = max(active, key=lambda item: (item[0], item[1], -item[2]))[2] if active else None
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


def _run_dual(seed: int) -> SeedResultV2:
    experiment = _DualWTAExperiment(seed)
    evaluator = ContinualMemoryEvaluator()
    seen: list[SequentialTask] = []
    for stage, task in enumerate(v1._tasks(seed)):
        experiment.train_task(task)
        experiment.maintain()
        seen.append(task)
        evaluator.record(stage=stage, scores={t.name: experiment.evaluate(t) for t in seen})
    acquisition, final, forgetting, learned_retention = _learned_metrics(evaluator)
    return SeedResultV2(
        seed=seed,
        acquisition_scores=acquisition,
        final_scores=final,
        forgetting=forgetting,
        learned_retention=learned_retention,
        synapse_count=experiment.brain_a.tracker.synapse_count + experiment.brain_b.tracker.synapse_count,
        bridge_count=experiment.bridge.synapse_count,
    )


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(len(ordered) - 1, lo + 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean", "std", "sem", "min", "p25", "median", "p75", "max")}
    std = pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": fmean(values),
        "std": std,
        "sem": std / sqrt(len(values)) if len(values) > 1 else 0.0,
        "min": min(values),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "max": max(values),
    }


def _aggregate(name: str, runs: list[SeedResultV2]) -> dict[str, object]:
    learned_instances = sum(run.learned_task_count for run in runs)
    forgotten_instances = sum(run.forgotten_learned_tasks for run in runs)
    learned_retention_values = [
        value
        for run in runs
        for value in run.learned_retention.values()
    ]
    return {
        "name": name,
        "runs": len(runs),
        "acquisition_accuracy": _stats([run.acquisition_accuracy for run in runs]),
        "final_accuracy": _stats([run.final_accuracy for run in runs]),
        "learned_task_retention": _stats(learned_retention_values),
        "mean_forgetting": _stats([run.mean_forgetting for run in runs]),
        "learned_task_instances": learned_instances,
        "forgotten_learned_task_instances": forgotten_instances,
        "catastrophic_forgetting_rate": (
            forgotten_instances / learned_instances if learned_instances else 0.0
        ),
        "per_task_acquisition": {
            task: _stats([run.acquisition_scores.get(task, 0.0) for run in runs])
            for task in ("task_A", "task_B", "task_C")
        },
        "per_task_final_accuracy": {
            task: _stats([run.final_scores.get(task, 0.0) for run in runs])
            for task in ("task_A", "task_B", "task_C")
        },
        "raw": [run.as_dict() for run in runs],
    }


def run_benchmark(*, runs: int = 100, seed_start: int = 0) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be >= 1")
    buckets: dict[str, list[SeedResultV2]] = {name: [] for name in v1.VARIANT_NAMES}
    for offset in range(runs):
        seed = seed_start + offset
        for name in v1.VARIANT_NAMES[:-1]:
            buckets[name].append(_run_single(seed, name))
        buckets["dual_brain_full"].append(_run_dual(seed))
    return {
        "experiment": "catastrophic_interference_v2",
        "runs_per_variant": runs,
        "variants": len(v1.VARIANT_NAMES),
        "total_model_runs": runs * len(v1.VARIANT_NAMES),
        "task_order": ["task_A", "task_B", "task_C"],
        "metrics_note": "Retention is computed only for task instances whose post-training acquisition score was above zero.",
        "results": [_aggregate(name, buckets[name]) for name in v1.VARIANT_NAMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquisition-aware DrosoMath interference benchmark")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(runs=args.runs, seed_start=args.seed_start), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
