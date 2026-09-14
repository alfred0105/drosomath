from __future__ import annotations

import json

from .benchmarks import BenchmarkVariant, ContinualAblationBenchmark
from .core import (
    ActivityBiasedCandidateConfig,
    ActivityBiasedCandidateGenerator,
    ConsolidationConfig,
    MemoryConsolidator,
    PlasticityTracker,
    RewardWeightRule,
    SpikingNetwork,
    STDPlasticity,
    STDPRule,
    StructuralPlasticityConfig,
    StructuralPlasticityManager,
    SynapseState,
)
from .experiments import AssociativeTrial, SequentialMemoryExperiment, SequentialTask


def _tasks() -> tuple[SequentialTask, ...]:
    return (
        SequentialTask(
            name="pattern_A",
            train_trials=(AssociativeTrial({0: 1.0}, 4),) * 6,
            eval_trials=(AssociativeTrial({0: 1.0}, 4),),
        ),
        SequentialTask(
            name="pattern_B",
            train_trials=(AssociativeTrial({1: 1.0}, 5),) * 6,
            eval_trials=(AssociativeTrial({1: 1.0}, 5),),
        ),
    )


def _build_experiment(
    *,
    use_stdp: bool,
    use_consolidation: bool,
) -> SequentialMemoryExperiment:
    synapses = [
        SynapseState(0, 2, 0.60),
        SynapseState(1, 3, 0.60),
        SynapseState(2, 4, 0.60),
        SynapseState(3, 5, 0.60),
        SynapseState(6, 7, 0.05),  # deliberately unused spare connection
    ]
    tracker = PlasticityTracker(
        synapses,
        reward_window=8,
        weight_rule=RewardWeightRule(learning_rate=0.03),
    )
    stdp = None
    if use_stdp:
        stdp = STDPlasticity(
            rule=STDPRule(
                potentiation_rate=0.01,
                depression_rate=0.012,
                window=8,
            )
        )
    network = SpikingNetwork.from_ids(
        range(8),
        tracker,
        threshold=0.5,
        decay=1.0,
        stdp=stdp,
    )
    consolidator = None
    if use_consolidation:
        consolidator = MemoryConsolidator(
            config=ConsolidationConfig(
                min_usage=2,
                min_reward=0.05,
                growth_rate=0.2,
                decay_rate=0.002,
            )
        )
    return SequentialMemoryExperiment(
        network,
        tracker,
        output_neurons=[4, 5],
        response_steps=3,
        consolidator=consolidator,
    )


def _rewire_after_stage(experiment: SequentialMemoryExperiment, stage: int) -> None:
    generator = ActivityBiasedCandidateGenerator(
        config=ActivityBiasedCandidateConfig(
            max_candidates=32,
            pool_size=8,
        )
    )
    candidates = generator.generate(
        experiment.tracker.synapses,
        step=experiment.network.step_index,
        neuron_ids=experiment.network.neurons,
    )
    manager = StructuralPlasticityManager(
        experiment.tracker,
        config=StructuralPlasticityConfig(
            min_age_cycles=1,
            stale_steps=1,
            max_rewire_per_cycle=1,
            regrow_weight=0.05,
        ),
    )
    manager.rewire(
        step=experiment.network.step_index,
        candidate_pairs=candidates,
    )


def main() -> None:
    benchmark = ContinualAblationBenchmark(_tasks(), epochs_per_task=1)
    variants = (
        BenchmarkVariant(
            "reward_only",
            lambda: _build_experiment(use_stdp=False, use_consolidation=False),
        ),
        BenchmarkVariant(
            "reward_plus_stdp",
            lambda: _build_experiment(use_stdp=True, use_consolidation=False),
        ),
        BenchmarkVariant(
            "reward_plus_consolidation",
            lambda: _build_experiment(use_stdp=False, use_consolidation=True),
        ),
        BenchmarkVariant(
            "reward_stdp_consolidation_rewiring",
            lambda: _build_experiment(use_stdp=True, use_consolidation=True),
            after_stage=_rewire_after_stage,
        ),
    )
    result = benchmark.run(variants)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
