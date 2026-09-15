from __future__ import annotations

import json

from .core import (
    ConsolidationConfig,
    MemoryConsolidator,
    PlasticityTracker,
    RewardWeightRule,
    SpikingNetwork,
    SynapseState,
)
from .experiments import AssociativeTrial, SequentialMemoryExperiment, SequentialTask


def build_demo() -> SequentialMemoryExperiment:
    synapses = [
        SynapseState(0, 1, 0.60),
        SynapseState(2, 3, 0.60),
    ]
    tracker = PlasticityTracker(
        synapses,
        reward_window=8,
        weight_rule=RewardWeightRule(learning_rate=0.03),
    )
    network = SpikingNetwork.from_ids(
        [0, 1, 2, 3],
        tracker,
        threshold=0.5,
        decay=1.0,
    )
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
        output_neurons=[1, 3],
        response_steps=2,
        consolidator=consolidator,
    )


def main() -> None:
    experiment = build_demo()
    tasks = [
        SequentialTask(
            name="pattern_A",
            train_trials=(AssociativeTrial({0: 1.0}, 1),) * 4,
            eval_trials=(AssociativeTrial({0: 1.0}, 1),),
        ),
        SequentialTask(
            name="pattern_B",
            train_trials=(AssociativeTrial({2: 1.0}, 3),) * 4,
            eval_trials=(AssociativeTrial({2: 1.0}, 3),),
        ),
    ]
    evaluator = experiment.run_sequence(tasks, epochs_per_task=1)
    payload = evaluator.summary()
    payload["synapses"] = [
        {
            "pre": synapse.pre_id,
            "post": synapse.post_id,
            "weight": round(synapse.weight, 6),
            "stability": round(synapse.stability, 6),
            "usage": synapse.usage_count,
            "reward_ema": round(synapse.reward_ema, 6),
        }
        for synapse in experiment.tracker.synapses
    ]
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
