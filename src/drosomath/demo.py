from __future__ import annotations

from .core import (
    ActivityBiasedCandidateConfig,
    ActivityBiasedCandidateGenerator,
    PlasticityTracker,
    RewardWeightRule,
    SpikingNetwork,
    StructuralPlasticityConfig,
    StructuralPlasticityManager,
    SynapseState,
)


def main() -> None:
    synapses = [
        SynapseState(0, 1, 0.60),
        SynapseState(1, 2, 0.60),
        SynapseState(3, 2, 0.10),  # deliberately unused edge
    ]
    tracker = PlasticityTracker(
        synapses,
        reward_window=8,
        weight_rule=RewardWeightRule(learning_rate=0.05),
    )
    network = SpikingNetwork.from_ids(
        range(4),
        tracker,
        threshold=0.5,
        decay=0.9,
    )

    print("DrosoMath executable v0 demo")
    print("initial:", [(s.pre_id, s.post_id, round(s.weight, 3)) for s in tracker.synapses])

    for result in network.run(steps=3, stimulus={0: 1.0}):
        print(
            f"step={result.step} fired={result.fired} "
            f"transfers={result.transferred_synapses}"
        )

    credited = tracker.apply_reward(reward=1.0, step=network.step_index)
    print("rewarded synapses:", credited)
    print("learned:", [(s.pre_id, s.post_id, round(s.weight, 3)) for s in tracker.synapses])

    generator = ActivityBiasedCandidateGenerator(
        config=ActivityBiasedCandidateConfig(max_candidates=16, pool_size=4)
    )
    candidates = generator.generate(
        tracker.synapses,
        step=network.step_index,
        neuron_ids=network.neurons,
    )
    structural = StructuralPlasticityManager(
        tracker,
        config=StructuralPlasticityConfig(
            min_age_cycles=1,
            stale_steps=1,
            max_rewire_per_cycle=1,
            regrow_weight=0.05,
        ),
    )
    rewired = structural.rewire(
        step=network.step_index,
        candidate_pairs=candidates,
    )
    print("rewired:", rewired)
    print("final:", [(s.pre_id, s.post_id, round(s.weight, 3)) for s in tracker.synapses])


if __name__ == "__main__":
    main()
