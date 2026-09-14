import unittest

from drosomath.core import (
    ActivityBiasedCandidateConfig,
    ActivityBiasedCandidateGenerator,
    PlasticityTracker,
    StructuralPlasticityConfig,
    StructuralPlasticityManager,
    SynapseState,
)


class ActivityBiasedCandidateGeneratorTests(unittest.TestCase):
    def test_recent_rewarded_activity_biases_candidates(self) -> None:
        hot = SynapseState(
            pre_id=0,
            post_id=1,
            weight=0.8,
            usage_count=50,
            reward_ema=0.9,
            last_used_step=100,
        )
        cold = SynapseState(
            pre_id=2,
            post_id=3,
            weight=0.2,
            usage_count=1,
            reward_ema=0.0,
            last_used_step=20,
        )
        generator = ActivityBiasedCandidateGenerator(
            config=ActivityBiasedCandidateConfig(
                max_candidates=6,
                pool_size=4,
            )
        )

        candidates = generator.generate(
            [hot, cold],
            step=100,
            neuron_ids=range(6),
        )

        self.assertGreater(len(candidates), 0)
        self.assertTrue(any(pre_id == 0 or post_id == 1 for pre_id, post_id in candidates[:3]))

    def test_existing_and_self_connections_are_excluded(self) -> None:
        existing = SynapseState(
            pre_id=0,
            post_id=1,
            weight=0.5,
            usage_count=5,
            reward_ema=0.5,
            last_used_step=10,
        )
        generator = ActivityBiasedCandidateGenerator(
            config=ActivityBiasedCandidateConfig(
                max_candidates=20,
                pool_size=4,
            )
        )

        candidates = generator.generate(
            [existing],
            step=10,
            neuron_ids=range(4),
        )

        self.assertNotIn((0, 1), candidates)
        self.assertTrue(all(pre_id != post_id for pre_id, post_id in candidates))
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_candidate_limit_is_respected(self) -> None:
        generator = ActivityBiasedCandidateGenerator(
            config=ActivityBiasedCandidateConfig(
                max_candidates=3,
                pool_size=5,
            )
        )

        candidates = generator.generate([], step=0, neuron_ids=range(10))

        self.assertEqual(len(candidates), 3)

    def test_generated_candidates_can_drive_fixed_budget_rewire(self) -> None:
        stale = SynapseState(
            pre_id=4,
            post_id=5,
            weight=0.01,
            reward_ema=-0.5,
            age=2,
            last_used_step=0,
        )
        hot = SynapseState(
            pre_id=0,
            post_id=1,
            weight=0.8,
            reward_ema=0.8,
            usage_count=40,
            age=2,
            last_used_step=100,
        )
        tracker = PlasticityTracker([stale, hot])
        generator = ActivityBiasedCandidateGenerator(
            config=ActivityBiasedCandidateConfig(
                max_candidates=8,
                pool_size=4,
            )
        )
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=2,
                stale_steps=10,
                max_rewire_per_cycle=1,
            ),
        )
        before = tracker.synapse_count
        candidates = generator.generate(
            tracker.synapses,
            step=100,
            neuron_ids=range(6),
        )

        result = manager.rewire(step=100, candidate_pairs=candidates)

        self.assertEqual(result.changed, 1)
        self.assertEqual(tracker.synapse_count, before)
        self.assertNotIn((4, 5), {(s.pre_id, s.post_id) for s in tracker.synapses})
        self.assertEqual(len(result.regrown), 1)


if __name__ == "__main__":
    unittest.main()
