import unittest

from drosomath.core import (
    PlasticityTracker,
    StructuralPlasticityConfig,
    StructuralPlasticityManager,
    SynapseState,
)


class StructuralPlasticityTests(unittest.TestCase):
    def test_rewire_preserves_synapse_budget(self) -> None:
        old = SynapseState(pre_id=1, post_id=2, weight=0.1)
        keep_a = SynapseState(pre_id=2, post_id=3, weight=0.5, reward_ema=0.5)
        keep_b = SynapseState(pre_id=3, post_id=4, weight=0.5, stability=0.9)
        tracker = PlasticityTracker([old, keep_a, keep_b])
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=1,
                stale_steps=0,
                max_rewire_per_cycle=1,
                regrow_weight=0.05,
            ),
        )

        before = tracker.synapse_count
        result = manager.rewire(step=10, candidate_pairs=[(10, 11)])

        self.assertEqual(before, 3)
        self.assertEqual(tracker.synapse_count, before)
        self.assertEqual(result.pruned, ((1, 2),))
        self.assertEqual(result.regrown, ((10, 11),))
        self.assertFalse(old.alive)
        self.assertTrue(tracker.has_synapse(pre_id=10, post_id=11))

    def test_recent_synapse_is_not_pruned(self) -> None:
        recent = SynapseState(pre_id=1, post_id=2, weight=0.1)
        tracker = PlasticityTracker([recent])
        tracker.record_transfer(pre_id=1, post_id=2, step=10)
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=1,
                stale_steps=5,
                max_rewire_per_cycle=1,
            ),
        )

        result = manager.rewire(step=12, candidate_pairs=[(10, 11)])

        self.assertEqual(result.changed, 0)
        self.assertTrue(tracker.has_synapse(pre_id=1, post_id=2))

    def test_stable_memory_synapse_is_protected(self) -> None:
        stable = SynapseState(
            pre_id=1,
            post_id=2,
            weight=0.1,
            stability=0.9,
        )
        tracker = PlasticityTracker([stable])
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=1,
                protected_stability=0.8,
                max_rewire_per_cycle=1,
            ),
        )

        result = manager.rewire(step=100, candidate_pairs=[(10, 11)])

        self.assertEqual(result.changed, 0)
        self.assertTrue(stable.alive)

    def test_positive_reward_synapse_is_protected(self) -> None:
        rewarded = SynapseState(
            pre_id=1,
            post_id=2,
            weight=0.1,
            reward_ema=0.2,
        )
        tracker = PlasticityTracker([rewarded])
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=1,
                reward_threshold=0.0,
                max_rewire_per_cycle=1,
            ),
        )

        result = manager.rewire(step=100, candidate_pairs=[(10, 11)])

        self.assertEqual(result.changed, 0)
        self.assertTrue(rewarded.alive)

    def test_no_candidate_means_no_pruning(self) -> None:
        old = SynapseState(pre_id=1, post_id=2, weight=0.1)
        tracker = PlasticityTracker([old])
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=1,
                stale_steps=0,
                max_rewire_per_cycle=1,
            ),
        )

        result = manager.rewire(
            step=10,
            candidate_pairs=[(1, 2), (3, 3)],
        )

        self.assertEqual(result.changed, 0)
        self.assertEqual(tracker.synapse_count, 1)
        self.assertTrue(old.alive)

    def test_regrowth_skips_duplicates_and_self_connections(self) -> None:
        first = SynapseState(pre_id=1, post_id=2, weight=0.1)
        second = SynapseState(pre_id=3, post_id=4, weight=0.1)
        tracker = PlasticityTracker([first, second])
        manager = StructuralPlasticityManager(
            tracker,
            config=StructuralPlasticityConfig(
                min_age_cycles=1,
                stale_steps=0,
                max_rewire_per_cycle=2,
            ),
        )

        result = manager.rewire(
            step=10,
            candidate_pairs=[
                (1, 2),
                (8, 8),
                (10, 11),
                (10, 11),
                (12, 13),
            ],
        )

        self.assertEqual(result.changed, 2)
        self.assertEqual(result.regrown, ((10, 11), (12, 13)))
        self.assertEqual(tracker.synapse_count, 2)


if __name__ == "__main__":
    unittest.main()
