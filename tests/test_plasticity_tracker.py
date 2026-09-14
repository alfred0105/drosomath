import unittest

from drosomath.core import PlasticityTracker, SynapseState


class PlasticityTrackerTests(unittest.TestCase):
    def test_record_transfer_tracks_only_existing_live_synapse(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker([synapse])

        self.assertTrue(tracker.record_transfer(pre_id=1, post_id=2, step=3))
        self.assertFalse(tracker.record_transfer(pre_id=9, post_id=9, step=3))
        self.assertEqual(synapse.usage_count, 1)
        self.assertEqual(synapse.last_used_step, 3)

    def test_delayed_reward_credits_recent_path(self) -> None:
        first = SynapseState(pre_id=1, post_id=2, weight=0.5)
        second = SynapseState(pre_id=2, post_id=3, weight=0.5)
        tracker = PlasticityTracker(
            [first, second],
            reward_window=4,
            reward_alpha=0.5,
        )

        tracker.record_transfer(pre_id=1, post_id=2, step=10)
        tracker.record_transfer(pre_id=2, post_id=3, step=11)
        credited = tracker.apply_reward(reward=1.0, step=12)

        self.assertEqual(credited, 2)
        self.assertAlmostEqual(first.reward_ema, 0.5)
        self.assertAlmostEqual(second.reward_ema, 0.5)

    def test_reward_does_not_credit_expired_transfer(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker(
            [synapse],
            reward_window=2,
            reward_alpha=0.5,
        )

        tracker.record_transfer(pre_id=1, post_id=2, step=1)
        credited = tracker.apply_reward(reward=1.0, step=4)

        self.assertEqual(credited, 0)
        self.assertAlmostEqual(synapse.reward_ema, 0.0)

    def test_same_synapse_is_credited_once_per_reward_event(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker(
            [synapse],
            reward_window=4,
            reward_alpha=0.5,
        )

        tracker.record_transfer(pre_id=1, post_id=2, step=1)
        tracker.record_transfer(pre_id=1, post_id=2, step=2)
        credited = tracker.apply_reward(reward=1.0, step=2)

        self.assertEqual(credited, 1)
        self.assertEqual(synapse.usage_count, 2)
        self.assertAlmostEqual(synapse.reward_ema, 0.5)

    def test_steps_must_be_monotonic(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker([synapse])
        tracker.record_transfer(pre_id=1, post_id=2, step=5)

        with self.assertRaises(ValueError):
            tracker.apply_reward(reward=1.0, step=4)


if __name__ == "__main__":
    unittest.main()
