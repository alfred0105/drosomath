import unittest

from drosomath.core import PlasticityTracker, RewardWeightRule, SynapseState


class RewardWeightRuleTests(unittest.TestCase):
    def test_positive_reward_strengthens_recent_synapse(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker(
            [synapse],
            reward_window=4,
            weight_rule=RewardWeightRule(learning_rate=0.1),
        )

        tracker.record_transfer(pre_id=1, post_id=2, step=1)
        tracker.apply_reward(reward=1.0, step=2)

        self.assertAlmostEqual(synapse.weight, 0.6)

    def test_negative_reward_weakens_recent_synapse(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker(
            [synapse],
            reward_window=4,
            weight_rule=RewardWeightRule(learning_rate=0.1),
        )

        tracker.record_transfer(pre_id=1, post_id=2, step=1)
        tracker.apply_reward(reward=-1.0, step=2)

        self.assertAlmostEqual(synapse.weight, 0.4)

    def test_weight_is_clamped_to_rule_bounds(self) -> None:
        upper = SynapseState(pre_id=1, post_id=2, weight=0.95)
        lower = SynapseState(pre_id=3, post_id=4, weight=0.05)
        rule = RewardWeightRule(
            learning_rate=0.2,
            min_weight=0.0,
            max_weight=1.0,
        )

        rule.apply(upper, reward=1.0)
        rule.apply(lower, reward=-1.0)

        self.assertAlmostEqual(upper.weight, 1.0)
        self.assertAlmostEqual(lower.weight, 0.0)

    def test_expired_synapse_is_not_changed_by_reward(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker(
            [synapse],
            reward_window=1,
            weight_rule=RewardWeightRule(learning_rate=0.1),
        )

        tracker.record_transfer(pre_id=1, post_id=2, step=1)
        credited = tracker.apply_reward(reward=1.0, step=3)

        self.assertEqual(credited, 0)
        self.assertAlmostEqual(synapse.weight, 0.5)

    def test_tracking_only_mode_does_not_change_weight(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        tracker = PlasticityTracker([synapse], reward_window=4)

        tracker.record_transfer(pre_id=1, post_id=2, step=1)
        tracker.apply_reward(reward=1.0, step=2)

        self.assertAlmostEqual(synapse.weight, 0.5)


if __name__ == "__main__":
    unittest.main()
