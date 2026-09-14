import unittest

from drosomath.core import SynapseState


class SynapseStateTests(unittest.TestCase):
    def test_record_use_updates_usage_and_reward(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)

        synapse.record_use(step=10, reward=1.0, reward_alpha=0.5)

        self.assertEqual(synapse.usage_count, 1)
        self.assertEqual(synapse.last_used_step, 10)
        self.assertAlmostEqual(synapse.reward_ema, 0.5)

    def test_decay_weight_respects_floor(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.1)

        synapse.decay_weight(0.9, floor=0.05)

        self.assertAlmostEqual(synapse.weight, 0.05)

    def test_pruned_synapse_stops_updating(self) -> None:
        synapse = SynapseState(pre_id=1, post_id=2, weight=0.5)
        synapse.prune()
        synapse.record_use(step=1, reward=1.0)
        synapse.decay_weight(0.5)

        self.assertFalse(synapse.alive)
        self.assertEqual(synapse.usage_count, 0)
        self.assertEqual(synapse.weight, 0.5)


if __name__ == "__main__":
    unittest.main()
