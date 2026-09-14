import unittest

from drosomath.core import ConsolidationConfig, MemoryConsolidator, SynapseState


class MemoryConsolidationTests(unittest.TestCase):
    def test_repeated_rewarded_use_increases_stability(self) -> None:
        synapse = SynapseState(1, 2, 0.5, usage_count=5, reward_ema=0.8)
        consolidator = MemoryConsolidator(
            config=ConsolidationConfig(min_usage=3, min_reward=0.1, growth_rate=0.5)
        )

        strengthened, weakened = consolidator.consolidate([synapse])

        self.assertEqual((strengthened, weakened), (1, 0))
        self.assertGreater(synapse.stability, 0.0)

    def test_usage_without_positive_reward_does_not_consolidate(self) -> None:
        synapse = SynapseState(1, 2, 0.5, stability=0.2, usage_count=100, reward_ema=0.0)
        consolidator = MemoryConsolidator(
            config=ConsolidationConfig(min_usage=3, min_reward=0.1, decay_rate=0.05)
        )

        strengthened, weakened = consolidator.consolidate([synapse])

        self.assertEqual((strengthened, weakened), (0, 1))
        self.assertAlmostEqual(synapse.stability, 0.15)


if __name__ == "__main__":
    unittest.main()
