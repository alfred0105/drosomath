import importlib.util
import unittest

from drosomath.core import ConsolidationConfig as CoreConsolidationConfig
from drosomath.core import MemoryConsolidator as CoreMemoryConsolidator
from drosomath.core import SynapseState


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class CoreMemoryConsolidationTests(unittest.TestCase):
    def test_repeated_rewarded_use_increases_stability(self) -> None:
        synapse = SynapseState(1, 2, 0.5, usage_count=5, reward_ema=0.8)
        consolidator = CoreMemoryConsolidator(
            config=CoreConsolidationConfig(min_usage=3, min_reward=0.1, growth_rate=0.5)
        )

        strengthened, weakened = consolidator.consolidate([synapse])

        self.assertEqual((strengthened, weakened), (1, 0))
        self.assertGreater(synapse.stability, 0.0)

    def test_usage_without_positive_reward_does_not_consolidate(self) -> None:
        synapse = SynapseState(1, 2, 0.5, stability=0.2, usage_count=100, reward_ema=0.0)
        consolidator = CoreMemoryConsolidator(
            config=CoreConsolidationConfig(min_usage=3, min_reward=0.1, decay_rate=0.05)
        )

        strengthened, weakened = consolidator.consolidate([synapse])

        self.assertEqual((strengthened, weakened), (0, 1))
        self.assertAlmostEqual(synapse.stability, 0.15)


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required")
class WholeBrainMemoryConsolidationTests(unittest.TestCase):
    def test_protected_rule_changes_stable_edge_less(self) -> None:
        import numpy as np

        from drosomath.whole_brain import PlasticStateConfig, SparsePlasticityState
        from drosomath.whole_brain.memory_consolidation import ProtectedRewardRule

        state = SparsePlasticityState(2, config=PlasticStateConfig(plastic_fraction=1.0))
        state.usage_ema[:] = 1.0
        state.eligibility[:] = 1.0
        state.stability[:] = np.asarray([0.0, 0.95], dtype=np.float32)
        before = state.multiplier.copy()

        ProtectedRewardRule(
            learning_rate=0.1,
            positive_protection=0.8,
            negative_protection=0.95,
        ).apply(state, reward=-1.0)

        fresh_delta = abs(float(state.multiplier[0] - before[0]))
        stable_delta = abs(float(state.multiplier[1] - before[1]))
        self.assertGreater(fresh_delta, stable_delta)
        self.assertGreater(stable_delta, 0.0)

    def test_consolidator_selects_reward_stable_used_edges(self) -> None:
        import numpy as np

        from drosomath.whole_brain import PlasticStateConfig, SparsePlasticityState
        from drosomath.whole_brain.memory_consolidation import ConsolidationConfig, MemoryConsolidator

        state = SparsePlasticityState(4, config=PlasticStateConfig(plastic_fraction=1.0))
        state.stability[:] = np.asarray([0.3, 0.2, 0.01, 0.0], dtype=np.float32)
        state.usage_ema[:] = np.asarray([0.8, 0.4, 0.8, 1.0], dtype=np.float32)
        state.multiplier[:] = np.asarray([1.4, 1.1, 1.0, 1.5], dtype=np.float32)
        before = state.stability.copy()

        stats = MemoryConsolidator(
            ConsolidationConfig(top_fraction=0.5, boost=0.5, min_stability=0.005)
        ).consolidate(state)

        self.assertEqual(stats["candidate_edges"], 3)
        self.assertEqual(stats["consolidated_edges"], 2)
        self.assertGreater(float(state.stability[0]), float(before[0]))
        self.assertEqual(float(state.stability[3]), 0.0)

    def test_homeostasis_can_protect_stable_edge(self) -> None:
        import numpy as np

        from drosomath.whole_brain import OutgoingBudgetNormalizer, PlasticStateConfig, SparsePlasticityState

        state = SparsePlasticityState(2, config=PlasticStateConfig(plastic_fraction=1.0))
        state.multiplier[:] = 2.0
        state.stability[:] = np.asarray([1.0, 0.0], dtype=np.float32)
        indptr = np.asarray([0, 2], dtype=np.int64)
        base = np.asarray([10.0, 10.0], dtype=np.float32)

        OutgoingBudgetNormalizer(strength=1.0, stability_protection=1.0).normalize_presynaptic(
            state, indptr=indptr, base_abs=base
        )

        self.assertAlmostEqual(float(state.multiplier[0]), 2.0, places=6)
        self.assertLess(float(state.multiplier[1]), 2.0)

    def test_replay_scheduler_interleaves_at_requested_interval(self) -> None:
        import numpy as np

        from drosomath.whole_brain.memory_consolidation import ReplayConfig, ReplayScheduler

        scheduler = ReplayScheduler(ReplayConfig(interval=4, max_prior_tasks=2))
        self.assertFalse(scheduler.should_replay(3, 1))
        self.assertTrue(scheduler.should_replay(4, 1))
        self.assertFalse(scheduler.should_replay(4, 0))
        rng = np.random.default_rng(1)
        for _ in range(20):
            idx = scheduler.choose_prior_index(rng, 4)
            self.assertIn(idx, (2, 3))


if __name__ == "__main__":
    unittest.main()
