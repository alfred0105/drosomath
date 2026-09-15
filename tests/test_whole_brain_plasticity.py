import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional whole-brain dependency")
class WholeBrainPlasticityTests(unittest.TestCase):
    def test_more_used_rewarded_edge_strengthens_more(self) -> None:
        from drosomath.whole_brain.plastic_state import SparsePlasticityState
        from drosomath.whole_brain.usage_learning import UsageRewardRule

        state = SparsePlasticityState(2)
        for _ in range(5):
            state.record_use_slice(0, 1, usage_alpha=0.5)
        state.record_use_slice(1, 2, usage_alpha=0.5)

        before = state.multiplier.copy()
        stats = UsageRewardRule(learning_rate=0.1).apply(state, reward=1.0)

        delta_frequent = float(state.multiplier[0] - before[0])
        delta_rare = float(state.multiplier[1] - before[1])
        self.assertEqual(stats.edge_updates, 2)
        self.assertGreater(delta_frequent, delta_rare)
        self.assertGreater(delta_rare, 0.0)

    def test_negative_reward_weakens_without_flipping_anatomical_sign(self) -> None:
        import numpy as np

        from drosomath.whole_brain.plastic_state import SparsePlasticityState
        from drosomath.whole_brain.usage_learning import UsageRewardRule

        state = SparsePlasticityState(1)
        state.record_use_slice(0, 1, usage_alpha=1.0)
        UsageRewardRule(learning_rate=0.2).apply(state, reward=-1.0)

        base = np.asarray([-12.0], dtype=np.float32)
        effective = state.effective_signed_slice(base, 0, 1)
        self.assertLess(float(state.multiplier[0]), 1.0)
        self.assertLess(float(effective[0]), 0.0)

    def test_outgoing_budget_prevents_runaway_strength(self) -> None:
        import numpy as np

        from drosomath.whole_brain.homeostasis import OutgoingBudgetNormalizer
        from drosomath.whole_brain.plastic_state import SparsePlasticityState

        state = SparsePlasticityState(2)
        state.multiplier[:] = 2.0
        indptr = np.asarray([0, 2], dtype=np.int64)
        base_abs = np.asarray([10.0, 10.0], dtype=np.float32)

        stats = OutgoingBudgetNormalizer(strength=1.0).normalize_presynaptic(
            state,
            indptr=indptr,
            base_abs=base_abs,
        )

        self.assertEqual(stats.neurons_touched, 1)
        self.assertAlmostEqual(float(state.multiplier[0]), 1.0, places=6)
        self.assertAlmostEqual(float(state.multiplier[1]), 1.0, places=6)

    def test_plastic_fraction_can_freeze_anatomical_edges(self) -> None:
        from drosomath.whole_brain.plastic_state import (
            PlasticStateConfig,
            SparsePlasticityState,
        )

        state = SparsePlasticityState(
            8,
            config=PlasticStateConfig(plastic_fraction=0.0),
        )
        credited = state.record_use_slice(0, 8)
        self.assertEqual(credited, 0)
        self.assertEqual(state.plastic_edge_count, 0)
        self.assertTrue((state.multiplier == 1.0).all())


if __name__ == "__main__":
    unittest.main()
