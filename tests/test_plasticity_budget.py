import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drosomath.malecns.checkpoint import restore_learning_checkpoint, save_learning_checkpoint
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.whole_brain.plasticity_budget import PlasticityBudgetManager


def state_with_four_plastic():
    state = SparsePlasticityState(10, config=PlasticStateConfig(plastic_fraction=1.0, seed=4))
    state.retire_edges([4, 5, 6, 7, 8, 9])
    return state


class PlasticStateBudgetTests(unittest.TestCase):
    def test_promote_existing_frozen_edge(self):
        state = state_with_four_plastic()
        result = state.exchange_plastic_edges([4], [0])
        self.assertTrue(state.plastic_mask[4]); self.assertFalse(state.plastic_mask[0])
        self.assertEqual(result["budget_delta"], 0)

    def test_retire_existing_plastic_edge(self):
        state = state_with_four_plastic()
        before = float(state.multiplier[0]); result = state.retire_edges([0])
        self.assertFalse(state.plastic_mask[0]); self.assertEqual(float(state.multiplier[0]), before)
        self.assertEqual(result["budget_delta"], -1)

    def test_atomic_exchange_keeps_budget_constant(self):
        state = state_with_four_plastic(); before = state.plastic_edge_count
        state.exchange_plastic_edges([4, 5], [2, 3])
        self.assertEqual(state.plastic_edge_count, before)

    def test_promotions_require_equal_retirements(self):
        with self.assertRaises(ValueError):
            state_with_four_plastic().exchange_plastic_edges([4], [])

    def test_invalid_and_duplicate_indices_rejected(self):
        with self.assertRaises(IndexError):
            state_with_four_plastic().exchange_plastic_edges([99], [0])
        with self.assertRaises(ValueError):
            state_with_four_plastic().exchange_plastic_edges([4, 4], [0, 1])

    def test_already_plastic_and_already_frozen_status_rejected(self):
        state = state_with_four_plastic()
        with self.assertRaises(ValueError):
            state.exchange_plastic_edges([0], [1])
        with self.assertRaises(ValueError):
            state.exchange_plastic_edges([4], [4])
        with self.assertRaises(ValueError):
            state.retire_edges([4])

    def test_protected_manual_retirement_requires_override(self):
        state = state_with_four_plastic(); state.stability[0] = 0.9
        with self.assertRaises(ValueError):
            state.retire_edges([0])
        state.retire_edges([0], allow_protected=True)
        self.assertFalse(state.plastic_mask[0])

    def test_protected_edge_not_selected_and_low_value_donors_preferred(self):
        state = state_with_four_plastic()
        state.stability[0] = 0.9
        state.usage_ema[1] = 0.8
        state.usage_ema[2] = 0.01
        state.usage_ema[3] = 0.02
        state.multiplier[1] = 1.8
        donors = PlasticityBudgetManager().choose_retirement_candidates(state, 2)
        np.testing.assert_array_equal(donors, [2, 3])

    def test_strongly_modified_edge_is_less_replaceable(self):
        state = state_with_four_plastic()
        state.multiplier[0] = 2.0
        state.multiplier[1] = 1.0
        state.usage_ema[0] = state.usage_ema[1] = 0.0
        donors = PlasticityBudgetManager().choose_retirement_candidates(state, 1)
        self.assertEqual(int(donors[0]), 1)

    def test_anatomy_count_and_functional_multiplier_survive_retirement(self):
        state = state_with_four_plastic(); before = state.multiplier.copy()
        state.retire_edges([0])
        self.assertEqual(state.edge_count, 10)
        self.assertEqual(float(state.multiplier[0]), float(before[0]))
        np.testing.assert_allclose(state.effective_signed_slice(np.ones(10), 0, 10), state.multiplier)

    def test_promotion_preserves_multiplier_and_is_learnable(self):
        state = state_with_four_plastic(); state.multiplier[4] = 1.7
        state.exchange_plastic_edges([4], [0])
        self.assertAlmostEqual(float(state.multiplier[4]), 1.7, places=6)
        self.assertEqual(state.record_use_indices([4]), 1)
        self.assertGreater(float(state.eligibility[4]), 0.0)

    def test_retired_edge_no_longer_receives_normal_plastic_use(self):
        state = state_with_four_plastic(); state.retire_edges([0])
        self.assertEqual(state.record_use_indices([0]), 0)
        self.assertEqual(float(state.eligibility[0]), 0.0)

    def test_lifecycle_indices_keep_dynamic_edge_state(self):
        state = state_with_four_plastic(); state.multiplier[0] = 1.4; state.usage_ema[0] = 0.5
        state.exchange_plastic_edges([4], [0])
        self.assertIn(0, state.lifecycle_indices.tolist())
        self.assertIn(4, state.lifecycle_indices.tolist())
        state.decay_episode(usage_decay=0.5)
        self.assertAlmostEqual(float(state.usage_ema[0]), 0.25, places=6)


class PlasticityBudgetManagerTests(unittest.TestCase):
    def test_manager_ranks_need_and_preserves_budget(self):
        state = state_with_four_plastic(); state.stability[0] = 0.9; state.usage_ema[1] = 0.8
        result = PlasticityBudgetManager().reallocate(state, [4, 5], need_scores=[0.2, 0.9], count=2)
        self.assertEqual(result["promoted_edges"].tolist(), [5, 4])
        self.assertEqual(result["retired_edges"].tolist(), [2, 3])
        self.assertEqual(result["budget_delta"], 0)

    def test_manager_is_deterministic(self):
        a = PlasticityBudgetManager().reallocate(state_with_four_plastic(), [4, 5], need_scores=[1.0, 1.0], count=2)
        b = PlasticityBudgetManager().reallocate(state_with_four_plastic(), [4, 5], need_scores=[1.0, 1.0], count=2)
        np.testing.assert_array_equal(a["promoted_edges"], b["promoted_edges"])
        np.testing.assert_array_equal(a["retired_edges"], b["retired_edges"])

    def test_manager_has_no_keyboard_import(self):
        from drosomath.whole_brain import plasticity_budget
        self.assertNotIn("keyboard", inspect.getsource(plasticity_budget).lower())


def fake_brain(state):
    return SimpleNamespace(
        np=np,
        connectome=SimpleNamespace(neuron_count=2, edge_count=state.edge_count, min_connection_synapses=1),
        plasticity=state,
    )


class PlasticityBudgetCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_dynamic_allocation_and_state(self):
        first_state = state_with_four_plastic(); first_state.multiplier[0] = 1.8; first_state.stability[0] = 0.2; first_state.usage_ema[0] = 0.4
        first_state.exchange_plastic_edges([4, 5], [2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget.npz"
            save_learning_checkpoint(path, brain=fake_brain(first_state))
            second_state = state_with_four_plastic()
            info = restore_learning_checkpoint(path, brain=fake_brain(second_state))
        np.testing.assert_array_equal(first_state.plastic_mask, second_state.plastic_mask)
        np.testing.assert_array_equal(first_state.plastic_indices, second_state.plastic_indices)
        np.testing.assert_array_equal(first_state.multiplier, second_state.multiplier)
        np.testing.assert_array_equal(first_state.stability, second_state.stability)
        np.testing.assert_array_equal(first_state.usage_ema, second_state.usage_ema)
        self.assertTrue(info["dynamic_allocation_restored"])

    def test_old_checkpoint_without_dynamic_overrides_restores_deterministic_mask(self):
        state = SparsePlasticityState(5, config=PlasticStateConfig(plastic_fraction=0.5, seed=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.npz"
            np.savez_compressed(
                path,
                format_version=np.array([3], dtype=np.int32), neuron_count=np.array([2]), edge_count=np.array([5]),
                changed_edge_indices=np.array([], dtype=np.int32), multipliers=np.array([], dtype=np.float32),
                stability=np.array([], dtype=np.float32), usage_ema=np.array([], dtype=np.float32),
                plastic_fraction=np.array([0.5], dtype=np.float32), plastic_seed=np.array([2]),
                completed_trials=np.array([0]), stage=np.array([""], dtype="U1"),
            )
            fresh = SparsePlasticityState(5, config=PlasticStateConfig(plastic_fraction=0.5, seed=2))
            info = restore_learning_checkpoint(path, brain=fake_brain(fresh))
        np.testing.assert_array_equal(state.plastic_mask, fresh.plastic_mask)
        self.assertFalse(info["dynamic_allocation_restored"])


if __name__ == "__main__":
    unittest.main()
