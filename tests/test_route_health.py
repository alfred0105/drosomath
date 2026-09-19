import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain.directional_modulation import DirectionalModulationConfig, PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from tests.run_route_health_phase_d22 import _group


def direct_brain(*, frozen=(), signs=(1.0, -1.0, 1.0, 1.0)):
    graph = SimpleNamespace(
        neuron_count=4,
        edge_count=4,
        indptr=np.asarray([0, 4, 4, 4, 4], dtype=np.int32),
        post_indices=np.asarray([2, 2, 2, 3], dtype=np.int32),
        signed_synapse_counts=np.asarray(signs, dtype=np.float32),
    )
    state = SparsePlasticityState(4, config=PlasticStateConfig(plastic_fraction=1.0, seed=3))
    if frozen:
        state.retire_edges(frozen)
    state.eligibility[:] = 1.0
    return SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic={0})


class RouteHealthTests(unittest.TestCase):
    def test_diagnostic_does_not_change_learning_state(self):
        first = direct_brain(); second = direct_brain()
        controller = PlasticityController()
        signal = LearningSignal(-1.0, {"out": 1.0}, success=False)
        before = [value.copy() for value in (first.plasticity.multiplier, first.plasticity.stability, first.plasticity.plastic_mask, first.plasticity.eligibility)]
        controller.diagnose_route_health(first, signal, {"out": [2]})
        for got, expected in zip((first.plasticity.multiplier, first.plasticity.stability, first.plasticity.plastic_mask, first.plasticity.eligibility), before):
            np.testing.assert_array_equal(got, expected)
        controller.apply_learning_signal(first, signal, {"out": [2]})
        controller.apply_learning_signal(second, signal, {"out": [2]})
        np.testing.assert_array_equal(first.plasticity.multiplier, second.plasticity.multiplier)
        np.testing.assert_array_equal(first.plasticity.stability, second.plasticity.stability)
        np.testing.assert_array_equal(first.plasticity.plastic_mask, second.plasticity.plastic_mask)

    def test_positive_and_negative_headroom_are_directional(self):
        brain = direct_brain(); brain.plasticity.multiplier[:2] = [2.0, 0.5]
        health = PlasticityController().diagnose_route_health(brain, LearningSignal(-1, {"out": 1.0}), {"out": [2]})["out"]
        self.assertEqual(health.increase_headroom_edges, 2)
        self.assertEqual(health.decrease_headroom_edges, 1)

    def test_saturated_max_and_min_edges_are_detected(self):
        brain = direct_brain()
        brain.plasticity.multiplier[0] = brain.plasticity.config.max_multiplier
        brain.plasticity.multiplier[1] = brain.plasticity.config.min_multiplier
        brain.plasticity.multiplier[2] = brain.plasticity.config.max_multiplier
        health = PlasticityController().diagnose_route_health(brain, LearningSignal(-1, {"out": 1.0}), {"out": [2]})["out"]
        self.assertEqual(health.saturated_edges, 3)
        self.assertEqual(health.saturated_fraction, 1.0)
        self.assertEqual(health.route_health_status, "PLASTIC_ROUTE_SATURATED")

    def test_aligned_and_opposing_edges_are_counted(self):
        health = PlasticityController().diagnose_route_health(
            direct_brain(), LearningSignal(-1, {"out": 1.0}), {"out": [2]}
        )["out"]
        self.assertEqual(health.active_plastic_edges, 3)
        self.assertEqual(health.aligned_plastic_edges, 2)
        self.assertEqual(health.opposing_plastic_edges, 1)

    def test_useful_frozen_direct_edge_is_counted_and_irrelevant_excluded(self):
        health = PlasticityController().diagnose_route_health(
            direct_brain(frozen=[2]), LearningSignal(-1, {"out": 1.0}), {"out": [2]}
        )["out"]
        self.assertEqual(health.useful_frozen_edges, 1)
        self.assertGreater(health.estimated_frozen_capacity, 0.0)

    def test_two_hop_frozen_route_and_ambiguous_route(self):
        two_hop_graph = SimpleNamespace(
            neuron_count=3, edge_count=2,
            indptr=np.asarray([0, 1, 2, 2], dtype=np.int32),
            post_indices=np.asarray([1, 2], dtype=np.int32),
            signed_synapse_counts=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        state = SparsePlasticityState(2, config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
        state.retire_edges([0, 1])
        brain = SimpleNamespace(np=np, connectome=two_hop_graph, plasticity=state, _recent_presynaptic={0})
        health = PlasticityController().diagnose_route_health(brain, LearningSignal(-1, {"out": 1.0}), {"out": [2]})["out"]
        self.assertEqual(health.useful_frozen_two_hop, 1)

        ambiguous_graph = SimpleNamespace(
            neuron_count=3, edge_count=3,
            indptr=np.asarray([0, 1, 3, 3], dtype=np.int32),
            post_indices=np.asarray([1, 2, 2], dtype=np.int32),
            signed_synapse_counts=np.asarray([1.0, 1.0, -1.0], dtype=np.float32),
        )
        ambiguous_state = SparsePlasticityState(3, config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
        ambiguous_state.retire_edges([0, 1, 2])
        ambiguous = SimpleNamespace(np=np, connectome=ambiguous_graph, plasticity=ambiguous_state, _recent_presynaptic={0})
        health = PlasticityController().diagnose_route_health(ambiguous, LearningSignal(-1, {"out": 1.0}), {"out": [2]})["out"]
        self.assertGreater(health.ambiguous_path_edges_skipped, 0)

    def test_capacity_increases_with_eligibility_and_headroom(self):
        low = direct_brain(); high = direct_brain()
        high.plasticity.eligibility[:] = 2.0
        high.plasticity.multiplier[:2] = [1.0, 1.0]
        low.plasticity.multiplier[:2] = [2.0, 0.5]
        signal = LearningSignal(-1, {"out": 1.0})
        low_health = PlasticityController().diagnose_route_health(low, signal, {"out": [2]})["out"]
        high_health = PlasticityController().diagnose_route_health(high, signal, {"out": [2]})["out"]
        self.assertGreater(high_health.estimated_available_adjustment, low_health.estimated_available_adjustment)

    def test_hop_decay_reduces_two_hop_capacity(self):
        graph = SimpleNamespace(
            neuron_count=3, edge_count=2,
            indptr=np.asarray([0, 1, 2, 2], dtype=np.int32),
            post_indices=np.asarray([1, 2], dtype=np.int32),
            signed_synapse_counts=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        state = SparsePlasticityState(2, config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
        state.eligibility[:] = 1.0
        brain = SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic={0, 1})
        health = PlasticityController().diagnose_route_health(brain, LearningSignal(-1, {"out": 1.0}), {"out": [2]})["out"]
        self.assertGreater(health.total_route_credit, 0.0)

    def test_route_health_is_task_independent(self):
        from drosomath.whole_brain import directional_modulation
        self.assertNotIn("keyboard", inspect.getsource(directional_modulation).lower())

    def test_keyboard_group_aggregation_is_external(self):
        per_key = {
            "A": {"attempts": 2, "failures": 2, "zero_update_failures": 1, "status_counts": {"LOW": 2}, **{metric: 1.0 for metric in ("mean_active_plastic_edges", "mean_total_eligibility", "mean_total_route_credit", "mean_estimated_available_adjustment", "mean_useful_frozen_edges", "mean_estimated_frozen_capacity", "mean_frozen_to_plastic_capacity_ratio", "mean_saturated_fraction", "mean_sum_abs_delta")}},
            "B": {"attempts": 2, "failures": 2, "zero_update_failures": 0, "status_counts": {"OK": 2}, **{metric: 2.0 for metric in ("mean_active_plastic_edges", "mean_total_eligibility", "mean_total_route_credit", "mean_estimated_available_adjustment", "mean_useful_frozen_edges", "mean_estimated_frozen_capacity", "mean_frozen_to_plastic_capacity_ratio", "mean_saturated_fraction", "mean_sum_abs_delta")}},
        }
        group = _group(["A", "B"], per_key)
        self.assertEqual(group["sample_size"], 4)
        self.assertEqual(group["status_counts"]["LOW"], 2)

    def test_real_diagnostic_runner_forces_clean_nonadaptive_config(self):
        from tests.run_route_health_phase_d22 import build_diagnostic_config
        config = build_diagnostic_config(trials=600, seed=7)
        self.assertFalse(config.resume)
        self.assertFalse(config.adaptive_plastic_budget)
        self.assertEqual(config.trials, 600)
        self.assertEqual(config.seed, 7)


if __name__ == "__main__":
    unittest.main()
