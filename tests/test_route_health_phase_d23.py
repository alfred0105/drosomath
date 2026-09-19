import unittest
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain.directional_modulation import (
    CreditEdges,
    DirectionalModulationConfig,
    PlasticityController,
)
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from tests.test_route_health import direct_brain
from tests.run_matched_route_health_phase_d23 import METRICS, result_keys_are_complete


def one_edge_brain(*, frozen=False, signed=1.0):
    graph = SimpleNamespace(
        neuron_count=3,
        edge_count=1,
        indptr=np.asarray([0, 1, 1, 1], dtype=np.int32),
        post_indices=np.asarray([2], dtype=np.int32),
        signed_synapse_counts=np.asarray([signed], dtype=np.float32),
    )
    state = SparsePlasticityState(1, config=PlasticStateConfig(plastic_fraction=1.0, seed=5))
    if frozen:
        state.retire_edges([0])
    state.eligibility[:] = 0.0
    return SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic={0})


class MatchedRouteHealthTests(unittest.TestCase):
    def test_counterfactual_probe_is_identical_for_success_and_failure(self):
        controller = PlasticityController()
        success = controller.diagnose_route_health(
            direct_brain(), LearningSignal(0.0, {"out": 1.0}, success=True), {"out": [2]}, standardized=True
        )["out"]
        failure = controller.diagnose_route_health(
            direct_brain(), LearningSignal(0.0, {"out": 1.0}, success=False), {"out": [2]}, standardized=True
        )["out"]
        self.assertEqual(success.requested_error_magnitude, 1.0)
        self.assertEqual(success.plastic_structural_opportunity, failure.plastic_structural_opportunity)
        self.assertEqual(success.frozen_structural_opportunity, failure.frozen_structural_opportunity)

    def test_probe_does_not_call_learning_or_change_state(self):
        brain = direct_brain()
        controller = PlasticityController()
        before = [value.copy() for value in (brain.plasticity.multiplier, brain.plasticity.stability, brain.plasticity.eligibility)]
        rng_before = np.random.get_state()
        controller.diagnose_route_health(
            brain, LearningSignal(0.0, {"out": 1.0}, success=None), {"out": [2]}, standardized=True
        )
        rng_after = np.random.get_state()
        for got, expected in zip((brain.plasticity.multiplier, brain.plasticity.stability, brain.plasticity.eligibility), before):
            np.testing.assert_array_equal(got, expected)
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])

    def test_structural_opportunity_excludes_eligibility_but_realized_uses_it(self):
        low = direct_brain(); high = direct_brain()
        low.plasticity.eligibility[:] = 0.0
        high.plasticity.eligibility[:] = 2.0
        signal = LearningSignal(0.0, {"out": 1.0}, success=None)
        controller = PlasticityController()
        low_health = controller.diagnose_route_health(low, signal, {"out": [2]}, standardized=True)["out"]
        high_health = controller.diagnose_route_health(high, signal, {"out": [2]}, standardized=True)["out"]
        self.assertEqual(low_health.plastic_structural_opportunity, high_health.plastic_structural_opportunity)
        self.assertEqual(low_health.realized_plastic_capacity, 0.0)
        self.assertGreater(high_health.realized_plastic_capacity, 0.0)

    def test_identical_plastic_and_frozen_route_has_identical_structural_opportunity(self):
        controller = PlasticityController()
        plastic = controller.diagnose_route_health(
            one_edge_brain(), LearningSignal(0.0, {"out": 1.0}), {"out": [2]}, standardized=True
        )["out"]
        frozen = controller.diagnose_route_health(
            one_edge_brain(frozen=True), LearningSignal(0.0, {"out": 1.0}), {"out": [2]}, standardized=True
        )["out"]
        self.assertAlmostEqual(plastic.plastic_structural_opportunity, frozen.frozen_structural_opportunity)

    def test_multiplier_headroom_and_influence_change_structural_opportunity(self):
        low = one_edge_brain(signed=1.0)
        high = one_edge_brain(signed=10.0)
        low.plasticity.multiplier[0] = low.plasticity.config.max_multiplier - 0.1
        signal = LearningSignal(0.0, {"out": 1.0})
        controller = PlasticityController()
        low_health = controller.diagnose_route_health(low, signal, {"out": [2]}, standardized=True)["out"]
        high_health = controller.diagnose_route_health(high, signal, {"out": [2]}, standardized=True)["out"]
        self.assertGreater(high_health.plastic_structural_opportunity, low_health.plastic_structural_opportunity)
        self.assertGreater(high_health.mean_effective_route_influence, low_health.mean_effective_route_influence)

    def test_hop_decay_changes_structural_opportunity(self):
        graph = SimpleNamespace(
            neuron_count=3, edge_count=2,
            indptr=np.asarray([0, 1, 2, 2], dtype=np.int32),
            post_indices=np.asarray([1, 2], dtype=np.int32),
            signed_synapse_counts=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        def brain(decay):
            state = SparsePlasticityState(2, config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
            state.eligibility[:] = 1.0
            return SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic={0}), PlasticityController(
                DirectionalModulationConfig(credit_decay_per_hop=decay)
            )
        near, near_controller = brain(0.9)
        far, far_controller = brain(0.1)
        signal = LearningSignal(0.0, {"out": 1.0})
        near_health = near_controller.diagnose_route_health(near, signal, {"out": [2]}, standardized=True)["out"]
        far_health = far_controller.diagnose_route_health(far, signal, {"out": [2]}, standardized=True)["out"]
        self.assertGreater(near_health.plastic_structural_opportunity, far_health.plastic_structural_opportunity)

    def test_duplicate_route_records_report_raw_and_unique_counts(self):
        brain = direct_brain(frozen=[0])
        controller = PlasticityController()
        original = controller._route_credit_edges

        def duplicated(*args, **kwargs):
            credit = original(*args, **kwargs)
            if not len(credit.edges):
                return credit
            influence = credit.effective_influence
            return CreditEdges(
                np.concatenate([credit.edges, credit.edges]),
                np.concatenate([credit.hops, credit.hops]),
                np.concatenate([credit.path_polarities, credit.path_polarities]),
                np.concatenate([credit.weights, credit.weights]),
                credit.ambiguous_path_edges_skipped,
                np.concatenate([influence, influence]),
            )

        controller._route_credit_edges = duplicated
        health = controller.diagnose_route_health(
            brain, LearningSignal(0.0, {"out": 1.0}), {"out": [2]}, standardized=True
        )["out"]
        self.assertEqual(health.frozen_raw_candidate_edges, 2)
        self.assertEqual(health.frozen_unique_candidate_edges, 1)

    def test_real_run_contract_requires_all_sixty_keys(self):
        self.assertEqual(len(METRICS), 19)
        incomplete = {str(index): {"attempts": 1} for index in range(59)}
        self.assertFalse(result_keys_are_complete(incomplete))
        complete = {str(index): {"attempts": 1} for index in range(60)}
        self.assertTrue(result_keys_are_complete(complete))


if __name__ == "__main__":
    unittest.main()
