from __future__ import annotations

import inspect
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.malecns.prospective_twohop import ProspectiveTwoHopAudit
from drosomath.whole_brain.directional_modulation import (
    DirectionalModulationConfig,
    PlasticityController,
)


def make_graph():
    # 0 -> 1 -> 4 is the only bounded useful route to output 4.
    # 2 -> 5 is an active but irrelevant anatomical branch.
    posts = np.asarray([1, 4, 5], dtype=np.int32)
    signed = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(100, 106, dtype=np.int64),
        indptr=np.asarray([0, 1, 2, 3, 3, 3, 3], dtype=np.int64),
        post_indices=posts,
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.ones(6, dtype=np.float32),
        presynaptic_sign=np.ones(6, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * 6, dtype=object),
        min_connection_synapses=5,
    )


def make_brain(*, edge0_plastic=True, edge0_eligibility=1.0):
    graph = make_graph()
    state = types.SimpleNamespace(
        plastic_mask=np.asarray([edge0_plastic, False, True], dtype=np.bool_),
        eligibility=np.asarray([edge0_eligibility, 1.0, 1.0], dtype=np.float32),
        multiplier=np.asarray([0.5, 0.8, 0.5], dtype=np.float32),
        stability=np.zeros(3, dtype=np.float32),
        config=types.SimpleNamespace(min_multiplier=0.1, max_multiplier=2.0),
        plastic_edge_count=int(edge0_plastic) + 1,
    )
    return types.SimpleNamespace(
        np=np,
        connectome=graph,
        plasticity=state,
        _recent_presynaptic={0},
    )


def output_context():
    return {"out": np.asarray([4], dtype=np.int32)}


class ProspectiveTwoHopPhaseF1B3Test(unittest.TestCase):
    def test_default_mode_is_legacy(self):
        self.assertEqual(DirectionalModulationConfig().two_hop_credit_mode, "active_chain")

    def test_prospective_mode_is_explicit(self):
        config = DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        self.assertEqual(config.two_hop_credit_mode, "prospective_anatomical")

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            DirectionalModulationConfig(two_hop_credit_mode="keyboard")

    def test_legacy_active_chain_does_not_look_through_ineligible_downstream(self):
        brain = make_brain()
        controller = PlasticityController()
        credit = controller._route_credit_edges(brain, np.asarray([0], dtype=np.int32), [4])
        self.assertEqual(len(credit.edges), 0)

    def test_prospective_discovers_frozen_downstream_route(self):
        brain = make_brain()
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        credit = controller._route_credit_edges(
            brain, np.asarray([0], dtype=np.int32), [4], discover_upstream_from_candidates=True
        )
        np.testing.assert_array_equal(credit.edges, [0])
        np.testing.assert_array_equal(credit.hops, [2])

    def test_direct_route_is_same_in_both_modes(self):
        brain = make_brain()
        active = np.asarray([1], dtype=np.int32)
        legacy = PlasticityController()._route_credit_edges(brain, active, [4])
        prospective = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )._route_credit_edges(brain, active, [4], discover_upstream_from_candidates=True)
        np.testing.assert_array_equal(legacy.edges, prospective.edges)
        np.testing.assert_array_equal(legacy.hops, prospective.hops)

    def test_frozen_downstream_edge_is_not_modified(self):
        brain = make_brain()
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        before = brain.plasticity.multiplier.copy()
        update = controller.apply_learning_signal(
            brain,
            LearningSignal(reward=0.0, directional_error={"out": 1.0}),
            output_context(),
        )
        self.assertGreater(update.edge_updates, 0)
        self.assertNotEqual(float(brain.plasticity.multiplier[0]), float(before[0]))
        self.assertEqual(float(brain.plasticity.multiplier[1]), float(before[1]))

    def test_inactive_upstream_edge_is_not_candidate(self):
        brain = make_brain()
        brain._recent_presynaptic = {2}
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        self.assertEqual(len(controller._active_edges(brain)), 1)
        credit = controller._route_credit_edges(
            brain, controller._active_edges(brain), [4], discover_upstream_from_candidates=True
        )
        self.assertEqual(len(credit.edges), 0)

    def test_zero_eligibility_is_not_active_candidate(self):
        brain = make_brain(edge0_eligibility=0.0)
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        self.assertEqual(len(controller._active_edges(brain)), 0)

    def test_irrelevant_frozen_branch_is_not_selected(self):
        brain = make_brain()
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        credit = controller._route_credit_edges(
            brain, np.asarray([0], dtype=np.int32), [4], discover_upstream_from_candidates=True
        )
        self.assertNotIn(2, set(int(edge) for edge in credit.edges))

    def test_inhibitory_downstream_route_requires_weakening(self):
        brain = make_brain()
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        update = controller.apply_learning_signal(
            brain, LearningSignal(reward=0.0, directional_error={"out": 1.0}), output_context()
        )
        self.assertLess(float(brain.plasticity.multiplier[0]), 0.5)
        self.assertEqual(update.channel_hop_counts["out"], {2: 1})

    def test_positive_upstream_route_can_be_strengthened(self):
        brain = make_brain()
        brain.connectome.signed_synapse_counts[1] = 2.0
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        controller.apply_learning_signal(
            brain, LearningSignal(reward=0.0, directional_error={"out": 1.0}), output_context()
        )
        self.assertGreater(float(brain.plasticity.multiplier[0]), 0.5)

    def test_frozen_upstream_edge_is_not_modified(self):
        brain = make_brain(edge0_plastic=False)
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        before = brain.plasticity.multiplier.copy()
        update = controller.apply_learning_signal(
            brain, LearningSignal(reward=0.0, directional_error={"out": 1.0}), output_context()
        )
        self.assertEqual(update.edge_updates, 0)
        np.testing.assert_array_equal(brain.plasticity.multiplier, before)

    def test_build_reward_credit_remains_legacy_in_intervention(self):
        brain = make_brain()
        signal = LearningSignal(reward=1.0, reinforcement={"out": 1.0})
        legacy = PlasticityController().build_reward_credit(brain, signal, output_context())
        intervention = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        ).build_reward_credit(brain, signal, output_context())
        np.testing.assert_array_equal(legacy.edge_indices, intervention.edge_indices)
        np.testing.assert_array_equal(legacy.weights, intervention.weights)

    def test_anatomy_is_unchanged_by_intervention(self):
        brain = make_brain()
        posts = brain.connectome.post_indices.copy()
        indptr = brain.connectome.indptr.copy()
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        controller.apply_learning_signal(
            brain, LearningSignal(reward=0.0, directional_error={"out": 1.0}), output_context()
        )
        np.testing.assert_array_equal(brain.connectome.post_indices, posts)
        np.testing.assert_array_equal(brain.connectome.indptr, indptr)

    def test_attribution_marks_only_intervention_edges(self):
        brain = make_brain()
        seen = []
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        controller.apply_learning_signal(
            brain,
            LearningSignal(reward=0.0, directional_error={"out": 1.0}),
            output_context(),
            attribution_observer=lambda **kwargs: seen.append(kwargs),
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(set(seen[0]["current_edge_indices"]), {0})
        self.assertEqual(len(seen[0]["legacy_edge_indices"]), 0)

    def test_audit_safety_passes_for_real_candidate(self):
        brain = make_brain()
        audit = ProspectiveTwoHopAudit(brain.connectome)
        audit.observe_attribution(
            target="A", decision="B", output_context=output_context(), channel="symbol/A",
            current_edge_indices=[], current_hops=[], legacy_edge_indices=[], legacy_hops=[],
            actual_deltas=[], eligibility=[], path_polarities=[], requested_direction=1.0,
            brain=brain,
        )
        # Generic channel names are intentionally required; an unknown route
        # is ignored rather than becoming a hidden task-specific selector.
        self.assertIn("symbol/A", audit.report())

    def test_unknown_audit_channel_is_ignored(self):
        brain = make_brain()
        audit = ProspectiveTwoHopAudit(brain.connectome)
        audit.observe_attribution(
            target="A", decision="B", output_context=output_context(), channel="keyboard/A",
            current_edge_indices=[0], current_hops=[2], legacy_edge_indices=[], legacy_hops=[],
            actual_deltas=[0.1], eligibility=[1.0], path_polarities=[1.0], requested_direction=1.0,
            brain=brain,
        )
        self.assertNotIn("keyboard/A", audit.report())

    def test_audit_counts_two_hop_updates(self):
        brain = make_brain()
        audit = ProspectiveTwoHopAudit(brain.connectome)
        audit.observe_attribution(
            target="A", decision="B", output_context={"symbol/A": [4]}, channel="symbol/A",
            current_edge_indices=[0], current_hops=[2], legacy_edge_indices=[], legacy_hops=[],
            actual_deltas=[0.1], eligibility=[1.0], path_polarities=[-1.0], requested_direction=1.0,
            brain=brain,
        )
        row = audit.report()["symbol/A"]["positive"]
        self.assertEqual(row["two_hop_updates"], 1)
        self.assertEqual(row["prospective_only_unique_edges"], 1)

    def test_audit_detects_safety_violation(self):
        brain = make_brain(edge0_plastic=False)
        audit = ProspectiveTwoHopAudit(brain.connectome)
        audit.observe_attribution(
            target="A", decision="B", output_context={"symbol/A": [4]}, channel="symbol/A",
            current_edge_indices=[0], current_hops=[2], legacy_edge_indices=[], legacy_hops=[],
            actual_deltas=[0.1], eligibility=[1.0], path_polarities=[1.0], requested_direction=1.0,
            brain=brain,
        )
        self.assertFalse(audit.report()["symbol/A"]["safety"]["passed"])

    def test_audit_absorb_is_additive(self):
        brain = make_brain()
        left = ProspectiveTwoHopAudit(brain.connectome)
        right = ProspectiveTwoHopAudit(brain.connectome)
        kwargs = dict(
            target="A", decision="B", output_context={"symbol/A": [4]}, channel="symbol/A",
            current_edge_indices=[0], current_hops=[2], legacy_edge_indices=[], legacy_hops=[],
            actual_deltas=[0.1], eligibility=[1.0], path_polarities=[-1.0], requested_direction=1.0,
            brain=brain,
        )
        left.observe_attribution(**kwargs)
        right.observe_attribution(**kwargs)
        left.absorb(right)
        self.assertEqual(left.report()["symbol/A"]["positive"]["direction_requests"], 2)

    def test_controller_source_has_no_keyboard_selector(self):
        source = inspect.getsource(PlasticityController.apply_learning_signal)
        self.assertNotIn("target_symbol", source)
        self.assertNotIn("keyboard", source.lower())

    def test_mode_is_passed_to_controller(self):
        from drosomath.malecns.symbol_learning import SymbolLearningConfig
        self.assertEqual(
            SymbolLearningConfig(two_hop_credit_mode="prospective_anatomical").two_hop_credit_mode,
            "prospective_anatomical",
        )

    def test_legacy_controller_has_no_prospective_only_edges(self):
        brain = make_brain()
        seen = []
        PlasticityController().apply_learning_signal(
            brain, LearningSignal(reward=0.0, directional_error={"out": 1.0}), output_context(),
            attribution_observer=lambda **kwargs: seen.append(kwargs),
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]["legacy_edge_indices"]), len(seen[0]["current_edge_indices"]))

    def test_saturated_candidate_has_no_effective_delta(self):
        brain = make_brain()
        brain.plasticity.multiplier[0] = brain.plasticity.config.min_multiplier
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        update = controller.apply_learning_signal(
            brain, LearningSignal(reward=0.0, directional_error={"out": 1.0}), output_context()
        )
        self.assertEqual(update.channel_unique_edge_updates["out"], 0)

    def test_empty_attribution_is_still_a_request(self):
        brain = make_brain()
        audit = ProspectiveTwoHopAudit(brain.connectome)
        audit.observe_attribution(
            target="A", decision="NO_DECISION", output_context={"symbol/A": [4]}, channel="symbol/A",
            current_edge_indices=[], current_hops=[], legacy_edge_indices=[], legacy_hops=[],
            actual_deltas=[], eligibility=[], path_polarities=[], requested_direction=1.0,
            brain=brain,
        )
        row = audit.report()["symbol/A"]["positive"]
        self.assertEqual(row["direction_requests"], 1)
        self.assertEqual(row["zero_update_count"], 1)


if __name__ == "__main__":
    unittest.main()
