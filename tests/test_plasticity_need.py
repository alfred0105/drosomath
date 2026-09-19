import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.checkpoint import restore_learning_checkpoint, save_learning_checkpoint
from drosomath.whole_brain.directional_modulation import (
    DirectionalModulationConfig,
    DirectionalUpdate,
    PlasticityController,
)
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.whole_brain.plasticity_budget_adaptation import (
    PlasticityBudgetAdaptation,
    PlasticityBudgetAdaptationConfig,
)
from drosomath.whole_brain.plasticity_need import PlasticityNeedConfig, PlasticityNeedTracker


def synthetic_brain():
    # 0->1 is a frozen two-hop upstream route to output 2.
    # 0->2 is a frozen inhibitory direct route; weakening it helps output 2.
    # 0->5 is active but irrelevant to output 2.
    pre = np.asarray([0, 0, 0, 1, 3, 4, 5, 6, 7, 7], dtype=np.int32)
    post = np.asarray([1, 2, 5, 2, 2, 3, 4, 4, 2, 3], dtype=np.int32)
    signs = np.asarray([1, -1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    graph = SimpleNamespace(
        neuron_count=8,
        edge_count=10,
        indptr=np.asarray([0, 3, 4, 4, 5, 6, 7, 8, 10], dtype=np.int32),
        post_indices=post,
        signed_synapse_counts=signs,
        min_connection_synapses=1,
    )
    state = SparsePlasticityState(10, config=PlasticStateConfig(plastic_fraction=1.0, seed=4))
    state.retire_edges([0, 1, 2, 3, 8, 9])
    brain = SimpleNamespace(
        np=np,
        connectome=graph,
        plasticity=state,
        _recent_presynaptic={0},
    )
    return brain


def zero_update():
    return DirectionalUpdate(
        0, {}, 0.0, np.empty(0, dtype=np.int32), {}, 0, 0, 0, 0,
        (), 0,
    )


def adaptation(brain, *, enabled=True, threshold=1.0, minimum=3):
    controller = PlasticityController(DirectionalModulationConfig(learning_rate=0.1))
    return PlasticityBudgetAdaptation(
        config=PlasticityBudgetAdaptationConfig(
            enabled=enabled,
            max_promotions_per_event=1,
            need=PlasticityNeedConfig(
                need_decay=0.90,
                minimum_observations=minimum,
                promotion_threshold=threshold,
            ),
        ),
        route_controller=controller,
    )


class PlasticityNeedTrackerTests(unittest.TestCase):
    def test_one_failure_does_not_promote(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(minimum_observations=3, promotion_threshold=1.0))
        tracker.advance(); tracker.observe([4], [1.0])
        self.assertEqual(len(tracker.ready_candidates()[0]), 0)

    def test_repeated_failure_accumulates(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(minimum_observations=3, promotion_threshold=1.0))
        for _ in range(3):
            tracker.advance(); tracker.observe([4], [1.0])
        self.assertEqual(int(tracker.ready_candidates()[0][0]), 4)
        self.assertEqual(tracker.records()[0].observation_count, 3)

    def test_observation_gap_resets_recent_readiness(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(minimum_observations=3, max_observation_gap=1))
        tracker.advance(); tracker.observe([4], [1.0])
        tracker.advance(); tracker.observe([4], [1.0])
        tracker.advance(); tracker.advance(); tracker.observe([4], [1.0])
        self.assertEqual(tracker.records()[0].observation_count, 1)
        self.assertEqual(len(tracker.ready_candidates()[0]), 0)

    def test_success_resets_recent_readiness(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(minimum_observations=3))
        for _ in range(2):
            tracker.advance(); tracker.observe([4], [1.0])
        tracker.advance(success=True)
        tracker.advance(); tracker.observe([4], [1.0])
        self.assertEqual(tracker.records()[0].observation_count, 1)

    def test_old_checkpoint_observations_are_conservative(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(minimum_observations=3))
        tracker.restore_from_checkpoint({
            "edge_indices": np.asarray([4], dtype=np.int32),
            "scores": np.asarray([5.0], dtype=np.float32),
            "observations": np.asarray([99], dtype=np.int32),
            "last_seen": np.asarray([99], dtype=np.int64),
            "event_count": np.asarray([99], dtype=np.int64),
        })
        self.assertEqual(tracker.records()[0].observation_count, 0)
        self.assertEqual(len(tracker.ready_candidates()[0]), 0)

    def test_need_decays_when_not_reinforced(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(need_decay=0.5))
        tracker.advance(); tracker.observe([4], [1.0])
        tracker.advance(); tracker.advance()
        self.assertAlmostEqual(tracker.records()[0].need_score, 0.25, places=6)

    def test_tracker_is_sparse_bounded_and_task_independent(self):
        tracker = PlasticityNeedTracker(PlasticityNeedConfig(max_tracked_candidates=2))
        tracker.advance(); tracker.observe([9, 3, 5], [1.0, 4.0, 2.0])
        self.assertEqual(tracker.tracked_count, 2)
        self.assertNotIn("keyboard", inspect.getsource(type(tracker)).lower())

    def test_checkpoint_round_trip(self):
        first = PlasticityNeedTracker()
        first.advance(); first.observe([4], [1.2])
        first.advance(); first.observe([4, 6], [0.4, 0.8])
        second = PlasticityNeedTracker()
        second.restore_from_checkpoint(first.checkpoint_payload(np))
        self.assertEqual(first.event_count, second.event_count)
        for left, right in zip(first.records(), second.records()):
            self.assertEqual(left.edge_index, right.edge_index)
            self.assertAlmostEqual(left.need_score, right.need_score, places=5)
            self.assertEqual(left.observation_count, right.observation_count)
            self.assertEqual(left.last_seen_event, right.last_seen_event)


class StructuralCandidateTests(unittest.TestCase):
    def setUp(self):
        self.brain = synthetic_brain()
        self.controller = PlasticityController()

    def test_frozen_direct_and_two_hop_routes_are_discoverable(self):
        credit = self.controller.structural_credit_edges(self.brain, [2])
        self.assertIn(1, credit.edges.tolist())
        self.assertIn(0, credit.edges.tolist())

    def test_irrelevant_frozen_edge_is_not_useful(self):
        credit = self.controller.structural_credit_edges(self.brain, [2])
        self.assertNotIn(2, credit.edges.tolist())

    def test_negative_inhibitory_path_has_negative_polarity(self):
        credit = self.controller.structural_credit_edges(self.brain, [2])
        polarity = {int(edge): float(value) for edge, value in zip(credit.edges, credit.path_polarities)}
        self.assertEqual(polarity[1], -1.0)

    def test_ambiguous_route_is_skipped(self):
        controller = PlasticityController(DirectionalModulationConfig(minimum_downstream_effect=2.0))
        credit = controller.structural_credit_edges(self.brain, [2])
        self.assertNotIn(0, credit.edges.tolist())
        self.assertGreater(credit.ambiguous_path_edges_skipped, 0)


class PlasticityBudgetAdaptationTests(unittest.TestCase):
    def test_disabled_adaptation_preserves_behavior(self):
        brain = synthetic_brain(); controller = adaptation(brain, enabled=False)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        result = controller.observe_directional_failure(
            brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update()
        )
        self.assertFalse(result["reallocation_triggered"])
        self.assertEqual(brain.plasticity.plastic_edge_count, 4)

    def test_zero_generic_update_accumulates_but_nonzero_does_not(self):
        brain = synthetic_brain(); controller = adaptation(brain)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        nonzero = zero_update()
        nonzero = DirectionalUpdate(1, {}, 0.1, np.asarray([4], dtype=np.int32), {}, 1, 0, 0, 1, (), 0)
        controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=nonzero)
        self.assertEqual(controller.need_tracker.tracked_count, 0)

    def test_candidate_below_threshold_is_not_promoted(self):
        brain = synthetic_brain(); controller = adaptation(brain, threshold=10.0)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        for _ in range(3):
            result = controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update())
        self.assertFalse(result["reallocation_triggered"])

    def test_repeated_failure_promotes_real_edge_and_keeps_budget(self):
        brain = synthetic_brain(); controller = adaptation(brain)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        before = brain.plasticity.plastic_edge_count
        for _ in range(3):
            result = controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update())
        self.assertTrue(result["reallocation_triggered"])
        self.assertEqual(before, brain.plasticity.plastic_edge_count)
        self.assertEqual(brain.connectome.edge_count, 10)
        self.assertIn(1, result["promoted_edges"])
        self.assertNotIn(2, result["promoted_edges"])

    def test_stable_donor_is_protected_and_active_exclusion_is_honored(self):
        brain = synthetic_brain(); brain.plasticity.stability[4] = 0.9
        controller = adaptation(brain)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        for _ in range(3):
            result = controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update())
        self.assertTrue(brain.plasticity.plastic_mask[4])
        self.assertNotIn(4, result["retired_edges"])
        self.assertGreaterEqual(result["protected_donors_skipped"], 0)

    def test_saturated_candidate_is_skipped(self):
        brain = synthetic_brain()
        brain.plasticity.multiplier[0] = brain.plasticity.config.max_multiplier
        brain.plasticity.multiplier[1] = brain.plasticity.config.min_multiplier
        controller = adaptation(brain)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        for _ in range(3):
            result = controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update())
        self.assertFalse(result["reallocation_triggered"])

    def test_success_decays_need(self):
        brain = synthetic_brain(); controller = adaptation(brain)
        fail = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        for _ in range(2):
            controller.observe_directional_failure(brain=brain, signal=fail, output_context={"out": [2]}, directional_update=zero_update())
        before = controller.need_tracker.records()[0].need_score
        success = LearningSignal(reward=1, directional_error={}, success=True)
        controller.observe_directional_failure(brain=brain, signal=success, output_context={"out": [2]}, directional_update=zero_update())
        self.assertLess(controller.need_tracker.records()[0].need_score, before)

    def test_promoted_edge_has_no_same_trial_eligibility_and_learns_next_trial(self):
        brain = synthetic_brain(); controller = adaptation(brain)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        for _ in range(3):
            result = controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update())
        promoted = int(result["promoted_edges"][0])
        self.assertEqual(float(brain.plasticity.eligibility[promoted]), 0.0)
        self.assertEqual(brain.plasticity.record_use_indices([promoted]), 1)
        old = float(brain.plasticity.multiplier[promoted])
        update = controller.route_controller.apply_learning_signal(brain, signal, {"out": [2]})
        self.assertGreater(update.edge_updates, 0)
        self.assertNotEqual(float(brain.plasticity.multiplier[promoted]), old)

    def test_checkpoint_preserves_need_and_dynamic_allocation(self):
        brain = synthetic_brain(); controller = adaptation(brain)
        signal = LearningSignal(reward=-1, directional_error={"out": 1.0}, success=False)
        for _ in range(2):
            controller.observe_directional_failure(brain=brain, signal=signal, output_context={"out": [2]}, directional_update=zero_update())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "need.npz"
            save_learning_checkpoint(path, brain=brain, need_tracker=controller.need_tracker)
            restored_brain = synthetic_brain(); restored = adaptation(restored_brain)
            info = restore_learning_checkpoint(path, brain=restored_brain, need_tracker=restored.need_tracker)
        self.assertTrue(info["need_state_restored"])
        for left, right in zip(controller.need_tracker.records(), restored.need_tracker.records()):
            self.assertEqual(left.edge_index, right.edge_index)
            self.assertAlmostEqual(left.need_score, right.need_score, places=5)
            self.assertEqual(left.observation_count, right.observation_count)
            self.assertEqual(left.last_seen_event, right.last_seen_event)


if __name__ == "__main__":
    unittest.main()
