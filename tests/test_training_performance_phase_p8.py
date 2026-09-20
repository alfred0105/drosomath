from __future__ import annotations

import copy
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.whole_brain.directional_modulation import (
    DirectionalModulationConfig,
    PlasticityController,
)
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.malecns.working_memory_learning import _compact_directional


def make_graph():
    # 0 -> 1 -> 4 and an unrelated 2 -> 5 branch.
    signed = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(100, 106, dtype=np.int64),
        indptr=np.asarray([0, 1, 2, 3, 3, 3, 3], dtype=np.int64),
        post_indices=np.asarray([1, 4, 5], dtype=np.int32),
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.ones(6, dtype=np.float32),
        presynaptic_sign=np.ones(6, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * 6, dtype=object),
        min_connection_synapses=5,
    )


def make_real_brain():
    graph = make_graph()
    state = SparsePlasticityState(
        3,
        config=PlasticStateConfig(plastic_fraction=1.0, seed=11),
    )
    state.eligibility[:] = 1.0
    return types.SimpleNamespace(
        np=np,
        connectome=graph,
        plasticity=state,
        _recent_presynaptic={0, 1, 2},
    )


def snapshot(brain):
    state = brain.plasticity
    return (
        state.multiplier.copy(),
        state.usage_ema.copy(),
        state.eligibility.copy(),
        state.stability.copy(),
        state.plastic_mask.copy(),
    )


class TrainingPerformanceP8Test(unittest.TestCase):
    def test_full_and_summary_have_identical_learning_state(self):
        full = make_real_brain()
        summary = make_real_brain()
        controller = PlasticityController(
            DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        )
        signal = LearningSignal(reward=0.0, directional_error={"out": 1.0})
        context = {"out": np.asarray([4], dtype=np.int32)}
        full_update = controller.apply_learning_signal(
            full, signal, context, telemetry_level="full"
        )
        summary_update = controller.apply_learning_signal(
            summary, signal, context, telemetry_level="summary"
        )
        for left, right in zip(snapshot(full), snapshot(summary)):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(full_update.unique_edge_updates, summary_update.unique_edge_updates)
        self.assertEqual(full_update.hop_counts, summary_update.hop_counts)
        self.assertTrue(len(full_update.updated_edge_indices))
        self.assertEqual(len(summary_update.updated_edge_indices), 0)
        self.assertIn("updated_edge_indices", _compact_directional(full_update))
        self.assertNotIn("updated_edge_indices", _compact_directional(summary_update))

    def test_plastic_row_index_preserves_order_and_invalidates_only_allocation(self):
        brain = make_real_brain()
        controller = PlasticityController()
        brain._recent_presynaptic = {2, 0}
        np.testing.assert_array_equal(controller._active_edges(brain), [0, 2])
        index = controller._plastic_row_index(brain)
        generation = brain.plasticity.allocation_generation
        self.assertEqual(index.allocation_generation, generation)
        self.assertIs(controller._plastic_row_index(brain), index)
        brain.plasticity.multiplier[0] += 0.25
        brain.plasticity.stability[0] = 0.5
        self.assertIs(controller._plastic_row_index(brain), index)
        brain.plasticity.retire_edges([1])
        self.assertGreater(brain.plasticity.allocation_generation, generation)
        new_index = controller._plastic_row_index(brain)
        self.assertIsNot(new_index, index)
        self.assertEqual(list(new_index.edge_indices), [0, 2])

    def test_reference_active_edges_match_cached_path(self):
        brain = make_real_brain()
        brain._recent_presynaptic = {2, 0}
        cached = PlasticityController(plastic_row_cache_enabled=True)._active_edges(brain)
        reference = PlasticityController(plastic_row_cache_enabled=False)._active_edges(brain)
        np.testing.assert_array_equal(cached, reference)

    def test_prospective_compact_index_matches_reference(self):
        brain = make_real_brain()
        candidate = np.asarray([0, 1], dtype=np.int32)
        context = np.asarray([4], dtype=np.int32)
        config = DirectionalModulationConfig(two_hop_credit_mode="prospective_anatomical")
        fast = PlasticityController(config, prospective_index_enabled=True)
        reference = PlasticityController(config, prospective_index_enabled=False)
        left = fast._route_credit_edges(brain, candidate, context, discover_upstream_from_candidates=True)
        right = reference._route_credit_edges(brain, candidate, context, discover_upstream_from_candidates=True)
        for left_value, right_value in zip(
            (left.edges, left.hops, left.path_polarities, left.weights, left.effective_influence),
            (right.edges, right.hops, right.path_polarities, right.weights, right.effective_influence),
        ):
            np.testing.assert_array_equal(left_value, right_value)
        self.assertEqual(left.ambiguous_path_edges_skipped, right.ambiguous_path_edges_skipped)

    def test_allocation_generation_does_not_change_on_learned_state_updates(self):
        state = SparsePlasticityState(4, config=PlasticStateConfig(plastic_fraction=0.5, seed=3))
        generation = state.allocation_generation
        state.multiplier[0] += 0.1
        state.stability[0] = 0.3
        state.eligibility[0] = 1.0
        state.usage_ema[0] = 0.5
        self.assertEqual(state.allocation_generation, generation)


if __name__ == "__main__":
    unittest.main()
