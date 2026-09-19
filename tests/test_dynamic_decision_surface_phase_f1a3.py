from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolSession,
    build_dynamic_generic_interface,
    collect_dynamic_candidate_features,
    partition_dynamic_generic_candidates,
    select_dynamic_generic_candidates,
)
from drosomath.malecns.dynamic_surface import DynamicCandidateFeatures  # noqa: E402
from drosomath.malecns.loader import MaleCNSConnectome  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


def make_connectome(n: int = 320) -> MaleCNSConnectome:
    posts = np.column_stack(
        (np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n)
    ).reshape(-1)
    signed = np.full(2 * n, 3.0, dtype=np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(10_000, 10_000 + n, dtype=np.int64),
        indptr=np.arange(0, 2 * n + 1, 2, dtype=np.int64),
        post_indices=posts,
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.full(n, 5.0, dtype=np.float32),
        presynaptic_sign=np.ones(n, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * n, dtype=object),
        min_connection_synapses=5,
    )


class DynamicDecisionSurfacePhaseF1A3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.config = SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
            output_selection="dynamic_generic",
        )
        random_config = SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
            output_selection="random_indegree",
        )
        random_interface = SymbolInterface(cls.connectome, random_config)
        sensory = set(int(value) for values in random_interface.sensory_populations.values() for value in values)
        available = [index for index in range(cls.connectome.neuron_count) if index not in sensory]
        cls.interface = SymbolInterface.with_output_populations(
            cls.connectome,
            cls.config,
            {
                symbol: np.asarray(available[index * 8 : (index + 1) * 8], dtype=np.int32)
                for index, symbol in enumerate(SYMBOLS)
            },
            source="dynamic_generic",
        )

    def test_dynamic_candidate_ranking_is_label_independent(self):
        runs = [
            {"input_symbol": "A", "brain_seed": 7, "active_neuron_indices": [10, 11]},
            {"input_symbol": "B", "brain_seed": 7, "active_neuron_indices": [10, 12]},
            {"input_symbol": "C", "brain_seed": 11, "active_neuron_indices": [10, 13]},
        ]
        swapped = [{**run, "input_symbol": {"A": "D", "B": "C", "C": "B"}[run["input_symbol"]]} for run in runs]
        left, _ = collect_dynamic_candidate_features(self.connectome, runs, sensory_indices=[])
        right, _ = collect_dynamic_candidate_features(self.connectome, swapped, sensory_indices=[])
        self.assertEqual([(f.neuron_index, f.distinct_symbol_count_active, f.distinct_seed_count_active) for f in left],
                         [(f.neuron_index, f.distinct_symbol_count_active, f.distinct_seed_count_active) for f in right])

    def test_ranking_priority_is_correct(self):
        features = [
            DynamicCandidateFeatures(1, 10001, 2, 3, 8),
            DynamicCandidateFeatures(2, 10002, 3, 1, 1),
            DynamicCandidateFeatures(3, 10003, 3, 2, 1),
            DynamicCandidateFeatures(4, 10004, 3, 2, 4),
        ]
        selected = select_dynamic_generic_candidates(features, selected_count=4)
        self.assertEqual([feature.neuron_index for feature in selected], [4, 3, 2, 1])

    def test_sensory_neurons_are_excluded(self):
        runs = [{"input_symbol": "A", "brain_seed": 7, "active_neuron_indices": [1, 2, 3]}]
        features, count = collect_dynamic_candidate_features(
            self.connectome, runs, sensory_indices=[1, 2]
        )
        self.assertEqual(count, 1)
        self.assertEqual(features[0].neuron_index, 3)

    def test_selected_count_exact_and_insufficient_pool_fails(self):
        features = [DynamicCandidateFeatures(i, 10_000 + i, 1, 1, 1) for i in range(128)]
        self.assertEqual(len(select_dynamic_generic_candidates(features)), 128)
        with self.assertRaisesRegex(ValueError, "required"):
            select_dynamic_generic_candidates(features[:127])

    def test_partition_is_deterministic_and_uses_local_rng(self):
        features = [DynamicCandidateFeatures(i, 10_000 + i, 1, 1, 1) for i in range(128)]
        rng = np.random.default_rng(55)
        before = copy.deepcopy(rng.bit_generator.state)
        left = partition_dynamic_generic_candidates(features, seed=17)
        right = partition_dynamic_generic_candidates(features, seed=17)
        self.assertEqual(before, rng.bit_generator.state)
        for symbol in SYMBOLS:
            np.testing.assert_array_equal(left[symbol], right[symbol])
            self.assertEqual(len(left[symbol]), 32)
        merged = np.concatenate(tuple(left.values()))
        self.assertEqual(len(np.unique(merged)), 128)

    def test_dynamic_interface_has_zero_overlap_and_frozen_surface(self):
        sensory = np.concatenate(tuple(self.interface.sensory_populations.values()))
        output = np.concatenate(tuple(self.interface.output_populations.values()))
        self.assertEqual(len(np.intersect1d(sensory, output)), 0)
        self.assertEqual(self.interface.output_selection, "dynamic_generic")
        for symbol in SYMBOLS:
            np.testing.assert_array_equal(
                self.interface.output_populations[symbol],
                self.interface.output_populations[symbol],
            )

    def test_heldout_seed_is_not_a_selection_input(self):
        discovery = [{"input_symbol": "A", "brain_seed": 7, "active_neuron_indices": [10]}]
        features, _ = collect_dynamic_candidate_features(self.connectome, discovery, sensory_indices=[])
        self.assertEqual({run["brain_seed"] for run in discovery}, {7})
        self.assertEqual(len(features), 1)

    def test_each_validation_symbol_gets_a_fresh_brain(self):
        from tests.run_dynamic_decision_surface_phase_f1a3 import _run_surface

        config = SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7)
        interface = SymbolInterface(self.connectome, config)
        rows, persistent, states = _run_surface(self.connectome, interface, config, (7,))
        self.assertTrue(persistent)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(states[(7, "A")] == states[(7, symbol)] for symbol in SYMBOLS))

    def test_readiness_is_per_symbol_not_pooled(self):
        counts = {"A": 0, "B": 3, "C": 2, "D": 0}
        self.assertFalse(all(counts[symbol] >= 2 for symbol in SYMBOLS))
        self.assertTrue(sum(counts.values()) >= 2)

    def test_no_external_trainable_decoder_in_dynamic_surface(self):
        source = inspect.getsource(__import__("drosomath.malecns.dynamic_surface", fromlist=["x"]))
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("softmax", source.lower())
        self.assertNotIn("linear", source.lower())


if __name__ == "__main__":
    unittest.main()
