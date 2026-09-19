from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    NO_DECISION,
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolPresentationResult,
    SymbolSession,
    audit_symbol_reachability,
    symbol_decision_surface_ready,
    symbol_f1b_ready,
)
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.whole_brain import PlasticStateConfig


def make_connectome(n: int = 320) -> MaleCNSConnectome:
    posts = np.column_stack(
        (np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n)
    ).reshape(-1)
    indptr = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    signed = np.where(np.arange(2 * n) % 3 == 0, -2.0, 3.0).astype(np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(10_000, 10_000 + n, dtype=np.int64),
        indptr=indptr,
        post_indices=posts.astype(np.int32),
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.full(n, 5.0, dtype=np.float32),
        presynaptic_sign=np.ones(n, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * n, dtype=object),
        min_connection_synapses=5,
    )


class SymbolInterfacePhaseF1ATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.config = SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7)
        cls.interface = SymbolInterface(cls.connectome, cls.config)

    def test_exact_vocabulary(self):
        self.assertEqual(tuple(self.interface.encoder.symbols), SYMBOLS)

    def test_same_seed_is_exactly_reproducible(self):
        left = SymbolInterface(self.connectome, self.config)
        right = SymbolInterface(self.connectome, self.config)
        for symbol in SYMBOLS:
            np.testing.assert_array_equal(left.sensory_populations[symbol], right.sensory_populations[symbol])
            np.testing.assert_array_equal(left.output_populations[symbol], right.output_populations[symbol])

    def test_sensory_size_and_no_duplicates(self):
        values = np.concatenate(tuple(self.interface.sensory_populations.values()))
        self.assertTrue(all(len(self.interface.sensory_populations[symbol]) == 8 for symbol in SYMBOLS))
        self.assertEqual(len(np.unique(values)), len(values))

    def test_sensory_mutual_disjoint(self):
        for left in SYMBOLS:
            for right in SYMBOLS:
                if left < right:
                    self.assertEqual(len(np.intersect1d(
                        self.interface.sensory_populations[left],
                        self.interface.sensory_populations[right],
                    )), 0)

    def test_output_equal_size_and_mutual_disjoint(self):
        values = np.concatenate(tuple(self.interface.output_populations.values()))
        self.assertTrue(all(len(self.interface.output_populations[symbol]) == 8 for symbol in SYMBOLS))
        self.assertEqual(len(np.unique(values)), len(values))

    def test_sensory_and_output_disjoint(self):
        sensory = np.concatenate(tuple(self.interface.sensory_populations.values()))
        output = np.concatenate(tuple(self.interface.output_populations.values()))
        self.assertEqual(len(np.intersect1d(sensory, output)), 0)

    def test_indices_are_real_valid_connectome_indices(self):
        all_indices = np.concatenate(
            tuple(self.interface.sensory_populations.values())
            + tuple(self.interface.output_populations.values())
        )
        self.assertGreaterEqual(int(all_indices.min()), 0)
        self.assertLess(int(all_indices.max()), self.connectome.neuron_count)
        body_ids = np.concatenate(
            tuple(self.interface.encoder.body_id_populations.values())
            + tuple(self.interface.decision_surface.body_id_populations.values())
        )
        self.assertTrue(set(int(x) for x in body_ids).issubset(set(int(x) for x in self.connectome.body_ids)))

    def test_unknown_symbol_fails_clearly(self):
        with self.assertRaisesRegex(KeyError, "unknown symbol"):
            self.interface.encoder.indices_for("?")

    def test_all_zero_is_no_decision(self):
        self.assertEqual(
            self.interface.decision_surface.decide({symbol: 0.0 for symbol in SYMBOLS}),
            NO_DECISION,
        )

    def test_unique_highest_is_selected(self):
        rates = {symbol: 0.0 for symbol in SYMBOLS}
        rates["C"] = 4.0
        self.assertEqual(self.interface.decision_surface.decide(rates), "C")

    def test_exact_tie_has_no_arbitrary_selection(self):
        rates = {symbol: 0.0 for symbol in SYMBOLS}
        rates["A"] = rates["B"] = 4.0
        self.assertEqual(self.interface.decision_surface.decide(rates), NO_DECISION)

    def test_zero_output_makes_decision_surface_not_ready(self):
        silent = {
            symbol: [SymbolPresentationResult(symbol, {}, NO_DECISION, 0, {})]
            for symbol in SYMBOLS
        }
        self.assertFalse(symbol_decision_surface_ready(silent))
        self.assertFalse(symbol_f1b_ready(sensory_interface_ready=True, decision_surface_ready=False))

    def test_nonzero_output_can_make_decision_surface_ready(self):
        active = {
            symbol: [SymbolPresentationResult(symbol, {}, symbol, 1, {})]
            for symbol in SYMBOLS
        }
        self.assertTrue(symbol_decision_surface_ready(active))
        self.assertTrue(symbol_f1b_ready(sensory_interface_ready=True, decision_surface_ready=True))

    def test_f1b_readiness_cannot_ignore_silent_output(self):
        silent = {
            symbol: [SymbolPresentationResult(symbol, {}, NO_DECISION, 0, {})]
            for symbol in SYMBOLS
        }
        self.assertFalse(symbol_f1b_ready(
            sensory_interface_ready=True,
            decision_surface_ready=symbol_decision_surface_ready(silent),
        ))

    def test_decision_consumes_no_rng(self):
        rng = np.random.default_rng(77)
        before = copy.deepcopy(rng.bit_generator.state)
        self.interface.decision_surface.decide({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0})
        self.assertEqual(before, rng.bit_generator.state)

    def test_allocation_does_not_consume_brain_rng(self):
        rng = np.random.default_rng(123)
        before = copy.deepcopy(rng.bit_generator.state)
        SymbolInterface(self.connectome, self.config)
        self.assertEqual(before, rng.bit_generator.state)

    def test_observational_session_preserves_persistent_learning_state(self):
        brain = self._make_brain()
        before = {
            "multiplier": brain.plasticity.multiplier.copy(),
            "usage": brain.plasticity.usage_ema.copy(),
            "eligibility": brain.plasticity.eligibility.copy(),
            "stability": brain.plasticity.stability.copy(),
            "mask": brain.plasticity.plastic_mask.copy(),
            "overrides": {key: value.copy() for key, value in brain.plasticity.allocation_overrides().items()},
            "budget": brain.plasticity.plastic_edge_count,
        }
        result = SymbolSession(brain, self.interface).present(symbol="A", duration_ms=2.0, stimulus_rate_hz=205.0)
        self.assertEqual(result.input_symbol, "A")
        np.testing.assert_array_equal(brain.plasticity.multiplier, before["multiplier"])
        np.testing.assert_array_equal(brain.plasticity.usage_ema, before["usage"])
        np.testing.assert_array_equal(brain.plasticity.eligibility, before["eligibility"])
        np.testing.assert_array_equal(brain.plasticity.stability, before["stability"])
        np.testing.assert_array_equal(brain.plasticity.plastic_mask, before["mask"])
        self.assertEqual(brain.plasticity.plastic_edge_count, before["budget"])
        for key, values in before["overrides"].items():
            np.testing.assert_array_equal(brain.plasticity.allocation_overrides()[key], values)

    def test_learn_true_is_not_silently_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "F.1B"):
            SymbolSession(self._make_brain(), self.interface).present(symbol="A", learn=True)

    def test_no_trainable_external_decoder_in_symbol_module(self):
        source = inspect.getsource(__import__("drosomath.malecns.symbol_interface", fromlist=["x"]))
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("softmax", source.lower())
        self.assertNotIn("linear", source.lower())

    def test_reachability_uses_real_graph_without_mutating_it(self):
        before_indptr = self.connectome.indptr.copy()
        before_posts = self.connectome.post_indices.copy()
        audit = audit_symbol_reachability(self.connectome, self.interface)
        self.assertEqual(set(audit), set(SYMBOLS))
        np.testing.assert_array_equal(self.connectome.indptr, before_indptr)
        np.testing.assert_array_equal(self.connectome.post_indices, before_posts)

    def test_reachability_hops_are_unique_and_bounded(self):
        audit = audit_symbol_reachability(self.connectome, self.interface)
        for input_symbol in SYMBOLS:
            for output_symbol in SYMBOLS:
                row = audit[input_symbol][output_symbol]
                self.assertLessEqual(row["hop1_output_neurons"], row["hop2_output_neurons"])
                self.assertLessEqual(row["hop2_output_neurons"], row["hop3_output_neurons"])
                self.assertLessEqual(row["hop3_output_neurons"], 8)

    def test_synthetic_one_two_three_hop_and_shortest_path(self):
        class Encoder:
            def indices_for(self, symbol):
                return np.asarray([0], dtype=np.int32)

        class Interface:
            encoder = Encoder()
            output_populations = {
                "A": np.asarray([3], dtype=np.int32),
                "B": np.asarray([2], dtype=np.int32),
                "C": np.asarray([4], dtype=np.int32),
                "D": np.asarray([5], dtype=np.int32),
            }

        graph = make_connectome(8)
        graph.indptr = np.asarray([0, 1, 3, 4, 5, 5, 5, 5, 5], dtype=np.int64)
        graph.post_indices = np.asarray([1, 2, 3, 3, 4], dtype=np.int32)
        audit = audit_symbol_reachability(graph, Interface())
        row = audit["A"]
        self.assertEqual(row["A"]["hop1_output_neurons"], 0)
        self.assertEqual(row["A"]["hop2_output_neurons"], 1)
        self.assertEqual(row["A"]["hop3_output_neurons"], 1)
        self.assertEqual(row["A"]["shortest_reachable_hop"], 2)
        self.assertEqual(row["B"]["shortest_reachable_hop"], 2)
        self.assertEqual(row["C"]["shortest_reachable_hop"], 3)
        self.assertIsNone(row["D"]["shortest_reachable_hop"])

    def _make_brain(self):
        return PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=11,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=11),
        )


if __name__ == "__main__":
    unittest.main()
