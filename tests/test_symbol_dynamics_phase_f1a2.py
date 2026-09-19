from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolSession,
    allocation_fingerprint,
    burst_outlier_symbols,
    classify_output_observability,
    count_candidate_dynamic_pool,
)
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


class SymbolDynamicsPhaseF1A2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.config = SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
            dt_ms=0.2,
            default_stimulus_rate_hz=205.0,
        )
        cls.interface = SymbolInterface(cls.connectome, cls.config)

    def test_observability_categories(self):
        self.assertEqual(classify_output_observability(2, 3), "reliably_observable")
        self.assertEqual(classify_output_observability(1, 3), "intermittently_observable")
        self.assertEqual(classify_output_observability(0, 3), "silent")

    def test_burst_outlier_calculation(self):
        events = burst_outlier_symbols({"A": 10, "B": 100, "C": 11, "D": 9})
        self.assertEqual([event["symbol"] for event in events], ["B"])
        self.assertAlmostEqual(events[0]["ratio"], 10.0)

    def test_candidate_dynamic_pool_counts_contexts_and_excludes_sensory(self):
        pool = count_candidate_dynamic_pool(
            {"A": {1, 2, 3, 8}, "B": {2, 3, 4}, "C": {3, 4, 5}, "D": {3, 6}},
            excluded_indices={1},
        )
        self.assertEqual(pool["active_for_at_least_1_symbol"], 6)
        self.assertEqual(pool["active_for_at_least_2_symbols"], 3)
        self.assertEqual(pool["active_for_at_least_3_symbols"], 1)
        self.assertEqual(pool["active_for_all_4_symbols"], 1)

    def test_every_symbol_gets_fresh_brain_and_matched_initial_rng(self):
        from tests.run_matched_symbol_dynamics_phase_f1a2 import _run_matched_condition

        rows, states, persistent, _ = _run_matched_condition(
            self.connectome,
            self.interface,
            self.config,
            seed=7,
            duration_ms=10.0,
        )
        self.assertTrue(persistent)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(states["A"] == states[symbol] for symbol in SYMBOLS))
        self.assertEqual({row["input_symbol"] for row in rows}, set(SYMBOLS))

    def test_execution_order_cannot_change_matched_setup(self):
        from tests.run_matched_symbol_dynamics_phase_f1a2 import _run_matched_condition

        first, _, _, _ = _run_matched_condition(
            self.connectome, self.interface, self.config, seed=11, duration_ms=10.0
        )
        second, _, _, _ = _run_matched_condition(
            self.connectome, self.interface, self.config, seed=11, duration_ms=10.0
        )
        self.assertEqual(first, second)

    def test_different_brain_seeds_are_distinct(self):
        states = []
        for seed in (7, 11, 19):
            brain = PlasticMaleCNSBrain(
                self.connectome,
                params=FlyBrainParams(dt_ms=0.2),
                seed=seed,
                plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=seed),
            )
            states.append(copy.deepcopy(brain.rng.bit_generator.state))
        self.assertEqual(len({repr(state) for state in states}), 3)

    def test_interface_allocation_remains_identical(self):
        left = allocation_fingerprint(self.interface)
        right = allocation_fingerprint(SymbolInterface(self.connectome, self.config))
        self.assertEqual(left, right)

    def test_diagnostics_preserve_persistent_plastic_state_and_do_not_learn(self):
        brain = PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=7,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=7),
        )
        before = brain.plasticity.multiplier.copy()
        with patch.object(
            PlasticMaleCNSBrain,
            "learn_from_reward",
            side_effect=AssertionError("F.1A.2 must not learn"),
        ):
            result = SymbolSession(brain, self.interface).present(
                symbol="A", duration_ms=2.0, learn=False
            )
        np.testing.assert_array_equal(brain.plasticity.multiplier, before)
        self.assertEqual(result.input_symbol, "A")

    def test_no_decision_behavior_remains_unchanged(self):
        rates = {symbol: 0.0 for symbol in SYMBOLS}
        self.assertEqual(self.interface.decision_surface.decide(rates), "NO_DECISION")


if __name__ == "__main__":
    unittest.main()
