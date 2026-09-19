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
    SymbolLearningConfig,
    SymbolLearningSession,
)
from drosomath.malecns.loader import MaleCNSConnectome  # noqa: E402
from drosomath.malecns.symbol_credit_interference import (  # noqa: E402
    SymbolCreditInterferenceAudit,
    classify_primary_hypotheses,
    jaccard,
)
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


def make_connectome(n: int = 320) -> MaleCNSConnectome:
    posts = np.column_stack(
        (np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n)
    ).reshape(-1)
    signed = np.where(np.arange(2 * n) % 3 == 0, -2.0, 3.0).astype(np.float32)
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


def fixed_interface(connectome):
    random = SymbolInterface(
        connectome,
        SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7),
    )
    sensory = np.concatenate(tuple(random.sensory_populations.values()))
    excluded = {int(value) for value in sensory}
    candidates = [index for index in range(connectome.neuron_count) if index not in excluded]
    config = SymbolInterfaceConfig(
        sensory_population_size=8,
        output_population_size=8,
        seed=7,
        output_selection="dynamic_generic",
    )
    return SymbolInterface.with_output_populations(
        connectome,
        config,
        {
            symbol: np.asarray(candidates[index * 8:(index + 1) * 8], dtype=np.int32)
            for index, symbol in enumerate(SYMBOLS)
        },
        source="dynamic_generic",
    )


class SymbolCreditInterferencePhaseF1B2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()

    def _audit(self):
        return SymbolCreditInterferenceAudit(self.connectome, max_route_health_requests=2)

    def _observe(self, audit, *, target, channel, edges, hops, deltas, direction):
        audit.observe_update(
            target=target,
            decision="B",
            channel=channel,
            edge_indices=np.asarray(edges, dtype=np.int32),
            hops=np.asarray(hops, dtype=np.int8),
            actual_deltas=np.asarray(deltas, dtype=np.float32),
            eligibility=np.ones(len(edges), dtype=np.float32),
            path_polarities=np.ones(len(edges), dtype=np.float32),
            requested_direction=direction,
        )

    def test_jaccard_and_empty_set_handling(self):
        self.assertEqual(jaccard({1, 2}, {2, 3}), 1 / 3)
        self.assertEqual(jaccard(set(), set()), 0.0)
        self.assertEqual(jaccard(set(), {1}), 0.0)

    def test_positive_negative_edge_sets_and_hop_split(self):
        audit = self._audit()
        self._observe(audit, target="A", channel="symbol/D", edges=[0, 1], hops=[1, 2], deltas=[0.2, 0.1], direction=1)
        self._observe(audit, target="B", channel="symbol/D", edges=[1, 2], hops=[2, 1], deltas=[-0.1, -0.2], direction=-1)
        report = audit.report()["symbol/D"]
        self.assertEqual(report["edge_overlap"]["positive_unique_edges"], 2)
        self.assertEqual(report["edge_overlap"]["negative_unique_edges"], 2)
        self.assertEqual(report["edge_overlap"]["intersection_count"], 1)
        self.assertEqual(report["hop_specific"]["1hop"]["jaccard_positive_negative"], 0.0)
        self.assertEqual(report["hop_specific"]["2hop"]["jaccard_positive_negative"], 1.0)

    def test_signed_delta_accumulation_and_cancellation(self):
        audit = self._audit()
        self._observe(audit, target="A", channel="symbol/D", edges=[4], hops=[1], deltas=[0.4], direction=1)
        self._observe(audit, target="B", channel="symbol/D", edges=[4], hops=[1], deltas=[-0.3], direction=-1)
        cancellation = audit.report()["symbol/D"]["cancellation"]
        self.assertAlmostEqual(cancellation["overlap_edge_total_abs_delta"], 0.7, places=5)
        self.assertAlmostEqual(cancellation["overlap_edge_abs_net_delta"], 0.1, places=5)
        self.assertAlmostEqual(cancellation["cancellation_fraction"], 1 - 0.1 / 0.7, places=5)

    def test_context_aggregation_and_presynaptic_overlap(self):
        audit = self._audit()
        self._observe(audit, target="A", channel="symbol/D", edges=[6, 8], hops=[1, 1], deltas=[0.2, 0.2], direction=1)
        self._observe(audit, target="B", channel="symbol/D", edges=[8, 10], hops=[1, 1], deltas=[-0.2, -0.2], direction=-1)
        report = audit.report()["symbol/D"]
        self.assertIn("target/A:positive", report["contexts"])
        self.assertIn("target/B:negative", report["contexts"])
        self.assertEqual(report["pairwise_context_overlap"]["target/A:positive|target/B:negative"], 1 / 3)
        self.assertGreaterEqual(report["presynaptic_overlap"]["intersection_count"], 1)

    def test_update_yield_and_zero_update_fraction(self):
        audit = self._audit()
        self._observe(audit, target="A", channel="symbol/A", edges=[], hops=[], deltas=[], direction=1)
        self._observe(audit, target="B", channel="symbol/A", edges=[0], hops=[1], deltas=[0.2], direction=1)
        self._observe(audit, target="C", channel="symbol/A", edges=[1], hops=[1], deltas=[0.0], direction=-1)
        report = audit.report()["symbol/A"]
        self.assertEqual(report["positive_requests"]["direction_requests"], 2)
        self.assertEqual(report["positive_requests"]["positive_zero_update_count"], 1)
        self.assertEqual(report["negative_requests"]["negative_zero_update_count"], 1)
        self.assertEqual(report["positive_requests"]["total_edge_updates"], 1)

    def test_route_health_aggregation(self):
        class Health:
            active_plastic_candidate_edges = 4
            useful_plastic_edges = 3
            plastic_structural_opportunity = 2.0
            realized_plastic_capacity = 1.0
            plastic_engagement_efficiency = 0.5
            useful_frozen_edges = 7

        audit = self._audit()
        signal = type("Signal", (), {"nonzero_directions": lambda self: {"symbol/A": 1.0}})()
        audit.observe_route_health(
            target="A",
            signal=signal,
            health_by_channel={"symbol/A": Health()},
        )
        report = audit.report()["symbol/A"]["route_health"]["target/A:positive"]
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["useful_frozen_edges"], 7.0)

    def test_d_strong_conflict_and_a_starvation_comparison(self):
        audit = self._audit()
        for _ in range(5):
            self._observe(audit, target="A", channel="symbol/A", edges=[], hops=[], deltas=[], direction=1)
        for channel in ("symbol/B", "symbol/D"):
            for _ in range(5):
                self._observe(audit, target="A", channel=channel, edges=[0, 1, 2, 3], hops=[1, 1, 2, 2], deltas=[0.1] * 4, direction=1)
        for _ in range(5):
            self._observe(audit, target="A", channel="symbol/D", edges=[0, 1, 2, 3], hops=[1, 1, 2, 2], deltas=[0.1] * 4, direction=1)
            self._observe(audit, target="B", channel="symbol/D", edges=[0, 1, 2, 3], hops=[1, 1, 2, 2], deltas=[-0.1] * 4, direction=-1)
        hypotheses = classify_primary_hypotheses(audit.report())
        self.assertTrue(hypotheses["A_credit_starvation_supported"])
        self.assertTrue(hypotheses["D_sign_conflict_supported"])
        self.assertEqual(hypotheses["primary_bottleneck"], "both")

    def test_diagnostics_off_and_on_preserve_learning(self):
        interface = fixed_interface(self.connectome)
        config = SymbolLearningConfig(duration_ms=1.0, training_trials=4, evaluation_repetitions=1)
        off_brain = PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=41,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=41),
        )
        audit = self._audit()
        on_brain = PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=41,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=41),
        )
        captured = []

        def observer(**kwargs):
            captured.append({
                "edges": np.asarray(kwargs["edge_indices"]).copy(),
                "hops": np.asarray(kwargs["hops"]).copy(),
                "deltas": np.asarray(kwargs["actual_deltas"]).copy(),
            })
            audit.observe_update(**kwargs)

        off = SymbolLearningSession(off_brain, interface, config=config)
        on = SymbolLearningSession(
            on_brain,
            interface,
            config=config,
            directional_telemetry_observer=observer,
        )
        off_result = off.train_trial("A")
        on_result = on.train_trial("A")
        self.assertEqual(off_result.decision, on_result.decision)
        self.assertEqual(off_result.directional_error, on_result.directional_error)
        np.testing.assert_array_equal(off_brain.plasticity.multiplier, on_brain.plasticity.multiplier)
        np.testing.assert_array_equal(off_brain.plasticity.stability, on_brain.plasticity.stability)
        np.testing.assert_array_equal(off_brain.plasticity.plastic_mask, on_brain.plasticity.plastic_mask)
        self.assertEqual(off_brain.plasticity.plastic_edge_count, on_brain.plasticity.plastic_edge_count)
        self.assertEqual(off_brain.rng.bit_generator.state, on_brain.rng.bit_generator.state)
        observed_edges = set(int(edge) for event in captured for edge in event["edges"])
        reported_edges = set(int(edge) for edge in on_result.directional_update["updated_edge_indices"])
        self.assertEqual(observed_edges, reported_edges)
        observed_abs_delta = sum(float(np.abs(event["deltas"]).sum()) for event in captured)
        self.assertAlmostEqual(
            observed_abs_delta,
            float(on_result.directional_update["sum_abs_delta"]),
            places=6,
        )

    def test_no_external_decoder_or_keyboard_teacher(self):
        source = inspect.getsource(__import__(
            "drosomath.malecns.symbol_credit_interference", fromlist=["x"]
        ))
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("keyboard_teacher", source)
        self.assertNotIn("keyboard_learning", source)


if __name__ == "__main__":
    unittest.main()
