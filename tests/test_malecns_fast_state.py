import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "fast MaleCNS tests need NumPy")
class MaleCNSFastStateTests(unittest.TestCase):
    def _graph(self):
        import numpy as np
        from drosomath.malecns.loader import MaleCNSConnectome

        # 0 -> {1,2}, 1 -> 3, 2 -> 3, 3 -> 4, 4 -> 5
        body_ids = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
        indptr = np.asarray([0, 2, 3, 4, 5, 6, 6], dtype=np.int64)
        posts = np.asarray([1, 2, 3, 3, 4, 5], dtype=np.int32)
        signed = np.asarray([10, 7, 8, 6, 9, 5], dtype=np.float32)
        return MaleCNSConnectome(
            body_ids=body_ids,
            indptr=indptr,
            post_indices=posts,
            synapse_counts=np.abs(signed).astype(np.int32),
            signed_synapse_counts=signed,
            outgoing_strength=np.asarray([17, 8, 6, 9, 5, 0], dtype=np.float32),
            presynaptic_sign=np.ones(6, dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine"] * 6, dtype=object),
            metadata={"superclass": np.asarray(["visual_projection"] * 6, dtype=object)},
            min_connection_synapses=1,
        )

    def test_fast_sparse_state_matches_dense_lif_steps(self):
        import numpy as np
        from drosomath.flywire_real import FlyBrainParams
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.whole_brain import PlasticSparseFlyBrain, PlasticStateConfig

        graph = self._graph()
        params = FlyBrainParams(dt_ms=0.2)
        cfg = PlasticStateConfig(plastic_fraction=1.0, seed=4)
        fast = PlasticMaleCNSBrain(graph, params=params, seed=11, plasticity_config=cfg)
        dense = PlasticSparseFlyBrain(graph, params=params, seed=11, plasticity_config=cfg)
        stimulus = np.asarray([0], dtype=np.int32)

        for _ in range(40):
            fired_fast, transfers_fast = fast.step(
                stimulus_indices=stimulus,
                stimulus_rate_hz=1000.0,
            )
            fired_dense, transfers_dense = dense.step(
                stimulus_indices=stimulus,
                stimulus_rate_hz=1000.0,
            )
            np.testing.assert_array_equal(fired_fast, fired_dense)
            self.assertEqual(transfers_fast, transfers_dense)
            np.testing.assert_allclose(fast.v, dense.v, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(fast.g, dense.g, rtol=0.0, atol=1e-6)
            np.testing.assert_array_equal(fast.refractory_until, dense.refractory_until)

        summary = fast.fast_state_summary()
        self.assertLessEqual(summary["peak_active_state_neurons"], graph.neuron_count)
        self.assertGreater(summary["peak_active_state_neurons"], 0)

    def test_batched_fast_output_matches_dense_delay_and_learning_state(self):
        import numpy as np
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.whole_brain import PlasticSparseFlyBrain, PlasticStateConfig

        graph = self._graph()
        cfg = PlasticStateConfig(plastic_fraction=1.0, seed=9)
        fast = PlasticMaleCNSBrain(graph, seed=13, plasticity_config=cfg)
        dense = PlasticSparseFlyBrain(graph, seed=13, plasticity_config=cfg)
        fired = np.asarray([2, 0, 1], dtype=np.int32)

        transferred_fast = fast._schedule_spike_outputs(fired)
        transferred_dense = dense._schedule_spike_outputs(fired)

        self.assertEqual(transferred_fast, transferred_dense)
        for fast_slot, dense_slot in zip(fast._delay_ring, dense._delay_ring):
            np.testing.assert_allclose(fast_slot, dense_slot, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            fast.plasticity.usage_ema,
            dense.plasticity.usage_ema,
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            fast.plasticity.eligibility,
            dense.plasticity.eligibility,
            rtol=0.0,
            atol=1e-6,
        )

    def test_due_mask_returns_sorted_unique_targets(self):
        import numpy as np
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.whole_brain import PlasticStateConfig

        brain = PlasticMaleCNSBrain(
            self._graph(),
            seed=3,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=3),
        )
        brain._fast_delay_touched[0].extend(
            [
                np.asarray([3, 1, 3], dtype=np.int32),
                np.asarray([2, 1], dtype=np.int32),
            ]
        )
        np.testing.assert_array_equal(
            brain._due_indices(0),
            np.asarray([1, 2, 3], dtype=np.int32),
        )

    def test_output_session_reuses_stimulus_index_cache(self):
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.malecns.output_readout import OutputPopulation, PopulationReadout
        from drosomath.malecns.output_session import MaleCNSOutputSession
        from drosomath.whole_brain import PlasticStateConfig

        graph = self._graph()
        brain = PlasticMaleCNSBrain(
            graph,
            seed=3,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=3),
        )
        output = OutputPopulation.from_body_ids(graph, [50, 60])
        readout = PopulationReadout(output, ("A", "B"))
        session = MaleCNSOutputSession(brain, readout)
        first = session._indices_for_stimulus((10, 20))
        second = session._indices_for_stimulus((10, 20))
        self.assertIs(first, second)
        self.assertEqual(len(session._stimulus_index_cache), 1)

    def test_fast_active_cache_uses_mask_and_stays_sorted(self):
        import numpy as np
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.whole_brain import PlasticStateConfig

        graph = self._graph()
        brain = PlasticMaleCNSBrain(
            graph,
            seed=3,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=3),
        )

        first = brain._activate_indices(
            np.asarray([5, 2, 5], dtype=np.int32),
            np.asarray([3, 2], dtype=np.int32),
        )
        np.testing.assert_array_equal(first, np.asarray([2, 3, 5], dtype=np.int32))
        self.assertTrue(bool(brain._fast_active_mask[2]))
        self.assertTrue(bool(brain._fast_active_mask[3]))
        self.assertTrue(bool(brain._fast_active_mask[5]))

        second = brain._activate_indices(np.asarray([1, 5], dtype=np.int32))
        np.testing.assert_array_equal(
            second,
            np.asarray([1, 2, 3, 5], dtype=np.int32),
        )

    def test_recent_key_mastery_requires_clean_window(self):
        from drosomath.malecns.keyboard_learning import recent_key_mastered

        self.assertFalse(recent_key_mastered([True] * 19, 20))
        self.assertFalse(recent_key_mastered([True] * 19 + [False], 20))
        self.assertTrue(recent_key_mastered([True] * 20, 20))

    def test_evaluation_schedule_is_randomized_and_balanced(self):
        from collections import Counter
        import numpy as np
        from drosomath.malecns.keyboard_learning import (
            build_evaluation_schedule,
        )

        labels = ("A", "B", "C")
        schedule = build_evaluation_schedule(
            labels,
            20,
            np.random.default_rng(5),
        )
        self.assertEqual(len(schedule), 60)
        self.assertEqual(Counter(schedule), {"A": 20, "B": 20, "C": 20})
        self.assertNotEqual(schedule, tuple(label for label in labels for _ in range(20)))

    def test_balanced_coverage_cycle_contains_each_key_once(self):
        import numpy as np
        from drosomath.malecns.keyboard_learning import build_balanced_coverage_cycle

        labels = ("A", "B", "C", "D")
        cycle = build_balanced_coverage_cycle(labels, np.random.default_rng(8))
        self.assertEqual(len(cycle), len(labels))
        self.assertEqual(set(cycle), set(labels))

    def test_click_teacher_requires_current_eligibility(self):
        from types import SimpleNamespace
        import numpy as np
        from drosomath.malecns.keyboard_learning import KeyboardNeuralSession
        from drosomath.whole_brain import PlasticStateConfig
        from drosomath.whole_brain.plastic_state import SparsePlasticityState

        state = SparsePlasticityState(
            4, config=PlasticStateConfig(plastic_fraction=1.0, seed=2)
        )
        state.usage_ema[:] = 1.0
        session = KeyboardNeuralSession.__new__(KeyboardNeuralSession)
        session.np = np
        session.config = SimpleNamespace(
            click_gate_threshold_hz=9.0,
            click_teacher_learning_rate=0.08,
            click_teacher_credit_floor=0.05,
            teacher_memory_edges_per_key=128,
        )
        session.click_teacher_edges_by_label = {"A": np.asarray([1, 3], dtype=np.int32)}
        session.click_teacher_edge_indices = np.asarray([1, 3], dtype=np.int32)
        session.click_teacher_pre_indices = np.asarray([0, 2], dtype=np.int32)
        session._pending_teacher_normalizer = None
        session.teacher_memory_edges_by_label = {"A": np.empty(0, dtype=np.int32)}
        session.brain = SimpleNamespace(
            plasticity=state,
            connectome=SimpleNamespace(signed_synapse_counts=np.ones(4, dtype=np.float32)),
            params=SimpleNamespace(mv_per_synapse=1.0),
        )

        before = state.multiplier.copy()
        zero = session._apply_low_peak_click_teacher("A", 3.0, 0.0)
        np.testing.assert_array_equal(state.multiplier, before)
        self.assertEqual(zero["active_eligible_edge_count"], 0)
        self.assertEqual(zero["edge_updates"], 0)

        state.eligibility[1] = 1.0
        active = session._apply_low_peak_click_teacher("A", 3.0, 0.0)
        self.assertEqual(active["active_eligible_edge_count"], 1)
        self.assertEqual(active["edge_updates"], 1)
        self.assertGreater(float(state.multiplier[1]), float(before[1]))
        self.assertEqual(float(state.multiplier[3]), float(before[3]))


if __name__ == "__main__":
    unittest.main()
