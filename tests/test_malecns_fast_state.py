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


if __name__ == "__main__":
    unittest.main()
