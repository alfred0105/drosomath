import importlib.util
import unittest

from drosomath.flywire_real import FlyBrainParams, FlyWireConnectome, SparseFlyBrain


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class FlyWireRealBrainTests(unittest.TestCase):
    def test_published_shiu_parameters_are_default(self) -> None:
        params = FlyBrainParams()
        self.assertEqual(params.resting_mv, -52.0)
        self.assertEqual(params.threshold_mv, -45.0)
        self.assertEqual(params.membrane_tau_ms, 20.0)
        self.assertEqual(params.synapse_tau_ms, 5.0)
        self.assertEqual(params.refractory_ms, 2.2)
        self.assertEqual(params.delay_ms, 1.8)
        self.assertEqual(params.mv_per_synapse, 0.275)

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional FlyWire dependency")
    def test_sparse_engine_keeps_real_ids_and_propagates(self) -> None:
        import numpy as np

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([101001, 101002, 101003], dtype=np.int64),
            indptr=np.asarray([0, 1, 2, 2], dtype=np.int64),
            post_indices=np.asarray([1, 2], dtype=np.int32),
            signed_synapse_counts=np.asarray([50.0, 50.0], dtype=np.float32),
            outgoing_strength=np.asarray([50.0, 50.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(connectome.index_of(101002), 1)
        self.assertEqual(connectome.strongest_outgoing_ids(1), (101002,))

        brain = SparseFlyBrain(connectome, params=FlyBrainParams(dt_ms=0.2), seed=0)
        result = brain.run(
            duration_ms=25.0,
            stimulus_ids=[101001],
            stimulus_rate_hz=1000.0,
        )
        self.assertEqual(result["neuron_count"], 3)
        self.assertEqual(result["stimulus_flywire_ids"], [101001])
        self.assertGreater(result["total_spikes"], 0)
        self.assertGreater(result["total_synaptic_edge_deliveries"], 0)


if __name__ == "__main__":
    unittest.main()
