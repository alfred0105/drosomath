import importlib.util
import unittest


GPU_AVAILABLE = importlib.util.find_spec("cupy") is not None


@unittest.skipUnless(GPU_AVAILABLE, "CuPy is an optional GPU dependency")
class MaleCNSGpuTests(unittest.TestCase):
    def test_gpu_csr_runner_propagates_on_tiny_graph(self):
        import numpy as np
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            self.skipTest("CUDA device is unavailable")

        from drosomath.flywire_real import FlyBrainParams
        from drosomath.malecns.gpu import GpuSparseFlyBrain
        from drosomath.malecns.loader import MaleCNSConnectome

        graph = MaleCNSConnectome(
            body_ids=np.asarray([10, 20, 30], dtype=np.int64),
            indptr=np.asarray([0, 1, 2, 2], dtype=np.int64),
            post_indices=np.asarray([1, 2], dtype=np.int32),
            synapse_counts=np.asarray([50, 50], dtype=np.int32),
            signed_synapse_counts=np.asarray([50.0, 50.0], dtype=np.float32),
            outgoing_strength=np.asarray([50.0, 50.0, 0.0], dtype=np.float32),
            presynaptic_sign=np.ones(3, dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine"] * 3, dtype=object),
            metadata={"superclass": np.asarray(["visual_projection"] * 3, dtype=object)},
            min_connection_synapses=1,
        )
        brain = GpuSparseFlyBrain(
            graph,
            params=FlyBrainParams(dt_ms=0.2),
            seed=0,
        )
        result = brain.run(
            duration_ms=25.0,
            stimulus_body_ids=[10],
            stimulus_rate_hz=1000.0,
        )

        self.assertTrue(result["gpu_used"])
        self.assertFalse(result["plasticity_used"])
        self.assertGreater(result["total_spikes"], 0)
        self.assertGreater(result["active_neurons"], 0)


if __name__ == "__main__":
    unittest.main()
