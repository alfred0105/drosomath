import importlib.util
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
ARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(
    NUMPY_AVAILABLE and ARROW_AVAILABLE,
    "MaleCNS tests need NumPy and PyArrow",
)
class MaleCNSLoaderTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        import pyarrow as pa
        import pyarrow.feather as feather

        from drosomath.malecns.download import FILES

        feather.write_feather(
            pa.table(
                {
                    "bodyId": pa.array([10, 20, 30, 999], type=pa.int64()),
                    "superclass": pa.array(
                        ["sensory", "descending_neuron", "vnc_intrinsic", None]
                    ),
                    "type": pa.array(["A", "B", "C", None]),
                    "rootSide": pa.array(["L", "R", "L", None]),
                }
            ),
            root / FILES["annotations"],
        )
        feather.write_feather(
            pa.table(
                {
                    "body": pa.array([10, 20, 30, 999], type=pa.int64()),
                    "consensus_nt": pa.array(
                        ["acetylcholine", "gaba", "glutamate", "acetylcholine"]
                    ),
                }
            ),
            root / FILES["neurotransmitters"],
        )
        feather.write_feather(
            pa.table(
                {
                    "body_pre": pa.array([10, 20, 30, 999], type=pa.int64()),
                    "body_post": pa.array([20, 30, 10, 10], type=pa.int64()),
                    "weight": pa.array([8, 4, 6, 100], type=pa.int32()),
                }
            ),
            root / FILES["weights"],
        )

    def test_loader_filters_non_neurons_thresholds_and_preserves_sign(self) -> None:
        import numpy as np

        from drosomath.malecns.loader import load_malecns_v1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            graph = load_malecns_v1(
                root,
                min_connection_synapses=5,
                require_published_neuron_count=False,
                batch_size=2,
            )

        np.testing.assert_array_equal(graph.body_ids, np.asarray([10, 20, 30]))
        self.assertEqual(graph.neuron_count, 3)
        self.assertEqual(graph.edge_count, 2)
        np.testing.assert_array_equal(graph.post_indices, np.asarray([1, 0]))
        np.testing.assert_array_equal(graph.synapse_counts, np.asarray([8, 6]))
        np.testing.assert_allclose(graph.signed_synapse_counts, np.asarray([8.0, -6.0]))
        np.testing.assert_array_equal(graph.presynaptic_sign, np.asarray([1, -1, -1]))
        self.assertEqual(graph.index_of(20), 1)
        self.assertEqual(graph.select_indices(superclass="descending_neuron").tolist(), [1])

    def test_malecns_connectome_drives_existing_plastic_engine(self) -> None:
        import numpy as np

        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.malecns.loader import load_malecns_v1
        from drosomath.whole_brain import PlasticStateConfig, UsageRewardRule

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            graph = load_malecns_v1(
                root,
                min_connection_synapses=5,
                require_published_neuron_count=False,
                batch_size=2,
            )

        brain = PlasticMaleCNSBrain(
            graph,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0),
            usage_alpha=1.0,
        )
        # Directly exercise the same sparse event path used when neuron 10 spikes.
        brain._schedule_spike_outputs(np.asarray([graph.index_of(10)], dtype=np.int32))
        before = float(brain.plasticity.multiplier[0])
        report = brain.learn_from_reward(
            reward=1.0,
            rule=UsageRewardRule(learning_rate=0.1),
        )
        after = float(brain.plasticity.multiplier[0])

        self.assertEqual(report["learning"]["edge_updates"], 1)
        self.assertGreater(after, before)
        # Anatomy itself is immutable: only the learned multiplier changed.
        self.assertEqual(float(graph.signed_synapse_counts[0]), 8.0)

    def test_report_uses_malecns_body_id_labels(self) -> None:
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.malecns.loader import load_malecns_v1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            graph = load_malecns_v1(
                root,
                min_connection_synapses=5,
                require_published_neuron_count=False,
                batch_size=2,
            )

        brain = PlasticMaleCNSBrain(graph, seed=0)
        result = brain.run_malecns(
            duration_ms=1.0,
            stimulus_body_ids=[10],
            stimulus_rate_hz=1000.0,
        )
        self.assertEqual(result["stimulus_body_ids"], [10])
        self.assertNotIn("stimulus_flywire_ids", result)
        self.assertEqual(result["dataset"], "MaleCNS v1.0 / HHMI Janelia")


if __name__ == "__main__":
    unittest.main()
