import importlib.util
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional whole-brain dependency")
class MaleCNSFirstTrainingTests(unittest.TestCase):
    def _tiny_connectome(self):
        import numpy as np

        from drosomath.malecns.loader import MaleCNSConnectome

        # 0 visual-L -> 2 DN-L; 1 visual-R -> 3 DN-R.
        body_ids = np.asarray([100, 101, 200, 201], dtype=np.int64)
        indptr = np.asarray([0, 1, 2, 2, 2], dtype=np.int64)
        posts = np.asarray([2, 3], dtype=np.int32)
        counts = np.asarray([50, 50], dtype=np.int32)
        signed = counts.astype(np.float32)
        return MaleCNSConnectome(
            body_ids=body_ids,
            indptr=indptr,
            post_indices=posts,
            synapse_counts=counts,
            signed_synapse_counts=signed,
            outgoing_strength=np.asarray([50.0, 50.0, 0.0, 0.0], dtype=np.float32),
            presynaptic_sign=np.ones(4, dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine"] * 4, dtype=object),
            metadata={
                "superclass": np.asarray(
                    ["visual_projection", "visual_projection", "descending_neuron", "descending_neuron"],
                    dtype=object,
                ),
                "rootSide": np.asarray(["L", "R", "L", "R"], dtype=object),
            },
            min_connection_synapses=5,
            source="tiny MaleCNS fixture",
        )

    def test_auto_populations_use_real_annotation_roles(self) -> None:
        from drosomath.malecns.first_training import choose_default_populations

        connectome = self._tiny_connectome()
        stimuli, output, info = choose_default_populations(
            connectome,
            input_per_side=1,
            output_population_size=2,
        )
        self.assertEqual(stimuli["LEFT"], (100,))
        self.assertEqual(stimuli["RIGHT"], (101,))
        self.assertEqual(set(output.body_ids), {200, 201})
        self.assertEqual(info["input_superclass"], "visual_projection")
        self.assertEqual(info["output_superclass"], "descending_neuron")

    def test_first_curriculum_runs_and_writes_compact_checkpoint(self) -> None:
        from drosomath.malecns.first_training import FirstTrainingConfig, run_first_training

        connectome = self._tiny_connectome()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "state.npz"
            report = run_first_training(
                connectome,
                config=FirstTrainingConfig(
                    input_per_side=1,
                    output_population_size=2,
                    decoder_epochs=2,
                    brain_trials=4,
                    validation_trials_per_class=1,
                    duration_ms=12.0,
                    decoder_stimulus_rate_hz=1000.0,
                    brain_stimulus_rate_hz=1000.0,
                    brain_input_fraction=1.0,
                    plastic_fraction=1.0,
                    seed=3,
                ),
                checkpoint_path=checkpoint,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(report["experiment"], "malecns_first_output_curriculum_v1")
            self.assertEqual(report["readout"]["frozen"], True)
            self.assertEqual(report["brain_training"]["trials"], 4)
            self.assertIn("after_brain_training_accuracy", report["evaluation"])
            self.assertEqual(report["checkpoint"]["path"], str(checkpoint))


if __name__ == "__main__":
    unittest.main()
