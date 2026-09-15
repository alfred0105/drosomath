from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required for MaleCNS live curriculum")
class MaleCNSLiveCurriculumTests(unittest.TestCase):
    def test_runner_constructs_without_loading_dataset(self) -> None:
        from drosomath.malecns.curriculum_v1 import CurriculumV1Config
        from drosomath.malecns.live_curriculum import LiveCurriculumRunner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = LiveCurriculumRunner(
                data_dir=root / "data",
                config=CurriculumV1Config(stage_trials=2, checkpoint_every=1),
                result_path=root / "result.json",
                html_path=root / "result.html",
                checkpoint_path=root / "brain.npz",
                readout_dir=root / "readouts",
                download=False,
                resume=True,
            )
            snap = runner.state.snapshot()
            self.assertEqual(snap["phase"], "loading")
            self.assertFalse(snap["finished"])

    def test_plastic_fraction_property_tracks_unlock(self) -> None:
        from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState

        state = SparsePlasticityState(1000, config=PlasticStateConfig(plastic_fraction=0.05, seed=3))
        self.assertAlmostEqual(state.plastic_fraction, 0.05)
        before = state.plastic_edge_count
        state.set_plastic_fraction(0.20)
        self.assertAlmostEqual(state.plastic_fraction, 0.20)
        self.assertGreaterEqual(state.plastic_edge_count, before)


if __name__ == "__main__":
    unittest.main()
