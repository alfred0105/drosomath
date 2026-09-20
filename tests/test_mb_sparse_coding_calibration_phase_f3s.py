import inspect
import json
from pathlib import Path
import unittest

from run_mb_sparse_coding_calibration_phase_f3s import (
    OFFSETS,
    RATES,
    _selection,
)


class MBSparseCodingCalibrationF3STests(unittest.TestCase):
    def test_declared_calibration_grid_is_fixed(self):
        self.assertEqual(RATES, (25.0, 50.0, 100.0, 150.0, 205.0))
        self.assertEqual(OFFSETS, (0.0, 2.0, 4.0, 6.0))

    def test_selection_uses_only_single_symbol_calibration_observables(self):
        point = {
            "rate_hz": 100.0,
            "kc_offset_mv": 0.0,
            "mean_kc_active_fraction": 0.075,
            "max_symbol_kc_active_fraction": 0.10,
            "pairwise_kc_active_set_jaccard_mean": 0.3,
            "mean_whole_network_active_fraction": 0.1,
            "physiological_acceptable": True,
        }
        selected = _selection([point])
        self.assertEqual(selected["rate_hz"], 100.0)

    def test_no_point_is_selected_when_physiology_guard_fails(self):
        self.assertIsNone(_selection([{"physiological_acceptable": False}]))

    def test_runner_has_no_task_grammar_or_training_selection_path(self):
        import run_mb_sparse_coding_calibration_phase_f3s as module

        source = inspect.getsource(module)
        self.assertNotIn("CONTEXTUAL_GRAMMAR", source)
        self.assertNotIn("learn=True", source)
        self.assertIn("single_symbol_only_during_calibration", source)
        self.assertIn("stage_b_only_after_selection", source)

    def test_stage_b_uses_declared_seed_set(self):
        import run_mb_sparse_coding_calibration_phase_f3s as module

        self.assertEqual(module.SEEDS, (251, 257, 263))
        self.assertEqual(len(module.ORDERED_CONTEXTS), 16)

    def test_corrected_stage_b_propagates_item_and_go_rates(self):
        import run_mb_sparse_coding_calibration_phase_f3s as module

        source = inspect.getsource(module._representation_audit)
        first_memory_source = inspect.getsource(module._first_memory)
        self.assertIn("item_stimulus_rate_hz=rate", source)
        self.assertIn("go_stimulus_rate_hz=go_rate", source)
        self.assertIn("item_stimulus_rate_hz=rate", first_memory_source)
        self.assertIn("go_stimulus_rate_hz=go_rate", first_memory_source)
        self.assertNotIn("calibration_grid(", inspect.getsource(module.run))

    def test_corrected_artifact_records_real_rate_trace_difference(self):
        artifact = Path(__file__).resolve().parents[1] / "results/latest_mb_sparse_coding_calibration_phase_f3s.json"
        if not artifact.exists():
            self.skipTest("generated F.3S artifact is unavailable")
        data = json.loads(artifact.read_text(encoding="utf-8"))
        if "correction" not in data:
            self.skipTest("corrected F.3S artifact has not been generated yet")
        self.assertTrue(data["correction"]["calibration_reused"])
        self.assertFalse(data["correction"]["calibration_rerun"])
        sanity = data["rate_trace_sanity"]
        self.assertTrue(sanity["actual_trace_differs"])
        self.assertEqual(sanity["effective_first_rate_hz"]["sparse"], 25.0)
        self.assertEqual(sanity["effective_second_rate_hz"]["sparse"], 25.0)
        self.assertEqual(sanity["effective_go_rate_hz"]["sparse"], 205.0)
        for arm, item_rate in (("CURRENT_RANDOM_ENCODER", 205.0), ("MB_ROUTED_GENERIC_205HZ", 205.0), ("MB_ROUTED_SPARSE_OPERATING_POINT", 25.0)):
            observed = data["representation_comparison"]["per_seed"][0][arm]
            self.assertEqual(observed["requested_item_rate_hz"], item_rate)
            self.assertEqual(observed["effective_first_rate_hz"], item_rate)
            self.assertEqual(observed["effective_second_rate_hz"], item_rate)
            self.assertEqual(observed["effective_go_rate_hz"], 205.0)


if __name__ == "__main__":
    unittest.main()
