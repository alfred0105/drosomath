import inspect
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


if __name__ == "__main__":
    unittest.main()
