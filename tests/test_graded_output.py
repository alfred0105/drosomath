import unittest

from drosomath.malecns.graded_output import (
    BETWEEN_LEVELS,
    NO_OUTPUT,
    GradedPopulationCode,
    GradedRateCodeConfig,
    calibrate_two_level_rate_code,
)


class GradedOutputTests(unittest.TestCase):
    def test_fixed_graded_population_levels_and_silence(self):
        code = GradedPopulationCode(
            100,
            GradedRateCodeConfig(
                base_rate_hz=1.0,
                level_step_hz=1.5,
                tolerance_hz=0.5,
                max_level=7,
            ),
        )

        level0 = code.observe(100, duration_ms=1000.0, target_level=0)
        self.assertEqual(level0.predicted_level, 0)
        self.assertEqual(level0.status, "LEVEL_0")
        self.assertTrue(level0.correct)
        self.assertEqual(level0.teaching_signal, 0.0)

        level1 = code.observe(250, duration_ms=1000.0, target_level=1)
        self.assertEqual(level1.predicted_level, 1)
        self.assertEqual(level1.status, "LEVEL_1")
        self.assertTrue(level1.correct)
        self.assertEqual(level1.teaching_signal, 0.0)

        silent = code.observe(0, duration_ms=1000.0, target_level=0)
        self.assertIsNone(silent.predicted_level)
        self.assertEqual(silent.status, NO_OUTPUT)

        between = code.observe(180, duration_ms=1000.0, target_level=1)
        self.assertIsNone(between.predicted_level)
        self.assertEqual(between.status, BETWEEN_LEVELS)

    def test_teaching_signal_has_direction_and_is_bounded(self):
        code = GradedPopulationCode(
            100,
            GradedRateCodeConfig(
                base_rate_hz=1.0,
                level_step_hz=1.5,
                tolerance_hz=0.5,
                max_level=7,
                max_teaching_signal=1.0,
            ),
        )

        too_low = code.observe(100, duration_ms=1000.0, target_level=1)
        self.assertGreater(too_low.signed_error_hz, 0.0)
        self.assertEqual(too_low.teaching_signal, 1.0)

        too_high = code.observe(400, duration_ms=1000.0, target_level=1)
        self.assertLess(too_high.signed_error_hz, 0.0)
        self.assertEqual(too_high.teaching_signal, -1.0)

        very_low = code.observe(0, duration_ms=1000.0, target_level=7)
        self.assertEqual(very_low.teaching_signal, 1.0)

    def test_output_code_is_explicitly_non_trainable(self):
        code = GradedPopulationCode(512)
        summary = code.summary()
        self.assertEqual(summary["type"], "fixed_population_rate_code")
        self.assertFalse(summary["trainable_decoder"])
        self.assertEqual(summary["population_size"], 512)

    def test_rate_code_is_calibrated_from_measured_distributions(self):
        calibration = calibrate_two_level_rate_code(
            [1.0, 1.1, 0.9],
            [2.9, 3.0, 3.1],
            min_separation_hz=0.5,
        )
        self.assertTrue(calibration.usable)
        config = calibration.code_config(max_level=1)
        self.assertAlmostEqual(config.base_rate_hz, 1.0)
        self.assertAlmostEqual(config.target_rate_hz(1), 3.0)
        self.assertLess(config.tolerance_hz, abs(config.level_step_hz) / 2.0)

    def test_rate_calibration_rejects_unseparated_classes(self):
        calibration = calibrate_two_level_rate_code(
            [1.0, 1.0],
            [1.02, 0.98],
            min_separation_hz=0.10,
        )
        self.assertFalse(calibration.usable)


if __name__ == "__main__":
    unittest.main()
