import unittest

from drosomath.malecns.presence_graded import PresenceMasteryConfig, mastery_passed


class PresenceGradedTests(unittest.TestCase):
    def test_mastery_gate_uses_requested_threshold(self):
        self.assertTrue(mastery_passed(0.85, 0.85))
        self.assertTrue(mastery_passed(0.91, 0.85))
        self.assertFalse(mastery_passed(0.849, 0.85))

    def test_presence_mastery_defaults_allow_two_retraining_cycles(self):
        config = PresenceMasteryConfig()
        self.assertEqual(config.max_attempts_per_phase, 3)
        self.assertEqual(config.fixed_full_trials, 2048)
        self.assertEqual(config.varied_full_trials, 3072)
        self.assertEqual(config.varied_dropout_trials, 3072)
        self.assertEqual(config.mastery_accuracy, 0.85)

    def test_presence_mastery_rejects_invalid_attempt_count(self):
        with self.assertRaises(ValueError):
            PresenceMasteryConfig(max_attempts_per_phase=0)


if __name__ == "__main__":
    unittest.main()
