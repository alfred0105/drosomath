import unittest

from drosomath.malecns.keyboard_diagnostics import jaccard
from drosomath.malecns.keyboard_learning import build_evaluation_schedule


class KeyboardDiagnosticTests(unittest.TestCase):
    def test_jaccard_handles_overlap_and_empty_sets(self):
        self.assertEqual(jaccard([1, 2], [2, 3]), 1 / 3)
        self.assertEqual(jaccard([], []), 1.0)
        self.assertEqual(jaccard([], [1]), 0.0)

    def test_retention_schedule_has_equal_attempts_per_key(self):
        import numpy as np
        from drosomath.malecns.virtual_keyboard import KEY_LABELS

        schedule = build_evaluation_schedule(KEY_LABELS, 5, np.random.default_rng(4))
        self.assertEqual(len(schedule), len(KEY_LABELS) * 5)
        self.assertTrue(all(schedule.count(label) == 5 for label in KEY_LABELS))


if __name__ == "__main__":
    unittest.main()
