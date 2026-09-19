import unittest

from drosomath.learning_signal import LearningSignal


class LearningSignalTests(unittest.TestCase):
    def test_signal_is_task_independent(self):
        signal = LearningSignal(reward=1.0, directional_error={"choice/left": 0.4})
        self.assertEqual(signal.nonzero_directions(), {"choice/left": 0.4})

    def test_zero_direction_is_ignored(self):
        signal = LearningSignal(reward=0.0, directional_error={"motor/click": 0.0})
        self.assertEqual(signal.nonzero_directions(), {})


if __name__ == "__main__":
    unittest.main()
