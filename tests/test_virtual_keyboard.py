import unittest


class VirtualKeyboardTests(unittest.TestCase):
    def test_required_key_set_and_one_hot_prompt(self):
        from drosomath.malecns.virtual_keyboard import KEY_LABELS, VirtualKeyboard

        keyboard = VirtualKeyboard()
        self.assertEqual(set(KEY_LABELS), set(keyboard.labels()))
        prompt = keyboard.prompt_vector("한글")
        self.assertEqual(len(prompt), 14)
        self.assertEqual(sum(prompt), 1)
        self.assertEqual(prompt[0], 1)

    def test_matching_task_exposes_prompt_and_keyboard(self):
        from drosomath.malecns.virtual_keyboard import KeyboardMatchingTask

        task = KeyboardMatchingTask(seed=1)
        observation = task.reset("X")
        self.assertEqual(observation["prompt"], "X")
        self.assertEqual(len(observation["keyboard"]), 14)
        self.assertEqual(observation["body"]["done"], False)

    def test_wrong_click_is_penalized_and_target_remains(self):
        from drosomath.malecns.virtual_body import ArmAction
        from drosomath.malecns.virtual_keyboard import KeyboardMatchingTask

        task = KeyboardMatchingTask(seed=1)
        task.reset("O")
        result = task.step((ArmAction(click=True), ArmAction(), ArmAction(), ArmAction()))
        self.assertLess(result.reward, 0.0)
        self.assertFalse(result.done)
        self.assertIsNone(result.clicked_label)


if __name__ == "__main__":
    unittest.main()
