import unittest


class VirtualKeyboardTests(unittest.TestCase):
    def test_required_key_set_and_one_hot_prompt(self):
        from drosomath.malecns.virtual_keyboard import KEY_LABELS, VirtualKeyboard

        keyboard = VirtualKeyboard()
        self.assertEqual(set(KEY_LABELS), set(keyboard.labels()))
        prompt = keyboard.prompt_vector("ㄱ")
        self.assertEqual(len(prompt), 60)
        self.assertEqual(sum(prompt), 1)
        self.assertEqual(prompt[0], 1)

    def test_label_schedule_is_shuffled_but_balanced(self):
        from collections import Counter
        from drosomath.malecns.keyboard_learning import build_label_schedule
        from drosomath.malecns.virtual_keyboard import KEY_LABELS

        schedule = build_label_schedule(len(KEY_LABELS) * 2, seed=7)
        self.assertNotEqual(schedule[: len(KEY_LABELS)], KEY_LABELS)
        counts = Counter(schedule)
        self.assertEqual(set(counts.values()), {2})

    def test_matching_task_exposes_prompt_and_keyboard(self):
        from drosomath.malecns.virtual_keyboard import KeyboardMatchingTask

        task = KeyboardMatchingTask(seed=1)
        observation = task.reset("X")
        self.assertEqual(observation["prompt"], "X")
        self.assertEqual(len(observation["keyboard"]), 60)
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
