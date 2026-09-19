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

    def test_circular_keyboard_works_with_one_arm(self):
        from drosomath.malecns.virtual_body import OneArmWorld
        from drosomath.malecns.virtual_keyboard import FanKeyboard, KeyboardMatchingTask
        from drosomath.malecns.virtual_motor import OneArmMotorAdapter

        world = OneArmWorld()
        task = KeyboardMatchingTask(
            keyboard=FanKeyboard.for_arm(world.arm_configs[0]),
            world=world,
            motor=OneArmMotorAdapter(),
        )
        observation = task.reset("O")
        self.assertEqual(len(observation["keyboard"]), 60)
        self.assertEqual(len(observation["body"]["arms"]), 1)
        self.assertEqual(task.motor.channel_count, 5)
        radii = sorted(
            ((key["x"] - 0.5) ** 2 + (key["y"] - 0.5) ** 2) ** 0.5
            for key in observation["keyboard"]
        )
        radial_layers = []
        for radius in radii:
            if not radial_layers or abs(radius - radial_layers[-1]) > 1e-8:
                radial_layers.append(radius)
        self.assertEqual(len(radial_layers), 5)

    def test_forward_fan_keys_are_all_reachable_without_box_overlap(self):
        import math

        from drosomath.malecns.virtual_body import OneArmWorld
        from drosomath.malecns.virtual_keyboard import FanKeyboard

        world = OneArmWorld()
        keyboard = FanKeyboard.for_arm(world.arm_configs[0])
        keys = list(keyboard.keys.values())
        for key in keys:
            world.reset()
            world.position_arm_to(0, key.x, key.y)
            endpoint = world.observation()["endpoints"][0]
            self.assertLess(math.dist(endpoint, (key.x, key.y)), 1e-8)
        closest_centers = min(
            math.dist((first.x, first.y), (second.x, second.y))
            for index, first in enumerate(keys)
            for second in keys[index + 1 :]
        )
        self.assertGreater(closest_centers, keys[0].width)


if __name__ == "__main__":
    unittest.main()
