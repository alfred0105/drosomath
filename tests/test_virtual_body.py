import unittest


class FourArmWorldTests(unittest.TestCase):
    def test_world_has_four_two_joint_arms_and_twelve_actions(self):
        from drosomath.malecns.virtual_body import ArmAction, FourArmWorld

        world = FourArmWorld()
        self.assertEqual(world.arm_count, 4)
        self.assertEqual(world.action_size, 12)
        self.assertEqual(len(world.observation()["endpoints"]), 4)
        result = world.step((ArmAction(),) * 4)
        self.assertFalse(result.done)
        self.assertEqual(len(result.endpoints), 4)

    def test_click_rewards_target_and_finishes_episode(self):
        from drosomath.malecns.virtual_body import ArmAction, FourArmWorld, VirtualTarget

        world = FourArmWorld()
        world.reset([VirtualTarget("arm0_target", 0.58, 0.18, radius=0.10, owner_arm=0)])
        world.arms[0].shoulder_angle_rad = 0.0
        world.arms[0].elbow_angle_rad = 0.0
        result = world.step((ArmAction(click=True), ArmAction(), ArmAction(), ArmAction()))

        self.assertEqual(result.clicked_arm, 0)
        self.assertEqual(result.clicked_target, "arm0_target")
        self.assertGreater(result.reward, 0.0)
        self.assertTrue(result.done)

    def test_wrong_arm_cannot_click_owned_target(self):
        from drosomath.malecns.virtual_body import ArmAction, FourArmWorld, VirtualTarget

        world = FourArmWorld()
        world.reset([VirtualTarget("arm0_target", 0.40, 0.0, radius=0.10, owner_arm=0)])
        world.arms[1].shoulder_angle_rad = 0.0
        world.arms[1].elbow_angle_rad = 0.0
        result = world.step((ArmAction(), ArmAction(click=True), ArmAction(), ArmAction()))

        self.assertIsNone(result.clicked_target)
        self.assertLess(result.reward, 0.0)
        self.assertFalse(result.done)

    def test_one_arm_world_has_three_actions(self):
        from drosomath.malecns.virtual_body import OneArmWorld

        world = OneArmWorld()
        self.assertEqual(world.arm_count, 1)
        self.assertEqual(world.action_size, 3)
        self.assertEqual(len(world.observation()["arms"]), 1)


if __name__ == "__main__":
    unittest.main()
