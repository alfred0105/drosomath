import unittest


class FourArmMotorTests(unittest.TestCase):
    def test_decoder_maps_antagonistic_rates_to_signed_actions(self):
        from drosomath.malecns.virtual_motor import FourArmMotorAdapter

        adapter = FourArmMotorAdapter()
        rates = [5.0] * 20
        rates[0] = 10.0
        rates[1] = 5.0
        rates[2] = 5.0
        rates[3] = 10.0
        actions = adapter.decode(rates)

        self.assertGreater(actions[0].shoulder_velocity, 0.0)
        self.assertLess(actions[0].elbow_velocity, 0.0)
        self.assertFalse(actions[0].click)

    def test_click_channel_has_refractory_period(self):
        from drosomath.malecns.virtual_motor import FourArmMotorAdapter

        adapter = FourArmMotorAdapter()
        rates = [5.0] * 20
        rates[4] = 20.0
        first = adapter.decode(rates)
        second = adapter.decode(rates)
        third = adapter.decode(rates)
        fourth = adapter.decode(rates)

        self.assertTrue(first[0].click)
        self.assertFalse(second[0].click)
        self.assertFalse(third[0].click)
        self.assertFalse(fourth[0].click)

    def test_one_arm_adapter_has_five_channels(self):
        from drosomath.malecns.virtual_motor import OneArmMotorAdapter

        adapter = OneArmMotorAdapter()
        self.assertEqual(adapter.channel_count, 5)
        actions = adapter.decode([5.0, 5.0, 5.0, 5.0, 20.0])
        self.assertEqual(len(actions), 1)
        self.assertTrue(actions[0].click)

    def test_click_evidence_integrates_multiple_control_windows(self):
        from drosomath.malecns.virtual_motor import FourArmMotorConfig, OneArmMotorAdapter

        adapter = OneArmMotorAdapter(FourArmMotorConfig(
            click_threshold_hz=9.0,
            click_integration_windows=5,
            click_refractory_steps=0,
        ))
        quiet = [5.0, 5.0, 5.0, 5.0, 0.0]
        active = [5.0, 5.0, 5.0, 5.0, 20.0]
        self.assertFalse(adapter.decode(active)[0].click)
        self.assertFalse(adapter.decode(quiet)[0].click)
        self.assertFalse(adapter.decode(active)[0].click)
        self.assertTrue(adapter.decode(active)[0].click)
        self.assertGreaterEqual(adapter.last_click_evidence_rates[0], 9.0)


if __name__ == "__main__":
    unittest.main()
