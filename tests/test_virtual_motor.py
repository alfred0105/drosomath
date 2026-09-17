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


if __name__ == "__main__":
    unittest.main()
