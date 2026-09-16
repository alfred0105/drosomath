import unittest

from drosomath.core import (
    PlasticityTracker,
    SpikingNetwork,
    STDPPlasticity,
    STDPRule,
    SynapseState,
)


class STDPTests(unittest.TestCase):
    def test_pre_before_post_potentiates(self) -> None:
        synapse = SynapseState(0, 1, 0.5)
        stdp = STDPPlasticity(
            rule=STDPRule(
                potentiation_rate=0.1,
                depression_rate=0.1,
                tau_plus=10.0,
                tau_minus=10.0,
                window=20,
            )
        )

        stdp.observe_spikes([0], [synapse], step=1)
        updates = stdp.observe_spikes([1], [synapse], step=2)

        self.assertEqual(updates, 1)
        self.assertGreater(synapse.weight, 0.5)

    def test_post_before_pre_depresses(self) -> None:
        synapse = SynapseState(0, 1, 0.5)
        stdp = STDPPlasticity(
            rule=STDPRule(
                potentiation_rate=0.1,
                depression_rate=0.1,
                tau_plus=10.0,
                tau_minus=10.0,
                window=20,
            )
        )

        stdp.observe_spikes([1], [synapse], step=1)
        updates = stdp.observe_spikes([0], [synapse], step=2)

        self.assertEqual(updates, 1)
        self.assertLess(synapse.weight, 0.5)

    def test_outside_window_does_not_change_weight(self) -> None:
        synapse = SynapseState(0, 1, 0.5)
        stdp = STDPPlasticity(rule=STDPRule(window=2))

        stdp.observe_spikes([0], [synapse], step=1)
        updates = stdp.observe_spikes([1], [synapse], step=5)

        self.assertEqual(updates, 0)
        self.assertAlmostEqual(synapse.weight, 0.5)

    def test_weight_bounds_are_respected(self) -> None:
        upper = SynapseState(0, 1, 0.99)
        lower = SynapseState(2, 3, 0.01)
        rule = STDPRule(
            potentiation_rate=1.0,
            depression_rate=1.0,
            tau_plus=100.0,
            tau_minus=100.0,
            window=10,
            min_weight=0.0,
            max_weight=1.0,
        )
        stdp = STDPPlasticity(rule=rule)

        stdp.observe_spikes([0, 3], [upper, lower], step=1)
        stdp.observe_spikes([1, 2], [upper, lower], step=2)

        self.assertLessEqual(upper.weight, 1.0)
        self.assertGreaterEqual(lower.weight, 0.0)

    def test_spiking_network_can_apply_stdp(self) -> None:
        synapse = SynapseState(0, 1, 0.6)
        tracker = PlasticityTracker([synapse])
        stdp = STDPPlasticity(
            rule=STDPRule(
                potentiation_rate=0.1,
                depression_rate=0.1,
                tau_plus=10.0,
                tau_minus=10.0,
                window=20,
            )
        )
        network = SpikingNetwork.from_ids(
            [0, 1],
            tracker,
            threshold=0.5,
            decay=1.0,
            stdp=stdp,
        )

        first = network.step({0: 1.0})
        second = network.step()

        self.assertEqual(first.fired, (0,))
        self.assertEqual(second.fired, (1,))
        self.assertEqual(second.stdp_updates, 1)
        self.assertGreater(synapse.weight, 0.6)


if __name__ == "__main__":
    unittest.main()
