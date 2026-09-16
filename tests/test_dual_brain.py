import unittest

from drosomath.core import (
    AdaptiveBridge,
    BrainCore,
    BridgeSynapseState,
    DualBrainSystem,
    PlasticityTracker,
    SpikingNetwork,
)


class DualBrainTests(unittest.TestCase):
    def _brain(self, name: str) -> BrainCore:
        tracker = PlasticityTracker([])
        network = SpikingNetwork.from_ids([0, 1], tracker, threshold=0.5, decay=1.0)
        return BrainCore(name=name, network=network, tracker=tracker)

    def test_bridge_spike_arrives_on_next_system_step(self) -> None:
        brain_a = self._brain("A")
        brain_b = self._brain("B")
        bridge = AdaptiveBridge([BridgeSynapseState("A", 0, "B", 1, 1.0)])
        system = DualBrainSystem(brain_a, brain_b, bridge)

        first = system.step(currents_a={0: 1.0})
        second = system.step()

        self.assertEqual(first.brain_a.fired, (0,))
        self.assertEqual(first.bridge_transfers, 1)
        self.assertEqual(second.brain_b.fired, (1,))

    def test_reward_strengthens_used_bridge_only(self) -> None:
        brain_a = self._brain("A")
        brain_b = self._brain("B")
        used = BridgeSynapseState("A", 0, "B", 1, 0.4)
        unused = BridgeSynapseState("B", 0, "A", 1, 0.4)
        bridge = AdaptiveBridge([used, unused], learning_rate=0.1, reward_alpha=0.5)
        system = DualBrainSystem(brain_a, brain_b, bridge)

        system.step(currents_a={0: 1.0})
        credited = system.apply_reward(1.0)

        self.assertEqual(credited[2], 1)
        self.assertAlmostEqual(used.weight, 0.5)
        self.assertAlmostEqual(used.reward_ema, 0.5)
        self.assertAlmostEqual(unused.weight, 0.4)


if __name__ == "__main__":
    unittest.main()
