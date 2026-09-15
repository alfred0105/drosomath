import unittest

from drosomath.core import PlasticityTracker, SpikingNetwork, SynapseState


class SpikingNetworkTests(unittest.TestCase):
    def test_spike_propagates_on_next_step(self) -> None:
        synapse = SynapseState(0, 1, 1.0)
        tracker = PlasticityTracker([synapse])
        network = SpikingNetwork.from_ids([0, 1], tracker, threshold=0.5, decay=1.0)

        first = network.step({0: 1.0})
        second = network.step()

        self.assertEqual(first.fired, (0,))
        self.assertEqual(first.transferred_synapses, 1)
        self.assertEqual(second.fired, (1,))
        self.assertEqual(synapse.usage_count, 1)

    def test_network_refreshes_after_rewire_registry_change(self) -> None:
        tracker = PlasticityTracker([SynapseState(0, 1, 1.0)])
        network = SpikingNetwork.from_ids([0, 1, 2], tracker, threshold=0.5, decay=1.0)

        removed = tracker.unregister_synapse(pre_id=0, post_id=1)
        self.assertIsNotNone(removed)
        tracker.register_synapse(SynapseState(0, 2, 1.0))

        network.step({0: 1.0})
        second = network.step()

        self.assertEqual(second.fired, (2,))

    def test_unknown_synapse_neuron_is_rejected(self) -> None:
        tracker = PlasticityTracker([SynapseState(0, 99, 1.0)])
        with self.assertRaises(ValueError):
            SpikingNetwork.from_ids([0, 1], tracker)


if __name__ == "__main__":
    unittest.main()
