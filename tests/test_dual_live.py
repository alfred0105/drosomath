import unittest

from drosomath.core import AdaptiveBridge, BridgeSynapseState, PlasticityTracker, SpikingNetwork
from drosomath.dual_live import DualLiveSimulation


class DualLiveTests(unittest.TestCase):
    def test_dual_snapshot_contains_brains_bridge_and_modules(self) -> None:
        simulation = DualLiveSimulation(hz=12.0)
        snapshot = simulation.snapshot()
        for _ in range(18):
            snapshot = simulation.advance_once()

        self.assertEqual(snapshot["mode"], "dual")
        self.assertEqual([brain["name"] for brain in snapshot["brains"]], ["A", "B"])
        self.assertEqual(snapshot["bridge"]["synapse_count"], 4)
        self.assertGreaterEqual(snapshot["completed_trials"], 3)
        self.assertIn("module_specialization", snapshot["metrics"])
        self.assertEqual(len(snapshot["brains"][0]["modules"]), 2)
        self.assertEqual(len(snapshot["brains"][1]["modules"]), 2)

    def test_controls_preserve_dual_mode(self) -> None:
        simulation = DualLiveSimulation(hz=8.0)
        paused = simulation.control("pause")
        self.assertFalse(paused["running"])

        before = paused["step"]
        stepped = simulation.control("step")
        self.assertEqual(stepped["step"], before + 1)
        self.assertFalse(stepped["running"])

        reset = simulation.control("reset")
        self.assertEqual(reset["mode"], "dual")
        self.assertEqual(reset["completed_trials"], 0)

    def test_bridge_clear_recent_prevents_cross_trial_reward_credit(self) -> None:
        tracker_a = PlasticityTracker([])
        tracker_b = PlasticityTracker([])
        network_a = SpikingNetwork.from_ids([0], tracker_a, threshold=0.5, decay=1.0)
        network_b = SpikingNetwork.from_ids([0], tracker_b, threshold=0.5, decay=1.0)
        synapse = BridgeSynapseState("A", 0, "B", 0, 0.4)
        bridge = AdaptiveBridge([synapse], learning_rate=0.1, reward_alpha=0.5)

        bridge.transfer(
            source_brain="A",
            fired=[0],
            target_networks={"A": network_a, "B": network_b},
            step=0,
        )
        bridge.clear_recent()
        credited = bridge.apply_reward(reward=1.0, step=1)

        self.assertEqual(credited, 0)
        self.assertAlmostEqual(synapse.weight, 0.4)
        self.assertAlmostEqual(synapse.reward_ema, 0.0)


if __name__ == "__main__":
    unittest.main()
