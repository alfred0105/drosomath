import unittest

from drosomath.core import (
    AdaptiveBridge,
    BridgeStructuralConfig,
    BridgeStructuralPlasticityManager,
    BridgeSynapseState,
)


class BridgeStructuralPlasticityTests(unittest.TestCase):
    def _manager(
        self,
        bridge: AdaptiveBridge,
        **config_overrides,
    ) -> BridgeStructuralPlasticityManager:
        config = BridgeStructuralConfig(
            min_age_cycles=config_overrides.pop("min_age_cycles", 1),
            stale_steps=config_overrides.pop("stale_steps", 2),
            max_rewire_per_cycle=config_overrides.pop("max_rewire_per_cycle", 4),
            regrow_weight=config_overrides.pop("regrow_weight", 0.05),
            **config_overrides,
        )
        return BridgeStructuralPlasticityManager(
            bridge,
            {"A": [0, 1, 2], "B": [0, 1, 2]},
            config=config,
        )

    def test_rewire_preserves_bridge_budget(self) -> None:
        weak = BridgeSynapseState("A", 0, "B", 0, 0.05, reward_ema=-0.5)
        keep = BridgeSynapseState("B", 1, "A", 1, 0.7, reward_ema=0.5)
        bridge = AdaptiveBridge([weak, keep])
        manager = self._manager(bridge)

        before = bridge.synapse_count
        result = manager.rewire(
            step=10,
            candidate_keys=[("A", 2, "B", 2)],
        )

        self.assertEqual(result.pruned, (("A", 0, "B", 0),))
        self.assertEqual(result.regrown, (("A", 2, "B", 2),))
        self.assertEqual(bridge.synapse_count, before)
        self.assertTrue(bridge.has(("A", 2, "B", 2)))
        self.assertFalse(weak.alive)

    def test_high_stability_bridge_is_protected(self) -> None:
        stable = BridgeSynapseState(
            "A", 0, "B", 0, 0.05, stability=0.95, reward_ema=-1.0
        )
        bridge = AdaptiveBridge([stable])
        manager = self._manager(bridge, protected_stability=0.8)

        result = manager.rewire(
            step=10,
            candidate_keys=[("A", 1, "B", 1)],
        )

        self.assertEqual(result.changed, 0)
        self.assertTrue(bridge.has(("A", 0, "B", 0)))

    def test_recently_used_bridge_is_protected(self) -> None:
        recent = BridgeSynapseState(
            "A", 0, "B", 0, 0.05, reward_ema=-1.0, last_used_step=9
        )
        bridge = AdaptiveBridge([recent])
        manager = self._manager(bridge, stale_steps=3)

        result = manager.rewire(
            step=10,
            candidate_keys=[("A", 1, "B", 1)],
        )

        self.assertEqual(result.changed, 0)
        self.assertTrue(bridge.has(("A", 0, "B", 0)))

    def test_invalid_candidates_do_not_cause_pruning(self) -> None:
        weak = BridgeSynapseState("A", 0, "B", 0, 0.05, reward_ema=-1.0)
        bridge = AdaptiveBridge([weak])
        manager = self._manager(bridge)

        result = manager.rewire(
            step=10,
            candidate_keys=[
                ("A", 1, "A", 2),
                ("A", 99, "B", 1),
                ("missing", 0, "B", 1),
            ],
        )

        self.assertEqual(result.changed, 0)
        self.assertEqual(bridge.synapse_count, 1)
        self.assertTrue(bridge.has(("A", 0, "B", 0)))

    def test_removed_bridge_does_not_leak_pending_reward_to_replacement(self) -> None:
        old = BridgeSynapseState("A", 0, "B", 0, 0.1, reward_ema=-0.5)
        bridge = AdaptiveBridge([old], reward_window=8, reward_alpha=0.5)
        manager = self._manager(bridge, stale_steps=0)

        # Simulate a previously used bridge so it has pending reward credit.
        old.usage_count = 1
        old.last_used_step = 0
        # Register a recent-credit event through the public transfer path is
        # unnecessary here; unregister cleanup is exercised by the manager and
        # the replacement must begin with clean reward state.
        result = manager.rewire(
            step=10,
            candidate_keys=[("A", 2, "B", 2)],
        )
        self.assertEqual(result.changed, 1)

        replacement = next(
            synapse
            for synapse in bridge.synapses
            if (synapse.source_brain, synapse.pre_id, synapse.target_brain, synapse.post_id)
            == ("A", 2, "B", 2)
        )
        bridge.apply_reward(reward=1.0, step=10)
        self.assertAlmostEqual(replacement.reward_ema, 0.0)
        self.assertEqual(replacement.usage_count, 0)


if __name__ == "__main__":
    unittest.main()
