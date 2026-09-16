import unittest

from drosomath.core import (
    AdaptiveBridge,
    BrainCore,
    BridgeActivityCandidateConfig,
    BridgeActivityCandidateGenerator,
    BridgeStructuralConfig,
    BridgeStructuralPlasticityManager,
    BridgeSynapseState,
    PlasticityTracker,
    SpikingNetwork,
    SynapseState,
)


class BridgeActivityCandidateTests(unittest.TestCase):
    def _brain(self, name: str, synapses: list[SynapseState]) -> BrainCore:
        tracker = PlasticityTracker(synapses)
        network = SpikingNetwork.from_ids([0, 1, 2], tracker, threshold=1.0, decay=1.0)
        return BrainCore(name=name, network=network, tracker=tracker)

    def test_successful_internal_activity_biases_bridge_endpoints(self) -> None:
        a_hot = SynapseState(
            0, 1, 0.8, usage_count=20, reward_ema=0.8, last_used_step=10
        )
        a_cold = SynapseState(1, 2, 0.2)
        b_hot = SynapseState(
            2, 1, 0.8, usage_count=15, reward_ema=0.7, last_used_step=10
        )
        b_cold = SynapseState(0, 1, 0.2)
        brain_a = self._brain("A", [a_hot, a_cold])
        brain_b = self._brain("B", [b_hot, b_cold])
        generator = BridgeActivityCandidateGenerator(
            config=BridgeActivityCandidateConfig(max_candidates=4, pool_size=2)
        )

        candidates = generator.generate(
            {"A": brain_a, "B": brain_b},
            AdaptiveBridge(),
            step=10,
        )

        self.assertTrue(candidates)
        # Hot A neurons (0/1) and hot B neurons (2/1) should dominate the front.
        first = candidates[0]
        self.assertIn(first[0], {"A", "B"})
        self.assertNotEqual(first[0], first[2])
        self.assertIn(first[1], {0, 1, 2})
        self.assertIn(first[3], {0, 1, 2})

    def test_existing_bridge_is_excluded(self) -> None:
        brain_a = self._brain(
            "A",
            [SynapseState(0, 1, 0.8, usage_count=10, reward_ema=1.0, last_used_step=5)],
        )
        brain_b = self._brain(
            "B",
            [SynapseState(0, 1, 0.8, usage_count=10, reward_ema=1.0, last_used_step=5)],
        )
        existing = BridgeSynapseState("A", 0, "B", 0, 0.5)
        bridge = AdaptiveBridge([existing])
        generator = BridgeActivityCandidateGenerator(
            config=BridgeActivityCandidateConfig(max_candidates=20, pool_size=3)
        )

        candidates = generator.generate(
            {"A": brain_a, "B": brain_b}, bridge, step=5
        )

        self.assertNotIn(("A", 0, "B", 0), candidates)

    def test_candidate_limit_is_respected(self) -> None:
        brain_a = self._brain("A", [SynapseState(0, 1, 0.5)])
        brain_b = self._brain("B", [SynapseState(1, 2, 0.5)])
        generator = BridgeActivityCandidateGenerator(
            config=BridgeActivityCandidateConfig(max_candidates=3, pool_size=3)
        )

        candidates = generator.generate(
            {"A": brain_a, "B": brain_b}, AdaptiveBridge(), step=0
        )

        self.assertLessEqual(len(candidates), 3)

    def test_generator_integrates_with_fixed_budget_rewire(self) -> None:
        brain_a = self._brain(
            "A",
            [SynapseState(0, 1, 0.8, usage_count=12, reward_ema=0.8, last_used_step=8)],
        )
        brain_b = self._brain(
            "B",
            [SynapseState(2, 1, 0.8, usage_count=12, reward_ema=0.8, last_used_step=8)],
        )
        stale = BridgeSynapseState("A", 2, "B", 0, 0.05, reward_ema=-1.0)
        bridge = AdaptiveBridge([stale])
        generator = BridgeActivityCandidateGenerator(
            config=BridgeActivityCandidateConfig(max_candidates=8, pool_size=3)
        )
        manager = BridgeStructuralPlasticityManager(
            bridge,
            {"A": brain_a.network.neurons, "B": brain_b.network.neurons},
            config=BridgeStructuralConfig(
                min_age_cycles=1,
                stale_steps=1,
                max_rewire_per_cycle=1,
            ),
        )

        candidates = generator.generate(
            {"A": brain_a, "B": brain_b}, bridge, step=10
        )
        before = bridge.synapse_count
        result = manager.rewire(step=10, candidate_keys=candidates)

        self.assertEqual(result.changed, 1)
        self.assertEqual(bridge.synapse_count, before)
        self.assertFalse(stale.alive)


if __name__ == "__main__":
    unittest.main()
