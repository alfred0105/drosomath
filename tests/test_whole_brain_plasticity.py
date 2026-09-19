import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional whole-brain dependency")
class WholeBrainPlasticityTests(unittest.TestCase):

    def test_sparse_lifecycle_cache_matches_dense_decay_and_clear(self):
        import numpy as np
        from drosomath.whole_brain import PlasticStateConfig
        from drosomath.whole_brain.plastic_state import SparsePlasticityState

        state = SparsePlasticityState(
            257, config=PlasticStateConfig(plastic_fraction=0.25, seed=12)
        )
        rng = np.random.default_rng(4)
        # Simulate a curriculum that first unlocked all edges and later froze
        # most of them: frozen historical state must still decay exactly.
        state.set_plastic_fraction(1.0)
        state.usage_ema[:] = rng.random(state.edge_count, dtype=np.float32)
        state.eligibility[:] = rng.random(state.edge_count, dtype=np.float32)
        state.stability[:] = rng.random(state.edge_count, dtype=np.float32)
        state.set_plastic_fraction(0.25)

        dense_usage = state.usage_ema.copy() * np.float32(0.995)
        dense_eligibility = state.eligibility.copy() * np.float32(0.90)
        dense_stability = state.stability.copy()
        state.decay_episode(usage_decay=0.995, eligibility_decay=0.90)
        np.testing.assert_allclose(state.usage_ema, dense_usage, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(state.eligibility, dense_eligibility, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(state.stability, dense_stability)

        state.clear_eligibility()
        np.testing.assert_array_equal(state.eligibility, np.zeros(state.edge_count, dtype=np.float32))

    def test_recent_presynaptic_reward_matches_full_scan(self):
        import numpy as np
        from drosomath.whole_brain import PlasticStateConfig
        from drosomath.whole_brain.plastic_state import SparsePlasticityState
        from drosomath.whole_brain.usage_learning import UsageRewardRule

        edge_count = 31
        cfg = PlasticStateConfig(plastic_fraction=1.0, seed=6)
        full = SparsePlasticityState(edge_count, config=cfg)
        recent = SparsePlasticityState(edge_count, config=cfg)
        rng = np.random.default_rng(8)
        for state in (full, recent):
            state.usage_ema[:] = rng.random(edge_count, dtype=np.float32)
            state.eligibility[:] = 0.0
            state.stability[:] = rng.random(edge_count, dtype=np.float32)
        # Only rows 1 and 3 received activity in this trial.
        indptr = np.asarray([0, 4, 10, 17, 24, 31], dtype=np.int64)
        active_edges = np.r_[4:10, 17:24]
        full.eligibility[active_edges] = 1.0
        recent.eligibility[active_edges] = 1.0
        recent.usage_ema[:] = full.usage_ema
        recent.stability[:] = full.stability
        rule = UsageRewardRule(learning_rate=0.1)

        expected = rule.apply(full, reward=-0.75)
        actual = rule.apply_recent_presynaptic(
            recent,
            reward=-0.75,
            indptr=indptr,
            presynaptic_indices=[1, 3],
        )
        self.assertEqual(actual.edge_updates, expected.edge_updates)
        self.assertAlmostEqual(actual.mean_abs_delta, expected.mean_abs_delta, places=8)
        self.assertAlmostEqual(actual.max_abs_delta, expected.max_abs_delta, places=8)
        np.testing.assert_allclose(recent.multiplier, full.multiplier, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(recent.stability, full.stability, rtol=0.0, atol=1e-7)
    def test_more_used_rewarded_edge_strengthens_more(self) -> None:
        from drosomath.whole_brain.plastic_state import SparsePlasticityState
        from drosomath.whole_brain.usage_learning import UsageRewardRule

        state = SparsePlasticityState(2)
        for _ in range(5):
            state.record_use_slice(0, 1, usage_alpha=0.5)
        state.record_use_slice(1, 2, usage_alpha=0.5)

        before = state.multiplier.copy()
        stats = UsageRewardRule(learning_rate=0.1).apply(state, reward=1.0)

        delta_frequent = float(state.multiplier[0] - before[0])
        delta_rare = float(state.multiplier[1] - before[1])
        self.assertEqual(stats.edge_updates, 2)
        self.assertGreater(delta_frequent, delta_rare)
        self.assertGreater(delta_rare, 0.0)

    def test_negative_reward_weakens_without_flipping_anatomical_sign(self) -> None:
        import numpy as np

        from drosomath.whole_brain.plastic_state import SparsePlasticityState
        from drosomath.whole_brain.usage_learning import UsageRewardRule

        state = SparsePlasticityState(1)
        state.record_use_slice(0, 1, usage_alpha=1.0)
        UsageRewardRule(learning_rate=0.2).apply(state, reward=-1.0)

        base = np.asarray([-12.0], dtype=np.float32)
        effective = state.effective_signed_slice(base, 0, 1)
        self.assertLess(float(state.multiplier[0]), 1.0)
        self.assertLess(float(effective[0]), 0.0)

    def test_outgoing_budget_prevents_runaway_strength(self) -> None:
        import numpy as np

        from drosomath.whole_brain.homeostasis import OutgoingBudgetNormalizer
        from drosomath.whole_brain.plastic_state import SparsePlasticityState

        state = SparsePlasticityState(2)
        state.multiplier[:] = 2.0
        indptr = np.asarray([0, 2], dtype=np.int64)
        base_abs = np.asarray([10.0, 10.0], dtype=np.float32)

        stats = OutgoingBudgetNormalizer(strength=1.0).normalize_presynaptic(
            state,
            indptr=indptr,
            base_abs=base_abs,
        )

        self.assertEqual(stats.neurons_touched, 1)
        self.assertAlmostEqual(float(state.multiplier[0]), 1.0, places=6)
        self.assertAlmostEqual(float(state.multiplier[1]), 1.0, places=6)

    def test_plastic_fraction_can_freeze_anatomical_edges(self) -> None:
        from drosomath.whole_brain.plastic_state import (
            PlasticStateConfig,
            SparsePlasticityState,
        )

        state = SparsePlasticityState(
            8,
            config=PlasticStateConfig(plastic_fraction=0.0),
        )
        credited = state.record_use_slice(0, 8)
        self.assertEqual(credited, 0)
        self.assertEqual(state.plastic_edge_count, 0)
        self.assertTrue((state.multiplier == 1.0).all())

    def test_sparse_brain_records_use_and_preserves_learned_memory_on_reset(self) -> None:
        import numpy as np

        from drosomath.flywire_real import FlyWireConnectome
        from drosomath.whole_brain import PlasticSparseFlyBrain, UsageRewardRule

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([101, 102], dtype=np.int64),
            indptr=np.asarray([0, 1, 1], dtype=np.int64),
            post_indices=np.asarray([1], dtype=np.int32),
            signed_synapse_counts=np.asarray([12.0], dtype=np.float32),
            outgoing_strength=np.asarray([12.0, 0.0], dtype=np.float32),
        )
        brain = PlasticSparseFlyBrain(connectome, usage_alpha=1.0)

        brain._schedule_spike_outputs(np.asarray([0], dtype=np.int32))
        before = float(brain.plasticity.multiplier[0])
        report = brain.learn_from_reward(
            reward=1.0,
            rule=UsageRewardRule(learning_rate=0.1),
        )
        learned = float(brain.plasticity.multiplier[0])

        self.assertEqual(report["learning"]["edge_updates"], 1)
        self.assertGreater(learned, before)

        brain.reset()
        self.assertAlmostEqual(float(brain.plasticity.multiplier[0]), learned)

        brain.reset_all()
        self.assertAlmostEqual(float(brain.plasticity.multiplier[0]), 1.0)

    def test_live_telemetry_samples_actual_transmitted_edges(self) -> None:
        import numpy as np

        from drosomath.flywire_real import FlyWireConnectome
        from drosomath.whole_brain import PlasticSparseFlyBrain

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([101, 102, 103], dtype=np.int64),
            indptr=np.asarray([0, 2, 2, 2], dtype=np.int64),
            post_indices=np.asarray([1, 2], dtype=np.int32),
            signed_synapse_counts=np.asarray([12.0, -8.0], dtype=np.float32),
            outgoing_strength=np.asarray([20.0, 0.0, 0.0], dtype=np.float32),
        )
        brain = PlasticSparseFlyBrain(connectome)
        brain.configure_live_telemetry(
            True,
            max_active_edges=1,
            edges_per_firing_neuron=2,
        )
        transferred = brain._schedule_spike_outputs(np.asarray([0], dtype=np.int32))
        snap = brain.live_telemetry_snapshot()

        self.assertEqual(transferred, 2)
        self.assertEqual(snap["fired_neuron_count"], 1)
        self.assertEqual(snap["fired_neuron_ids"], [101])
        self.assertEqual(snap["transferred_synapses"], 2)
        self.assertEqual(len(snap["active_edges"]), 1)
        self.assertEqual(snap["active_edges"][0]["pre_id"], 101)
        self.assertEqual(snap["active_edges"][0]["post_id"], 102)
        self.assertGreater(abs(snap["active_edges"][0]["signal_mv"]), 0.0)


if __name__ == "__main__":
    unittest.main()
