import copy
import inspect
import unittest


class SlowAdaptationPhaseF3CTests(unittest.TestCase):
    @staticmethod
    def graph():
        import numpy as np
        from drosomath.malecns.loader import MaleCNSConnectome

        signed = np.asarray([10, 8, 6], dtype=np.float32)
        return MaleCNSConnectome(
            body_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
            indptr=np.asarray([0, 1, 2, 3, 3], dtype=np.int64),
            post_indices=np.asarray([1, 2, 3], dtype=np.int32),
            synapse_counts=np.asarray([10, 8, 6], dtype=np.int32),
            signed_synapse_counts=signed,
            outgoing_strength=np.asarray([10, 8, 6, 0], dtype=np.float32),
            presynaptic_sign=np.ones(4, dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine"] * 4, dtype=object),
            metadata={"superclass": np.asarray(["test"] * 4, dtype=object)},
            min_connection_synapses=1,
        )

    @staticmethod
    def brain(*, enabled=False, seed=5):
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.whole_brain import PlasticStateConfig, SlowAdaptationConfig

        return PlasticMaleCNSBrain(
            SlowAdaptationPhaseF3CTests.graph(),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=seed),
            slow_adaptation_config=SlowAdaptationConfig(enabled=enabled),
        )

    def test_default_is_disabled_with_protocol_values(self):
        from drosomath.whole_brain import SlowAdaptationConfig

        config = SlowAdaptationConfig()
        self.assertFalse(config.enabled)
        self.assertEqual((config.tau_ms, config.spike_increment_mv, config.max_adaptation_mv), (30.0, 0.75, 3.0))

    def test_config_rejects_invalid_values(self):
        from drosomath.whole_brain import SlowAdaptationConfig

        with self.assertRaises(ValueError):
            SlowAdaptationConfig(tau_ms=0.0)
        with self.assertRaises(ValueError):
            SlowAdaptationConfig(spike_increment_mv=4.0, max_adaptation_mv=3.0)

    def test_adaptation_is_float32_and_bounded(self):
        brain = self.brain(enabled=True)
        self.assertEqual(str(brain.adaptation_mv.dtype), "float32")
        brain._apply_slow_adaptation_spikes(brain.np.asarray([0], dtype=brain.np.int32))
        brain._apply_slow_adaptation_spikes(brain.np.asarray([0], dtype=brain.np.int32))
        brain._apply_slow_adaptation_spikes(brain.np.asarray([0], dtype=brain.np.int32))
        self.assertLessEqual(float(brain.adaptation_mv[0]), 3.0)
        self.assertAlmostEqual(float(brain.adaptation_mv[0]), 2.25, places=5)

    def test_decay_applies_only_to_relevant_indices(self):
        brain = self.brain(enabled=True)
        brain.adaptation_mv[:] = 2.0
        active = brain.np.asarray([0, 2], dtype=brain.np.int32)
        brain._apply_slow_adaptation_decay(active)
        self.assertLess(float(brain.adaptation_mv[0]), 2.0)
        self.assertLess(float(brain.adaptation_mv[2]), 2.0)
        self.assertEqual(float(brain.adaptation_mv[1]), 2.0)

    def test_effective_threshold_uses_pre_increment_adaptation(self):
        import numpy as np
        from drosomath.flywire_real import FlyBrainParams

        brain = self.brain(enabled=True)
        brain.v[0] = np.float32(brain.params.threshold_mv + 0.5)
        brain.adaptation_mv[0] = np.float32(0.75)
        active = np.asarray([0], dtype=np.int32)
        fired = brain._advance_python_with_slow_adaptation(
            active, np.empty(0, dtype=np.int32), brain._delay_ring[0],
            np.empty(0, dtype=np.int32), brain.params,
        )
        self.assertEqual(len(fired), 0)
        self.assertLess(float(brain.adaptation_mv[0]), 0.75)

    def test_fired_neuron_increments_after_decision(self):
        import numpy as np

        brain = self.brain(enabled=True)
        brain.v[0] = np.float32(brain.params.threshold_mv + 2.0)
        fired = brain._advance_python_with_slow_adaptation(
            np.asarray([0], dtype=np.int32), np.empty(0, dtype=np.int32), brain._delay_ring[0],
            np.empty(0, dtype=np.int32), brain.params,
        )
        np.testing.assert_array_equal(fired, np.asarray([0], dtype=np.int32))
        self.assertAlmostEqual(float(brain.adaptation_mv[0]), 0.75, places=5)

    def test_reset_clears_transient_adaptation(self):
        brain = self.brain(enabled=True)
        brain.adaptation_mv[0] = 2.0
        brain.reset()
        self.assertTrue(bool((brain.adaptation_mv == 0.0).all()))

    def test_adaptation_is_not_in_persistent_learning_snapshot(self):
        from drosomath.malecns.symbol_interface import _snapshot_persistent_state

        brain = self.brain(enabled=True)
        brain.adaptation_mv[0] = 1.0
        snapshot = _snapshot_persistent_state(brain)
        self.assertNotIn("adaptation_mv", snapshot)

    def test_disabled_path_is_exactly_equal_to_explicit_disabled_path(self):
        import numpy as np
        from drosomath.whole_brain import SlowAdaptationConfig
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.whole_brain import PlasticStateConfig

        graph = self.graph()
        left = PlasticMaleCNSBrain(graph, seed=9, plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=9))
        right = PlasticMaleCNSBrain(graph, seed=9, plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=9), slow_adaptation_config=SlowAdaptationConfig(enabled=False))
        stimulus = np.asarray([0], dtype=np.int32)
        for _ in range(20):
            left_result = left.step(stimulus_indices=stimulus, stimulus_rate_hz=1000.0)
            right_result = right.step(stimulus_indices=stimulus, stimulus_rate_hz=1000.0)
            np.testing.assert_array_equal(left_result[0], right_result[0])
            self.assertEqual(left_result[1], right_result[1])
            np.testing.assert_array_equal(left.v, right.v)
            np.testing.assert_array_equal(left.g, right.g)
            np.testing.assert_array_equal(left.refractory_until, right.refractory_until)
            np.testing.assert_array_equal(left._fast_active, right._fast_active)
            self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)

    def test_enabled_replay_is_deterministic(self):
        import numpy as np

        left = self.brain(enabled=True, seed=12)
        right = self.brain(enabled=True, seed=12)
        stimulus = np.asarray([0], dtype=np.int32)
        for _ in range(20):
            l = left.step(stimulus_indices=stimulus, stimulus_rate_hz=1000.0)
            r = right.step(stimulus_indices=stimulus, stimulus_rate_hz=1000.0)
            np.testing.assert_array_equal(l[0], r[0])
            np.testing.assert_array_equal(left.adaptation_mv, right.adaptation_mv)
            np.testing.assert_array_equal(left.v, right.v)
            self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)

    def test_numba_and_reference_adaptation_paths_match(self):
        import numpy as np

        compiled = self.brain(enabled=True, seed=18)
        reference = self.brain(enabled=True, seed=18)
        reference._numba_enabled = False
        stimulus = np.asarray([0], dtype=np.int32)
        for _ in range(20):
            left = compiled.step(stimulus_indices=stimulus, stimulus_rate_hz=1000.0)
            right = reference.step(stimulus_indices=stimulus, stimulus_rate_hz=1000.0)
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_allclose(compiled.v, reference.v, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(compiled.g, reference.g, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(compiled.adaptation_mv, reference.adaptation_mv, rtol=0.0, atol=1e-6)

    def test_summary_reports_compact_telemetry(self):
        brain = self.brain(enabled=True)
        summary = brain.slow_adaptation_summary()
        self.assertTrue(summary["enabled"])
        self.assertIn("extra_state_bytes", summary)
        self.assertNotIn("values", summary)

    def test_generic_core_does_not_import_keyboard_modules(self):
        from drosomath.whole_brain import slow_adaptation

        source = inspect.getsource(slow_adaptation)
        self.assertNotIn("keyboard", source.lower())

    def test_adaptation_config_is_not_persistent_state(self):
        brain = self.brain(enabled=True)
        before = copy.deepcopy(brain.plasticity.multiplier)
        brain.adaptation_mv[0] = 3.0
        self.assertTrue((brain.plasticity.multiplier == before).all())


if __name__ == "__main__":
    unittest.main()
