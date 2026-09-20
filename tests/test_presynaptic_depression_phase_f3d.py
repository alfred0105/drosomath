import inspect
import unittest

import numpy as np

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns.brain import PlasticMaleCNSBrain
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.whole_brain import PlasticSparseFlyBrain, PlasticStateConfig, PresynapticDepressionConfig


def _graph():
    # 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> 4, 4 -> 5.
    body_ids = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
    indptr = np.asarray([0, 2, 3, 4, 5, 6, 6], dtype=np.int64)
    posts = np.asarray([1, 2, 3, 3, 4, 5], dtype=np.int32)
    signed = np.asarray([10, 7, 8, 6, 9, 5], dtype=np.float32)
    return MaleCNSConnectome(
        body_ids=body_ids,
        indptr=indptr,
        post_indices=posts,
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.asarray([17, 8, 6, 9, 5, 0], dtype=np.float32),
        presynaptic_sign=np.ones(6, dtype=np.int8),
        consensus_nt=np.asarray(["acetylcholine"] * 6, dtype=object),
        metadata={"superclass": np.asarray(["visual_projection"] * 6, dtype=object),},
        min_connection_synapses=1,
    )


def _brain(*, enabled=True, seed=41):
    return PlasticMaleCNSBrain(
        _graph(),
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=seed),
        presynaptic_depression_config=PresynapticDepressionConfig(enabled=enabled),
    )


def _target_slot(brain):
    return brain._delay_ring[(brain.step_index + brain.delay_steps) % len(brain._delay_ring)]


class PresynapticDepressionPhaseF3DTests(unittest.TestCase):
    def test_exact_config_defaults_and_validation(self):
        config = PresynapticDepressionConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.recovery_tau_ms, 30.0)
        self.assertEqual(config.depression_fraction, 0.15)
        self.assertEqual(config.min_release_factor, 0.50)
        with self.assertRaises(ValueError):
            PresynapticDepressionConfig(recovery_tau_ms=0.0)
        with self.assertRaises(ValueError):
            PresynapticDepressionConfig(depression_fraction=1.0)
        with self.assertRaises(ValueError):
            PresynapticDepressionConfig(min_release_factor=0.0)

    def test_disabled_path_is_exact_reference_and_has_no_dense_std_state(self):
        fast = _brain(enabled=False)
        reference = _brain(enabled=False)
        fired = np.asarray([2, 0, 1], dtype=np.int32)
        self.assertEqual(fast._schedule_spike_outputs(fired), reference._schedule_spike_outputs(fired))
        for left, right in zip(fast._delay_ring, reference._delay_ring):
            np.testing.assert_array_equal(left, right)
        for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask"):
            np.testing.assert_array_equal(getattr(fast.plasticity, name), getattr(reference.plasticity, name))
        self.assertEqual(fast.release_factor.size, 0)
        self.assertEqual(fast.last_release_step.size, 0)
        self.assertFalse(fast.presynaptic_depression_summary()["enabled"])

    def test_transmission_uses_recovered_factor_then_depresses(self):
        brain = _brain(enabled=True)
        fired = np.asarray([0], dtype=np.int32)
        start = int(brain.connectome.indptr[0])
        stop = int(brain.connectome.indptr[1])
        post = int(brain.connectome.post_indices[start])
        expected_full = float(brain.connectome.signed_synapse_counts[start]) * brain.params.mv_per_synapse

        brain._schedule_spike_outputs(fired)
        self.assertAlmostEqual(float(_target_slot(brain)[post]), expected_full, places=6)
        self.assertAlmostEqual(float(brain.release_factor[0]), 0.85, places=6)
        self.assertEqual(int(brain.last_release_step[0]), 0)

        brain.step_index = 1
        brain._schedule_spike_outputs(fired)
        recovered = 1.0 - 0.15 * np.exp(-0.2 / 30.0)
        self.assertAlmostEqual(float(_target_slot(brain)[post]), expected_full * recovered, places=5)
        self.assertAlmostEqual(float(brain.release_factor[0]), recovered * 0.85, places=5)
        self.assertGreater(float(brain.release_factor[0]), 0.50)

    def test_release_factor_is_lazy_and_floor_is_respected(self):
        brain = _brain(enabled=True)
        brain.step_index = 0
        for step in range(32):
            brain.step_index = step
            brain._schedule_spike_outputs(np.asarray([0], dtype=np.int32))
        self.assertGreaterEqual(float(brain.release_factor[0]), 0.50)
        self.assertLessEqual(float(brain.release_factor[0]), 1.0)
        untouched = np.flatnonzero(brain.last_release_step < 0)
        np.testing.assert_array_equal(brain.release_factor[untouched], np.ones(len(untouched), dtype=np.float32))
        summary = brain.presynaptic_depression_summary()
        self.assertGreater(summary["depressed_neurons"], 0)
        self.assertGreater(summary["extra_state_bytes"], 0)

    def test_reset_clears_only_transient_std_state(self):
        brain = _brain(enabled=True)
        brain._schedule_spike_outputs(np.asarray([0], dtype=np.int32))
        before = {name: getattr(brain.plasticity, name).copy() for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")}
        brain.reset()
        np.testing.assert_array_equal(brain.release_factor, np.ones(brain.connectome.neuron_count, dtype=np.float32))
        np.testing.assert_array_equal(brain.last_release_step, -np.ones(brain.connectome.neuron_count, dtype=np.int64))
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(brain.plasticity, name), value)

    def test_python_fallback_matches_numba_semantics(self):
        compiled = _brain(enabled=True, seed=53)
        fallback = _brain(enabled=True, seed=53)
        fallback._numba_enabled = False
        fired = np.asarray([2, 0, 1], dtype=np.int32)
        for step in range(5):
            compiled.step_index = step
            fallback.step_index = step
            self.assertEqual(compiled._schedule_spike_outputs(fired), fallback._schedule_spike_outputs(fired))
            for left, right in zip(compiled._delay_ring, fallback._delay_ring):
                np.testing.assert_allclose(left, right, rtol=0.0, atol=2e-6)
            np.testing.assert_allclose(compiled.release_factor, fallback.release_factor, rtol=0.0, atol=2e-6)
            np.testing.assert_array_equal(compiled.last_release_step, fallback.last_release_step)

    def test_std_is_task_independent_and_not_serialized_as_plastic_memory(self):
        from drosomath.whole_brain.presynaptic_depression import PresynapticDepressionConfig as Exported

        self.assertIs(Exported, PresynapticDepressionConfig)
        source = inspect.getsource(Exported)
        self.assertNotIn("keyboard", source.lower())
        self.assertNotIn("context", source.lower())
        brain = _brain(enabled=True)
        self.assertEqual(brain.plasticity.edge_count, len(brain.connectome.signed_synapse_counts))
        self.assertEqual(brain.presynaptic_depression_summary()["extra_state_bytes"], brain.release_factor.nbytes + brain.last_release_step.nbytes)


if __name__ == "__main__":
    unittest.main()
