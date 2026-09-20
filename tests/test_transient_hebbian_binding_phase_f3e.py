import inspect
import unittest

import numpy as np

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns.brain import PlasticMaleCNSBrain
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.whole_brain import (
    PlasticSparseFlyBrain,
    PlasticStateConfig,
    TransientHebbianBindingConfig,
)


def graph():
    signed = np.asarray([10.0, 7.0, 8.0, 6.0, 9.0], dtype=np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(10, 16, dtype=np.int64),
        indptr=np.asarray([0, 2, 3, 4, 5, 5, 5], dtype=np.int64),
        post_indices=np.asarray([1, 2, 3, 3, 4], dtype=np.int32),
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.asarray([17, 8, 6, 9, 0, 0], dtype=np.float32),
        presynaptic_sign=np.ones(6, dtype=np.int8),
        consensus_nt=np.asarray(["acetylcholine"] * 6, dtype=object),
        metadata={"superclass": np.asarray(["visual_projection"] * 6, dtype=object)},
        min_connection_synapses=1,
    )


def brain(enabled=True, seed=19):
    return PlasticMaleCNSBrain(
        graph(),
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=seed),
        transient_hebbian_config=TransientHebbianBindingConfig(enabled=enabled),
    )


class TransientHebbianBindingF3ETests(unittest.TestCase):
    def test_disabled_by_default(self):
        value = TransientHebbianBindingConfig()
        self.assertFalse(value.enabled)
        self.assertEqual(value.pre_trace_tau_ms, 20.0)
        self.assertEqual(value.binding_tau_ms, 40.0)
        self.assertEqual(value.binding_increment, 0.10)
        self.assertEqual(value.max_binding_gain, 0.50)
        self.assertFalse(brain(enabled=False)._hebb_enabled)

    def test_existing_edge_and_previous_trace_are_required(self):
        value = brain()
        value._refresh_hebbian_pre_traces(np.asarray([0], dtype=np.int32))
        value._apply_hebbian_bindings(np.asarray([1], dtype=np.int32))
        self.assertGreater(float(value.binding_gain[0]), 0.0)
        self.assertEqual(int(value._binding_active_count[0]), 1)
        value._apply_hebbian_bindings(np.asarray([5], dtype=np.int32))
        self.assertEqual(int(value._binding_active_count[0]), 1)

    def test_pre_trace_decay_is_lazy_and_local(self):
        value = brain()
        value._refresh_hebbian_pre_traces(np.asarray([0], dtype=np.int32))
        value.step_index = 20
        value._apply_hebbian_bindings(np.asarray([1], dtype=np.int32))
        expected = 0.10 * np.exp(-(20 * 0.2) / 20.0)
        self.assertAlmostEqual(float(value.binding_gain[0]), float(expected), places=5)
        self.assertEqual(float(value.pre_trace[0]), 1.0)

    def test_update_is_after_causal_transmission(self):
        value = brain()
        value._refresh_hebbian_pre_traces(np.asarray([0], dtype=np.int32))
        target = value._delay_ring[(value.step_index + value.delay_steps) % len(value._delay_ring)]
        value._schedule_spike_outputs(np.asarray([0], dtype=np.int32))
        causal_value = float(target[1])
        value._apply_hebbian_bindings(np.asarray([1], dtype=np.int32))
        self.assertAlmostEqual(float(target[1]), causal_value, places=6)
        self.assertGreater(float(value.binding_gain[0]), 0.0)

    def test_binding_transmission_and_decay(self):
        value = brain()
        value.binding_gain[0] = 0.20
        value.binding_last_step[0] = 0
        value.binding_active_mask[0] = True
        value.binding_active_edges[0] = 0
        value._binding_active_count[0] = 1
        target = value._delay_ring[(value.step_index + value.delay_steps) % len(value._delay_ring)]
        value._schedule_spike_outputs(np.asarray([0], dtype=np.int32))
        expected = 10.0 * value.params.mv_per_synapse * 1.20
        self.assertAlmostEqual(float(target[1]), expected, places=5)
        value.step_index = 20
        decayed = value._hebb_gain_for_edge(0)
        self.assertAlmostEqual(decayed, 0.20 * np.exp(-4.0 / 40.0), places=5)

    def test_binding_cap_and_reset(self):
        value = brain()
        value._refresh_hebbian_pre_traces(np.asarray([0], dtype=np.int32))
        for _ in range(10):
            value._apply_hebbian_bindings(np.asarray([1], dtype=np.int32))
        self.assertLessEqual(float(value.binding_gain[0]), 0.50)
        persistent = value.plasticity.multiplier.copy()
        value.reset()
        self.assertEqual(int(value._binding_active_count[0]), 0)
        self.assertEqual(int(value._hebb_pre_active_count), 0)
        self.assertTrue(np.all(value.binding_gain == 0.0))
        self.assertTrue(np.all(value.pre_trace == 0.0))
        np.testing.assert_array_equal(value.plasticity.multiplier, persistent)

    def test_disabled_path_matches_reference_and_replay_is_deterministic(self):
        fast = brain(enabled=False)
        reference = PlasticSparseFlyBrain(
            graph(), params=FlyBrainParams(dt_ms=0.2), seed=19,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=19),
        )
        for indices in (np.asarray([0], dtype=np.int32), np.asarray([1, 2], dtype=np.int32)):
            fast._schedule_spike_outputs(indices)
            reference._schedule_spike_outputs(indices)
        for left, right in zip(fast._delay_ring, reference._delay_ring):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-6)

        left = brain(enabled=True, seed=23)
        right = brain(enabled=True, seed=23)
        events = ((0, (0,)), (1, (1,)), (2, (3,)), (3, (0, 1)))
        for step, fired in events:
            left.step_index = right.step_index = step
            for value in (left, right):
                value._schedule_spike_outputs(np.asarray(fired, dtype=np.int32))
                value._apply_hebbian_bindings(np.asarray(fired, dtype=np.int32))
                value._refresh_hebbian_pre_traces(np.asarray(fired, dtype=np.int32))
        np.testing.assert_array_equal(left.pre_trace, right.pre_trace)
        np.testing.assert_array_equal(left.binding_gain, right.binding_gain)
        np.testing.assert_array_equal(left._binding_active_count, right._binding_active_count)
        count = int(left._binding_active_count[0])
        np.testing.assert_array_equal(left.binding_active_edges[:count], right.binding_active_edges[:count])

    def test_mechanism_contains_no_task_semantics(self):
        from drosomath.whole_brain.transient_hebbian import TransientHebbianBindingConfig

        source = inspect.getsource(TransientHebbianBindingConfig)
        for forbidden in ("keyboard", "grammar", "target", "context identity"):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__":
    unittest.main()
