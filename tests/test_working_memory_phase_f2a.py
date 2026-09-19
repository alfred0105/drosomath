from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    DelayedCueSession,
    GO_ALLOCATION_SEED,
    GO_SYMBOL,
    NO_DECISION,
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    pairwise_set_jaccard,
)
from drosomath.malecns.loader import MaleCNSConnectome
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler


def make_connectome(n: int = 320) -> MaleCNSConnectome:
    posts = np.column_stack(
        (np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n)
    ).reshape(-1)
    indptr = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    signed = np.where(np.arange(2 * n) % 3 == 0, -2.0, 3.0).astype(np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(10_000, 10_000 + n, dtype=np.int64),
        indptr=indptr,
        post_indices=posts.astype(np.int32),
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.full(n, 5.0, dtype=np.float32),
        presynaptic_sign=np.ones(n, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * n, dtype=object),
        min_connection_synapses=5,
    )


class StubBrain:
    """Small protocol spy: no learning state and no hidden target signal."""

    def __init__(self, connectome, interface):
        self.connectome = connectome
        self.np = np
        self.params = FlyBrainParams(dt_ms=0.2)
        self.v = np.full(connectome.neuron_count, self.params.resting_mv, dtype=np.float32)
        self.g = np.zeros(connectome.neuron_count, dtype=np.float32)
        self.refractory_until = np.zeros(connectome.neuron_count, dtype=np.int64)
        self._fast_active = np.empty(0, dtype=np.int32)
        self.step_index = 0
        self.reset_count = 0
        self.tracking_calls = []
        self.step_calls = []
        self.output_index = int(interface.output_populations["A"][0])
        self.persistent = np.asarray([1.25, 2.5], dtype=np.float32)

    def set_plasticity_tracking(self, enabled: bool) -> bool:
        previous = getattr(self, "tracking_enabled", True)
        self.tracking_enabled = bool(enabled)
        self.tracking_calls.append(bool(enabled))
        return previous

    def reset(self) -> None:
        self.reset_count += 1
        self.v.fill(self.params.resting_mv)
        self.g.fill(0.0)
        self.refractory_until.fill(0)
        self._fast_active = np.empty(0, dtype=np.int32)

    def step(self, *, stimulus_indices=None, stimulus_rate_hz=0.0):
        indices = None if stimulus_indices is None else np.asarray(stimulus_indices, dtype=np.int32)
        self.step_calls.append((indices, float(stimulus_rate_hz)))
        self.step_index += 1
        if indices is not None and len(indices):
            self._fast_active = np.asarray([0], dtype=np.int32)
            self.v[0] = self.params.resting_mv + 1.0
        else:
            self._fast_active = np.asarray([0], dtype=np.int32)
        return np.asarray([self.output_index], dtype=np.int32), 1


class WorkingMemoryFoundationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.config = SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
        )
        cls.interface = SymbolInterface(cls.connectome, cls.config)
        cls.wm = WorkingMemoryInterface(cls.connectome, cls.interface)

    def test_input_and_output_vocabularies_are_separate(self):
        self.assertEqual(self.wm.input_vocabulary, ("A", "B", "C", "D", "GO"))
        self.assertEqual(self.wm.output_vocabulary, SYMBOLS)
        self.assertNotIn(GO_SYMBOL, self.wm.output_populations)

    def test_go_population_is_exactly_32_real_indices(self):
        go = self.wm.go_population
        self.assertEqual(len(go), 32)
        self.assertEqual(len(np.unique(go)), 32)
        self.assertTrue(set(go).issubset(set(range(self.connectome.neuron_count))))

    def test_go_population_is_deterministic(self):
        other = WorkingMemoryInterface(self.connectome, self.interface, go_seed=GO_ALLOCATION_SEED)
        np.testing.assert_array_equal(self.wm.go_population, other.go_population)

    def test_go_population_is_disjoint_from_symbol_sensory(self):
        sensory = np.concatenate(tuple(self.interface.sensory_populations.values()))
        self.assertEqual(len(np.intersect1d(self.wm.go_population, sensory)), 0)

    def test_go_population_is_disjoint_from_outputs(self):
        outputs = np.concatenate(tuple(self.interface.output_populations.values()))
        self.assertEqual(len(np.intersect1d(self.wm.go_population, outputs)), 0)

    def test_go_allocation_does_not_consume_external_rng(self):
        rng = np.random.default_rng(919)
        before = copy.deepcopy(rng.bit_generator.state)
        WorkingMemoryInterface(self.connectome, self.interface)
        self.assertEqual(before, rng.bit_generator.state)

    def test_allocation_summary_reports_invariants(self):
        summary = self.wm.allocation_summary()
        self.assertEqual(summary["go_population_size"], 32)
        self.assertEqual(summary["go_seed"], 23)
        self.assertEqual(summary["sensory_overlap"], 0)
        self.assertEqual(summary["output_overlap"], 0)

    def test_one_trial_resets_once_at_start(self):
        brain = StubBrain(self.connectome, self.interface)
        row = DelayedCueSession(brain, self.wm).run_trial("A")
        self.assertEqual(brain.reset_count, 1)
        self.assertFalse(row.reset_before_go)

    def test_reset_ablation_adds_exactly_one_reset_before_go(self):
        brain = StubBrain(self.connectome, self.interface)
        row = DelayedCueSession(brain, self.wm).run_trial("A", reset_before_go=True)
        self.assertEqual(brain.reset_count, 2)
        self.assertTrue(row.reset_before_go)
        self.assertEqual(row.post_reset_active_count, 0)

    def test_delay_is_silent_and_has_zero_rate(self):
        brain = StubBrain(self.connectome, self.interface)
        DelayedCueSession(brain, self.wm).run_trial("B")
        self.assertGreaterEqual(len(brain.step_calls), 3)
        delay_calls = brain.step_calls[100:200]
        self.assertTrue(all(indices is None and rate == 0.0 for indices, rate in delay_calls))

    def test_cue_and_go_use_205_hz_for_20_ms(self):
        brain = StubBrain(self.connectome, self.interface)
        DelayedCueSession(brain, self.wm).run_trial("C")
        self.assertEqual(len(brain.step_calls), 300)
        self.assertTrue(all(rate == 205.0 for _, rate in brain.step_calls[:100]))
        self.assertTrue(all(rate == 205.0 for _, rate in brain.step_calls[200:]))

    def test_decision_is_based_on_go_only(self):
        brain = StubBrain(self.connectome, self.interface)

        class ScriptedSession(DelayedCueSession):
            def __init__(self, *args):
                super().__init__(*args)
                self.phase = 0

            def _run_phase(self, *args):
                self.phase += 1
                if self.phase == 1:
                    return ({"A": 99.0, "B": 0.0, "C": 0.0, "D": 0.0}, 1, 1)
                if self.phase == 2:
                    return ({"A": 0.0, "B": 99.0, "C": 0.0, "D": 0.0}, 1, 1)
                return ({"A": 0.0, "B": 0.0, "C": 99.0, "D": 0.0}, 1, 1)

        row = ScriptedSession(brain, self.wm).run_trial("D")
        self.assertEqual(row.decision, "C")
        self.assertNotEqual(row.decision, row.cue)

    def test_go_is_not_an_output_decision(self):
        brain = StubBrain(self.connectome, self.interface)
        row = DelayedCueSession(brain, self.wm).run_trial("A")
        self.assertNotIn(GO_SYMBOL, row.go_output_rates_hz)
        self.assertNotEqual(row.decision, GO_SYMBOL)

    def test_pre_go_state_has_compact_hash_and_metrics(self):
        brain = StubBrain(self.connectome, self.interface)
        row = DelayedCueSession(brain, self.wm).run_trial("A")
        self.assertEqual(len(row.pre_go_fingerprint), 64)
        self.assertEqual(row.pre_go_active_count, len(row.pre_go_active_indices))
        self.assertGreaterEqual(row.pre_go_membrane_norm, 0.0)
        self.assertGreaterEqual(row.pre_go_conductance_norm, 0.0)

    def test_reset_ablation_removes_transient_pre_go_state(self):
        brain = StubBrain(self.connectome, self.interface)
        row = DelayedCueSession(brain, self.wm).run_trial("A", reset_before_go=True)
        self.assertGreater(row.pre_go_active_count, 0)
        self.assertEqual(row.post_reset_active_count, 0)

    def test_learning_disabled_session_never_calls_learning(self):
        brain = StubBrain(self.connectome, self.interface)
        brain.learn_from_reward = lambda *args, **kwargs: self.fail("learning was called")
        DelayedCueSession(brain, self.wm, learning_enabled=False).run_trial("A")

    def test_learning_enabled_is_rejected_until_f2b(self):
        with self.assertRaisesRegex(NotImplementedError, "learning-disabled"):
            DelayedCueSession(StubBrain(self.connectome, self.interface), self.wm, learning_enabled=True)

    def test_learning_signal_is_only_prepared_for_future_phase(self):
        brain = StubBrain(self.connectome, self.interface)
        session = DelayedCueSession(brain, self.wm)
        row = session.run_trial("A")
        signal = session.build_learning_signal(row)
        self.assertTrue(hasattr(signal, "directional_error"))
        self.assertEqual(brain.reset_count, 1)

    def test_tracking_is_disabled_for_f2a_and_restored(self):
        brain = StubBrain(self.connectome, self.interface)
        brain.tracking_enabled = True
        DelayedCueSession(brain, self.wm).run_trial("A", track_eligibility=False)
        self.assertEqual(brain.tracking_calls[0], False)
        self.assertEqual(brain.tracking_calls[-1], True)

    def test_tracking_can_span_episode_for_f2b_without_learning(self):
        brain = StubBrain(self.connectome, self.interface)
        brain.tracking_enabled = False
        DelayedCueSession(brain, self.wm).run_trial(
            "A", track_eligibility=True, retain_tracking=True
        )
        self.assertEqual(brain.tracking_calls[0], True)
        self.assertTrue(brain.tracking_enabled)

    def test_pairwise_jaccard_is_deterministic_and_bounded(self):
        brain = StubBrain(self.connectome, self.interface)
        session = DelayedCueSession(brain, self.wm)
        rows = [session.run_trial(symbol) for symbol in SYMBOLS]
        values = pairwise_set_jaccard(rows)
        self.assertEqual(set(values), {"A|B", "A|C", "A|D", "B|C", "B|D", "C|D"})
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values.values()))

    def test_source_has_no_external_decoder(self):
        source = inspect.getsource(sys.modules[DelayedCueSession.__module__])
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("softmax", source.lower())

    def test_source_does_not_select_edges_by_keyboard_label(self):
        source = inspect.getsource(sys.modules[DelayedCueSession.__module__])
        self.assertNotIn("teacher_edges", source)
        self.assertNotIn("click_teacher_edges_by_label", source)

    def test_f2a_result_serialization_omits_dense_active_array(self):
        brain = StubBrain(self.connectome, self.interface)
        row = DelayedCueSession(brain, self.wm).run_trial("A")
        payload = row.to_dict()
        self.assertNotIn("pre_go_active_indices", payload)
        self.assertIn("pre_go_fingerprint", payload)

    def test_brain_reset_preserves_persistent_arrays(self):
        brain = PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=11,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=11),
        )
        before = brain.plasticity.multiplier.copy()
        brain.plasticity.multiplier[0] = 1.2345
        brain.reset()
        self.assertAlmostEqual(float(brain.plasticity.multiplier[0]), 1.2345, places=5)
        np.testing.assert_array_equal(brain.plasticity.multiplier[1:], before[1:])

    def test_profiler_report_is_state_free_and_serializable(self):
        profiler = TimingProfiler()
        profiler.add("reward_update_seconds", 1.0)
        profiler.add("post_reward_directional_seconds", 2.0)
        report = profiler.report()
        self.assertTrue(report["enabled"])
        self.assertEqual(report["seconds"]["reward_update_seconds"], 1.0)
        self.assertEqual(report["counts"]["post_reward_directional_seconds"], 1)

    def test_corrected_timing_sections_are_named_disjointly(self):
        source = inspect.getsource(PlasticMaleCNSBrain.learn_from_reward)
        for name in (
            "reward_update_seconds",
            "post_reward_directional_seconds",
            "normalizer_seconds",
            "plastic_lifecycle_seconds",
        ):
            self.assertIn(name, source)
        self.assertNotIn("normalization_seconds", source)

    def test_no_decision_is_allowed_when_go_is_silent(self):
        class SilentSession(DelayedCueSession):
            def _run_phase(self, *args):
                return ({symbol: 0.0 for symbol in SYMBOLS}, 0, 0)

        brain = StubBrain(self.connectome, self.interface)
        row = SilentSession(brain, self.wm).run_trial("A")
        self.assertEqual(row.decision, NO_DECISION)

    def test_go_rate_changes_do_not_change_input_vocabularies(self):
        self.assertEqual(self.wm.input_vocabulary, ("A", "B", "C", "D", "GO"))
        self.assertEqual(self.wm.output_vocabulary, SYMBOLS)

    def test_real_connectome_anatomy_is_not_modified_by_allocation(self):
        indptr = self.connectome.indptr.copy()
        posts = self.connectome.post_indices.copy()
        WorkingMemoryInterface(self.connectome, self.interface)
        np.testing.assert_array_equal(self.connectome.indptr, indptr)
        np.testing.assert_array_equal(self.connectome.post_indices, posts)


if __name__ == "__main__":
    unittest.main()
