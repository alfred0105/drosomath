from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    DelayedCueLearningSession,
    DelayedCueSession,
    DelayedCueTrialResult,
    GO_SYMBOL,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterface,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    WorkingMemoryLearningConfig,
    balanced_symbol_schedule,
    build_symbol_learning_signal,
)
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler
from run_delayed_cue_learning_phase_f2b import (
    _evaluate_checkpoint,
    _memory_comparison,
    _persistent_digest,
    _summarize_rows,
    _make_brain,
)
from test_working_memory_phase_f2a import make_connectome


class CountingBrain(PlasticMaleCNSBrain):
    def __init__(self, *args, **kwargs):
        self.reset_calls = 0
        self.tracking_calls = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.reset_calls += 1
        return super().reset()

    def set_plasticity_tracking(self, enabled):
        self.tracking_calls.append(bool(enabled))
        return super().set_plasticity_tracking(enabled)


class WorkingMemoryLearningPhaseF2BTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.interface_config = SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
        )
        cls.interface = SymbolInterface(cls.connectome, cls.interface_config)
        cls.wm = WorkingMemoryInterface(cls.connectome, cls.interface)
        cls.config = WorkingMemoryLearningConfig()

    def make_brain(self, seed=83, counting=False):
        cls = CountingBrain if counting else PlasticMaleCNSBrain
        return cls(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        )

    def test_f2a_guard_remains_learning_disabled(self):
        with self.assertRaises(NotImplementedError):
            DelayedCueSession(self.make_brain(), self.wm, learning_enabled=True)

    def test_f2b_is_a_separate_learning_session(self):
        self.assertIsNot(DelayedCueLearningSession, DelayedCueSession)
        self.assertIsInstance(DelayedCueLearningSession(self.make_brain(), self.wm), DelayedCueLearningSession)

    def test_protocol_timings_are_fixed(self):
        self.assertEqual((self.config.cue_ms, self.config.delay_ms, self.config.go_ms), (20.0, 20.0, 20.0))
        self.assertEqual(self.config.stimulus_rate_hz, 205.0)
        self.assertEqual(self.config.training_trials, 800)

    def test_prospective_anatomical_and_route_cache_are_required(self):
        self.assertEqual(self.config.two_hop_credit_mode, "prospective_anatomical")
        self.assertTrue(self.config.route_cache_enabled)
        with self.assertRaises(ValueError):
            WorkingMemoryLearningConfig(route_cache_enabled=False)

    def test_same_go_population_is_used_for_all_cues(self):
        expected = self.wm.go_population
        for cue in SYMBOLS:
            np.testing.assert_array_equal(self.wm.population_for_input(GO_SYMBOL), expected)
            self.assertFalse(np.array_equal(self.wm.population_for_input(cue), expected))

    def test_balanced_schedule_is_exactly_200_per_cue(self):
        schedule = balanced_symbol_schedule(cycles=200, seed=83 + self.config.schedule_seed_offset)
        self.assertEqual(len(schedule), 800)
        self.assertEqual({cue: schedule.count(cue) for cue in SYMBOLS}, {cue: 200 for cue in SYMBOLS})

    def test_intact_training_episode_has_exactly_one_reset(self):
        brain = self.make_brain(counting=True)
        session = DelayedCueLearningSession(brain, self.wm, config=self.config)
        session.train_trial("A")
        self.assertEqual(brain.reset_calls, 1)

    def test_no_reset_occurs_between_cue_delay_go(self):
        brain = self.make_brain(counting=True)
        session = DelayedCueLearningSession(brain, self.wm, config=self.config)
        session.train_trial("A")
        self.assertEqual(brain.reset_calls, 1)

    def test_tracking_is_enabled_before_episode_and_restored_after_learning(self):
        brain = self.make_brain(counting=True)
        DelayedCueLearningSession(brain, self.wm, config=self.config).train_trial("A")
        self.assertIn(True, brain.tracking_calls)
        self.assertEqual(brain.tracking_calls[-1], True)

    def test_eligibility_is_visible_after_cue_and_delay_without_clear(self):
        brain = self.make_brain()
        session = DelayedCueSession(brain, self.wm)
        observed = {}

        def observe(phase, _brain):
            observed[phase] = int(np.count_nonzero(brain.plasticity.eligibility))

        session.run_trial("A", track_eligibility=True, retain_tracking=True, phase_observer=observe)
        self.assertIn("cue", observed)
        self.assertIn("delay", observed)
        self.assertIn("go", observed)
        self.assertGreaterEqual(observed["delay"], observed["cue"])

    def test_learning_lifecycle_clears_episode_eligibility(self):
        brain = self.make_brain()
        DelayedCueLearningSession(brain, self.wm, config=self.config).train_trial("A")
        self.assertEqual(float(brain.plasticity.eligibility.max()), 0.0)
        self.assertEqual(len(brain._recent_presynaptic), 0)

    def test_next_trial_has_no_stale_eligibility(self):
        brain = self.make_brain()
        session = DelayedCueLearningSession(brain, self.wm, config=self.config)
        session.train_trial("A")
        before = brain.plasticity.eligibility.copy()
        session.train_trial("B")
        np.testing.assert_array_equal(before, np.zeros_like(before))
        self.assertEqual(float(brain.plasticity.eligibility.max()), 0.0)

    def test_learning_signal_enters_after_go_decision_and_is_generic(self):
        source = inspect.getsource(DelayedCueLearningSession.train_trial)
        self.assertLess(source.index("result = self.episode.run_trial"), source.index("build_symbol_learning_signal"))
        signal = build_symbol_learning_signal(
            target="A", decision="D", output_rates_hz={symbol: (5.0 if symbol == "D" else 0.0) for symbol in SYMBOLS}
        )
        self.assertEqual(signal.directional_error, {"symbol/A": 1.0, "symbol/D": -1.0})
        self.assertNotIn("cue", signal.directional_error)

    def test_controller_call_does_not_receive_working_memory_labels(self):
        source = inspect.getsource(DelayedCueLearningSession.train_trial)
        controller_call = source[source.index("self.controller.apply_learning_signal"):source.index("reward_report =")]
        self.assertNotIn("cue=", controller_call)
        self.assertNotIn("GO", controller_call)
        self.assertIn("self.output_context", controller_call)

    def test_learning_happens_after_go_only(self):
        source = inspect.getsource(DelayedCueLearningSession.train_trial)
        self.assertLess(source.index("result = self.episode.run_trial"), source.index("self.brain.learn_from_reward"))
        self.assertLess(source.index("build_symbol_learning_signal"), source.index("self.brain.learn_from_reward"))

    def test_training_record_is_compact(self):
        brain = self.make_brain()
        trial = DelayedCueLearningSession(brain, self.wm, config=self.config).train_trial("A")
        payload = trial.to_dict()
        self.assertNotIn("pre_go_active_indices", payload)
        self.assertIn("go_output_rates_hz", payload)

    def test_checkpoint_twins_have_identical_persistent_state(self):
        brain = self.make_brain()
        persistent = __import__("drosomath.malecns.symbol_interface", fromlist=["_snapshot_persistent_state"])._snapshot_persistent_state(brain)
        left = _make_brain(self.connectome, self.interface_config, 83)
        right = _make_brain(self.connectome, self.interface_config, 83)
        from drosomath.malecns.symbol_interface import _restore_persistent_state
        _restore_persistent_state(left, persistent)
        _restore_persistent_state(right, persistent)
        self.assertEqual(_persistent_digest(left), _persistent_digest(right))

    def test_checkpoint_evaluation_does_not_use_training_brain(self):
        brain = self.make_brain()
        persistent = __import__("drosomath.malecns.symbol_interface", fromlist=["_snapshot_persistent_state"])._snapshot_persistent_state(brain)
        rng = copy.deepcopy(brain.rng.bit_generator.state)
        digest = _persistent_digest(brain)
        _evaluate_checkpoint(self.connectome, self.wm, self.interface_config, 83, persistent, rng, 0)
        self.assertEqual(digest, _persistent_digest(brain))
        self.assertEqual(rng, brain.rng.bit_generator.state)

    def test_reset_twin_changes_transient_not_persistent_state(self):
        brain = self.make_brain()
        persistent = __import__("drosomath.malecns.symbol_interface", fromlist=["_snapshot_persistent_state"])._snapshot_persistent_state(brain)
        rng = copy.deepcopy(brain.rng.bit_generator.state)
        result = _evaluate_checkpoint(self.connectome, self.wm, self.interface_config, 83, persistent, rng, 0)
        self.assertEqual(result["intact"]["state_memory"]["mean_post_reset_active_count"], result["intact"]["state_memory"]["mean_pre_go_active_count"])
        self.assertEqual(result["reset"]["state_memory"]["mean_post_reset_active_count"], 0.0)

    def test_memory_comparison_has_required_gaps(self):
        intact = {
            "accuracy": 0.6,
            "target_minus_best_competitor_margin_hz": 4.0,
            "mean_target_rank": 1.4,
            "cue_conditioned_go_output_separation_hz": 20.0,
            "state_memory": {"mean_pre_go_active_count": 100.0},
        }
        reset = {
            "accuracy": 0.2,
            "target_minus_best_competitor_margin_hz": -2.0,
            "mean_target_rank": 2.3,
            "cue_conditioned_go_output_separation_hz": 2.0,
            "state_memory": {"mean_pre_go_active_count": 0.0},
        }
        result = _memory_comparison(intact, reset)
        self.assertAlmostEqual(result["accuracy_gap_intact_minus_reset"], 0.4)
        self.assertEqual(result["margin_gap_intact_minus_reset_hz"], 6.0)
        self.assertEqual(result["go_separation_ratio_intact_over_reset"], 10.0)

    def _fake_row(self, cue, decision, rate=1.0):
        rates = {symbol: 0.0 for symbol in SYMBOLS}
        rates[decision] = rate
        return SimpleNamespace(
            cue=cue,
            decision=decision,
            go_output_rates_hz=rates,
            go_output_spikes=1,
            pre_go_active_count=1,
            pre_go_active_indices=(1,),
            pre_go_fingerprint=f"{cue}-{decision}",
            pre_go_membrane_norm=1.0,
            pre_go_conductance_norm=1.0,
            post_reset_active_count=0,
        )

    def test_collapse_diagnostic_uses_largest_prediction_fraction(self):
        rows = [self._fake_row(SYMBOLS[index % 4], "B") for index in range(40)]
        summary = _summarize_rows(rows)
        self.assertTrue(summary["output_collapse"])
        self.assertGreaterEqual(summary["largest_prediction_fraction"], 0.80)

    def test_noncollapsed_prediction_distribution_is_reported(self):
        rows = [self._fake_row(SYMBOLS[index % 4], SYMBOLS[index % 4]) for index in range(40)]
        summary = _summarize_rows(rows)
        self.assertFalse(summary["output_collapse"])
        self.assertEqual(summary["distinct_predicted_symbols"], 4)

    def test_fixed_plastic_budget_after_one_learned_episode(self):
        brain = self.make_brain()
        start = brain.plasticity.plastic_edge_count
        DelayedCueLearningSession(brain, self.wm, config=self.config).train_trial("A")
        self.assertEqual(brain.plasticity.plastic_edge_count, start)

    def test_adaptive_budget_is_disabled(self):
        self.assertFalse(self.config.adaptive_plastic_budget)
        self.assertEqual(self.config.episode_credit_limit, 16)

    def test_bounded_phase_credit_has_no_dense_state_field(self):
        brain = self.make_brain()
        session = DelayedCueLearningSession(brain, self.wm, config=self.config)
        trials, records = session.train(tuple("ABCDABCDABCDABCD"), capture_first_incorrect=16)
        self.assertLessEqual(len(records), 16)
        self.assertTrue(all("eligible_edges_by_end_go" in record for record in records))
        self.assertTrue(all("dense" not in record for record in records))

    def test_profiler_sections_are_disjoint_at_top_level(self):
        names = {
            "network_simulation_seconds",
            "reward_update_seconds",
            "post_reward_directional_seconds",
            "normalizer_seconds",
            "plastic_lifecycle_seconds",
        }
        source = inspect.getsource(DelayedCueLearningSession.train_trial)
        source += inspect.getsource(__import__("drosomath.whole_brain.brain_adapter", fromlist=["PlasticSparseFlyBrain"]).PlasticSparseFlyBrain.learn_from_reward)
        for name in names:
            self.assertIn(name, source)
        self.assertNotIn("normalization_seconds", source)

    def test_profiler_on_off_is_state_free(self):
        profiler = TimingProfiler()
        profiler.add("network_simulation_seconds", 1.0)
        self.assertTrue(profiler.report()["enabled"])
        disabled = TimingProfiler(enabled=False)
        disabled.add("network_simulation_seconds", 100.0)
        self.assertEqual(disabled.report()["seconds"], {})

    def test_route_cache_benchmark_is_counterbalanced(self):
        source = inspect.getsource(__import__("run_delayed_cue_learning_phase_f2b", fromlist=["_paired_route_cache_benchmark"])._paired_route_cache_benchmark)
        self.assertIn("sequence = (False, True, True, False)", source)

    def test_f2b_artifact_has_separate_conclusion_and_pass_schema(self):
        source = inspect.getsource(__import__("run_delayed_cue_learning_phase_f2b", fromlist=["run"]).run)
        self.assertIn('"conclusion"', source)
        self.assertIn('"pass"', source)

    def test_no_external_decoder_or_backpropagation(self):
        source = inspect.getsource(DelayedCueLearningSession)
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("backprop", source.lower())

    def test_learning_session_does_not_use_go_as_output_channel(self):
        session = DelayedCueLearningSession(self.make_brain(), self.wm, config=self.config)
        self.assertNotIn(GO_SYMBOL, session.output_context)


if __name__ == "__main__":
    unittest.main()
