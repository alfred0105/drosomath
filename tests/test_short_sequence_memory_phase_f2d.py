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
    ORDERED_PAIRS,
    PlasticMaleCNSBrain,
    SequenceLearningSession,
    SequenceMemoryConfig,
    SYMBOLS,
    SymbolInterface,
    SymbolInterfaceConfig,
    TwoCueSequenceSession,
    WorkingMemoryInterface,
    balanced_pair_schedule,
)
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler
from run_short_sequence_memory_phase_f2d import (
    _schedule_balance,
    _summarize,
)
from test_working_memory_phase_f2a import make_connectome


class CountingSequenceBrain(PlasticMaleCNSBrain):
    def __init__(self, *args, **kwargs):
        self.reset_calls = 0
        super().__init__(*args, **kwargs)

    def reset(self):
        self.reset_calls += 1
        return super().reset()


class ShortSequenceMemoryPhaseF2DTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.interface_config = SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7)
        cls.interface = SymbolInterface(cls.connectome, cls.interface_config)
        cls.wm = WorkingMemoryInterface(cls.connectome, cls.interface)

    def make_brain(self, seed=109, brain_type=PlasticMaleCNSBrain):
        return brain_type(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        )

    def test_exactly_16_ordered_pairs(self):
        self.assertEqual(len(ORDERED_PAIRS), 16)
        self.assertEqual(set(ORDERED_PAIRS), {(first, second) for first in SYMBOLS for second in SYMBOLS})

    def test_schedule_has_50_of_each_pair_and_200_each_item(self):
        schedule = balanced_pair_schedule(50, seed=109)
        self.assertEqual(len(schedule), 800)
        self.assertTrue(all(schedule.count(pair) == 50 for pair in ORDERED_PAIRS))
        self.assertTrue(all(sum(first == cue for first, _ in schedule) == 200 for cue in SYMBOLS))
        self.assertTrue(all(sum(second == cue for _, second in schedule) == 200 for cue in SYMBOLS))

    def test_schedule_is_shuffled_cyclewise(self):
        schedule = balanced_pair_schedule(2, seed=109)
        self.assertNotEqual(schedule[:16], ORDERED_PAIRS)
        self.assertEqual(set(schedule[:16]), set(ORDERED_PAIRS))
        self.assertEqual(set(schedule[16:]), set(ORDERED_PAIRS))

    def test_second_cue_conditional_targets_are_uniform(self):
        balance = _schedule_balance(balanced_pair_schedule(50, seed=109))
        self.assertTrue(balance["second_cue_target_independence"])
        self.assertEqual(balance["second_only_nominal_shortcut_accuracy"], 0.25)
        for second in SYMBOLS:
            self.assertEqual(set(balance["target_count_by_second"][second].values()), {50})

    def test_target_is_first_item(self):
        session = TwoCueSequenceSession(self.make_brain(), self.wm)
        result = session.run_trial("A", "D")
        self.assertEqual(result.target, "A")
        self.assertNotEqual(result.target, result.second)

    def test_intact_episode_resets_once(self):
        brain = CountingSequenceBrain(
            self.connectome, params=FlyBrainParams(dt_ms=0.2), seed=109,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=109),
        )
        TwoCueSequenceSession(brain, self.wm).run_trial("A", "B")
        self.assertEqual(brain.reset_calls, 1)

    def test_between_item_reset_occurs_after_first_and_before_second(self):
        brain = CountingSequenceBrain(
            self.connectome, params=FlyBrainParams(dt_ms=0.2), seed=109,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=109),
        )
        TwoCueSequenceSession(brain, self.wm).run_trial("A", "B", reset_between_items=True)
        self.assertEqual(brain.reset_calls, 2)

    def test_no_reset_between_first_second_go_in_intact_mode(self):
        brain = CountingSequenceBrain(
            self.connectome, params=FlyBrainParams(dt_ms=0.2), seed=109,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=109),
        )
        TwoCueSequenceSession(brain, self.wm).run_trial("A", "B", reset_between_items=False)
        self.assertEqual(brain.reset_calls, 1)

    def test_between_reset_preserves_persistent_state(self):
        brain = self.make_brain()
        before = {name: value.copy() for name, value in {
            "multiplier": brain.plasticity.multiplier,
            "stability": brain.plasticity.stability,
            "plastic_mask": brain.plasticity.plastic_mask,
        }.items()}
        TwoCueSequenceSession(brain, self.wm).run_trial("A", "B", reset_between_items=True)
        np.testing.assert_array_equal(brain.plasticity.multiplier, before["multiplier"])
        np.testing.assert_array_equal(brain.plasticity.stability, before["stability"])
        np.testing.assert_array_equal(brain.plasticity.plastic_mask, before["plastic_mask"])

    def test_go_is_the_only_decision_phase(self):
        source = inspect.getsource(TwoCueSequenceSession.run_trial)
        self.assertLess(source.index("rates, go_spikes"), source.index("decision ="))
        self.assertNotIn("decision =", source[:source.index("rates, go_spikes")])

    def test_learning_target_enters_after_go(self):
        source = inspect.getsource(SequenceLearningSession.train_trial)
        self.assertLess(source.index("result = self.episode.run_trial"), source.index("build_symbol_learning_signal"))
        self.assertIn("target=result.first", source)

    def test_eligibility_observer_spans_all_three_phases(self):
        brain = self.make_brain()
        observed = {}
        session = TwoCueSequenceSession(brain, self.wm)
        session.run_trial(
            "A", "C", track_eligibility=True, retain_tracking=True,
            phase_observer=lambda phase, _: observed.setdefault(phase, int(np.count_nonzero(brain.plasticity.eligibility))),
        )
        self.assertEqual(set(observed), {"first", "second", "go"})
        self.assertGreaterEqual(observed["second"], observed["first"])
        self.assertGreaterEqual(observed["go"], observed["second"])

    def test_learning_session_has_generic_signal_and_post_go_learning(self):
        source = inspect.getsource(SequenceLearningSession.train_trial)
        self.assertLess(source.index("build_symbol_learning_signal"), source.index("self.brain.learn_from_reward"))
        self.assertNotIn("pair=", source)
        self.assertNotIn("second=", source[source.index("build_symbol_learning_signal"):])

    def test_pair_metrics_and_same_different_groups(self):
        rows = []
        for first, second in ORDERED_PAIRS:
            rates = {symbol: 0.0 for symbol in SYMBOLS}
            rates[first] = 10.0
            rows.append(SimpleNamespace(first=first, second=second, decision=first, go_output_rates_hz=rates, go_output_spikes=1))
        result = _summarize(rows)
        self.assertEqual(len(result["pair_accuracy"]), 16)
        self.assertEqual(result["same_symbol_accuracy"], 1.0)
        self.assertEqual(result["different_symbol_accuracy"], 1.0)

    def test_reversal_diagnostic_is_present(self):
        rows = []
        for first, second in ORDERED_PAIRS:
            rates = {symbol: 0.0 for symbol in SYMBOLS}
            rates[first] = 10.0
            rows.append(SimpleNamespace(first=first, second=second, decision=first, go_output_rates_hz=rates, go_output_spikes=1))
        result = _summarize(rows)
        self.assertIn("AB_vs_BA", result["order_reversal"])
        self.assertIn("reversed_pair_target_sensitivity", result["order_reversal"])

    def test_collapse_diagnostic(self):
        rows = []
        for index in range(16):
            first, second = ORDERED_PAIRS[index]
            rates = {symbol: 0.0 for symbol in SYMBOLS}
            rates["B"] = 10.0
            rows.append(SimpleNamespace(first=first, second=second, decision="B", go_output_rates_hz=rates, go_output_spikes=1))
        self.assertTrue(_summarize(rows)["output_collapse"])

    def test_config_keeps_f2b_learning_constants(self):
        config = SequenceMemoryConfig()
        self.assertEqual((config.first_ms, config.second_ms, config.go_ms), (20.0, 20.0, 20.0))
        self.assertEqual(config.directional_learning_rate, 0.02)
        self.assertEqual(config.reward_learning_rate, 0.02)
        self.assertEqual(config.two_hop_credit_mode, "prospective_anatomical")
        self.assertFalse(config.adaptive_plastic_budget)

    def test_fixed_budget_after_sequence_learning_episode(self):
        brain = self.make_brain()
        start = brain.plasticity.plastic_edge_count
        SequenceLearningSession(brain, self.wm).train_trial("A", "C")
        self.assertEqual(brain.plasticity.plastic_edge_count, start)

    def test_phase_credit_is_bounded_and_does_not_store_raw_sets(self):
        brain = self.make_brain()
        session = SequenceLearningSession(brain, self.wm)
        record = session.train_trial("A", "D", capture_phase_credit=True)
        self.assertIn("phase_credit", record)
        if record["phase_credit"] is not None:
            self.assertNotIn("edges", record["phase_credit"])

    def test_profiler_off_is_state_free(self):
        brain = self.make_brain()
        profiler = TimingProfiler(enabled=False)
        brain.configure_neural_timing(profiler)
        TwoCueSequenceSession(brain, self.wm).run_trial("A", "B")
        self.assertEqual(profiler.report()["seconds"], {})

    def test_delay_ring_sections_are_split(self):
        source = inspect.getsource(PlasticMaleCNSBrain.step)
        self.assertIn("due_index_collection_seconds", source)
        self.assertIn("active_set_merge_seconds", source)

    def test_sequence_exact_replay_is_available(self):
        # Small-connectome exactness is exercised by running the same seed twice.
        left = self.make_brain(333)
        right = self.make_brain(333)
        TwoCueSequenceSession(left, self.wm).run_trial("A", "B")
        TwoCueSequenceSession(right, self.wm).run_trial("A", "B")
        np.testing.assert_array_equal(left.v, right.v)
        np.testing.assert_array_equal(left.g, right.g)
        self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
