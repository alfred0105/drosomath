from __future__ import annotations

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
    CONTEXTUAL_GRAMMAR,
    ContextualPredictionConfig,
    ContextualPredictionLearningSession,
    ContextualPredictionSession,
    ORDERED_PAIRS,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterface,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    balanced_pair_schedule,
)
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler
from run_contextual_prediction_phase_f3a import _grammar_balance, _summarize
from test_working_memory_phase_f2a import make_connectome


class CountingBrain(PlasticMaleCNSBrain):
    def __init__(self, *args, **kwargs):
        self.reset_calls = 0
        super().__init__(*args, **kwargs)

    def reset(self):
        self.reset_calls += 1
        return super().reset()


class ContextualPredictionPhaseF3ATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.interface_config = SymbolInterfaceConfig(
            sensory_population_size=8, output_population_size=8, seed=7
        )
        cls.interface = SymbolInterface(cls.connectome, cls.interface_config)
        cls.wm = WorkingMemoryInterface(cls.connectome, cls.interface)

    def make_brain(self, seed=131, brain_type=PlasticMaleCNSBrain):
        return brain_type(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        )

    def test_fixed_contextual_grammar(self):
        expected = {
            ("A", "A"): "A", ("A", "B"): "B", ("A", "C"): "C", ("A", "D"): "D",
            ("B", "A"): "C", ("B", "B"): "D", ("B", "C"): "A", ("B", "D"): "B",
            ("C", "A"): "D", ("C", "B"): "C", ("C", "C"): "B", ("C", "D"): "A",
            ("D", "A"): "B", ("D", "B"): "A", ("D", "C"): "D", ("D", "D"): "C",
        }
        self.assertEqual(CONTEXTUAL_GRAMMAR, expected)
        self.assertEqual(set(CONTEXTUAL_GRAMMAR), set(ORDERED_PAIRS))

    def test_balanced_800_schedule_and_shortcut_controls(self):
        schedule = balanced_pair_schedule(50, seed=131)
        self.assertEqual(len(schedule), 800)
        self.assertTrue(all(schedule.count(pair) == 50 for pair in ORDERED_PAIRS))
        balance = _grammar_balance(schedule)
        self.assertTrue(balance["first_conditional_targets_uniform"])
        self.assertTrue(balance["second_conditional_targets_uniform"])
        self.assertTrue(balance["global_targets_uniform"])
        self.assertEqual(balance["first_only_nominal_accuracy"], 0.25)
        self.assertEqual(balance["second_only_nominal_accuracy"], 0.25)
        self.assertEqual(balance["global_majority_target_accuracy"], 0.25)

    def test_order_reversal_is_not_ignored(self):
        self.assertNotEqual(CONTEXTUAL_GRAMMAR[("A", "B")], CONTEXTUAL_GRAMMAR[("B", "A")])
        self.assertEqual(CONTEXTUAL_GRAMMAR[("A", "B")], "B")
        self.assertEqual(CONTEXTUAL_GRAMMAR[("B", "A")], "C")
        self.assertEqual(_grammar_balance(balanced_pair_schedule(50, seed=131))["order_sensitive_reversed_context_count"], 6)

    def test_target_lookup_is_after_go_and_not_a_stimulus(self):
        brain = self.make_brain()
        session = ContextualPredictionSession(brain, self.wm)
        calls = []
        original = self.wm.population_for_input

        def recorded(symbol):
            calls.append(symbol)
            return original(symbol)

        self.wm.population_for_input = recorded
        result = session.run_trial("A", "B")
        self.assertEqual(calls, ["A", "B", "GO"])
        self.assertEqual(result.target, "B")
        source = inspect.getsource(ContextualPredictionSession.run_trial)
        self.assertLess(source.index("super().run_trial"), source.index("target = CONTEXTUAL_GRAMMAR"))

    def test_intact_and_between_item_reset_counts(self):
        brain = CountingBrain(
            self.connectome, params=FlyBrainParams(dt_ms=0.2), seed=131,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=131),
        )
        ContextualPredictionSession(brain, self.wm).run_trial("A", "B")
        self.assertEqual(brain.reset_calls, 1)
        ContextualPredictionSession(brain, self.wm).run_trial("A", "B", reset_between_items=True)
        self.assertEqual(brain.reset_calls, 3)

    def test_config_preserves_fixed_f3a_constants(self):
        config = ContextualPredictionConfig()
        self.assertEqual((config.first_ms, config.second_ms, config.go_ms), (20.0, 20.0, 20.0))
        self.assertEqual(config.stimulus_rate_hz, 205.0)
        self.assertEqual(config.training_trials, 800)
        self.assertEqual(config.directional_learning_rate, 0.02)
        self.assertEqual(config.reward_learning_rate, 0.02)
        self.assertEqual(config.two_hop_credit_mode, "prospective_anatomical")
        self.assertFalse(config.adaptive_plastic_budget)

    def test_learning_uses_generic_signal_after_episode(self):
        source = inspect.getsource(ContextualPredictionLearningSession.train_trial)
        self.assertLess(source.index("result = self.episode.run_trial"), source.index("build_symbol_learning_signal"))
        self.assertIn("target=result.target", source)
        self.assertNotIn("CONTEXTUAL_GRAMMAR", source)
        self.assertNotIn("teacher", source.lower())

    def test_phase_credit_observes_first_second_and_go(self):
        brain = self.make_brain()
        observed = {}
        session = ContextualPredictionSession(brain, self.wm)
        session.run_trial(
            "A", "C", track_eligibility=True, retain_tracking=True,
            phase_observer=lambda phase, _: observed.setdefault(
                phase, int(np.count_nonzero(brain.plasticity.eligibility))
            ),
        )
        self.assertEqual(set(observed), {"first", "second", "go"})
        self.assertGreaterEqual(observed["second"], observed["first"])
        self.assertGreaterEqual(observed["go"], observed["second"])

    def test_context_metrics_use_grammar_target(self):
        rows = []
        for first, second in ORDERED_PAIRS:
            target = CONTEXTUAL_GRAMMAR[(first, second)]
            rates = {symbol: 0.0 for symbol in SYMBOLS}
            rates[target] = 10.0
            rows.append(SimpleNamespace(
                first=first, second=second, target=target, decision=target,
                go_output_rates_hz=rates, go_output_spikes=1,
            ))
        summary = _summarize(rows)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(len(summary["context_accuracy"]), 16)
        self.assertEqual(summary["same_symbol_context_accuracy"], 1.0)
        self.assertEqual(summary["different_symbol_context_accuracy"], 1.0)
        self.assertFalse(summary["output_collapse"])

    def test_context_output_collapse_is_reported(self):
        rows = []
        for first, second in ORDERED_PAIRS:
            rates = {symbol: 0.0 for symbol in SYMBOLS}
            rates["A"] = 10.0
            rows.append(SimpleNamespace(
                first=first, second=second, target=CONTEXTUAL_GRAMMAR[(first, second)],
                decision="A", go_output_rates_hz=rates, go_output_spikes=1,
            ))
        self.assertTrue(_summarize(rows)["output_collapse"])

    def test_fixed_budget_after_contextual_learning_episode(self):
        brain = self.make_brain()
        start = brain.plasticity.plastic_edge_count
        record = ContextualPredictionLearningSession(brain, self.wm).train_trial("A", "D")
        self.assertEqual(brain.plasticity.plastic_edge_count, start)
        self.assertIn("directional_update", record)

    def test_profiler_sections_remain_split(self):
        source = inspect.getsource(PlasticMaleCNSBrain.step)
        self.assertIn("due_index_collection_seconds", source)
        self.assertIn("active_set_merge_seconds", source)
        profiler = TimingProfiler(enabled=False)
        brain = self.make_brain()
        brain.configure_neural_timing(profiler)
        ContextualPredictionSession(brain, self.wm).run_trial("A", "B")
        self.assertEqual(profiler.report()["seconds"], {})

    def test_same_seed_replays_exactly(self):
        left = self.make_brain(333)
        right = self.make_brain(333)
        left_result = ContextualPredictionSession(left, self.wm).run_trial("C", "D")
        right_result = ContextualPredictionSession(right, self.wm).run_trial("C", "D")
        np.testing.assert_array_equal(left.v, right.v)
        np.testing.assert_array_equal(left.g, right.g)
        np.testing.assert_array_equal(left._fast_active, right._fast_active)
        self.assertEqual(left_result.decision, right_result.decision)
        self.assertEqual(left.rng.bit_generator.state, right.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
