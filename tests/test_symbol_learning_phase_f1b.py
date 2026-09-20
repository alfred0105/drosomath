from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    CHANNELS,
    NO_DECISION,
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolLearningConfig,
    SymbolLearningSession,
    balanced_symbol_schedule,
    build_symbol_learning_signal,
    symbol_output_context,
)
from drosomath.malecns.loader import MaleCNSConnectome  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


def make_connectome(n: int = 320) -> MaleCNSConnectome:
    posts = np.column_stack(
        (np.arange(n, dtype=np.int32), (np.arange(n, dtype=np.int32) + 1) % n)
    ).reshape(-1)
    signed = np.where(np.arange(2 * n) % 3 == 0, -2.0, 3.0).astype(np.float32)
    return MaleCNSConnectome(
        body_ids=np.arange(10_000, 10_000 + n, dtype=np.int64),
        indptr=np.arange(0, 2 * n + 1, 2, dtype=np.int64),
        post_indices=posts,
        synapse_counts=np.abs(signed).astype(np.int32),
        signed_synapse_counts=signed,
        outgoing_strength=np.full(n, 5.0, dtype=np.float32),
        presynaptic_sign=np.ones(n, dtype=np.int8),
        consensus_nt=np.asarray(["Glutamate"] * n, dtype=object),
        min_connection_synapses=5,
    )


class SymbolLearningPhaseF1BTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.config = SymbolInterfaceConfig(
            sensory_population_size=8,
            output_population_size=8,
            seed=7,
            output_selection="dynamic_generic",
        )
        random_interface = SymbolInterface(
            cls.connectome,
            SymbolInterfaceConfig(
                sensory_population_size=8,
                output_population_size=8,
                seed=7,
            ),
        )
        sensory = np.concatenate(tuple(random_interface.sensory_populations.values()))
        candidates = [
            index for index in range(cls.connectome.neuron_count)
            if index not in set(int(value) for value in sensory)
        ]
        cls.interface = SymbolInterface.with_output_populations(
            cls.connectome,
            cls.config,
            {
                symbol: np.asarray(candidates[i * 8:(i + 1) * 8], dtype=np.int32)
                for i, symbol in enumerate(SYMBOLS)
            },
            source="dynamic_generic",
        )

    def _brain(self, seed=41):
        return PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.2, seed=seed),
        )

    def test_channels_are_exactly_generic_symbol_channels(self):
        self.assertEqual(CHANNELS, ("symbol/A", "symbol/B", "symbol/C", "symbol/D"))
        self.assertEqual(tuple(symbol_output_context(self.interface)), CHANNELS)

    def test_target_label_does_not_enter_controller_module(self):
        source = inspect.getsource(__import__(
            "drosomath.whole_brain.directional_modulation",
            fromlist=["PlasticityController"],
        ))
        self.assertNotIn("keyboard", source.lower())
        self.assertNotIn("click_teacher", source)

    def test_incorrect_unique_winner(self):
        signal = build_symbol_learning_signal(
            target="A", decision="B", output_rates_hz={"A": 0, "B": 5, "C": 0, "D": 0}
        )
        self.assertEqual(signal.reward, 0.0)
        self.assertFalse(signal.success)
        self.assertEqual(signal.directional_error, {"symbol/A": 1.0, "symbol/B": -1.0})

    def test_silent_has_only_target_direction(self):
        signal = build_symbol_learning_signal(
            target="C", decision=NO_DECISION, output_rates_hz={symbol: 0 for symbol in SYMBOLS}
        )
        self.assertEqual(signal.directional_error, {"symbol/C": 1.0})

    def test_tied_no_decision_splits_negative_correction(self):
        signal = build_symbol_learning_signal(
            target="A", decision=NO_DECISION, output_rates_hz={"A": 2, "B": 4, "C": 4, "D": 0}
        )
        self.assertEqual(signal.directional_error["symbol/A"], 1.0)
        self.assertEqual(signal.directional_error["symbol/B"], -0.5)
        self.assertEqual(signal.directional_error["symbol/C"], -0.5)
        self.assertNotIn("symbol/D", signal.directional_error)

    def test_tied_target_does_not_receive_negative_correction(self):
        signal = build_symbol_learning_signal(
            target="A", decision=NO_DECISION, output_rates_hz={"A": 4, "B": 4, "C": 0, "D": 0}
        )
        self.assertEqual(signal.directional_error, {"symbol/A": 1.0, "symbol/B": -1.0})

    def test_correct_has_reward_and_target_reinforcement_only(self):
        signal = build_symbol_learning_signal(
            target="D", decision="D", output_rates_hz={"A": 0, "B": 0, "C": 0, "D": 5}
        )
        self.assertEqual(signal.directional_error, {})
        self.assertEqual(signal.reward, 1.0)
        self.assertTrue(signal.success)
        self.assertEqual(signal.reinforcement, {"symbol/D": 1.0})

    def test_directions_are_bounded(self):
        for rates in (
            {"A": 0, "B": 10, "C": 10, "D": 10},
            {"A": 10, "B": 0, "C": 0, "D": 0},
        ):
            signal = build_symbol_learning_signal(target="A", decision=NO_DECISION, output_rates_hz=rates)
            self.assertTrue(all(-1.0 <= value <= 1.0 for value in signal.directional_error.values()))
            self.assertLessEqual(sum(abs(value) for value in signal.directional_error.values()), 2.0)

    def test_surface_is_frozen_and_disjoint(self):
        before = {key: value.copy() for key, value in self.interface.output_populations.items()}
        sensory = np.concatenate(tuple(self.interface.sensory_populations.values()))
        output = np.concatenate(tuple(self.interface.output_populations.values()))
        self.assertEqual(len(np.intersect1d(sensory, output)), 0)
        session = SymbolLearningSession(
            self._brain(), self.interface,
            config=SymbolLearningConfig(duration_ms=1.0, evaluation_repetitions=1, training_trials=4),
        )
        session.evaluate(seed=41)
        for symbol in SYMBOLS:
            np.testing.assert_array_equal(before[symbol], self.interface.output_populations[symbol])

    def test_balanced_schedule_is_deterministic_and_exact(self):
        left = balanced_symbol_schedule(cycles=100, seed=41)
        right = balanced_symbol_schedule(cycles=100, seed=41)
        self.assertEqual(left, right)
        self.assertEqual({symbol: left.count(symbol) for symbol in SYMBOLS}, {symbol: 100 for symbol in SYMBOLS})

    def test_independent_seed_brains_do_not_share_state(self):
        left = self._brain(41)
        right = self._brain(43)
        self.assertIsNot(left.plasticity.multiplier, right.plasticity.multiplier)
        before = right.plasticity.multiplier.copy()
        left.plasticity.multiplier[0] += 0.25
        np.testing.assert_array_equal(right.plasticity.multiplier, before)

    def test_evaluation_does_not_change_learning_state_or_budget(self):
        brain = self._brain()
        session = SymbolLearningSession(
            brain, self.interface,
            config=SymbolLearningConfig(duration_ms=1.0, evaluation_repetitions=1, training_trials=4),
        )
        before = {
            name: getattr(brain.plasticity, name).copy()
            for name in ("multiplier", "stability", "usage_ema", "eligibility", "plastic_mask")
        }
        budget = brain.plasticity.plastic_edge_count
        session.evaluate(seed=41)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(brain.plasticity, name), values)
        self.assertEqual(brain.plasticity.plastic_edge_count, budget)

    def test_baseline_twin_does_not_perturb_training_brain(self):
        training_brain = self._brain(41)
        training_session = SymbolLearningSession(
            training_brain, self.interface,
            config=SymbolLearningConfig(duration_ms=1.0, evaluation_repetitions=1, training_trials=4),
        )
        rng_before = copy.deepcopy(training_brain.rng.bit_generator.state)
        state_before = training_brain.plasticity.multiplier.copy()
        baseline_brain = self._brain(41)
        SymbolLearningSession(
            baseline_brain, self.interface,
            config=SymbolLearningConfig(duration_ms=1.0, evaluation_repetitions=1, training_trials=4),
        ).evaluate(seed=41)
        self.assertEqual(training_brain.rng.bit_generator.state, rng_before)
        np.testing.assert_array_equal(training_brain.plasticity.multiplier, state_before)

    def test_training_changes_only_existing_plastic_routes(self):
        brain = self._brain()
        session = SymbolLearningSession(
            brain, self.interface,
            config=SymbolLearningConfig(duration_ms=1.0, evaluation_repetitions=1, training_trials=4),
        )
        before = brain.plasticity.multiplier.copy()
        mask = brain.plasticity.plastic_mask.copy()
        session.train_trial("A")
        changed = np.flatnonzero(before != brain.plasticity.multiplier)
        self.assertTrue(np.all(mask[changed]))
        self.assertEqual(brain.plasticity.plastic_edge_count, int(mask.sum()))

    def test_no_external_decoder_in_generic_learning_module(self):
        source = inspect.getsource(__import__(
            "drosomath.malecns.symbol_learning", fromlist=["x"]
        ))
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("softmax", source.lower())
        self.assertNotIn("keyboard_learning", source)

    def test_collapse_diagnostic(self):
        from drosomath.malecns.symbol_learning import SymbolTrialResult, summarize_results
        rows = [SymbolTrialResult("A", "A", {"A": 1, "B": 0, "C": 0, "D": 0}, 1, {}, 1, True, {})] * 8
        rows += [SymbolTrialResult("B", "A", {"A": 1, "B": 0, "C": 0, "D": 0}, 1, {}, 0, False, {})] * 2
        summary = summarize_results(rows)
        self.assertTrue(summary["collapsed"])
        self.assertEqual(summary["distinct_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
