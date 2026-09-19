import ast
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.keyboard_learning import (
    KeyboardNeuralSession,
    KeyboardTrainingConfig,
    calculate_success_margin_deficit,
    summarize_recent_outcome_transitions,
)
from drosomath.malecns.checkpoint import save_learning_checkpoint, restore_learning_checkpoint
from drosomath.whole_brain.directional_modulation import PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from tests.test_directional_modulation import make_brain


class SuccessMarginSignalTests(unittest.TestCase):
    def test_disabled_success_reproduces_old_signal(self):
        session = SimpleNamespace(config=SimpleNamespace(click_only=True))
        signal = KeyboardNeuralSession._learning_signal_for_trial(
            session,
            reward=1.0,
            correct=True,
            current_deficit=0.3,
            click_gain=1.0,
        )
        self.assertEqual(signal.nonzero_directions(), {})
        self.assertEqual(signal.positive_reinforcements(), {"motor/click": 1.0})

    def test_margin_deficit_is_positive_for_marginal_success(self):
        deficit = calculate_success_margin_deficit(
            peak_click_rate_hz=9.5, threshold_hz=9.0, target_hz=15.0
        )
        self.assertAlmostEqual(deficit, 5.5 / 6.0)

    def test_margin_deficit_is_zero_at_or_above_target(self):
        self.assertEqual(
            calculate_success_margin_deficit(
                peak_click_rate_hz=15.0, threshold_hz=9.0, target_hz=15.0
            ),
            0.0,
        )
        self.assertEqual(
            calculate_success_margin_deficit(
                peak_click_rate_hz=18.0, threshold_hz=9.0, target_hz=15.0
            ),
            0.0,
        )

    def test_margin_signal_is_generic_and_failure_is_unchanged(self):
        session = SimpleNamespace(config=SimpleNamespace(click_only=True))
        success = KeyboardNeuralSession._learning_signal_for_trial(
            session,
            reward=1.0,
            correct=True,
            current_deficit=0.0,
            click_gain=1.0,
            success_margin_directional_error=0.2,
        )
        failure = KeyboardNeuralSession._learning_signal_for_trial(
            session,
            reward=0.0,
            correct=False,
            current_deficit=0.3,
            click_gain=1.2,
            success_margin_directional_error=0.2,
        )
        self.assertEqual(success.nonzero_directions(), {"motor/click": 0.2})
        self.assertEqual(success.positive_reinforcements(), {"motor/click": 1.0})
        self.assertEqual(failure.nonzero_directions(), {"motor/click": 0.36})
        self.assertEqual(failure.positive_reinforcements(), {})

    def test_config_default_is_backward_compatible(self):
        config = KeyboardTrainingConfig()
        self.assertFalse(config.success_margin_directional_learning)
        self.assertGreater(config.success_margin_directional_scale, 0.0)


class GenericDirectionalInteractionTests(unittest.TestCase):
    context = {"motor/click": np.asarray([2], dtype=np.int32)}

    def test_margin_update_uses_generic_controller_without_labels(self):
        brain = make_brain([(0, 2, 1)])
        before = float(brain.plasticity.multiplier[0])
        update = PlasticityController().apply_learning_signal(
            brain,
            LearningSignal(
                reward=1.0,
                directional_error={"motor/click": 0.2},
                reinforcement={"motor/click": 1.0},
                success=True,
            ),
            self.context,
        )
        self.assertEqual(update.edge_updates, 1)
        self.assertEqual(update.reinforced_channels, ("motor/click",))
        self.assertGreater(float(brain.plasticity.multiplier[0]), before)

        root = pathlib.Path(__file__).parents[1] / "src" / "drosomath"
        controller_path = root / "whole_brain" / "directional_modulation.py"
        tree = ast.parse(controller_path.read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("keyboard" in module for module in imports))

    def test_margin_update_and_consolidation_apply_once_and_respect_bound(self):
        brain = make_brain([(0, 2, 1)])
        brain.plasticity.multiplier[:] = brain.plasticity.config.max_multiplier - 0.01
        update = PlasticityController().apply_learning_signal(
            brain,
            LearningSignal(
                reward=1.0,
                directional_error={"motor/click": 10.0},
                reinforcement={"motor/click": 1.0},
                success=True,
            ),
            self.context,
        )
        self.assertEqual(update.edge_updates, 1)
        self.assertEqual(update.consolidated_edges, 1)
        self.assertLessEqual(
            float(brain.plasticity.multiplier[0]),
            brain.plasticity.config.max_multiplier,
        )
        self.assertGreater(float(brain.plasticity.stability[0]), 0.0)

    def test_inhibitory_path_uses_existing_polarity(self):
        brain = make_brain([(0, 2, -1)])
        before = float(brain.plasticity.multiplier[0])
        PlasticityController().apply_learning_signal(
            brain,
            LearningSignal(1.0, {"motor/click": 0.2}, reinforcement={"motor/click": 1.0}),
            self.context,
        )
        self.assertLess(float(brain.plasticity.multiplier[0]), before)

    def test_high_stability_protection_remains_active(self):
        brain = make_brain([(0, 2, 1)])
        brain.plasticity.stability[:] = 1.0
        before = float(brain.plasticity.multiplier[0])
        PlasticityController().apply_learning_signal(
            brain,
            LearningSignal(1.0, {"motor/click": 0.2}, reinforcement={"motor/click": 1.0}),
            self.context,
        )
        self.assertGreater(float(brain.plasticity.multiplier[0]), before)
        self.assertLess(
            float(brain.plasticity.multiplier[0]) - before,
            0.2 * PlasticityController().config.learning_rate,
        )

    def test_transition_summary_uses_recent_binary_history(self):
        result = summarize_recent_outcome_transitions({"A": [True, False, True, True]})
        self.assertEqual(result["per_key"]["A"]["transition_count"], 2)
        self.assertAlmostEqual(result["per_key"]["A"]["transition_rate"], 2 / 3)


class SuccessMarginCheckpointTests(unittest.TestCase):
    def test_checkpoint_reports_margin_flag(self):
        brain = make_brain([(0, 2, 1)])
        brain.connectome.edge_count = 1
        config = KeyboardTrainingConfig(
            success_margin_directional_learning=True,
            checkpoint_every=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "margin.npz"
            save_learning_checkpoint(path, brain=brain, config=config, completed_trials=1)
            with np.load(path, allow_pickle=False) as data:
                self.assertTrue(bool(data["config__success_margin_directional_learning"][0]))

            restored = restore_learning_checkpoint(path, brain=brain)
            self.assertFalse(restored["need_state_restored"])


if __name__ == "__main__":
    unittest.main()
