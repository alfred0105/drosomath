import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from correct_presynaptic_std_binding_phase_f3d import (
    ARTIFACT,
    derive_guards,
    derive_prediction_criteria,
    derive_representation_criteria,
)


class PresynapticStdBindingCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(Path(ARTIFACT).read_text(encoding="utf-8"))

    def test_prediction_improved_requires_all_behavioral_conditions(self):
        result = derive_prediction_criteria(self.artifact)
        self.assertTrue(result["std_prediction_improved"])
        self.assertEqual(result["improved_seed_count"], 3)
        self.assertTrue(result["no_output_collapse"])

    def test_context_dependence_requires_margin_improvement(self):
        result = derive_prediction_criteria(self.artifact)
        self.assertFalse(result["std_context_dependence_supported"])
        self.assertEqual(result["intact_gt_reset_seed_count"], 3)

    def test_strong_prediction_is_not_claimed(self):
        result = derive_prediction_criteria(self.artifact)
        self.assertFalse(result["strong_prediction"])
        self.assertFalse(result["discrete_contextual_prediction_demonstrated"])

    def test_representation_criterion_is_preserved(self):
        result = derive_representation_criteria(self.artifact)
        self.assertAlmostEqual(result["std_to_control_ratio"], 0.9956104025, places=8)
        self.assertEqual(result["broader_context_counts_by_seed"], [9, 9, 8])
        self.assertFalse(result["context_binding_improved"])

    def test_second_domination_guard(self):
        result = derive_guards(self.artifact)["second_domination_guard"]
        self.assertAlmostEqual(result["std_same_first_jaccard"], 0.6763205886, places=8)
        self.assertAlmostEqual(result["std_same_second_jaccard"], 0.23683, delta=0.01)
        self.assertFalse(result["second_cue_domination"])

    def test_activity_guard(self):
        result = derive_guards(self.artifact)["activity_guard"]
        self.assertAlmostEqual(result["std_to_control_activity_ratio"], 0.858, delta=0.02)
        self.assertFalse(result["activity_pathology"])

    def test_first_memory_guard(self):
        guard = self.artifact["first_memory_guard"]
        self.assertAlmostEqual(guard["control_initial_accuracy"], 0.2447917, places=6)
        self.assertAlmostEqual(guard["std_initial_accuracy"], 0.2604167, places=6)
        self.assertFalse(guard["flagged"])

    def test_next_step_is_deterministic(self):
        result = derive_representation_criteria(self.artifact)
        next_step = (
            "test_transient_hebbian_synaptic_binding"
            if not result["context_binding_improved"]
            else "continue_context_binding_validation"
        )
        self.assertEqual(next_step, "test_transient_hebbian_synaptic_binding")


if __name__ == "__main__":
    unittest.main()
