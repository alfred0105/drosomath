import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from correct_transient_hebbian_binding_phase_f3e import ARTIFACT  # noqa: E402


class TransientHebbianStageATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(Path(ARTIFACT).read_text(encoding="utf-8"))

    def test_representation_gate_and_stage_b_guard(self):
        self.assertFalse(self.artifact["representation"]["summary"]["context_binding_improved"])
        self.assertFalse(self.artifact["training_stage_executed"])
        self.assertEqual(self.artifact["behavior"], "not_executed_due_to_representation_gate")

    def test_primary_gate_values(self):
        summary = self.artifact["representation"]["summary"]
        self.assertLess(summary["hebb_to_control_ratio"], 1.01)
        self.assertEqual(summary["broader_context_counts_by_seed"], [11, 8, 10])

    def test_guards_and_next_step(self):
        guards = self.artifact["guards"]
        self.assertFalse(guards["first_memory"]["first_memory_destroyed"])
        self.assertFalse(guards["second_domination"]["second_cue_domination"])
        self.assertFalse(guards["activity"]["activity_pathology"])
        self.assertFalse(guards["binding_saturation"]["binding_saturation"])
        self.assertEqual(self.artifact["conclusion"]["recommended_next_step"], "redesign_generic_binding_dynamics")

    def test_performance_pathology_is_reported(self):
        performance = self.artifact["performance"]
        self.assertTrue(performance["performance_pathology"])
        self.assertLess(performance["hebb_episodes_per_second"], 0.50 * performance["control_episodes_per_second"])


if __name__ == "__main__":
    unittest.main()
