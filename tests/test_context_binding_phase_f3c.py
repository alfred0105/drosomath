import inspect
import unittest


class ContextBindingPhaseF3CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        import run_context_binding_phase_f3c as module

        cls.module = module

    def _representation_run(self, control_jaccard=0.5, binding_jaccard=0.3, broad=16):
        per_context = {f"{left}{right}": {"broader_conjunction_count": broad} for left in "ABCD" for right in "ABCD"}
        return {
            "seed": 163,
            "representation_before": {
                "control": {"pairwise_context_similarity": {"pre_go": {"same_first_different_second": {"mean": control_jaccard}}}, "conjunctive_specific_pools": {"pre_go": {"per_context": per_context}}},
                "binding": {"pairwise_context_similarity": {"pre_go": {"same_first_different_second": {"mean": binding_jaccard}}}, "conjunctive_specific_pools": {"pre_go": {"per_context": per_context}}},
            },
        }

    @staticmethod
    def _behavior_row(accuracy, margin, collapse=False):
        return {"accuracy": accuracy, "target_minus_best_competitor_margin_hz": margin, "output_collapse": collapse}

    def _prediction_run(self, control=(0.2, 0.1), binding=(0.4, 0.3), reset=(0.2, 0.1), collapse=False):
        row = lambda value: {"intact": self._behavior_row(*value, collapse), "between_item_reset": self._behavior_row(*reset, collapse)}
        return {"behavior": {"control": {"400": row(control)}, "binding": {"400": row(binding)}}}

    def test_required_seed_set(self):
        self.assertEqual(self.module.SEEDS, (163, 167, 173))

    def test_training_episode_count_is_four_hundred(self):
        self.assertEqual(self.module.TRAINING_EPISODES, 400)

    def test_representation_ratio_is_reported(self):
        result = self.module._representation_criterion([self._representation_run() for _ in range(3)])
        self.assertAlmostEqual(result["binding_to_control_ratio"], 0.6, places=6)

    def test_representation_requires_two_seeds_for_broad_pools(self):
        good = [self._representation_run() for _ in range(2)]
        bad = self._representation_run(broad=0)
        result = self.module._representation_criterion(good + [bad])
        self.assertTrue(result["criterion_broader_pools"])

    def test_representation_fails_when_separation_is_not_twenty_percent(self):
        result = self.module._representation_criterion([self._representation_run(binding_jaccard=0.45) for _ in range(3)])
        self.assertFalse(result["criterion_same_first"])
        self.assertFalse(result["supported"])

    def test_prediction_improvement_requires_two_seeds(self):
        result = self.module._prediction_criterion([self._prediction_run()])
        self.assertFalse(result["prediction_improved"])

    def test_prediction_improvement_passes_with_two_seeds(self):
        result = self.module._prediction_criterion([self._prediction_run(), self._prediction_run(), self._prediction_run()])
        self.assertTrue(result["prediction_improved"])

    def test_context_dependence_uses_reset_arm(self):
        result = self.module._prediction_criterion([self._prediction_run(reset=(0.1, 0.0)) for _ in range(3)])
        self.assertTrue(result["context_dependent"])
        self.assertEqual(result["context_dependence_seed_count"], 3)

    def test_strong_prediction_requires_half_accuracy(self):
        weak = self.module._prediction_criterion([self._prediction_run(binding=(0.4, 0.3), reset=(0.1, 0.0)) for _ in range(3)])
        strong = self.module._prediction_criterion([self._prediction_run(binding=(0.6, 0.4), reset=(0.1, 0.0)) for _ in range(3)])
        self.assertFalse(weak["strong_prediction"])
        self.assertTrue(strong["strong_prediction"])

    def test_output_collapse_blocks_prediction_claim(self):
        result = self.module._prediction_criterion([self._prediction_run(collapse=True) for _ in range(3)])
        self.assertFalse(result["no_output_collapse"])
        self.assertFalse(result["prediction_improved"])

    def test_f3c_core_is_task_independent(self):
        from drosomath.whole_brain import slow_adaptation

        self.assertNotIn("keyboard", inspect.getsource(slow_adaptation).lower())

    def test_protocol_does_not_enable_adaptive_budget(self):
        self.assertFalse(self.module._interface_config().plastic_fraction == 0.0)
        self.assertIn("adaptive_plastic_budget", inspect.getsource(self.module.run))

    def test_protocol_uses_fixed_adaptation_values(self):
        from drosomath.whole_brain import SlowAdaptationConfig

        config = SlowAdaptationConfig(enabled=True)
        self.assertEqual(config.tau_ms, 30.0)
        self.assertEqual(config.spike_increment_mv, 0.75)
        self.assertEqual(config.max_adaptation_mv, 3.0)

    def test_persistent_snapshot_helpers_are_used(self):
        source = inspect.getsource(self.module)
        self.assertIn("_snapshot_persistent_state", source)
        self.assertIn("_restore_persistent_state", source)

    def test_no_external_decoder_is_created(self):
        source = inspect.getsource(self.module)
        self.assertNotIn("backprop", source.lower())
        self.assertNotIn("classifier", source.lower())

    def test_performance_reports_extra_bytes(self):
        source = inspect.getsource(self.module._benchmark)
        self.assertIn("extra_adaptation_bytes", source)

    def test_performance_reports_control_and_binding(self):
        source = inspect.getsource(self.module._benchmark)
        self.assertIn("control_episodes_per_second", source)
        self.assertIn("binding_episodes_per_second", source)

    def test_adaptation_telemetry_reports_both_phases(self):
        source = inspect.getsource(self.module._adaptation_summary)
        self.assertIn('"first"', source)
        self.assertIn('"pre_go"', source)

    def test_binding_criterion_requires_broader_pool_coverage(self):
        result = self.module._representation_criterion([self._representation_run(broad=0) for _ in range(3)])
        self.assertFalse(result["criterion_broader_pools"])

    def test_prediction_margin_is_part_of_improvement(self):
        result = self.module._prediction_criterion([self._prediction_run(binding=(0.4, 0.05), control=(0.3, 0.2)) for _ in range(3)])
        self.assertFalse(result["prediction_improved"])

    def test_schedule_is_matched_by_protocol_code(self):
        source = inspect.getsource(self.module.run)
        self.assertIn("control_schedule == binding_schedule", source)

    def test_safety_requires_disabled_exact_equivalence(self):
        source = inspect.getsource(self.module.run)
        self.assertIn('"disabled_exact_equivalence"', source)


if __name__ == "__main__":
    unittest.main()
