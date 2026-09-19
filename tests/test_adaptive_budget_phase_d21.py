import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from tests import run_adaptive_budget_phase_d21_ab as ab_runner


def fake_report(config, trials):
    return {
        "accuracy": 0.5,
        "training_accuracy": 0.5,
        "macro_recent_accuracy": 0.5,
        "median_recent_accuracy": 0.5,
        "minimum_recent_accuracy": 0.0,
        "per_key": {"A": {"recent_accuracy": 0.0}},
        "resume": None,
        "completed_trials": trials,
        "final_plasticity": {"mean_stability": 0.1},
        "adaptive_budget": {
            "directional_generic_update_count": 0,
            "localized_positive_reward_updated_edge_count": 0,
            "legacy_rescue_events": 0,
            "protected_edges_retired": 0,
            "plastic_budget_start": 10,
            "plastic_budget_end": 10,
            "budget_delta": 0,
            "reallocation_events": 0,
            "total_promoted_edges": 0,
            "total_retired_edges": 0,
            "unique_promoted_edges": 0,
            "unique_retired_edges": 0,
            "promoted_edges_later_activated": 0,
            "promoted_edges_later_updated": 0,
            "promotion_to_learning_rate": 0.0,
            "allocation_churn_rate": 0.0,
            "candidates_reaching_minimum_observations": 0,
            "candidates_reaching_promotion_threshold": 0,
            "reallocation_attempts": 0,
            "successful_reallocations": 0,
            "failed_reallocations_no_safe_donor": 0,
            "promotion_observation_counts": [],
            "promotion_need_scores": [],
        },
    }


class PhaseD21RunnerTests(unittest.TestCase):
    def test_ab_runner_changes_only_adaptive_flag_and_forces_clean_runs(self):
        captured = []

        def fake_run(connectome, *, config, **paths):
            captured.append(config)
            return fake_report(config, 4)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ab_runner, "run_keyboard_training", side_effect=fake_run
        ), patch.object(ab_runner, "load_malecns_v1", return_value=object()):
            result = ab_runner.run(
                data_dir=Path(tmp),
                trials=4,
                seed=7,
                output=Path(tmp) / "ab.json",
            )
        self.assertTrue(result["protocol"]["same_config_except_adaptive_plastic_budget"])
        self.assertEqual(len(captured), 2)
        self.assertFalse(captured[0].adaptive_plastic_budget)
        self.assertTrue(captured[1].adaptive_plastic_budget)
        self.assertFalse(captured[0].resume)
        self.assertFalse(captured[1].resume)
        self.assertEqual(captured[0].trials, captured[1].trials)
        self.assertEqual(captured[0].seed, captured[1].seed)
        self.assertEqual(captured[0].learning_rate, captured[1].learning_rate)
        self.assertEqual(captured[0].plastic_fraction, captured[1].plastic_fraction)
        left = asdict(captured[0]); right = asdict(captured[1])
        left.pop("adaptive_plastic_budget"); right.pop("adaptive_plastic_budget")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
