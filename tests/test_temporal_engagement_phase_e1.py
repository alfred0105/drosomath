import unittest

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.keyboard_learning import build_temporal_window_record
from drosomath.whole_brain.directional_modulation import PlasticityController
from tests.run_temporal_engagement_phase_e1 import _group, _matched_success_failure, _summarize
from tests.test_route_health import direct_brain


def temporal_row(label, correct, *, first_causal=2, first_click=3, early_edges=4.0, windows=4.0, eligibility=2.0):
    fields = {
        "mean_windows_per_trial": windows,
        "first_click_causal_activity_window": first_causal,
        "first_click_output_spike_window": first_click,
        "first_any_network_activity_window": 1,
        "early_active_click_causal_edges": early_edges,
        "early_active_plastic_click_causal_edges": 3.0,
        "early_active_frozen_click_causal_edges": 1.0,
        "early_net_click_route_influence": 2.0,
        "mean_net_click_route_influence_per_window": 1.5,
        "eligibility_per_window": eligibility,
        "early_eligibility_per_window": 1.0,
        "raw_total_click_route_eligibility": eligibility * windows,
        "eligibility_accumulation_rate": 0.5,
        "mean_adjacent_window_active_neuron_jaccard": 0.5,
        "mean_adjacent_window_click_causal_jaccard": 0.25,
        "activity_persistence_windows": 2.0,
        "mean_positive_effect_magnitude_per_window": 3.0,
        "mean_negative_effect_magnitude_per_window": 1.0,
        "unique_click_output_neurons_recruited": 5.0,
        "total_click_output_spikes": 8.0,
        "peak_click_output_rate_hz": 10.0,
        "time_to_peak_click_rate_window": 3,
        "mean_click_output_synchrony_proxy": 0.5,
        "peak_simultaneous_click_output_recruitment": 4.0,
        "first_window_click_rate_above_25pct_threshold": 1,
        "first_window_click_rate_above_50pct_threshold": 2,
        "first_window_click_rate_above_75pct_threshold": 3,
        "first_window_click_rate_above_threshold": 4,
    }
    return {"label": label, "correct": correct, "temporal_engagement": fields}


class TemporalEngagementTests(unittest.TestCase):
    def test_window_record_counts_spikes_unique_clicks_and_rates(self):
        record = build_temporal_window_record(
            window_index=2,
            total_spikes=7,
            fired_neurons={1, 3, 8},
            recent_presynaptic_count=5,
            click_output_spikes=3,
            click_output_neurons={8, 9},
            click_rate_hz=12.5,
            motor_channel_rates=np.asarray([1.0, 2.0]),
            causal_summary={"active_click_causal_edges": 4},
            active_neuron_jaccard=0.5,
            click_causal_jaccard=0.25,
        )
        self.assertEqual(record["total_spikes"], 7)
        self.assertEqual(record["unique_fired_neurons"], 3)
        self.assertEqual(record["click_output_spikes"], 3)
        self.assertEqual(record["active_click_output_neurons"], 2)
        self.assertEqual(record["active_click_causal_edges"], 4)

    def test_causal_counts_and_path_polarity_are_read_only(self):
        brain = direct_brain(signs=(1.0, -1.0, 1.0, 1.0))
        controller = PlasticityController()
        before = [value.copy() for value in (brain.plasticity.multiplier, brain.plasticity.stability, brain.plasticity.eligibility)]
        result = controller.diagnose_activity_window(
            brain, np.asarray([0], dtype=np.int32), {"motor/click": [2]}
        )["motor/click"]
        self.assertEqual(result["active_click_causal_edges"], 3)
        self.assertEqual(result["positive_effect_route_count"], 2)
        self.assertEqual(result["negative_effect_route_count"], 1)
        self.assertGreater(result["positive_effect_magnitude"], 0.0)
        self.assertGreater(result["negative_effect_magnitude"], 0.0)
        for got, expected in zip((brain.plasticity.multiplier, brain.plasticity.stability, brain.plasticity.eligibility), before):
            np.testing.assert_array_equal(got, expected)

    def test_eligibility_trajectory_is_sampled_without_mutation(self):
        brain = direct_brain()
        controller = PlasticityController()
        brain.plasticity.eligibility[:] = 2.0
        before = brain.plasticity.eligibility.copy()
        first = controller.diagnose_activity_window(brain, [0], {"motor/click": [2]})["motor/click"]
        second = controller.diagnose_activity_window(brain, [0], {"motor/click": [2]})["motor/click"]
        np.testing.assert_array_equal(brain.plasticity.eligibility, before)
        self.assertEqual(first["total_click_route_eligibility"], second["total_click_route_eligibility"])

    def test_first_timing_thresholds_and_persistence_are_aggregated(self):
        row = temporal_row("A", True, first_causal=2, first_click=4)
        summary = _summarize([row])
        self.assertEqual(summary["mean_first_causal_window"]["mean"], 2.0)
        self.assertEqual(summary["mean_first_click_output_window"]["mean"], 4.0)
        self.assertEqual(summary["mean_first_window_click_rate_above_75pct_threshold"]["mean"], 3.0)
        self.assertEqual(summary["mean_activity_persistence"], 0.5)

    def test_early_metrics_are_independent_of_late_trial_length(self):
        short = temporal_row("A", True, windows=4.0, early_edges=7.0)
        long = temporal_row("A", True, windows=20.0, early_edges=7.0)
        summary = _summarize([short, long])
        self.assertEqual(summary["mean_early_causal_edges"], 7.0)
        self.assertEqual(summary["mean_windows_per_trial"], 12.0)

    def test_same_key_success_failure_aggregation(self):
        grouped = _matched_success_failure({"A": [temporal_row("A", True), temporal_row("A", False)]})
        self.assertEqual(grouped["eligible_key_count"], 1)
        self.assertEqual(grouped["success"]["trial_count"], 1)
        self.assertEqual(grouped["failure"]["trial_count"], 1)

    def test_group_aggregation_uses_all_trial_temporal_samples(self):
        rows_by_key = {"A": [temporal_row("A", True)], "B": [temporal_row("B", False)]}
        group = _group(["A", "B"], rows_by_key)
        self.assertEqual(group["trial_count"], 2)
        self.assertEqual(group["mean_early_causal_edges"], 4.0)


if __name__ == "__main__":
    unittest.main()
