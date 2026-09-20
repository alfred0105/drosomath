import unittest

import numpy as np

from drosomath.malecns.keyboard_learning import (
    build_credit_targeting_trial_summary,
    pair_next_same_key_occurrences,
    summarize_key_edge_sharing,
    summarize_next_same_key_pairs,
)


def window(index, rate, edges, hops=None):
    return {
        "window_index": index,
        "click_rate_hz": rate,
        "causal_edges": tuple(edges),
        "causal_hops": hops or {},
    }


class CreditTargetingSummaryTests(unittest.TestCase):
    def test_empty_sets_are_zero_and_do_not_raise(self):
        summary = build_credit_targeting_trial_summary(
            window_records=[],
            updated_edge_indices=np.empty(0, dtype=np.int32),
            early_window_limit=1,
        )
        self.assertEqual(summary["updated_edge_count"], 0)
        self.assertEqual(summary["fraction_updated_in_peak_window_causal_set"], 0.0)
        self.assertEqual(summary["peak_final_causal_jaccard"], 0.0)

    def test_peak_window_is_selected_by_click_rate(self):
        summary = build_credit_targeting_trial_summary(
            window_records=[window(1, 4.0, [1]), window(2, 12.0, [2]), window(3, 8.0, [3])],
            updated_edge_indices=[2],
            updated_edge_hops={2: 1},
            early_window_limit=1,
        )
        self.assertEqual(summary["peak_window_index"], 2)
        self.assertEqual(summary["fraction_updated_in_peak_window_causal_set"], 1.0)

    def test_early_union_and_final_window_are_distinct(self):
        summary = build_credit_targeting_trial_summary(
            window_records=[window(1, 4.0, [1, 2]), window(2, 5.0, [2, 3]), window(3, 15.0, [3, 4])],
            updated_edge_indices=[1, 3, 4],
            updated_edge_hops={1: 1, 3: 2, 4: 1},
            early_window_limit=2,
        )
        self.assertAlmostEqual(summary["fraction_updated_in_early_causal_set"], 2 / 3)
        self.assertAlmostEqual(summary["fraction_updated_in_final_window_causal_set"], 2 / 3)
        self.assertAlmostEqual(summary["updated_direct_1hop_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["updated_2hop_fraction"], 1 / 3)

    def test_peak_final_overlap_is_reported(self):
        summary = build_credit_targeting_trial_summary(
            window_records=[window(1, 4.0, [1, 2]), window(2, 10.0, [2, 3]), window(3, 5.0, [3, 4])],
            updated_edge_indices=[2],
            updated_edge_hops={2: 1},
            early_window_limit=1,
        )
        self.assertAlmostEqual(summary["peak_final_causal_jaccard"], 1 / 3)

    def test_summary_does_not_mutate_inputs(self):
        records = [window(1, 4.0, [1, 2]), window(2, 10.0, [2, 3])]
        before = [dict(item) for item in records]
        updated = np.asarray([2], dtype=np.int32)
        summary = build_credit_targeting_trial_summary(
            window_records=records,
            updated_edge_indices=updated,
            updated_edge_hops={2: 1},
            early_window_limit=1,
        )
        self.assertTrue(summary)
        self.assertEqual(records, before)
        np.testing.assert_array_equal(updated, np.asarray([2], dtype=np.int32))


class CreditTargetingAggregationTests(unittest.TestCase):
    def test_key_sharing_counts_pairwise_and_threshold_usage(self):
        result = summarize_key_edge_sharing({
            "A": {1, 2},
            "B": {2, 3},
            "C": {2},
            "D": {4},
            "E": {2},
        })
        self.assertEqual(result["unique_margin_updated_edge_count"], 4)
        self.assertAlmostEqual(result["fraction_updated_edges_used_by_at_least_2_keys"], 0.25)
        self.assertAlmostEqual(result["fraction_updated_edges_used_by_at_least_5_keys"], 0.0)
        self.assertGreater(result["maximum_pairwise_updated_edge_overlap"], 0.0)

    def test_empty_key_sharing_is_zero(self):
        result = summarize_key_edge_sharing({"A": set(), "B": set()})
        self.assertEqual(result["mean_pairwise_updated_edge_overlap"], 0.0)
        self.assertEqual(result["unique_margin_updated_edge_count"], 0)

    def test_next_same_key_pairing(self):
        records = [
            {"label": "A", "peak_click_rate_hz": 10.0, "correct": True, "margin_update": True, "below_margin_success": True},
            {"label": "B", "peak_click_rate_hz": 14.0, "correct": True, "margin_update": False, "below_margin_success": True},
            {"label": "A", "peak_click_rate_hz": 12.0, "correct": False, "margin_update": False, "below_margin_success": False},
        ]
        pairs = pair_next_same_key_occurrences(records, eligible_field="margin_update")
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["next_same_key_peak_delta"], 2.0)
        self.assertEqual(pairs[0]["next_same_key_success_change"], -1)
        summary = summarize_next_same_key_pairs(pairs)
        self.assertEqual(summary["pair_count"], 1)

    def test_baseline_and_margin_bookkeeping_are_separable(self):
        records = [
            {"label": "A", "peak_click_rate_hz": 10.0, "correct": True, "margin_update": False, "below_margin_success": True},
            {"label": "A", "peak_click_rate_hz": 11.0, "correct": True, "margin_update": True, "below_margin_success": True},
        ]
        baseline = pair_next_same_key_occurrences(records, eligible_field="below_margin_success")
        margin = pair_next_same_key_occurrences(records, eligible_field="margin_update")
        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(margin), 0)


if __name__ == "__main__":
    unittest.main()
