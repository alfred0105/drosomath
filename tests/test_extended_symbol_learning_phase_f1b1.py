from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.malecns.symbol_learning import SymbolTrialResult  # noqa: E402
from drosomath.malecns.symbol_learning_extended import (  # noqa: E402
    EXTENDED_CHECKPOINTS,
    boundary_crossings,
    channel_credit_telemetry,
    classify_margin_interval,
    extended_checkpoint_metrics,
    extended_symbol_schedule,
    target_rank,
)
from drosomath.malecns.symbol_interface import NO_DECISION, SYMBOLS  # noqa: E402


def row(target: str, rates: dict[str, float], decision: str | None = None, *, directional=None):
    decision = decision or max(rates, key=rates.get) if any(rates.values()) else NO_DECISION
    return SymbolTrialResult(
        target=target,
        decision=decision,
        output_rates_hz=rates,
        total_output_spikes=int(sum(rates.values())),
        directional_error=directional or {},
        reward=1.0 if decision == target else 0.0,
        success=decision == target,
        reinforcement={f"symbol/{target}": 1.0} if decision == target else {},
    )


class ExtendedSymbolLearningPhaseF1B1Test(unittest.TestCase):
    def test_exact_1600_balanced_schedule_and_400_each(self):
        schedule = extended_symbol_schedule(seed=41)
        self.assertEqual(len(schedule), 1600)
        self.assertEqual({symbol: schedule.count(symbol) for symbol in SYMBOLS}, {symbol: 400 for symbol in SYMBOLS})

    def test_checkpoints_are_exact(self):
        self.assertEqual(EXTENDED_CHECKPOINTS, (0, 400, 800, 1200, 1600))

    def test_target_rank_unique_and_tie(self):
        self.assertEqual(target_rank({"A": 5, "B": 2, "C": 1, "D": 0}, "A"), 1.0)
        self.assertEqual(target_rank({"A": 3, "B": 3, "C": 1, "D": 0}, "A"), 1.5)
        self.assertEqual(target_rank({"A": 1, "B": 3, "C": 3, "D": 0}, "A"), 3.0)

    def test_silent_rank_is_null(self):
        self.assertIsNone(target_rank({symbol: 0 for symbol in SYMBOLS}, "A"))

    def test_positive_margin_fraction(self):
        results = [
            row("A", {"A": 5, "B": 1, "C": 0, "D": 0}),
            row("A", {"A": 1, "B": 5, "C": 0, "D": 0}, decision="B"),
        ]
        metrics = extended_checkpoint_metrics(results)
        self.assertEqual(metrics["per_symbol"]["A"]["fraction_positive_margin"], 0.5)

    def test_boundary_crossing_detection(self):
        checkpoints = {}
        for checkpoint, margin, accuracy in ((0, -2, 0.0), (400, -1, 0.25), (800, 1, 0.5), (1200, 2, 0.5), (1600, 3, 0.75)):
            metrics = extended_checkpoint_metrics([
                row("A", {"A": margin + 2, "B": 2, "C": 0, "D": 0}, decision="A" if margin > 0 else "B")
            ])
            metrics["per_symbol"]["A"]["target_minus_best_competitor_margin"] = margin
            metrics["per_symbol"]["A"]["accuracy"] = accuracy
            metrics["target_minus_best_competitor_margin"] = margin
            checkpoints[str(checkpoint)] = metrics
        for symbol in ("B", "C", "D"):
            checkpoints[str(0)]["per_symbol"][symbol]["target_minus_best_competitor_margin"] = -1
            checkpoints[str(0)]["per_symbol"][symbol]["accuracy"] = 0
            for checkpoint in (400, 800, 1200, 1600):
                checkpoints[str(checkpoint)]["per_symbol"][symbol]["target_minus_best_competitor_margin"] = -1
                checkpoints[str(checkpoint)]["per_symbol"][symbol]["accuracy"] = 0
        crossings = boundary_crossings(checkpoints)
        self.assertEqual(crossings["A"]["first_mean_positive_margin_checkpoint"], "800")
        self.assertEqual(crossings["A"]["first_accuracy_at_least_0_5_checkpoint"], "800")
        self.assertIsNone(crossings["B"]["first_accuracy_at_least_0_5_checkpoint"])

    def test_margin_classification(self):
        self.assertEqual(classify_margin_interval(10.1), "still_improving")
        self.assertEqual(classify_margin_interval(10.0), "plateau")
        self.assertEqual(classify_margin_interval(-10.0), "plateau")
        self.assertEqual(classify_margin_interval(-10.1), "regressing")

    def test_channel_specific_telemetry(self):
        trial = row(
            "A",
            {"A": 0, "B": 2, "C": 0, "D": 0},
            decision="B",
            directional={"symbol/A": 1.0, "symbol/B": -1.0},
        )
        trial.directional_update.update({
            "channel_updates": {"symbol/A": 3, "symbol/B": 2},
            "channel_sum_abs_delta": {"symbol/A": 0.3, "symbol/B": 0.2},
            "channel_unique_edge_updates": {"symbol/A": 3, "symbol/B": 2},
            "channel_hop_counts": {"symbol/A": {1: 2, 2: 1}, "symbol/B": {1: 1, 2: 1}},
        })
        telemetry = channel_credit_telemetry([trial], unique_edge_counts={"symbol/A": 3, "symbol/B": 2, "symbol/C": 0, "symbol/D": 0})
        self.assertEqual(telemetry["symbol/A"]["positive_direction_requests"], 1)
        self.assertEqual(telemetry["symbol/B"]["negative_direction_requests"], 1)
        self.assertEqual(telemetry["symbol/A"]["1hop_updates"], 2)
        self.assertEqual(telemetry["symbol/A"]["2hop_updates"], 1)
        self.assertEqual(telemetry["symbol/B"]["unique_edges_modified"], 2)

    def test_output_collapse_detection(self):
        results = [row("A", {"A": 4, "B": 0, "C": 0, "D": 0})] * 8
        results += [row("B", {"B": 4, "A": 0, "C": 0, "D": 0})] * 2
        self.assertTrue(extended_checkpoint_metrics(results)["collapsed"])

    def test_learning_config_and_adaptive_budget_guard(self):
        from drosomath.malecns.symbol_learning import SymbolLearningConfig
        config = SymbolLearningConfig(training_trials=1600)
        self.assertEqual((config.duration_ms, config.stimulus_rate_hz), (40.0, 205.0))
        self.assertEqual((config.directional_learning_rate, config.reward_learning_rate), (0.02, 0.02))
        self.assertFalse(config.adaptive_plastic_budget)
        with self.assertRaises(ValueError):
            SymbolLearningConfig(training_trials=1600, adaptive_plastic_budget=True)

    def test_no_external_decoder_or_keyboard_teacher(self):
        module = __import__("drosomath.malecns.symbol_learning", fromlist=["x"])
        source = inspect.getsource(module)
        self.assertNotIn("PopulationReadout", source)
        self.assertNotIn("keyboard_learning", source)
        self.assertNotIn("legacy_rescue", source)


if __name__ == "__main__":
    unittest.main()
