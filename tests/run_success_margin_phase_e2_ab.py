"""Run the clean Phase E.2 OFF/ON success-margin directional A/B."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.malecns.keyboard_learning import KeyboardTrainingConfig, run_keyboard_training
from drosomath.malecns.loader import load_malecns_v1


def _mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _temporal_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    temporal = [row["temporal_engagement"] for row in rows if isinstance(row.get("temporal_engagement"), dict)]
    return {
        "trial_count": len(temporal),
        "mean_early_plastic_causal_edges": _mean(
            [float(row.get("early_active_plastic_click_causal_edges", 0.0)) for row in temporal]
        ),
        "mean_early_eligibility_per_window": _mean(
            [float(row.get("early_eligibility_per_window", 0.0)) for row in temporal]
        ),
        "mean_peak_click_rate_hz": _mean(
            [float(row.get("peak_click_output_rate_hz", 0.0)) for row in temporal]
        ),
        "mean_synchrony_proxy": _mean(
            [float(row.get("mean_click_output_synchrony_proxy", 0.0)) for row in temporal]
        ),
        "mean_unique_click_output_neurons": _mean(
            [float(row.get("unique_click_output_neurons_recruited", 0.0)) for row in temporal]
        ),
    }


def _run_summary(report: dict[str, object]) -> dict[str, object]:
    per_key = report["per_key"]
    ranked = sorted(
        per_key.items(),
        key=lambda item: (float(item[1]["recent_accuracy"]), item[0]),
    )
    weakest = [label for label, _ in ranked[:10]]
    weakest_mean = _mean([float(per_key[label]["recent_accuracy"]) for label in weakest])
    success_margin = report["success_margin_summary"]
    transitions = report["outcome_transition_summary"]
    return {
        "overall_accuracy": float(report["accuracy"]),
        "macro_recent_accuracy": float(report["macro_recent_accuracy"]),
        "median_recent_accuracy": float(report["median_recent_accuracy"]),
        "minimum_recent_accuracy": float(report["minimum_recent_accuracy"]),
        "per_key_recent_accuracy": {
            label: float(item["recent_accuracy"]) for label, item in per_key.items()
        },
        "failure_count": int(report["completed_trials"] - round(float(report["accuracy"]) * int(report["completed_trials"]))),
        "weakest_10_keys": weakest,
        "weakest_10_mean_accuracy": weakest_mean,
        "successful_trials_below_margin_target": int(success_margin["successful_trials_below_margin_target"]),
        "successful_trials_at_or_above_margin_target": int(success_margin["successful_trials_at_or_above_margin_target"]),
        "margin_update_trial_count": int(success_margin["margin_update_trial_count"]),
        "margin_directional_edge_update_count": int(success_margin["margin_directional_edge_update_count"]),
        "margin_directional_sum_abs_delta": float(success_margin["margin_directional_sum_abs_delta"]),
        "mean_success_peak_click_rate_hz": float(success_margin["mean_success_peak_click_rate_hz"]),
        "mean_failure_peak_click_rate_hz": float(success_margin["mean_failure_peak_click_rate_hz"]),
        "mean_success_click_output_synchrony": float(success_margin["mean_success_click_output_synchrony"]),
        "mean_failure_click_output_synchrony": float(success_margin["mean_failure_click_output_synchrony"]),
        "mean_success_unique_click_output_neurons": float(success_margin["mean_success_unique_click_output_neurons"]),
        "mean_failure_unique_click_output_neurons": float(success_margin["mean_failure_unique_click_output_neurons"]),
        "mean_recent_outcome_transition_rate": float(transitions["mean_recent_outcome_transition_rate"]),
        "weak_key_outcome_transition_rate": float(transitions["weak_key_outcome_transition_rate"]),
        "generic_directional_update_count": int(report["adaptive_budget"]["directional_generic_update_count"]),
        "mean_stability": float(report["final_plasticity"]["mean_stability"]),
        "multiplier_saturation_fraction": float(report["final_plasticity"]["multiplier_saturation_fraction"]),
        "plastic_budget_start": int(report["adaptive_budget"]["plastic_budget_start"]),
        "plastic_budget_end": int(report["adaptive_budget"]["plastic_budget_end"]),
        "plastic_budget_drift": int(report["adaptive_budget"]["budget_delta"]),
        "temporal": _temporal_summary(report["temporal_engagement_trials"]),
    }


def _config(*, trials: int, seed: int, enabled: bool) -> KeyboardTrainingConfig:
    return KeyboardTrainingConfig(
        trials=trials,
        seed=seed,
        resume=False,
        adaptive_plastic_budget=False,
        route_health_diagnostic=False,
        temporal_engagement_diagnostic=True,
        success_margin_directional_learning=enabled,
        checkpoint_every=trials,
        retention_probe_interval=trials + 1,
        dashboard_update_interval_seconds=3600.0,
    )


def run(*, data_dir: Path, trials: int, seed: int, output: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    with tempfile.TemporaryDirectory(prefix="drosomath_e2_success_margin_") as temporary:
        root = Path(temporary)
        reports = {}
        for name, enabled in (("baseline", False), ("success_margin", True)):
            reports[name] = run_keyboard_training(
                connectome,
                config=_config(trials=trials, seed=seed, enabled=enabled),
                result_path=root / f"{name}.json",
                progress_path=root / f"{name}_progress.json",
                checkpoint_path=root / f"{name}.npz",
                html_path=root / f"{name}.html",
            )

    baseline = _run_summary(reports["baseline"])
    margin = _run_summary(reports["success_margin"])
    baseline_top = sorted(
        baseline["per_key_recent_accuracy"].items(),
        key=lambda item: (-float(item[1]), item[0]),
    )[:10]
    baseline_top_labels = [label for label, _ in baseline_top]
    baseline_top_mean = _mean([float(baseline["per_key_recent_accuracy"][label]) for label in baseline_top_labels])
    margin_on_baseline_top_mean = _mean(
        [float(margin["per_key_recent_accuracy"][label]) for label in baseline_top_labels]
    )
    strong_regressions = sum(
        int(
            float(baseline["per_key_recent_accuracy"][label]) >= 0.9
            and float(margin["per_key_recent_accuracy"][label])
            <= float(baseline["per_key_recent_accuracy"][label]) - 0.2
        )
        for label in baseline_top_labels
    )
    comparison = {
        "accuracy_delta": margin["overall_accuracy"] - baseline["overall_accuracy"],
        "macro_recent_delta": margin["macro_recent_accuracy"] - baseline["macro_recent_accuracy"],
        "minimum_recent_delta": margin["minimum_recent_accuracy"] - baseline["minimum_recent_accuracy"],
        "failure_delta": margin["failure_count"] - baseline["failure_count"],
        "weak_group_delta": margin["weakest_10_mean_accuracy"] - baseline["weakest_10_mean_accuracy"],
        "outcome_transition_rate_delta": (
            margin["mean_recent_outcome_transition_rate"]
            - baseline["mean_recent_outcome_transition_rate"]
        ),
        "weak_key_outcome_transition_rate_delta": (
            margin["weak_key_outcome_transition_rate"]
            - baseline["weak_key_outcome_transition_rate"]
        ),
        "mean_success_peak_rate_delta": (
            margin["mean_success_peak_click_rate_hz"]
            - baseline["mean_success_peak_click_rate_hz"]
        ),
        "mean_synchrony_delta": (
            margin["mean_success_click_output_synchrony"]
            - baseline["mean_success_click_output_synchrony"]
        ),
        "mean_unique_click_neurons_delta": (
            margin["mean_success_unique_click_output_neurons"]
            - baseline["mean_success_unique_click_output_neurons"]
        ),
    }
    safety = {
        "plastic_budget_drift_baseline": baseline["plastic_budget_drift"],
        "plastic_budget_drift_success_margin": margin["plastic_budget_drift"],
        "strong_key_regressions": int(strong_regressions),
        "baseline_top_10_keys": baseline_top_labels,
        "baseline_top_10_mean_recent_accuracy": baseline_top_mean,
        "success_margin_on_baseline_top_10_mean_recent_accuracy": margin_on_baseline_top_mean,
        "multiplier_saturation_change": margin["multiplier_saturation_fraction"] - baseline["multiplier_saturation_fraction"],
    }
    conclusion = {
        "margin_learning_improved_robustness": bool(
            comparison["outcome_transition_rate_delta"] < 0.0
        ),
        "weak_keys_improved": bool(comparison["weak_group_delta"] > 0.0),
        "strong_keys_preserved": bool(strong_regressions == 0),
        "evidence_supports_margin_hypothesis": bool(
            comparison["outcome_transition_rate_delta"] <= 0.0
            and comparison["weak_group_delta"] >= 0.0
            and strong_regressions == 0
        ),
    }
    result = {
        "protocol": {
            "seed": seed,
            "trials": trials,
            "resume": False,
            "same_config_except_success_margin": True,
            "connectome": "MaleCNS v1 min_connection_synapses=5",
            "adaptive_plastic_budget": False,
            "temporal_engagement_diagnostic": True,
        },
        "baseline": baseline,
        "success_margin": margin,
        "comparison": comparison,
        "safety": safety,
        "temporal_confirmation": {
            "baseline": baseline["temporal"],
            "success_margin": margin["temporal"],
        },
        "conclusion": conclusion,
        "pass": bool(
            reports["baseline"]["completed_trials"] == trials
            and reports["success_margin"]["completed_trials"] == trials
            and reports["baseline"]["config"]["success_margin_directional_learning"] is False
            and reports["success_margin"]["config"]["success_margin_directional_learning"] is True
            and baseline["plastic_budget_drift"] == 0
            and margin["plastic_budget_drift"] == 0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase E.2 success-margin directional A/B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/latest_success_margin_phase_e2_ab.json"))
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, trials=args.trials, seed=args.seed, output=args.output), ensure_ascii=False, indent=2))
