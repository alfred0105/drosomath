"""Run the diagnostic-only Phase E.3A credit-targeting A/B."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.malecns.keyboard_learning import (
    KeyboardTrainingConfig,
    pair_next_same_key_occurrences,
    run_keyboard_training,
    summarize_next_same_key_pairs,
)
from drosomath.malecns.loader import load_malecns_v1


def _mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values) -> float:
    return float(np.median(values)) if values else 0.0


def _targeting_summary(report: dict[str, object], *, margin_records: bool) -> dict[str, object]:
    records = list(report.get("credit_targeting_trials", []))
    selected = [
        row for row in records
        if bool(row.get("margin_update")) is bool(margin_records)
        and (
            bool(row.get("margin_update"))
            if margin_records
            else bool(row.get("below_margin_success"))
        )
    ]
    fields = (
        "updated_edge_count",
        "fraction_updated_in_any_trial_causal_set",
        "fraction_updated_in_early_causal_set",
        "fraction_updated_in_peak_window_causal_set",
        "fraction_updated_in_final_window_causal_set",
        "fraction_peak_causal_edges_updated",
        "updated_direct_1hop_fraction",
        "updated_2hop_fraction",
        "peak_final_causal_jaccard",
    )
    values = {
        field: [float(row.get("credit_targeting", {}).get(field, 0.0)) for row in selected]
        for field in fields
    }
    return {
        "selected_trial_count": len(selected),
        "mean_updated_edge_count": _mean(values["updated_edge_count"]),
        "mean_fraction_updated_in_any_trial_causal_set": _mean(values["fraction_updated_in_any_trial_causal_set"]),
        "mean_fraction_updated_in_early_causal_set": _mean(values["fraction_updated_in_early_causal_set"]),
        "mean_fraction_updated_in_peak_window_causal_set": _mean(values["fraction_updated_in_peak_window_causal_set"]),
        "mean_fraction_updated_in_final_window_causal_set": _mean(values["fraction_updated_in_final_window_causal_set"]),
        "mean_fraction_peak_causal_edges_updated": _mean(values["fraction_peak_causal_edges_updated"]),
        "mean_updated_direct_1hop_fraction": _mean(values["updated_direct_1hop_fraction"]),
        "mean_updated_2hop_fraction": _mean(values["updated_2hop_fraction"]),
        "mean_peak_final_causal_jaccard": _mean(values["peak_final_causal_jaccard"]),
        "median_peak_final_causal_jaccard": _median(values["peak_final_causal_jaccard"]),
        "next_same_key": summarize_next_same_key_pairs(
            pair_next_same_key_occurrences(
                records,
                eligible_field="margin_update" if margin_records else "below_margin_success",
            )
        ),
    }


def _config(*, trials: int, seed: int, margin: bool) -> KeyboardTrainingConfig:
    return KeyboardTrainingConfig(
        trials=trials,
        seed=seed,
        resume=False,
        adaptive_plastic_budget=False,
        route_health_diagnostic=False,
        temporal_engagement_diagnostic=True,
        credit_targeting_diagnostic=True,
        success_margin_directional_learning=margin,
        checkpoint_every=trials,
        retention_probe_interval=trials + 1,
        dashboard_update_interval_seconds=3600.0,
    )


def _run_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "accuracy": float(report["accuracy"]),
        "failure_count": int(
            sum(int(item["trials"] - item["correct"]) for item in report["per_key"].values())
        ),
        "macro_recent_accuracy": float(report["macro_recent_accuracy"]),
        "minimum_recent_accuracy": float(report["minimum_recent_accuracy"]),
        "credit_targeting": _targeting_summary(report, margin_records=bool(report["config"]["success_margin_directional_learning"])),
        "baseline_below_margin": _targeting_summary(report, margin_records=False),
        "key_sharing": report.get("credit_targeting_key_edge_sharing", {}),
        "plastic_budget": {
            "start": int(report["adaptive_budget"]["plastic_budget_start"]),
            "end": int(report["adaptive_budget"]["plastic_budget_end"]),
            "drift": int(report["adaptive_budget"]["budget_delta"]),
        },
        "config": {
            "success_margin_directional_learning": bool(report["config"]["success_margin_directional_learning"]),
            "credit_targeting_diagnostic": bool(report["config"]["credit_targeting_diagnostic"]),
        },
    }


def run(*, data_dir: Path, trials: int, seed: int, output: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    with tempfile.TemporaryDirectory(prefix="drosomath_e3a_credit_targeting_") as temporary:
        root = Path(temporary)
        reports = {}
        for name, margin in (("baseline", False), ("margin", True)):
            reports[name] = run_keyboard_training(
                connectome,
                config=_config(trials=trials, seed=seed, margin=margin),
                result_path=root / f"{name}.json",
                progress_path=root / f"{name}_progress.json",
                checkpoint_path=root / f"{name}.npz",
                html_path=root / f"{name}.html",
            )

    baseline = _run_summary(reports["baseline"])
    margin = _run_summary(reports["margin"])
    baseline_targeting = baseline["baseline_below_margin"]
    margin_targeting = margin["credit_targeting"]
    comparison = {
        "peak_window_update_overlap_delta": (
            margin_targeting["mean_fraction_updated_in_peak_window_causal_set"]
            - baseline_targeting["mean_fraction_updated_in_peak_window_causal_set"]
        ),
        "early_window_update_overlap": margin_targeting["mean_fraction_updated_in_early_causal_set"],
        "final_window_update_overlap": margin_targeting["mean_fraction_updated_in_final_window_causal_set"],
        "fraction_peak_causal_edges_updated": margin_targeting["mean_fraction_peak_causal_edges_updated"],
        "cross_key_mean_overlap": margin["key_sharing"].get("mean_pairwise_updated_edge_overlap", 0.0),
        "cross_key_median_overlap": margin["key_sharing"].get("median_pairwise_updated_edge_overlap", 0.0),
        "cross_key_maximum_overlap": margin["key_sharing"].get("maximum_pairwise_updated_edge_overlap", 0.0),
        "fraction_edges_used_by_at_least_2_keys": margin["key_sharing"].get("fraction_updated_edges_used_by_at_least_2_keys", 0.0),
        "fraction_edges_used_by_at_least_5_keys": margin["key_sharing"].get("fraction_updated_edges_used_by_at_least_5_keys", 0.0),
        "next_same_key_peak_delta_baseline": baseline_targeting["next_same_key"]["mean_next_same_key_peak_delta"],
        "next_same_key_peak_delta_margin": margin_targeting["next_same_key"]["mean_next_same_key_peak_delta"],
        "next_same_key_success_change_baseline": baseline_targeting["next_same_key"]["mean_next_same_key_success_change"],
        "next_same_key_success_change_margin": margin_targeting["next_same_key"]["mean_next_same_key_success_change"],
    }
    conclusion = {
        "margin_updates_strongly_aligned_with_peak_window": bool(
            margin_targeting["mean_fraction_updated_in_peak_window_causal_set"] >= 0.5
            and margin_targeting["mean_fraction_peak_causal_edges_updated"] >= 0.5
        ),
        "margin_updates_mostly_final_window_only": bool(
            margin_targeting["mean_fraction_updated_in_final_window_causal_set"]
            > margin_targeting["mean_fraction_updated_in_peak_window_causal_set"]
            and margin_targeting["mean_fraction_updated_in_final_window_causal_set"] >= 0.5
        ),
        "updated_edges_heavily_shared_across_keys": bool(
            margin["key_sharing"].get("fraction_updated_edges_used_by_at_least_2_keys", 0.0) >= 0.5
        ),
        "next_same_key_peak_improved_more_than_baseline": bool(
            comparison["next_same_key_peak_delta_margin"]
            > comparison["next_same_key_peak_delta_baseline"]
        ),
    }
    result = {
        "protocol": {
            "seed": seed,
            "trials": trials,
            "resume": False,
            "same_config_except_success_margin": True,
            "credit_targeting_diagnostic": True,
            "temporal_engagement_diagnostic": True,
            "connectome": "MaleCNS v1 min_connection_synapses=5",
        },
        "baseline": baseline,
        "margin": margin,
        "comparison": comparison,
        "conclusion": conclusion,
        "pass": bool(
            reports["baseline"]["completed_trials"] == trials
            and reports["margin"]["completed_trials"] == trials
            and baseline["config"]["success_margin_directional_learning"] is False
            and margin["config"]["success_margin_directional_learning"] is True
            and baseline["config"]["credit_targeting_diagnostic"] is True
            and margin["config"]["credit_targeting_diagnostic"] is True
            and baseline["plastic_budget"]["drift"] == 0
            and margin["plastic_budget"]["drift"] == 0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase E.3A credit-targeting diagnostic A/B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/latest_credit_targeting_phase_e3a.json"))
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, trials=args.trials, seed=args.seed, output=args.output), ensure_ascii=False, indent=2))
