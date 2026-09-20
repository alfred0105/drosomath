"""Run the clean real-keyboard Phase D.2.2 route-health diagnostic."""

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


METRICS = (
    "mean_active_plastic_edges",
    "mean_total_eligibility",
    "mean_total_route_credit",
    "mean_estimated_available_adjustment",
    "mean_useful_frozen_edges",
    "mean_estimated_frozen_capacity",
    "mean_frozen_to_plastic_capacity_ratio",
    "mean_saturated_fraction",
    "mean_sum_abs_delta",
)


def _correlation(x, y):
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[0, 1])


def _group(labels, per_key):
    rows = [per_key[label] for label in labels]
    observed = [row for row in rows if int(row["attempts"]) > 0]
    group = {
        "labels": list(labels),
        "key_count": len(labels),
        "observed_key_count": len(observed),
        "sample_size": sum(int(row["attempts"]) for row in observed),
        "failures": sum(int(row["failures"]) for row in observed),
        "zero_update_failures": sum(int(row["zero_update_failures"]) for row in observed),
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in observed]
        group[metric] = float(np.mean(values)) if values else 0.0
    status_counts = {}
    for row in observed:
        for status, count in row["status_counts"].items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
    group["status_counts"] = status_counts
    group["status_fractions"] = {
        status: count / max(1, group["sample_size"])
        for status, count in status_counts.items()
    }
    return group


def build_diagnostic_config(*, trials: int, seed: int) -> KeyboardTrainingConfig:
    """Build the fixed, clean diagnostic configuration."""
    return KeyboardTrainingConfig(
        trials=trials,
        seed=seed,
        resume=False,
        adaptive_plastic_budget=False,
        checkpoint_every=trials,
        retention_probe_interval=trials + 1,
        dashboard_update_interval_seconds=3600.0,
    )


def run(*, data_dir: Path, trials: int, seed: int, output: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = build_diagnostic_config(trials=trials, seed=seed)
    with tempfile.TemporaryDirectory(prefix="drosomath_d22_route_health_") as temporary:
        root = Path(temporary)
        report = run_keyboard_training(
            connectome,
            config=config,
            result_path=root / "diagnostic.json",
            progress_path=root / "diagnostic_progress.json",
            checkpoint_path=root / "diagnostic.npz",
            html_path=root / "diagnostic.html",
        )

    per_key = report["route_health_by_key"]
    ranked = sorted(
        (
            {"key": label, "recent_accuracy": float(item["recent_accuracy"]), **per_key[label]}
            for label, item in report["per_key"].items()
        ),
        key=lambda row: (row["recent_accuracy"], row["key"]),
    )
    weak_rows = ranked[:10]
    strong_rows = list(reversed(ranked[-10:]))
    weak_labels = [row["key"] for row in weak_rows]
    strong_labels = [row["key"] for row in strong_rows]
    weak_group = _group(weak_labels, per_key)
    strong_group = _group(strong_labels, per_key)

    correlation_rows = [
        row for row in ranked if int(row["attempts"]) > 0
    ]
    accuracy = [row["recent_accuracy"] for row in correlation_rows]
    correlations = {
        "sample_size": len(correlation_rows),
        "recent_accuracy_vs_mean_plastic_capacity": _correlation(
            accuracy, [row["mean_estimated_available_adjustment"] for row in correlation_rows]
        ),
        "recent_accuracy_vs_mean_frozen_plastic_ratio": _correlation(
            accuracy, [row["mean_frozen_to_plastic_capacity_ratio"] for row in correlation_rows]
        ),
        "recent_accuracy_vs_mean_saturated_fraction": _correlation(
            accuracy, [row["mean_saturated_fraction"] for row in correlation_rows]
        ),
        "recent_accuracy_vs_mean_directional_update_magnitude": _correlation(
            accuracy, [row["mean_sum_abs_delta"] for row in correlation_rows]
        ),
        "recent_accuracy_vs_active_plastic_route_edges": _correlation(
            accuracy, [row["mean_active_plastic_edges"] for row in correlation_rows]
        ),
    }
    group_comparison_available = bool(
        weak_group["observed_key_count"] > 0 and strong_group["observed_key_count"] > 0
    )
    weak_capacity_lower = bool(
        group_comparison_available
        and weak_group["mean_estimated_available_adjustment"] < strong_group["mean_estimated_available_adjustment"]
    )
    weak_saturation_higher = bool(
        group_comparison_available
        and weak_group["mean_saturated_fraction"] > strong_group["mean_saturated_fraction"]
    )
    weak_frozen_higher = bool(
        group_comparison_available
        and weak_group["mean_estimated_frozen_capacity"] > strong_group["mean_estimated_frozen_capacity"]
    )
    summary = report["route_health_summary"]
    result = {
        "protocol": {
            "seed": seed,
            "trials": trials,
            "resume": False,
            "adaptive_plastic_budget": False,
            "connectome": "MaleCNS v1 min_connection_synapses=5",
        },
        "overall": {
            "accuracy": float(report["accuracy"]),
            "macro_recent_accuracy": float(report["macro_recent_accuracy"]),
            "minimum_recent_accuracy": float(report["minimum_recent_accuracy"]),
            "generic_directional_update_count": int(report["adaptive_budget"]["directional_generic_update_count"]),
            "zero_update_failure_count": int(summary["zero_update_failure_count"]),
            "failed_directional_trial_count": int(summary["failed_directional_trial_count"]),
        },
        "weak_keys": weak_rows,
        "strong_keys": strong_rows,
        "weak_group": weak_group,
        "strong_group": strong_group,
        "correlations": correlations,
        "route_health_status_counts": dict(summary["route_health_status_counts"]),
        "conclusion": {
            "reason_d2_did_not_trigger": (
                "The conservative D.2 trigger requires generic directional edge_updates == 0; "
                f"the clean run recorded {summary['zero_update_failure_count']} zero-update failures."
            ),
            "evidence_for_weak_route_capacity": bool(weak_capacity_lower),
            "evidence_for_saturation": bool(weak_saturation_higher),
            "evidence_for_frozen_alternative_capacity": bool(
                weak_frozen_higher or weak_group["mean_useful_frozen_edges"] > 0.0
            ),
            "weak_strong_comparison_available": group_comparison_available,
            "comparison_insufficient_data": not group_comparison_available,
            "weak_vs_strong_capacity_comparison": {
                "weak_mean_plastic_capacity": weak_group["mean_estimated_available_adjustment"],
                "strong_mean_plastic_capacity": strong_group["mean_estimated_available_adjustment"],
                "weak_mean_frozen_capacity": weak_group["mean_estimated_frozen_capacity"],
                "strong_mean_frozen_capacity": strong_group["mean_estimated_frozen_capacity"],
            },
        },
        "pass": bool(
            report["completed_trials"] == trials
            and report["resume"] is None
            and report["adaptive_budget"]["plastic_budget_start"] == report["adaptive_budget"]["plastic_budget_end"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase D.2.2 route-health diagnostic")
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/latest_route_health_phase_d22.json"))
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, trials=args.trials, seed=args.seed, output=args.output), ensure_ascii=False, indent=2))
