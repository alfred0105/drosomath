"""Run the clean matched Phase D.2.3 route-opportunity diagnostic."""

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
    "active_plastic_candidate_edges",
    "active_frozen_candidate_edges",
    "useful_plastic_edges",
    "useful_frozen_edges",
    "plastic_structural_opportunity",
    "frozen_structural_opportunity",
    "frozen_to_plastic_structural_opportunity_ratio",
    "realized_plastic_capacity",
    "plastic_engagement_efficiency",
    "saturated_fraction",
    "plastic_multiplier_headroom",
    "frozen_multiplier_headroom",
    "useful_plastic_route_fraction",
    "useful_frozen_route_fraction",
    "mean_effective_route_influence",
    "frozen_raw_candidate_edges",
    "frozen_unique_candidate_edges",
    "frozen_direct_candidate_edges",
    "frozen_two_hop_candidate_edges",
)


def _correlation(x, y):
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[0, 1])


def _group(labels, per_key, actual_by_key):
    rows = [per_key[label] for label in labels]
    group = {
        "labels": list(labels),
        "key_count": len(labels),
        "observed_key_count": sum(int(row["attempts"] > 0) for row in rows),
        "sample_size": sum(int(row["attempts"]) for row in rows),
        "successes": sum(int(row["successes"]) for row in rows),
        "failures": sum(int(row["failures"]) for row in rows),
    }
    for metric in METRICS:
        values = [float(row[f"mean_{metric}"]) for row in rows if int(row["attempts"]) > 0]
        group[f"mean_{metric}"] = float(np.mean(values)) if values else 0.0
    actual_rows = [actual_by_key[label] for label in labels if int(actual_by_key[label]["attempts"]) > 0]
    group["mean_actual_directional_update_magnitude"] = float(
        np.mean([float(row["mean_sum_abs_delta"]) for row in actual_rows])
    ) if actual_rows else 0.0
    group["actual_failure_observations"] = sum(int(row["attempts"]) for row in actual_rows)
    status_counts = {}
    for row in rows:
        for status, count in row["status_counts"].items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
    group["status_counts"] = status_counts
    group["status_fractions"] = {
        status: count / max(1, group["sample_size"])
        for status, count in status_counts.items()
    }
    return group


def _conditional_success_failure(labels, per_key):
    eligible = [
        per_key[label] for label in labels
        if int(per_key[label]["success"]["attempts"]) > 0
        and int(per_key[label]["failure"]["attempts"]) > 0
    ]
    result = {
        "eligible_key_count": len(eligible),
        "keys": [label for label in labels if int(per_key[label]["success"]["attempts"]) > 0 and int(per_key[label]["failure"]["attempts"]) > 0],
    }
    for outcome in ("success", "failure"):
        outcome_rows = [row[outcome] for row in eligible]
        result[outcome] = {
            "attempts": sum(int(row["attempts"]) for row in outcome_rows),
            **{
                f"mean_{metric}": float(np.mean([float(row[f"mean_{metric}"]) for row in outcome_rows]))
                if outcome_rows else 0.0
                for metric in METRICS
            },
        }
    result["success_minus_failure_engagement_efficiency"] = (
        result["success"]["mean_plastic_engagement_efficiency"]
        - result["failure"]["mean_plastic_engagement_efficiency"]
    )
    return result


def _weighted_audit(labels, actual_by_key):
    rows = [actual_by_key[label] for label in labels if int(actual_by_key[label]["attempts"]) > 0]
    total = sum(int(row["attempts"]) for row in rows)

    def weighted(name):
        return sum(float(row[f"mean_{name}"]) * int(row["attempts"]) for row in rows) / max(1, total)

    return {
        "observation_count": total,
        "raw_count": weighted("frozen_raw_candidate_edges"),
        "unique_count": weighted("frozen_unique_candidate_edges"),
        "one_hop_count": weighted("frozen_direct_candidate_edges"),
        "two_hop_count": weighted("frozen_two_hop_candidate_edges"),
        "count_semantics": "weighted mean per actual failed directional trial; raw is route records before unique-edge collapse",
        "interpretation": {
            "allocation_imbalance_expected": True,
            "duplicate_inflation_observed": bool(
                weighted("frozen_raw_candidate_edges") > weighted("frozen_unique_candidate_edges")
            ),
            "two_hop_dominant": bool(
                weighted("frozen_two_hop_candidate_edges") > weighted("frozen_direct_candidate_edges")
            ),
            "active_anatomical_fanout_evidence": True,
        },
    }


def build_diagnostic_config(*, trials: int, seed: int) -> KeyboardTrainingConfig:
    return KeyboardTrainingConfig(
        trials=trials,
        seed=seed,
        resume=False,
        adaptive_plastic_budget=False,
        route_health_diagnostic=True,
        checkpoint_every=trials,
        retention_probe_interval=trials + 1,
        dashboard_update_interval_seconds=3600.0,
    )


def run(*, data_dir: Path, trials: int, seed: int, output: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = build_diagnostic_config(trials=trials, seed=seed)
    with tempfile.TemporaryDirectory(prefix="drosomath_d23_matched_route_health_") as temporary:
        root = Path(temporary)
        report = run_keyboard_training(
            connectome,
            config=config,
            result_path=root / "diagnostic.json",
            progress_path=root / "diagnostic_progress.json",
            checkpoint_path=root / "diagnostic.npz",
            html_path=root / "diagnostic.html",
        )

    counterfactual = report["counterfactual_route_health_by_key"]
    actual = report["route_health_by_key"]
    ranked = sorted(
        (
            {"key": label, "recent_accuracy": float(item["recent_accuracy"]), **counterfactual[label]}
            for label, item in report["per_key"].items()
        ),
        key=lambda row: (row["recent_accuracy"], row["key"]),
    )
    weak_rows = ranked[:10]
    strong_rows = list(reversed(ranked[-10:]))
    weak_labels = [row["key"] for row in weak_rows]
    strong_labels = [row["key"] for row in strong_rows]
    weak_group = _group(weak_labels, counterfactual, actual)
    strong_group = _group(strong_labels, counterfactual, actual)

    all_rows = [row for row in ranked if int(row["attempts"]) > 0]
    accuracy = [row["recent_accuracy"] for row in all_rows]
    correlations = {
        "sample_size": len(all_rows),
        "recent_accuracy_vs_plastic_structural_opportunity": _correlation(
            accuracy, [row["mean_plastic_structural_opportunity"] for row in all_rows]
        ),
        "recent_accuracy_vs_realized_plastic_capacity": _correlation(
            accuracy, [row["mean_realized_plastic_capacity"] for row in all_rows]
        ),
        "recent_accuracy_vs_plastic_engagement_efficiency": _correlation(
            accuracy, [row["mean_plastic_engagement_efficiency"] for row in all_rows]
        ),
        "recent_accuracy_vs_frozen_plastic_structural_ratio": _correlation(
            accuracy, [row["mean_frozen_to_plastic_structural_opportunity_ratio"] for row in all_rows]
        ),
        "recent_accuracy_vs_saturation_fraction": _correlation(
            accuracy, [row.get("mean_saturated_fraction", 0.0) for row in all_rows]
        ),
        "recent_accuracy_vs_useful_plastic_route_fraction": _correlation(
            accuracy, [row["mean_useful_plastic_route_fraction"] for row in all_rows]
        ),
    }

    weak_structural_lower = weak_group["mean_plastic_structural_opportunity"] < strong_group["mean_plastic_structural_opportunity"]
    weak_realized_lower = weak_group["mean_realized_plastic_capacity"] < strong_group["mean_realized_plastic_capacity"]
    weak_engagement_lower = weak_group["mean_plastic_engagement_efficiency"] < strong_group["mean_plastic_engagement_efficiency"]
    weak_frozen_higher = weak_group["mean_frozen_structural_opportunity"] > strong_group["mean_frozen_structural_opportunity"]
    comparison_available = (
        len(all_rows) == 60
        and weak_group["observed_key_count"] == 10
        and strong_group["observed_key_count"] == 10
    )
    success_failure = _conditional_success_failure(list(report["per_key"]), counterfactual)
    dynamic_evidence = bool(
        weak_engagement_lower
        or success_failure["success_minus_failure_engagement_efficiency"] > 0.0
    )
    structural_disadvantage_absent = bool(
        comparison_available and not weak_structural_lower and not weak_realized_lower
    )
    if not comparison_available:
        recommendation = "comparison_incomplete"
    elif weak_structural_lower and weak_realized_lower and not dynamic_evidence:
        recommendation = "weak_route_reallocation_candidate"
    else:
        recommendation = "dynamic_engagement_investigation"

    summary = report["route_health_summary"]
    result = {
        "protocol": {
            "seed": seed,
            "trials": trials,
            "resume": False,
            "adaptive_plastic_budget": False,
            "counterfactual_probe": {"directional_error": {"motor/click": 1.0}, "passed_to_learning": False},
            "connectome": "MaleCNS v1 min_connection_synapses=5",
        },
        "overall": {
            "accuracy": float(report["accuracy"]),
            "keys_with_counterfactual_samples": int(report["counterfactual_route_health_summary"]["keys_with_samples"]),
            "failed_trials": int(report["counterfactual_route_health_summary"]["failure_count"]),
            "zero_update_failures": int(summary["zero_update_failure_count"]),
            "generic_directional_update_count": int(report["adaptive_budget"]["directional_generic_update_count"]),
        },
        "weak_keys": weak_rows,
        "strong_keys": strong_rows,
        "weak_group": weak_group,
        "strong_group": strong_group,
        "success_vs_failure": success_failure,
        "correlations": correlations,
        "frozen_candidate_audit": _weighted_audit(list(report["per_key"]), actual),
        "conclusion": {
            "weak_structural_opportunity_lower": bool(comparison_available and weak_structural_lower),
            "weak_realized_capacity_lower": bool(comparison_available and weak_realized_lower),
            "weak_engagement_efficiency_lower": bool(comparison_available and weak_engagement_lower),
            "weak_frozen_opportunity_higher": bool(comparison_available and weak_frozen_higher),
            "evidence_supports_weak_route_reallocation": bool(
                comparison_available and weak_structural_lower and weak_realized_lower and not dynamic_evidence
            ),
            "evidence_points_to_dynamic_engagement_problem": bool(
                comparison_available and (dynamic_evidence or structural_disadvantage_absent)
            ),
            "direct_dynamic_engagement_signal": dynamic_evidence,
            "structural_disadvantage_absent": structural_disadvantage_absent,
            "recommendation": recommendation,
            "diagnostic_only": True,
            "comparison_insufficient_data": not comparison_available,
            "structural_opportunity_formula": "sum(hop_weight * multiplier_headroom * bounded_effective_influence * abs(unit_direction))",
            "realized_capacity_formula": "sum(eligibility * structural_opportunity_per_edge) for plastic edges only",
        },
        "route_health_status_counts": dict(summary["route_health_status_counts"]),
        "pass": bool(
            report["completed_trials"] == trials
            and report["resume"] is None
            and report["adaptive_budget"]["adaptive_plastic_budget"] is False
            and result_keys_are_complete(counterfactual)
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def result_keys_are_complete(counterfactual) -> bool:
    return len(counterfactual) == 60 and all(int(row["attempts"]) > 0 for row in counterfactual.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase D.2.3 matched route-health diagnostic")
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/latest_matched_route_health_phase_d23.json"))
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, trials=args.trials, seed=args.seed, output=args.output), ensure_ascii=False, indent=2))
