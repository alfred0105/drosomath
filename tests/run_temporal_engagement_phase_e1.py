"""Run the diagnostic-only Phase E.1 temporal engagement analysis."""

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


def build_diagnostic_config(*, trials: int, seed: int) -> KeyboardTrainingConfig:
    return KeyboardTrainingConfig(
        trials=trials,
        seed=seed,
        resume=False,
        adaptive_plastic_budget=False,
        route_health_diagnostic=False,
        temporal_engagement_diagnostic=True,
        checkpoint_every=trials,
        retention_probe_interval=trials + 1,
        dashboard_update_interval_seconds=3600.0,
    )


def _mean(rows, name):
    values = [float(row["temporal_engagement"][name]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def _mean_reached(rows, name):
    values = [
        float(row["temporal_engagement"][name])
        for row in rows
        if row["temporal_engagement"].get(name) is not None
    ]
    return {
        "mean": float(np.mean(values)) if values else None,
        "reached_fraction": len(values) / max(1, len(rows)),
        "reached_count": len(values),
    }


def _summarize(rows):
    summary = {
        "trial_count": len(rows),
        "mean_windows_per_trial": _mean(rows, "mean_windows_per_trial"),
        "mean_first_causal_window": _mean_reached(rows, "first_click_causal_activity_window"),
        "mean_first_click_output_window": _mean_reached(rows, "first_click_output_spike_window"),
        "mean_first_network_activity_window": _mean_reached(rows, "first_any_network_activity_window"),
        "mean_early_causal_edges": _mean(rows, "early_active_click_causal_edges"),
        "mean_early_plastic_causal_edges": _mean(rows, "early_active_plastic_click_causal_edges"),
        "mean_early_frozen_causal_edges": _mean(rows, "early_active_frozen_click_causal_edges"),
        "mean_early_net_route_influence": _mean(rows, "early_net_click_route_influence"),
        "mean_net_route_influence_per_window": _mean(rows, "mean_net_click_route_influence_per_window"),
        "mean_eligibility_per_window": _mean(rows, "eligibility_per_window"),
        "mean_early_eligibility_per_window": _mean(rows, "early_eligibility_per_window"),
        "mean_raw_total_eligibility": _mean(rows, "raw_total_click_route_eligibility"),
        "mean_eligibility_accumulation_rate": _mean(rows, "eligibility_accumulation_rate"),
        "mean_activity_persistence": _mean(rows, "mean_adjacent_window_active_neuron_jaccard"),
        "mean_click_causal_persistence": _mean(rows, "mean_adjacent_window_click_causal_jaccard"),
        "mean_activity_persistence_windows": _mean(rows, "activity_persistence_windows"),
        "mean_positive_route_influence": _mean(rows, "mean_positive_effect_magnitude_per_window"),
        "mean_negative_route_influence": _mean(rows, "mean_negative_effect_magnitude_per_window"),
        "mean_unique_click_output_neurons": _mean(rows, "unique_click_output_neurons_recruited"),
        "mean_total_click_output_spikes": _mean(rows, "total_click_output_spikes"),
        "mean_peak_click_output_rate_hz": _mean(rows, "peak_click_output_rate_hz"),
        "mean_time_to_peak_click_rate_window": _mean_reached(rows, "time_to_peak_click_rate_window"),
        "mean_click_output_synchrony": _mean(rows, "mean_click_output_synchrony_proxy"),
        "mean_peak_simultaneous_click_output_recruitment": _mean(rows, "peak_simultaneous_click_output_recruitment"),
        "mean_first_window_click_rate_above_25pct_threshold": _mean_reached(rows, "first_window_click_rate_above_25pct_threshold"),
        "mean_first_window_click_rate_above_50pct_threshold": _mean_reached(rows, "first_window_click_rate_above_50pct_threshold"),
        "mean_first_window_click_rate_above_75pct_threshold": _mean_reached(rows, "first_window_click_rate_above_75pct_threshold"),
        "mean_first_window_click_rate_above_threshold": _mean_reached(rows, "first_window_click_rate_above_threshold"),
    }
    return summary


def _by_outcome(rows):
    return {
        "success": _summarize([row for row in rows if row["correct"]]),
        "failure": _summarize([row for row in rows if not row["correct"]]),
    }


def _group(labels, rows_by_key):
    rows = [row for label in labels for row in rows_by_key[label]]
    return {"labels": list(labels), **_summarize(rows)}


def _matched_success_failure(rows_by_key):
    eligible = {
        label: rows
        for label, rows in rows_by_key.items()
        if any(row["correct"] for row in rows) and any(not row["correct"] for row in rows)
    }
    success = _summarize([row for rows in eligible.values() for row in rows if row["correct"]])
    failure = _summarize([row for rows in eligible.values() for row in rows if not row["correct"]])
    return {
        "eligible_key_count": len(eligible),
        "eligible_keys": sorted(eligible),
        "success": success,
        "failure": failure,
    }


def run(*, data_dir: Path, trials: int, seed: int, output: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = build_diagnostic_config(trials=trials, seed=seed)
    with tempfile.TemporaryDirectory(prefix="drosomath_e1_temporal_engagement_") as temporary:
        root = Path(temporary)
        report = run_keyboard_training(
            connectome,
            config=config,
            result_path=root / "diagnostic.json",
            progress_path=root / "diagnostic_progress.json",
            checkpoint_path=root / "diagnostic.npz",
            html_path=root / "diagnostic.html",
        )

    temporal_rows = report["temporal_engagement_trials"]
    rows_by_key = {label: [] for label in report["token_labels"]}
    for row in temporal_rows:
        rows_by_key[row["label"]].append(row)
    ranked = sorted(
        report["per_key"].items(),
        key=lambda item: (float(item[1]["recent_accuracy"]), item[0]),
    )
    weak_labels = [label for label, _ in ranked[:10]]
    strong_labels = [label for label, _ in reversed(ranked[-10:])]
    weak_group = _group(weak_labels, rows_by_key)
    strong_group = _group(strong_labels, rows_by_key)
    matched = _matched_success_failure(rows_by_key)
    success_rows = [row for row in temporal_rows if row["correct"]]
    failure_rows = [row for row in temporal_rows if not row["correct"]]
    trial_length = {
        "mean_success_windows": _mean(success_rows, "mean_windows_per_trial"),
        "mean_failure_windows": _mean(failure_rows, "mean_windows_per_trial"),
        "success_trial_count": len(success_rows),
        "failure_trial_count": len(failure_rows),
    }

    success_first = matched["success"]["mean_first_causal_window"]["mean"]
    failure_first = matched["failure"]["mean_first_causal_window"]["mean"]
    success_click_first = matched["success"]["mean_first_click_output_window"]["mean"]
    failure_click_first = matched["failure"]["mean_first_click_output_window"]["mean"]
    success_early_net = matched["success"]["mean_early_net_route_influence"]
    failure_early_net = matched["failure"]["mean_early_net_route_influence"]
    success_eligibility = matched["success"]["mean_eligibility_per_window"]
    failure_eligibility = matched["failure"]["mean_eligibility_per_window"]
    failure_longer = trial_length["mean_failure_windows"] > trial_length["mean_success_windows"]
    duration_inflated = bool(
        failure_longer
        and matched["failure"]["mean_raw_total_eligibility"] > matched["success"]["mean_raw_total_eligibility"]
        and failure_eligibility <= success_eligibility
    )
    success_earlier = bool(
        success_first is not None and failure_first is not None and success_first < failure_first
    )
    success_stronger_net = success_early_net > failure_early_net
    success_better_click = bool(
        success_click_first is not None and failure_click_first is not None and success_click_first < failure_click_first
    ) or (
        matched["success"]["mean_unique_click_output_neurons"]
        > matched["failure"]["mean_unique_click_output_neurons"]
    )
    temporal_bottleneck = bool(
        matched["eligible_key_count"] > 0
        and (success_earlier or success_stronger_net or success_better_click or duration_inflated)
    )

    result = {
        "protocol": {
            "seed": seed,
            "trials": trials,
            "resume": False,
            "adaptive_plastic_budget": False,
            "temporal_engagement_diagnostic": True,
            "extra_brain_steps": 0,
            "extra_rng_calls": 0,
            "early_window_fraction": 0.25,
            "connectome": "MaleCNS v1 min_connection_synapses=5",
        },
        "overall": {
            "accuracy": float(report["accuracy"]),
            "success_trials": len(success_rows),
            "failure_trials": len(failure_rows),
            "temporal_trial_samples": len(temporal_rows),
        },
        "trial_length": trial_length,
        "same_key_success_vs_failure": matched,
        "weak_group": weak_group,
        "strong_group": strong_group,
        "click_output_recruitment": {
            "success": {
                "mean_unique_neurons": _mean(success_rows, "unique_click_output_neurons_recruited"),
                "mean_total_spikes": _mean(success_rows, "total_click_output_spikes"),
                "mean_peak_rate_hz": _mean(success_rows, "peak_click_output_rate_hz"),
                "mean_time_to_peak_window": _mean_reached(success_rows, "time_to_peak_click_rate_window"),
                "mean_synchrony_proxy": _mean(success_rows, "mean_click_output_synchrony_proxy"),
            },
            "failure": {
                "mean_unique_neurons": _mean(failure_rows, "unique_click_output_neurons_recruited"),
                "mean_total_spikes": _mean(failure_rows, "total_click_output_spikes"),
                "mean_peak_rate_hz": _mean(failure_rows, "peak_click_output_rate_hz"),
                "mean_time_to_peak_window": _mean_reached(failure_rows, "time_to_peak_click_rate_window"),
                "mean_synchrony_proxy": _mean(failure_rows, "mean_click_output_synchrony_proxy"),
            },
        },
        "conclusion": {
            "failure_trials_are_longer": bool(failure_longer),
            "failure_eligibility_is_duration_inflated": duration_inflated,
            "success_has_earlier_causal_recruitment": success_earlier,
            "success_has_stronger_early_net_influence": bool(success_stronger_net),
            "success_has_better_click_output_recruitment": success_better_click,
            "evidence_for_temporal_engagement_bottleneck": temporal_bottleneck,
            "eligibility_duration_audit": {
                "raw_total_success": matched["success"]["mean_raw_total_eligibility"],
                "raw_total_failure": matched["failure"]["mean_raw_total_eligibility"],
                "per_window_success": success_eligibility,
                "per_window_failure": failure_eligibility,
                "early_window_success": matched["success"]["mean_early_eligibility_per_window"],
                "early_window_failure": matched["failure"]["mean_early_eligibility_per_window"],
            },
            "diagnostic_only": True,
        },
        "pass": bool(
            report["completed_trials"] == trials
            and report["resume"] is None
            and report["adaptive_budget"]["adaptive_plastic_budget"] is False
            and len(temporal_rows) == trials
            and all(len(rows) > 0 for rows in rows_by_key.values())
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase E.1 temporal engagement diagnostic")
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/latest_temporal_engagement_phase_e1.json"))
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, trials=args.trials, seed=args.seed, output=args.output), ensure_ascii=False, indent=2))
