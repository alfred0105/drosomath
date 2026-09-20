"""Run a clean, same-seed real MaleCNS keyboard A/B for Phase D.2.1."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drosomath.malecns.keyboard_learning import KeyboardTrainingConfig, run_keyboard_training
from drosomath.malecns.loader import load_malecns_v1


def _run(connectome, *, adaptive: bool, trials: int, seed: int, root: Path) -> dict[str, object]:
    config = KeyboardTrainingConfig(
        trials=trials,
        seed=seed,
        resume=False,
        adaptive_plastic_budget=adaptive,
        checkpoint_every=trials,
        retention_probe_interval=trials + 1,
        dashboard_update_interval_seconds=3600.0,
    )
    label = "adaptive_on" if adaptive else "adaptive_off"
    report = run_keyboard_training(
        connectome,
        config=config,
        result_path=root / f"{label}.json",
        progress_path=root / f"{label}_progress.json",
        checkpoint_path=root / f"{label}.npz",
        html_path=root / f"{label}.html",
    )
    budget = dict(report["adaptive_budget"])
    return {
        "accuracy": float(report["accuracy"]),
        "training_accuracy": float(report["training_accuracy"]),
        "macro_recent_accuracy": float(report["macro_recent_accuracy"]),
        "median_recent_accuracy": float(report["median_recent_accuracy"]),
        "minimum_recent_accuracy": float(report["minimum_recent_accuracy"]),
        "per_key_recent_accuracy": {
            key: float(value["recent_accuracy"])
            for key, value in report["per_key"].items()
        },
        "directional_generic_update_count": int(budget["directional_generic_update_count"]),
        "localized_positive_reward_updated_edge_count": int(
            budget["localized_positive_reward_updated_edge_count"]
        ),
        "legacy_rescue_events": int(budget["legacy_rescue_events"]),
        "mean_stability": float(report["final_plasticity"]["mean_stability"]),
        "plastic_budget_start": int(budget["plastic_budget_start"]),
        "plastic_budget_end": int(budget["plastic_budget_end"]),
        "budget_delta": int(budget["budget_delta"]),
        "adaptive_telemetry": budget,
        "completed_trials": int(report["completed_trials"]),
        "resume": report["resume"],
    }


def run(*, data_dir: Path, trials: int, seed: int, output: Path) -> dict[str, object]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    # Temporary run files ensure both arms are clean and never resume a prior
    # checkpoint. The compact, generated comparison is the committed artifact.
    with tempfile.TemporaryDirectory(prefix="drosomath_d21_ab_") as temporary:
        root = Path(temporary)
        adaptive_off = _run(connectome, adaptive=False, trials=trials, seed=seed, root=root)
        adaptive_on = _run(connectome, adaptive=True, trials=trials, seed=seed, root=root)

    off_budget = adaptive_off["adaptive_telemetry"]
    on_budget = adaptive_on["adaptive_telemetry"]
    result = {
        "protocol": {
            "seed": seed,
            "trials": trials,
            "resume": False,
            "connectome": "MaleCNS v1 min_connection_synapses=5",
            "same_config_except_adaptive_plastic_budget": True,
        },
        "adaptive_off": adaptive_off,
        "adaptive_on": adaptive_on,
        "comparison": {
            "accuracy_delta_on_minus_off": adaptive_on["accuracy"] - adaptive_off["accuracy"],
            "macro_recent_accuracy_delta_on_minus_off": (
                adaptive_on["macro_recent_accuracy"] - adaptive_off["macro_recent_accuracy"]
            ),
            "legacy_rescue_rate_off": adaptive_off["legacy_rescue_events"] / max(1, trials),
            "legacy_rescue_rate_on": adaptive_on["legacy_rescue_events"] / max(1, trials),
            "legacy_rescue_delta_on_minus_off": (
                adaptive_on["legacy_rescue_events"] - adaptive_off["legacy_rescue_events"]
            ),
        },
        "pass": bool(
            adaptive_off["completed_trials"] == trials
            and adaptive_on["completed_trials"] == trials
            and adaptive_off["resume"] is None
            and adaptive_on["resume"] is None
            and adaptive_on["budget_delta"] == 0
            and int(on_budget["protected_edges_retired"]) == 0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase D.2.1 real MaleCNS keyboard A/B")
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest_adaptive_budget_phase_d21_ab.json"),
    )
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, trials=args.trials, seed=args.seed, output=args.output), ensure_ascii=False, indent=2))
