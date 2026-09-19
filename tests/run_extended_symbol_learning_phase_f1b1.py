"""Run the fixed-mechanics 1600-trial Phase F.1B.1 curve."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    SYMBOLS,
    SymbolInterfaceConfig,
    SymbolLearningConfig,
    SymbolLearningSession,
    load_malecns_v1,
)
from drosomath.malecns.symbol_learning_extended import (  # noqa: E402
    EXTENDED_CHECKPOINTS,
    boundary_crossings,
    channel_credit_telemetry,
    cumulative_plasticity_telemetry,
    extended_checkpoint_metrics,
    extended_symbol_schedule,
    margin_intervals,
)
from run_symbol_learning_phase_f1b import (  # noqa: E402
    _make_brain,
    _surface_match,
    build_f1b_dynamic_interface,
)


TRAINING_SEEDS = (41, 43, 47)
F1B_CHECKPOINT = 400


def _zero_channel_telemetry():
    return {
        f"symbol/{symbol}": {
            "positive_direction_requests": 0,
            "negative_direction_requests": 0,
            "edge_updates": 0,
            "sum_abs_multiplier_delta": 0.0,
            "1hop_updates": 0,
            "2hop_updates": 0,
            "unique_edges_modified": 0,
        }
        for symbol in SYMBOLS
    }


def _evaluate_isolated(session: SymbolLearningSession, *, seed: int):
    """Evaluate without leaving RNG changes that can affect later training."""
    rng_state = copy.deepcopy(session.brain.rng.bit_generator.state)
    try:
        return session.evaluate(seed=seed)
    finally:
        session.brain.rng.bit_generator.state = rng_state


def _config() -> tuple[SymbolInterfaceConfig, SymbolLearningConfig]:
    interface_config = SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=40.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )
    learning_config = SymbolLearningConfig(
        duration_ms=40.0,
        stimulus_rate_hz=205.0,
        training_trials=1600,
        evaluation_repetitions=10,
        directional_learning_rate=0.02,
        reward_learning_rate=0.02,
        plastic_fraction=0.05,
        adaptive_plastic_budget=False,
    )
    return interface_config, learning_config


def _run_seed(connectome, interface, interface_config, learning_config, seed):
    schedule = extended_symbol_schedule(seed=seed)
    training_brain = _make_brain(connectome, interface_config, seed)
    rng_before_baseline = copy.deepcopy(training_brain.rng.bit_generator.state)
    baseline_state = training_brain.plasticity.multiplier.copy()

    baseline_brain = _make_brain(connectome, interface_config, seed)
    baseline_session = SymbolLearningSession(baseline_brain, interface, config=learning_config)
    baseline_results = baseline_session.evaluate(seed=seed)
    baseline_twin_isolated = bool(
        training_brain.rng.bit_generator.state == rng_before_baseline
        and np.array_equal(training_brain.plasticity.multiplier, baseline_state)
    )
    session = SymbolLearningSession(training_brain, interface, config=learning_config)

    checkpoints: dict[str, dict[str, object]] = {
        "0": extended_checkpoint_metrics(baseline_results)
    }
    plasticity: dict[str, dict[str, object]] = {
        "0": cumulative_plasticity_telemetry(session, trial_count=0)
    }
    channel_credit: dict[str, dict[str, object]] = {
        "0": _zero_channel_telemetry()
    }

    for trial_number, target in enumerate(schedule, start=1):
        session.train_trial(target)
        if trial_number in EXTENDED_CHECKPOINTS[1:]:
            evaluation = _evaluate_isolated(session, seed=seed)
            key = str(trial_number)
            checkpoints[key] = extended_checkpoint_metrics(evaluation)
            plasticity[key] = cumulative_plasticity_telemetry(
                session, trial_count=trial_number
            )
            channel_credit[key] = channel_credit_telemetry(
                session.trial_results,
                unique_edge_counts=session.channel_unique_edges(),
            )

    schedule_400 = extended_symbol_schedule(seed=seed)[:400]
    f1b_schedule_same_prefix = schedule[:400] == schedule_400
    return {
        "seed": int(seed),
        "checkpoints": checkpoints,
        "margin_intervals": margin_intervals(checkpoints),
        "boundary_crossings": boundary_crossings(checkpoints),
        "plasticity": plasticity,
        "channel_credit": channel_credit,
        "invariants": {
            "schedule_length": int(len(schedule)),
            "schedule_count_per_symbol": {symbol: int(schedule.count(symbol)) for symbol in SYMBOLS},
            "checkpoints_exact": sorted(int(key) for key in checkpoints) == list(EXTENDED_CHECKPOINTS),
            "baseline_twin_isolated": baseline_twin_isolated,
            "f1b_schedule_same_prefix": f1b_schedule_same_prefix,
            "budget_fixed": all(row["budget_drift"] == 0 for row in plasticity.values()),
            "adaptive_reallocation_count": 0,
            "learning_config": {
                "duration_ms": learning_config.duration_ms,
                "stimulus_rate_hz": learning_config.stimulus_rate_hz,
                "directional_learning_rate": learning_config.directional_learning_rate,
                "reward_learning_rate": learning_config.reward_learning_rate,
                "adaptive_plastic_budget": learning_config.adaptive_plastic_budget,
            },
        },
    }


def _aggregate(runs):
    accuracy_curve = {}
    margin_curve = {}
    target_rank_curve = {}
    for checkpoint in EXTENDED_CHECKPOINTS:
        key = str(checkpoint)
        accuracy_curve[key] = float(np.mean([run["checkpoints"][key]["accuracy"] for run in runs]))
        margin_curve[key] = float(np.mean([
            run["checkpoints"][key]["target_minus_best_competitor_margin"]
            for run in runs
        ]))
        ranks = [run["checkpoints"][key]["target_rank"]["mean"] for run in runs]
        ranks = [rank for rank in ranks if rank is not None]
        target_rank_curve[key] = float(np.mean(ranks)) if ranks else None
    final = [run["checkpoints"]["1600"] for run in runs]
    return {
        "accuracy_curve": accuracy_curve,
        "macro_accuracy_curve": {
            str(checkpoint): float(np.mean([
                run["checkpoints"][str(checkpoint)]["macro_accuracy"] for run in runs
            ]))
            for checkpoint in EXTENDED_CHECKPOINTS
        },
        "margin_curve": margin_curve,
        "target_rank_curve": target_rank_curve,
        "final_per_seed_accuracy": {str(run["seed"]): run["checkpoints"]["1600"]["accuracy"] for run in runs},
        "final_per_seed_macro_accuracy": {str(run["seed"]): run["checkpoints"]["1600"]["macro_accuracy"] for run in runs},
        "final_per_symbol_accuracy": {
            symbol: float(np.mean([
                row["per_symbol"][symbol]["accuracy"] for row in final
            ]))
            for symbol in SYMBOLS
        },
        "final_per_symbol_margin": {
            symbol: float(np.mean([
                row["per_symbol"][symbol]["target_minus_best_competitor_margin"]
                for row in final
            ]))
            for symbol in SYMBOLS
        },
        "final_no_decision_fraction": float(np.mean([
            row["no_decision_fraction"] for row in final
        ])),
        "final_collapsed_seed_count": int(sum(row["collapsed"] for row in final)),
    }


def _conclusion(runs, aggregate, *, surface_match, schedule_match):
    final_accuracy = aggregate["accuracy_curve"]["1600"]
    final_rows = [run["checkpoints"]["1600"] for run in runs]
    no_collapse = aggregate["final_collapsed_seed_count"] == 0
    outcome_a = bool(
        final_accuracy >= 0.50
        and sum(row["accuracy"] >= 0.50 for row in final_rows) >= 2
        and no_collapse
    )
    seed_400_to_1600 = []
    for run in runs:
        movement = (
            run["checkpoints"]["1600"]["target_minus_best_competitor_margin"]
            - run["checkpoints"]["400"]["target_minus_best_competitor_margin"]
        )
        seed_400_to_1600.append(movement)
    outcome_b = bool(
        not outcome_a
        and aggregate["margin_curve"]["1600"] > aggregate["margin_curve"]["400"]
        and sum(value > 0.0 for value in seed_400_to_1600) >= 2
    )
    interval_labels = [
        run["margin_intervals"]["1200->1600"]["classification"] for run in runs
    ]
    plateau = bool(
        not outcome_a
        and not outcome_b
        and aggregate["margin_curve"]["1200"] >= aggregate["margin_curve"]["400"]
        and all(label != "still_improving" for label in interval_labels)
    )
    continuous_effect = bool(
        aggregate["margin_curve"]["1600"] > aggregate["margin_curve"]["0"]
    )
    if outcome_a:
        recommendation = "sequence_working_memory"
    elif outcome_b:
        recommendation = "diagnose_learning_efficiency_before_sequence"
    else:
        recommendation = "diagnose_symbol_credit_assignment"
    return {
        "continuous_learning_signal_effect": continuous_effect,
        "discrete_symbol_mapping_learned": outcome_a,
        "learning_plateau_detected": plateau,
        "output_collapse_detected": not no_collapse,
        "recommended_next_step": recommendation,
        "outcome_class": "A" if outcome_a else ("B" if outcome_b else "C"),
        "mean_400_to_1600_margin_deltas": {
            str(run["seed"]): float(value) for run, value in zip(runs, seed_400_to_1600)
        },
        "surface_fingerprint_identical": bool(surface_match),
        "training_schedule_semantics_identical": bool(schedule_match),
    }


def run(*, data_dir: Path, output_path: Path, f1a3_artifact: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config, learning_config = _config()
    interface, candidate_count, _selected = build_f1b_dynamic_interface(
        connectome, interface_config
    )
    surface_match = _surface_match(interface, f1a3_artifact)
    runs = [
        _run_seed(connectome, interface, interface_config, learning_config, seed)
        for seed in TRAINING_SEEDS
    ]
    aggregate = _aggregate(runs)
    schedule_match = all(
        run["invariants"]["f1b_schedule_same_prefix"] for run in runs
    )
    protocol_pass = bool(
        surface_match
        and schedule_match
        and all(run["invariants"]["schedule_length"] == 1600 for run in runs)
        and all(all(value == 400 for value in run["invariants"]["schedule_count_per_symbol"].values()) for run in runs)
        and all(run["invariants"]["checkpoints_exact"] for run in runs)
        and all(run["invariants"]["baseline_twin_isolated"] for run in runs)
        and all(run["invariants"]["budget_fixed"] for run in runs)
        and all(run["invariants"]["adaptive_reallocation_count"] == 0 for run in runs)
        and not learning_config.adaptive_plastic_budget
    )
    artifact = {
        "phase": "F.1B.1",
        "protocol": {
            "training_trials_per_seed": 1600,
            "trials_per_symbol": 400,
            "checkpoints": list(EXTENDED_CHECKPOINTS),
            "evaluation_presentations_per_symbol": 10,
            "evaluation_total_presentations": 40,
            "learning_disabled_during_evaluation": True,
            "evaluation_rng_isolated": True,
            "training_seeds": list(TRAINING_SEEDS),
            "dynamic_surface": "F.1A.3 exact deterministic reconstruction",
            "dynamic_candidate_count": int(candidate_count),
            "duration_ms": learning_config.duration_ms,
            "stimulus_rate_hz": learning_config.stimulus_rate_hz,
            "directional_learning_rate": learning_config.directional_learning_rate,
            "reward_learning_rate": learning_config.reward_learning_rate,
            "adaptive_plastic_budget": False,
            "external_decoder": False,
            "keyboard_teacher": False,
            "structural_reallocation": False,
        },
        "surface": {
            "matches_f1a3_artifact": surface_match,
            "frozen_before_learning": True,
        },
        "runs": {str(run["seed"]): run for run in runs},
        "aggregate": aggregate,
        "conclusion": _conclusion(
            runs,
            aggregate,
            surface_match=surface_match,
            schedule_match=schedule_match,
        ),
        "pass": protocol_pass,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest_extended_symbol_learning_phase_f1b1.json"),
    )
    parser.add_argument(
        "--f1a3-artifact",
        type=Path,
        default=Path("results/latest_dynamic_decision_surface_phase_f1a3.json"),
    )
    args = parser.parse_args()
    result = run(
        data_dir=args.data_dir,
        output_path=args.output,
        f1a3_artifact=args.f1a3_artifact,
    )
    print(json.dumps({
        "pass": result["pass"],
        "conclusion": result["conclusion"],
        "accuracy_curve": result["aggregate"]["accuracy_curve"],
        "margin_curve": result["aggregate"]["margin_curve"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
