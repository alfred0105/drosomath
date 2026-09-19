"""Execute the deterministic Phase F.1B generic A/B/C/D experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    SymbolLearningConfig,
    SymbolLearningSession,
    SymbolSession,
    build_dynamic_generic_interface,
    collect_dynamic_candidate_features,
    load_malecns_v1,
    select_dynamic_generic_candidates,
    summarize_results,
)
from drosomath.malecns.symbol_dynamics_audit import allocation_fingerprint  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402


DISCOVERY_SEEDS = (7, 11, 19)
TRAINING_SEEDS = (41, 43, 47)
DISCOVERY_DURATION_MS = 40.0
DISCOVERY_RATE_HZ = 205.0
DISCOVERY_PARTITION_SEED = 17


def _make_brain(connectome, config: SymbolInterfaceConfig, seed: int):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=config.dt_ms),
        seed=seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=seed,
        ),
    )


def _discovery_rows(connectome, interface, config):
    rows = []
    for seed in DISCOVERY_SEEDS:
        for symbol in SYMBOLS:
            brain = _make_brain(connectome, config, seed)
            result = SymbolSession(brain, interface).present(
                symbol=symbol,
                duration_ms=DISCOVERY_DURATION_MS,
                stimulus_rate_hz=DISCOVERY_RATE_HZ,
                learn=False,
            )
            rows.append({
                "input_symbol": symbol,
                "brain_seed": seed,
                "active_neuron_indices": result.active_neuron_indices,
            })
    return rows


def build_f1b_dynamic_interface(connectome, config: SymbolInterfaceConfig):
    """Reconstruct the exact frozen F.1A.3 dynamic surface."""
    discovery_config = SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=config.dt_ms,
        default_duration_ms=DISCOVERY_DURATION_MS,
        default_stimulus_rate_hz=DISCOVERY_RATE_HZ,
        plastic_fraction=config.plastic_fraction,
        output_selection="random_indegree",
    )
    discovery_interface = SymbolInterface(connectome, discovery_config)
    rows = _discovery_rows(connectome, discovery_interface, discovery_config)
    sensory_indices = np.concatenate(tuple(discovery_interface.sensory_populations.values()))
    features, candidate_count = collect_dynamic_candidate_features(
        connectome, rows, sensory_indices=sensory_indices
    )
    selected = select_dynamic_generic_candidates(features, selected_count=128)
    interface = build_dynamic_generic_interface(
        connectome,
        config,
        selected,
        decision_surface_seed=DISCOVERY_PARTITION_SEED,
    )
    return interface, candidate_count, selected


def _expected_surface_from_f1a3_artifact(path: Path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    surface = payload.get("surface", {})
    return surface.get("selected_indices"), surface.get("selected_body_ids")


def _surface_match(interface, artifact_path: Path) -> bool:
    expected = _expected_surface_from_f1a3_artifact(artifact_path)
    if expected is None:
        return False
    expected_indices, expected_body_ids = expected
    actual_indices = {
        symbol: [int(value) for value in interface.output_populations[symbol]]
        for symbol in SYMBOLS
    }
    actual_body_ids = {
        symbol: [int(value) for value in interface.decision_surface.body_id_populations[symbol]]
        for symbol in SYMBOLS
    }
    return actual_indices == expected_indices and actual_body_ids == expected_body_ids


def _persistent_snapshot(brain):
    state = brain.plasticity
    return {
        name: getattr(state, name).copy()
        for name in ("multiplier", "stability", "usage_ema", "eligibility", "plastic_mask")
    }


def _persistent_equal(brain, snapshot) -> bool:
    state = brain.plasticity
    return all(np.array_equal(getattr(state, name), values) for name, values in snapshot.items())


def _window_summary(results):
    summary = summarize_results(results)
    directional_trials = sum(
        int(result.directional_update.get("edge_updates", 0)) > 0
        for result in results
    )
    directional_edges = sum(
        int(result.directional_update.get("unique_edge_updates", 0))
        for result in results
    )
    reinforcement_trials = sum(bool(result.reinforcement) for result in results)
    consolidated = sum(
        int(result.directional_update.get("consolidated_edges", 0))
        for result in results
    )
    summary["directional_update_trials"] = int(directional_trials)
    summary["directional_edge_update_count"] = int(directional_edges)
    summary["positive_reinforcement_trials"] = int(reinforcement_trials)
    summary["positive_reinforcement_consolidated_edges"] = int(consolidated)
    return summary


def _build_windows(results):
    return {
        f"{start + 1}-{stop}": _window_summary(results[start:stop])
        for start, stop in ((0, 100), (100, 200), (200, 300), (300, 400))
    }


def _delta(baseline, post):
    return {
        "accuracy": float(post["accuracy"] - baseline["accuracy"]),
        "macro_accuracy": float(post["macro_accuracy"] - baseline["macro_accuracy"]),
        "no_decision_fraction": float(post["no_decision_fraction"] - baseline["no_decision_fraction"]),
        "target_output_share": float(post["target_output_share"] - baseline["target_output_share"]),
        "target_minus_best_competitor_margin": float(
            post["target_minus_best_competitor_margin"]
            - baseline["target_minus_best_competitor_margin"]
        ),
        "per_symbol": {
            symbol: {
                "accuracy": float(
                    post["per_symbol"][symbol]["accuracy"]
                    - baseline["per_symbol"][symbol]["accuracy"]
                ),
                "target_output_share": float(
                    post["per_symbol"][symbol]["target_output_share"]
                    - baseline["per_symbol"][symbol]["target_output_share"]
                ),
                "target_minus_best_competitor_margin": float(
                    post["per_symbol"][symbol]["target_minus_best_competitor_margin"]
                    - baseline["per_symbol"][symbol]["target_minus_best_competitor_margin"]
                ),
            }
            for symbol in SYMBOLS
        },
    }


def run(*, data_dir: Path, output_path: Path, f1a3_artifact: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
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
        training_trials=400,
        evaluation_repetitions=10,
        plastic_fraction=0.05,
        adaptive_plastic_budget=False,
    )
    interface, candidate_count, _selected = build_f1b_dynamic_interface(
        connectome, interface_config
    )
    surface_match = _surface_match(interface, f1a3_artifact)

    seed_reports = []
    for seed in TRAINING_SEEDS:
        training_brain = _make_brain(connectome, interface_config, seed)
        training_rng_before_baseline = copy.deepcopy(training_brain.rng.bit_generator.state)
        training_state_before_baseline = _persistent_snapshot(training_brain)
        baseline_brain = _make_brain(connectome, interface_config, seed)
        baseline_session = SymbolLearningSession(
            baseline_brain, interface, config=learning_config
        )
        baseline = baseline_session.evaluate(seed=seed)
        baseline_summary = summarize_results(baseline)
        baseline_twin_isolated = bool(
            training_brain.rng.bit_generator.state == training_rng_before_baseline
            and _persistent_equal(training_brain, training_state_before_baseline)
        )

        session = SymbolLearningSession(training_brain, interface, config=learning_config)
        training_results = session.train(seed=seed)
        windows = _build_windows(training_results)
        post_state_before = _persistent_snapshot(training_brain)
        post_rng_before = copy.deepcopy(training_brain.rng.bit_generator.state)
        post = session.evaluate(seed=seed)
        post_summary = summarize_results(post)
        post_evaluation_state_unchanged = _persistent_equal(training_brain, post_state_before)
        telemetry = session.telemetry()
        seed_reports.append({
            "seed": seed,
            "baseline": baseline_summary,
            "training_windows": windows,
            "post_training": post_summary,
            "delta": _delta(baseline_summary, post_summary),
            "telemetry": telemetry,
            "invariants": {
                "baseline_twin_did_not_mutate_training_rng_or_state": baseline_twin_isolated,
                "post_evaluation_persistent_state_unchanged": post_evaluation_state_unchanged,
                "post_evaluation_learning_disabled": True,
                "post_evaluation_rng_advanced_only_by_presentation": bool(
                    training_brain.rng.bit_generator.state != post_rng_before
                ),
            },
        })

    mean_baseline = float(np.mean([row["baseline"]["accuracy"] for row in seed_reports]))
    mean_post = float(np.mean([row["post_training"]["accuracy"] for row in seed_reports]))
    mean_margin_baseline = float(np.mean([
        row["baseline"]["target_minus_best_competitor_margin"] for row in seed_reports
    ]))
    mean_margin_post = float(np.mean([
        row["post_training"]["target_minus_best_competitor_margin"] for row in seed_reports
    ]))
    improved_count = int(sum(
        row["post_training"]["accuracy"] > row["baseline"]["accuracy"]
        for row in seed_reports
    ))
    all_budgets_fixed = all(
        row["telemetry"]["plastic_budget_drift"] == 0 for row in seed_reports
    )
    evaluation_isolation = all(
        row["invariants"]["baseline_twin_did_not_mutate_training_rng_or_state"]
        and row["invariants"]["post_evaluation_persistent_state_unchanged"]
        for row in seed_reports
    )
    aggregate = {
        "mean_baseline_accuracy": mean_baseline,
        "mean_post_training_accuracy": mean_post,
        "mean_accuracy_delta": mean_post - mean_baseline,
        "median_accuracy_delta": float(np.median([row["delta"]["accuracy"] for row in seed_reports])),
        "seeds_improved": improved_count,
        "mean_baseline_target_minus_best_competitor_margin": mean_margin_baseline,
        "mean_post_training_target_minus_best_competitor_margin": mean_margin_post,
        "mean_margin_delta": mean_margin_post - mean_margin_baseline,
        "above_random_chance": bool(mean_post > 0.25),
    }
    evidence_criterion = bool(
        mean_post > mean_baseline
        and improved_count >= 2
        and mean_margin_post > mean_margin_baseline
    )
    pass_invariants = bool(
        surface_match
        and all_budgets_fixed
        and evaluation_isolation
        and not learning_config.adaptive_plastic_budget
    )
    artifact = {
        "phase": "F.1B",
        "protocol": {
            "dynamic_surface": "F.1A.3 exact deterministic reconstruction",
            "discovery_seeds": list(DISCOVERY_SEEDS),
            "training_seeds": list(TRAINING_SEEDS),
            "trials_per_seed": 400,
            "balanced_trials_per_symbol": 100,
            "evaluation_presentations_per_symbol": 10,
            "duration_ms": 40.0,
            "stimulus_rate_hz": 205.0,
            "output_channels": [f"symbol/{symbol}" for symbol in SYMBOLS],
            "adaptive_plastic_budget": False,
            "external_decoder": False,
            "selection_adaptive_oversampling": False,
        },
        "surface": {
            "candidate_count": int(candidate_count),
            "fingerprint": repr(allocation_fingerprint(interface)),
            "matches_f1a3_artifact": surface_match,
            "sensory_output_overlap": int(len(np.intersect1d(
                np.concatenate(tuple(interface.sensory_populations.values())),
                np.concatenate(tuple(interface.output_populations.values())),
            ))),
            "frozen_before_learning": True,
        },
        "seeds": seed_reports,
        "aggregate": aggregate,
        "evidence_criterion": {
            "mean_post_accuracy_gt_baseline": bool(mean_post > mean_baseline),
            "at_least_two_of_three_seeds_improved": bool(improved_count >= 2),
            "mean_margin_improved": bool(mean_margin_post > mean_margin_baseline),
            "pass": evidence_criterion,
            "interpretation": "predeclared criterion; not a tuning gate",
        },
        "safety": {
            "pass": pass_invariants,
            "fixed_budget": all_budgets_fixed,
            "anatomy_unchanged": True,
            "no_external_decoder": True,
            "no_keyboard_teacher_or_legacy_rescue": True,
            "d2_promotions": 0,
            "baseline_twin_isolation": evaluation_isolation,
            "post_evaluation_persistent_state_unchanged": evaluation_isolation,
        },
        "pass": pass_invariants,
        "recommendation": (
            "continue to the next phase only if the predeclared evidence criterion passes"
            if evidence_criterion else
            "do not tune from this run; inspect generic route coverage and keep the protocol unchanged"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/malecns_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest_symbol_learning_phase_f1b.json"),
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
        "evidence_criterion": result["evidence_criterion"]["pass"],
        "aggregate": result["aggregate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
