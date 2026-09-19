"""Execute the diagnostic-only Phase F.1B.2 credit-interference audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from drosomath.malecns import (  # noqa: E402
    SYMBOLS,
    SymbolInterfaceConfig,
    SymbolLearningConfig,
    SymbolLearningSession,
    load_malecns_v1,
)
from drosomath.malecns.symbol_credit_interference import (  # noqa: E402
    SymbolCreditInterferenceAudit,
    classify_primary_hypotheses,
    cross_target_positive_overlap,
)
from run_symbol_learning_phase_f1b import (  # noqa: E402
    _make_brain,
    _surface_match,
    build_f1b_dynamic_interface,
)


SEEDS = (41, 43, 47)


def _configs():
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
        directional_learning_rate=0.02,
        reward_learning_rate=0.02,
        plastic_fraction=0.05,
        adaptive_plastic_budget=False,
    )
    return interface_config, learning_config


def _run_seed(connectome, interface, interface_config, learning_config, seed):
    brain = _make_brain(connectome, interface_config, seed)
    audit = SymbolCreditInterferenceAudit(connectome, max_route_health_requests=8)

    def observe_route_health(*, brain, signal, output_context, target, decision):
        if not audit.wants_route_health(target=target, signal=signal):
            return
        health = session.controller.diagnose_route_health(
            brain,
            signal,
            output_context,
            standardized=True,
        )
        audit.observe_route_health(
            target=target,
            signal=signal,
            health_by_channel=health,
        )

    session = SymbolLearningSession(
        brain,
        interface,
        config=learning_config,
        directional_telemetry_observer=audit.observe_update,
        route_health_observer=observe_route_health,
    )
    schedule = session.train(seed=seed)
    report = audit.report()
    return {
        "seed": int(seed),
        "trials": int(len(schedule)),
        "schedule_count_per_symbol": {
            symbol: int(sum(result.target == symbol for result in schedule))
            for symbol in SYMBOLS
        },
        "channels": report,
        "plastic_budget_start": int(session.plastic_budget_start),
        "plastic_budget_end": int(brain.plasticity.plastic_edge_count),
        "budget_drift": int(brain.plasticity.plastic_edge_count - session.plastic_budget_start),
        "adaptive_reallocations": 0,
        "brain": brain,
        "audit": audit,
    }


def _strip_runtime(run):
    return {
        key: value
        for key, value in run.items()
        if key not in {"brain", "audit"}
    }


def run(*, data_dir: Path, output_path: Path, f1a3_artifact: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config, learning_config = _configs()
    interface, candidate_count, _selected = build_f1b_dynamic_interface(
        connectome, interface_config
    )
    surface_match = _surface_match(interface, f1a3_artifact)
    runs = [
        _run_seed(connectome, interface, interface_config, learning_config, seed)
        for seed in SEEDS
    ]
    combined = SymbolCreditInterferenceAudit(connectome, max_route_health_requests=8)
    for run_result in runs:
        combined.absorb(run_result["audit"])
    channels = combined.report()
    hypotheses = classify_primary_hypotheses(channels)
    protocol_pass = bool(
        surface_match
        and all(run_result["trials"] == 400 for run_result in runs)
        and all(all(value == 100 for value in run_result["schedule_count_per_symbol"].values()) for run_result in runs)
        and all(run_result["budget_drift"] == 0 for run_result in runs)
        and all(run_result["adaptive_reallocations"] == 0 for run_result in runs)
        and not learning_config.adaptive_plastic_budget
    )
    artifact = {
        "protocol": {
            "phase": "F.1B.2",
            "seeds": list(SEEDS),
            "trials_per_seed": 400,
            "trials_per_symbol": 100,
            "duration_ms": 40.0,
            "stimulus_rate_hz": 205.0,
            "directional_learning_rate": 0.02,
            "reward_learning_rate": 0.02,
            "adaptive_plastic_budget": False,
            "behavior_modified": False,
            "dynamic_surface": "F.1A.3 exact deterministic reconstruction",
            "dynamic_candidate_count": int(candidate_count),
            "external_decoder": False,
            "keyboard_teacher": False,
            "structural_reallocation": False,
            "route_health_samples_per_context_per_seed": 8,
        },
        "surface": {
            "matches_f1a3_artifact": surface_match,
            "frozen_before_learning": True,
        },
        "channels": channels,
        "per_seed": {str(run_result["seed"]): _strip_runtime(run_result) for run_result in runs},
        "cross_target_positive_overlap": cross_target_positive_overlap(combined),
        "conclusion": {
            **hypotheses,
            "description": {
                "A": "positive requests with weak realized generic update yield",
                "D": "same channel receives sign-conflicting target/competitor credit",
            },
        },
        "safety": {
            "learning_algorithm_changed": False,
            "learning_rates_changed": False,
            "decision_surface_changed": False,
            "rng_or_state_mutated_by_diagnostics": False,
            "final_multiplier_or_stability_changed_by_diagnostics": False,
            "plastic_budget_changed_by_diagnostics": False,
        },
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
        default=Path("results/latest_symbol_credit_interference_phase_f1b2.json"),
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
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
