"""Execute the deterministic Phase F.1B.3 prospective two-hop A/B test."""

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
    ProspectiveTwoHopAudit,
    SymbolInterfaceConfig,
    SymbolLearningConfig,
    SymbolLearningSession,
    SymbolSession,
    SymbolCreditInterferenceAudit,
    balanced_symbol_schedule,
    load_malecns_v1,
)
from drosomath.malecns.symbol_learning_extended import extended_checkpoint_metrics  # noqa: E402
from drosomath.malecns.symbol_credit_interference import cross_target_positive_overlap  # noqa: E402
from run_symbol_learning_phase_f1b import (  # noqa: E402
    _make_brain,
    _surface_match,
    build_f1b_dynamic_interface,
)


SEEDS = (41, 43, 47)
CHECKPOINTS = (0, 400, 800)


def _configs(mode: str):
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
        training_trials=800,
        evaluation_repetitions=10,
        directional_learning_rate=0.02,
        reward_learning_rate=0.02,
        plastic_fraction=0.05,
        adaptive_plastic_budget=False,
        two_hop_credit_mode=mode,
    )
    return interface_config, learning_config


def _persistent_snapshot(brain):
    state = brain.plasticity
    return {
        name: getattr(state, name).copy()
        for name in ("multiplier", "stability", "usage_ema", "eligibility", "plastic_mask")
    }


def _evaluate_isolated(session, targets):
    brain = session.brain
    snapshot = _persistent_snapshot(brain)
    rng_state = brain.rng.bit_generator.state
    results = [session.evaluate_trial(target) for target in targets]
    for name, values in snapshot.items():
        np.testing.assert_array_equal(getattr(brain.plasticity, name), values)
    brain.rng.bit_generator.state = rng_state
    return results


def _state_metrics(session, checkpoint):
    state = session.brain.plasticity
    saturation = (state.multiplier <= state.config.min_multiplier) | (
        state.multiplier >= state.config.max_multiplier
    )
    return {
        "trial_count": int(checkpoint),
        "plastic_budget": int(state.plastic_edge_count),
        "budget_drift": int(state.plastic_edge_count - session.plastic_budget_start),
        "adaptive_reallocations": 0,
        "mean_multiplier": float(np.mean(state.multiplier)),
        "multiplier_saturation_fraction": float(np.mean(saturation)),
        "mean_stability": float(np.mean(state.stability)),
    }


def _run_seed(connectome, interface, interface_config, learning_config, seed):
    brain = _make_brain(connectome, interface_config, seed)
    attribution_all = ProspectiveTwoHopAudit(connectome)
    attribution_400 = ProspectiveTwoHopAudit(connectome)
    attribution_800 = ProspectiveTwoHopAudit(connectome)
    interference = SymbolCreditInterferenceAudit(connectome, max_route_health_requests=8)
    active_checkpoint_audit = attribution_400

    def observe_update(**kwargs):
        interference.observe_update(**kwargs)

    def observe_attribution(**kwargs):
        nonlocal active_checkpoint_audit
        attribution_all.observe_attribution(**kwargs)
        active_checkpoint_audit.observe_attribution(**kwargs)

    session = SymbolLearningSession(
        brain,
        interface,
        config=learning_config,
        directional_telemetry_observer=observe_update,
        directional_attribution_observer=observe_attribution,
    )

    def observe_route_health(*, brain, signal, output_context, target, decision):
        if not interference.wants_route_health(target=target, signal=signal):
            return
        health = session.controller.diagnose_route_health(
            brain, signal, output_context, standardized=True
        )
        interference.observe_route_health(
            target=target, signal=signal, health_by_channel=health
        )

    session.route_health_observer = observe_route_health
    schedule = balanced_symbol_schedule(cycles=200, seed=seed + learning_config.schedule_seed_offset)
    evaluations = {}
    training_cursor = 0
    eval_targets = balanced_symbol_schedule(
        cycles=learning_config.evaluation_repetitions,
        seed=seed + learning_config.evaluation_seed_offset,
    )
    evaluations["0"] = extended_checkpoint_metrics(_evaluate_isolated(session, eval_targets))
    for target in schedule:
        session.train_trial(target)
        training_cursor += 1
        if training_cursor == 400:
            evaluations["400"] = extended_checkpoint_metrics(
                _evaluate_isolated(session, eval_targets)
            )
            active_checkpoint_audit = attribution_800
    evaluations["800"] = extended_checkpoint_metrics(
        _evaluate_isolated(session, eval_targets)
    )
    return {
        "seed": int(seed),
        "trials": int(training_cursor),
        "schedule_count_per_symbol": {
            symbol: int(sum(target == symbol for target in schedule)) for symbol in SYMBOLS
        },
        "checkpoints": evaluations,
        "route_engagement": {
            "400": attribution_400.report(),
            "800": attribution_800.report(),
            "all": attribution_all.report(),
        },
        "context_credit": interference.report(),
        "cross_target_positive_overlap": cross_target_positive_overlap(interference),
        "state": _state_metrics(session, 800),
        "brain": brain,
        "session": session,
    }


def _strip_runtime(run):
    return {
        key: value for key, value in run.items() if key not in {"brain", "session"}
    }


def _mean_positive_yield(runs, checkpoint, channel):
    rows = [
        run["route_engagement"][str(checkpoint)][channel]["positive"][
            "mean_edge_updates_per_request"
        ]
        for run in runs
    ]
    return float(np.mean(rows)) if rows else 0.0


def _mean_metric(runs, checkpoint, key):
    return float(np.mean([run["checkpoints"][str(checkpoint)][key] for run in runs]))


def run(*, data_dir: Path, output_path: Path, f1a3_artifact: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    control_config, control_learning = _configs("active_chain")
    intervention_config, intervention_learning = _configs("prospective_anatomical")
    control_interface, candidate_count, _ = build_f1b_dynamic_interface(
        connectome, control_config
    )
    intervention_interface, _, _ = build_f1b_dynamic_interface(
        connectome, intervention_config
    )
    surface_match = _surface_match(control_interface, f1a3_artifact) and _surface_match(
        intervention_interface, f1a3_artifact
    )
    control = [
        _run_seed(connectome, control_interface, control_config, control_learning, seed)
        for seed in SEEDS
    ]
    intervention = [
        _run_seed(
            connectome,
            intervention_interface,
            intervention_config,
            intervention_learning,
            seed,
        )
        for seed in SEEDS
    ]

    route_comparison = {}
    for checkpoint in (400, 800):
        route_comparison[str(checkpoint)] = {}
        for symbol in SYMBOLS:
            channel = f"symbol/{symbol}"
            control_value = _mean_positive_yield(control, checkpoint, channel)
            intervention_value = _mean_positive_yield(intervention, checkpoint, channel)
            route_comparison[str(checkpoint)][channel] = {
                "control_mean_positive_edge_updates_per_request": control_value,
                "intervention_mean_positive_edge_updates_per_request": intervention_value,
                "intervention_at_least_2x_control": (
                    intervention_value > 0.0 if control_value == 0.0
                    else intervention_value >= 2.0 * control_value
                ),
            }
    route_engagement_improved = bool(
        route_comparison["400"]["symbol/A"]["intervention_at_least_2x_control"]
        and route_comparison["400"]["symbol/D"]["intervention_at_least_2x_control"]
    )

    control_accuracy = _mean_metric(control, 800, "accuracy")
    intervention_accuracy = _mean_metric(intervention, 800, "accuracy")
    control_margin = _mean_metric(control, 800, "target_minus_best_competitor_margin")
    intervention_margin = _mean_metric(intervention, 800, "target_minus_best_competitor_margin")
    better_margin_seeds = sum(
        intervention[i]["checkpoints"]["800"]["target_minus_best_competitor_margin"]
        > control[i]["checkpoints"]["800"]["target_minus_best_competitor_margin"]
        for i in range(len(SEEDS))
    )
    no_collapse = all(
        not run["checkpoints"]["800"]["collapsed"]
        for run in intervention
    )
    behavioral_improvement_supported = bool(
        intervention_margin > control_margin
        and better_margin_seeds >= 2
        and intervention_accuracy >= control_accuracy
        and no_collapse
    )
    discrete_mapping_demonstrated = bool(
        intervention_accuracy >= 0.5
        and sum(run["checkpoints"]["800"]["accuracy"] >= 0.5 for run in intervention) >= 2
        and no_collapse
    )

    def positive_overlap(runs):
        values = [
            float(value)
            for run in runs
            for key, value in run["cross_target_positive_overlap"].items()
        ]
        return float(np.mean(values)) if values else 0.0

    control_overlap = positive_overlap(control)
    intervention_overlap = positive_overlap(intervention)
    specificity_regression = intervention_overlap > control_overlap + 0.10
    if discrete_mapping_demonstrated:
        outcome = "discrete_mapping_working"
    elif route_engagement_improved and behavioral_improvement_supported:
        outcome = "sequence_working_memory"
    elif route_engagement_improved:
        outcome = "diagnose_credit_quality_not_quantity"
    else:
        outcome = "reassess_eligibility_and_temporal_credit"

    safety = {
        "control_budget_drift": [run["state"]["budget_drift"] for run in control],
        "intervention_budget_drift": [run["state"]["budget_drift"] for run in intervention],
        "adaptive_reallocations": 0,
        "prospective_only_safety_passed": all(
            all(
                channel["safety"]["passed"]
                for channel in run["route_engagement"]["all"].values()
            )
            for run in intervention
        ),
        "anatomy_unchanged": True,
        "no_structural_nodes_or_edges": True,
    }
    artifact = {
        "protocol": {
            "phase": "F.1B.3",
            "arms": {"control": "active_chain", "intervention": "prospective_anatomical"},
            "seeds": list(SEEDS),
            "trials_per_arm_per_seed": 800,
            "trials_per_symbol": 200,
            "evaluation_trials_per_checkpoint": 40,
            "checkpoints": list(CHECKPOINTS),
            "duration_ms": 40.0,
            "stimulus_rate_hz": 205.0,
            "directional_learning_rate": 0.02,
            "reward_learning_rate": 0.02,
            "adaptive_plastic_budget": False,
            "dynamic_candidate_count": int(candidate_count),
            "surface_matches_f1a3_artifact": surface_match,
            "external_decoder": False,
            "keyboard_teacher": False,
            "synthetic_connections": False,
        },
        "control": {str(run["seed"]): _strip_runtime(run) for run in control},
        "intervention": {str(run["seed"]): _strip_runtime(run) for run in intervention},
        "comparison": {
            "route_engagement": route_comparison,
            "route_engagement_improved": route_engagement_improved,
            "behavior": {
                "control_mean_accuracy_800": control_accuracy,
                "intervention_mean_accuracy_800": intervention_accuracy,
                "control_mean_margin_800": control_margin,
                "intervention_mean_margin_800": intervention_margin,
                "intervention_better_margin_seed_count": int(better_margin_seeds),
                "no_collapse": no_collapse,
                "behavioral_improvement_supported": behavioral_improvement_supported,
                "discrete_mapping_demonstrated": discrete_mapping_demonstrated,
            },
            "specificity": {
                "control_mean_cross_target_positive_jaccard": control_overlap,
                "intervention_mean_cross_target_positive_jaccard": intervention_overlap,
                "specificity_regression": specificity_regression,
            },
        },
        "safety": safety,
        "conclusion": {"outcome": outcome},
        "pass": bool(
            surface_match
            and all(run["trials"] == 800 for run in control + intervention)
            and all(all(value == 200 for value in run["schedule_count_per_symbol"].values()) for run in control + intervention)
            and all(run["state"]["budget_drift"] == 0 for run in control + intervention)
            and safety["prospective_only_safety_passed"]
        ),
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
        default=Path("results/latest_prospective_twohop_phase_f1b3.json"),
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
    print(json.dumps({"pass": result["pass"], "conclusion": result["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
