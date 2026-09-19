"""Run the P.1 baseline and the held-out F.1B.4 replication.

The benchmark is intentionally separate from the scientific replication so
its diagnostic observers stay disabled and its timing protocol is explicit.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
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
    SymbolCreditInterferenceAudit,
    balanced_symbol_schedule,
    load_malecns_v1,
)
from drosomath.whole_brain import TimingProfiler  # noqa: E402
from drosomath.malecns.symbol_learning_extended import extended_checkpoint_metrics  # noqa: E402
from run_prospective_twohop_phase_f1b3 import (  # noqa: E402
    _configs,
    _evaluate_isolated,
    _make_brain,
    _run_seed,
    _surface_match,
    build_f1b_dynamic_interface,
)


REPLICATION_SEEDS = (53, 59, 61)
BENCHMARK_SEED = 53


def _benchmark_config() -> tuple[SymbolInterfaceConfig, SymbolLearningConfig]:
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
        training_trials=200,
        evaluation_repetitions=10,
        directional_learning_rate=0.02,
        reward_learning_rate=0.02,
        plastic_fraction=0.05,
        adaptive_plastic_budget=False,
        two_hop_credit_mode="prospective_anatomical",
        profile_timing=False,
    )
    return interface_config, learning_config


def _timed_training(connectome, interface, interface_config, config, seed, *, route_cache_enabled=True):
    profiler = TimingProfiler()
    brain = _make_brain(connectome, interface_config, seed)
    session = SymbolLearningSession(
        brain,
        interface,
        config=config,
        timing_profiler=profiler,
        route_cache_enabled=route_cache_enabled,
    )
    started = time.perf_counter()
    session.train(seed=seed)
    wall = time.perf_counter() - started
    return {
        "wall_seconds": float(wall),
        "trials": int(config.training_trials),
        "timing": profiler.report(),
        "session": session,
    }


def _warmup(connectome, interface, interface_config, config):
    # This call is deliberately outside all measured warm runs.  It warms the
    # optional Numba/JIT path without changing any measured run's state.
    brain = _make_brain(connectome, interface_config, BENCHMARK_SEED)
    session = SymbolLearningSession(brain, interface, config=config)
    session.train_trial("A")


def _median_timing(runs):
    names = sorted({
        name
        for run in runs
        for name in run["timing"]["seconds"]
    })
    seconds = {
        name: float(np.median([run["timing"]["seconds"].get(name, 0.0) for run in runs]))
        for name in names
    }
    total = max(seconds.get("total_training_seconds", 0.0), 1e-12)
    fractions = {name: float(value / total) for name, value in seconds.items()}
    return {
        "seconds": seconds,
        "fraction_of_total_training": fractions,
        "hotspots_at_least_15_percent": [
            name for name, fraction in fractions.items()
            if name not in {"total_training_seconds", "total_seconds"} and fraction >= 0.15
        ],
    }


def run_benchmark(connectome, interface, interface_config, *, route_cache_enabled=True):
    _, base_config = _benchmark_config()
    cold_started = time.perf_counter()
    cold = _timed_training(
        connectome, interface, interface_config, base_config, BENCHMARK_SEED,
        route_cache_enabled=route_cache_enabled,
    )
    cold_end_to_end = time.perf_counter() - cold_started
    _warmup(connectome, interface, interface_config, base_config)
    warm_runs = [
        _timed_training(
            connectome, interface, interface_config, base_config, BENCHMARK_SEED,
            route_cache_enabled=route_cache_enabled,
        )
        for _ in range(3)
    ]
    warm_median_wall = float(np.median([run["wall_seconds"] for run in warm_runs]))
    warm_profile = _median_timing(warm_runs)
    baseline_total = max(warm_profile["seconds"].get("total_training_seconds", warm_median_wall), 1e-12)
    return {
        "protocol": {
            "mode": "prospective_anatomical",
            "brain_seed": BENCHMARK_SEED,
            "training_trials": 200,
            "balanced_symbols": True,
            "duration_ms": 40.0,
            "stimulus_rate_hz": 205.0,
            "diagnostic_audit_observers": False,
            "jit_warmup_excluded_from_warm_runs": True,
            "route_cache_enabled": bool(route_cache_enabled),
        },
        "cold": {
            "end_to_end_seconds": float(cold_end_to_end),
            "training_wall_seconds": float(cold["wall_seconds"]),
            "timing": cold["timing"],
        },
        "warm": {
            "runs": [
                {"wall_seconds": run["wall_seconds"], "timing": run["timing"]}
                for run in warm_runs
            ],
            "median_wall_seconds": warm_median_wall,
            "median_trials_per_second": float(200.0 / warm_median_wall),
            "median_milliseconds_per_trial": float(1000.0 * warm_median_wall / 200.0),
            "median_profile": warm_profile,
        },
        "baseline_warm_seconds": float(baseline_total),
    }


def _nested_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    try:
        return bool(left == right)
    except ValueError:
        return False


def run_route_cache_equivalence(connectome, interface, interface_config):
    _, base = _benchmark_config()
    config = replace(base, training_trials=16, evaluation_repetitions=1)
    events = {"reference": [], "optimized": []}

    def observer_for(name):
        def observe(**kwargs):
            events[name].append({
                key: np.asarray(kwargs[key]).copy()
                if key in {"edge_indices", "hops", "actual_deltas", "eligibility", "path_polarities"}
                else kwargs[key]
                for key in kwargs
            })
        return observe

    brains = {}
    sessions = {}
    for name, enabled in (("reference", False), ("optimized", True)):
        brain = _make_brain(connectome, interface_config, BENCHMARK_SEED)
        brains[name] = brain
        sessions[name] = SymbolLearningSession(
            brain,
            interface,
            config=config,
            directional_telemetry_observer=observer_for(name),
            route_cache_enabled=enabled,
        )
    schedule = balanced_symbol_schedule(cycles=4, seed=BENCHMARK_SEED + config.schedule_seed_offset)
    results = {"reference": [], "optimized": []}
    for target in schedule:
        for name in ("reference", "optimized"):
            results[name].append(sessions[name].train_trial(target))

    result_equal = all(
        left.decision == right.decision
        and left.total_output_spikes == right.total_output_spikes
        and left.output_rates_hz == right.output_rates_hz
        and left.directional_error == right.directional_error
        for left, right in zip(results["reference"], results["optimized"])
    )
    state_equal = all(
        np.array_equal(getattr(brains["reference"].plasticity, name), getattr(brains["optimized"].plasticity, name))
        for name in ("multiplier", "stability", "usage_ema", "eligibility", "plastic_mask")
    )
    event_equal = len(events["reference"]) == len(events["optimized"]) and all(
        left.keys() == right.keys()
        and all(
            np.array_equal(left[key], right[key])
            if isinstance(left[key], np.ndarray) else left[key] == right[key]
            for key in left
        )
        for left, right in zip(events["reference"], events["optimized"])
    )
    return {
        "trials": len(schedule),
        "decisions_and_rates_equal": bool(result_equal),
        "directional_telemetry_equal": bool(event_equal),
        "multiplier_stability_mask_eligibility_equal": bool(state_equal),
        "plastic_budget_equal": brains["reference"].plasticity.plastic_edge_count == brains["optimized"].plasticity.plastic_edge_count,
        "rng_equal": _nested_equal(
            brains["reference"].rng.bit_generator.state,
            brains["optimized"].rng.bit_generator.state,
        ),
        "reference_route_cache_bytes": int(sessions["reference"].controller.route_cache_bytes()),
        "optimized_route_cache_bytes": int(sessions["optimized"].controller.route_cache_bytes()),
        "passed": bool(
            result_equal and event_equal and state_equal
            and brains["reference"].plasticity.plastic_edge_count == brains["optimized"].plasticity.plastic_edge_count
            and _nested_equal(brains["reference"].rng.bit_generator.state, brains["optimized"].rng.bit_generator.state)
        ),
    }


def _replication_result(connectome, interface, interface_config, learning_config, seed):
    # F.1B.3 already established prospective safety.  The held-out run uses
    # normal learning telemetry only; heavy per-update attribution stays out
    # of the replication protocol.
    brain = _make_brain(connectome, interface_config, seed)
    session = SymbolLearningSession(brain, interface, config=learning_config)
    schedule = balanced_symbol_schedule(
        cycles=learning_config.training_trials // len(SYMBOLS),
        seed=seed + learning_config.schedule_seed_offset,
    )
    eval_schedule = balanced_symbol_schedule(
        cycles=learning_config.evaluation_repetitions,
        seed=seed + learning_config.evaluation_seed_offset,
    )
    checkpoints = {
        "0": extended_checkpoint_metrics(_evaluate_isolated(session, eval_schedule))
    }
    for trial, target in enumerate(schedule, start=1):
        session.train_trial(target)
        if trial == 400:
            checkpoints["400"] = extended_checkpoint_metrics(
                _evaluate_isolated(session, eval_schedule)
            )
    checkpoints["800"] = extended_checkpoint_metrics(
        _evaluate_isolated(session, eval_schedule)
    )
    return {
        "seed": int(seed),
        "trials": int(len(schedule)),
        "schedule_count_per_symbol": {
            symbol: int(sum(target == symbol for target in schedule)) for symbol in SYMBOLS
        },
        "checkpoints": checkpoints,
        "state": {
            "plastic_budget": int(brain.plasticity.plastic_edge_count),
            "budget_drift": int(brain.plasticity.plastic_edge_count - session.plastic_budget_start),
            "adaptive_reallocations": 0,
        },
        "brain": brain,
        "session": session,
    }


def _strip_replication(run):
    return {
        key: value for key, value in run.items()
        if key not in {"brain", "session"}
    }


def _replication_summary(control, intervention):
    control800 = [run["checkpoints"]["800"] for run in control]
    intervention800 = [run["checkpoints"]["800"] for run in intervention]
    control_margin = [float(row["target_minus_best_competitor_margin"]) for row in control800]
    intervention_margin = [float(row["target_minus_best_competitor_margin"]) for row in intervention800]
    control_accuracy = [float(row["accuracy"]) for row in control800]
    intervention_accuracy = [float(row["accuracy"]) for row in intervention800]
    better = [intervention_margin[i] > control_margin[i] for i in range(3)]
    no_collapse = all(not bool(row["collapsed"]) for row in intervention800)
    replicated = bool(
        float(np.mean(intervention_margin)) > float(np.mean(control_margin))
        and sum(better) >= 2
        and float(np.mean(intervention_accuracy)) >= float(np.mean(control_accuracy))
        and no_collapse
    )
    discrete = bool(
        float(np.mean(intervention_accuracy)) >= 0.5
        and sum(value >= 0.5 for value in intervention_accuracy) >= 2
        and no_collapse
    )
    return {
        "accuracy_comparison": {
            "control_mean_800": float(np.mean(control_accuracy)),
            "intervention_mean_800": float(np.mean(intervention_accuracy)),
            "control_per_seed": control_accuracy,
            "intervention_per_seed": intervention_accuracy,
        },
        "margin_comparison": {
            "control_mean_800": float(np.mean(control_margin)),
            "intervention_mean_800": float(np.mean(intervention_margin)),
            "control_per_seed": control_margin,
            "intervention_per_seed": intervention_margin,
            "intervention_better_seed_count": int(sum(better)),
        },
        "rank_comparison": {
            "control_mean_rank_800": float(np.mean([
                row["target_rank"]["mean"] or 0.0 for row in control800
            ])),
            "intervention_mean_rank_800": float(np.mean([
                row["target_rank"]["mean"] or 0.0 for row in intervention800
            ])),
        },
        "no_decision": {
            "control_mean_fraction_800": float(np.mean([row["no_decision_fraction"] for row in control800])),
            "intervention_mean_fraction_800": float(np.mean([row["no_decision_fraction"] for row in intervention800])),
            "intervention_no_collapse": no_collapse,
        },
        "prospective_effect_replicated": replicated,
        "discrete_mapping_demonstrated": discrete,
    }


def run(*, data_dir: Path, output_path: Path, f1a3_artifact: Path) -> dict[str, object]:
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config, _ = _configs("active_chain")
    interface, candidate_count, _ = build_f1b_dynamic_interface(connectome, interface_config)
    baseline = run_benchmark(
        connectome, interface, interface_config, route_cache_enabled=False
    )
    optimized = run_benchmark(
        connectome, interface, interface_config, route_cache_enabled=True
    )
    equivalence = run_route_cache_equivalence(connectome, interface, interface_config)
    control_config, control_learning = _configs("active_chain")
    intervention_config, intervention_learning = _configs("prospective_anatomical")
    control = [
        _replication_result(connectome, interface, control_config, control_learning, seed)
        for seed in REPLICATION_SEEDS
    ]
    intervention = [
        _replication_result(connectome, interface, intervention_config, intervention_learning, seed)
        for seed in REPLICATION_SEEDS
    ]
    replication = _replication_summary(control, intervention)
    performance_profile = baseline["warm"]["median_profile"]
    hotspots = performance_profile["hotspots_at_least_15_percent"]
    # No speculative optimization is accepted without a measured hotspot.
    # This phase's baseline is therefore explicit even if the list is empty.
    artifact = {
        "protocol": {
            "phase": "F.1B.4+P.1",
            "replication_seeds": list(REPLICATION_SEEDS),
            "trials_per_arm_per_seed": 800,
            "trials_per_symbol": 200,
            "checkpoints": [0, 400, 800],
            "dynamic_candidate_count": int(candidate_count),
            "surface_matches_f1a3_artifact": _surface_match(interface, f1a3_artifact),
            "optimization_tuning_seeds": [],
        },
        "performance": {
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
            },
            "baseline_profile": baseline,
            "optimized_profile": optimized,
            "hotspots": hotspots,
            "speedup": float(
                baseline["warm"]["median_wall_seconds"]
                / optimized["warm"]["median_wall_seconds"]
            ),
            "trials_per_second_before": baseline["warm"]["median_trials_per_second"],
            "trials_per_second_after": optimized["warm"]["median_trials_per_second"],
            "additional_cache_bytes": equivalence["optimized_route_cache_bytes"],
            "equivalence_passed": equivalence["passed"],
            "equivalence": equivalence,
        },
        "control": {str(run["seed"]): _strip_replication(run) for run in control},
        "intervention": {str(run["seed"]): _strip_replication(run) for run in intervention},
        "replication": replication,
        "conclusion": {
            "prospective_effect_replicated": replication["prospective_effect_replicated"],
            "discrete_mapping_demonstrated": replication["discrete_mapping_demonstrated"],
            "symbol_stage_complete_for_now": replication["prospective_effect_replicated"],
            "performance_equivalence_passed": equivalence["passed"],
            "performance_speedup": float(
                baseline["warm"]["median_wall_seconds"]
                / optimized["warm"]["median_wall_seconds"]
            ),
            "recommended_next_step": (
                "sequence_working_memory"
                if replication["prospective_effect_replicated"]
                else "reassess_prospective_credit_generalization"
            ),
        },
        "pass": bool(
            replication["prospective_effect_replicated"]
            and equivalence["passed"]
            and (
                baseline["warm"]["median_wall_seconds"]
                / optimized["warm"]["median_wall_seconds"]
            ) >= 1.25
            and baseline["protocol"]["diagnostic_audit_observers"] is False
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
        "--output", type=Path,
        default=Path("results/latest_symbol_replication_performance_phase_f1b4.json"),
    )
    parser.add_argument(
        "--f1a3-artifact", type=Path,
        default=Path("results/latest_dynamic_decision_surface_phase_f1a3.json"),
    )
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.benchmark_only:
        connectome = load_malecns_v1(args.data_dir, min_connection_synapses=5)
        interface_config, _ = _benchmark_config()
        interface, candidate_count, _ = build_f1b_dynamic_interface(connectome, interface_config)
        benchmark = run_benchmark(
            connectome, interface, interface_config, route_cache_enabled=False
        )
        result = {
            "phase": "P.1 baseline",
            "candidate_count": int(candidate_count),
            "performance": benchmark,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    else:
        result = run(data_dir=args.data_dir, output_path=args.output, f1a3_artifact=args.f1a3_artifact)
        print(json.dumps({"pass": result.get("pass", True), "conclusion": result.get("conclusion", {})}, indent=2))


if __name__ == "__main__":
    main()
