"""Run F.2C delay generalization and the P.3 neural runtime audit."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    DelayedCueLearningSession,
    DelayedCueSession,
    GO_ALLOCATION_SEED,
    GO_SYMBOL,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    WorkingMemoryLearningConfig,
    balanced_symbol_schedule,
    load_malecns_v1,
    pairwise_set_jaccard,
)
from drosomath.malecns.symbol_interface import (  # noqa: E402
    _restore_persistent_state,
    _snapshot_persistent_state,
)
from drosomath.malecns.symbol_learning_extended import target_rank  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler  # noqa: E402
from run_delayed_cue_learning_phase_f2b import (  # noqa: E402
    _array_digest,
    _make_brain,
    _prediction_label,
    _summarize_rows,
)
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_symbol_replication_performance_phase_f1b4 import run_route_cache_equivalence  # noqa: E402


TRAINING_SEEDS = (101, 103, 107)
EVALUATION_DELAYS_MS = (0, 10, 20, 40, 80)
TRAINING_TRIALS_PER_CUE = 200
EVALUATION_TRIALS_PER_CUE = 10
ARTIFACT = Path("results/latest_delay_generalization_phase_f2c.json")
DATA_DIR = Path("data/malecns_v1")


def _interface_config() -> SymbolInterfaceConfig:
    return SymbolInterfaceConfig(
        symbols=SYMBOLS,
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


def _persistent_digest_from_snapshot(snapshot) -> dict[str, str]:
    return {
        name: _array_digest(snapshot[name])
        for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    } | {
        "promoted_edges": _array_digest(snapshot["allocation_overrides"]["promoted_edges"]),
        "retired_edges": _array_digest(snapshot["allocation_overrides"]["retired_edges"]),
    }


def _rng_state(brain):
    return copy.deepcopy(brain.rng.bit_generator.state)


def _make_twin(connectome, interface_config, seed, persistent, rng_state):
    brain = _make_brain(connectome, interface_config, seed)
    _restore_persistent_state(brain, persistent)
    brain.rng.bit_generator.state = copy.deepcopy(rng_state)
    brain.reset()
    return brain


def _summarize_delay_rows(rows) -> dict[str, object]:
    summary = _summarize_rows(rows)
    summary["delay_ms"] = float(getattr(rows[0], "delay_ms", 0.0)) if rows else None
    return summary


def _evaluate_delay(connectome, wm_interface, interface_config, seed, persistent, rng_state, delay_ms):
    schedule = balanced_symbol_schedule(cycles=EVALUATION_TRIALS_PER_CUE, seed=seed + 20_000)
    arms = {}
    for arm_name, reset_before_go in (("intact", False), ("reset", True)):
        twin = _make_twin(connectome, interface_config, seed, persistent, rng_state)
        session = DelayedCueSession(twin, wm_interface, delay_ms=delay_ms)
        rows = [session.run_trial(cue, reset_before_go=reset_before_go) for cue in schedule]
        arms[arm_name] = _summarize_delay_rows(rows)
        arms[arm_name]["delay_ms"] = float(delay_ms)
    arms["memory_comparison"] = {
        "accuracy_gap_intact_minus_reset": float(arms["intact"]["accuracy"] - arms["reset"]["accuracy"]),
        "margin_gap_intact_minus_reset_hz": float(
            arms["intact"]["target_minus_best_competitor_margin_hz"]
            - arms["reset"]["target_minus_best_competitor_margin_hz"]
        ),
        "go_separation_ratio_intact_over_reset": float(
            arms["intact"]["cue_conditioned_go_output_separation_hz"]
            / max(arms["reset"]["cue_conditioned_go_output_separation_hz"], 1e-9)
        ),
    }
    return arms


def _mean_by_delay(runs, arm, field):
    return {
        str(delay): float(np.mean([
            runs[str(seed)]["evaluations"][str(delay)][arm][field]
            for seed in TRAINING_SEEDS
        ]))
        for delay in EVALUATION_DELAYS_MS
    }


def _retention_curve(runs, field):
    result = {}
    for seed in TRAINING_SEEDS:
        base = float(runs[str(seed)]["evaluations"]["20"]["intact"][field])
        result[str(seed)] = {
            str(delay): float(runs[str(seed)]["evaluations"][str(delay)]["intact"][field] / max(base, 1e-9))
            for delay in EVALUATION_DELAYS_MS
        }
    return result


def _memory_horizon(intact_curve, reset_curve):
    base = float(intact_curve["20"])
    eligible = [
        delay for delay in EVALUATION_DELAYS_MS
        if delay >= 20 and intact_curve[str(delay)] > reset_curve[str(delay)]
        and intact_curve[str(delay)] >= 0.75 * base
    ]
    return int(max(eligible) if eligible else 20)


def _criterion(runs, delay, baseline_delay=20):
    intact = [runs[str(seed)]["evaluations"][str(delay)]["intact"] for seed in TRAINING_SEEDS]
    reset = [runs[str(seed)]["evaluations"][str(delay)]["reset"] for seed in TRAINING_SEEDS]
    mean_intact = float(np.mean([row["accuracy"] for row in intact]))
    mean_reset = float(np.mean([row["accuracy"] for row in reset]))
    baseline = float(np.mean([
        runs[str(seed)]["evaluations"][str(baseline_delay)]["intact"]["accuracy"]
        for seed in TRAINING_SEEDS
    ]))
    no_collapse = not any(row["output_collapse"] for row in (*intact, *reset))
    return {
        "mean_intact_accuracy": mean_intact,
        "mean_reset_accuracy": mean_reset,
        "intact_gt_reset": bool(mean_intact > mean_reset),
        "seeds_intact_gt_reset": int(sum(a["accuracy"] > b["accuracy"] for a, b in zip(intact, reset))),
        "mean_intact_vs_20ms_fraction": float(mean_intact / max(baseline, 1e-9)),
        "no_global_output_collapse": bool(no_collapse),
        "supported": bool(
            mean_intact > mean_reset
            and sum(a["accuracy"] > b["accuracy"] for a, b in zip(intact, reset)) >= 2
            and mean_intact + 1e-12 >= 0.75 * baseline
            and no_collapse
        ),
    }


def _state_decay_curve(runs):
    curve = {}
    for delay in EVALUATION_DELAYS_MS:
        intact = [runs[str(seed)]["evaluations"][str(delay)]["intact"] for seed in TRAINING_SEEDS]
        reset = [runs[str(seed)]["evaluations"][str(delay)]["reset"] for seed in TRAINING_SEEDS]
        curve[str(delay)] = {
            "mean_pre_go_active_count": float(np.mean([
                row["state_memory"]["mean_pre_go_active_count"] for row in intact
            ])),
            "mean_pre_go_membrane_norm": float(np.mean([
                row["state_memory"]["mean_pre_go_membrane_norm"] for row in intact
            ])),
            "mean_pre_go_conductance_norm": float(np.mean([
                row["state_memory"]["mean_pre_go_conductance_norm"] for row in intact
            ])),
            "pairwise_cue_active_set_jaccard": {
                key: float(np.mean([
                    row["state_memory"]["pairwise_pre_go_active_set_jaccard"].get(key, 0.0)
                    for row in intact
                ]))
                for key in ("A|B", "A|C", "A|D", "B|C", "B|D", "C|D")
            },
            "fingerprint_diversity": float(np.mean([
                row["state_memory"]["pre_go_fingerprint_diversity"] for row in intact
            ])),
            "reset_mean_post_reset_active_count": float(np.mean([
                row["state_memory"]["mean_post_reset_active_count"] for row in reset
            ])),
        }
    baseline = curve["20"]
    for delay, row in curve.items():
        row["pre_go_active_count_ratio_vs_20ms"] = float(
            row["mean_pre_go_active_count"] / max(baseline["mean_pre_go_active_count"], 1e-9)
        )
        row["pre_go_conductance_norm_ratio_vs_20ms"] = float(
            row["mean_pre_go_conductance_norm"] / max(baseline["mean_pre_go_conductance_norm"], 1e-9)
        )
    return curve


def _trace_episode(brain, wm_interface, delay_ms=20.0):
    captured = []
    original = brain.step

    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(np.asarray(result[0], dtype=np.int32).copy())
        return result

    brain.step = traced
    try:
        session = DelayedCueSession(brain, wm_interface, delay_ms=delay_ms)
        session.run_trial("A")
    finally:
        brain.step = original
    return captured


def _neural_exact_replay(connectome, wm_interface, interface_config):
    left = _make_brain(connectome, interface_config, 911)
    right = _make_brain(connectome, interface_config, 911)
    left_trace = _trace_episode(left, wm_interface)
    right_trace = _trace_episode(right, wm_interface)
    same_fired = len(left_trace) == len(right_trace) and all(
        np.array_equal(a, b) for a, b in zip(left_trace, right_trace)
    )
    same_state = all(np.array_equal(getattr(left, name), getattr(right, name)) for name in ("v", "g", "refractory_until"))
    return {
        "passed": bool(same_fired and same_state and left.step_index == right.step_index and _rng_state(left) == _rng_state(right)),
        "fired_indices_equal": bool(same_fired),
        "membrane_equal": bool(np.array_equal(left.v, right.v)),
        "conductance_equal": bool(np.array_equal(left.g, right.g)),
        "refractory_equal": bool(np.array_equal(left.refractory_until, right.refractory_until)),
        "rng_equal": bool(_rng_state(left) == _rng_state(right)),
    }


def _profile_network(connectome, wm_interface, interface_config, *, learning_enabled, config):
    schedule = balanced_symbol_schedule(cycles=25, seed=44_401)
    warm_brain = _make_brain(connectome, interface_config, 4_321)
    warm_session = (
        DelayedCueLearningSession(warm_brain, wm_interface, config=config)
        if learning_enabled else DelayedCueSession(warm_brain, wm_interface)
    )
    if learning_enabled:
        warm_session.train(schedule[:4])
    else:
        for cue in schedule[:4]:
            warm_session.run_trial(cue)

    runs = []
    for run_index in range(3):
        brain = _make_brain(connectome, interface_config, 4_321)
        profiler = TimingProfiler()
        brain.configure_neural_timing(profiler)
        session = (
            DelayedCueLearningSession(
                brain, wm_interface, config=config, timing_profiler=profiler,
                route_cache_enabled=True,
            )
            if learning_enabled else DelayedCueSession(brain, wm_interface)
        )
        started = time.perf_counter()
        if learning_enabled:
            session.train(schedule)
        else:
            for cue in schedule:
                session.run_trial(cue)
        wall = time.perf_counter() - started
        report = profiler.report()
        runs.append({
            "run": int(run_index + 1),
            "wall_seconds": float(wall),
            "episodes": int(len(schedule)),
            "episodes_per_second": float(len(schedule) / max(wall, 1e-12)),
            "profiler": report,
        })
    names = (
        "stimulus_injection_seconds",
        "active_neuron_state_update_seconds",
        "synaptic_scheduling_seconds",
        "delay_ring_handling_seconds",
    )
    medians = {
        name: float(np.median([run["profiler"]["seconds"].get(name, 0.0) for run in runs]))
        for name in names
    }
    median_wall = float(np.median([run["wall_seconds"] for run in runs]))
    return {
        "learning_enabled": bool(learning_enabled),
        "warmup_episodes": 4,
        "measured_episodes": len(schedule),
        "warm_runs": 3,
        "runs": runs,
        "median_wall_seconds": median_wall,
        "median_episodes_per_second": float(len(schedule) / max(median_wall, 1e-12)),
        "median_disjoint_neural_seconds": medians,
        "median_disjoint_sum_seconds": float(sum(medians.values())),
    }


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface, go_seed=GO_ALLOCATION_SEED)
    config = WorkingMemoryLearningConfig()
    runs = {}

    for seed in TRAINING_SEEDS:
        brain = _make_brain(connectome, interface_config, seed)
        session = DelayedCueLearningSession(brain, wm_interface, config=config, route_cache_enabled=True)
        schedule = balanced_symbol_schedule(cycles=TRAINING_TRIALS_PER_CUE, seed=seed + config.schedule_seed_offset)
        session.train(schedule)
        persistent = _snapshot_persistent_state(brain)
        rng_state = _rng_state(brain)
        evaluations = {
            str(delay): _evaluate_delay(
                connectome, wm_interface, interface_config, seed, persistent, rng_state, delay
            )
            for delay in EVALUATION_DELAYS_MS
        }
        runs[str(seed)] = {
            "seed": seed,
            "training": {
                "trials": len(session.trial_results),
                "accuracy": float(np.mean([row.success for row in session.trial_results])),
                "persistent_state": _persistent_digest_from_snapshot(persistent),
            },
            "evaluations": evaluations,
            "safety": {
                "plastic_budget_start": int(session.plastic_budget_start),
                "plastic_budget_end": int(brain.plasticity.plastic_edge_count),
                "plastic_budget_drift": int(brain.plasticity.plastic_edge_count - session.plastic_budget_start),
                "evaluation_twins_share_learned_state": True,
                "learning_disabled_during_evaluation": True,
            },
        }

    intact_accuracy = _mean_by_delay(runs, "intact", "accuracy")
    reset_accuracy = _mean_by_delay(runs, "reset", "accuracy")
    intact_margin = _mean_by_delay(runs, "intact", "target_minus_best_competitor_margin_hz")
    reset_margin = _mean_by_delay(runs, "reset", "target_minus_best_competitor_margin_hz")
    intact_target_rank = _mean_by_delay(runs, "intact", "mean_target_rank")
    reset_target_rank = _mean_by_delay(runs, "reset", "mean_target_rank")
    memory_gap = {
        str(delay): {
            "accuracy": float(intact_accuracy[str(delay)] - reset_accuracy[str(delay)]),
            "margin_hz": float(intact_margin[str(delay)] - reset_margin[str(delay)]),
        }
        for delay in EVALUATION_DELAYS_MS
    }
    state_decay = _state_decay_curve(runs)
    explosion = bool(
        state_decay["80"]["pre_go_active_count_ratio_vs_20ms"] > 4.0
        or state_decay["80"]["pre_go_conductance_norm_ratio_vs_20ms"] > 4.0
    )
    delay_criterion = {str(delay): _criterion(runs, delay) for delay in (40, 80)}
    normalized = {
        "accuracy_retention": _retention_curve(runs, "accuracy"),
        "margin_retention": _retention_curve(runs, "target_minus_best_competitor_margin_hz"),
    }

    network_disabled = _profile_network(connectome, wm_interface, interface_config, learning_enabled=False, config=config)
    network_training = _profile_network(connectome, wm_interface, interface_config, learning_enabled=True, config=config)
    neural_equivalence = _neural_exact_replay(connectome, wm_interface, interface_config)
    route_equivalence = run_route_cache_equivalence(connectome, interface, interface_config)
    horizon = _memory_horizon(intact_accuracy, reset_accuracy)
    f2b_replication = bool(
        intact_accuracy["20"] > 0.0
        and reset_accuracy["20"] < intact_accuracy["20"]
        and not any(runs[str(seed)]["evaluations"]["20"][arm]["output_collapse"] for seed in TRAINING_SEEDS for arm in ("intact", "reset"))
    )
    if delay_criterion["80"]["supported"]:
        next_step = "short_sequence_learning"
    elif delay_criterion["40"]["supported"]:
        next_step = "short_sequence_learning"
    elif f2b_replication:
        next_step = "short_sequence_with_short_context_then_memory_extension"
    else:
        next_step = "reassess_working_memory_replication"

    artifact = {
        "protocol": {
            "phase": "F.2C+P.3",
            "training_seeds": list(TRAINING_SEEDS),
            "training_delay_ms": 20,
            "evaluation_delays_ms": list(EVALUATION_DELAYS_MS),
            "training_trials_per_seed": 800,
            "evaluation_trials_per_cue": EVALUATION_TRIALS_PER_CUE,
            "stimulus_rate_hz": 205.0,
            "input_vocabulary": [*SYMBOLS, GO_SYMBOL],
            "output_vocabulary": list(SYMBOLS),
            "two_hop_credit_mode": "prospective_anatomical",
            "directional_learning_rate": 0.02,
            "reward_learning_rate": 0.02,
            "adaptive_plastic_budget": False,
            "route_cache_enabled": True,
            "dynamic_candidate_count": int(candidate_count),
            "selected_output_count": int(len(selected)),
        },
        "runs": runs,
        "retention": {
            "accuracy_curve": {"intact": intact_accuracy, "reset": reset_accuracy},
            "memory_gap_curve": memory_gap,
            "margin_curve_hz": {"intact": intact_margin, "reset": reset_margin},
            "target_rank_curve": {"intact": intact_target_rank, "reset": reset_target_rank},
            "normalized_retention": normalized,
            "state_decay_curve": state_decay,
            "memory_horizon_ms": horizon,
            "delay_criteria": delay_criterion,
        },
        "performance": {
            "network_profile_learning_disabled": network_disabled,
            "network_profile_training": network_training,
            "optimization_applied": False,
            "optimization_note": "P.3 retained profiling only; no subcomponent was changed before exact-equivalence evidence.",
            "exact_equivalence": bool(neural_equivalence["passed"]),
            "neural_exact_equivalence": neural_equivalence,
            "route_cache_equivalence": route_equivalence,
            "network_speedup": 1.0,
            "total_workload_speedup": 1.0,
        },
        "conclusion": {
            "f2b_replication_supported": f2b_replication,
            "delay_generalization_supported": bool(delay_criterion["40"]["supported"]),
            "long_delay_memory_supported": bool(delay_criterion["80"]["supported"]),
            "recurrent_activity_explosion": explosion,
            "recommended_next_step": next_step,
        },
        "pass": bool(route_equivalence["passed"] and neural_equivalence["passed"] and all(
            runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS
        )),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    artifact = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({
        "pass": artifact["pass"],
        "conclusion": artifact["conclusion"],
        "accuracy_curve": artifact["retention"]["accuracy_curve"],
        "memory_horizon_ms": artifact["retention"]["memory_horizon_ms"],
        "network_profile": artifact["performance"]["network_profile_learning_disabled"]["median_disjoint_neural_seconds"],
        "runtime_seconds": artifact["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
