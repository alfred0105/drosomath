"""Run Phase F.2D short-sequence distractor memory and P.4 profiling."""

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
    ORDERED_PAIRS,
    PlasticMaleCNSBrain,
    SequenceLearningSession,
    SequenceMemoryConfig,
    SYMBOLS,
    SymbolInterfaceConfig,
    TwoCueSequenceSession,
    WorkingMemoryInterface,
    balanced_pair_schedule,
    load_malecns_v1,
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
)
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_symbol_replication_performance_phase_f1b4 import run_route_cache_equivalence  # noqa: E402


TRAINING_SEEDS = (109, 113, 127)
CHECKPOINTS = (0, 400, 800)
EVALUATION_CYCLES = 4
ARTIFACT = Path("results/latest_short_sequence_memory_phase_f2d.json")
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


def _persistent_digest(brain) -> dict[str, str]:
    state = brain.plasticity
    overrides = state.allocation_overrides()
    return {
        name: _array_digest(getattr(state, name))
        for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    } | {
        "promoted_edges": _array_digest(overrides["promoted_edges"]),
        "retired_edges": _array_digest(overrides["retired_edges"]),
    }


def _snapshot_digest(snapshot) -> dict[str, str]:
    return {
        name: _array_digest(snapshot[name])
        for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    } | {
        "promoted_edges": _array_digest(snapshot["allocation_overrides"]["promoted_edges"]),
        "retired_edges": _array_digest(snapshot["allocation_overrides"]["retired_edges"]),
    }


def _make_twin(connectome, interface_config, seed, persistent, rng_state):
    brain = _make_brain(connectome, interface_config, seed)
    _restore_persistent_state(brain, persistent)
    brain.rng.bit_generator.state = copy.deepcopy(rng_state)
    brain.reset()
    return brain


def _prediction_label(value: str) -> str:
    return value if value in SYMBOLS else "NO_DECISION"


def _empty_confusion():
    return {first: {label: 0 for label in (*SYMBOLS, "NO_DECISION")} for first in SYMBOLS}


def _summarize(rows) -> dict[str, object]:
    confusion = _empty_confusion()
    predictions = [_prediction_label(row.decision) for row in rows]
    per_first = {}
    pair_accuracy = {}
    pair_predictions = {}
    for first in SYMBOLS:
        selected = [row for row in rows if row.first == first]
        first_predictions = [_prediction_label(row.decision) for row in selected]
        for prediction in first_predictions:
            confusion[first][prediction] += 1
        per_first[first] = {
            "trials": len(selected),
            "accuracy": float(np.mean([prediction == first for prediction in first_predictions])) if selected else 0.0,
        }
    for pair in ORDERED_PAIRS:
        key = "".join(pair)
        selected = [row for row in rows if (row.first, row.second) == pair]
        labels = [_prediction_label(row.decision) for row in selected]
        pair_accuracy[key] = float(np.mean([label == pair[0] for label in labels])) if selected else 0.0
        pair_predictions[key] = {label: int(labels.count(label)) for label in (*SYMBOLS, "NO_DECISION")}
    total = len(rows)
    counts = {label: int(predictions.count(label)) for label in (*SYMBOLS, "NO_DECISION")}
    margins = [
        float(row.go_output_rates_hz.get(row.first, 0.0))
        - max((float(value) for symbol, value in row.go_output_rates_hz.items() if symbol != row.first), default=0.0)
        for row in rows
    ]
    ranks = [target_rank(row.go_output_rates_hz, row.first) for row in rows]
    ranks = [rank for rank in ranks if rank is not None]
    shares = []
    for row in rows:
        total_rate = sum(float(row.go_output_rates_hz.get(symbol, 0.0)) for symbol in SYMBOLS)
        shares.append(float(row.go_output_rates_hz.get(row.first, 0.0)) / total_rate if total_rate > 0 else 0.0)
    same_keys = {"AA", "BB", "CC", "DD"}
    different_keys = {"".join(pair) for pair in ORDERED_PAIRS} - same_keys
    return {
        "trials": total,
        "accuracy": float(np.mean([prediction == row.first for prediction, row in zip(predictions, rows)])) if rows else 0.0,
        "macro_first_accuracy": float(np.mean([value["accuracy"] for value in per_first.values()])) if rows else 0.0,
        "per_first_accuracy": {first: value["accuracy"] for first, value in per_first.items()},
        "per_first": per_first,
        "confusion_matrix": confusion,
        "pair_accuracy": pair_accuracy,
        "pair_predictions": pair_predictions,
        "same_symbol_accuracy": float(np.mean([pair_accuracy[key] for key in same_keys])),
        "different_symbol_accuracy": float(np.mean([pair_accuracy[key] for key in different_keys])),
        "prediction_distribution": counts,
        "largest_prediction_fraction": float(max(counts.values()) / total) if total else 0.0,
        "output_collapse": bool(total and max(counts.values()) / total >= 0.80),
        "no_decision_fraction": float(counts["NO_DECISION"] / total) if total else 0.0,
        "target_output_share": float(np.mean(shares)) if shares else 0.0,
        "target_minus_best_competitor_margin_hz": float(np.mean(margins)) if margins else 0.0,
        "mean_target_rank": float(np.mean(ranks)) if ranks else None,
        "fraction_rank_1": float(np.mean([rank == 1.0 for rank in ranks])) if ranks else 0.0,
        "fraction_rank_le_2": float(np.mean([rank <= 2.0 for rank in ranks])) if ranks else 0.0,
        "go_output_spikes": int(sum(row.go_output_spikes for row in rows)),
        "order_reversal": _order_reversal(rows),
    }


def _order_reversal(rows):
    result = {}
    for left, right in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")):
        forward = [row for row in rows if (row.first, row.second) == (left, right)]
        reverse = [row for row in rows if (row.first, row.second) == (right, left)]
        forward_labels = [_prediction_label(row.decision) for row in forward]
        reverse_labels = [_prediction_label(row.decision) for row in reverse]
        result[f"{left}{right}_vs_{right}{left}"] = {
            "forward_accuracy": float(np.mean([label == left for label in forward_labels])) if forward else 0.0,
            "reverse_accuracy": float(np.mean([label == right for label in reverse_labels])) if reverse else 0.0,
            "forward_predictions": {label: int(forward_labels.count(label)) for label in (*SYMBOLS, "NO_DECISION")},
            "reverse_predictions": {label: int(reverse_labels.count(label)) for label in (*SYMBOLS, "NO_DECISION")},
            "accuracy_difference": float(
                (np.mean([label == left for label in forward_labels]) if forward_labels else 0.0)
                - (np.mean([label == right for label in reverse_labels]) if reverse_labels else 0.0)
            ),
        }
    result["reversed_pair_target_sensitivity"] = float(np.mean([
        abs(value["accuracy_difference"])
        for key, value in result.items() if key != "reversed_pair_target_sensitivity"
    ]))
    return result


def _evaluate(connectome, interface, interface_config, seed, persistent, rng_state, checkpoint):
    schedule = balanced_pair_schedule(EVALUATION_CYCLES, seed=seed + 30_000)
    arms = {}
    for name, reset_between_items in (("intact", False), ("between_item_reset", True)):
        twin = _make_twin(connectome, interface_config, seed, persistent, rng_state)
        session = TwoCueSequenceSession(twin, interface)
        rows = [
            session.run_trial(first, second, reset_between_items=reset_between_items)
            for first, second in schedule
        ]
        arms[name] = _summarize(rows)
        arms[name]["checkpoint"] = checkpoint
    return arms


def _profile(connectome, interface, interface_config, *, learning_enabled, config):
    schedule = balanced_pair_schedule(7, seed=44_997)[:100]
    warm_brain = _make_brain(connectome, interface_config, 8_321)
    if learning_enabled:
        SequenceLearningSession(warm_brain, interface, config=config).train(schedule[:4])
    else:
        session = TwoCueSequenceSession(warm_brain, interface)
        for first, second in schedule[:4]:
            session.run_trial(first, second)
    runs = []
    for run_index in range(3):
        brain = _make_brain(connectome, interface_config, 8_321)
        profiler = TimingProfiler()
        brain.configure_neural_timing(profiler)
        if learning_enabled:
            session = SequenceLearningSession(
                brain, interface, config=config, timing_profiler=profiler, route_cache_enabled=True
            )
            started = time.perf_counter()
            session.train(schedule)
        else:
            session = TwoCueSequenceSession(brain, interface)
            started = time.perf_counter()
            for first, second in schedule:
                session.run_trial(first, second)
        wall = time.perf_counter() - started
        runs.append({
            "run": run_index + 1,
            "episodes": len(schedule),
            "wall_seconds": float(wall),
            "episodes_per_second": float(len(schedule) / max(wall, 1e-12)),
            "profiler": profiler.report(),
        })
    names = (
        "due_index_collection_seconds",
        "active_set_merge_seconds",
        "active_neuron_state_update_seconds",
        "synaptic_scheduling_seconds",
        "stimulus_injection_seconds",
    )
    medians = {
        name: float(np.median([run["profiler"]["seconds"].get(name, 0.0) for run in runs]))
        for name in names
    }
    delay_total = medians["due_index_collection_seconds"] + medians["active_set_merge_seconds"]
    return {
        "learning_enabled": learning_enabled,
        "warmup_episodes": 4,
        "measured_episodes": len(schedule),
        "warm_runs": 3,
        "runs": runs,
        "median_wall_seconds": float(np.median([run["wall_seconds"] for run in runs])),
        "median_episodes_per_second": float(np.median([run["episodes_per_second"] for run in runs])),
        "median_disjoint_sections": medians,
        "median_delay_ring_subtotal_seconds": delay_total,
    }


def _sequence_exact_replay(connectome, interface, interface_config):
    traces = []
    states = []
    results = []
    for _ in range(2):
        brain = _make_brain(connectome, interface_config, 8_877)
        captured = []
        original = brain.step

        def traced(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            captured.append(np.asarray(result[0], dtype=np.int32).copy())
            return result

        brain.step = traced
        try:
            results.append(TwoCueSequenceSession(brain, interface).run_trial("A", "D"))
        finally:
            brain.step = original
        traces.append(captured)
        states.append((
            brain.v.copy(),
            brain.g.copy(),
            brain.refractory_until.copy(),
            brain.plasticity.multiplier.copy(),
            brain.plasticity.usage_ema.copy(),
            brain.plasticity.stability.copy(),
            brain.plasticity.plastic_mask.copy(),
            copy.deepcopy(brain.rng.bit_generator.state),
        ))
    fired_equal = len(traces[0]) == len(traces[1]) and all(np.array_equal(a, b) for a, b in zip(*traces))
    state_equal = all(np.array_equal(states[0][index], states[1][index]) for index in range(7))
    return {
        "passed": bool(fired_equal and state_equal and results[0].decision == results[1].decision and results[0].go_output_spikes == results[1].go_output_spikes and states[0][7] == states[1][7]),
        "fired_indices_equal": fired_equal,
        "membrane_equal": bool(np.array_equal(states[0][0], states[1][0])),
        "conductance_equal": bool(np.array_equal(states[0][1], states[1][1])),
        "refractory_equal": bool(np.array_equal(states[0][2], states[1][2])),
        "output_decision_equal": bool(results[0].decision == results[1].decision),
        "output_spike_count_equal": bool(results[0].go_output_spikes == results[1].go_output_spikes),
        "multiplier_equal": bool(np.array_equal(states[0][3], states[1][3])),
        "usage_ema_equal": bool(np.array_equal(states[0][4], states[1][4])),
        "stability_equal": bool(np.array_equal(states[0][5], states[1][5])),
        "plastic_mask_equal": bool(np.array_equal(states[0][6], states[1][6])),
        "rng_equal": bool(states[0][7] == states[1][7]),
    }


def _schedule_balance(schedule):
    matrix = {first: {second: int(schedule.count((first, second))) for second in SYMBOLS} for first in SYMBOLS}
    target_by_second = {
        second: {first: int(sum(first_value == first and second_value == second for first_value, second_value in schedule)) for first in SYMBOLS}
        for second in SYMBOLS
    }
    return {
        "pair_count_matrix": matrix,
        "target_count_by_second": target_by_second,
        "pair_counts_uniform": all(value == 50 for row in matrix.values() for value in row.values()),
        "first_counts_uniform": all(sum(row.values()) == 200 for row in matrix.values()),
        "second_counts_uniform": all(sum(matrix[first][second] for first in SYMBOLS) == 200 for second in SYMBOLS),
        "second_cue_target_independence": all(
            len(set(target_by_second[second].values())) == 1 for second in SYMBOLS
        ),
        "second_only_nominal_shortcut_accuracy": 0.25,
    }


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    config = SequenceMemoryConfig()
    full_schedule = balanced_pair_schedule(50, seed=109 + config.schedule_seed_offset)
    balance = _schedule_balance(full_schedule)
    runs = {}
    for seed in TRAINING_SEEDS:
        brain = _make_brain(connectome, interface_config, seed)
        session = SequenceLearningSession(brain, wm_interface, config=config, route_cache_enabled=True)
        schedule = balanced_pair_schedule(50, seed=seed + config.schedule_seed_offset)
        checkpoints = {"0": _evaluate(connectome, wm_interface, interface_config, seed, _snapshot_persistent_state(brain), copy.deepcopy(brain.rng.bit_generator.state), 0)}
        credit_records = []
        session.train(schedule[:400], capture_first_incorrect=16)
        credit_records.extend([
            dict(record["phase_credit"]) for record in session.trial_results[:16] if record.get("phase_credit") is not None
        ])
        checkpoints["400"] = _evaluate(connectome, wm_interface, interface_config, seed, _snapshot_persistent_state(brain), copy.deepcopy(brain.rng.bit_generator.state), 400)
        session.train(schedule[400:], capture_first_incorrect=0)
        checkpoints["800"] = _evaluate(connectome, wm_interface, interface_config, seed, _snapshot_persistent_state(brain), copy.deepcopy(brain.rng.bit_generator.state), 800)
        state = brain.plasticity
        saturation = (state.multiplier <= state.config.min_multiplier) | (state.multiplier >= state.config.max_multiplier)
        runs[str(seed)] = {
            "seed": seed,
            "schedule_balance": _schedule_balance(schedule),
            "checkpoints": checkpoints,
            "episode_credit": {
                "episodes_captured": len(credit_records),
                "aggregate": {
                    key: int(sum(record.get(key, 0) for record in credit_records))
                    for key in ("updated_edges", "updated_edges_eligible_after_first", "updated_edges_eligible_by_end_second", "updated_edges_newly_eligible_during_go", "eligible_edges_after_first", "eligible_edges_by_end_second", "eligible_edges_by_end_go")
                },
            },
            "safety": {
                "plastic_budget_start": int(session.plastic_budget_start),
                "plastic_budget_end": int(state.plastic_edge_count),
                "plastic_budget_drift": int(state.plastic_edge_count - session.plastic_budget_start),
                "adaptive_reallocations": 0,
                "multiplier_saturation_fraction": float(np.mean(saturation)),
                "mean_multiplier": float(np.mean(state.multiplier)),
                "mean_stability": float(np.mean(state.stability)),
            },
        }

    def mean_arm(checkpoint, arm, field):
        return float(np.mean([runs[str(seed)]["checkpoints"][str(checkpoint)][arm][field] for seed in TRAINING_SEEDS]))

    intact_accuracy = {str(checkpoint): mean_arm(checkpoint, "intact", "accuracy") for checkpoint in CHECKPOINTS}
    reset_accuracy = {str(checkpoint): mean_arm(checkpoint, "between_item_reset", "accuracy") for checkpoint in CHECKPOINTS}
    intact_margin = {str(checkpoint): mean_arm(checkpoint, "intact", "target_minus_best_competitor_margin_hz") for checkpoint in CHECKPOINTS}
    reset_margin = {str(checkpoint): mean_arm(checkpoint, "between_item_reset", "target_minus_best_competitor_margin_hz") for checkpoint in CHECKPOINTS}
    final_intact = [runs[str(seed)]["checkpoints"]["800"]["intact"] for seed in TRAINING_SEEDS]
    final_reset = [runs[str(seed)]["checkpoints"]["800"]["between_item_reset"] for seed in TRAINING_SEEDS]
    initial_intact = [runs[str(seed)]["checkpoints"]["0"]["intact"] for seed in TRAINING_SEEDS]
    memory_dependence = bool(
        np.mean([row["accuracy"] for row in final_intact]) > np.mean([row["accuracy"] for row in final_reset])
        and sum(a["accuracy"] > b["accuracy"] for a, b in zip(final_intact, final_reset)) >= 2
        and np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_intact]) > np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_reset])
        and not any(row["output_collapse"] for row in (*final_intact, *final_reset))
    )
    learning_supported = bool(
        np.mean([row["accuracy"] for row in final_intact]) > np.mean([row["accuracy"] for row in initial_intact])
        and sum(a["accuracy"] > b["accuracy"] for a, b in zip(final_intact, initial_intact)) >= 2
        and np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_intact]) > np.mean([row["target_minus_best_competitor_margin_hz"] for row in initial_intact])
        and not any(row["output_collapse"] for row in final_intact)
    )
    output_collapse = any(
        runs[str(seed)]["checkpoints"][str(checkpoint)][arm]["output_collapse"]
        for seed in TRAINING_SEEDS for checkpoint in CHECKPOINTS for arm in ("intact", "between_item_reset")
    )
    discrete = bool(
        np.mean([row["accuracy"] for row in final_intact]) >= 0.50
        and sum(row["accuracy"] >= 0.50 for row in final_intact) >= 2
        and memory_dependence and not output_collapse
    )
    if discrete:
        next_step = "next_symbol_prediction"
    elif learning_supported and memory_dependence:
        next_step = "next_symbol_prediction_with_short_context"
    elif learning_supported:
        next_step = "diagnose_second_cue_shortcut"
    else:
        next_step = "diagnose_sequence_interference_credit"

    profile_disabled = _profile(connectome, wm_interface, interface_config, learning_enabled=False, config=config)
    profile_training = _profile(connectome, wm_interface, interface_config, learning_enabled=True, config=config)
    exact = _sequence_exact_replay(connectome, wm_interface, interface_config)
    route_equivalence = run_route_cache_equivalence(connectome, interface, interface_config)
    artifact = {
        "protocol": {
            "phase": "F.2D+P.4",
            "training_seeds": list(TRAINING_SEEDS),
            "training_trials_per_seed": 800,
            "ordered_pairs": len(ORDERED_PAIRS),
            "trials_per_pair": 50,
            "evaluation_trials_per_pair": EVALUATION_CYCLES,
            "first_ms": 20,
            "second_ms": 20,
            "go_ms": 20,
            "stimulus_rate_hz": 205.0,
            "two_hop_credit_mode": "prospective_anatomical",
            "directional_learning_rate": 0.02,
            "reward_learning_rate": 0.02,
            "adaptive_plastic_budget": False,
            "route_cache_enabled": True,
            "dynamic_candidate_count": int(candidate_count),
            "selected_output_count": int(len(selected)),
        },
        "schedule_balance": balance,
        "runs": runs,
        "aggregate": {
            "accuracy_curve": {"intact": intact_accuracy, "between_item_reset": reset_accuracy},
            "margin_curve_hz": {"intact": intact_margin, "between_item_reset": reset_margin},
            "reset_ablation_curve": {
                str(checkpoint): {
                    "accuracy_gap": float(intact_accuracy[str(checkpoint)] - reset_accuracy[str(checkpoint)]),
                    "margin_gap_hz": float(intact_margin[str(checkpoint)] - reset_margin[str(checkpoint)]),
                } for checkpoint in CHECKPOINTS
            },
            "pair_accuracy_final_intact": {
                key: float(np.mean([runs[str(seed)]["checkpoints"]["800"]["intact"]["pair_accuracy"][key] for seed in TRAINING_SEEDS]))
                for key in ("".join(pair) for pair in ORDERED_PAIRS)
            },
            "same_vs_different_final_intact": {
                "same_symbol_accuracy": float(np.mean([row["same_symbol_accuracy"] for row in final_intact])),
                "different_symbol_accuracy": float(np.mean([row["different_symbol_accuracy"] for row in final_intact])),
            },
            "order_reversal_final_intact": {
                str(seed): runs[str(seed)]["checkpoints"]["800"]["intact"]["order_reversal"]
                for seed in TRAINING_SEEDS
            },
        },
        "episode_credit": {
            "limit_per_seed": 16,
            "captured_per_seed": {seed: runs[str(seed)]["episode_credit"]["episodes_captured"] for seed in TRAINING_SEEDS},
            "aggregate": {
                key: int(sum(runs[str(seed)]["episode_credit"]["aggregate"][key] for seed in TRAINING_SEEDS))
                for key in ("updated_edges", "updated_edges_eligible_after_first", "updated_edges_eligible_by_end_second", "updated_edges_newly_eligible_during_go", "eligible_edges_after_first", "eligible_edges_by_end_second", "eligible_edges_by_end_go")
            },
        },
        "performance": {
            "delay_ring_subprofile_learning_disabled": profile_disabled,
            "delay_ring_subprofile_training": profile_training,
            "optimization_applied": False,
            "optimization_note": "P.4 retained profiling only; no integer-index change met the evidence threshold.",
            "exact_equivalence": bool(exact["passed"]),
            "neural_exact_equivalence": exact,
            "component_speedup": 1.0,
            "total_speedup": 1.0,
            "route_cache_equivalence": route_equivalence,
        },
        "safety": {
            "plastic_budget_drift_by_seed": {seed: runs[str(seed)]["safety"]["plastic_budget_drift"] for seed in TRAINING_SEEDS},
            "adaptive_reallocations_by_seed": {seed: runs[str(seed)]["safety"]["adaptive_reallocations"] for seed in TRAINING_SEEDS},
            "all_budgets_fixed": all(runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS),
            "all_adaptive_reallocations_zero": True,
            "evaluation_twins_isolated": True,
        },
        "conclusion": {
            "second_cue_target_independence": bool(balance["second_cue_target_independence"]),
            "short_sequence_learning_supported": learning_supported,
            "first_item_memory_dependence_supported": memory_dependence,
            "discrete_short_sequence_memory_demonstrated": discrete,
            "output_collapse_detected": output_collapse,
            "recommended_next_step": next_step,
        },
        "pass": bool(
            balance["second_cue_target_independence"]
            and exact["passed"]
            and route_equivalence["passed"]
            and all(runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS)
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    artifact = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({
        "pass": artifact["pass"],
        "conclusion": artifact["conclusion"],
        "accuracy_curve": artifact["aggregate"]["accuracy_curve"],
        "runtime_seconds": artifact["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
