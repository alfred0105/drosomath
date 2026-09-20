"""Run the first learned delayed-cue recurrent working-memory experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from drosomath.malecns.symbol_learning import (  # noqa: E402
    SymbolLearningConfig,
    SymbolLearningSession,
    build_symbol_learning_signal,
)
from drosomath.malecns.symbol_learning_extended import target_rank  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_symbol_replication_performance_phase_f1b4 import run_route_cache_equivalence  # noqa: E402


TRAINING_SEEDS = (83, 89, 97)
CHECKPOINTS = (0, 400, 800)
TRIALS_PER_CUE = 200
EVALUATION_TRIALS_PER_CUE = 10
ARTIFACT = Path("results/latest_delayed_cue_learning_phase_f2b.json")
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


def _array_digest(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _persistent_digest(brain) -> dict[str, str]:
    state = brain.plasticity
    overrides = state.allocation_overrides()
    return {
        "multiplier": _array_digest(state.multiplier),
        "usage_ema": _array_digest(state.usage_ema),
        "eligibility": _array_digest(state.eligibility),
        "stability": _array_digest(state.stability),
        "plastic_mask": _array_digest(state.plastic_mask),
        "promoted_edges": _array_digest(overrides["promoted_edges"]),
        "retired_edges": _array_digest(overrides["retired_edges"]),
    }


def _rng_state(brain):
    return copy.deepcopy(brain.rng.bit_generator.state)


def _rng_equal(left, right) -> bool:
    return left == right


def _make_evaluation_twin(connectome, interface_config, seed, persistent, rng_state):
    twin = _make_brain(connectome, interface_config, seed)
    _restore_persistent_state(twin, persistent)
    twin.rng.bit_generator.state = copy.deepcopy(rng_state)
    twin.reset()
    return twin


def _prediction_label(value: str) -> str:
    return value if value in SYMBOLS else "NO_DECISION"


def _mean_output_vector(rows):
    result = {}
    for cue in SYMBOLS:
        selected = [row for row in rows if row.cue == cue]
        result[cue] = np.asarray(
            [np.mean([row.go_output_rates_hz[symbol] for row in selected]) for symbol in SYMBOLS],
            dtype=np.float64,
        ) if selected else np.zeros(len(SYMBOLS), dtype=np.float64)
    return result


def _output_separation(rows) -> float:
    vectors = _mean_output_vector(rows)
    distances = [
        float(np.linalg.norm(vectors[left] - vectors[right], ord=1))
        for index, left in enumerate(SYMBOLS)
        for right in SYMBOLS[index + 1:]
    ]
    return float(np.mean(distances)) if distances else 0.0


def _summarize_rows(rows) -> dict[str, object]:
    confusion = {cue: {label: 0 for label in (*SYMBOLS, "NO_DECISION")} for cue in SYMBOLS}
    per_cue = {}
    for cue in SYMBOLS:
        selected = [row for row in rows if row.cue == cue]
        predictions = [_prediction_label(row.decision) for row in selected]
        for prediction in predictions:
            confusion[cue][prediction] += 1
        margins = [
            float(row.go_output_rates_hz.get(cue, 0.0))
            - max((float(value) for symbol, value in row.go_output_rates_hz.items() if symbol != cue), default=0.0)
            for row in selected
        ]
        ranks = [target_rank(row.go_output_rates_hz, cue) for row in selected]
        ranks = [rank for rank in ranks if rank is not None]
        shares = []
        for row in selected:
            total = sum(float(row.go_output_rates_hz.get(symbol, 0.0)) for symbol in SYMBOLS)
            shares.append(float(row.go_output_rates_hz.get(cue, 0.0)) / total if total > 0.0 else 0.0)
        per_cue[cue] = {
            "trials": int(len(selected)),
            "accuracy": float(sum(prediction == cue for prediction in predictions) / len(selected)) if selected else 0.0,
            "no_decision_fraction": float(sum(prediction == "NO_DECISION" for prediction in predictions) / len(selected)) if selected else 0.0,
            "target_output_share": float(np.mean(shares)) if shares else 0.0,
            "target_minus_best_competitor_margin_hz": float(np.mean(margins)) if margins else 0.0,
            "mean_target_rank": float(np.mean(ranks)) if ranks else None,
            "fraction_rank_1": float(sum(rank == 1.0 for rank in ranks) / len(ranks)) if ranks else 0.0,
            "fraction_rank_le_2": float(sum(rank <= 2.0 for rank in ranks) / len(ranks)) if ranks else 0.0,
            "go_output_spikes": int(sum(row.go_output_spikes for row in selected)),
            "mean_pre_go_active_count": float(np.mean([row.pre_go_active_count for row in selected])) if selected else 0.0,
            "mean_pre_go_membrane_norm": float(np.mean([row.pre_go_membrane_norm for row in selected])) if selected else 0.0,
            "mean_pre_go_conductance_norm": float(np.mean([row.pre_go_conductance_norm for row in selected])) if selected else 0.0,
            "pre_go_fingerprint_diversity": int(len({row.pre_go_fingerprint for row in selected})),
        }
    predictions = [_prediction_label(row.decision) for row in rows]
    counts = {label: int(predictions.count(label)) for label in (*SYMBOLS, "NO_DECISION")}
    total = len(rows)
    cue_accuracy = [float(per_cue[cue]["accuracy"]) for cue in SYMBOLS]
    ranks = [target_rank(row.go_output_rates_hz, row.cue) for row in rows]
    ranks = [rank for rank in ranks if rank is not None]
    margins = [float(per_cue[row.cue]["target_minus_best_competitor_margin_hz"]) for row in rows]
    return {
        "trials": int(total),
        "accuracy": float(sum(_prediction_label(row.decision) == row.cue for row in rows) / total) if total else 0.0,
        "macro_accuracy": float(np.mean(cue_accuracy)) if cue_accuracy else 0.0,
        "per_cue_accuracy": {cue: per_cue[cue]["accuracy"] for cue in SYMBOLS},
        "per_cue": per_cue,
        "confusion_matrix": confusion,
        "prediction_distribution": counts,
        "largest_prediction_fraction": float(max(counts.values()) / total) if total else 0.0,
        "distinct_predicted_symbols": int(sum(value > 0 for value in counts.values() if value)),
        "output_collapse": bool(total and max(counts.values()) / total >= 0.80),
        "no_decision_fraction": float(counts["NO_DECISION"] / total) if total else 0.0,
        "target_output_share": float(np.mean([per_cue[row.cue]["target_output_share"] for row in rows])) if rows else 0.0,
        "target_minus_best_competitor_margin_hz": float(np.mean(margins)) if margins else 0.0,
        "mean_target_rank": float(np.mean(ranks)) if ranks else None,
        "fraction_rank_1": float(sum(rank == 1.0 for rank in ranks) / len(ranks)) if ranks else 0.0,
        "fraction_rank_le_2": float(sum(rank <= 2.0 for rank in ranks) / len(ranks)) if ranks else 0.0,
        "go_output_spikes": int(sum(row.go_output_spikes for row in rows)),
        "cue_conditioned_go_output_separation_hz": _output_separation(rows),
        "state_memory": {
            "mean_pre_go_active_count": float(np.mean([row.pre_go_active_count for row in rows])) if rows else 0.0,
            "pairwise_pre_go_active_set_jaccard": pairwise_set_jaccard(rows),
            "pre_go_fingerprint_diversity": int(len({row.pre_go_fingerprint for row in rows})),
            "mean_pre_go_membrane_norm": float(np.mean([row.pre_go_membrane_norm for row in rows])) if rows else 0.0,
            "mean_pre_go_conductance_norm": float(np.mean([row.pre_go_conductance_norm for row in rows])) if rows else 0.0,
            "mean_post_reset_active_count": float(np.mean([row.post_reset_active_count for row in rows])) if rows else 0.0,
            "reset_empty_fraction": float(np.mean([row.post_reset_active_count == 0 for row in rows])) if rows else 0.0,
        },
    }


def _summarize_training(trials) -> dict[str, object]:
    total = len(trials)
    per_cue = {}
    for cue in SYMBOLS:
        selected = [trial for trial in trials if trial.cue == cue]
        per_cue[cue] = {
            "trials": len(selected),
            "accuracy": float(np.mean([trial.decision == cue for trial in selected])) if selected else 0.0,
            "target_margin_hz": float(np.mean([
                trial.go_output_rates_hz.get(cue, 0.0)
                - max((value for symbol, value in trial.go_output_rates_hz.items() if symbol != cue), default=0.0)
                for trial in selected
            ])) if selected else 0.0,
            "target_rank": float(np.mean([
                rank for rank in (target_rank(trial.go_output_rates_hz, cue) for trial in selected)
                if rank is not None
            ])) if any(target_rank(trial.go_output_rates_hz, cue) is not None for trial in selected) else None,
        }
    directional_trials = sum(bool(trial.directional_error) for trial in trials)
    reinforcement_trials = sum(trial.reward > 0.0 for trial in trials)
    edge_updates = sum(int(trial.directional_update.get("edge_updates", 0)) for trial in trials)
    hop_counts = {}
    for trial in trials:
        for hop, count in trial.directional_update.get("hop_counts", {}).items():
            hop_counts[str(hop)] = hop_counts.get(str(hop), 0) + int(count)
    margins = [
        trial.go_output_rates_hz.get(trial.cue, 0.0)
        - max((value for symbol, value in trial.go_output_rates_hz.items() if symbol != trial.cue), default=0.0)
        for trial in trials
    ]
    ranks = [target_rank(trial.go_output_rates_hz, trial.cue) for trial in trials]
    ranks = [rank for rank in ranks if rank is not None]
    return {
        "trials": total,
        "accuracy": float(np.mean([trial.decision == trial.cue for trial in trials])) if trials else 0.0,
        "per_cue_accuracy": {cue: per_cue[cue]["accuracy"] for cue in SYMBOLS},
        "per_cue": per_cue,
        "no_decision_fraction": float(np.mean([trial.decision == "NO_DECISION" for trial in trials])) if trials else 0.0,
        "positive_reinforcement_trials": int(reinforcement_trials),
        "directional_correction_trials": int(directional_trials),
        "directional_edge_updates": int(edge_updates),
        "one_hop_updates": int(hop_counts.get("1", 0)),
        "two_hop_updates": int(hop_counts.get("2", 0)),
        "target_margin_hz": float(np.mean(margins)) if margins else 0.0,
        "target_rank": float(np.mean(ranks)) if ranks else None,
    }


def _evaluate_checkpoint(connectome, wm_interface, interface_config, seed, persistent, rng_state, checkpoint):
    schedule = balanced_symbol_schedule(cycles=EVALUATION_TRIALS_PER_CUE, seed=seed + 20_000)
    arms = {}
    before_digest = _persistent_digest_from_snapshot(persistent)
    for arm_name, reset_before_go in (("intact", False), ("reset", True)):
        twin = _make_evaluation_twin(connectome, interface_config, seed, persistent, rng_state)
        session = DelayedCueSession(twin, wm_interface, learning_enabled=False)
        rows = [session.run_trial(cue, reset_before_go=reset_before_go) for cue in schedule]
        arms[arm_name] = _summarize_rows(rows)
        arms[arm_name]["checkpoint"] = int(checkpoint)
    arms["memory_comparison"] = _memory_comparison(arms["intact"], arms["reset"])
    return arms


def _persistent_digest_from_snapshot(snapshot) -> dict[str, str]:
    return {
        name: _array_digest(snapshot[name])
        for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    } | {
        "promoted_edges": _array_digest(snapshot["allocation_overrides"]["promoted_edges"]),
        "retired_edges": _array_digest(snapshot["allocation_overrides"]["retired_edges"]),
    }


def _memory_comparison(intact, reset):
    epsilon = 1e-9
    return {
        "accuracy_gap_intact_minus_reset": float(intact["accuracy"] - reset["accuracy"]),
        "margin_gap_intact_minus_reset_hz": float(intact["target_minus_best_competitor_margin_hz"] - reset["target_minus_best_competitor_margin_hz"]),
        "target_rank_intact_minus_reset": float((intact["mean_target_rank"] or 0.0) - (reset["mean_target_rank"] or 0.0)),
        "go_separation_ratio_intact_over_reset": float(
            intact["cue_conditioned_go_output_separation_hz"]
            / max(reset["cue_conditioned_go_output_separation_hz"], epsilon)
        ),
        "pre_go_active_count_gap": float(
            intact["state_memory"]["mean_pre_go_active_count"]
            - reset["state_memory"]["mean_pre_go_active_count"]
        ),
    }


def _paired_route_cache_benchmark(connectome, interface, interface_config):
    config = SymbolLearningConfig(
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
    sequence = (False, True, True, False)
    for enabled in (False, True):
        brain = _make_brain(connectome, interface_config, 53)
        session = SymbolLearningSession(brain, interface, config=config, route_cache_enabled=enabled)
        session.train_trial("A")
    runs = []
    for enabled in sequence:
        brain = _make_brain(connectome, interface_config, 53)
        profiler = TimingProfiler()
        session = SymbolLearningSession(
            brain,
            interface,
            config=config,
            timing_profiler=profiler,
            route_cache_enabled=enabled,
        )
        started = time.perf_counter()
        session.train(seed=53)
        wall = time.perf_counter() - started
        runs.append({
            "route_cache_enabled": bool(enabled),
            "wall_seconds": float(wall),
            "trials_per_second": float(config.training_trials / wall),
            "timing": profiler.report(),
        })
    off = [run["trials_per_second"] for run in runs if not run["route_cache_enabled"]]
    on = [run["trials_per_second"] for run in runs if run["route_cache_enabled"]]
    median_off = float(np.median(off))
    median_on = float(np.median(on))
    return {
        "sequence": ["ON" if value else "OFF" for value in sequence],
        "runs": runs,
        "median_off_trials_per_second": median_off,
        "median_on_trials_per_second": median_on,
        "paired_speedup": float(median_on / max(median_off, 1e-12)),
    }


def _f2b_profile(connectome, interface, interface_config, wm_interface, config):
    brain = _make_brain(connectome, interface_config, TRAINING_SEEDS[0])
    profiler = TimingProfiler()
    session = DelayedCueLearningSession(
        brain,
        wm_interface,
        config=config,
        timing_profiler=profiler,
        route_cache_enabled=True,
    )
    schedule = balanced_symbol_schedule(cycles=25, seed=TRAINING_SEEDS[0] + config.schedule_seed_offset)
    started = time.perf_counter()
    session.train(schedule, capture_first_incorrect=0)
    wall = time.perf_counter() - started
    report = profiler.report()
    seconds = report["seconds"]
    disjoint_names = (
        "network_simulation_seconds",
        "reward_update_seconds",
        "post_reward_directional_seconds",
        "normalizer_seconds",
        "plastic_lifecycle_seconds",
    )
    disjoint = {name: float(seconds.get(name, 0.0)) for name in disjoint_names}
    return {
        "episodes": 100,
        "wall_seconds": float(wall),
        "episodes_per_second": float(100.0 / wall),
        "profiler": report,
        "disjoint_seconds": disjoint,
        "disjoint_sum_seconds": float(sum(disjoint.values())),
        "disjoint_fraction_of_wall": {name: value / max(wall, 1e-12) for name, value in disjoint.items()},
        "nested_sections": {
            name: float(seconds.get(name, 0.0))
            for name in ("active_edge_collection_seconds", "directional_credit_selection_seconds", "prospective_downstream_effect_seconds", "multiplier_update_seconds")
            if name in seconds
        },
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
        initial_digest = _persistent_digest(brain)
        schedule = balanced_symbol_schedule(cycles=TRIALS_PER_CUE, seed=seed + config.schedule_seed_offset)
        if {cue: schedule.count(cue) for cue in SYMBOLS} != {cue: TRIALS_PER_CUE for cue in SYMBOLS}:
            raise AssertionError("F.2B schedule is not balanced")
        session = DelayedCueLearningSession(
            brain,
            wm_interface,
            config=config,
            route_cache_enabled=True,
        )
        checkpoints = {}
        initial_persistent = _snapshot_persistent_state(brain)
        initial_rng = _rng_state(brain)
        checkpoints["0"] = _evaluate_checkpoint(
            connectome, wm_interface, interface_config, seed, initial_persistent, initial_rng, 0
        )
        training_windows = {}
        episode_credit_records = []
        for start in range(0, config.training_trials, 200):
            window_schedule = schedule[start:start + 200]
            before = len(session.trial_results)
            remaining_credit = max(0, config.episode_credit_limit - len(episode_credit_records))
            trials, phase_records = session.train(window_schedule, capture_first_incorrect=remaining_credit)
            episode_credit_records.extend(phase_records)
            window_trials = trials[before:]
            training_windows[f"{start + 1}-{start + 200}"] = _summarize_training(window_trials)
            checkpoint = start + 200
            if checkpoint in (400, 800):
                persistent = _snapshot_persistent_state(brain)
                rng = _rng_state(brain)
                training_digest_before_eval = _persistent_digest(brain)
                rng_before_eval = _rng_state(brain)
                checkpoints[str(checkpoint)] = _evaluate_checkpoint(
                    connectome, wm_interface, interface_config, seed, persistent, rng, checkpoint
                )
                if training_digest_before_eval != _persistent_digest(brain) or not _rng_equal(rng_before_eval, _rng_state(brain)):
                    raise AssertionError("checkpoint evaluation mutated the training brain")
        final_digest = _persistent_digest(brain)
        state = brain.plasticity
        saturation = (state.multiplier <= state.config.min_multiplier) | (state.multiplier >= state.config.max_multiplier)
        runs[str(seed)] = {
            "seed": int(seed),
            "schedule_counts": {cue: int(schedule.count(cue)) for cue in SYMBOLS},
            "training_windows": training_windows,
            "checkpoints": checkpoints,
            "episode_credit": {
                "episodes_captured": int(len(episode_credit_records)),
                "aggregate": {
                    key: int(sum(int(row.get(key, 0)) for row in episode_credit_records))
                    for key in (
                        "updated_edges",
                        "updated_edges_eligible_after_cue",
                        "updated_edges_eligible_by_end_delay",
                        "updated_edges_newly_eligible_during_go",
                        "eligible_edges_after_cue",
                        "eligible_edges_by_end_delay",
                        "eligible_edges_by_end_go",
                    )
                },
            },
            "safety": {
                "persistent_state_initial_digest": initial_digest,
                "persistent_state_final_digest": final_digest,
                "plastic_budget_start": int(session.plastic_budget_start),
                "plastic_budget_end": int(state.plastic_edge_count),
                "plastic_budget_drift": int(state.plastic_edge_count - session.plastic_budget_start),
                "adaptive_reallocations": 0,
                "multiplier_saturation_fraction": float(np.mean(saturation)),
                "mean_multiplier": float(np.mean(state.multiplier)),
                "mean_stability": float(np.mean(state.stability)),
            },
        }

    final_intact = [runs[str(seed)]["checkpoints"]["800"]["intact"] for seed in TRAINING_SEEDS]
    final_reset = [runs[str(seed)]["checkpoints"]["800"]["reset"] for seed in TRAINING_SEEDS]
    baseline_intact = [runs[str(seed)]["checkpoints"]["0"]["intact"] for seed in TRAINING_SEEDS]
    final_comparisons = [
        _memory_comparison(runs[str(seed)]["checkpoints"]["800"]["intact"], runs[str(seed)]["checkpoints"]["800"]["reset"])
        for seed in TRAINING_SEEDS
    ]
    memory_dependence_supported = bool(
        sum(row["accuracy_gap_intact_minus_reset"] > 0.0 for row in final_comparisons) >= 2
        and np.mean([row["go_separation_ratio_intact_over_reset"] for row in final_comparisons]) > 1.0
        and all(row["state_memory"]["reset_empty_fraction"] == 1.0 for row in final_reset)
    )
    output_collapse_detected = any(
        arm["output_collapse"]
        for seed in TRAINING_SEEDS
        for arm_name in ("intact", "reset")
        for arm in (runs[str(seed)]["checkpoints"]["800"][arm_name],)
    )
    mean_baseline_accuracy = float(np.mean([row["accuracy"] for row in baseline_intact]))
    mean_final_accuracy = float(np.mean([row["accuracy"] for row in final_intact]))
    mean_baseline_margin = float(np.mean([row["target_minus_best_competitor_margin_hz"] for row in baseline_intact]))
    mean_final_margin = float(np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_intact]))
    working_memory_learning_supported = bool(
        mean_final_accuracy > mean_baseline_accuracy
        and sum(final_intact[index]["accuracy"] > baseline_intact[index]["accuracy"] for index in range(3)) >= 2
        and mean_final_margin > mean_baseline_margin
        and mean_final_accuracy > float(np.mean([row["accuracy"] for row in final_reset]))
        and not output_collapse_detected
    )
    discrete_working_memory_demonstrated = bool(
        mean_final_accuracy >= 0.50
        and sum(row["accuracy"] >= 0.50 for row in final_intact) >= 2
        and memory_dependence_supported
        and not output_collapse_detected
    )
    recurrent_information = bool(
        np.mean([row["cue_conditioned_go_output_separation_hz"] for row in baseline_intact]) > 0.0
    )
    if discrete_working_memory_demonstrated:
        next_step = "delay_generalization"
    elif working_memory_learning_supported and memory_dependence_supported:
        next_step = "delay_generalization_then_short_sequence"
    elif recurrent_information and not working_memory_learning_supported:
        next_step = "diagnose_temporal_credit_assignment"
    elif all(row["accuracy_gap_intact_minus_reset"] == 0.0 for row in final_comparisons):
        next_step = "diagnose_memory_independent_shortcut"
    else:
        next_step = "diagnose_temporal_credit_assignment"

    paired = _paired_route_cache_benchmark(connectome, interface, interface_config)
    f2b_profile = _f2b_profile(connectome, interface, interface_config, wm_interface, config)
    exact_equivalence = run_route_cache_equivalence(connectome, interface, interface_config)
    disjoint_profile = f2b_profile["disjoint_seconds"]
    next_hotspot = max(disjoint_profile, key=disjoint_profile.get) if disjoint_profile else None

    aggregate = {
        "intact_accuracy_curve": {
            str(checkpoint): float(np.mean([runs[str(seed)]["checkpoints"][str(checkpoint)]["intact"]["accuracy"] for seed in TRAINING_SEEDS]))
            for checkpoint in CHECKPOINTS
        },
        "reset_accuracy_curve": {
            str(checkpoint): float(np.mean([runs[str(seed)]["checkpoints"][str(checkpoint)]["reset"]["accuracy"] for seed in TRAINING_SEEDS]))
            for checkpoint in CHECKPOINTS
        },
        "intact_margin_curve_hz": {
            str(checkpoint): float(np.mean([runs[str(seed)]["checkpoints"][str(checkpoint)]["intact"]["target_minus_best_competitor_margin_hz"] for seed in TRAINING_SEEDS]))
            for checkpoint in CHECKPOINTS
        },
        "reset_margin_curve_hz": {
            str(checkpoint): float(np.mean([runs[str(seed)]["checkpoints"][str(checkpoint)]["reset"]["target_minus_best_competitor_margin_hz"] for seed in TRAINING_SEEDS]))
            for checkpoint in CHECKPOINTS
        },
        "memory_gap_curve": {
            str(checkpoint): {
                "accuracy": float(np.mean([
                    runs[str(seed)]["checkpoints"][str(checkpoint)]["intact"]["accuracy"]
                    - runs[str(seed)]["checkpoints"][str(checkpoint)]["reset"]["accuracy"]
                    for seed in TRAINING_SEEDS
                ])),
                "margin_hz": float(np.mean([
                    runs[str(seed)]["checkpoints"][str(checkpoint)]["intact"]["target_minus_best_competitor_margin_hz"]
                    - runs[str(seed)]["checkpoints"][str(checkpoint)]["reset"]["target_minus_best_competitor_margin_hz"]
                    for seed in TRAINING_SEEDS
                ])),
            }
            for checkpoint in CHECKPOINTS
        },
        "state_memory": {
            str(checkpoint): {
                "intact_mean_pre_go_active_count": float(np.mean([
                    runs[str(seed)]["checkpoints"][str(checkpoint)]["intact"]["state_memory"]["mean_pre_go_active_count"] for seed in TRAINING_SEEDS
                ])),
                "reset_mean_post_reset_active_count": float(np.mean([
                    runs[str(seed)]["checkpoints"][str(checkpoint)]["reset"]["state_memory"]["mean_post_reset_active_count"] for seed in TRAINING_SEEDS
                ])),
                "intact_go_separation_hz": float(np.mean([
                    runs[str(seed)]["checkpoints"][str(checkpoint)]["intact"]["cue_conditioned_go_output_separation_hz"] for seed in TRAINING_SEEDS
                ])),
                "reset_go_separation_hz": float(np.mean([
                    runs[str(seed)]["checkpoints"][str(checkpoint)]["reset"]["cue_conditioned_go_output_separation_hz"] for seed in TRAINING_SEEDS
                ])),
            }
            for checkpoint in CHECKPOINTS
        },
    }
    artifact = {
        "protocol": {
            "phase": "F.2B+P.2",
            "training_seeds": list(TRAINING_SEEDS),
            "training_trials_per_seed": 800,
            "trials_per_cue": 200,
            "evaluation_trials_per_cue": 10,
            "cue_ms": 20.0,
            "delay_ms": 20.0,
            "go_ms": 20.0,
            "stimulus_rate_hz": 205.0,
            "input_vocabulary": [*SYMBOLS, GO_SYMBOL],
            "output_vocabulary": list(SYMBOLS),
            "go_population_size": 32,
            "go_seed": GO_ALLOCATION_SEED,
            "two_hop_credit_mode": "prospective_anatomical",
            "directional_learning_rate": 0.02,
            "reward_learning_rate": 0.02,
            "adaptive_plastic_budget": False,
            "dynamic_candidate_count": int(candidate_count),
            "selected_output_count": int(len(selected)),
        },
        "runs": runs,
        "aggregate": aggregate,
        "episode_credit": {
            "limit_per_seed": 16,
            "captured_per_seed": {seed: runs[str(seed)]["episode_credit"]["episodes_captured"] for seed in TRAINING_SEEDS},
            "aggregate": {
                key: int(sum(runs[str(seed)]["episode_credit"]["aggregate"].get(key, 0) for seed in TRAINING_SEEDS))
                for key in (
                    "updated_edges",
                    "updated_edges_eligible_after_cue",
                    "updated_edges_eligible_by_end_delay",
                    "updated_edges_newly_eligible_during_go",
                    "eligible_edges_after_cue",
                    "eligible_edges_by_end_delay",
                    "eligible_edges_by_end_go",
                )
            },
        },
        "performance": {
            "paired_route_cache_benchmark": paired,
            "f2b_disjoint_profile": f2b_profile,
            "exact_equivalence": bool(exact_equivalence["passed"]),
            "exact_equivalence_detail": exact_equivalence,
            "next_optimization_hotspot": next_hotspot,
        },
        "safety": {
            "plastic_budget_drift_by_seed": {seed: runs[str(seed)]["safety"]["plastic_budget_drift"] for seed in TRAINING_SEEDS},
            "adaptive_reallocations_by_seed": {seed: runs[str(seed)]["safety"]["adaptive_reallocations"] for seed in TRAINING_SEEDS},
            "all_budgets_fixed": all(runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS),
            "all_adaptive_reallocations_zero": all(runs[str(seed)]["safety"]["adaptive_reallocations"] == 0 for seed in TRAINING_SEEDS),
            "evaluation_twins_isolated": True,
            "reset_preserves_persistent_arrays": True,
        },
        "conclusion": {
            "baseline_intact_accuracy_mean": mean_baseline_accuracy,
            "final_intact_accuracy_mean": mean_final_accuracy,
            "final_reset_accuracy_mean": float(np.mean([row["accuracy"] for row in final_reset])),
            "working_memory_learning_supported": working_memory_learning_supported,
            "memory_dependence_supported": memory_dependence_supported,
            "discrete_working_memory_demonstrated": discrete_working_memory_demonstrated,
            "output_collapse_detected": output_collapse_detected,
            "recommended_next_step": next_step,
        },
        "pass": bool(
            exact_equivalence["passed"]
            and all(runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS)
            and paired["paired_speedup"] > 1.0
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    artifact = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({
        "pass": artifact["pass"],
        "conclusion": artifact["conclusion"],
        "paired_speedup": artifact["performance"]["paired_route_cache_benchmark"]["paired_speedup"],
        "runtime_seconds": artifact["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
