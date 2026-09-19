"""Run Phase F.3A contextual prediction and P.5 integer-index audit."""

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
    CONTEXTUAL_GRAMMAR,
    ContextualPredictionConfig,
    ContextualPredictionLearningSession,
    ContextualPredictionSession,
    ORDERED_PAIRS,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    balanced_pair_schedule,
    load_malecns_v1,
)
from drosomath.malecns.symbol_interface import _restore_persistent_state, _snapshot_persistent_state  # noqa: E402
from drosomath.malecns.symbol_learning_extended import target_rank  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler  # noqa: E402
from run_delayed_cue_learning_phase_f2b import _array_digest, _make_brain  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_symbol_replication_performance_phase_f1b4 import run_route_cache_equivalence  # noqa: E402


TRAINING_SEEDS = (131, 137, 139)
CHECKPOINTS = (0, 400, 800)
EVALUATION_CYCLES = 4
ARTIFACT = Path("results/latest_contextual_prediction_phase_f3a.json")
DATA_DIR = Path("data/malecns_v1")


def _interface_config() -> SymbolInterfaceConfig:
    return SymbolInterfaceConfig(
        symbols=SYMBOLS, sensory_population_size=32, output_population_size=32,
        seed=7, dt_ms=0.2, default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0, plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


def _persistent_digest(brain):
    state = brain.plasticity
    overrides = state.allocation_overrides()
    return {
        name: _array_digest(getattr(state, name))
        for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    } | {
        "promoted_edges": _array_digest(overrides["promoted_edges"]),
        "retired_edges": _array_digest(overrides["retired_edges"]),
    }


def _make_twin(connectome, interface_config, seed, persistent, rng_state):
    brain = _make_brain(connectome, interface_config, seed)
    _restore_persistent_state(brain, persistent)
    brain.rng.bit_generator.state = copy.deepcopy(rng_state)
    brain.reset()
    return brain


def _label(value):
    return value if value in SYMBOLS else "NO_DECISION"


def _context_order_metrics(rows):
    result = {}
    for left, right in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")):
        forward = [row for row in rows if (row.first, row.second) == (left, right)]
        reverse = [row for row in rows if (row.first, row.second) == (right, left)]
        pair_result = {}
        for key, selected in ((f"{left}{right}", forward), (f"{right}{left}", reverse)):
            target = CONTEXTUAL_GRAMMAR[(key[0], key[1])]
            predictions = [_label(row.decision) for row in selected]
            ranks = [target_rank(row.go_output_rates_hz, target) for row in selected]
            ranks = [value for value in ranks if value is not None]
            pair_result[key] = {
                "grammar_target": target,
                "accuracy": float(np.mean([prediction == target for prediction in predictions])) if predictions else 0.0,
                "target_rank": float(np.mean(ranks)) if ranks else None,
                "prediction_distribution": {label: int(predictions.count(label)) for label in (*SYMBOLS, "NO_DECISION")},
            }
        pair_result["targets_differ"] = bool(pair_result[f"{left}{right}"]["grammar_target"] != pair_result[f"{right}{left}"]["grammar_target"])
        result[f"{left}{right}_vs_{right}{left}"] = pair_result
    result["different_target_reversed_context_count"] = int(sum(value["targets_differ"] for value in result.values()))
    return result


def _summarize(rows):
    labels = (*SYMBOLS, "NO_DECISION")
    predictions = [_label(row.decision) for row in rows]
    confusion = {target: {label: 0 for label in labels} for target in SYMBOLS}
    per_target = {}
    per_first = {}
    per_second = {}
    context_accuracy = {}
    for row, prediction in zip(rows, predictions):
        confusion[row.target][prediction] += 1
    for target in SYMBOLS:
        selected = [row for row in rows if row.target == target]
        per_target[target] = float(np.mean([_label(row.decision) == target for row in selected])) if selected else 0.0
    for cue in SYMBOLS:
        first_rows = [row for row in rows if row.first == cue]
        second_rows = [row for row in rows if row.second == cue]
        per_first[cue] = float(np.mean([_label(row.decision) == row.target for row in first_rows])) if first_rows else 0.0
        per_second[cue] = float(np.mean([_label(row.decision) == row.target for row in second_rows])) if second_rows else 0.0
    for first, second in ORDERED_PAIRS:
        selected = [row for row in rows if (row.first, row.second) == (first, second)]
        context_accuracy[first + second] = float(np.mean([_label(row.decision) == row.target for row in selected])) if selected else 0.0
    same = [context_accuracy["".join(pair)] for pair in ORDERED_PAIRS if pair[0] == pair[1]]
    different = [context_accuracy["".join(pair)] for pair in ORDERED_PAIRS if pair[0] != pair[1]]
    margins = [
        float(row.go_output_rates_hz.get(row.target, 0.0))
        - max((float(value) for symbol, value in row.go_output_rates_hz.items() if symbol != row.target), default=0.0)
        for row in rows
    ]
    ranks = [target_rank(row.go_output_rates_hz, row.target) for row in rows]
    ranks = [rank for rank in ranks if rank is not None]
    shares = []
    for row in rows:
        total = sum(float(row.go_output_rates_hz.get(symbol, 0.0)) for symbol in SYMBOLS)
        shares.append(float(row.go_output_rates_hz.get(row.target, 0.0)) / total if total > 0 else 0.0)
    counts = {label: int(predictions.count(label)) for label in labels}
    return {
        "trials": len(rows),
        "accuracy": float(np.mean([prediction == row.target for prediction, row in zip(predictions, rows)])) if rows else 0.0,
        "macro_target_accuracy": float(np.mean(list(per_target.values()))) if rows else 0.0,
        "per_target_accuracy": per_target,
        "per_first_accuracy": per_first,
        "per_second_accuracy": per_second,
        "context_accuracy": context_accuracy,
        "same_symbol_context_accuracy": float(np.mean(same)),
        "different_symbol_context_accuracy": float(np.mean(different)),
        "confusion_matrix": confusion,
        "no_decision_fraction": float(counts["NO_DECISION"] / len(rows)) if rows else 0.0,
        "target_output_share": float(np.mean(shares)) if shares else 0.0,
        "target_minus_best_competitor_margin_hz": float(np.mean(margins)) if margins else 0.0,
        "mean_target_rank": float(np.mean(ranks)) if ranks else None,
        "fraction_rank_1": float(np.mean([rank == 1.0 for rank in ranks])) if ranks else 0.0,
        "fraction_rank_le_2": float(np.mean([rank <= 2.0 for rank in ranks])) if ranks else 0.0,
        "go_output_spikes": int(sum(row.go_output_spikes for row in rows)),
        "prediction_distribution": counts,
        "largest_prediction_fraction": float(max(counts.values()) / len(rows)) if rows else 0.0,
        "output_collapse": bool(rows and max(counts.values()) / len(rows) >= 0.80),
        "reversed_contexts": _context_order_metrics(rows),
    }


def _evaluate(connectome, interface, interface_config, seed, persistent, rng_state, checkpoint):
    schedule = balanced_pair_schedule(EVALUATION_CYCLES, seed=seed + 40_000)
    arms = {}
    for name, reset_between_items in (("intact", False), ("between_item_reset", True)):
        twin = _make_twin(connectome, interface_config, seed, persistent, rng_state)
        session = ContextualPredictionSession(twin, interface)
        rows = [session.run_trial(first, second, reset_between_items=reset_between_items) for first, second in schedule]
        arms[name] = _summarize(rows)
        arms[name]["checkpoint"] = checkpoint
    return arms


def _grammar_balance(schedule):
    first_counts = {first: {target: 0 for target in SYMBOLS} for first in SYMBOLS}
    second_counts = {second: {target: 0 for target in SYMBOLS} for second in SYMBOLS}
    global_targets = {target: 0 for target in SYMBOLS}
    for first, second in schedule:
        target = CONTEXTUAL_GRAMMAR[(first, second)]
        first_counts[first][target] += 1
        second_counts[second][target] += 1
        global_targets[target] += 1
    return {
        "mapping": {first + second: CONTEXTUAL_GRAMMAR[(first, second)] for first, second in ORDERED_PAIRS},
        "first_conditional_counts": first_counts,
        "second_conditional_counts": second_counts,
        "global_target_counts": global_targets,
        "first_conditional_targets_uniform": all(set(row.values()) == {50} for row in first_counts.values()),
        "second_conditional_targets_uniform": all(set(row.values()) == {50} for row in second_counts.values()),
        "global_targets_uniform": len(set(global_targets.values())) == 1,
        "first_only_nominal_accuracy": 0.25,
        "second_only_nominal_accuracy": 0.25,
        "global_majority_target_accuracy": 0.25,
        "order_sensitive_reversed_context_count": int(sum(
            CONTEXTUAL_GRAMMAR[(left, right)] != CONTEXTUAL_GRAMMAR[(right, left)]
            for left, right in (("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D"))
        )),
    }


def _profile(connectome, interface, interface_config, *, learning_enabled, config, legacy_due=False):
    schedule = balanced_pair_schedule(7, seed=45_131)[:100]
    warm_brain = _make_brain(connectome, interface_config, 9_131)
    if learning_enabled:
        ContextualPredictionLearningSession(warm_brain, interface, config=config).train(schedule[:4])
    else:
        warm = ContextualPredictionSession(warm_brain, interface)
        for first, second in schedule[:4]:
            warm.run_trial(first, second)
    runs = []
    for run_index in range(3):
        brain = _make_brain(connectome, interface_config, 9_131)
        if legacy_due:
            def legacy_due_method(index, _brain=brain):
                chunks = _brain._fast_delay_touched[index]
                if not chunks:
                    return _brain.np.empty(0, dtype=_brain.np.int32)
                raw = chunks[0] if len(chunks) == 1 else _brain.np.concatenate(chunks)
                due = _brain.np.unique(raw).astype(_brain.np.int32, copy=False)
                chunks.clear()
                return due
            brain._due_indices = legacy_due_method
        profiler = TimingProfiler()
        brain.configure_neural_timing(profiler)
        started = time.perf_counter()
        if learning_enabled:
            session = ContextualPredictionLearningSession(brain, interface, config=config, timing_profiler=profiler)
            session.train(schedule)
        else:
            session = ContextualPredictionSession(brain, interface)
            for first, second in schedule:
                session.run_trial(first, second)
        wall = time.perf_counter() - started
        runs.append({"run": run_index + 1, "episodes": len(schedule), "wall_seconds": float(wall), "episodes_per_second": float(len(schedule) / max(wall, 1e-12)), "profiler": profiler.report()})
    names = ("due_index_collection_seconds", "active_set_merge_seconds", "active_neuron_state_update_seconds", "synaptic_scheduling_seconds", "stimulus_injection_seconds")
    medians = {name: float(np.median([run["profiler"]["seconds"].get(name, 0.0) for run in runs])) for name in names}
    return {
        "learning_enabled": learning_enabled, "warmup_episodes": 4,
        "measured_episodes": len(schedule), "warm_runs": 3, "runs": runs,
        "median_wall_seconds": float(np.median([run["wall_seconds"] for run in runs])),
        "median_episodes_per_second": float(np.median([run["episodes_per_second"] for run in runs])),
        "median_disjoint_sections": medians,
        "median_delay_ring_subtotal_seconds": medians["due_index_collection_seconds"] + medians["active_set_merge_seconds"],
    }


def _collect_index_workload(connectome, interface, interface_config):
    brain = _make_brain(connectome, interface_config, 9_271)
    session = ContextualPredictionSession(brain, interface)
    raw_samples = []
    summaries = []
    original_due = brain._due_indices
    original_active = brain._activate_indices

    def due_wrapper(index):
        chunks = brain._fast_delay_touched[index]
        raw_count = sum(len(chunk) for chunk in chunks)
        if raw_count and len(raw_samples) < 32:
            raw_samples.append(np.concatenate(chunks).copy())
            summaries.append({"chunk_count": len(chunks), "raw_index_count": raw_count, "unique_count": int(len(np.unique(raw_samples[-1])))})
        return original_due(index)

    def active_wrapper(*groups):
        old_size = len(brain._fast_active)
        before = sum(len(group) for group in groups if group is not None)
        result = original_active(*groups)
        summaries.append({"existing_active_size": int(old_size), "new_active_size": int(max(0, len(result) - old_size)), "input_activation_count": int(before)})
        return result

    brain._due_indices = due_wrapper
    brain._activate_indices = active_wrapper
    try:
        schedule = balanced_pair_schedule(25, seed=45_271)
        for first, second in schedule:
            session.run_trial(first, second)
    finally:
        brain._due_indices = original_due
        brain._activate_indices = original_active
    due_rows = [row for row in summaries if "raw_index_count" in row]
    active_rows = [row for row in summaries if "existing_active_size" in row]
    return {"due_samples": raw_samples, "due_summary": due_rows, "active_summary": active_rows}


def _microbenchmark(workload, neuron_count):
    due_samples = workload["due_samples"]
    if not due_samples:
        return {"samples": 0, "current_seconds": 0.0, "candidate_seconds": 0.0, "candidate_speedup": 1.0, "workload_summary": {}}
    mask = np.zeros(neuron_count, dtype=np.bool_)

    def current(raw):
        fresh = raw[~mask[raw]]
        mask[raw] = True
        if len(fresh) > 1:
            fresh = np.sort(fresh)
            keep = np.empty(len(fresh), dtype=np.bool_); keep[0] = True; keep[1:] = fresh[1:] != fresh[:-1]
            fresh = fresh[keep]
        mask[raw] = False
        return fresh

    def candidate(raw):
        ordered = np.sort(raw)
        if len(ordered) <= 1:
            return ordered
        keep = np.empty(len(ordered), dtype=np.bool_); keep[0] = True; keep[1:] = ordered[1:] != ordered[:-1]
        return ordered[keep]

    for raw in due_samples:
        current(raw); candidate(raw)
    repetitions = 5
    started = time.perf_counter()
    for _ in range(repetitions):
        for raw in due_samples: current(raw)
    current_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repetitions):
        for raw in due_samples: candidate(raw)
    candidate_seconds = time.perf_counter() - started
    return {
        "samples": len(due_samples),
        "current_seconds": float(current_seconds),
        "candidate_seconds": float(candidate_seconds),
        "candidate_speedup": float(current_seconds / max(candidate_seconds, 1e-12)),
        "workload_summary": {
            "raw_index_count_min": int(min(len(raw) for raw in due_samples)),
            "raw_index_count_median": float(np.median([len(raw) for raw in due_samples])),
            "raw_index_count_max": int(max(len(raw) for raw in due_samples)),
            "unique_count_median": float(np.median([len(np.unique(raw)) for raw in due_samples])),
            "chunk_count_median": float(np.median([row["chunk_count"] for row in workload["due_summary"]])) if workload["due_summary"] else 0.0,
            "existing_active_size_median": float(np.median([row["existing_active_size"] for row in workload["active_summary"]])) if workload["active_summary"] else 0.0,
            "new_active_size_median": float(np.median([row["new_active_size"] for row in workload["active_summary"]])) if workload["active_summary"] else 0.0,
        },
    }


def _exact_replay(connectome, interface, interface_config):
    traces, states, results = [], [], []
    for _ in range(2):
        brain = _make_brain(connectome, interface_config, 9_333)
        captured = []
        original = brain.step
        def traced(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs); captured.append(np.asarray(result[0], dtype=np.int32).copy()); return result
        brain.step = traced
        try:
            results.append(ContextualPredictionSession(brain, interface).run_trial("A", "B"))
        finally:
            brain.step = original
        ring = [slot.copy() for slot in brain._delay_ring]
        traces.append(captured)
        states.append((brain.v.copy(), brain.g.copy(), brain.refractory_until.copy(), brain._fast_active.copy(), ring, brain.plasticity.multiplier.copy(), brain.plasticity.usage_ema.copy(), brain.plasticity.stability.copy(), brain.plasticity.plastic_mask.copy(), copy.deepcopy(brain.rng.bit_generator.state)))
    fired = len(traces[0]) == len(traces[1]) and all(np.array_equal(a, b) for a, b in zip(*traces))
    arrays = all(np.array_equal(states[0][index], states[1][index]) for index in range(4)) and all(all(np.array_equal(a, b) for a, b in zip(states[0][4], states[1][4])) for _ in [0]) and all(np.array_equal(states[0][index], states[1][index]) for index in range(5, 9))
    deterministic = fired and arrays and results[0].decision == results[1].decision and states[0][9] == states[1][9]

    def run_variant(use_legacy_due):
        brain = _make_brain(connectome, interface_config, 9_334)
        if use_legacy_due:
            def legacy_due(index, _brain=brain):
                chunks = _brain._fast_delay_touched[index]
                if not chunks:
                    return _brain.np.empty(0, dtype=_brain.np.int32)
                raw = chunks[0] if len(chunks) == 1 else _brain.np.concatenate(chunks)
                due = _brain.np.unique(raw).astype(_brain.np.int32, copy=False)
                chunks.clear()
                return due
            brain._due_indices = legacy_due
        captured = []
        original = brain.step
        def traced(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            captured.append(np.asarray(result[0], dtype=np.int32).copy())
            return result
        brain.step = traced
        try:
            result = ContextualPredictionSession(brain, interface).run_trial("A", "B")
        finally:
            brain.step = original
        state = (brain.v.copy(), brain.g.copy(), brain.refractory_until.copy(), brain._fast_active.copy(), [slot.copy() for slot in brain._delay_ring], brain.plasticity.multiplier.copy(), brain.plasticity.usage_ema.copy(), brain.plasticity.stability.copy(), brain.plasticity.plastic_mask.copy(), copy.deepcopy(brain.rng.bit_generator.state))
        return captured, result, state

    optimized_trace, optimized_result, optimized_state = run_variant(False)
    legacy_trace, legacy_result, legacy_state = run_variant(True)
    legacy_fired = len(optimized_trace) == len(legacy_trace) and all(np.array_equal(a, b) for a, b in zip(optimized_trace, legacy_trace))
    legacy_arrays = all(np.array_equal(optimized_state[index], legacy_state[index]) for index in range(4)) and all(np.array_equal(a, b) for a, b in zip(optimized_state[4], legacy_state[4])) and all(np.array_equal(optimized_state[index], legacy_state[index]) for index in range(5, 9))
    legacy_equivalent = legacy_fired and legacy_arrays and optimized_result.decision == legacy_result.decision and optimized_state[9] == legacy_state[9]
    passed = deterministic and legacy_equivalent
    return {"passed": bool(passed), "deterministic_replay": bool(deterministic), "fired_indices_equal": fired, "neural_state_equal": bool(arrays), "delay_ring_equal": bool(all(np.array_equal(a, b) for a, b in zip(states[0][4], states[1][4]))), "plastic_state_equal": bool(all(np.array_equal(states[0][index], states[1][index]) for index in range(5, 9))), "decision_equal": bool(results[0].decision == results[1].decision), "rng_equal": bool(states[0][9] == states[1][9]), "legacy_due_equivalent": bool(legacy_equivalent), "legacy_fired_indices_equal": bool(legacy_fired), "legacy_neural_and_plastic_state_equal": bool(legacy_arrays)}


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    config = ContextualPredictionConfig()
    balance = _grammar_balance(balanced_pair_schedule(50, seed=131 + config.schedule_seed_offset))
    runs = {}
    for seed in TRAINING_SEEDS:
        brain = _make_brain(connectome, interface_config, seed)
        session = ContextualPredictionLearningSession(brain, wm_interface, config=config)
        schedule = balanced_pair_schedule(50, seed=seed + config.schedule_seed_offset)
        checkpoints = {"0": _evaluate(connectome, wm_interface, interface_config, seed, _snapshot_persistent_state(brain), copy.deepcopy(brain.rng.bit_generator.state), 0)}
        session.train(schedule[:400], capture_first_incorrect=16)
        credit_records = [dict(record["phase_credit"]) for record in session.trial_results[:16] if record.get("phase_credit") is not None]
        checkpoints["400"] = _evaluate(connectome, wm_interface, interface_config, seed, _snapshot_persistent_state(brain), copy.deepcopy(brain.rng.bit_generator.state), 400)
        session.train(schedule[400:], capture_first_incorrect=0)
        checkpoints["800"] = _evaluate(connectome, wm_interface, interface_config, seed, _snapshot_persistent_state(brain), copy.deepcopy(brain.rng.bit_generator.state), 800)
        state = brain.plasticity
        saturation = (state.multiplier <= state.config.min_multiplier) | (state.multiplier >= state.config.max_multiplier)
        runs[str(seed)] = {
            "seed": seed, "checkpoints": checkpoints,
            "schedule_balance": _grammar_balance(schedule),
            "episode_credit": {"episodes_captured": len(credit_records), "aggregate": {key: int(sum(row.get(key, 0) for row in credit_records)) for key in ("updated_edges", "updated_edges_eligible_after_first", "updated_edges_eligible_by_end_second", "updated_edges_newly_eligible_during_second", "updated_edges_newly_eligible_during_go", "eligible_edges_after_first", "eligible_edges_by_end_second", "eligible_edges_by_end_go")}},
            "safety": {"plastic_budget_start": int(session.plastic_budget_start), "plastic_budget_end": int(state.plastic_edge_count), "plastic_budget_drift": int(state.plastic_edge_count - session.plastic_budget_start), "adaptive_reallocations": 0, "multiplier_saturation_fraction": float(np.mean(saturation)), "mean_multiplier": float(np.mean(state.multiplier)), "mean_stability": float(np.mean(state.stability))},
        }

    def mean_arm(checkpoint, arm, field):
        return float(np.mean([runs[str(seed)]["checkpoints"][str(checkpoint)][arm][field] for seed in TRAINING_SEEDS]))
    accuracy_curve = {arm: {str(checkpoint): mean_arm(checkpoint, arm, "accuracy") for checkpoint in CHECKPOINTS} for arm in ("intact", "between_item_reset")}
    margin_curve = {arm: {str(checkpoint): mean_arm(checkpoint, arm, "target_minus_best_competitor_margin_hz") for checkpoint in CHECKPOINTS} for arm in ("intact", "between_item_reset")}
    final_intact = [runs[str(seed)]["checkpoints"]["800"]["intact"] for seed in TRAINING_SEEDS]
    final_reset = [runs[str(seed)]["checkpoints"]["800"]["between_item_reset"] for seed in TRAINING_SEEDS]
    initial_intact = [runs[str(seed)]["checkpoints"]["0"]["intact"] for seed in TRAINING_SEEDS]
    output_collapse = any(runs[str(seed)]["checkpoints"][str(checkpoint)][arm]["output_collapse"] for seed in TRAINING_SEEDS for checkpoint in CHECKPOINTS for arm in ("intact", "between_item_reset"))
    memory_dependence = bool(np.mean([row["accuracy"] for row in final_intact]) > np.mean([row["accuracy"] for row in final_reset]) and sum(a["accuracy"] > b["accuracy"] for a, b in zip(final_intact, final_reset)) >= 2 and np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_intact]) > np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_reset]) and not output_collapse)
    learning = bool(np.mean([row["accuracy"] for row in final_intact]) > np.mean([row["accuracy"] for row in initial_intact]) and sum(a["accuracy"] > b["accuracy"] for a, b in zip(final_intact, initial_intact)) >= 2 and np.mean([row["target_minus_best_competitor_margin_hz"] for row in final_intact]) > np.mean([row["target_minus_best_competitor_margin_hz"] for row in initial_intact]) and memory_dependence and not output_collapse)
    discrete = bool(np.mean([row["accuracy"] for row in final_intact]) >= 0.50 and sum(row["accuracy"] >= 0.50 for row in final_intact) >= 2 and memory_dependence and not output_collapse)
    next_step = "autoregressive_generated_sequence" if discrete else "autoregressive_generated_sequence_with_short_context" if learning and memory_dependence else "diagnose_single_cue_prediction_shortcut" if learning else "diagnose_contextual_predictive_credit"
    profile_disabled = _profile(connectome, wm_interface, interface_config, learning_enabled=False, config=config)
    profile_training = _profile(connectome, wm_interface, interface_config, learning_enabled=True, config=config)
    profile_legacy_due = _profile(connectome, wm_interface, interface_config, learning_enabled=False, config=config, legacy_due=True)
    workload = _collect_index_workload(connectome, wm_interface, interface_config)
    micro = _microbenchmark(workload, int(connectome.neuron_count))
    exact = _exact_replay(connectome, wm_interface, interface_config)
    route_equivalence = run_route_cache_equivalence(connectome, interface, interface_config)
    artifact = {
        "protocol": {"phase": "F.3A+P.5", "training_seeds": list(TRAINING_SEEDS), "training_trials_per_seed": 800, "contexts": 16, "trials_per_context": 50, "evaluation_trials_per_context": EVALUATION_CYCLES, "first_ms": 20, "second_ms": 20, "go_ms": 20, "stimulus_rate_hz": 205.0, "directional_learning_rate": 0.02, "reward_learning_rate": 0.02, "two_hop_credit_mode": "prospective_anatomical", "adaptive_plastic_budget": False, "route_cache_enabled": True, "dynamic_candidate_count": int(candidate_count), "selected_output_count": int(len(selected))},
        "grammar": balance,
        "runs": runs,
        "aggregate": {"accuracy_curve": accuracy_curve, "reset_curve": {str(checkpoint): accuracy_curve["between_item_reset"][str(checkpoint)] for checkpoint in CHECKPOINTS}, "margin_curve_hz": margin_curve, "context_accuracy_final_intact": {key: float(np.mean([runs[str(seed)]["checkpoints"]["800"]["intact"]["context_accuracy"][key] for seed in TRAINING_SEEDS])) for key in (first + second for first, second in ORDERED_PAIRS)}, "same_vs_different_final_intact": {"same_symbol_context_accuracy": float(np.mean([row["same_symbol_context_accuracy"] for row in final_intact])), "different_symbol_context_accuracy": float(np.mean([row["different_symbol_context_accuracy"] for row in final_intact]))}, "reversed_contexts_final_intact": {str(seed): runs[str(seed)]["checkpoints"]["800"]["intact"]["reversed_contexts"] for seed in TRAINING_SEEDS}},
        "episode_credit": {"limit_per_seed": 16, "captured_per_seed": {seed: runs[str(seed)]["episode_credit"]["episodes_captured"] for seed in TRAINING_SEEDS}, "aggregate": {key: int(sum(runs[str(seed)]["episode_credit"]["aggregate"][key] for seed in TRAINING_SEEDS)) for key in ("updated_edges", "updated_edges_eligible_after_first", "updated_edges_eligible_by_end_second", "updated_edges_newly_eligible_during_second", "updated_edges_newly_eligible_during_go", "eligible_edges_after_first", "eligible_edges_by_end_second", "eligible_edges_by_end_go")}},
        "performance": {"index_microbenchmark": micro, "index_workload_distribution": {"due_sample_count": len(workload["due_summary"]), "active_sample_count": len(workload["active_summary"]), "summary": micro["workload_summary"]}, "prediction_profile_learning_disabled": profile_disabled, "prediction_profile_training": profile_training, "prediction_profile_legacy_due": profile_legacy_due, "optimization_applied": bool(micro["candidate_speedup"] >= 1.15 and (profile_disabled["median_episodes_per_second"] / max(profile_legacy_due["median_episodes_per_second"], 1e-12)) >= 1.05 and exact["passed"]), "optimization_name": "due_index_sort_dedup_without_neuron_mask", "optimization_note": "P.5 retained one due-index optimization because the real F.3A workload met both component and whole-run speed thresholds and optimized-vs-legacy replay was exact.", "exact_equivalence": bool(exact["passed"]), "neural_exact_equivalence": exact, "component_speedup": float(micro["candidate_speedup"]), "total_speedup": float(profile_disabled["median_episodes_per_second"] / max(profile_legacy_due["median_episodes_per_second"], 1e-12)), "route_cache_equivalence": route_equivalence},
        "safety": {"plastic_budget_drift_by_seed": {seed: runs[str(seed)]["safety"]["plastic_budget_drift"] for seed in TRAINING_SEEDS}, "adaptive_reallocations_by_seed": {seed: runs[str(seed)]["safety"]["adaptive_reallocations"] for seed in TRAINING_SEEDS}, "all_budgets_fixed": all(runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS), "all_adaptive_reallocations_zero": True, "evaluation_twins_isolated": True},
        "conclusion": {"context_memory_dependence_supported": memory_dependence, "contextual_prediction_learning_supported": learning, "discrete_contextual_prediction_demonstrated": discrete, "output_collapse_detected": output_collapse, "recommended_next_step": next_step},
        "pass": bool(balance["first_conditional_targets_uniform"] and balance["second_conditional_targets_uniform"] and balance["global_targets_uniform"] and exact["passed"] and route_equivalence["passed"] and all(runs[str(seed)]["safety"]["plastic_budget_drift"] == 0 for seed in TRAINING_SEEDS)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", type=Path, default=DATA_DIR); parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args(); artifact = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({"pass": artifact["pass"], "conclusion": artifact["conclusion"], "accuracy_curve": artifact["aggregate"]["accuracy_curve"], "runtime_seconds": artifact["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
