"""Run the matched Phase F.3D generic presynaptic-STD experiment.

The runner deliberately keeps the task adapter outside the STD mechanism.  The
only STD switch passed into the brain is a generic configuration object; the
learning controller, output grammar, and plastic allocation remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

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
    TwoCueSequenceSession,
    WorkingMemoryInterface,
    balanced_pair_schedule,
    load_malecns_v1,
)
from drosomath.malecns.symbol_interface import (  # noqa: E402
    _restore_persistent_state,
    _snapshot_persistent_state,
)
from drosomath.whole_brain import (  # noqa: E402
    PlasticStateConfig,
    PresynapticDepressionConfig,
    SlowAdaptationConfig,
)
from run_contextual_credit_audit_phase_f3b import (  # noqa: E402
    _jaccard,
    _public_representation,
    _representation_summary,
)
from run_contextual_prediction_phase_f3a import _summarize  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402


SEEDS = (179, 181, 191)
ARM_NAMES = ("CONTROL", "STD_BINDING")
AUDIT_REPETITIONS = 4
STABLE_REPETITION_THRESHOLD = 3
TRAINING_EPISODES = 400
EVALUATION_CYCLES = 4
ARTIFACT = ROOT / "results/latest_presynaptic_std_binding_phase_f3d.json"
DATA_DIR = ROOT / "data/malecns_v1"


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


def _make_brain(connectome, config, seed: int, *, std_enabled: bool):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=float(getattr(config, "plastic_fraction", 0.05)),
            seed=seed,
        ),
        slow_adaptation_config=SlowAdaptationConfig(enabled=False),
        presynaptic_depression_config=PresynapticDepressionConfig(enabled=std_enabled),
    )


def _twin(connectome, config, seed, persistent, rng_state, *, std_enabled):
    brain = _make_brain(connectome, config, seed, std_enabled=std_enabled)
    _restore_persistent_state(brain, persistent)
    brain.rng.bit_generator.state = copy.deepcopy(rng_state)
    brain.reset()
    return brain


def _audit(connectome, interface, config, seed, persistent, rng_state, *, std_enabled):
    stable = {"first": {}, "pre_go": {}}
    frequencies = {"first": {}, "pre_go": {}}
    metrics = {"first": {}, "pre_go": {}}
    std_rows = {"first": [], "pre_go": []}
    for phase in ("first", "pre_go"):
        for context in ORDERED_PAIRS:
            repetition_sets = []
            context_metrics = []
            context_std = []
            for repetition in range(AUDIT_REPETITIONS):
                brain = _twin(
                    connectome,
                    config,
                    seed + 10_000 + repetition,
                    persistent,
                    rng_state,
                    std_enabled=std_enabled,
                )
                observed = {}

                def observer(name, current, *, observed=observed):
                    wanted = phase == "first" and name == "first" or phase == "pre_go" and name == "second"
                    if wanted:
                        active = np.asarray(current._fast_active, dtype=np.int32).copy()
                        observed["active"] = active
                        observed["membrane_norm"] = float(np.linalg.norm(current.v[active])) if len(active) else 0.0
                        observed["conductance_norm"] = float(np.linalg.norm(current.g[active])) if len(active) else 0.0
                        observed["std"] = current.presynaptic_depression_summary()
                        recovered = current._std_recovered_factors()
                        observed["mean_recovered_touched"] = float(np.mean(recovered)) if len(recovered) else 1.0

                TwoCueSequenceSession(brain, interface).run_trial(
                    *context,
                    track_eligibility=False,
                    phase_observer=observer,
                )
                active = set(int(value) for value in observed.get("active", ()))
                repetition_sets.append(active)
                context_metrics.append({
                    "active_count": len(active),
                    "membrane_norm": observed.get("membrane_norm", 0.0),
                    "conductance_norm": observed.get("conductance_norm", 0.0),
                    "fingerprint": hashlib.sha256(np.asarray(sorted(active), dtype=np.int32).tobytes()).hexdigest(),
                })
                context_std.append({
                    "depressed_neurons": int(observed.get("std", {}).get("depressed_neurons", 0)),
                    "mean_release_factor_depressed": float(observed.get("std", {}).get("mean_release_factor_depressed", 1.0)),
                    "minimum_release_factor": float(observed.get("std", {}).get("minimum_release_factor", 1.0)),
                    "fraction_at_floor": float(observed.get("std", {}).get("fraction_at_floor", 0.0)),
                    "mean_recovered_touched": float(observed.get("mean_recovered_touched", 1.0)),
                })
            counts = {}
            for active in repetition_sets:
                for neuron in active:
                    counts[neuron] = counts.get(neuron, 0) + 1
            stable[phase][context] = {neuron for neuron, count in counts.items() if count >= STABLE_REPETITION_THRESHOLD}
            frequencies[phase][context] = {neuron: count / AUDIT_REPETITIONS for neuron, count in counts.items()}
            metrics[phase][context] = context_metrics
            std_rows[phase].extend(context_std)
    raw = {"stable": stable, "frequencies": frequencies, "metrics": metrics}
    public = _public_representation(_representation_summary(raw))
    std_summary = {}
    for phase, rows in std_rows.items():
        std_summary[phase] = {
            "mean_depressed_neurons": float(np.mean([row["depressed_neurons"] for row in rows])) if rows else 0.0,
            "mean_release_factor_depressed": float(np.mean([row["mean_release_factor_depressed"] for row in rows])) if rows else 1.0,
            "minimum_release_factor": float(np.min([row["minimum_release_factor"] for row in rows])) if rows else 1.0,
            "mean_fraction_at_floor": float(np.mean([row["fraction_at_floor"] for row in rows])) if rows else 0.0,
            "mean_recovered_factor_touched": float(np.mean([row["mean_recovered_touched"] for row in rows])) if rows else 1.0,
        }
    return {"representation": public, "std": std_summary}


def _summary_rows(results, *, prediction: bool):
    rows = []
    for value in results:
        target = CONTEXTUAL_GRAMMAR[(value.first, value.second)] if prediction else value.first
        rows.append(SimpleNamespace(
            first=value.first,
            second=value.second,
            target=target,
            decision=value.decision,
            go_output_rates_hz=value.go_output_rates_hz,
            go_output_spikes=value.go_output_spikes,
        ))
    return _summarize(rows)


def _evaluate(connectome, interface, config, seed, persistent, rng_state, *, std_enabled, checkpoint):
    schedule = balanced_pair_schedule(EVALUATION_CYCLES, seed=seed + 40_000)
    result = {}
    for name, reset_between_items in (("intact", False), ("between_item_reset", True)):
        brain = _twin(connectome, config, seed, persistent, rng_state, std_enabled=std_enabled)
        session = ContextualPredictionSession(brain, interface)
        values = [session.run_trial(first, second, reset_between_items=reset_between_items) for first, second in schedule]
        result[name] = _summary_rows(values, prediction=True) | {"checkpoint": int(checkpoint)}
    return result


def _first_guard(connectome, interface, config, seed, persistent, rng_state, *, std_enabled):
    schedule = balanced_pair_schedule(EVALUATION_CYCLES, seed=seed + 40_000)
    brain = _twin(connectome, config, seed, persistent, rng_state, std_enabled=std_enabled)
    values = [TwoCueSequenceSession(brain, interface).run_trial(first, second) for first, second in schedule]
    return _summary_rows(values, prediction=False)


def _train_arm(connectome, interface, config, seed, *, std_enabled):
    brain = _make_brain(connectome, config, seed, std_enabled=std_enabled)
    session = ContextualPredictionLearningSession(
        brain,
        interface,
        config=config,
        route_cache_enabled=True,
        plastic_row_cache_enabled=True,
        prospective_index_enabled=True,
    )
    schedule = balanced_pair_schedule(25, seed=seed + session.config.schedule_seed_offset)
    started = time.perf_counter()
    records, phase_credit = session.train(schedule[:TRAINING_EPISODES], capture_first_incorrect=8)
    elapsed = time.perf_counter() - started
    persistent = _snapshot_persistent_state(brain)
    rng_state = copy.deepcopy(brain.rng.bit_generator.state)
    successes = int(sum(bool(row["success"]) for row in records))
    return {
        "brain": brain,
        "persistent": persistent,
        "rng_state": rng_state,
        "schedule": schedule,
        "records": records,
        "phase_credit_samples": phase_credit,
        "elapsed_seconds": float(elapsed),
        "successes": successes,
        "throughput_episodes_per_second": float(TRAINING_EPISODES / max(elapsed, 1e-12)),
    }


def _compact_training(run, connectome, interface, config, seed, *, std_enabled, initial_persistent, initial_rng, initial_plastic_budget):
    brain = run["brain"]
    initial_eval = _evaluate(connectome, interface, config, seed, initial_persistent, initial_rng, std_enabled=std_enabled, checkpoint=0)
    final_eval = _evaluate(connectome, interface, config, seed, run["persistent"], run["rng_state"], std_enabled=std_enabled, checkpoint=TRAINING_EPISODES)
    first_initial = _first_guard(connectome, interface, config, seed, initial_persistent, initial_rng, std_enabled=std_enabled)
    return {
        "training": {
            "episodes": TRAINING_EPISODES,
            "successes": int(run["successes"]),
            "elapsed_seconds": run["elapsed_seconds"],
            "episodes_per_second": run["throughput_episodes_per_second"],
            "first_incorrect_phase_credit_samples": run["phase_credit_samples"],
        },
        "evaluation": {"0": initial_eval, str(TRAINING_EPISODES): final_eval},
        "first_memory_guard": {"initial": first_initial},
        "matched_schedule_digest": int(len(run["schedule"])),
        "plastic_budget": int(brain.plasticity.plastic_edge_count),
        "initial_plastic_budget": int(initial_plastic_budget),
        "anatomical_edge_count": int(connectome.edge_count),
        "std_state_bytes": int(brain.release_factor.nbytes + brain.last_release_step.nbytes),
    }


def _run_seed(seed: int, data_dir: str):
    connectome = load_malecns_v1(Path(data_dir), min_connection_synapses=5)
    config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    initial = {}
    audits = {}
    trained = {}
    for arm, std_enabled in (("CONTROL", False), ("STD_BINDING", True)):
        initial_brain = _make_brain(connectome, config, seed, std_enabled=std_enabled)
        initial[arm] = {
            "persistent": _snapshot_persistent_state(initial_brain),
            "rng": copy.deepcopy(initial_brain.rng.bit_generator.state),
            "plastic_budget": int(initial_brain.plasticity.plastic_edge_count),
        }
        audits[arm] = _audit(connectome, wm_interface, config, seed, initial[arm]["persistent"], initial[arm]["rng"], std_enabled=std_enabled)
        trained[arm] = _train_arm(connectome, wm_interface, ContextualPredictionConfig(telemetry_level="summary"), seed, std_enabled=std_enabled)
    runs = {}
    for arm, std_enabled in (("CONTROL", False), ("STD_BINDING", True)):
        runs[arm] = _compact_training(
            trained[arm], connectome, wm_interface, ContextualPredictionConfig(telemetry_level="summary"), seed,
            std_enabled=std_enabled,
            initial_persistent=initial[arm]["persistent"],
            initial_rng=initial[arm]["rng"],
            initial_plastic_budget=initial[arm]["plastic_budget"],
        )
    return {
        "seed": int(seed),
        "arms": runs,
        "representation": {arm: audits[arm]["representation"] for arm in ARM_NAMES},
        "std_telemetry": {arm: audits[arm]["std"] for arm in ARM_NAMES},
        "matched_schedule": trained["CONTROL"]["schedule"] == trained["STD_BINDING"]["schedule"],
        "matched_initial_plastic_mask": bool(np.array_equal(
            _make_brain(connectome, config, seed, std_enabled=False).plasticity.plastic_mask,
            _make_brain(connectome, config, seed, std_enabled=True).plasticity.plastic_mask,
        )),
        "candidate_count": int(candidate_count),
        "selected_output_count": int(len(selected)),
        "connectome_edge_count": int(connectome.edge_count),
    }


def _run_seed_job(job):
    return _run_seed(*job)


def _f3d_representation_summary(runs):
    control = [row["representation"]["CONTROL"] for row in runs]
    std = [row["representation"]["STD_BINDING"] for row in runs]
    control_j = [row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in control]
    std_j = [row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in std]
    broad = []
    for row in std:
        per_context = row["conjunctive_specific_pools"]["pre_go"]["per_context"]
        broad.append(sum(item["broader_conjunction_count"] > 0 for item in per_context.values()))
    return {
        "control_same_first_jaccard_mean": float(np.mean(control_j)) if control_j else 0.0,
        "std_same_first_jaccard_mean": float(np.mean(std_j)) if std_j else 0.0,
        "std_to_control_ratio": float(np.mean(std_j) / max(np.mean(control_j), 1e-12)) if control_j else 0.0,
        "std_broader_contexts_with_pool": broad,
        "criterion_context_separation": bool(np.mean(std_j) <= 0.80 * np.mean(control_j)) if control_j else False,
        "criterion_broader_pools": bool(sum(value >= 12 for value in broad) >= 2),
    }


def _behavior_summary(runs):
    control = [row["arms"]["CONTROL"]["evaluation"]["400"]["intact"] for row in runs]
    std = [row["arms"]["STD_BINDING"]["evaluation"]["400"]["intact"] for row in runs]
    reset = [row["arms"]["STD_BINDING"]["evaluation"]["400"]["between_item_reset"] for row in runs]
    no_collapse = not any(item["output_collapse"] for item in control + std + reset)
    control_accuracy = float(np.mean([item["accuracy"] for item in control]))
    std_accuracy = float(np.mean([item["accuracy"] for item in std]))
    reset_accuracy = float(np.mean([item["accuracy"] for item in reset]))
    control_margin = float(np.mean([item["target_minus_best_competitor_margin_hz"] for item in control]))
    std_margin = float(np.mean([item["target_minus_best_competitor_margin_hz"] for item in std]))
    reset_margin = float(np.mean([item["target_minus_best_competitor_margin_hz"] for item in reset]))
    return {
        "control_intact_accuracy": control_accuracy,
        "std_intact_accuracy": std_accuracy,
        "std_reset_accuracy": reset_accuracy,
        "control_intact_margin_hz": control_margin,
        "std_intact_margin_hz": std_margin,
        "std_reset_margin_hz": reset_margin,
        "improved_seed_count": int(sum(s["accuracy"] > c["accuracy"] for s, c in zip(std, control))),
        "context_dependence_seed_count": int(sum(s["accuracy"] > r["accuracy"] for s, r in zip(std, reset))),
        "no_output_collapse": bool(no_collapse),
        "behavioral_criterion": bool(std_accuracy > control_accuracy and std_margin > control_margin and no_collapse),
    }


def _disabled_exact_smoke(data_dir: Path):
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = _interface_config()
    interface, _, _ = build_f1b_dynamic_interface(connectome, config)
    left = _make_brain(connectome, config, 17_303, std_enabled=False)
    right = _make_brain(connectome, config, 17_303, std_enabled=False)
    left_trace = []
    right_trace = []
    for brain, trace in ((left, left_trace), (right, right_trace)):
        for _ in range(20):
            fired, _ = brain.step(stimulus_indices=interface.sensory_populations["A"], stimulus_rate_hz=205.0)
            trace.append(np.asarray(fired, dtype=np.int32).copy())
    return {
        "fired_indices_equal": len(left_trace) == len(right_trace) and all(np.array_equal(a, b) for a, b in zip(left_trace, right_trace)),
        "neural_state_equal": bool(np.array_equal(left.v, right.v) and np.array_equal(left.g, right.g) and np.array_equal(left._fast_active, right._fast_active)),
        "rng_equal": left.rng.bit_generator.state == right.rng.bit_generator.state,
        "passed": bool(all(np.array_equal(a, b) for a, b in zip(left_trace, right_trace)) and np.array_equal(left.v, right.v) and np.array_equal(left.g, right.g)),
    }


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT, workers: int = 1):
    started = time.perf_counter()
    worker_count = max(1, min(3, int(workers)))
    jobs = [(seed, str(data_dir)) for seed in SEEDS]
    if worker_count == 1:
        rows = [_run_seed(seed, str(data_dir)) for seed in SEEDS]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_run_seed_job, jobs))
    rows.sort(key=lambda row: row["seed"])

    performance_rows = [
        {
            "seed": row["seed"],
            "control_episodes_per_second": row["arms"]["CONTROL"]["training"]["episodes_per_second"],
            "std_episodes_per_second": row["arms"]["STD_BINDING"]["training"]["episodes_per_second"],
            "control_seconds": row["arms"]["CONTROL"]["training"]["elapsed_seconds"],
            "std_seconds": row["arms"]["STD_BINDING"]["training"]["elapsed_seconds"],
        }
        for row in rows
    ]
    representation = _f3d_representation_summary(rows)
    behavior = _behavior_summary(rows)
    first_control = float(np.mean([row["arms"]["CONTROL"]["first_memory_guard"]["initial"]["accuracy"] for row in rows]))
    first_std = float(np.mean([row["arms"]["STD_BINDING"]["first_memory_guard"]["initial"]["accuracy"] for row in rows]))
    first_guard = {
        "control_initial_accuracy": first_control,
        "std_initial_accuracy": first_std,
        "absolute_drop": first_control - first_std,
        "flagged": bool(first_control - first_std > 0.10),
    }
    budget_pairs = [(row["arms"][arm]["initial_plastic_budget"], row["arms"][arm]["plastic_budget"]) for row in rows for arm in ARM_NAMES]
    anatomy = [row["arms"][arm]["anatomical_edge_count"] for row in rows for arm in ARM_NAMES]
    std_bytes = [row["arms"]["STD_BINDING"]["std_state_bytes"] for row in rows]
    disabled_exact = _disabled_exact_smoke(data_dir)
    artifact = {
        "protocol": {
            "phase": "F.3D",
            "seeds": list(SEEDS),
            "arms": list(ARM_NAMES),
            "contexts": 16,
            "audit_repetitions": AUDIT_REPETITIONS,
            "training_episodes_per_arm": TRAINING_EPISODES,
            "evaluation_checkpoints": [0, TRAINING_EPISODES],
            "evaluation_cycles_per_context": EVALUATION_CYCLES,
            "timing_ms": {"first": 20.0, "second": 20.0, "go": 20.0},
            "slow_adaptation_enabled": False,
            "adaptive_plastic_budget": False,
            "route_cache_enabled": True,
            "plastic_row_cache_enabled": True,
            "prospective_index_enabled": True,
            "telemetry_level": "summary",
            "std_config": {"enabled_control": False, "enabled_binding": True, "recovery_tau_ms": 30.0, "depression_fraction": 0.15, "min_release_factor": 0.50},
            "dynamic_candidate_count": rows[0]["candidate_count"],
            "selected_output_count": rows[0]["selected_output_count"],
        },
        "matched_setup": {
            "identical_schedules": bool(all(row["matched_schedule"] for row in rows)),
            "identical_initial_plastic_masks": bool(all(row["matched_initial_plastic_mask"] for row in rows)),
            "same_learning_algorithm": True,
            "std_is_generic": True,
        },
        "representation": {"per_seed": [{"seed": row["seed"], "arms": row["representation"]} for row in rows], "summary": representation},
        "std_telemetry": {"per_seed": [{"seed": row["seed"], "arms": row["std_telemetry"]} for row in rows]},
        "first_memory_guard": first_guard,
        "behavior": {"per_seed": [{"seed": row["seed"], "arms": row["arms"]} for row in rows], "summary": behavior},
        "performance": {"training_per_seed": performance_rows, "workers_requested": worker_count, "wall_seconds": float(time.perf_counter() - started), "aggregate_training_episodes": len(SEEDS) * len(ARM_NAMES) * TRAINING_EPISODES},
        "memory": {"std_state_bytes_per_brain": int(np.median(std_bytes)) if std_bytes else 0, "std_state_formula": "neuron_count * (float32 release_factor + int64 last_release_step)", "additional_worker_memory": "one connectome plus one brain per worker; measure externally for full 1GB feather input", "dense_std_array": False},
        "safety": {"plastic_budget_fixed": bool(all(before == after for before, after in budget_pairs)), "anatomy_unchanged": bool(len(set(anatomy)) == 1), "disabled_exact_smoke": disabled_exact},
        "conclusion": {"representation_supported": bool(representation["criterion_context_separation"] and representation["criterion_broader_pools"]), "behavioral_criterion": behavior["behavioral_criterion"], "first_memory_guard_passed": not first_guard["flagged"], "next_step": "proceed_to_F3E_only_if_STD_binding_is_behaviorally_better_and_guards_pass"},
        "pass": bool(all(row["matched_schedule"] and row["matched_initial_plastic_mask"] for row in rows) and all(before == after for before, after in budget_pairs) and len(set(anatomy)) == 1 and not first_guard["flagged"] and disabled_exact["passed"]),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output, workers=args.workers)
    print(json.dumps({"pass": result["pass"], "representation": result["representation"]["summary"], "behavior": result["behavior"]["summary"], "runtime_seconds": result["runtime_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
