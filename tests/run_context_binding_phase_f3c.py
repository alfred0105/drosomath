"""Run the diagnostic-only Phase F.3C slow-adaptation context-binding study."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    CONTEXTUAL_GRAMMAR,
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
from drosomath.whole_brain import PlasticStateConfig, SlowAdaptationConfig  # noqa: E402
from run_contextual_credit_audit_phase_f3b import (  # noqa: E402
    _jaccard,
    _public_representation,
    _representation_summary,
)
from run_contextual_prediction_phase_f3a import _summarize as _contextual_summary  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_contextual_credit_audit_phase_f3b import _run_performance as _p5_performance  # noqa: E402


SEEDS = (163, 167, 173)
AUDIT_REPETITIONS = 4
STABLE_REPETITION_THRESHOLD = 3
TRAINING_EPISODES = 400
EVALUATION_CYCLES = 4
ARTIFACT = Path("results/latest_context_binding_phase_f3c.json")
DATA_DIR = Path("data/malecns_v1")


def _interface_config() -> SymbolInterfaceConfig:
    return SymbolInterfaceConfig(
        symbols=SYMBOLS, sensory_population_size=32, output_population_size=32,
        seed=7, dt_ms=0.2, default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0, plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


def _make_brain(connectome, config, seed, *, binding: bool):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=config.dt_ms),
        seed=seed,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=seed,
        ),
        slow_adaptation_config=SlowAdaptationConfig(enabled=binding),
    )


def _twin(connectome, config, seed, persistent, rng_state, *, binding):
    brain = _make_brain(connectome, config, seed, binding=binding)
    _restore_persistent_state(brain, persistent)
    brain.rng.bit_generator.state = copy.deepcopy(rng_state)
    brain.reset()
    return brain


def _audit_state(connectome, interface, config, seed, persistent, rng_state, *, binding):
    stable = {"first": {}, "pre_go": {}}
    frequencies = {"first": {}, "pre_go": {}}
    metrics = {"first": {}, "pre_go": {}}
    adaptation = {"first": {}, "pre_go": {}}
    for phase in ("first", "pre_go"):
        for context in ORDERED_PAIRS:
            repetitions = []
            rows = []
            adaptation_rows = []
            for repetition in range(AUDIT_REPETITIONS):
                brain = _twin(
                    connectome, config, seed + 10_000 + repetition,
                    persistent, rng_state, binding=binding,
                )
                observed = {}

                def observer(name, current, *, observed=observed):
                    if (phase == "first" and name == "first") or (phase == "pre_go" and name == "second"):
                        active = np.asarray(current._fast_active, dtype=np.int32).copy()
                        observed["active"] = active
                        observed["voltage"] = float(np.linalg.norm(current.v[active])) if len(active) else 0.0
                        observed["conductance"] = float(np.linalg.norm(current.g[active])) if len(active) else 0.0
                        payload = active.tobytes()
                        payload += np.asarray(current.v[active], dtype=np.float32).round(4).tobytes()
                        payload += np.asarray(current.g[active], dtype=np.float32).round(4).tobytes()
                        observed["fingerprint"] = hashlib.sha256(payload).hexdigest()
                        observed["adaptation"] = current.slow_adaptation_summary()

                session = TwoCueSequenceSession(brain, interface)
                session.run_trial(*context, track_eligibility=False, phase_observer=observer)
                active = set(int(value) for value in observed.get("active", ()))
                repetitions.append(active)
                rows.append({
                    "active_count": len(active),
                    "membrane_norm": observed.get("voltage", 0.0),
                    "conductance_norm": observed.get("conductance", 0.0),
                    "fingerprint": observed.get("fingerprint", ""),
                })
                adaptation_rows.append(observed.get("adaptation", brain.slow_adaptation_summary()))
            counts = {}
            for active in repetitions:
                for neuron in active:
                    counts[neuron] = counts.get(neuron, 0) + 1
            stable[phase][context] = {
                neuron for neuron, count in counts.items()
                if count >= STABLE_REPETITION_THRESHOLD
            }
            frequencies[phase][context] = {
                neuron: count / AUDIT_REPETITIONS for neuron, count in counts.items()
            }
            metrics[phase][context] = rows
            adaptation[phase][context] = adaptation_rows
    return {
        "stable": stable,
        "frequencies": frequencies,
        "metrics": metrics,
        "adaptation": adaptation,
    }


def _adaptation_summary(audit):
    result = {}
    for phase in ("first", "pre_go"):
        rows = [row for context in ORDERED_PAIRS for row in audit["adaptation"][phase][context]]
        result[phase] = {
            "mean_adapted_neurons": float(np.mean([row["adapted_neurons"] for row in rows])) if rows else 0.0,
            "mean_adaptation_mv": float(np.mean([row["mean_adaptation_mv"] for row in rows])) if rows else 0.0,
            "max_adaptation_mv": float(np.max([row["max_adaptation_mv"] for row in rows])) if rows else 0.0,
            "extra_state_bytes": int(rows[0]["extra_state_bytes"]) if rows else 0,
        }
    return result


def _behavior(connectome, interface, config, seed, persistent, rng_state, *, binding, prediction, reset):
    brain = _twin(connectome, config, seed, persistent, rng_state, binding=binding)
    session = ContextualPredictionSession(brain, interface) if prediction else TwoCueSequenceSession(brain, interface)
    schedule = balanced_pair_schedule(EVALUATION_CYCLES, seed=seed + 40_000)
    rows = []
    for first, second in schedule:
        value = session.run_trial(first, second, reset_between_items=reset)
        target = CONTEXTUAL_GRAMMAR[(first, second)] if prediction else first
        rows.append(SimpleNamespace(
            first=first, second=second, target=target,
            decision=value.decision, go_output_rates_hz=value.go_output_rates_hz,
            go_output_spikes=value.go_output_spikes,
        ))
    return _contextual_summary(rows)


def _train(connectome, interface, config, seed, *, binding):
    brain = _make_brain(connectome, config, seed, binding=binding)
    session = ContextualPredictionLearningSession(brain, interface)
    schedule = balanced_pair_schedule(25, seed=seed + session.config.schedule_seed_offset)
    session.train(schedule[:TRAINING_EPISODES], capture_first_incorrect=0)
    return brain, schedule


def _state_equal(left, right):
    keys = ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    return all(np.array_equal(getattr(left.plasticity, key), getattr(right.plasticity, key)) for key in keys) and left.plasticity.allocation_overrides() == right.plasticity.allocation_overrides()


def _allocation_equal(left, right):
    return all(
        np.array_equal(left[key], right[key])
        for key in ("promoted_edges", "retired_edges")
    )


def _trace_trial(brain, interface):
    trace = []
    original = brain.step

    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        trace.append(np.asarray(result[0], dtype=np.int32).copy())
        return result

    brain.step = traced
    try:
        result = TwoCueSequenceSession(brain, interface).run_trial("A", "B")
    finally:
        brain.step = original
    return trace, result


def _disabled_exact(connectome, interface, config):
    left = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=config.dt_ms),
        seed=17_301,
        plasticity_config=PlasticStateConfig(
            plastic_fraction=config.plastic_fraction,
            seed=17_301,
        ),
    )
    right = _make_brain(connectome, config, 17_301, binding=False)
    left_trace, left_result = _trace_trial(left, interface)
    right_trace, right_result = _trace_trial(right, interface)
    arrays = all(np.array_equal(getattr(left, name), getattr(right, name)) for name in ("v", "g", "refractory_until", "_fast_active"))
    rings = all(np.array_equal(a, b) for a, b in zip(left._delay_ring, right._delay_ring))
    return {
        "fired_indices_equal": len(left_trace) == len(right_trace) and all(np.array_equal(a, b) for a, b in zip(left_trace, right_trace)),
        "neural_state_equal": bool(arrays and rings),
        "rng_equal": left.rng.bit_generator.state == right.rng.bit_generator.state,
        "decision_equal": left_result.decision == right_result.decision,
    } | {"passed": bool(len(left_trace) == len(right_trace) and all(np.array_equal(a, b) for a, b in zip(left_trace, right_trace)) and arrays and rings and left.rng.bit_generator.state == right.rng.bit_generator.state and left_result.decision == right_result.decision)}


def _binding_replay(connectome, interface, config):
    left = _make_brain(connectome, config, 17_302, binding=True)
    right = _make_brain(connectome, config, 17_302, binding=True)
    left_trace, left_result = _trace_trial(left, interface)
    right_trace, right_result = _trace_trial(right, interface)
    return {
        "fired_indices_equal": len(left_trace) == len(right_trace) and all(np.array_equal(a, b) for a, b in zip(left_trace, right_trace)),
        "adaptation_equal": bool(np.array_equal(left.adaptation_mv, right.adaptation_mv)),
        "v_equal": bool(np.array_equal(left.v, right.v)),
        "g_equal": bool(np.array_equal(left.g, right.g)),
        "rng_equal": left.rng.bit_generator.state == right.rng.bit_generator.state,
        "decision_equal": left_result.decision == right_result.decision,
    }


def _persistent_safety(connectome, interface, config):
    result = {}
    for binding in (False, True):
        brain = _make_brain(connectome, config, 17_303, binding=binding)
        before = _snapshot_persistent_state(brain)
        session = TwoCueSequenceSession(brain, interface)
        for first, second in (("A", "B"), ("C", "D")):
            session.run_trial(first, second, track_eligibility=False)
        after = _snapshot_persistent_state(brain)
        result["binding" if binding else "control"] = {
            key: bool(np.array_equal(before[key], after[key]))
            for key in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
        }
        result["binding" if binding else "control"]["allocation_unchanged"] = _allocation_equal(before["allocation_overrides"], after["allocation_overrides"])
    return result


def _benchmark(connectome, interface, config):
    schedule = balanced_pair_schedule(2, seed=17_304)[:32]

    def bench(binding):
        brain = _make_brain(connectome, config, 17_305, binding=binding)
        session = ContextualPredictionSession(brain, interface)
        started = time.perf_counter()
        for first, second in schedule:
            session.run_trial(first, second)
        elapsed = time.perf_counter() - started
        return float(len(schedule) / max(elapsed, 1e-12))

    control = [bench(False) for _ in range(2)]
    binding = [bench(True) for _ in range(2)]
    control_median = float(np.median(control))
    binding_median = float(np.median(binding))
    return {
        "control_episodes_per_second": control_median,
        "binding_episodes_per_second": binding_median,
        "control_runs": control,
        "binding_runs": binding,
        "binding_overhead_fraction": float(max(0.0, 1.0 - binding_median / max(control_median, 1e-12))),
        "extra_adaptation_bytes": int(connectome.neuron_count * 4),
        "episodes": len(schedule),
    }


def _mean(rows, path):
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _representation_criterion(runs):
    control_values = []
    binding_values = []
    broad_pass = []
    for run in runs:
        control = run["representation_before"]["control"]
        binding = run["representation_before"]["binding"]
        control_values.append(control["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"])
        binding_values.append(binding["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"])
        per_context = binding["conjunctive_specific_pools"]["pre_go"]["per_context"]
        broad_pass.append(sum(row["broader_conjunction_count"] > 0 for row in per_context.values()) >= 12)
    return {
        "control_mean_same_first_jaccard": float(np.mean(control_values)),
        "binding_mean_same_first_jaccard": float(np.mean(binding_values)),
        "binding_to_control_ratio": float(np.mean(binding_values) / max(np.mean(control_values), 1e-12)),
        "per_seed": [{"seed": run["seed"], "same_first_control": c, "same_first_binding": b, "broader_pool_12_of_16": broad} for run, c, b, broad in zip(runs, control_values, binding_values, broad_pass)],
        "criterion_same_first": bool(np.mean(binding_values) <= 0.80 * np.mean(control_values)),
        "criterion_broader_pools": bool(sum(broad_pass) >= 2),
        "supported": bool(np.mean(binding_values) <= 0.80 * np.mean(control_values) and sum(broad_pass) >= 2),
    }


def _prediction_criterion(runs):
    control = [run["behavior"]["control"]["400"]["intact"] for run in runs]
    binding = [run["behavior"]["binding"]["400"]["intact"] for run in runs]
    reset = [run["behavior"]["binding"]["400"]["between_item_reset"] for run in runs]
    no_collapse = not any(row["output_collapse"] for row in control + binding + reset)
    improved_seeds = sum(b["accuracy"] > c["accuracy"] for c, b in zip(control, binding))
    dependence_seeds = sum(b["accuracy"] > r["accuracy"] for b, r in zip(binding, reset))
    control_acc = float(np.mean([row["accuracy"] for row in control]))
    binding_acc = float(np.mean([row["accuracy"] for row in binding]))
    reset_acc = float(np.mean([row["accuracy"] for row in reset]))
    control_margin = float(np.mean([row["target_minus_best_competitor_margin_hz"] for row in control]))
    binding_margin = float(np.mean([row["target_minus_best_competitor_margin_hz"] for row in binding]))
    reset_margin = float(np.mean([row["target_minus_best_competitor_margin_hz"] for row in reset]))
    return {
        "control_intact_accuracy": control_acc,
        "binding_intact_accuracy": binding_acc,
        "binding_reset_accuracy": reset_acc,
        "control_intact_margin_hz": control_margin,
        "binding_intact_margin_hz": binding_margin,
        "binding_reset_margin_hz": reset_margin,
        "improved_seed_count": int(improved_seeds),
        "context_dependence_seed_count": int(dependence_seeds),
        "no_output_collapse": bool(no_collapse),
        "prediction_improved": bool(binding_acc > control_acc and improved_seeds >= 2 and binding_margin > control_margin and no_collapse),
        "context_dependent": bool(binding_acc > reset_acc and dependence_seeds >= 2 and binding_margin > reset_margin and no_collapse),
        "strong_prediction": bool(binding_acc >= 0.50 and sum(row["accuracy"] >= 0.50 for row in binding) >= 2 and binding_acc > reset_acc and dependence_seeds >= 2 and no_collapse),
    }


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    runs = []
    for seed in SEEDS:
        control_initial = _make_brain(connectome, interface_config, seed, binding=False)
        binding_initial = _make_brain(connectome, interface_config, seed, binding=True)
        control_persistent = _snapshot_persistent_state(control_initial)
        binding_persistent = _snapshot_persistent_state(binding_initial)
        control_rng = copy.deepcopy(control_initial.rng.bit_generator.state)
        binding_rng = copy.deepcopy(binding_initial.rng.bit_generator.state)
        control_audit_raw = _audit_state(connectome, wm_interface, interface_config, seed, control_persistent, control_rng, binding=False)
        control_audit = _public_representation(_representation_summary(control_audit_raw))
        binding_audit_raw = _audit_state(connectome, wm_interface, interface_config, seed, binding_persistent, binding_rng, binding=True)
        binding_audit = _public_representation(_representation_summary(binding_audit_raw))
        control_brain, control_schedule = _train(connectome, wm_interface, interface_config, seed, binding=False)
        binding_brain, binding_schedule = _train(connectome, wm_interface, interface_config, seed, binding=True)
        control_after = _snapshot_persistent_state(control_brain)
        binding_after = _snapshot_persistent_state(binding_brain)
        control_after_rng = copy.deepcopy(control_brain.rng.bit_generator.state)
        binding_after_rng = copy.deepcopy(binding_brain.rng.bit_generator.state)
        behavior = {"control": {}, "binding": {}}
        for label, binding, persistent, rng_state in (
            ("control", False, control_after, control_after_rng),
            ("binding", True, binding_after, binding_after_rng),
        ):
            behavior[label]["0"] = {
                "intact": _behavior(connectome, wm_interface, interface_config, seed, control_persistent if not binding else binding_persistent, control_rng if not binding else binding_rng, binding=binding, prediction=True, reset=False),
                "between_item_reset": _behavior(connectome, wm_interface, interface_config, seed, control_persistent if not binding else binding_persistent, control_rng if not binding else binding_rng, binding=binding, prediction=True, reset=True),
            }
            behavior[label]["400"] = {
                "intact": _behavior(connectome, wm_interface, interface_config, seed, persistent, rng_state, binding=binding, prediction=True, reset=False),
                "between_item_reset": _behavior(connectome, wm_interface, interface_config, seed, persistent, rng_state, binding=binding, prediction=True, reset=True),
            }
        runs.append({
            "seed": seed,
            "matched_schedule": control_schedule == binding_schedule,
            "matched_initial_plastic_mask": bool(np.array_equal(control_initial.plasticity.plastic_mask, binding_initial.plasticity.plastic_mask)),
            "representation_before": {"control": control_audit, "binding": binding_audit},
            "adaptation_telemetry": {"control": _adaptation_summary(control_audit_raw), "binding": _adaptation_summary(binding_audit_raw)},
            "behavior": behavior,
            "safety": {"control_budget": int(control_brain.plasticity.plastic_edge_count), "binding_budget": int(binding_brain.plasticity.plastic_edge_count)},
        })

    representation = _representation_criterion(runs)
    prediction = _prediction_criterion(runs)
    first_guard = {
        "control_accuracy": float(np.mean([run["behavior"]["control"]["0"]["intact"]["accuracy"] for run in runs])),
        "binding_accuracy": float(np.mean([run["behavior"]["binding"]["0"]["intact"]["accuracy"] for run in runs])),
    }
    first_guard["absolute_drop"] = float(first_guard["control_accuracy"] - first_guard["binding_accuracy"])
    first_guard["flagged"] = bool(first_guard["absolute_drop"] > 0.10)
    activity = {
        "control_mean_pre_go_active": float(np.mean([run["representation_before"]["control"]["state_metrics"]["pre_go"]["per_context"]["AA"]["active_count_mean"] for run in runs])),
        "binding_mean_pre_go_active": float(np.mean([run["representation_before"]["binding"]["state_metrics"]["pre_go"]["per_context"]["AA"]["active_count_mean"] for run in runs])),
        "binding_mean_pre_go_adapted": float(np.mean([run["adaptation_telemetry"]["binding"]["pre_go"]["mean_adapted_neurons"] for run in runs])),
        "binding_near_silence": bool(all(run["representation_before"]["binding"]["state_metrics"]["pre_go"]["per_context"]["AA"]["stable_active_count"] <= 1 for run in runs)),
    }
    activity["binding_more_than_two_x_control"] = bool(
        activity["binding_mean_pre_go_active"] > 2.0 * max(activity["control_mean_pre_go_active"], 1.0)
    )
    activity["pathological"] = bool(
        activity["binding_more_than_two_x_control"] or activity["binding_near_silence"]
    )
    safety = {
        "persistent_learning_disabled": _persistent_safety(connectome, wm_interface, interface_config),
        "disabled_exact_equivalence": _disabled_exact(connectome, wm_interface, interface_config),
        "binding_deterministic_replay": _binding_replay(connectome, wm_interface, interface_config),
        "plastic_budget_fixed": all(run["safety"]["control_budget"] == run["safety"]["binding_budget"] for run in runs),
    }
    performance = _benchmark(connectome, wm_interface, interface_config)
    p5 = _p5_performance(connectome, wm_interface, interface_config)
    artifact = {
        "protocol": {
            "phase": "F.3C+P.7",
            "seeds": list(SEEDS),
            "training_episodes_per_arm": TRAINING_EPISODES,
            "contexts": 16,
            "evaluation_trials_per_context": EVALUATION_CYCLES,
            "slow_adaptation": {"enabled_binding": True, "enabled_control": False, "tau_ms": 30.0, "spike_increment_mv": 0.75, "max_adaptation_mv": 3.0},
            "learning_rates_unchanged": True,
            "grammar_unchanged": True,
            "adaptive_plastic_budget": False,
            "route_cache_enabled": True,
            "dynamic_candidate_count": int(candidate_count),
            "selected_output_count": int(len(selected)),
        },
        "matched_setup": {"identical_initialization": all(run["matched_initial_plastic_mask"] for run in runs), "identical_schedules": all(run["matched_schedule"] for run in runs), "same_interface": True, "same_learning_algorithm": True},
        "runs": runs,
        "representation_criterion": representation,
        "first_memory_guard": first_guard,
        "activity_guard": activity,
        "prediction_criterion": prediction,
        "safety": safety,
        "performance": {"f3c_binding_vs_control": performance, "p5_reference": p5, "p5_optimized_path_preserved": bool(p5["optimized_episodes_per_second"] >= p5["legacy_reference_episodes_per_second"] * 1.05 and not p5["regression_detected"])},
        "conclusion": {"context_binding_supported": bool(representation["supported"] and not first_guard["flagged"] and not activity["binding_near_silence"]), "prediction_improved": prediction["prediction_improved"], "context_dependence_supported": prediction["context_dependent"], "strong_prediction_supported": prediction["strong_prediction"], "next_step": "retain_slow_adaptation_for_followup" if prediction["prediction_improved"] else "do_not_promote_adaptation_without_behavioral_gain"},
        "pass": bool(all(run["matched_schedule"] and run["matched_initial_plastic_mask"] for run in runs) and safety["disabled_exact_equivalence"]["passed"] and safety["binding_deterministic_replay"]["fired_indices_equal"] and safety["persistent_learning_disabled"]["binding"]["allocation_unchanged"] and performance["control_episodes_per_second"] > 0.0 and safety["plastic_budget_fixed"]),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    artifact = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({"pass": artifact["pass"], "representation": artifact["representation_criterion"], "prediction": artifact["prediction_criterion"], "runtime_seconds": artifact["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
