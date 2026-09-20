"""Run the gated Phase F.3E transient-Hebbian representation screen."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    CONTEXTUAL_GRAMMAR,
    ContextualPredictionSession,
    ORDERED_PAIRS,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterfaceConfig,
    TwoCueSequenceSession,
    TransientHebbianBindingConfig,
    WorkingMemoryInterface,
    balanced_pair_schedule,
    load_malecns_v1,
)
from drosomath.malecns.symbol_interface import _restore_persistent_state, _snapshot_persistent_state  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig, SlowAdaptationConfig  # noqa: E402
from run_contextual_credit_audit_phase_f3b import _jaccard, _public_representation, _representation_summary  # noqa: E402
from run_contextual_prediction_phase_f3a import _summarize  # noqa: E402
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402


SEEDS = (193, 197, 199)
AUDIT_REPETITIONS = 4
STABLE_THRESHOLD = 3
AUDIT_EPISODES = len(ORDERED_PAIRS) * AUDIT_REPETITIONS
ARTIFACT = ROOT / "results/latest_transient_hebbian_binding_phase_f3e.json"
DATA_DIR = ROOT / "data/malecns_v1"


def interface_config():
    return SymbolInterfaceConfig(
        symbols=SYMBOLS, sensory_population_size=32, output_population_size=32,
        seed=7, dt_ms=0.2, default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0, plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


def make_brain(connectome, seed, *, binding):
    return PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        slow_adaptation_config=SlowAdaptationConfig(enabled=False),
        transient_hebbian_config=TransientHebbianBindingConfig(enabled=binding),
    )


def twin(connectome, seed, persistent, rng_state, *, binding):
    value = make_brain(connectome, seed, binding=binding)
    _restore_persistent_state(value, persistent)
    value.rng.bit_generator.state = copy.deepcopy(rng_state)
    value.reset()
    return value


def state_fingerprint(brain):
    active = np.asarray(brain._fast_active, dtype=np.int32)
    vector = np.zeros(64, dtype=np.float64)
    if len(active):
        values = (brain.v[active] - brain.params.resting_mv) / 50.0 + brain.g[active] / 10.0
        np.add.at(vector, active % 64, values)
        vector[0] += len(active) / max(1, brain.connectome.neuron_count)
    return vector


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def cosine_groups(values):
    grouped = defaultdict(list)
    for index, left in enumerate(ORDERED_PAIRS):
        for right in ORDERED_PAIRS[index + 1:]:
            if left[0] == right[0] and left[1] != right[1]:
                name = "same_first_different_second"
            elif left[1] == right[1] and left[0] != right[0]:
                name = "same_second_different_first"
            else:
                name = "different_first_and_second"
            grouped[name].append(cosine(values[left], values[right]))
    return {name: {"mean": float(np.mean(items)), "pairs": len(items)} for name, items in grouped.items()}


def audit(connectome, interface, seed, persistent, rng_state, *, binding):
    stable = {"first": {}, "pre_go": {}}
    frequencies = {"first": {}, "pre_go": {}}
    metrics = {"first": {}, "pre_go": {}}
    vectors = {"first": {}, "pre_go": {}}
    telemetry = {"first": [], "pre_go": []}
    started = time.perf_counter()
    for phase in ("first", "pre_go"):
        for context in ORDERED_PAIRS:
            repetitions = []
            rows = []
            state_rows = []
            for repetition in range(AUDIT_REPETITIONS):
                brain = twin(connectome, seed + 10_000 + repetition, persistent, rng_state, binding=binding)
                observed = {}

                def observer(name, current, *, observed=observed):
                    wanted = phase == "first" and name == "first" or phase == "pre_go" and name == "second"
                    if wanted:
                        active = np.asarray(current._fast_active, dtype=np.int32).copy()
                        observed["active"] = active
                        observed["vector"] = state_fingerprint(current)
                        observed["telemetry"] = current.transient_hebbian_summary()
                        observed["membrane_norm"] = float(np.linalg.norm(current.v[active])) if len(active) else 0.0
                        observed["conductance_norm"] = float(np.linalg.norm(current.g[active])) if len(active) else 0.0

                TwoCueSequenceSession(brain, interface).run_trial(*context, track_eligibility=False, phase_observer=observer)
                active = set(int(value) for value in observed.get("active", ()))
                repetitions.append(active)
                state_rows.append(observed.get("vector", np.zeros(64, dtype=np.float64)))
                rows.append({
                    "active_count": len(active),
                    "membrane_norm": observed.get("membrane_norm", 0.0),
                    "conductance_norm": observed.get("conductance_norm", 0.0),
                    "fingerprint": hashlib.sha256(np.asarray(sorted(active), dtype=np.int32).tobytes()).hexdigest(),
                })
                telemetry[phase].append(observed.get("telemetry", brain.transient_hebbian_summary()))
            counts = {}
            for active in repetitions:
                for neuron in active:
                    counts[neuron] = counts.get(neuron, 0) + 1
            stable[phase][context] = {neuron for neuron, count in counts.items() if count >= STABLE_THRESHOLD}
            frequencies[phase][context] = {neuron: count / AUDIT_REPETITIONS for neuron, count in counts.items()}
            metrics[phase][context] = rows
            vectors[phase][context] = np.mean(np.asarray(state_rows), axis=0)
    raw = {"stable": stable, "frequencies": frequencies, "metrics": metrics}
    public = _public_representation(_representation_summary(raw))
    compact_telemetry = {}
    for phase in ("first", "pre_go"):
        rows = telemetry[phase]
        compact_telemetry[phase] = {
            "recent_pre_neurons_mean": float(np.mean([row["recent_pre_neurons"] for row in rows])),
            "active_binding_edges_mean": float(np.mean([row["active_binding_edges"] for row in rows])),
            "mean_binding_gain": float(np.mean([row["mean_binding_gain"] for row in rows])),
            "max_binding_gain": float(np.max([row["max_binding_gain"] for row in rows])),
            "fraction_at_cap_mean": float(np.mean([row["fraction_at_cap"] for row in rows])),
            "mean_age_ms": float(np.mean([row["mean_age_ms"] for row in rows])),
            "pre_trace_bytes": int(rows[0]["pre_trace_bytes"]),
            "binding_state_bytes": int(rows[0]["binding_state_bytes"]),
            "reverse_index_bytes": int(rows[0]["reverse_index_bytes"]),
            "active_index_bytes": int(rows[0]["active_index_bytes"]),
        }
    return {
        "representation": public,
        "cosine_similarity": {phase: cosine_groups(vectors[phase]) for phase in ("first", "pre_go")},
        "telemetry": compact_telemetry,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def first_memory(connectome, interface, seed, persistent, rng_state, *, binding):
    brain = twin(connectome, seed, persistent, rng_state, binding=binding)
    schedule = balanced_pair_schedule(4, seed=seed + 40_000)
    rows = []
    for first, second in schedule:
        value = TwoCueSequenceSession(brain, interface).run_trial(first, second)
        rows.append(SimpleNamespace(first=first, second=second, target=first, decision=value.decision, go_output_rates_hz=value.go_output_rates_hz, go_output_spikes=value.go_output_spikes))
    return _summarize(rows)


def deterministic_safety(connectome, interface, seed):
    left = make_brain(connectome, seed, binding=True)
    right = make_brain(connectome, seed, binding=True)
    trace_equal = True
    for _ in range(40):
        fired_left, _ = left.step(stimulus_indices=interface.population_for_input("A"), stimulus_rate_hz=205.0)
        fired_right, _ = right.step(stimulus_indices=interface.population_for_input("A"), stimulus_rate_hz=205.0)
        trace_equal &= bool(np.array_equal(fired_left, fired_right))
    count = int(left._binding_active_count[0])
    return {
        "hebb_replay_fired_equal": trace_equal,
        "hebb_replay_neural_equal": bool(np.array_equal(left.v, right.v) and np.array_equal(left.g, right.g)),
        "hebb_replay_binding_equal": bool(np.array_equal(left.binding_gain, right.binding_gain) and np.array_equal(left.binding_active_edges[:count], right.binding_active_edges[:count])),
        "hebb_replay_rng_equal": left.rng.bit_generator.state == right.rng.bit_generator.state,
    }


def representation_criteria(rows):
    control = [row["representation"]["CONTROL"] for row in rows]
    hebb = [row["representation"]["HEBB_BINDING"] for row in rows]
    control_j = [row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in control]
    hebb_j = [row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in hebb]
    broader = [sum(item["broader_conjunction_count"] > 0 for item in row["conjunctive_specific_pools"]["pre_go"]["per_context"].values()) for row in hebb]
    control_mean = float(np.mean(control_j)); hebb_mean = float(np.mean(hebb_j))
    return {
        "control_same_first_jaccard": control_mean,
        "hebb_same_first_jaccard": hebb_mean,
        "hebb_to_control_ratio": float(hebb_mean / max(control_mean, 1e-12)),
        "broader_context_counts_by_seed": broader,
        "context_binding_improved": bool(hebb_mean <= 0.80 * control_mean and sum(value >= 12 for value in broader) >= 2),
    }


def guards(rows, first_rows):
    control = [row["representation"]["CONTROL"] for row in rows]
    hebb = [row["representation"]["HEBB_BINDING"] for row in rows]
    hebb_first = float(np.mean([row["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in hebb]))
    hebb_second = float(np.mean([row["pairwise_context_similarity"]["pre_go"]["same_second_different_first"]["mean"] for row in hebb]))
    control_active = float(np.mean([item["active_count_mean"] for row in control for item in row["state_metrics"]["pre_go"]["per_context"].values()]))
    hebb_active = float(np.mean([item["active_count_mean"] for row in hebb for item in row["state_metrics"]["pre_go"]["per_context"].values()]))
    near_silence = all(item["stable_active_count"] <= 1 for row in hebb for item in row["state_metrics"]["pre_go"]["per_context"].values())
    first_control = float(np.mean([row["CONTROL"]["accuracy"] for row in first_rows]))
    first_hebb = float(np.mean([row["HEBB_BINDING"]["accuracy"] for row in first_rows]))
    saturation = any(row["telemetry"]["HEBB_BINDING"]["pre_go"]["fraction_at_cap_mean"] > 0.50 for row in rows)
    return {
        "first_memory": {"control_accuracy": first_control, "hebb_accuracy": first_hebb, "first_memory_destroyed": bool(first_hebb < first_control - 0.10)},
        "second_domination": {"hebb_same_first_jaccard": hebb_first, "hebb_same_second_jaccard": hebb_second, "second_cue_domination": bool(hebb_second >= 0.80 and hebb_second > hebb_first)},
        "activity": {"control_mean_pre_go_active": control_active, "hebb_mean_pre_go_active": hebb_active, "hebb_to_control_ratio": hebb_active / max(control_active, 1e-12), "near_total_silence": bool(near_silence), "activity_pathology": bool(hebb_active > 2.0 * control_active or near_silence)},
        "binding_saturation": {"threshold": 0.50, "binding_saturation": bool(saturation)},
    }


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, config)
    wm = WorkingMemoryInterface(connectome, interface)
    rows = []
    for seed in SEEDS:
        initial = {}
        audits = {}
        first_rows = {}
        for arm, binding in (("CONTROL", False), ("HEBB_BINDING", True)):
            initial_brain = make_brain(connectome, seed, binding=binding)
            persistent = _snapshot_persistent_state(initial_brain)
            rng = copy.deepcopy(initial_brain.rng.bit_generator.state)
            initial[arm] = (persistent, rng)
            audits[arm] = audit(connectome, wm, seed, persistent, rng, binding=binding)
            first_rows[arm] = first_memory(connectome, wm, seed, persistent, rng, binding=binding)
        rows.append({
            "seed": seed,
            "representation": {arm: audits[arm]["representation"] for arm in ("CONTROL", "HEBB_BINDING")},
            "cosine_similarity": {arm: audits[arm]["cosine_similarity"] for arm in ("CONTROL", "HEBB_BINDING")},
            "telemetry": {arm: audits[arm]["telemetry"] for arm in ("CONTROL", "HEBB_BINDING")},
            "first_memory": first_rows,
            "runtime_seconds": {arm: audits[arm]["runtime_seconds"] for arm in ("CONTROL", "HEBB_BINDING")},
            "matched_initial_plastic_mask": bool(np.array_equal(make_brain(connectome, seed, binding=False).plasticity.plastic_mask, make_brain(connectome, seed, binding=True).plasticity.plastic_mask)),
        })
    rep = representation_criteria(rows)
    first = [{"CONTROL": row["first_memory"]["CONTROL"], "HEBB_BINDING": row["first_memory"]["HEBB_BINDING"]} for row in rows]
    guard = guards(rows, first)
    safety = deterministic_safety(connectome, wm, SEEDS[0])
    training_executed = False
    conclusion = {
        "context_binding_improved": rep["context_binding_improved"],
        "first_memory_destroyed": guard["first_memory"]["first_memory_destroyed"],
        "second_cue_domination": guard["second_domination"]["second_cue_domination"],
        "activity_pathology": guard["activity"]["activity_pathology"],
        "binding_saturation": guard["binding_saturation"]["binding_saturation"],
        "recommended_next_step": "redesign_generic_binding_dynamics" if not rep["context_binding_improved"] else "stage_b_prediction_learning",
        "interpretation": {"stage_a_only": True, "conjunctive_representation_demonstrated": rep["context_binding_improved"]},
    }
    artifact = {
        "protocol": {"phase": "F.3E", "seeds": list(SEEDS), "arms": ["CONTROL", "HEBB_BINDING"], "contexts": 16, "audit_repetitions": 4, "std_enabled": False, "slow_adaptation_enabled": False, "transient_hebbian_config": {"enabled_hebb": True, "pre_trace_tau_ms": 20.0, "binding_tau_ms": 40.0, "binding_increment": 0.10, "max_binding_gain": 0.50}, "dynamic_candidate_count": int(candidate_count), "selected_output_count": int(len(selected)), "route_cache_enabled": True, "plastic_row_cache_enabled": True, "prospective_index_enabled": True},
        "representation": {"per_seed": rows, "summary": rep},
        "binding_telemetry": {"per_seed": [{"seed": row["seed"], "arms": row["telemetry"]} for row in rows]},
        "guards": guard,
        "performance": {
            "per_seed_runtime_seconds": [row["runtime_seconds"] for row in rows],
            "control_episodes_per_second": float(np.mean([AUDIT_EPISODES / row["runtime_seconds"]["CONTROL"] for row in rows])),
            "hebb_episodes_per_second": float(np.mean([AUDIT_EPISODES / row["runtime_seconds"]["HEBB_BINDING"] for row in rows])),
            "hebb_overhead_fraction": float(max(0.0, 1.0 - np.mean([row["runtime_seconds"]["CONTROL"] for row in rows]) / max(np.mean([row["runtime_seconds"]["HEBB_BINDING"] for row in rows]), 1e-12))),
            "performance_pathology_threshold": 0.50,
            "performance_pathology": bool(
                np.mean([AUDIT_EPISODES / row["runtime_seconds"]["HEBB_BINDING"] for row in rows])
                < 0.50 * np.mean([AUDIT_EPISODES / row["runtime_seconds"]["CONTROL"] for row in rows])
            ),
            "stage_a_runtime_seconds": float(time.perf_counter() - started),
        },
        "memory": {"pre_trace_bytes": int(rows[0]["telemetry"]["HEBB_BINDING"]["pre_go"]["pre_trace_bytes"]), "binding_state_bytes": int(rows[0]["telemetry"]["HEBB_BINDING"]["pre_go"]["binding_state_bytes"]), "reverse_index_bytes": int(rows[0]["telemetry"]["HEBB_BINDING"]["pre_go"]["reverse_index_bytes"]), "active_index_bytes": int(rows[0]["telemetry"]["HEBB_BINDING"]["pre_go"]["active_index_bytes"]), "peak_rss": "not sampled by runner"},
        "safety": {"identical_initial_masks": bool(all(row["matched_initial_plastic_mask"] for row in rows)), **safety},
        "training_stage_executed": training_executed,
        "behavior": "not_executed_due_to_representation_gate",
        "conclusion": conclusion,
        "pass": bool(not training_executed and safety["hebb_replay_fired_equal"] and safety["hebb_replay_neural_equal"] and safety["hebb_replay_binding_equal"] and all(row["matched_initial_plastic_mask"] for row in rows)),
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
    result = run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({"pass": result["pass"], "training_stage_executed": result["training_stage_executed"], "conclusion": result["conclusion"], "runtime_seconds": result["runtime_seconds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
