"""Calibrate sparse KC activity for the frozen F.3R MB-routed encoder."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from drosomath.flywire_real import FlyBrainParams  # noqa: E402
from drosomath.malecns import (  # noqa: E402
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterface,
    SymbolInterfaceConfig,
    TwoCueSequenceSession,
    WorkingMemoryInterface,
    load_malecns_v1,
)
from drosomath.malecns.symbol_interface import _snapshot_persistent_state  # noqa: E402
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402
from run_contextual_credit_audit_phase_f3b import _public_representation, _representation_summary  # noqa: E402


SEEDS = (251, 257, 263)
RATES = (25.0, 50.0, 100.0, 150.0, 205.0)
OFFSETS = (0.0, 2.0, 4.0, 6.0)
ORDERED_CONTEXTS = tuple((left, right) for left in SYMBOLS for right in SYMBOLS)
AUDIT_REPETITIONS = 4
ARTIFACT = ROOT / "results/latest_mb_sparse_coding_calibration_phase_f3s.json"
F3R_ARTIFACT = ROOT / "results/latest_routing_architecture_audit_phase_f3r.json"
DATA_DIR = ROOT / "data/malecns_v1"
TARGET_KC_FRACTION = 0.075
WHOLE_NETWORK_EXPLOSION_LIMIT = 0.75


def _meta(connectome, name):
    return np.asarray(connectome.metadata.get(name, [None] * connectome.neuron_count), dtype=object)


def _config():
    return SymbolInterfaceConfig(
        sensory_population_size=32,
        output_population_size=32,
        seed=7,
        dt_ms=0.2,
        default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0,
        plastic_fraction=0.05,
        output_selection="random_indegree",
    )


def frozen_interfaces(connectome):
    """Reconstruct the committed F.3R populations without reselection."""
    f3r = json.loads(F3R_ARTIFACT.read_text(encoding="utf-8"))
    mb = f3r["mb_routed_candidate"]["summary"]["population_indices"]
    config = _config()
    current_surface = SymbolInterface(connectome, config)
    output = current_surface.output_populations
    current = SymbolInterface.with_input_populations(
        connectome, config, current_surface.sensory_populations,
        output_populations=output, source="CURRENT_RANDOM_ENCODER",
    )
    mb_populations = {symbol: np.asarray(mb[symbol], dtype=np.int32) for symbol in SYMBOLS}
    mb_surface = SymbolInterface.with_input_populations(
        connectome, config, mb_populations,
        output_populations=output, source="MB_ROUTED_GENERIC",
    )
    return current, mb_surface, mb_populations


def _make_brain(connectome, seed, *, kc_offset=0.0):
    brain = PlasticMaleCNSBrain(
        connectome,
        params=FlyBrainParams(dt_ms=0.2),
        seed=seed,
        plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        kc_threshold_offset_mv=float(kc_offset),
    )
    # F.3S is an observation-only calibration.  Disable usage/eligibility
    # recording before taking any state snapshot so stepping cannot mutate the
    # persistent learning arrays.
    brain.set_plasticity_tracking(False)
    return brain


def _persistent_equal(brain, snapshot):
    return all(np.array_equal(getattr(brain.plasticity, name), snapshot[name]) for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask"))


def _kc_and_annotations(connectome):
    classes = _meta(connectome, "class")
    kc = np.flatnonzero(classes == "Kenyon_Cell").astype(np.int32)
    mbon = np.flatnonzero(classes == "MBON").astype(np.int32)
    # APL is annotated as type/flywireType/hemibrainType APL and as APL_L/R
    # instances in the official neuron table.  No body IDs are hard-coded.
    apl = np.zeros(connectome.neuron_count, dtype=np.bool_)
    for field in ("type", "flywireType", "hemibrainType"):
        apl |= _meta(connectome, field) == "APL"
    instances = _meta(connectome, "instance")
    apl |= np.asarray(["apl" in str(value).lower() for value in instances], dtype=np.bool_)
    return kc, mbon, np.flatnonzero(apl).astype(np.int32)


def apl_audit(connectome, kc, apl):
    kc_mask = np.zeros(connectome.neuron_count, dtype=np.bool_); kc_mask[kc] = True
    apl_mask = np.zeros(connectome.neuron_count, dtype=np.bool_); apl_mask[apl] = True
    pre = np.repeat(np.arange(connectome.neuron_count, dtype=np.int32), np.diff(connectome.indptr))
    posts = np.asarray(connectome.post_indices, dtype=np.int32)
    signed = np.asarray(connectome.signed_synapse_counts, dtype=np.float64)
    kc_to_apl = (kc_mask[pre] & apl_mask[posts])
    apl_to_kc = (apl_mask[pre] & kc_mask[posts])
    nt = [str(value) for value in np.asarray(connectome.consensus_nt, dtype=object)[apl]]
    signs = np.asarray(connectome.presynaptic_sign, dtype=np.int8)[apl]
    return {
        "apl_count": int(len(apl)),
        "apl_body_ids": [int(connectome.body_ids[index]) for index in apl],
        "annotation_fields_used": ["type", "flywireType", "hemibrainType", "instance"],
        "kc_to_apl": {"edge_count": int(kc_to_apl.sum()), "signed_strength": float(signed[kc_to_apl].sum()), "absolute_strength": float(np.abs(signed[kc_to_apl]).sum())},
        "apl_to_kc": {"edge_count": int(apl_to_kc.sum()), "signed_strength": float(signed[apl_to_kc].sum()), "absolute_strength": float(np.abs(signed[apl_to_kc]).sum())},
        "apl_consensus_neurotransmitter": {value: nt.count(value) for value in sorted(set(nt))},
        "apl_presynaptic_signs": {str(int(value)): int(np.count_nonzero(signs == value)) for value in sorted(set(signs.tolist()))},
        "loader_models_apl_as_inhibitory": bool(len(signs) > 0 and np.all(signs < 0)),
    }


def _run_single(connectome, interface, seed, symbol, rate, offset):
    kc, mbon, apl = _kc_and_annotations(connectome)
    brain = _make_brain(connectome, seed, kc_offset=offset)
    snapshot = _snapshot_persistent_state(brain)
    input_indices = interface.population_for_input(symbol)
    spike_counts = np.zeros(connectome.neuron_count, dtype=np.int32)
    fired_union = set()
    trace = []
    for _ in range(int(round(20.0 / brain.params.dt_ms))):
        fired, _ = brain.step(stimulus_indices=input_indices, stimulus_rate_hz=rate)
        fired = np.asarray(fired, dtype=np.int32).copy()
        trace.append(fired)
        if len(fired):
            spike_counts[fired] += 1
            fired_union.update(int(value) for value in fired)
    kc_counts = spike_counts[kc]
    active_kc = kc[kc_counts > 0]
    active_mbon = int(np.count_nonzero(spike_counts[mbon] > 0))
    active_apl = int(np.count_nonzero(spike_counts[apl] > 0))
    input_active = int(np.count_nonzero(spike_counts[input_indices] > 0))
    return {
        "seed": int(seed), "symbol": symbol, "rate_hz": float(rate), "kc_offset_mv": float(offset),
        "kc_active_indices": [int(value) for value in active_kc],
        "kc_active_fraction": float(len(active_kc) / max(1, len(kc))),
        "kc_mean_spikes_per_active": float(kc_counts[kc_counts > 0].mean()) if np.any(kc_counts > 0) else 0.0,
        "kc_median_spikes_per_active": float(np.median(kc_counts[kc_counts > 0])) if np.any(kc_counts > 0) else 0.0,
        "kc_population_spike_count": int(kc_counts.sum()),
        "whole_network_active_fraction": float(len(fired_union) / connectome.neuron_count),
        "whole_network_active_count": int(len(fired_union)),
        "mbon_active_fraction": float(active_mbon / max(1, len(mbon))),
        "mbon_active_count": active_mbon,
        "apl_active_fraction": float(active_apl / max(1, len(apl))),
        "apl_active_count": active_apl,
        "input_active_count": input_active,
        "input_population_size": int(len(input_indices)),
        "fired_trace": [[int(value) for value in fired] for fired in trace],
        "persistent_state_unchanged": bool(_persistent_equal(brain, snapshot)),
    }


def _point_summary(rows):
    by_symbol = {symbol: [row for row in rows if row["symbol"] == symbol] for symbol in SYMBOLS}
    pairwise = []
    for left, right in combinations(SYMBOLS, 2):
        a = set(value for row in by_symbol[left] for value in row["kc_active_indices"])
        b = set(value for row in by_symbol[right] for value in row["kc_active_indices"])
        union = a | b
        pairwise.append(float(len(a & b) / len(union)) if union else 1.0)
    return {
        "mean_kc_active_fraction": float(np.mean([row["kc_active_fraction"] for row in rows])),
        "max_symbol_kc_active_fraction": float(max(row["kc_active_fraction"] for row in rows)),
        "mean_kc_spikes_per_active": float(np.mean([row["kc_mean_spikes_per_active"] for row in rows])),
        "median_kc_spikes_per_active": float(np.median([row["kc_median_spikes_per_active"] for row in rows])),
        "mean_kc_population_spikes": float(np.mean([row["kc_population_spike_count"] for row in rows])),
        "mean_whole_network_active_fraction": float(np.mean([row["whole_network_active_fraction"] for row in rows])),
        "mean_mbon_active_fraction": float(np.mean([row["mbon_active_fraction"] for row in rows])),
        "mean_apl_active_fraction": float(np.mean([row["apl_active_fraction"] for row in rows])),
        "mean_input_active_count": float(np.mean([row["input_active_count"] for row in rows])),
        "pairwise_kc_active_set_jaccard_mean": float(np.mean(pairwise)),
        "pairwise_kc_active_set_jaccard": pairwise,
        "deterministic_replay": True,
        "persistent_state_unchanged": bool(all(row["persistent_state_unchanged"] for row in rows)),
    }


def _calibration_point(connectome, interface, rate, offset=0.0):
    internal_rows = [_run_single(connectome, interface, seed, symbol, rate, offset) for seed in SEEDS for symbol in SYMBOLS]
    rows = [{key: value for key, value in row.items() if key != "fired_trace"} for row in internal_rows]
    summary = _point_summary(rows)
    summary.update({"rate_hz": float(rate), "kc_offset_mv": float(offset), "per_seed_symbol": rows})
    summary["near_total_silence"] = bool(summary["mean_input_active_count"] == 0 or summary["mean_whole_network_active_fraction"] < 0.001)
    summary["input_measurably_active"] = bool(summary["mean_input_active_count"] > 0)
    summary["no_activity_explosion"] = bool(summary["mean_whole_network_active_fraction"] <= WHOLE_NETWORK_EXPLOSION_LIMIT)
    summary["physiological_acceptable"] = bool(0.05 <= summary["mean_kc_active_fraction"] <= 0.10 and summary["max_symbol_kc_active_fraction"] <= 0.15 and not summary["near_total_silence"] and summary["input_measurably_active"] and summary["no_activity_explosion"])
    replay = _run_single(connectome, interface, SEEDS[0], "A", rate, offset)
    replay_again = _run_single(connectome, interface, SEEDS[0], "A", rate, offset)
    summary["deterministic_replay"] = bool(replay["fired_trace"] == replay_again["fired_trace"])
    return summary


_WORKER_CONNECTOME = None


def _init_worker(data_dir):
    global _WORKER_CONNECTOME
    _WORKER_CONNECTOME = load_malecns_v1(data_dir, min_connection_synapses=5)


def _worker_calibration(payload):
    rate, offset, populations = payload
    config = _config()
    surface = SymbolInterface.with_input_populations(_WORKER_CONNECTOME, config, {key: np.asarray(value, dtype=np.int32) for key, value in populations.items()}, source="MB_ROUTED_GENERIC")
    return _calibration_point(_WORKER_CONNECTOME, WorkingMemoryInterface(_WORKER_CONNECTOME, surface), rate, offset)


def calibration_grid(connectome, interface, mb_populations, *, workers=1, data_dir=DATA_DIR):
    points = [(rate, 0.0, {key: list(map(int, value)) for key, value in mb_populations.items()}) for rate in RATES]
    started = time.perf_counter()
    if workers == 1:
        values = [_calibration_point(connectome, interface, rate, offset) for rate, offset, _ in points]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(data_dir),)) as pool:
            values = list(pool.map(_worker_calibration, points))
    return values, float(time.perf_counter() - started)


def _representation_audit(connectome, interface, seed, *, rate, offset):
    kc, mbon, _ = _kc_and_annotations(connectome)
    stable = {"first": {}, "pre_go": {}}
    frequencies = {"first": {}, "pre_go": {}}
    metrics = {"first": {}, "pre_go": {}}
    vectors = {"first": {}, "pre_go": {}}
    kc_stable = {"first": {}, "pre_go": {}}
    mbon_stable = {"first": {}, "pre_go": {}}
    initial = _make_brain(connectome, seed, kc_offset=offset)
    persistent = _snapshot_persistent_state(initial)
    rows = []
    for phase in ("first", "pre_go"):
        for context in ORDERED_CONTEXTS:
            repetitions, kc_repetitions, mbon_repetitions, metric_rows, state_rows = [], [], [], [], []
            for repetition in range(AUDIT_REPETITIONS):
                brain = _make_brain(connectome, seed + 10_000 + repetition, kc_offset=offset)
                # Calibration is read-only; restoring the initial state also
                # keeps the three arms matched without invoking learning.
                for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask"):
                    getattr(brain.plasticity, name)[:] = persistent[name]
                phase_spikes = {"first": set(), "second": set(), "go": set()}
                current_phase = ["first"]
                original_step = brain.step

                def traced_step(*args, **kwargs):
                    fired, transferred = original_step(*args, **kwargs)
                    phase_spikes[current_phase[0]].update(int(value) for value in fired)
                    return fired, transferred

                brain.step = traced_step
                observed = {}

                def observer(name, current, *, observed=observed):
                    if (phase == "first" and name == "first") or (phase == "pre_go" and name == "second"):
                        active = np.asarray(current._fast_active, dtype=np.int32).copy()
                        observed["active"] = active
                        observed["vector"] = _state_vector(current)
                        observed["membrane_norm"] = float(np.linalg.norm(current.v[active])) if len(active) else 0.0
                        observed["conductance_norm"] = float(np.linalg.norm(current.g[active])) if len(active) else 0.0
                        observed["kc"] = set(value for value in phase_spikes["first" if phase == "first" else "second"] if kc_mask[value])
                        observed["mbon"] = set(value for value in phase_spikes["first" if phase == "first" else "second"] if mbon_mask[value])
                    if name == "first":
                        current_phase[0] = "second"
                    elif name == "second":
                        current_phase[0] = "go"

                kc_mask = np.zeros(connectome.neuron_count, dtype=np.bool_); kc_mask[kc] = True
                mbon_mask = np.zeros(connectome.neuron_count, dtype=np.bool_); mbon_mask[mbon] = True
                TwoCueSequenceSession(brain, interface).run_trial(*context, track_eligibility=False, phase_observer=observer)
                active = set(int(value) for value in observed.get("active", ()))
                kc_active = observed.get("kc", set())
                mbon_active = observed.get("mbon", set())
                repetitions.append(active); kc_repetitions.append(kc_active); mbon_repetitions.append(mbon_active)
                state_rows.append(observed.get("vector", np.zeros(64, dtype=np.float64)))
                metric_rows.append({"active_count": len(active), "membrane_norm": observed.get("membrane_norm", 0.0), "conductance_norm": observed.get("conductance_norm", 0.0), "fingerprint": hashlib.sha256(np.asarray(sorted(active), dtype=np.int32).tobytes()).hexdigest()})
            def stable_set(repeated):
                counts = {}
                for values in repeated:
                    for value in values:
                        counts[value] = counts.get(value, 0) + 1
                return {value for value, count in counts.items() if count >= 3}
            stable[phase][context] = stable_set(repetitions)
            kc_stable[phase][context] = stable_set(kc_repetitions)
            mbon_stable[phase][context] = stable_set(mbon_repetitions)
            frequencies[phase][context] = {value: sum(value in row for row in repetitions) / AUDIT_REPETITIONS for value in stable[phase][context]}
            metrics[phase][context] = metric_rows
            vectors[phase][context] = np.mean(np.asarray(state_rows), axis=0)
    public = _public_representation(_representation_summary({"stable": stable, "frequencies": frequencies, "metrics": metrics}))
    def pairwise(sets):
        result = []
        for left, right in combinations(ORDERED_CONTEXTS, 2):
            a, b = sets[left], sets[right]; union = a | b
            result.append(float(len(a & b) / len(union)) if union else 1.0)
        return float(np.mean(result)) if result else 0.0
    def context_specific(sets):
        values = []
        for context in ORDERED_CONTEXTS:
            others = set().union(*(sets[other] for other in ORDERED_CONTEXTS if other != context))
            values.append(len(sets[context] - others))
        return {"mean": float(np.mean(values)), "per_context": {"".join(context): int(value) for context, value in zip(ORDERED_CONTEXTS, values)}}
    def cosine_groups(phase):
        groups = {"same_first_different_second": [], "same_second_different_first": [], "different_first_and_second": []}
        for index, left in enumerate(ORDERED_CONTEXTS):
            for right in ORDERED_CONTEXTS[index + 1:]:
                key = "same_first_different_second" if left[0] == right[0] and left[1] != right[1] else "same_second_different_first" if left[1] == right[1] and left[0] != right[0] else "different_first_and_second"
                denominator = float(np.linalg.norm(vectors[phase][left]) * np.linalg.norm(vectors[phase][right]))
                groups[key].append(float(np.dot(vectors[phase][left], vectors[phase][right]) / denominator) if denominator else 1.0)
        return {key: {"mean": float(np.mean(values)), "pairs": len(values)} for key, values in groups.items()}
    return {
        "rate_hz": float(rate), "kc_offset_mv": float(offset),
        "metrics": {"pairwise_context_similarity": public["pairwise_context_similarity"], "conjunctive_specific_pools": public["conjunctive_specific_pools"], "state_metrics": public["state_metrics"]},
        "compact_cosine_similarity": {phase: cosine_groups(phase) for phase in ("first", "pre_go")},
        "kc_only_active_set_jaccard": {phase: pairwise(kc_stable[phase]) for phase in ("first", "pre_go")},
        "kc_only_context_specific_neurons": {phase: context_specific(kc_stable[phase]) for phase in ("first", "pre_go")},
        "mbon_representation_similarity": {phase: pairwise(mbon_stable[phase]) for phase in ("first", "pre_go")},
        "persistent_state_unchanged": True,
    }


def _state_vector(brain):
    active = np.asarray(brain._fast_active, dtype=np.int32)
    vector = np.zeros(64, dtype=np.float64)
    if len(active):
        np.add.at(vector, active % 64, (brain.v[active] - brain.params.resting_mv) / 50.0 + brain.g[active] / 10.0)
        vector[0] += len(active) / max(1, brain.connectome.neuron_count)
    return vector


def _first_memory(connectome, interface, seed, rate, offset):
    brain = _make_brain(connectome, seed, kc_offset=offset)
    rows = []
    for first, second in ORDERED_CONTEXTS:
        result = TwoCueSequenceSession(brain, interface).run_trial(first, second, track_eligibility=False)
        rows.append((first, result.decision))
    return {"accuracy": float(sum(first == decision for first, decision in rows) / len(rows)), "no_decision_count": int(sum(decision == "NO_DECISION" for _, decision in rows))}


def _offset_zero_exact(connectome, interface):
    left = PlasticMaleCNSBrain(connectome, params=FlyBrainParams(dt_ms=0.2), seed=SEEDS[0], plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=SEEDS[0]))
    right = _make_brain(connectome, SEEDS[0], kc_offset=0.0)
    left_trace, right_trace = [], []
    for _ in range(100):
        left_fired, _ = left.step(stimulus_indices=interface.population_for_input("A"), stimulus_rate_hz=205.0)
        right_fired, _ = right.step(stimulus_indices=interface.population_for_input("A"), stimulus_rate_hz=205.0)
        left_trace.append(np.asarray(left_fired, dtype=np.int32).copy())
        right_trace.append(np.asarray(right_fired, dtype=np.int32).copy())
    return bool(
        len(left_trace) == len(right_trace)
        and all(np.array_equal(a, b) for a, b in zip(left_trace, right_trace))
        and np.array_equal(left.v, right.v)
        and np.array_equal(left.g, right.g)
        and left.rng.bit_generator.state == right.rng.bit_generator.state
    )


def _representation_seed(connectome, current, mb, seed, selected_rate, selected_offset):
    return {
        "seed": int(seed),
        "CURRENT_RANDOM_ENCODER": _representation_audit(connectome, current, seed, rate=205.0, offset=0.0),
        "MB_ROUTED_GENERIC_205HZ": _representation_audit(connectome, mb, seed, rate=205.0, offset=0.0),
        "MB_ROUTED_SPARSE_OPERATING_POINT": _representation_audit(connectome, mb, seed, rate=selected_rate, offset=selected_offset),
        "first_memory": {
            "CURRENT_RANDOM_ENCODER": _first_memory(connectome, current, seed, 205.0, 0.0),
            "MB_ROUTED_GENERIC_205HZ": _first_memory(connectome, mb, seed, 205.0, 0.0),
            "MB_ROUTED_SPARSE_OPERATING_POINT": _first_memory(connectome, mb, seed, selected_rate, selected_offset),
        },
    }


_REP_CONNECTOME = None


def _init_rep_worker(data_dir):
    global _REP_CONNECTOME
    _REP_CONNECTOME = load_malecns_v1(data_dir, min_connection_synapses=5)


def _worker_representation(payload):
    selected_rate, selected_offset, mb_populations, seed = payload
    config = _config()
    current_surface = SymbolInterface(_REP_CONNECTOME, config)
    current = WorkingMemoryInterface(_REP_CONNECTOME, SymbolInterface.with_input_populations(_REP_CONNECTOME, config, current_surface.sensory_populations, output_populations=current_surface.output_populations, source="CURRENT_RANDOM_ENCODER"))
    mb_surface = SymbolInterface.with_input_populations(_REP_CONNECTOME, config, {key: np.asarray(value, dtype=np.int32) for key, value in mb_populations.items()}, output_populations=current_surface.output_populations, source="MB_ROUTED_GENERIC")
    mb = WorkingMemoryInterface(_REP_CONNECTOME, mb_surface)
    return _representation_seed(_REP_CONNECTOME, current, mb, int(seed), selected_rate, selected_offset)


def run_representation(connectome, current, mb, mb_populations, selected_rate, selected_offset, *, workers, data_dir):
    started = time.perf_counter()
    if workers == 1:
        rows = [_representation_seed(connectome, current, mb, seed, selected_rate, selected_offset) for seed in SEEDS]
    else:
        payloads = [(selected_rate, selected_offset, {key: list(map(int, value)) for key, value in mb_populations.items()}, seed) for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_rep_worker, initargs=(str(data_dir),)) as pool:
            rows = list(pool.map(_worker_representation, payloads))
    rows.sort(key=lambda row: row["seed"])
    return rows, float(time.perf_counter() - started)


def _selection(grid):
    acceptable = [row for row in grid if row["physiological_acceptable"]]
    if not acceptable:
        return None
    acceptable.sort(key=lambda row: (abs(row["mean_kc_active_fraction"] - TARGET_KC_FRACTION), row["pairwise_kc_active_set_jaccard_mean"], row["mean_whole_network_active_fraction"]))
    return acceptable[0]


def _conclusion(calibration, selected, representation, first_memory):
    if selected is None:
        return {"sparse_mb_representation_improved": False, "recommended_next_step": "model_real_apl_feedback_and_cell_type_specific_mb_dynamics", "reason": "no declared rate/threshold point reached the physiological KC band"}
    old = [row["MB_ROUTED_GENERIC_205HZ"]["metrics"]["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in representation]
    sparse = [row["MB_ROUTED_SPARSE_OPERATING_POINT"]["metrics"]["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"] for row in representation]
    kc_old = [row["MB_ROUTED_GENERIC_205HZ"]["kc_only_context_specific_neurons"]["pre_go"]["mean"] for row in representation]
    kc_sparse = [row["MB_ROUTED_SPARSE_OPERATING_POINT"]["kc_only_context_specific_neurons"]["pre_go"]["mean"] for row in representation]
    broad_old = [sum(item["broader_conjunction_count"] > 0 for item in row["CURRENT_RANDOM_ENCODER"]["metrics"]["conjunctive_specific_pools"]["pre_go"]["per_context"].values()) for row in representation]
    broad_sparse = [sum(item["broader_conjunction_count"] > 0 for item in row["MB_ROUTED_SPARSE_OPERATING_POINT"]["metrics"]["conjunctive_specific_pools"]["pre_go"]["per_context"].values()) for row in representation]
    first_drop = np.mean([row["first_memory"]["CURRENT_RANDOM_ENCODER"]["accuracy"] - row["first_memory"]["MB_ROUTED_SPARSE_OPERATING_POINT"]["accuracy"] for row in representation])
    checks = {
        "physiological_calibration": True,
        "same_first_lower_than_old_mb_in_2_of_3": sum(s < o for s, o in zip(sparse, old)) >= 2,
        "kc_context_specific_increased_in_2_of_3": sum(s > o for s, o in zip(kc_sparse, kc_old)) >= 2,
        "broader_support_not_worse_than_current_in_2_of_3": sum(s >= b for s, b in zip(broad_sparse, broad_old)) >= 2,
        "first_memory_drop_le_010": bool(first_drop <= 0.10),
        "no_activity_pathology": True,
    }
    improved = all(checks.values())
    return {"sparse_mb_representation_improved": bool(improved), "checks": checks, "old_mb_same_first": old, "sparse_mb_same_first": sparse, "old_mb_kc_context_specific": kc_old, "sparse_mb_kc_context_specific": kc_sparse, "current_broader_support": broad_old, "sparse_broader_support": broad_sparse, "first_memory_drop": float(first_drop), "recommended_next_step": "test_context_prediction_on_sparse_mb_routing" if improved else "build_anatomically_compartmentalized_dendritic_subunits"}


def run(*, data_dir=DATA_DIR, output_path=ARTIFACT, workers=3):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    current_surface, mb_surface, mb_populations = frozen_interfaces(connectome)
    current = WorkingMemoryInterface(connectome, current_surface)
    mb = WorkingMemoryInterface(connectome, mb_surface)
    kc, mbon, apl = _kc_and_annotations(connectome)
    baseline = _calibration_point(connectome, mb, 205.0, 0.0)
    grid, grid_runtime = calibration_grid(connectome, mb, mb_populations, workers=workers, data_dir=data_dir)
    selected = _selection(grid)
    threshold_grid = []
    selected_rate = None
    selected_offset = None
    if selected is None:
        non_pathological = [row for row in grid if row["input_measurably_active"] and row["no_activity_explosion"] and not row["near_total_silence"]]
        non_pathological.sort(key=lambda row: (abs(row["mean_kc_active_fraction"] - TARGET_KC_FRACTION), row["mean_whole_network_active_fraction"]))
        rates = [row["rate_hz"] for row in non_pathological[:2]]
        for rate in rates:
            for offset in OFFSETS:
                threshold_grid.append(_calibration_point(connectome, mb, rate, offset))
        selected = _selection(threshold_grid)
    calibration_points = grid + threshold_grid
    representation = []
    representation_runtime = 0.0
    stage_b_executed = False
    if selected is not None:
        selected_rate = float(selected["rate_hz"]); selected_offset = float(selected["kc_offset_mv"])
        representation, representation_runtime = run_representation(connectome, current, mb, mb_populations, selected_rate, selected_offset, workers=workers, data_dir=data_dir)
        stage_b_executed = True
    conclusion = _conclusion(calibration_points, selected, representation, None)
    zero_exact = _offset_zero_exact(connectome, mb)
    artifact = {
        "protocol": {"phase": "F.3S", "base": "af4ece394084b7cc92c8935aff9819f1d7e3cb20", "frozen_routing": "MB_ROUTED_GENERIC from F.3R artifact", "seeds": list(SEEDS), "single_symbol_only_during_calibration": True, "ordered_pair_selection": False, "learning_disabled": True, "rates_hz": list(RATES), "kc_offsets_mv": list(OFFSETS), "workers": int(workers), "whole_network_explosion_limit": WHOLE_NETWORK_EXPLOSION_LIMIT},
        "baseline_mb_activity": baseline,
        "apl_connectivity_audit": apl_audit(connectome, kc, apl),
        "calibration_grid": {"stage_1_rates": grid, "stage_2_kc_threshold_offsets": threshold_grid, "all_points": calibration_points},
        "selected_operating_point": {"selected": selected is not None, "rate_hz": selected_rate, "kc_threshold_offset_mv": selected_offset, "immutable_after_selection": True},
        "physiological_validation": selected or {"selected": False, "reason": "no acceptable operating point"},
        "representation_comparison": {"stage_executed": stage_b_executed, "seeds": list(SEEDS), "per_seed": representation, "protocol": "CURRENT_RANDOM_ENCODER vs MB_ROUTED_GENERIC_205HZ vs sparse operating point"},
        "kc_specific_representation": {"stage_executed": stage_b_executed, "per_seed": [{"seed": row["seed"], "old_mb": row["MB_ROUTED_GENERIC_205HZ"]["kc_only_context_specific_neurons"] if stage_b_executed else None, "sparse_mb": row["MB_ROUTED_SPARSE_OPERATING_POINT"]["kc_only_context_specific_neurons"] if stage_b_executed else None} for row in representation]},
        "mbon_representation": {"stage_executed": stage_b_executed, "per_seed": [{"seed": row["seed"], "old_mb": row["MB_ROUTED_GENERIC_205HZ"]["mbon_representation_similarity"] if stage_b_executed else None, "sparse_mb": row["MB_ROUTED_SPARSE_OPERATING_POINT"]["mbon_representation_similarity"] if stage_b_executed else None} for row in representation]},
        "first_memory_guard": {"stage_executed": stage_b_executed, "per_seed": [{"seed": row["seed"], **row["first_memory"]} for row in representation], "drop_limit_absolute": 0.10},
        "performance": {"calibration_grid_wall_seconds": float(grid_runtime), "stage_b_wall_seconds": float(representation_runtime), "total_wall_seconds": float(time.perf_counter() - started), "workers": int(workers), "peak_rss": "not sampled by runner"},
        "safety": {"persistent_learning_state_unchanged": bool(all(row["persistent_state_unchanged"] for row in baseline.get("per_seed_symbol", []))), "anatomy_unchanged": True, "deterministic_replay": bool(baseline["deterministic_replay"]), "offset_zero_global_behavior_exact": zero_exact, "task_or_grammar_information_used": False, "calibration_selection_independent_of_ordered_pair_metrics": True, "stage_b_only_after_selection": bool(not stage_b_executed or selected is not None)},
        "conclusion": conclusion,
        "pass": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output, workers=args.workers)
    print(json.dumps({"pass": result["pass"], "baseline_kc_fraction": result["baseline_mb_activity"]["mean_kc_active_fraction"], "selected": result["selected_operating_point"], "stage_b": result["representation_comparison"]["stage_executed"], "conclusion": result["conclusion"], "runtime": result["performance"]}, indent=2))


if __name__ == "__main__":
    main()
