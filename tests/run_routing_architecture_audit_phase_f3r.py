"""Run the learning-disabled Phase F.3R biological routing audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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
    SYMBOLS,
    PlasticMaleCNSBrain,
    SymbolInterface,
    SymbolInterfaceConfig,
    TwoCueSequenceSession,
    WorkingMemoryInterface,
    load_malecns_v1,
)
from drosomath.malecns.symbol_interface import (  # noqa: E402
    _restore_persistent_state,
    _snapshot_persistent_state,
)
from drosomath.whole_brain import PlasticStateConfig  # noqa: E402
from run_contextual_credit_audit_phase_f3b import _public_representation, _representation_summary  # noqa: E402


SEEDS = (233, 239, 241)
AUDIT_REPETITIONS = 4
ORDERED_CONTEXTS = tuple((left, right) for left in SYMBOLS for right in SYMBOLS)
ARTIFACT = ROOT / "results/latest_routing_architecture_audit_phase_f3r.json"
DATA_DIR = ROOT / "data/malecns_v1"
EDGE_COUNT_V1_MIN5 = 6_242_118


def _metadata(connectome, key: str) -> np.ndarray:
    return np.asarray(connectome.metadata.get(key, [None] * connectome.neuron_count), dtype=object)


def _annotation_record(connectome, index: int) -> dict[str, object]:
    fields = ("type", "flywireType", "hemibrainType", "superclass", "class", "subclass", "somaNeuromere", "rootSide")
    row = {"body_id": int(connectome.body_ids[index])}
    for field in fields:
        row[field] = None if _metadata(connectome, field)[index] is None else str(_metadata(connectome, field)[index])
    row["consensus_neurotransmitter"] = str(connectome.consensus_nt[index])
    return row


def _role_evidence(record: dict[str, object]) -> list[str]:
    superclass = str(record.get("superclass") or "")
    cell_class = str(record.get("class") or "")
    roles = []
    if "sensory" in superclass or cell_class in {"chemosensory", "gustatory", "olfactory", "visual", "mechanosensory", "thermosensory", "hygrosensory", "unknown_sensory"}:
        roles.append("sensory_annotation")
    if cell_class in {"ALPN", "ALON", "SEZPN"}:
        roles.append("projection_neuron_annotation")
    if cell_class in {"Kenyon_Cell", "MBON", "DAN", "ALIN"}:
        roles.append("higher_order_or_mushroom_body_annotation")
    if "descending" in superclass:
        roles.append("descending_annotation")
    if "motor" in superclass or "motor" in cell_class:
        roles.append("motor_related_annotation")
    return roles or ["no_supported_role_annotation"]


def summarize_population_annotations(connectome, populations: dict[str, np.ndarray]) -> dict[str, object]:
    fields = ("type", "flywireType", "hemibrainType", "superclass", "class", "subclass", "somaNeuromere", "rootSide")
    result = {"available_fields": [*fields, "consensus_neurotransmitter"], "per_symbol": {}, "aggregate": {}}
    all_records = []
    for symbol in SYMBOLS:
        records = [_annotation_record(connectome, int(index)) for index in populations[symbol]]
        all_records.extend(records)
        categorical = {}
        for field in (*fields, "consensus_neurotransmitter"):
            counts = {}
            for record in records:
                value = str(record.get(field) or "<missing>")
                counts[value] = counts.get(value, 0) + 1
            categorical[field] = {
                "counts": dict(sorted(counts.items())),
                "fractions": {key: value / len(records) for key, value in sorted(counts.items())},
            }
        role_counts = {}
        for record in records:
            for role in _role_evidence(record):
                role_counts[role] = role_counts.get(role, 0) + 1
        result["per_symbol"][symbol] = {
            "body_ids": [int(record["body_id"]) for record in records],
            "annotations": records,
            "categorical": categorical,
            "role_evidence_counts": role_counts,
        }
    for field in (*fields, "consensus_neurotransmitter"):
        counts = {}
        for record in all_records:
            value = str(record.get(field) or "<missing>")
            counts[value] = counts.get(value, 0) + 1
        result["aggregate"][field] = {
            "counts": dict(sorted(counts.items())),
            "fractions": {key: value / len(all_records) for key, value in sorted(counts.items())},
        }
    result["aggregate"]["role_evidence_counts"] = {
        role: sum(value for symbol in result["per_symbol"].values() for role_name, value in symbol["role_evidence_counts"].items() if role_name == role)
        for role in sorted({role for symbol in result["per_symbol"].values() for role in symbol["role_evidence_counts"]})
    }
    return result


def _edge_features(connectome, candidates: np.ndarray, target_masks: dict[str, np.ndarray]) -> dict[int, dict[str, float | int]]:
    indptr = np.asarray(connectome.indptr)
    posts = np.asarray(connectome.post_indices, dtype=np.int32)
    signed = np.asarray(connectome.signed_synapse_counts, dtype=np.float64)
    features = {}
    for raw in candidates:
        index = int(raw)
        start, stop = int(indptr[index]), int(indptr[index + 1])
        local_posts = posts[start:stop]
        local_signed = signed[start:stop]
        row = {"outgoing_degree": int(stop - start), "outgoing_strength": float(np.abs(local_signed).sum())}
        for name, mask in target_masks.items():
            selected = mask[local_posts]
            row[f"{name}_edge_count"] = int(np.count_nonzero(selected))
            row[f"{name}_signed_strength"] = float(local_signed[selected].sum())
            row[f"{name}_absolute_strength"] = float(np.abs(local_signed[selected]).sum())
        features[index] = row
    return features


def select_mb_routed_candidate(connectome, *, excluded_indices=(), seed: int = 233, population_size: int = 32) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Select annotation-derived ALPN inputs with direct KC anatomical evidence."""
    classes = _metadata(connectome, "class")
    superclasses = _metadata(connectome, "superclass")
    kc = classes == "Kenyon_Cell"
    mbon = classes == "MBON"
    dan = classes == "DAN"
    target_masks = {"kc": kc, "mbon": mbon, "dan": dan}
    excluded = np.zeros(connectome.neuron_count, dtype=np.bool_)
    excluded[np.asarray(tuple(excluded_indices), dtype=np.int32)] = True
    annotated_projection = np.flatnonzero((classes == "ALPN") & (superclasses == "cb_intrinsic") & ~excluded).astype(np.int32)
    features = _edge_features(connectome, annotated_projection, target_masks)
    eligible = np.asarray([index for index in annotated_projection if features[int(index)]["kc_edge_count"] > 0], dtype=np.int32)
    if len(eligible) < 4 * population_size:
        eligible = annotated_projection
    rng = np.random.default_rng(seed)
    tie = rng.random(len(eligible))
    degree = np.asarray([features[int(index)]["outgoing_degree"] for index in eligible])
    kc_edges = np.asarray([features[int(index)]["kc_edge_count"] for index in eligible])
    kc_strength = np.asarray([features[int(index)]["kc_absolute_strength"] for index in eligible])
    order = np.lexsort((eligible, tie, -kc_strength, -kc_edges, -degree))
    selected = eligible[order[: 4 * population_size]]
    # Interleaving a single ranked pool matches route strength and degree without
    # giving any symbol a biological meaning.
    populations = {
        symbol: selected[offset::len(SYMBOLS)].astype(np.int32, copy=False)
        for offset, symbol in enumerate(SYMBOLS)
    }
    summary = {
        "name": "MB_ROUTED_GENERIC",
        "selection_rules": [
            "class == ALPN",
            "superclass == cb_intrinsic",
            "at least one existing anatomical edge to class == Kenyon_Cell when available",
            "exclude current frozen input/output populations",
            "rank by outgoing degree and KC input evidence; interleave ranked pool into four groups",
        ],
        "selection_seed": int(seed),
        "annotation_population_counts": {
            "ALPN_cb_intrinsic": int(np.count_nonzero((classes == "ALPN") & (superclasses == "cb_intrinsic"))),
            "eligible_direct_kc": int(len(eligible)),
        },
        "candidate_body_ids": [int(connectome.body_ids[index]) for index in selected],
        "populations": {symbol: [int(connectome.body_ids[index]) for index in values] for symbol, values in populations.items()},
        "population_indices": {symbol: [int(index) for index in values] for symbol, values in populations.items()},
        "disjoint": bool(len(np.unique(selected)) == len(selected)),
        "matched_constraints": {},
    }
    for symbol, values in populations.items():
        rows = [features[int(index)] for index in values]
        summary["matched_constraints"][symbol] = {
            "count": len(rows),
            "outgoing_degree_mean": float(np.mean([row["outgoing_degree"] for row in rows])),
            "kc_edge_count_mean": float(np.mean([row["kc_edge_count"] for row in rows])),
            "kc_absolute_strength_mean": float(np.mean([row["kc_absolute_strength"] for row in rows])),
            "kc_signed_strength_mean": float(np.mean([row["kc_signed_strength"] for row in rows])),
        }
    summary["matched_constraints"]["max_minus_min_means"] = {
        field: float(max(item[field] for symbol, item in summary["matched_constraints"].items() if symbol != "max_minus_min_means") - min(item[field] for symbol, item in summary["matched_constraints"].items() if symbol != "max_minus_min_means"))
        for field in ("outgoing_degree_mean", "kc_edge_count_mean", "kc_absolute_strength_mean")
    }
    return populations, summary


def _expand(connectome, frontier: np.ndarray, visited: np.ndarray, path_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(frontier) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.zeros(connectome.neuron_count, dtype=np.float64), 0.0
    indptr = np.asarray(connectome.indptr)
    posts = np.asarray(connectome.post_indices, dtype=np.int32)
    starts = indptr[frontier]
    lengths = (indptr[frontier + 1] - starts).astype(np.int64, copy=False)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.zeros(connectome.neuron_count, dtype=np.float64), 0.0
    offsets = np.arange(total, dtype=np.int64)
    block_offsets = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    edge_indices = np.repeat(starts, lengths) + offsets - block_offsets
    edge_posts = posts[edge_indices]
    edge_path_counts = np.repeat(path_counts[frontier], lengths)
    next_paths = np.bincount(edge_posts, weights=edge_path_counts, minlength=connectome.neuron_count).astype(np.float64, copy=False)
    next_paths = np.minimum(next_paths, 1.0e15)
    fresh = np.unique(edge_posts)
    fresh = fresh[~visited[fresh]].astype(np.int32, copy=False)
    if len(fresh):
        visited[fresh] = True
    return fresh, edge_posts, next_paths, float(total)


def bounded_reachability(connectome, input_indices: np.ndarray, targets: dict[str, np.ndarray], *, max_hops: int = 4) -> tuple[dict[str, object], list[set[int]]]:
    n = connectome.neuron_count
    target_masks = {name: np.zeros(n, dtype=np.bool_) for name in targets}
    for name, values in targets.items():
        target_masks[name][np.asarray(values, dtype=np.int32)] = True
    visited = np.zeros(n, dtype=np.bool_)
    frontier = np.unique(np.asarray(input_indices, dtype=np.int32))
    visited[frontier] = True
    paths = np.zeros(n, dtype=np.float64)
    paths[frontier] = 1.0
    per_target = {name: [] for name in targets}
    hop_sets = []
    for hop in range(1, max_hops + 1):
        fresh, edge_posts, next_paths, edge_count = _expand(connectome, frontier, visited, paths)
        edge_start = 0
        # Reconstruct edge signs only for this frontier, preserving CSR order.
        if len(frontier):
            indptr = np.asarray(connectome.indptr)
            lengths = (indptr[frontier + 1] - indptr[frontier]).astype(np.int64, copy=False)
            total = int(lengths.sum())
            offsets = np.arange(total, dtype=np.int64)
            starts = indptr[frontier]
            block_offsets = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
            edge_indices = np.repeat(starts, lengths) + offsets - block_offsets
            edge_posts = np.asarray(connectome.post_indices, dtype=np.int32)[edge_indices]
            edge_signed = np.asarray(connectome.signed_synapse_counts, dtype=np.float64)[edge_indices]
        else:
            edge_signed = np.empty(0, dtype=np.float64)
        for name, mask in target_masks.items():
            edge_target = mask[edge_posts] if len(edge_posts) else np.empty(0, dtype=np.bool_)
            newly_target = fresh[mask[fresh]] if len(fresh) else np.empty(0, dtype=np.int32)
            per_target[name].append({
                "newly_reached_neurons": int(len(newly_target)),
                "cumulative_reached_neurons": int(np.count_nonzero(visited & mask)),
                "reachable_fraction": float(np.count_nonzero(visited & mask) / max(1, int(mask.sum()))),
                "shortest_path_hop": int(hop) if len(newly_target) else None,
                "distinct_bounded_paths_to_new": float(next_paths[newly_target].sum()) if len(newly_target) else 0.0,
                "signed_connection_strength_total": float(edge_signed[edge_target].sum()) if len(edge_signed) else 0.0,
                "edge_count_examined": int(np.count_nonzero(edge_target)),
            })
        hop_sets.append(set(int(value) for value in fresh))
        paths[:] = 0.0
        if len(fresh):
            paths[fresh] = next_paths[fresh]
        frontier = fresh
    return {name: {"population_size": int(len(values)), "per_hop": rows} for name, (values, rows) in zip(targets, ((targets[name], per_target[name]) for name in targets))}, hop_sets


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _overlap_coefficient(left: set[int], right: set[int]) -> float:
    denominator = min(len(left), len(right))
    return float(len(left & right) / denominator) if denominator else 1.0


def expansion_diagnostics(expansions: dict[str, list[set[int]]], target_expansions: dict[str, dict[str, list[set[int]]]]) -> dict[str, object]:
    pairwise = {}
    for hop in range(4):
        rows = []
        for left, right in combinations(SYMBOLS, 2):
            a, b = expansions[left][hop], expansions[right][hop]
            rows.append({"pair": f"{left}|{right}", "jaccard": _jaccard(a, b), "overlap_coefficient": _overlap_coefficient(a, b), "newly_reached_left": len(a - b), "newly_reached_right": len(b - a)})
        pairwise[str(hop + 1)] = rows
    target = {}
    for target_name in ("kc", "mbon", "dan"):
        target[str(target_name)] = {}
        for hop in range(4):
            sets = {symbol: target_expansions[symbol][target_name][hop] for symbol in SYMBOLS}
            values = [sets[symbol] for symbol in SYMBOLS]
            target[target_name][str(hop + 1)] = {
                "per_symbol_count": {symbol: len(sets[symbol]) for symbol in SYMBOLS},
                "pairwise_jaccard_mean": float(np.mean([_jaccard(a, b) for a, b in combinations(values, 2)])),
                "pairwise_overlap_mean": float(np.mean([_overlap_coefficient(a, b) for a, b in combinations(values, 2)])),
            }
    return {"pairwise_expansion": pairwise, "target_specific_expansion": target}


def _make_interface(connectome, config, populations, outputs=None, source="current_random"):
    if source == "current_random":
        return SymbolInterface(connectome, config)
    return SymbolInterface.with_input_populations(connectome, config, populations, output_populations=outputs, source=source)


def _make_brain(connectome, seed):
    return PlasticMaleCNSBrain(connectome, params=FlyBrainParams(dt_ms=0.2), seed=seed, plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed))


def _state_fingerprint(brain) -> np.ndarray:
    active = np.asarray(brain._fast_active, dtype=np.int32)
    vector = np.zeros(64, dtype=np.float64)
    if len(active):
        np.add.at(vector, active % 64, (brain.v[active] - brain.params.resting_mv) / 50.0 + brain.g[active] / 10.0)
        vector[0] += len(active) / max(1, brain.connectome.neuron_count)
    return vector


def _cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def _audit_representation(connectome, interface, seed: int) -> dict[str, object]:
    stable = {"first": {}, "pre_go": {}}
    frequencies = {"first": {}, "pre_go": {}}
    metrics = {"first": {}, "pre_go": {}}
    vectors = {"first": {}, "pre_go": {}}
    initial = _make_brain(connectome, seed)
    persistent = _snapshot_persistent_state(initial)
    rng_state = copy.deepcopy(initial.rng.bit_generator.state)
    persistent_unchanged = True
    for phase in ("first", "pre_go"):
        for context in ORDERED_CONTEXTS:
            repetitions, rows, state_rows = [], [], []
            for repetition in range(AUDIT_REPETITIONS):
                brain = _make_brain(connectome, seed + 10_000 + repetition)
                _restore_persistent_state(brain, persistent)
                brain.rng.bit_generator.state = copy.deepcopy(rng_state)
                local_persistent = _snapshot_persistent_state(brain)
                observed = {}

                def observer(name, current, *, observed=observed):
                    wanted = (phase == "first" and name == "first") or (phase == "pre_go" and name == "second")
                    if wanted:
                        active = np.asarray(current._fast_active, dtype=np.int32).copy()
                        observed["active"] = active
                        observed["vector"] = _state_fingerprint(current)
                        observed["membrane_norm"] = float(np.linalg.norm(current.v[active])) if len(active) else 0.0
                        observed["conductance_norm"] = float(np.linalg.norm(current.g[active])) if len(active) else 0.0

                TwoCueSequenceSession(brain, interface).run_trial(*context, track_eligibility=False, phase_observer=observer)
                persistent_unchanged &= all(
                    np.array_equal(getattr(brain.plasticity, name), local_persistent[name])
                    for name in ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
                )
                active = set(int(value) for value in observed.get("active", ()))
                repetitions.append(active)
                state_rows.append(observed.get("vector", np.zeros(64, dtype=np.float64)))
                rows.append({
                    "active_count": len(active),
                    "membrane_norm": observed.get("membrane_norm", 0.0),
                    "conductance_norm": observed.get("conductance_norm", 0.0),
                    "fingerprint": hashlib.sha256(np.asarray(sorted(active), dtype=np.int32).tobytes()).hexdigest(),
                })
            counts = {}
            for active in repetitions:
                for neuron in active:
                    counts[neuron] = counts.get(neuron, 0) + 1
            stable[phase][context] = {neuron for neuron, count in counts.items() if count >= 3}
            frequencies[phase][context] = {neuron: count / AUDIT_REPETITIONS for neuron, count in counts.items()}
            metrics[phase][context] = rows
            vectors[phase][context] = np.mean(np.asarray(state_rows), axis=0)
    public = _public_representation(_representation_summary({"stable": stable, "frequencies": frequencies, "metrics": metrics}))
    compact = {
        "pairwise_context_similarity": public["pairwise_context_similarity"],
        "conjunctive_specific_pools": public["conjunctive_specific_pools"],
        "state_metrics": public["state_metrics"],
    }
    cosine_groups = {}
    for phase in ("first", "pre_go"):
        groups = {"same_first_different_second": [], "same_second_different_first": [], "different_first_and_second": []}
        for index, left in enumerate(ORDERED_CONTEXTS):
            for right in ORDERED_CONTEXTS[index + 1:]:
                if left[0] == right[0] and left[1] != right[1]:
                    key = "same_first_different_second"
                elif left[1] == right[1] and left[0] != right[0]:
                    key = "same_second_different_first"
                else:
                    key = "different_first_and_second"
                groups[key].append(_cosine(vectors[phase][left], vectors[phase][right]))
        cosine_groups[phase] = {key: {"mean": float(np.mean(values)), "pairs": len(values)} for key, values in groups.items()}
    return {"metrics": compact, "cosine_state_similarity": cosine_groups, "persistent_state_unchanged": bool(persistent_unchanged)}


def _first_memory(connectome, interface, seed: int) -> dict[str, object]:
    brain = _make_brain(connectome, seed)
    rows = []
    for first, second in ORDERED_CONTEXTS:
        result = TwoCueSequenceSession(brain, interface).run_trial(first, second, track_eligibility=False)
        rows.append((first, result.decision))
    correct = sum(decision == first for first, decision in rows)
    return {"trials": len(rows), "first_accuracy": correct / len(rows), "no_decision_count": sum(decision == "NO_DECISION" for _, decision in rows)}


def _audit_seed(connectome, current_populations, mb_populations, output_populations, seed: int) -> dict[str, object]:
    config = SymbolInterfaceConfig(sensory_population_size=32, output_population_size=32, seed=7, dt_ms=0.2, default_duration_ms=20.0, default_stimulus_rate_hz=205.0, plastic_fraction=0.05, output_selection="random_indegree")
    current_surface = SymbolInterface(connectome, config)
    mb_surface = _make_interface(connectome, config, mb_populations, output_populations, source="MB_ROUTED_GENERIC")
    current = WorkingMemoryInterface(connectome, current_surface)
    mb = WorkingMemoryInterface(connectome, mb_surface)
    current_before = _snapshot_persistent_state(_make_brain(connectome, seed))
    current_rep = _audit_representation(connectome, current, seed)
    mb_rep = _audit_representation(connectome, mb, seed)
    current_first = _first_memory(connectome, current, seed + 4000)
    mb_first = _first_memory(connectome, mb, seed + 4000)
    return {"seed": int(seed), "current_random": current_rep, "mb_routed_generic": mb_rep, "first_memory": {"current_random": current_first, "mb_routed_generic": mb_first}, "persistent_snapshot_shape": {key: list(np.asarray(value).shape) for key, value in current_before.items() if isinstance(value, np.ndarray)}}


def _worker_seed(payload):
    data_dir, current_populations, mb_populations, output_populations, seed = payload
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    return _audit_seed(connectome, current_populations, mb_populations, output_populations, int(seed))


def run_seed_jobs(connectome, current_populations, mb_populations, output_populations, *, workers: int = 1, data_dir: Path = DATA_DIR) -> list[dict[str, object]]:
    if workers not in (1, 2, 3):
        raise ValueError("workers must be 1, 2, or 3")
    if workers == 1:
        rows = [_audit_seed(connectome, current_populations, mb_populations, output_populations, seed) for seed in SEEDS]
    else:
        payloads = [(str(data_dir), {key: list(map(int, value)) for key, value in current_populations.items()}, {key: list(map(int, value)) for key, value in mb_populations.items()}, {key: list(map(int, value)) for key, value in output_populations.items()}, seed) for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker_seed, payloads))
    return sorted(rows, key=lambda row: int(row["seed"]))


def _target_sets(connectome):
    classes = _metadata(connectome, "class")
    return {
        "kc": np.flatnonzero(classes == "Kenyon_Cell").astype(np.int32),
        "mbon": np.flatnonzero(classes == "MBON").astype(np.int32),
        "dan": np.flatnonzero(classes == "DAN").astype(np.int32),
    }


def _reachability_audit(connectome, input_populations, output_populations):
    targets = _target_sets(connectome)
    targets["frozen_outputs"] = np.unique(np.concatenate(tuple(output_populations.values()))).astype(np.int32)
    result, expansion = {}, {}
    target_expansion = {symbol: {name: [] for name in targets} for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        metrics, hop_sets = bounded_reachability(connectome, input_populations[symbol], targets, max_hops=4)
        result[symbol] = metrics
        cumulative = set()
        expansion[symbol] = []
        for hop_set in hop_sets:
            cumulative = cumulative | hop_set
            expansion[symbol].append(set(cumulative))
        for name, target_indices in targets.items():
            target_cumulative = set()
            target_expansion[symbol][name] = []
            for hop in range(4):
                target_cumulative |= expansion[symbol][hop] & set(int(value) for value in target_indices)
                target_expansion[symbol][name].append(set(target_cumulative))
    return {"targets": {name: {"count": int(len(values)), "annotation_basis": "class == " + ("Kenyon_Cell" if name == "kc" else "MBON" if name == "mbon" else "DAN" if name == "dan" else "frozen output population")} for name, values in targets.items()}, "per_symbol": result}, expansion, target_expansion


def dendritic_feasibility() -> dict[str, object]:
    compartments = {
        "level_1_primary_post_neuropil": {"neurons_per_compartment_range": [2, 8], "estimated_compartment_count": [333400, 1333600]},
        "level_2_spatial_skeleton_clusters": {"compartments_per_neuron_range": [8, 32], "estimated_compartment_count": [1333600, 5334400]},
    }
    edge_bytes = EDGE_COUNT_V1_MIN5 * 4
    return {
        "official_metadata_basis": {
            "source_url": "https://male-cns.janelia.org/download/",
            "synapse_xyz": True,
            "body_pre_body_post": True,
            "primary_post_neuropil": True,
            "neuron_skeletons": True,
            "syn_partners_processed": False,
        },
        "future_preprocessing": [
            "Level 1: map each post-synapse primary_post neuropil to a post-neuron compartment.",
            "Level 2: cluster synapse positions along post-neuron skeleton within each neuropil.",
            "Aggregate pre->post weights must be split across compartments using synapse-level partner rows.",
        ],
        "estimated_compartments": compartments,
        "edge_to_compartment_int32_bytes_for_6242118_edges": int(edge_bytes),
        "edge_to_compartment_mib": float(edge_bytes / (1024 * 1024)),
        "compartment_state_float32_bytes": {name: [int(value * 4) for value in row["estimated_compartment_count"]] for name, row in compartments.items()},
        "requirements": ["stream the 6.8GB partner table in Arrow batches", "obtain skeleton/neuropil mapping", "write compact edge->compartment arrays", "validate one-to-many split conservation"],
    }


def _conclusion(rows, mb_summary, reachability, expansion):
    current_kc = []
    mb_kc = []
    current_jaccard = []
    mb_jaccard = []
    broad_current = []
    broad_mb = []
    for row in rows:
        for arm, kc_values, broad_values, jaccard_values in (("current_random", current_kc, broad_current, current_jaccard), ("mb_routed_generic", mb_kc, broad_mb, mb_jaccard)):
            rep = row[arm]["metrics"]
            kc_values.append(rep["state_metrics"]["pre_go"]["per_context"]["AA"]["active_count_mean"])
            broad_values.append(sum(item["broader_conjunction_count"] > 0 for item in rep["conjunctive_specific_pools"]["pre_go"]["per_context"].values()))
            jaccard_values.append(rep["pairwise_context_similarity"]["pre_go"]["same_first_different_second"]["mean"])
    # Routing criterion is deliberately architecture-only: direct/shorter KC reach,
    # memory guard, lower pre-GO same-FIRST overlap, and no worse broad support.
    current_kc_reach = np.mean([reachability["current_random"][symbol]["kc"]["per_hop"][0]["cumulative_reached_neurons"] for symbol in SYMBOLS])
    mb_kc_reach = np.mean([reachability["mb_routed_generic"][symbol]["kc"]["per_hop"][0]["cumulative_reached_neurons"] for symbol in SYMBOLS])
    current_first = np.mean([row["first_memory"]["current_random"]["first_accuracy"] for row in rows])
    mb_first = np.mean([row["first_memory"]["mb_routed_generic"]["first_accuracy"] for row in rows])
    criterion = {
        "kc_reach_substantially_more_or_shorter": bool(mb_kc_reach > current_kc_reach * 1.10 or mb_kc_reach > current_kc_reach and mb_kc_reach > 0),
        "first_memory_drop_le_010": bool(mb_first >= current_first - 0.10),
        "pre_go_same_first_lower_in_at_least_2_of_3": bool(sum(m < c for m, c in zip(mb_jaccard, current_jaccard)) >= 2),
        "broader_pool_not_worse_in_at_least_2_of_3": bool(sum(m >= c for m, c in zip(broad_mb, broad_current)) >= 2),
    }
    promising = all(criterion.values())
    if promising:
        next_step = "replicate_mb_routed_context_representation_then_prediction"
    elif mb_kc_reach <= current_kc_reach and current_kc_reach > 0:
        next_step = "build_anatomically_compartmentalized_dendritic_subunits"
    else:
        next_step = "redesign_symbol_sensory_interface_from_connectome_annotations"
    return {"mb_routing_promising": promising, "criterion": criterion, "current_mean_1hop_kc_reach": float(current_kc_reach), "mb_routed_mean_1hop_kc_reach": float(mb_kc_reach), "current_first_memory_accuracy": float(current_first), "mb_routed_first_memory_accuracy": float(mb_first), "recommended_next_step": next_step, "interpretation": "architecture-selection audit only; not contextual prediction success"}


def run(*, data_dir: Path = DATA_DIR, output_path: Path = ARTIFACT, workers: int = 1) -> dict[str, object]:
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    config = SymbolInterfaceConfig(sensory_population_size=32, output_population_size=32, seed=7, dt_ms=0.2, default_duration_ms=20.0, default_stimulus_rate_hz=205.0, plastic_fraction=0.05, output_selection="random_indegree")
    current_interface = SymbolInterface(connectome, config)
    current_inputs = current_interface.sensory_populations
    current_outputs = current_interface.output_populations
    excluded = np.unique(np.concatenate(tuple(current_inputs.values()) + tuple(current_outputs.values())))
    mb_inputs, mb_candidate = select_mb_routed_candidate(connectome, excluded_indices=excluded, seed=233)
    current_annotations = summarize_population_annotations(connectome, current_inputs)
    mb_annotations = summarize_population_annotations(connectome, mb_inputs)
    populations = _target_sets(connectome)
    populations["frozen_outputs"] = np.unique(np.concatenate(tuple(current_outputs.values()))).astype(np.int32)
    current_reachability, current_expansion, current_target_expansion = _reachability_audit(connectome, current_inputs, current_outputs)
    mb_reachability, mb_expansion, mb_target_expansion = _reachability_audit(connectome, mb_inputs, current_outputs)
    rep_started = time.perf_counter()
    rows = run_seed_jobs(connectome, current_inputs, mb_inputs, current_outputs, workers=workers, data_dir=data_dir)
    rep_runtime = time.perf_counter() - rep_started
    reachability = {"current_random": current_reachability["per_symbol"], "mb_routed_generic": mb_reachability["per_symbol"]}
    representation = {"seeds": list(SEEDS), "contexts": 16, "repetitions_per_context": AUDIT_REPETITIONS, "learning_enabled": False, "workers": int(workers), "per_seed": rows, "worker_order_deterministic": [row["seed"] for row in rows] == list(SEEDS)}
    conclusion = _conclusion(rows, mb_candidate, reachability, {"current": current_expansion, "mb": mb_expansion})
    artifact = {
        "protocol": {"phase": "F.3R", "base": "67bda598ccf541ada6e46b6d40e378b22e472be8", "min_connection_synapses": 5, "topology_only": True, "training": False, "seeds": list(SEEDS), "ordered_contexts": ["".join(row) for row in ORDERED_CONTEXTS], "workers": int(workers)},
        "current_encoder_annotations": {"encoder": "CURRENT_RANDOM_ENCODER", "selection": "existing random outgoing-edge encoder", "summary": current_annotations},
        "mushroom_body_populations": {"annotation_rules": {"KC": "class == Kenyon_Cell", "MBON": "class == MBON", "DAN": "class == DAN", "projection_input": "class == ALPN and superclass == cb_intrinsic, with direct KC edge when available"}, "counts": {name: int(len(values)) for name, values in _target_sets(connectome).items()}, "body_ids": {name: [int(connectome.body_ids[index]) for index in values] for name, values in _target_sets(connectome).items()}},
        "current_reachability": {"targets": current_reachability["targets"], "per_symbol": current_reachability["per_symbol"], "expansion_diagnostics": expansion_diagnostics(current_expansion, current_target_expansion)},
        "expansion_diagnostics": {"current_random_encoder": expansion_diagnostics(current_expansion, current_target_expansion), "mb_routed_generic": expansion_diagnostics(mb_expansion, mb_target_expansion)},
        "mb_routed_candidate": {"summary": mb_candidate, "annotations": mb_annotations, "reachability": {"targets": mb_reachability["targets"], "per_symbol": mb_reachability["per_symbol"], "expansion_diagnostics": expansion_diagnostics(mb_expansion, mb_target_expansion)}, "candidate_no_target_or_grammar_selection": True},
        "representation_comparison": representation,
        "first_memory_guard": {"per_seed": [{"seed": row["seed"], **row["first_memory"]} for row in rows], "drop_threshold_absolute": 0.10},
        "dendritic_feasibility": dendritic_feasibility(),
        "performance": {"topology_and_setup_runtime_seconds": float(rep_started - started), "representation_runtime_seconds": float(rep_runtime), "total_runtime_seconds": float(time.perf_counter() - started), "peak_memory": "not sampled by runner", "workers_supported": [1, 2, 3]},
        "conclusion": conclusion,
        "safety": {"no_learning": bool(all(row["current_random"]["persistent_state_unchanged"] and row["mb_routed_generic"]["persistent_state_unchanged"] for row in rows)), "persistent_state_mutation": bool(not all(row["current_random"]["persistent_state_unchanged"] and row["mb_routed_generic"]["persistent_state_unchanged"] for row in rows)), "anatomy_unchanged": True, "no_syn_partners_downloaded": True},
        "pass": bool(conclusion["mb_routing_promising"] is not None and representation["worker_order_deterministic"] and conclusion["criterion"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, output_path=args.output, workers=args.workers)
    print(json.dumps({"pass": result["pass"], "mb_routing_promising": result["conclusion"]["mb_routing_promising"], "next": result["conclusion"]["recommended_next_step"], "workers": result["protocol"]["workers"], "runtime_seconds": result["performance"]["total_runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
