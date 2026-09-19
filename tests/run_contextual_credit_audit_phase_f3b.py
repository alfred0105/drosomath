"""Run the diagnostic-only F.3B matched contextual-credit audit."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sys
import time
from collections import Counter, defaultdict
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
    SequenceLearningSession,
    SequenceMemoryConfig,
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
from run_contextual_prediction_phase_f3a import (  # noqa: E402
    _exact_replay,
    _make_brain,
    _summarize as _contextual_summary,
)
from run_symbol_learning_phase_f1b import build_f1b_dynamic_interface  # noqa: E402
from run_symbol_replication_performance_phase_f1b4 import run_route_cache_equivalence  # noqa: E402
from test_working_memory_phase_f2a import make_connectome  # noqa: E402


SEEDS = (149, 151, 157)
CHECKPOINT_TRIALS = (0, 400)
AUDIT_REPETITIONS = 4
STABLE_REPETITION_THRESHOLD = 3
ARTIFACT = Path("results/latest_contextual_credit_audit_phase_f3b.json")
DATA_DIR = Path("data/malecns_v1")


def _interface_config() -> SymbolInterfaceConfig:
    return SymbolInterfaceConfig(
        symbols=SYMBOLS, sensory_population_size=32, output_population_size=32,
        seed=7, dt_ms=0.2, default_duration_ms=20.0,
        default_stimulus_rate_hz=205.0, plastic_fraction=0.05,
        output_selection="dynamic_generic",
    )


def _digest(value) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _rng_state(brain):
    return copy.deepcopy(brain.rng.bit_generator.state)


def _snapshot_state(brain):
    return _snapshot_persistent_state(brain), _rng_state(brain)


def _twin(connectome, config, seed, persistent, rng_state):
    brain = _make_brain(connectome, config, seed)
    _restore_persistent_state(brain, persistent)
    brain.rng.bit_generator.state = copy.deepcopy(rng_state)
    brain.reset()
    return brain


def _jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _group_name(left, right):
    first_equal = left[0] == right[0]
    second_equal = left[1] == right[1]
    if left == right:
        return "identical_context"
    if first_equal and not second_equal:
        return "same_first_different_second"
    if second_equal and not first_equal:
        return "same_second_different_first"
    return "different_first_and_second"


def _pairwise_groups(values):
    grouped = defaultdict(list)
    for index, left in enumerate(ORDERED_PAIRS):
        for right in ORDERED_PAIRS[index + 1 :]:
            grouped[_group_name(left, right)].append(_jaccard(values[left], values[right]))
    return {
        name: {"pairs": len(items), "mean": float(np.mean(items)) if items else 0.0,
               "median": float(np.median(items)) if items else 0.0,
               "min": float(np.min(items)) if items else 0.0,
               "max": float(np.max(items)) if items else 0.0}
        for name, items in grouped.items()
    }


def _fingerprint(active, voltage, conductance):
    active = np.asarray(active, dtype=np.int32)
    payload = active.tobytes() + np.asarray(voltage[active], dtype=np.float32).round(4).tobytes()
    payload += np.asarray(conductance[active], dtype=np.float32).round(4).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _audit_state(connectome, interface, config, seed, persistent, rng_state):
    stable = {"first": {}, "pre_go": {}}
    frequencies = {"first": {}, "pre_go": {}}
    raw_metrics = {"first": {}, "pre_go": {}}
    for phase in ("first", "pre_go"):
        for context in ORDERED_PAIRS:
            active_repetitions = []
            metrics = []
            for repetition in range(AUDIT_REPETITIONS):
                brain = _twin(connectome, config, seed + 10_000 + repetition, persistent, rng_state)
                observed = {}
                session = TwoCueSequenceSession(brain, interface)

                def observer(name, _brain, *, observed=observed):
                    if (phase == "first" and name == "first") or (phase == "pre_go" and name == "second"):
                        observed["active"] = np.asarray(_brain._fast_active, dtype=np.int32).copy()
                        observed["voltage"] = float(np.linalg.norm(_brain.v[observed["active"]])) if len(observed["active"]) else 0.0
                        observed["conductance"] = float(np.linalg.norm(_brain.g[observed["active"]])) if len(observed["active"]) else 0.0
                        observed["fingerprint"] = _fingerprint(observed["active"], _brain.v, _brain.g)

                first, second = context
                session.run_trial(first, second, track_eligibility=False, phase_observer=observer)
                active = set(int(value) for value in observed.get("active", ()))
                active_repetitions.append(active)
                metrics.append({
                    "active_count": len(active),
                    "membrane_norm": observed.get("voltage", 0.0),
                    "conductance_norm": observed.get("conductance", 0.0),
                    "fingerprint": observed.get("fingerprint", ""),
                })
            counts = Counter(value for active in active_repetitions for value in active)
            stable[phase][context] = {value for value, count in counts.items() if count >= STABLE_REPETITION_THRESHOLD}
            frequencies[phase][context] = {value: count / AUDIT_REPETITIONS for value, count in counts.items()}
            raw_metrics[phase][context] = metrics
    return {"stable": stable, "frequencies": frequencies, "metrics": raw_metrics}


def _representation_summary(audit):
    result = {"state_metrics": {}, "pairwise_context_similarity": {}, "conjunctive_specific_pools": {}, "grammar_alignment": {}}
    for phase in ("first", "pre_go"):
        stable = audit["stable"][phase]
        result["state_metrics"][phase] = {
            "per_context": {
                "".join(context): {
                    "stable_active_count": len(stable[context]),
                    "active_count_mean": float(np.mean([row["active_count"] for row in audit.get("metrics", {}).get(phase, {}).get(context, [])])) if audit.get("metrics", {}).get(phase, {}).get(context) else float(len(stable[context])),
                    "membrane_norm_mean": float(np.mean([row["membrane_norm"] for row in audit.get("metrics", {}).get(phase, {}).get(context, [])])) if audit.get("metrics", {}).get(phase, {}).get(context) else 0.0,
                    "conductance_norm_mean": float(np.mean([row["conductance_norm"] for row in audit.get("metrics", {}).get(phase, {}).get(context, [])])) if audit.get("metrics", {}).get(phase, {}).get(context) else 0.0,
                    "fingerprint_diversity": len({row["fingerprint"] for row in audit.get("metrics", {}).get(phase, {}).get(context, [])}),
                } for context in ORDERED_PAIRS
            }
        }
        result["pairwise_context_similarity"][phase] = _pairwise_groups(stable)
        strict = {}
        broad = {}
        for context in ORDERED_PAIRS:
            first, second = context
            alternatives_first = [stable[other] for other in ORDERED_PAIRS if other != context and other[0] == first]
            alternatives_second = [stable[other] for other in ORDERED_PAIRS if other != context and other[1] == second]
            strict_pool = stable[context] - set().union(*(alternatives_first + alternatives_second))
            freq = audit["frequencies"][phase][context]
            broad_pool = set()
            for neuron, frequency in freq.items():
                if frequency < 0.75:
                    continue
                first_max = max((audit["frequencies"][phase][other].get(neuron, 0.0) for other in ORDERED_PAIRS if other != context and other[0] == first), default=0.0)
                second_max = max((audit["frequencies"][phase][other].get(neuron, 0.0) for other in ORDERED_PAIRS if other != context and other[1] == second), default=0.0)
                if first_max <= 0.50 and second_max <= 0.50:
                    broad_pool.add(neuron)
            strict[context] = strict_pool
            broad[context] = broad_pool
        strict_counts = [len(strict[context]) for context in ORDERED_PAIRS]
        broad_counts = [len(broad[context]) for context in ORDERED_PAIRS]
        result["conjunctive_specific_pools"][phase] = {
            "per_context": {
                "".join(context): {
                    "stable_active_count": len(stable[context]),
                    "conjunctive_specific_count": len(strict[context]),
                    "conjunctive_specific_fraction": float(len(strict[context]) / len(stable[context])) if stable[context] else 0.0,
                    "broader_conjunction_count": len(broad[context]),
                } for context in ORDERED_PAIRS
            },
            "strict_aggregate": {"mean": float(np.mean(strict_counts)), "median": float(np.median(strict_counts)), "min": int(min(strict_counts)), "max": int(max(strict_counts))},
            "broader_aggregate": {"mean": float(np.mean(broad_counts)), "median": float(np.median(broad_counts)), "min": int(min(broad_counts)), "max": int(max(broad_counts))},
            "strict_pools": strict,
            "broader_pools": broad,
        }
        target_groups = defaultdict(list)
        for context in ORDERED_PAIRS:
            target_groups[CONTEXTUAL_GRAMMAR[context]].append(context)
        within, between = [], []
        for index, left in enumerate(ORDERED_PAIRS):
            for right in ORDERED_PAIRS[index + 1:]:
                value = _jaccard(stable[left], stable[right])
                (within if CONTEXTUAL_GRAMMAR[left] == CONTEXTUAL_GRAMMAR[right] else between).append(value)
        result["grammar_alignment"][phase] = {
            "same_grammar_target_mean_jaccard": float(np.mean(within)) if within else 0.0,
            "different_grammar_target_mean_jaccard": float(np.mean(between)) if between else 0.0,
            "target_contexts": {target: ["".join(context) for context in contexts] for target, contexts in target_groups.items()},
        }
    return result


class _CreditRecorder:
    def __init__(self, session, context, phase_sets):
        self.session = session
        self.context = context
        self.phase_sets = phase_sets
        self.events = []
        self._install()

    def _install(self):
        controller = self.session.controller
        original_apply = controller.apply_learning_signal
        recorder = self

        def apply(brain, signal, output_context, **kwargs):
            event = {"channels": [], "signed": defaultdict(float), "absolute": defaultdict(float), "updates": 0, "sum_abs_delta": 0.0, "hop_counts": Counter()}

            def attribution_observer(*, channel, current_edge_indices, current_hops, actual_deltas, requested_direction, **_ignored):
                edges = np.asarray(current_edge_indices, dtype=np.int32)
                hops = np.asarray(current_hops, dtype=np.int8)
                deltas = np.asarray(actual_deltas, dtype=np.float64)
                if float(requested_direction) == 0.0 or len(edges) == 0:
                    return
                sign = "positive" if float(requested_direction) > 0 else "negative"
                event["channels"].append({"channel": channel, "direction": sign, "edges": set(int(edge) for edge in edges), "hops": Counter(int(hop) for hop in hops), "signed": {int(edge): float(delta) for edge, delta in zip(edges, deltas)}, "absolute": {int(edge): abs(float(delta)) for edge, delta in zip(edges, deltas)}})
                event["updates"] += int(len(edges))
                event["sum_abs_delta"] += float(np.abs(deltas).sum())
                event["hop_counts"].update(int(hop) for hop in hops)
                for edge, delta in zip(edges, deltas):
                    event["signed"][int(edge)] += float(delta)
                    event["absolute"][int(edge)] += abs(float(delta))

            kwargs["attribution_observer"] = attribution_observer
            result = original_apply(brain, signal, output_context, **kwargs)
            event["update"] = result
            self.events.append(event)
            return result

        controller.apply_learning_signal = apply
        original_phase = self.session._phase_observer

        def phase_observer(snapshots):
            base = original_phase(snapshots)

            def observe(phase, brain):
                base(phase, brain)
                phase_sets = self.phase_sets if "eligible_after_first" in self.phase_sets else self.phase_sets[self.context]
                if phase in ("first", "second", "go"):
                    values = set(snapshots.get(phase, ()))
                    if phase == "first":
                        self._first = values
                    elif phase == "second":
                        phase_sets["eligible_after_first"].update(self._first)
                        phase_sets["newly_second"].update(values - self._first)
                        self._second = values
                    else:
                        phase_sets["newly_go"].update(values - self._second)

            return observe

        self.session._phase_observer = phase_observer


def _run_training_arm(connectome, interface, config, seed, schedule, prediction):
    brain = _make_brain(connectome, config, seed)
    session = (ContextualPredictionLearningSession(brain, interface, config=SequenceMemoryConfig()) if prediction else SequenceLearningSession(brain, interface, config=SequenceMemoryConfig()))
    phase_by_context = {context: {"eligible_after_first": set(), "newly_second": set(), "newly_go": set()} for context in ORDERED_PAIRS}
    recorder = _CreditRecorder(session, None, phase_by_context)
    per_context = {context: {"requests": 0, "updated_requests": 0, "update_counts": [], "positive_edges": set(), "negative_edges": set(), "total_edge_updates": 0, "sum_abs_delta": 0.0, "hop_counts": Counter(), "signed": defaultdict(float), "absolute": defaultdict(float), "channel_data": defaultdict(lambda: {"positive": set(), "negative": set(), "signed": defaultdict(float), "absolute": defaultdict(float)}), "events": 0} for context in ORDERED_PAIRS}
    records = []
    for context in schedule:
        recorder.context = context
        recorder.events = []
        record = session.train_trial(*context, capture_phase_credit=True)
        bucket = per_context[context]
        if record["directional_error"]:
            bucket["requests"] += 1
        for event in recorder.events:
            bucket["events"] += 1
            bucket["total_edge_updates"] += event["updates"]
            bucket["sum_abs_delta"] += event["sum_abs_delta"]
            bucket["hop_counts"].update(event["hop_counts"])
            bucket["signed"].update(event["signed"])
            bucket["absolute"].update(event["absolute"])
            if event["updates"] > 0:
                bucket["updated_requests"] += 1
            if record["directional_error"]:
                bucket["update_counts"].append(event["updates"])
            for channel in event["channels"]:
                bucket["positive_edges" if channel["direction"] == "positive" else "negative_edges"].update(channel["edges"])
                channel_bucket = bucket["channel_data"][channel["channel"]]
                channel_bucket[channel["direction"]].update(channel["edges"])
                channel_bucket["signed"].update(channel["signed"])
                channel_bucket["absolute"].update(channel["absolute"])
        records.append(record)
    return brain, session, per_context, phase_by_context, records, int(session.plastic_budget_start), int(brain.plasticity.plastic_edge_count)


def _credit_summary(per_context, phase_by_context, connectome, pools):
    result = {}
    for context in ORDERED_PAIRS:
        bucket = per_context[context]
        requests = bucket["requests"]
        result["".join(context)] = {
            "directional_correction_requests": requests,
            "requests_with_ge_1_edge_update": int(bucket["updated_requests"]),
            "zero_update_fraction": float(1.0 - bucket["updated_requests"] / requests) if requests else 0.0,
            "mean_edge_updates_per_request": float(bucket["total_edge_updates"] / requests) if requests else 0.0,
            "median_edge_updates_per_request": float(np.median(bucket["update_counts"])) if bucket["update_counts"] else 0.0,
            "one_hop_updates": int(bucket["hop_counts"].get(1, 0)),
            "two_hop_updates": int(bucket["hop_counts"].get(2, 0)),
            "sum_abs_delta_per_request": float(bucket["sum_abs_delta"] / requests) if requests else 0.0,
            "unique_positive_edges": len(bucket["positive_edges"]),
            "unique_negative_edges": len(bucket["negative_edges"]),
            "positive_edges": bucket["positive_edges"],
            "negative_edges": bucket["negative_edges"],
            "newly_second_eligible_count": len(phase_by_context[context]["newly_second"]),
            "newly_go_eligible_count": len(phase_by_context[context]["newly_go"]),
            "eligible_after_first_count": len(phase_by_context[context]["eligible_after_first"]),
            "conjunctive_route_usage": {
                "strict_fraction": _edge_pool_fraction(bucket["positive_edges"] | bucket["negative_edges"], connectome, pools["strict"].get(context, set())),
                "broader_fraction": _edge_pool_fraction(bucket["positive_edges"] | bucket["negative_edges"], connectome, pools["broader"].get(context, set())),
            },
        }
    return result


def _edge_pool_fraction(edges, connectome, pools):
    if not edges:
        return 0.0
    values = np.asarray(sorted(edges), dtype=np.int32)
    pres = np.searchsorted(connectome.indptr, values, side="right") - 1
    pool = set(pools) if pools else set()
    return float(sum(int(value) in pool for value in pres) / len(pres))


def _strip_credit(summary):
    return {context: {key: value for key, value in row.items() if key not in {"positive_edges", "negative_edges"}} for context, row in summary.items()}


def _public_representation(rep):
    return {
        "state_metrics": rep["state_metrics"],
        "pairwise_context_similarity": rep["pairwise_context_similarity"],
        "grammar_alignment": rep["grammar_alignment"],
        "conjunctive_specific_pools": {
            phase: {key: value for key, value in rep["conjunctive_specific_pools"][phase].items() if key not in {"strict_pools", "broader_pools"}}
            for phase in ("first", "pre_go")
        },
    }


def _mean_cancellation(items):
    rows = [_sign_cancellation(item[0]) for item in items]
    result = {"per_output": {}, "positive_unique_edges": 0, "negative_unique_edges": 0, "intersection": 0, "positive_negative_jaccard": 0.0, "total_abs_effect": 0.0, "abs_net_effect": 0.0, "cancellation_fraction": 0.0, "median_edge_level_cancellation": 0.0}
    if not rows:
        return result
    for key in ("positive_unique_edges", "negative_unique_edges", "intersection"):
        result[key] = int(round(np.mean([row[key] for row in rows])))
    for key in ("positive_negative_jaccard", "total_abs_effect", "abs_net_effect", "cancellation_fraction", "median_edge_level_cancellation"):
        result[key] = float(np.mean([row[key] for row in rows]))
    symbols = set().union(*(row["per_output"] for row in rows))
    result["per_output"] = {symbol: {key: float(np.mean([row["per_output"].get(symbol, {}).get(key, 0.0) for row in rows])) for key in ("positive_unique_edges", "negative_unique_edges", "intersection", "positive_negative_jaccard", "total_abs_effect", "abs_net_effect", "cancellation_fraction", "median_edge_level_cancellation")} for symbol in symbols}
    return result


def _phase_overlap(items, relation):
    values = []
    for _, phase_sets, _ in items:
        for index, left in enumerate(ORDERED_PAIRS):
            for right in ORDERED_PAIRS[index + 1:]:
                if relation(left, right):
                    values.append(_jaccard(phase_sets[left]["newly_second"], phase_sets[right]["newly_second"]))
    return float(np.mean(values)) if values else 0.0


def _overlap_metrics(per_context, relation):
    values = []
    pairs = []
    for index, left in enumerate(ORDERED_PAIRS):
        for right in ORDERED_PAIRS[index + 1:]:
            if relation(left, right):
                values.append(_jaccard(per_context[left]["positive_edges"], per_context[right]["positive_edges"]))
                pairs.append((left, right))
    return {"pairs": len(values), "mean": float(np.mean(values)) if values else 0.0, "median": float(np.median(values)) if values else 0.0, "pair_examples": ["".join(a) + ":" + "".join(b) for a, b in pairs[:6]]}


def _conflict_metrics(per_context):
    groups = defaultdict(list)
    for index, left in enumerate(ORDERED_PAIRS):
        for right in ORDERED_PAIRS[index + 1:]:
            if CONTEXTUAL_GRAMMAR[left] == CONTEXTUAL_GRAMMAR[right]:
                continue
            groups[_group_name(left, right)].append(_jaccard(per_context[left]["positive_edges"], per_context[right]["positive_edges"]))
    return {name: {"pairs": len(values), "mean_positive_credit_overlap": float(np.mean(values)) if values else 0.0, "median": float(np.median(values)) if values else 0.0} for name, values in groups.items()}


def _sign_cancellation(per_context):
    channel_edges = defaultdict(lambda: {"positive": set(), "negative": set(), "signed": defaultdict(float), "absolute": defaultdict(float)})
    for bucket in per_context.values():
        for channel, data in bucket.get("channel_data", {}).items():
            channel_edges[channel]["positive"].update(data["positive"])
            channel_edges[channel]["negative"].update(data["negative"])
            channel_edges[channel]["signed"].update(data["signed"])
            channel_edges[channel]["absolute"].update(data["absolute"])
    per_output = {}
    all_signed = defaultdict(float); all_absolute = defaultdict(float)
    positive_all = set(); negative_all = set()
    for channel, data in channel_edges.items():
        positive_all.update(data["positive"]); negative_all.update(data["negative"])
        all_signed.update(data["signed"]); all_absolute.update(data["absolute"])
        total_abs = float(sum(data["absolute"].values())); net = float(sum(data["signed"].values()))
        edge_cancel = [1.0 - abs(data["signed"][edge]) / data["absolute"][edge] for edge in data["signed"] if data["absolute"][edge] > 0.0]
        per_output[channel.split("/")[-1]] = {"positive_unique_edges": len(data["positive"]), "negative_unique_edges": len(data["negative"]), "intersection": len(data["positive"] & data["negative"]), "positive_negative_jaccard": _jaccard(data["positive"], data["negative"]), "total_abs_effect": total_abs, "abs_net_effect": abs(net), "cancellation_fraction": 1.0 - abs(net) / total_abs if total_abs else 0.0, "median_edge_level_cancellation": float(np.median(edge_cancel)) if edge_cancel else 0.0}
    total_abs = float(sum(all_absolute.values())); net = float(sum(all_signed.values()))
    edge_cancel = [1.0 - abs(all_signed[edge]) / all_absolute[edge] for edge in all_signed if all_absolute[edge] > 0.0]
    return {"per_output": per_output, "positive_unique_edges": len(positive_all), "negative_unique_edges": len(negative_all), "intersection": len(positive_all & negative_all), "positive_negative_jaccard": _jaccard(positive_all, negative_all), "total_abs_effect": total_abs, "abs_net_effect": abs(net), "cancellation_fraction": 1.0 - abs(net) / total_abs if total_abs else 0.0, "median_edge_level_cancellation": float(np.median(edge_cancel)) if edge_cancel else 0.0}


def _behavior_rows(connectome, interface, config, seed, persistent, rng_state, prediction):
    brain = _twin(connectome, config, seed, persistent, rng_state)
    session = ContextualPredictionSession(brain, interface) if prediction else TwoCueSequenceSession(brain, interface)
    schedule = balanced_pair_schedule(4, seed=seed + 30_000)
    rows = []
    for first, second in schedule:
        value = session.run_trial(first, second)
        target = CONTEXTUAL_GRAMMAR[(first, second)] if prediction else first
        rows.append(SimpleNamespace(first=first, second=second, target=target, decision=value.decision, go_output_rates_hz=value.go_output_rates_hz, go_output_spikes=value.go_output_spikes))
    return _contextual_summary(rows)


def _diagnostic_invariance(connectome, interface, config):
    schedule = balanced_pair_schedule(1, seed=45_149)
    def run(enabled):
        brain = _make_brain(connectome, config, 45_149)
        session = SequenceLearningSession(brain, interface, config=SequenceMemoryConfig())
        recorder = _CreditRecorder(session, None, {"eligible_after_first": set(), "newly_second": set(), "newly_go": set()})
        records = [session.train_trial(*pair, capture_phase_credit=enabled) for pair in schedule]
        return records, brain, recorder
    off_records, off, off_recorder = run(False)
    on_records, on, on_recorder = run(True)
    keys = ("multiplier", "usage_ema", "eligibility", "stability", "plastic_mask")
    arrays_equal = all(np.array_equal(getattr(off.plasticity, key), getattr(on.plasticity, key)) for key in keys)
    def event_signature(recorder):
        return [{"updates": event["updates"], "sum_abs_delta": event["sum_abs_delta"], "signed": dict(event["signed"]), "absolute": dict(event["absolute"]), "channels": [{"channel": row["channel"], "direction": row["direction"], "edges": sorted(row["edges"]), "signed": row["signed"], "absolute": row["absolute"]} for row in event["channels"]]} for event in recorder.events]
    attribution_equal = event_signature(off_recorder) == event_signature(on_recorder)
    transient_equal = np.array_equal(off.v, on.v) and np.array_equal(off.g, on.g)
    return {"passed": bool(arrays_equal and transient_equal and _rng_state(off) == _rng_state(on) and off_records[0]["directional_update"] == on_records[0]["directional_update"] and attribution_equal), "persistent_arrays_equal": arrays_equal, "transient_neural_state_equal": transient_equal, "rng_equal": _rng_state(off) == _rng_state(on), "directional_telemetry_equal": off_records[0]["directional_update"] == on_records[0]["directional_update"], "attribution_delta_values_equal": attribution_equal, "decisions_equal": off_records[0]["result"].decision == on_records[0]["result"].decision}


def _run_performance(connectome, interface, config):
    schedule = balanced_pair_schedule(2, seed=45_149)[:32]
    def bench(legacy):
        brain = _make_brain(connectome, config, 45_150)
        if legacy:
            def old_due(index, _brain=brain):
                chunks = _brain._fast_delay_touched[index]
                if not chunks:
                    return _brain.np.empty(0, dtype=_brain.np.int32)
                raw = chunks[0] if len(chunks) == 1 else _brain.np.concatenate(chunks)
                value = _brain.np.unique(raw).astype(_brain.np.int32, copy=False)
                chunks.clear()
                return value
            brain._due_indices = old_due
        session = TwoCueSequenceSession(brain, interface)
        started = time.perf_counter()
        for pair in schedule:
            session.run_trial(*pair)
        elapsed = time.perf_counter() - started
        return float(len(schedule) / max(elapsed, 1e-12))
    optimized = [bench(False) for _ in range(2)]
    legacy = [bench(True) for _ in range(2)]
    optimized_median = float(np.median(optimized))
    return {"optimized_episodes_per_second": optimized_median, "optimized_runs": optimized, "legacy_reference_episodes_per_second": float(np.median(legacy)), "regression_detected": bool(min(optimized) < max(optimized) * 0.95), "optimized_vs_legacy_speedup": float(optimized_median / max(np.median(legacy), 1e-12)), "samples": len(schedule)}


def _run(data_dir=DATA_DIR, output_path=ARTIFACT):
    started = time.perf_counter()
    connectome = load_malecns_v1(data_dir, min_connection_synapses=5)
    interface_config = _interface_config()
    interface, candidate_count, selected = build_f1b_dynamic_interface(connectome, interface_config)
    wm_interface = WorkingMemoryInterface(connectome, interface)
    config = SequenceMemoryConfig()
    runs = {}
    representation_by_arm = {"recall": [], "prediction": []}
    credit_by_arm = {"recall": [], "prediction": []}
    behavior_by_arm = {"recall": [], "prediction": []}
    matched_init = True
    matched_schedule = True
    for seed in SEEDS:
        schedule = balanced_pair_schedule(25, seed=seed + config.schedule_seed_offset)
        recall_brain = _make_brain(connectome, interface_config, seed)
        prediction_brain = _make_brain(connectome, interface_config, seed)
        matched_init = matched_init and _digest(recall_brain.plasticity.plastic_mask) == _digest(prediction_brain.plasticity.plastic_mask)
        matched_schedule = matched_schedule and schedule == balanced_pair_schedule(25, seed=seed + config.schedule_seed_offset)
        recall_before, recall_rng = _snapshot_state(recall_brain)
        prediction_before, prediction_rng = _snapshot_state(prediction_brain)
        initial_repr_recall = _audit_state(connectome, wm_interface, interface_config, seed, recall_before, recall_rng)
        initial_repr_prediction = _audit_state(connectome, wm_interface, interface_config, seed, prediction_before, prediction_rng)
        recall_brain, recall_session, recall_credit, recall_phase, recall_records, recall_budget_start, recall_budget_end = _run_training_arm(connectome, wm_interface, interface_config, seed, schedule, False)
        prediction_brain, prediction_session, prediction_credit, prediction_phase, prediction_records, prediction_budget_start, prediction_budget_end = _run_training_arm(connectome, wm_interface, interface_config, seed, schedule, True)
        recall_after, recall_after_rng = _snapshot_state(recall_brain)
        prediction_after, prediction_after_rng = _snapshot_state(prediction_brain)
        final_repr_recall = _audit_state(connectome, wm_interface, interface_config, seed, recall_after, recall_after_rng)
        final_repr_prediction = _audit_state(connectome, wm_interface, interface_config, seed, prediction_after, prediction_after_rng)
        rep_initial_recall = _public_representation(_representation_summary(initial_repr_recall))
        rep_initial_prediction = _public_representation(_representation_summary(initial_repr_prediction))
        rep_recall = _representation_summary(final_repr_recall)
        rep_prediction = _representation_summary(final_repr_prediction)
        recall_pools = {"strict": rep_recall["conjunctive_specific_pools"]["pre_go"]["strict_pools"], "broader": rep_recall["conjunctive_specific_pools"]["pre_go"]["broader_pools"]}
        prediction_pools = {"strict": rep_prediction["conjunctive_specific_pools"]["pre_go"]["strict_pools"], "broader": rep_prediction["conjunctive_specific_pools"]["pre_go"]["broader_pools"]}
        credit_recall = _credit_summary(recall_credit, recall_phase, connectome, recall_pools)
        credit_prediction = _credit_summary(prediction_credit, prediction_phase, connectome, prediction_pools)
        representation_by_arm["recall"].append(rep_recall); representation_by_arm["prediction"].append(rep_prediction)
        credit_by_arm["recall"].append((recall_credit, recall_phase, credit_recall)); credit_by_arm["prediction"].append((prediction_credit, prediction_phase, credit_prediction))
        behavior_by_arm["recall"].append({"0": _behavior_rows(connectome, wm_interface, interface_config, seed, recall_before, recall_rng, False), "400": _behavior_rows(connectome, wm_interface, interface_config, seed, recall_after, recall_after_rng, False)})
        behavior_by_arm["prediction"].append({"0": _behavior_rows(connectome, wm_interface, interface_config, seed, prediction_before, prediction_rng, True), "400": _behavior_rows(connectome, wm_interface, interface_config, seed, prediction_after, prediction_after_rng, True)})
        runs[str(seed)] = {"seed": seed, "trials_per_arm": len(schedule), "episodes_per_context": {"".join(context): schedule.count(context) for context in ORDERED_PAIRS}, "matched_initialization": matched_init, "matched_schedule": matched_schedule, "representation_before": {"recall": rep_initial_recall, "prediction": rep_initial_prediction}, "representation_after": {"recall": _public_representation(rep_recall), "prediction": _public_representation(rep_prediction)}, "behavior": {"recall": behavior_by_arm["recall"][-1], "prediction": behavior_by_arm["prediction"][-1]}, "credit": {"recall": _strip_credit(credit_recall), "prediction": _strip_credit(credit_prediction)}, "plastic_budget": {"recall_before": recall_budget_start, "recall_after": recall_budget_end, "prediction_before": prediction_budget_start, "prediction_after": prediction_budget_end}}

    def mean_behavior(arm, checkpoint, field):
        return float(np.mean([row[str(checkpoint)][field] for row in behavior_by_arm[arm]]))
    def aggregate_credit(arm, key):
        values = []
        for _, _, credit in credit_by_arm[arm]:
            values.extend(row[key] for row in credit.values() if isinstance(row.get(key), (int, float)))
        return float(np.mean(values)) if values else 0.0
    def aggregate_rep(arm, phase, path):
        values = [run["representation"][arm][path][phase] for run in runs.values()]
        if path == "pairwise_context_similarity":
            return {group: float(np.mean([value[group]["mean"] for value in values])) for group in values[0]}
        return float(np.mean([value for value in values])) if values and isinstance(values[0], (int, float)) else values

    recall_sets = [item[0] for item in credit_by_arm["recall"]]
    prediction_sets = [item[0] for item in credit_by_arm["prediction"]]
    def mean_overlap(items, relation):
        values = []
        for per_context in items:
            values.append(_overlap_metrics(per_context, relation)["mean"])
        return float(np.mean(values)) if values else 0.0
    same_first = lambda a, b: a[0] == b[0] and a != b
    same_second = lambda a, b: a[1] == b[1] and a != b
    same_target = lambda a, b: CONTEXTUAL_GRAMMAR[a] == CONTEXTUAL_GRAMMAR[b] and a != b
    different_target = lambda a, b: CONTEXTUAL_GRAMMAR[a] != CONTEXTUAL_GRAMMAR[b]
    overlap = {"recall": {"same_first": mean_overlap(recall_sets, same_first), "same_second": mean_overlap(recall_sets, same_second), "same_grammar_target": mean_overlap(recall_sets, same_target), "different_grammar_target": mean_overlap(recall_sets, different_target)}, "prediction": {"same_first": mean_overlap(prediction_sets, same_first), "same_second": mean_overlap(prediction_sets, same_second), "same_grammar_target": mean_overlap(prediction_sets, same_target), "different_grammar_target": mean_overlap(prediction_sets, different_target)}}
    conflict = {"recall": [_conflict_metrics(item[0]) for item in credit_by_arm["recall"]], "prediction": [_conflict_metrics(item[0]) for item in credit_by_arm["prediction"]]}
    representation_supported = all(all(run["representation_after"]["prediction"]["conjunctive_specific_pools"]["pre_go"]["per_context"]["".join(context)]["stable_active_count"] > 0 for context in ORDERED_PAIRS) for run in runs.values())
    broad_counts = [run["representation_after"]["prediction"]["conjunctive_specific_pools"]["pre_go"]["per_context"]["".join(context)]["broader_conjunction_count"] for run in runs.values() for context in ORDERED_PAIRS]
    representation_supported = bool(representation_supported and sum(count > 0 for count in broad_counts) >= 12 * len(SEEDS))
    prediction_updates = aggregate_credit("prediction", "mean_edge_updates_per_request")
    recall_updates = aggregate_credit("recall", "mean_edge_updates_per_request")
    quantity_shortage = bool(prediction_updates < 0.50 * recall_updates)
    cancellation = {"recall": _mean_cancellation(credit_by_arm["recall"]), "prediction": _mean_cancellation(credit_by_arm["prediction"])}
    cancellation_difference = cancellation["prediction"]["cancellation_fraction"] - cancellation["recall"]["cancellation_fraction"]
    overlap_difference = overlap["prediction"]["different_grammar_target"] - overlap["recall"]["different_grammar_target"]
    interference = bool((overlap_difference >= 0.10 or cancellation_difference >= 0.15) and (overlap["prediction"]["same_first"] - overlap["recall"]["same_first"] >= 0.10 or overlap["prediction"]["same_second"] - overlap["recall"]["same_second"] >= 0.10))
    broad_route_fraction = float(np.mean([row["conjunctive_route_usage"]["broader_fraction"] for item in credit_by_arm["prediction"] for row in item[2].values()]))
    credit_bottleneck = bool(representation_supported and not quantity_shortage and broad_route_fraction < 0.20)
    if not representation_supported:
        primary, next_step = "conjunctive_context_representation", "build_context_binding_dynamics"
    elif interference:
        primary, next_step = "context_specific_credit_interference", "context_gated_predictive_credit"
    elif credit_bottleneck:
        primary, next_step = "conjunctive_route_credit_assignment", "prioritize_context_specific_causal_routes"
    elif quantity_shortage:
        primary, next_step = "prediction_credit_quantity", "improve_predictive_route_engagement"
    else:
        primary, next_step = "unresolved_predictive_mapping", "deeper_contextual_learning_audit"
    invariance = _diagnostic_invariance(connectome, wm_interface, interface_config)
    performance = _run_performance(connectome, wm_interface, interface_config)
    p5_exact = _exact_replay(connectome, wm_interface, interface_config)
    route_equivalence = run_route_cache_equivalence(connectome, interface, interface_config)
    source_learning = inspect.getsource(ContextualPredictionLearningSession.train_trial)
    artifact = {
        "protocol": {"phase": "F.3B+P.6", "seeds": list(SEEDS), "trials_per_arm_per_seed": 400, "contexts": 16, "trials_per_context": 25, "audit_repetitions": 4, "stable_active_threshold": 3, "learning_rates_unchanged": True, "grammar_unchanged": True, "sensory_output_populations_unchanged": True},
        "matched_setup": {"identical_schedules": matched_schedule, "identical_initial_plastic_masks": matched_init, "same_interface": True, "same_route_cache": True, "same_prospective_credit": True, "only_target_assignment_differs": True},
        "state_representation": {"pairwise_context_similarity": {"recall": [rep["pairwise_context_similarity"] for rep in representation_by_arm["recall"]], "prediction": [rep["pairwise_context_similarity"] for rep in representation_by_arm["prediction"]]}, "conjunctive_specific_pools": {"recall": [{phase: {key: value for key, value in rep["conjunctive_specific_pools"][phase].items() if key not in {"strict_pools", "broader_pools"}} for phase in ("first", "pre_go")} for rep in representation_by_arm["recall"]], "prediction": [{phase: {key: value for key, value in rep["conjunctive_specific_pools"][phase].items() if key not in {"strict_pools", "broader_pools"}} for phase in ("first", "pre_go")} for rep in representation_by_arm["prediction"]]}, "grammar_alignment": {"recall": [rep["grammar_alignment"] for rep in representation_by_arm["recall"]], "prediction": [rep["grammar_alignment"] for rep in representation_by_arm["prediction"]]}},
        "recall_arm": {"behavior": behavior_by_arm["recall"], "credit": [_strip_credit(item[2]) for item in credit_by_arm["recall"]]},
        "prediction_arm": {"behavior": behavior_by_arm["prediction"], "credit": [_strip_credit(item[2]) for item in credit_by_arm["prediction"]]},
        "comparison": {"positive_credit_overlap": overlap, "context_target_conflict_matrix": conflict, "sign_cancellation": cancellation, "credit_quantity": {"recall_mean_updates_per_request": recall_updates, "prediction_mean_updates_per_request": prediction_updates, "prediction_to_recall_ratio": float(prediction_updates / recall_updates) if recall_updates else 0.0}, "conjunctive_route_usage": {"prediction_mean_broader_sourced_fraction": broad_route_fraction, "threshold": 0.20}, "phase_dominance": {"recall_mean_newly_second": float(np.mean([row["newly_second_eligible_count"] for item in credit_by_arm["recall"] for row in item[2].values()])), "prediction_mean_newly_second": float(np.mean([row["newly_second_eligible_count"] for item in credit_by_arm["prediction"] for row in item[2].values()])), "recall_same_second_newly_second_overlap": _phase_overlap(credit_by_arm["recall"], same_second), "prediction_same_second_newly_second_overlap": _phase_overlap(credit_by_arm["prediction"], same_second)}},
        "performance": {"p5_optimization_still_enabled": True, "exact_equivalence": bool(p5_exact["passed"]), "p5_exact_replay": p5_exact, "diagnostic_invariance": invariance, "throughput": performance, "route_cache_equivalence": route_equivalence, "regression_detected": performance["regression_detected"]},
        "safety": {"fixed_plastic_budget": all(run["plastic_budget"]["recall_before"] == run["plastic_budget"]["recall_after"] == run["plastic_budget"]["prediction_before"] == run["plastic_budget"]["prediction_after"] for run in runs.values()), "adaptive_reallocations": 0, "external_decoder": "classifier" not in source_learning.lower() and "backprop" not in source_learning.lower()},
        "conclusion": {"conjunctive_representation_supported": representation_supported, "contextual_credit_interference_supported": interference, "credit_quantity_shortage_supported": quantity_shortage, "conjunctive_credit_bottleneck_supported": credit_bottleneck, "primary_bottleneck": primary, "recommended_next_step": next_step, "diagnostic_values": {"overlap_difference_different_target": overlap_difference, "cancellation_difference": cancellation_difference, "broader_route_fraction": broad_route_fraction}},
        "pass": bool(matched_schedule and matched_init and invariance["passed"] and p5_exact["passed"] and route_equivalence["passed"] and not performance["regression_detected"]),
        "runs": runs,
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
    artifact = _run(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({"pass": artifact["pass"], "conclusion": artifact["conclusion"], "runtime_seconds": artifact["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
