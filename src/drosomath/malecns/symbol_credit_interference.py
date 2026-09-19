"""Diagnostic-only aggregation for context-specific generic credit reuse."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from statistics import median
from typing import Mapping

import numpy as np

from .symbol_interface import SYMBOLS


CHANNELS = tuple(f"symbol/{symbol}" for symbol in SYMBOLS)
SIGNS = ("positive", "negative")
HOPS = (1, 2)
ROUTE_HEALTH_FIELDS = (
    "active_plastic_candidate_edges",
    "useful_plastic_edges",
    "plastic_structural_opportunity",
    "realized_plastic_capacity",
    "plastic_engagement_efficiency",
    "useful_frozen_edges",
)


def jaccard(left, right) -> float:
    """Return a safe Jaccard score, including the empty/empty case."""
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def _new_channel_state():
    return {
        "request_counts": {sign: 0 for sign in SIGNS},
        "request_yields": {sign: [] for sign in SIGNS},
        "edges": {sign: set() for sign in SIGNS},
        "edges_by_hop": {sign: {hop: set() for hop in HOPS} for sign in SIGNS},
        "presynaptic": {sign: set() for sign in SIGNS},
        "contexts": defaultdict(set),
        "edge_effects": defaultdict(lambda: {sign: 0.0 for sign in SIGNS}),
        "hop_effects": defaultdict(lambda: {sign: 0.0 for sign in SIGNS}),
        "route_health": defaultdict(list),
    }


def _merge_defaultdict_sets(target, source):
    for key, values in source.items():
        target[key].update(values)


def _merge_channel(target, source):
    for sign in SIGNS:
        target["request_counts"][sign] += source["request_counts"][sign]
        target["request_yields"][sign].extend(source["request_yields"][sign])
        target["edges"][sign].update(source["edges"][sign])
        target["presynaptic"][sign].update(source["presynaptic"][sign])
        for hop in HOPS:
            target["edges_by_hop"][sign][hop].update(source["edges_by_hop"][sign][hop])
    _merge_defaultdict_sets(target["contexts"], source["contexts"])
    for edge, values in source["edge_effects"].items():
        for sign in SIGNS:
            target["edge_effects"][edge][sign] += values[sign]
    for edge_hop, values in source["hop_effects"].items():
        for sign in SIGNS:
            target["hop_effects"][edge_hop][sign] += values[sign]
    for context, rows in source["route_health"].items():
        target["route_health"][context].extend(rows)


def _safe_cancellation(effect_rows, *, epsilon=1e-12):
    overlap = []
    total_abs = 0.0
    net = 0.0
    for values in effect_rows.values():
        positive = float(values["positive"])
        negative = float(values["negative"])
        if abs(positive) <= epsilon or abs(negative) <= epsilon:
            continue
        overlap.append(1.0 - min(1.0, abs(positive + negative) / max(epsilon, abs(positive) + abs(negative))))
        total_abs += abs(positive) + abs(negative)
        net += positive + negative
    return {
        "overlap_edge_total_abs_delta": float(total_abs),
        "overlap_edge_abs_net_delta": float(abs(net)),
        "cancellation_fraction": float(1.0 - abs(net) / total_abs) if total_abs > epsilon else 0.0,
        "median_edge_level_cancellation_fraction": float(median(overlap)) if overlap else 0.0,
        "overlap_edge_count": int(len(overlap)),
    }


def _yield_summary(values):
    values = [int(value) for value in values]
    requests = len(values)
    nonzero = sum(value > 0 for value in values)
    return {
        "direction_requests": int(requests),
        "requests_with_at_least_one_edge_update": int(nonzero),
        "total_edge_updates": int(sum(values)),
        "mean_edge_updates_per_request": float(np.mean(values)) if values else 0.0,
        "median_edge_updates_per_request": float(median(values)) if values else 0.0,
        "zero_update_count": int(requests - nonzero),
        "zero_update_fraction": float((requests - nonzero) / requests) if requests else 0.0,
    }


def _context_overlap(contexts):
    result = {}
    for left, right in combinations(sorted(contexts), 2):
        result[f"{left}|{right}"] = jaccard(contexts[left], contexts[right])
    return result


class SymbolCreditInterferenceAudit:
    """Collect sparse credit reuse without changing any learning state."""

    def __init__(self, connectome, *, max_route_health_requests: int = 8):
        self.connectome = connectome
        self.max_route_health_requests = int(max_route_health_requests)
        self.channels = {channel: _new_channel_state() for channel in CHANNELS}

    def observe_update(
        self,
        *,
        target: str,
        decision: str,
        channel: str,
        edge_indices,
        hops,
        actual_deltas,
        eligibility,
        path_polarities,
        requested_direction: float,
    ) -> None:
        if channel not in self.channels:
            return
        sign = "positive" if float(requested_direction) > 0.0 else "negative"
        state = self.channels[channel]
        state["request_counts"][sign] += 1
        edges = np.asarray(edge_indices, dtype=np.int32)
        hop_array = np.asarray(hops, dtype=np.int8)
        deltas = np.asarray(actual_deltas, dtype=np.float64)
        changed = np.abs(deltas) > 1e-12
        changed_edges = edges[changed]
        changed_hops = hop_array[changed]
        changed_deltas = deltas[changed]
        state["request_yields"][sign].append(int(len(changed_edges)))
        context = f"target/{target}:{sign}"
        if len(changed_edges):
            state["edges"][sign].update(int(edge) for edge in changed_edges)
            state["contexts"][context].update(int(edge) for edge in changed_edges)
            pre_indices = np.searchsorted(
                np.asarray(self.connectome.indptr), changed_edges, side="right"
            ).astype(np.int32) - 1
            state["presynaptic"][sign].update(int(pre) for pre in pre_indices)
        for edge, hop, delta in zip(changed_edges, changed_hops, changed_deltas):
            edge = int(edge)
            hop = int(hop)
            if hop in HOPS:
                state["edges_by_hop"][sign][hop].add(edge)
            state["edge_effects"][edge][sign] += float(delta)
            state["hop_effects"][(edge, hop)][sign] += float(delta)

    def observe_route_health(
        self,
        *,
        target: str,
        signal,
        health_by_channel: Mapping[str, object],
    ) -> None:
        for channel, direction in signal.nonzero_directions().items():
            if channel not in self.channels or channel not in health_by_channel:
                continue
            sign = "positive" if direction > 0.0 else "negative"
            context = f"target/{target}:{sign}"
            rows = self.channels[channel]["route_health"][context]
            if len(rows) >= self.max_route_health_requests:
                continue
            health = health_by_channel[channel]
            rows.append({
                field: float(getattr(health, field, 0.0))
                for field in ROUTE_HEALTH_FIELDS
            })

    def wants_route_health(self, *, target: str, signal) -> bool:
        """Return whether this bounded diagnostic still has sample capacity."""
        for channel, direction in signal.nonzero_directions().items():
            if channel not in self.channels:
                continue
            sign = "positive" if direction > 0.0 else "negative"
            if len(self.channels[channel]["route_health"][f"target/{target}:{sign}"]) < self.max_route_health_requests:
                return True
        return False

    def absorb(self, other: "SymbolCreditInterferenceAudit") -> None:
        for channel in CHANNELS:
            _merge_channel(self.channels[channel], other.channels[channel])

    def _route_health_report(self, state):
        result = {}
        for context, rows in sorted(state["route_health"].items()):
            result[context] = {
                "sample_count": int(len(rows)),
                **{
                    field: float(np.mean([row[field] for row in rows]))
                    for field in ROUTE_HEALTH_FIELDS
                },
            }
        return result

    def report(self) -> dict[str, object]:
        channels = {}
        for channel, state in self.channels.items():
            positive = state["edges"]["positive"]
            negative = state["edges"]["negative"]
            positive_yield = _yield_summary(state["request_yields"]["positive"])
            negative_yield = _yield_summary(state["request_yields"]["negative"])
            positive_yield["positive_zero_update_count"] = positive_yield["zero_update_count"]
            positive_yield["positive_zero_update_fraction"] = positive_yield["zero_update_fraction"]
            negative_yield["negative_zero_update_count"] = negative_yield["zero_update_count"]
            negative_yield["negative_zero_update_fraction"] = negative_yield["zero_update_fraction"]
            channels[channel] = {
                "positive_requests": positive_yield,
                "negative_requests": negative_yield,
                "edge_overlap": {
                    "positive_unique_edges": int(len(positive)),
                    "negative_unique_edges": int(len(negative)),
                    "intersection_count": int(len(positive & negative)),
                    "jaccard_positive_negative": jaccard(positive, negative),
                    "fraction_positive_edges_also_negative": float(len(positive & negative) / len(positive)) if positive else 0.0,
                    "fraction_negative_edges_also_positive": float(len(positive & negative) / len(negative)) if negative else 0.0,
                },
                "cancellation": _safe_cancellation(state["edge_effects"]),
                "hop_specific": {
                    "1hop": {
                        "jaccard_positive_negative": jaccard(state["edges_by_hop"]["positive"][1], state["edges_by_hop"]["negative"][1]),
                        **_safe_cancellation({key: value for key, value in state["hop_effects"].items() if key[1] == 1}),
                    },
                    "2hop": {
                        "jaccard_positive_negative": jaccard(state["edges_by_hop"]["positive"][2], state["edges_by_hop"]["negative"][2]),
                        **_safe_cancellation({key: value for key, value in state["hop_effects"].items() if key[1] == 2}),
                    },
                },
                "presynaptic_overlap": {
                    "positive_presynaptic_neurons": int(len(state["presynaptic"]["positive"])),
                    "negative_presynaptic_neurons": int(len(state["presynaptic"]["negative"])),
                    "intersection_count": int(len(state["presynaptic"]["positive"] & state["presynaptic"]["negative"])),
                    "jaccard": jaccard(state["presynaptic"]["positive"], state["presynaptic"]["negative"]),
                },
                "contexts": {
                    context: {"unique_edges": int(len(edges))}
                    for context, edges in sorted(state["contexts"].items())
                },
                "pairwise_context_overlap": _context_overlap(state["contexts"]),
                "route_health": self._route_health_report(state),
            }
        return channels


def cross_target_positive_overlap(audit: SymbolCreditInterferenceAudit) -> dict[str, object]:
    """Compare positive credit sets across input/target contexts."""
    target_sets = defaultdict(set)
    for state in audit.channels.values():
        for context, edges in state["contexts"].items():
            if context.endswith(":positive"):
                target_sets[context.split(":", 1)[0]].update(edges)
    return {
        f"{left}|{right}": jaccard(target_sets[left], target_sets[right])
        for left, right in combinations(sorted(target_sets), 2)
    }


def classify_primary_hypotheses(channels: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Apply predeclared descriptive thresholds without changing training."""
    d = channels["symbol/D"]
    d_jaccard = float(d["edge_overlap"]["jaccard_positive_negative"])
    d_cancellation = float(d["cancellation"]["cancellation_fraction"])
    d_strong = bool(d_jaccard >= 0.25 and d_cancellation >= 0.25)

    a_positive = channels["symbol/A"]["positive_requests"]
    comparison = {
        symbol: {
            "positive_zero_update_fraction": float(channels[f"symbol/{symbol}"]["positive_requests"]["positive_zero_update_fraction"]),
            "mean_edge_updates_per_positive_request": float(channels[f"symbol/{symbol}"]["positive_requests"]["mean_edge_updates_per_request"]),
        }
        for symbol in SYMBOLS
    }
    other_zero = float(np.mean([
        comparison["B"]["positive_zero_update_fraction"],
        comparison["D"]["positive_zero_update_fraction"],
    ]))
    other_yield = float(np.mean([
        comparison["B"]["mean_edge_updates_per_positive_request"],
        comparison["D"]["mean_edge_updates_per_positive_request"],
    ]))
    a_starved = bool(
        a_positive["positive_zero_update_fraction"] > max(0.0, other_zero * 1.25)
        and a_positive["mean_edge_updates_per_request"] < other_yield * 0.75
    )
    if a_starved and d_strong:
        primary = "both"
    elif a_starved:
        primary = "credit_starvation"
    elif d_strong:
        primary = "sign_conflicting_context_reuse"
    else:
        primary = "neither"
    upstream = bool(
        np.mean([
            channels[channel]["presynaptic_overlap"]["jaccard"]
            for channel in CHANNELS
        ]) >= 0.25
    )
    return {
        "A_credit_starvation_supported": a_starved,
        "D_sign_conflict_supported": d_strong,
        "upstream_context_interference_supported": upstream,
        "primary_bottleneck": primary,
        "D_thresholds": {
            "edge_jaccard": d_jaccard,
            "cancellation_fraction": d_cancellation,
            "strong_sign_conflict": d_strong,
        },
        "A_credit_starvation_comparison": comparison,
        "A_vs_BD_thresholds": {
            "A_zero_update_fraction": float(a_positive["positive_zero_update_fraction"]),
            "B_D_mean_zero_update_fraction": other_zero,
            "A_mean_edge_updates_per_request": float(a_positive["mean_edge_updates_per_request"]),
            "B_D_mean_edge_updates_per_request": other_yield,
        },
        "recommended_next_step": (
            "combined_credit_specificity_repair" if primary == "both" else
            "context_specific_credit_gating" if primary == "sign_conflicting_context_reuse" else
            "improve_route_engagement" if primary == "credit_starvation" else
            "reassess_symbol_learning_design"
        ),
    }


__all__ = [
    "CHANNELS",
    "SymbolCreditInterferenceAudit",
    "classify_primary_hypotheses",
    "cross_target_positive_overlap",
    "jaccard",
]
