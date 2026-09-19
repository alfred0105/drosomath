"""Diagnostics for the controlled F.1B.3 prospective two-hop intervention.

The audit is deliberately downstream of the generic controller.  It records
which real plastic edges changed under the selected credit mode; it never
selects edges and never modifies brain state.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median

import numpy as np

from .symbol_interface import SYMBOLS


CHANNELS = tuple(f"symbol/{symbol}" for symbol in SYMBOLS)
SIGNS = ("positive", "negative")


def _empty_state():
    return {
        "requests": {sign: 0 for sign in SIGNS},
        "yields": {sign: [] for sign in SIGNS},
        "legacy_updates": {sign: [] for sign in SIGNS},
        "prospective_updates": {sign: [] for sign in SIGNS},
        "legacy_edges": {sign: set() for sign in SIGNS},
        "prospective_edges": {sign: set() for sign in SIGNS},
        "prospective_hops": {sign: {1: set(), 2: set()} for sign in SIGNS},
        "sum_abs_delta": {sign: 0.0 for sign in SIGNS},
        "prospective_delta": {sign: 0.0 for sign in SIGNS},
        "hop_updates": {sign: {1: 0, 2: 0} for sign in SIGNS},
        "safety": {
            "checked": 0,
            "violations": 0,
            "wrong_hop": 0,
            "not_plastic": 0,
            "zero_eligibility": 0,
            "inactive_presynaptic": 0,
            "missing_downstream_route": 0,
        },
    }


def _yield(values):
    values = [int(value) for value in values]
    count = len(values)
    nonzero = sum(value > 0 for value in values)
    return {
        "direction_requests": count,
        "requests_with_at_least_one_edge_update": nonzero,
        "total_edge_updates": int(sum(values)),
        "mean_edge_updates_per_request": float(np.mean(values)) if values else 0.0,
        "median_edge_updates_per_request": float(median(values)) if values else 0.0,
        "zero_update_count": int(count - nonzero),
        "zero_update_fraction": float((count - nonzero) / count) if count else 0.0,
    }


class ProspectiveTwoHopAudit:
    """Collect intervention attribution and frozen-route safety evidence."""

    def __init__(self, connectome):
        self.connectome = connectome
        self.channels = {channel: _empty_state() for channel in CHANNELS}

    def observe_attribution(
        self,
        *,
        target: str,
        decision: str,
        output_context,
        channel: str,
        current_edge_indices,
        current_hops,
        legacy_edge_indices,
        legacy_hops,
        actual_deltas,
        eligibility,
        path_polarities,
        requested_direction: float,
        brain,
    ) -> None:
        if channel not in self.channels:
            return
        sign = "positive" if requested_direction > 0.0 else "negative"
        state = self.channels[channel]
        state["requests"][sign] += 1
        edges = np.asarray(current_edge_indices, dtype=np.int32)
        hops = np.asarray(current_hops, dtype=np.int8)
        deltas = np.asarray(actual_deltas, dtype=np.float64)
        eligibility = np.asarray(eligibility, dtype=np.float64)
        changed = np.abs(deltas) > 1e-12
        changed_edges = edges[changed]
        changed_hops = hops[changed]
        state["yields"][sign].append(int(len(changed_edges)))
        legacy = set(int(edge) for edge in np.asarray(legacy_edge_indices, dtype=np.int32))
        legacy_changed = [int(edge) for edge in changed_edges if int(edge) in legacy]
        prospective_changed = [
            int(edge) for edge in changed_edges if int(edge) not in legacy
        ]
        state["legacy_updates"][sign].append(len(legacy_changed))
        state["prospective_updates"][sign].append(len(prospective_changed))
        state["legacy_edges"][sign].update(legacy_changed)
        state["prospective_edges"][sign].update(prospective_changed)
        state["sum_abs_delta"][sign] += float(np.abs(deltas[changed]).sum())
        prospective_mask = changed & ~np.isin(
            edges, np.asarray(tuple(legacy), dtype=np.int32)
        )
        state["prospective_delta"][sign] += float(np.abs(deltas[prospective_mask]).sum())

        graph = brain.connectome
        plastic = brain.plasticity.plastic_mask
        recent = {int(value) for value in brain._recent_presynaptic}
        outputs = set(int(value) for value in output_context.get(channel, ()))
        for edge, hop, eligibility_value in zip(changed_edges, changed_hops, eligibility[changed]):
            edge = int(edge)
            hop = int(hop)
            if hop in (1, 2):
                state["hop_updates"][sign][hop] += 1
            if edge in legacy:
                continue
            state["prospective_hops"][sign].setdefault(hop, set()).add(edge)
            state["safety"]["checked"] += 1
            pre = int(np.searchsorted(np.asarray(graph.indptr), edge, side="right") - 1)
            intermediate = int(graph.post_indices[edge])
            start, stop = int(graph.indptr[intermediate]), int(graph.indptr[intermediate + 1])
            has_downstream = bool(
                np.any(np.isin(graph.post_indices[start:stop], tuple(outputs)))
            )
            violation = False
            if hop != 2:
                state["safety"]["wrong_hop"] += 1
                violation = True
            if not bool(plastic[edge]):
                state["safety"]["not_plastic"] += 1
                violation = True
            if float(eligibility_value) <= 0.0:
                state["safety"]["zero_eligibility"] += 1
                violation = True
            if pre not in recent:
                state["safety"]["inactive_presynaptic"] += 1
                violation = True
            if not has_downstream:
                state["safety"]["missing_downstream_route"] += 1
                violation = True
            state["safety"]["violations"] += int(violation)

    def absorb(self, other: "ProspectiveTwoHopAudit") -> None:
        for channel in CHANNELS:
            target = self.channels[channel]
            source = other.channels[channel]
            for sign in SIGNS:
                target["requests"][sign] += source["requests"][sign]
                target["yields"][sign].extend(source["yields"][sign])
                target["legacy_updates"][sign].extend(source["legacy_updates"][sign])
                target["prospective_updates"][sign].extend(source["prospective_updates"][sign])
                target["legacy_edges"][sign].update(source["legacy_edges"][sign])
                target["prospective_edges"][sign].update(source["prospective_edges"][sign])
                target["sum_abs_delta"][sign] += source["sum_abs_delta"][sign]
                target["prospective_delta"][sign] += source["prospective_delta"][sign]
                for hop in (1, 2):
                    target["prospective_hops"][sign][hop].update(
                        source["prospective_hops"][sign][hop]
                    )
                    target["hop_updates"][sign][hop] += source["hop_updates"][sign][hop]
            for key in target["safety"]:
                target["safety"][key] += source["safety"][key]

    def report(self) -> dict[str, object]:
        result = {}
        for channel, state in self.channels.items():
            signs = {}
            for sign in SIGNS:
                yields = _yield(state["yields"][sign])
                total = len(state["yields"][sign])
                prospective = _yield(state["prospective_updates"][sign])
                legacy = _yield(state["legacy_updates"][sign])
                hop_total = sum(state["hop_updates"][sign].values())
                signs[sign] = {
                    **yields,
                    "unique_edges_modified": int(
                        len(state["legacy_edges"][sign] | state["prospective_edges"][sign])
                    ),
                    "sum_abs_multiplier_delta": float(state["sum_abs_delta"][sign]),
                    "direct_updates": int(state["hop_updates"][sign][1]),
                    "two_hop_updates": int(state["hop_updates"][sign][2]),
                    "direct_fraction": float(state["hop_updates"][sign][1] / hop_total) if hop_total else 0.0,
                    "two_hop_fraction": float(state["hop_updates"][sign][2] / hop_total) if hop_total else 0.0,
                    "legacy_compatible": legacy,
                    "prospective_only": prospective,
                    "prospective_only_unique_edges": int(len(state["prospective_edges"][sign])),
                    "prospective_only_sum_abs_delta": float(state["prospective_delta"][sign]),
                    "prospective_only_fraction_of_updates": float(
                        sum(state["prospective_updates"][sign]) / max(1, sum(state["yields"][sign]))
                    ),
                    "requests_with_prospective_only_updates": int(
                        sum(value > 0 for value in state["prospective_updates"][sign])
                    ),
                }
            result[channel] = {
                "positive": signs["positive"],
                "negative": signs["negative"],
                "safety": {
                    **state["safety"],
                    "passed": state["safety"]["violations"] == 0,
                },
            }
        return result


__all__ = ["CHANNELS", "ProspectiveTwoHopAudit"]
