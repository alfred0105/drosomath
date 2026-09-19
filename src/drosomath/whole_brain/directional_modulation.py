"""Sparse, task-independent output-direction neuromodulation."""

from __future__ import annotations

from dataclasses import dataclass

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain.usage_learning import RewardCredit


@dataclass(frozen=True, slots=True)
class DirectionalModulationConfig:
    learning_rate: float = 0.02
    stability_protection: float = 0.5
    minimum_learning_factor: float = 0.10
    max_credit_hops: int = 2
    credit_decay_per_hop: float = 0.5
    consolidation_gain: float = 0.01
    minimum_downstream_effect: float = 1e-8


@dataclass(frozen=True, slots=True)
class CreditEdges:
    """Internal bounded causal credit selection for one output channel."""

    edges: object
    hops: object
    path_polarities: object
    weights: object
    ambiguous_path_edges_skipped: int = 0


@dataclass(frozen=True, slots=True)
class DirectionalUpdate:
    edge_updates: int
    channel_updates: dict[str, int]
    mean_abs_delta: float
    updated_edge_indices: object
    hop_counts: dict[int, int]
    excitatory_updates: int
    inhibitory_updates: int
    consolidated_edges: int
    unique_edge_updates: int
    reinforced_channels: tuple[str, ...]
    ambiguous_path_edges_skipped: int


class PlasticityController:
    """Apply local output feedback using active, eligible, plastic routes.

    ``output_context`` maps generic channel names to output-neuron indices. No
    task label, teacher edge set, or environment-specific data enters here.
    """

    def __init__(self, config: DirectionalModulationConfig | None = None) -> None:
        self.config = config or DirectionalModulationConfig()

    def _active_edges(self, brain):
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        rows = sorted(brain._recent_presynaptic)
        chunks = [
            np.arange(int(graph.indptr[pre]), int(graph.indptr[pre + 1]), dtype=np.int32)
            for pre in rows if int(graph.indptr[pre + 1]) > int(graph.indptr[pre])
        ]
        edges = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        return edges[state.plastic_mask[edges] & (state.eligibility[edges] > 0.0)]

    @staticmethod
    def _pre_indices(np, graph, edges):
        return np.searchsorted(graph.indptr, edges, side="right").astype(np.int32) - 1

    def _credit_edges(self, brain, active_edges, outputs) -> CreditEdges:
        """Select one/two-hop active routes and their net effect on outputs."""
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        empty = np.empty(0, dtype=np.int32)
        if not len(active_edges):
            return CreditEdges(empty, empty, empty, np.empty(0, dtype=np.float32))
        output_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
        output_mask[np.asarray(outputs, dtype=np.int32)] = True
        direct = active_edges[output_mask[graph.post_indices[active_edges]]]
        direct_sign = np.sign(graph.signed_synapse_counts[direct])
        direct_keep = direct_sign != 0.0
        direct, direct_sign = direct[direct_keep], direct_sign[direct_keep]
        parts = [(direct, np.ones(len(direct), dtype=np.int8), direct_sign, np.ones(len(direct), dtype=np.float32))]
        ambiguous = 0
        if self.config.max_credit_hops < 2 or not len(direct):
            return self._join_credit(np, parts, ambiguous)

        intermediates = np.unique(self._pre_indices(np, graph, direct))
        intermediate_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
        intermediate_mask[intermediates] = True
        upstream = active_edges[
            intermediate_mask[graph.post_indices[active_edges]]
            & ~output_mask[graph.post_indices[active_edges]]
        ]
        if not len(upstream):
            return self._join_credit(np, parts, ambiguous)
        upstream_post = graph.post_indices[upstream]
        downstream_effect = {}
        for intermediate in np.unique(upstream_post):
            start, stop = int(graph.indptr[intermediate]), int(graph.indptr[intermediate + 1])
            outgoing = np.arange(start, stop, dtype=np.int32)
            outgoing = outgoing[output_mask[graph.post_indices[outgoing]]]
            # Net anatomical signed influence, scaled by current learned
            # strength; never pick an arbitrary first outgoing edge.
            downstream_effect[int(intermediate)] = float(
                (graph.signed_synapse_counts[outgoing] * state.multiplier[outgoing]).sum()
            ) if len(outgoing) else 0.0
        effects = np.asarray([downstream_effect[int(post)] for post in upstream_post], dtype=np.float32)
        downstream_sign = np.sign(effects)
        keep = np.abs(effects) > self.config.minimum_downstream_effect
        ambiguous = int((~keep).sum())
        upstream = upstream[keep]
        if len(upstream):
            polarity = np.sign(graph.signed_synapse_counts[upstream]) * downstream_sign[keep]
            valid = polarity != 0.0
            ambiguous += int((~valid).sum())
            upstream, polarity = upstream[valid], polarity[valid]
            parts.append((
                upstream,
                np.full(len(upstream), 2, dtype=np.int8),
                polarity,
                np.full(len(upstream), self.config.credit_decay_per_hop, dtype=np.float32),
            ))
        return self._join_credit(np, parts, ambiguous)

    @staticmethod
    def _join_credit(np, parts, ambiguous):
        nonempty = [part for part in parts if len(part[0])]
        if not nonempty:
            empty = np.empty(0, dtype=np.int32)
            return CreditEdges(empty, empty, empty, np.empty(0, dtype=np.float32), ambiguous)
        return CreditEdges(
            np.concatenate([part[0] for part in nonempty]),
            np.concatenate([part[1] for part in nonempty]),
            np.concatenate([part[2] for part in nonempty]),
            np.concatenate([part[3] for part in nonempty]),
            ambiguous,
        )

    def build_reward_credit(self, brain, signal: LearningSignal, output_context) -> RewardCredit:
        """Build sparse positive credit from the same bounded causal routes."""
        np = brain.np
        active_edges = self._active_edges(brain)
        edge_parts = []
        weight_parts = []
        aligned_one = aligned_two = opposing = ambiguous = 0
        for name, magnitude in signal.positive_reinforcements().items():
            outputs = output_context.get(name)
            if outputs is None:
                continue
            credit = self._credit_edges(brain, active_edges, outputs)
            ambiguous += credit.ambiguous_path_edges_skipped
            for hop in (1, 2):
                hop_mask = credit.hops == hop
                aligned = hop_mask & (credit.path_polarities > 0.0)
                if hop == 1:
                    aligned_one += int(aligned.sum())
                else:
                    aligned_two += int(aligned.sum())
                opposing += int((hop_mask & (credit.path_polarities < 0.0)).sum())
            aligned = credit.path_polarities > 0.0
            if aligned.any():
                edge_parts.append(credit.edges[aligned])
                weight_parts.append(
                    credit.weights[aligned] * abs(float(magnitude))
                )
        if not edge_parts:
            return RewardCredit(
                np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32),
                aligned_one, aligned_two, opposing + ambiguous,
                opposing, ambiguous,
            )
        edges = np.concatenate(edge_parts).astype(np.int32, copy=False)
        weights = np.concatenate(weight_parts).astype(np.float32, copy=False)
        unique, inverse = np.unique(edges, return_inverse=True)
        combined = np.zeros(len(unique), dtype=np.float32)
        np.add.at(combined, inverse, weights)
        return RewardCredit(
            unique.astype(np.int32, copy=False), combined,
            aligned_one, aligned_two, opposing + ambiguous,
            opposing, ambiguous,
        )

    def apply_learning_signal(self, brain, signal: LearningSignal, output_context) -> DirectionalUpdate:
        np = brain.np
        state = brain.plasticity
        active_edges = self._active_edges(brain)
        total, sum_abs = 0, 0.0
        per_channel: dict[str, int] = {}
        changed_all = []
        hops: dict[int, int] = {}
        excitatory = inhibitory = consolidated = ambiguous = 0
        credits = {}

        def credit_for(name):
            nonlocal ambiguous
            if name not in credits:
                outputs = output_context.get(name)
                credits[name] = self._credit_edges(brain, active_edges, outputs) if outputs is not None else None
                if credits[name] is not None:
                    ambiguous += credits[name].ambiguous_path_edges_skipped
            return credits[name]

        for name, direction in signal.nonzero_directions().items():
            credit = credit_for(name)
            if credit is None or not len(credit.edges):
                per_channel[name] = 0
                continue
            edges = credit.edges
            factor = np.maximum(self.config.minimum_learning_factor, 1.0 - self.config.stability_protection * state.stability[edges])
            delta = self.config.learning_rate * float(direction) * credit.path_polarities * state.eligibility[edges] * factor * credit.weights
            old = state.multiplier[edges].copy()
            state.multiplier[edges] = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
            actual = state.multiplier[edges] - old
            count = int(len(edges)); total += count; per_channel[name] = count
            sum_abs += float(np.abs(actual).sum()); changed_all.append(edges)
            for hop in np.unique(credit.hops):
                hops[int(hop)] = hops.get(int(hop), 0) + int((credit.hops == hop).sum())
            anatomical_sign = np.sign(brain.connectome.signed_synapse_counts[edges])
            excitatory += int((anatomical_sign > 0).sum())
            inhibitory += int((anatomical_sign < 0).sum())

        reinforced = []
        for name in signal.positive_reinforcements():
            credit = credit_for(name)
            if credit is None or not len(credit.edges):
                continue
            # Reinforcement stabilizes causal routes; success alone does not
            # directly perturb their learned multipliers.
            aligned = credit.path_polarities > 0.0
            if not aligned.any():
                continue
            edges = np.unique(credit.edges[aligned])
            state.stability[edges] += self.config.consolidation_gain * (1.0 - state.stability[edges])
            np.clip(state.stability[edges], 0.0, 1.0, out=state.stability[edges])
            consolidated += int(len(edges)); reinforced.append(name)

        updated = np.unique(np.concatenate(changed_all)).astype(np.int32, copy=False) if changed_all else np.empty(0, dtype=np.int32)
        return DirectionalUpdate(
            total, per_channel, sum_abs / total if total else 0.0, updated, hops,
            excitatory, inhibitory, consolidated, int(len(updated)), tuple(reinforced), ambiguous,
        )
