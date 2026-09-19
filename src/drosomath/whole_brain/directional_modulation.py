"""Sparse, task-independent output-direction neuromodulation."""

from __future__ import annotations

from dataclasses import dataclass

from drosomath.learning_signal import LearningSignal


@dataclass(frozen=True, slots=True)
class DirectionalModulationConfig:
    learning_rate: float = 0.02
    stability_protection: float = 0.5
    minimum_learning_factor: float = 0.10
    max_credit_hops: int = 2
    credit_decay_per_hop: float = 0.5
    consolidation_gain: float = 0.01


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


class PlasticityController:
    """Apply local output-direction feedback using only active eligibility.

    ``output_context`` maps generic channel names to output-neuron indices.  It
    contains no environment-specific label or teacher edge set.
    """

    def __init__(self, config: DirectionalModulationConfig | None = None) -> None:
        self.config = config or DirectionalModulationConfig()

    def apply_learning_signal(self, brain, signal: LearningSignal, output_context) -> DirectionalUpdate:
        np = brain.np
        state, graph = brain.plasticity, brain.connectome
        total, sum_abs = 0, 0.0
        per_channel: dict[str, int] = {}
        changed_all = []
        hops: dict[int, int] = {}
        excitatory = inhibitory = consolidated = 0
        # Only rows from currently/recently firing presynaptic neurons can
        # hold current-trial eligibility; avoid a full graph scan.
        active_rows = sorted(brain._recent_presynaptic)
        row_chunks = [
            np.arange(int(graph.indptr[pre]), int(graph.indptr[pre + 1]), dtype=np.int32)
            for pre in active_rows if int(graph.indptr[pre + 1]) > int(graph.indptr[pre])
        ]
        active_edges = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int32)
        active_edges = active_edges[state.plastic_mask[active_edges] & (state.eligibility[active_edges] > 0.0)]
        for name, direction in signal.nonzero_directions().items():
            outputs = output_context.get(name)
            if outputs is None:
                continue
            output_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
            output_mask[np.asarray(outputs, dtype=np.int32)] = True
            one_hop = active_edges[output_mask[graph.post_indices[active_edges]]]
            credited = [(one_hop, 1.0, 1)]
            if self.config.max_credit_hops >= 2 and len(one_hop):
                one_pre = np.searchsorted(graph.indptr, one_hop, side="right").astype(np.int32) - 1
                target_pre = np.zeros(graph.neuron_count, dtype=bool)
                target_pre[np.unique(one_pre)] = True
                two_hop = active_edges[target_pre[graph.post_indices[active_edges]] & ~output_mask[graph.post_indices[active_edges]]]
                credited.append((two_hop, self.config.credit_decay_per_hop, 2))
            active = np.concatenate([edges for edges, _, _ in credited if len(edges)]) if any(len(edges) for edges, _, _ in credited) else np.empty(0, dtype=np.int32)
            if not len(active):
                per_channel[name] = 0
                continue
            channel_count = 0
            for edges, hop_weight, hop in credited:
                if not len(edges):
                    continue
                # Sign-correct: positive requested output strengthens
                # excitatory and weakens inhibitory contributions.
                sign = np.sign(graph.signed_synapse_counts[edges])
                factor = np.maximum(self.config.minimum_learning_factor, 1.0 - self.config.stability_protection * state.stability[edges])
                delta = self.config.learning_rate * float(direction) * sign * state.eligibility[edges] * factor * hop_weight
                old = state.multiplier[edges].copy()
                state.multiplier[edges] = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
                actual = state.multiplier[edges] - old
                if signal.success:
                    state.stability[edges] += self.config.consolidation_gain * (1.0 - state.stability[edges])
                    np.clip(state.stability[edges], 0.0, 1.0, out=state.stability[edges])
                    consolidated += int(len(edges))
                count = int(len(edges)); total += count; channel_count += count
                sum_abs += float(np.abs(actual).sum()); changed_all.append(edges)
                hops[hop] = hops.get(hop, 0) + count
                excitatory += int((sign > 0).sum()); inhibitory += int((sign < 0).sum())
            per_channel[name] = channel_count
        updated = np.unique(np.concatenate(changed_all)).astype(np.int32, copy=False) if changed_all else np.empty(0, dtype=np.int32)
        return DirectionalUpdate(total, per_channel, sum_abs / total if total else 0.0, updated, hops, excitatory, inhibitory, consolidated)
