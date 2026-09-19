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
    effective_influence: object = None


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
    sum_abs_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class DirectionalRouteHealth:
    """Read-only capacity diagnostics for one directional output channel."""

    channel: str
    requested_error_magnitude: float
    active_plastic_edges: int
    aligned_plastic_edges: int
    opposing_plastic_edges: int
    total_eligibility: float
    total_route_credit: float
    increase_headroom_edges: int
    decrease_headroom_edges: int
    saturated_edges: int
    unsaturated_useful_edges: int
    saturated_useful_edges: int
    estimated_available_adjustment: float
    useful_frozen_edges: int
    useful_frozen_one_hop: int
    useful_frozen_two_hop: int
    estimated_frozen_capacity: float
    frozen_to_plastic_capacity_ratio: float
    saturated_fraction: float
    ambiguous_path_edges_skipped: int
    route_health_status: str
    diagnostic_mode: str = "actual_failure"
    active_plastic_candidate_edges: int = 0
    active_frozen_candidate_edges: int = 0
    useful_plastic_edges: int = 0
    plastic_structural_opportunity: float = 0.0
    frozen_structural_opportunity: float = 0.0
    frozen_to_plastic_structural_opportunity_ratio: float = 0.0
    realized_plastic_capacity: float = 0.0
    plastic_engagement_efficiency: float = 0.0
    plastic_multiplier_headroom: float = 0.0
    frozen_multiplier_headroom: float = 0.0
    useful_plastic_route_fraction: float = 0.0
    useful_frozen_route_fraction: float = 0.0
    mean_effective_route_influence: float = 0.0
    frozen_raw_candidate_edges: int = 0
    frozen_unique_candidate_edges: int = 0
    frozen_direct_candidate_edges: int = 0
    frozen_two_hop_candidate_edges: int = 0


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

    def _active_anatomical_edges(self, brain, *, plastic: bool):
        """Return active anatomical edges for structural, allocation-matched probes."""
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        rows = sorted(int(pre) for pre in brain._recent_presynaptic)
        chunks = [
            np.arange(int(graph.indptr[pre]), int(graph.indptr[pre + 1]), dtype=np.int32)
            for pre in rows if int(graph.indptr[pre + 1]) > int(graph.indptr[pre])
        ]
        anatomical = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        if not len(anatomical):
            return anatomical
        mask = state.plastic_mask[anatomical]
        return anatomical[mask if plastic else ~mask]

    @staticmethod
    def _bounded_influence(values):
        """Normalize effective anatomical influence to [0, 1) for telemetry."""
        return abs(values) / (1.0 + abs(values))

    @staticmethod
    def _pre_indices(np, graph, edges):
        return np.searchsorted(graph.indptr, edges, side="right").astype(np.int32) - 1

    @staticmethod
    def _unique_credit(np, credit: CreditEdges) -> CreditEdges:
        """Collapse repeated route records without changing the anatomical edge set."""
        if len(credit.edges) <= 1:
            return credit
        unique, first = np.unique(credit.edges, return_index=True)
        influence = credit.effective_influence
        if influence is None:
            influence = np.ones(len(credit.edges), dtype=np.float32)
        return CreditEdges(
            unique.astype(np.int32, copy=False),
            credit.hops[first],
            credit.path_polarities[first],
            credit.weights[first],
            credit.ambiguous_path_edges_skipped,
            influence[first],
        )

    def _credit_edges(self, brain, active_edges, outputs) -> CreditEdges:
        """Select one/two-hop active routes and their net effect on outputs."""
        return self._route_credit_edges(brain, active_edges, outputs)

    def _route_credit_edges(self, brain, candidate_edges, outputs, *,
                            discover_upstream_from_candidates: bool = False) -> CreditEdges:
        """Shared bounded route analysis for plastic and frozen candidates."""
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        empty = np.empty(0, dtype=np.int32)
        if not len(candidate_edges):
            return CreditEdges(
                empty, empty, empty, np.empty(0, dtype=np.float32),
                effective_influence=np.empty(0, dtype=np.float32),
            )
        output_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
        output_mask[np.asarray(outputs, dtype=np.int32)] = True
        direct = candidate_edges[output_mask[graph.post_indices[candidate_edges]]]
        direct_sign = np.sign(graph.signed_synapse_counts[direct])
        direct_keep = direct_sign != 0.0
        direct, direct_sign = direct[direct_keep], direct_sign[direct_keep]
        direct_influence = self._bounded_influence(
            graph.signed_synapse_counts[direct]
        ).astype(np.float32, copy=False)
        parts = [(
            direct,
            np.ones(len(direct), dtype=np.int8),
            direct_sign,
            np.ones(len(direct), dtype=np.float32),
            direct_influence,
        )]
        ambiguous = 0
        if self.config.max_credit_hops < 2:
            return self._join_credit(np, parts, ambiguous)

        if discover_upstream_from_candidates:
            intermediates = np.unique(
                graph.post_indices[candidate_edges][
                    ~output_mask[graph.post_indices[candidate_edges]]
                ]
            )
        else:
            if not len(direct):
                return self._join_credit(np, parts, ambiguous)
            intermediates = np.unique(self._pre_indices(np, graph, direct))
        intermediate_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
        intermediate_mask[intermediates] = True
        upstream = candidate_edges[
            intermediate_mask[graph.post_indices[candidate_edges]]
            & ~output_mask[graph.post_indices[candidate_edges]]
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
            upstream_influence = self._bounded_influence(
                graph.signed_synapse_counts[upstream]
            ) * self._bounded_influence(effects[keep])
            upstream, polarity = upstream[valid], polarity[valid]
            parts.append((
                upstream,
                np.full(len(upstream), 2, dtype=np.int8),
                polarity,
                np.full(len(upstream), self.config.credit_decay_per_hop, dtype=np.float32),
                upstream_influence[valid].astype(np.float32, copy=False),
            ))
        return self._join_credit(np, parts, ambiguous)

    def structural_credit_edges(self, brain, outputs) -> CreditEdges:
        """Find useful frozen anatomical routes from actually active inputs.

        Frozen edges intentionally bypass eligibility here: structural need is
        the evidence used to decide whether an edge should become plastic.
        Activity still comes only from ``brain._recent_presynaptic``.
        """
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        rows = sorted(int(pre) for pre in brain._recent_presynaptic)
        chunks = [
            np.arange(int(graph.indptr[pre]), int(graph.indptr[pre + 1]), dtype=np.int32)
            for pre in rows if int(graph.indptr[pre + 1]) > int(graph.indptr[pre])
        ]
        anatomical = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        frozen = anatomical[~state.plastic_mask[anatomical]] if len(anatomical) else anatomical
        return self._route_credit_edges(
            brain,
            frozen,
            outputs,
            discover_upstream_from_candidates=True,
        )

    def diagnose_route_health(
        self,
        brain,
        signal: LearningSignal,
        output_context,
        *,
        standardized: bool = False,
    ):
        """Return read-only capacity diagnostics for nonzero directions.

        This method deliberately performs no state mutation. It uses the same
        bounded causal credit selection as directional learning, but compares
        active plastic routes with active frozen anatomical alternatives.
        """
        np = brain.np
        state = brain.plasticity
        active_edges = (
            self._active_anatomical_edges(brain, plastic=True)
            if standardized else self._active_edges(brain)
        )
        frozen_candidates = self._active_anatomical_edges(brain, plastic=False)
        diagnostics: dict[str, DirectionalRouteHealth] = {}

        for name, direction in signal.nonzero_directions().items():
            outputs = output_context.get(name)
            if outputs is None:
                continue
            magnitude = abs(float(direction))
            plastic_credit = self._route_credit_edges(
                brain,
                active_edges,
                outputs,
                discover_upstream_from_candidates=standardized,
            )
            frozen_credit = self._route_credit_edges(
                brain,
                frozen_candidates,
                outputs,
                discover_upstream_from_candidates=True,
            )
            raw_frozen_route_edges = int(len(frozen_credit.edges))
            plastic_credit = self._unique_credit(np, plastic_credit)
            frozen_credit = self._unique_credit(np, frozen_credit)

            def measure(credit, *, frozen: bool):
                if not len(credit.edges):
                    return {
                        "useful": np.empty(0, dtype=np.int32),
                        "required": np.empty(0, dtype=np.float32),
                        "headroom": np.empty(0, dtype=np.float32),
                        "structural": 0.0,
                        "realized": 0.0,
                    }
                edges = credit.edges
                required = np.sign(float(direction) * credit.path_polarities)
                current = state.multiplier[edges]
                increase = required > 0.0
                decrease = required < 0.0
                headroom = np.where(
                    increase,
                    state.config.max_multiplier - current,
                    np.where(decrease, current - state.config.min_multiplier, 0.0),
                ).astype(np.float32, copy=False)
                useful = np.flatnonzero(
                    (required != 0.0) & (headroom > 1e-7)
                ).astype(np.int32, copy=False)
                influence = credit.effective_influence
                if influence is None:
                    influence = np.ones(len(edges), dtype=np.float32)
                structural_weights = (
                    credit.weights[useful]
                    * headroom[useful]
                    * influence[useful]
                    * magnitude
                )
                structural = float(structural_weights.sum())
                realized = float(
                    (state.eligibility[edges[useful]] * structural_weights).sum()
                ) if not frozen else 0.0
                return {
                    "useful": useful,
                    "required": required,
                    "headroom": headroom,
                    "structural": structural,
                    "realized": realized,
                }

            plastic = measure(plastic_credit, frozen=False)
            frozen = measure(frozen_credit, frozen=True)
            plastic_edges = plastic_credit.edges
            required = plastic["required"]
            headroom = plastic["headroom"]
            useful = plastic["useful"]
            saturated = (required != 0.0) & (headroom <= 1e-7)
            aligned = plastic_credit.path_polarities > 0.0
            opposing = plastic_credit.path_polarities < 0.0
            frozen_useful = frozen["useful"]
            frozen_hops = (
                frozen_credit.hops[frozen_useful]
                if len(frozen_useful) else np.empty(0, dtype=np.int8)
            )
            plastic_structural = float(plastic["structural"])
            frozen_structural = float(frozen["structural"])
            realized_plastic = float(plastic["realized"])
            structural_ratio = frozen_structural / max(plastic_structural, 1e-9)
            realized_ratio = frozen_structural / max(realized_plastic, 1e-9)
            if not len(plastic_edges):
                status = "FROZEN_ALTERNATIVES_AVAILABLE" if len(frozen_useful) else "NO_PLASTIC_ROUTE"
            elif len(saturated) and int(saturated.sum()) == len(plastic_edges):
                status = "PLASTIC_ROUTE_SATURATED"
            elif frozen_structural > plastic_structural and len(frozen_useful):
                status = "FROZEN_ALTERNATIVES_AVAILABLE"
            elif not len(useful):
                status = "LOW_PLASTIC_CAPACITY"
            else:
                status = "PLASTIC_CAPACITY_ADEQUATE"

            influence_parts = []
            if len(useful) and plastic_credit.effective_influence is not None:
                influence_parts.append(plastic_credit.effective_influence[useful])
            if len(frozen_useful) and frozen_credit.effective_influence is not None:
                influence_parts.append(frozen_credit.effective_influence[frozen_useful])
            mean_influence = float(np.mean(np.concatenate(influence_parts))) if influence_parts else 0.0

            diagnostics[name] = DirectionalRouteHealth(
                channel=name,
                requested_error_magnitude=magnitude,
                active_plastic_edges=int(len(plastic_edges)),
                aligned_plastic_edges=int(aligned.sum()),
                opposing_plastic_edges=int(opposing.sum()),
                total_eligibility=float(state.eligibility[plastic_edges].sum()) if len(plastic_edges) else 0.0,
                total_route_credit=float(
                    (state.eligibility[plastic_edges] * plastic_credit.weights).sum()
                ) if len(plastic_edges) else 0.0,
                increase_headroom_edges=int(((required > 0.0) & (headroom > 1e-7)).sum()),
                decrease_headroom_edges=int(((required < 0.0) & (headroom > 1e-7)).sum()),
                saturated_edges=int(saturated.sum()),
                unsaturated_useful_edges=int(len(useful)),
                saturated_useful_edges=int(saturated.sum()),
                estimated_available_adjustment=realized_plastic,
                useful_frozen_edges=int(len(frozen_useful)),
                useful_frozen_one_hop=int((frozen_hops == 1).sum()),
                useful_frozen_two_hop=int((frozen_hops == 2).sum()),
                estimated_frozen_capacity=frozen_structural,
                frozen_to_plastic_capacity_ratio=float(realized_ratio),
                saturated_fraction=float(saturated.sum() / max(1, len(plastic_edges))),
                ambiguous_path_edges_skipped=int(
                    plastic_credit.ambiguous_path_edges_skipped
                    + frozen_credit.ambiguous_path_edges_skipped
                ),
                route_health_status=status,
                diagnostic_mode="counterfactual_unit" if standardized else "actual_failure",
                active_plastic_candidate_edges=int(len(active_edges)),
                active_frozen_candidate_edges=int(len(frozen_candidates)),
                useful_plastic_edges=int(len(useful)),
                plastic_structural_opportunity=plastic_structural,
                frozen_structural_opportunity=frozen_structural,
                frozen_to_plastic_structural_opportunity_ratio=float(structural_ratio),
                realized_plastic_capacity=realized_plastic,
                plastic_engagement_efficiency=float(
                    realized_plastic / max(plastic_structural, 1e-9)
                ),
                plastic_multiplier_headroom=float(
                    (headroom[useful] * plastic_credit.weights[useful]).sum()
                    if len(useful) else 0.0
                ),
                frozen_multiplier_headroom=float(
                    (frozen["headroom"][frozen_useful] * frozen_credit.weights[frozen_useful]).sum()
                    if len(frozen_useful) else 0.0
                ),
                useful_plastic_route_fraction=float(
                    len(useful) / max(1, len(active_edges))
                ),
                useful_frozen_route_fraction=float(
                    len(frozen_useful) / max(1, len(frozen_candidates))
                ),
                mean_effective_route_influence=mean_influence,
                frozen_raw_candidate_edges=raw_frozen_route_edges,
                frozen_unique_candidate_edges=int(len(frozen_credit.edges)),
                frozen_direct_candidate_edges=int((frozen_credit.hops == 1).sum()),
                frozen_two_hop_candidate_edges=int((frozen_credit.hops == 2).sum()),
            )
        return diagnostics

    @staticmethod
    def _join_credit(np, parts, ambiguous):
        nonempty = [part for part in parts if len(part[0])]
        if not nonempty:
            empty = np.empty(0, dtype=np.int32)
            return CreditEdges(
                empty, empty, empty, np.empty(0, dtype=np.float32), ambiguous,
                np.empty(0, dtype=np.float32),
            )
        return CreditEdges(
            np.concatenate([part[0] for part in nonempty]),
            np.concatenate([part[1] for part in nonempty]),
            np.concatenate([part[2] for part in nonempty]),
            np.concatenate([part[3] for part in nonempty]),
            ambiguous,
            np.concatenate([part[4] for part in nonempty]),
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
            sum_abs,
        )
