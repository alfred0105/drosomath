"""Sparse, task-independent output-direction neuromodulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext as _nullcontext
from types import MappingProxyType

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
    # F.1B.3 intervention switch.  The default deliberately preserves the
    # pre-intervention active-chain credit rule exactly.
    two_hop_credit_mode: str = "active_chain"

    def __post_init__(self) -> None:
        if self.two_hop_credit_mode not in {"active_chain", "prospective_anatomical"}:
            raise ValueError(
                "two_hop_credit_mode must be 'active_chain' or "
                "'prospective_anatomical'"
            )


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
class OutputRouteIndex:
    """Immutable topology index for one generic output population.

    Learned multipliers are intentionally absent.  They are read from the
    live plastic state whenever a route is scored.
    """

    output_indices: object
    output_mask: object
    downstream_intermediates: object
    downstream_edges_by_intermediate: object
    # Compact lookup for the prospective two-hop path.  The mapping above is
    # retained for backwards-compatible diagnostic consumers; the learning
    # path uses these immutable CSR-like arrays to avoid repeated dictionary
    # construction and lookup.
    downstream_edge_indices: object = None
    downstream_indptr: object = None

    @property
    def nbytes(self) -> int:
        total = sum(
            int(getattr(value, "nbytes", 0))
            for value in (
                self.output_indices,
                self.output_mask,
                self.downstream_intermediates,
                self.downstream_edge_indices,
                self.downstream_indptr,
            )
        )
        total += sum(int(getattr(value, "nbytes", 0)) for value in self.downstream_edges_by_intermediate.values())
        return int(total)


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
    # Read-only telemetry for bounded diagnostic consumers.  This does not
    # participate in learning or checkpoint state.
    updated_edge_hops: dict[int, int] = field(default_factory=dict)
    # Read-only per-channel telemetry.  These fields do not participate in
    # route selection or state updates.
    channel_sum_abs_delta: dict[str, float] = field(default_factory=dict)
    channel_unique_edge_updates: dict[str, int] = field(default_factory=dict)
    channel_hop_counts: dict[str, dict[int, int]] = field(default_factory=dict)
    channel_edge_indices: dict[str, tuple[int, ...]] = field(default_factory=dict)
    telemetry_level: str = "full"


@dataclass(frozen=True, slots=True)
class PlasticRowIndex:
    """Allocation-only CSR index of plastic edges grouped by presynaptic row."""

    indptr: object
    edge_indices: object
    allocation_generation: int

    @property
    def nbytes(self) -> int:
        return int(
            getattr(self.indptr, "nbytes", 0)
            + getattr(self.edge_indices, "nbytes", 0)
        )


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

    def __init__(
        self,
        config: DirectionalModulationConfig | None = None,
        *,
        route_cache_enabled: bool = True,
        plastic_row_cache_enabled: bool = True,
        prospective_index_enabled: bool = True,
    ) -> None:
        self.config = config or DirectionalModulationConfig()
        self.route_cache_enabled = bool(route_cache_enabled)
        self.plastic_row_cache_enabled = bool(plastic_row_cache_enabled)
        self.prospective_index_enabled = bool(prospective_index_enabled)
        self._output_route_cache: dict[tuple[int, tuple[int, ...]], OutputRouteIndex] = {}
        self._plastic_row_cache: dict[tuple[int, int], PlasticRowIndex] = {}

    def _output_route_index(self, brain, outputs) -> OutputRouteIndex:
        """Build/read immutable output topology without caching learned state."""
        np = brain.np
        graph = brain.connectome
        values = np.asarray(outputs, dtype=np.int32)
        key = (id(graph), tuple(int(value) for value in values))
        cached = self._output_route_cache.get(key)
        if cached is not None:
            return cached
        output_indices = values.copy()
        output_mask = np.zeros(graph.neuron_count, dtype=np.bool_)
        output_mask[output_indices] = True
        all_edges = np.arange(len(graph.post_indices), dtype=np.int32)
        selected = all_edges[output_mask[graph.post_indices]]
        presynaptic = self._pre_indices(np, graph, selected)
        downstream_edges: dict[int, object] = {}
        if len(selected):
            unique_pre, starts = np.unique(presynaptic, return_index=True)
            stops = np.concatenate((starts[1:], np.asarray([len(selected)], dtype=np.int64)))
            for pre, start, stop in zip(unique_pre, starts, stops):
                downstream_edges[int(pre)] = selected[int(start):int(stop)].copy()
        output_indices.setflags(write=False)
        output_mask.setflags(write=False)
        downstream_intermediates = np.asarray(sorted(downstream_edges), dtype=np.int32)
        downstream_intermediates.setflags(write=False)
        downstream_indptr = np.zeros(len(downstream_intermediates) + 1, dtype=np.int32)
        if len(downstream_intermediates):
            downstream_indptr[1:] = np.cumsum(
                [len(downstream_edges[int(value)]) for value in downstream_intermediates],
                dtype=np.int64,
            ).astype(np.int32, copy=False)
            downstream_edge_indices = np.concatenate(
                [downstream_edges[int(value)] for value in downstream_intermediates]
            ).astype(np.int32, copy=False)
        else:
            downstream_edge_indices = np.empty(0, dtype=np.int32)
        downstream_edge_indices.setflags(write=False)
        downstream_indptr.setflags(write=False)
        for edges in downstream_edges.values():
            edges.setflags(write=False)
        index = OutputRouteIndex(
            output_indices,
            output_mask,
            downstream_intermediates,
            MappingProxyType(downstream_edges),
            downstream_edge_indices,
            downstream_indptr,
        )
        self._output_route_cache[key] = index
        return index

    def route_cache_bytes(self) -> int:
        return int(sum(index.nbytes for index in self._output_route_cache.values()))

    def _plastic_row_index(self, brain) -> PlasticRowIndex:
        """Return the current allocation topology without reading learned values."""
        graph, state = brain.connectome, brain.plasticity
        key = (id(graph), int(getattr(state, "allocation_generation", 0)))
        cached = self._plastic_row_cache.get(key)
        if cached is not None:
            return cached
        np = brain.np
        # Small unit-test doubles and legacy callers may expose only the
        # original dense mask.  Keep their reference path fully compatible.
        if not self.plastic_row_cache_enabled or not hasattr(state, "plastic_indices"):
            all_edges = np.arange(len(graph.post_indices), dtype=np.int32)
            plastic_edges = all_edges[np.asarray(state.plastic_mask, dtype=np.bool_)]
        else:
            plastic_edges = np.asarray(state.plastic_indices, dtype=np.int32)
        if len(plastic_edges):
            presynaptic = self._pre_indices(np, graph, plastic_edges)
            counts = np.bincount(presynaptic, minlength=graph.neuron_count)
            indptr = np.empty(graph.neuron_count + 1, dtype=np.int32)
            indptr[0] = 0
            np.cumsum(counts, dtype=np.int64, out=indptr[1:])
        else:
            indptr = np.zeros(graph.neuron_count + 1, dtype=np.int32)
        # plastic_indices is sorted in anatomical CSR edge order, so keeping
        # it intact preserves the reference active-edge order exactly.
        plastic_edges = plastic_edges.copy()
        indptr.setflags(write=False)
        plastic_edges.setflags(write=False)
        index = PlasticRowIndex(indptr, plastic_edges, int(state.allocation_generation))
        for old_key in tuple(self._plastic_row_cache):
            if old_key[0] == id(graph) and old_key != key:
                del self._plastic_row_cache[old_key]
        self._plastic_row_cache[key] = index
        return index

    def plastic_row_cache_bytes(self) -> int:
        return int(sum(index.nbytes for index in self._plastic_row_cache.values()))

    def _active_edges(self, brain):
        np = brain.np
        state = brain.plasticity
        if not self.plastic_row_cache_enabled or not hasattr(state, "plastic_indices"):
            graph = brain.connectome
            rows = sorted(brain._recent_presynaptic)
            chunks = [
                np.arange(int(graph.indptr[pre]), int(graph.indptr[pre + 1]), dtype=np.int32)
                for pre in rows if int(graph.indptr[pre + 1]) > int(graph.indptr[pre])
            ]
            edges = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
            return edges[state.plastic_mask[edges] & (state.eligibility[edges] > 0.0)]
        row_index = self._plastic_row_index(brain)
        rows = sorted(brain._recent_presynaptic)
        chunks = [
            row_index.edge_indices[int(row_index.indptr[pre]):int(row_index.indptr[pre + 1])]
            for pre in rows if int(row_index.indptr[pre + 1]) > int(row_index.indptr[pre])
        ]
        edges = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        # Allocation is cached; eligibility remains a live per-episode filter.
        # Keep the live mask check for low-level/debug callers that may edit
        # the mask directly; managed allocation changes invalidate the index
        # through allocation_generation.
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
                            discover_upstream_from_candidates: bool = False,
                            timing_profiler=None) -> CreditEdges:
        """Shared bounded route analysis for plastic and frozen candidates."""
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        empty = np.empty(0, dtype=np.int32)
        if not len(candidate_edges):
            return CreditEdges(
                empty, empty, empty, np.empty(0, dtype=np.float32),
                effective_influence=np.empty(0, dtype=np.float32),
            )
        route_index = self._output_route_index(brain, outputs) if self.route_cache_enabled else None
        output_mask = (
            route_index.output_mask
            if route_index is not None
            else np.zeros(graph.neuron_count, dtype=np.bool_)
        )
        if route_index is None:
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

        if timing_profiler is not None:
            timing_context = timing_profiler.section("prospective_downstream_effect_seconds")
        else:
            timing_context = _nullcontext()
        with timing_context:
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
            unique_posts, inverse = np.unique(upstream_post, return_inverse=True)
            downstream_effect = np.zeros(len(unique_posts), dtype=np.float32)
            if route_index is not None and self.prospective_index_enabled:
                # Search the immutable sorted intermediate index, then reduce
                # each original anatomical edge slice in its original order.
                # The reduction order is deliberately the same as the old
                # per-intermediate np.sum path, so learned values remain exact.
                positions = np.searchsorted(
                    route_index.downstream_intermediates,
                    unique_posts,
                )
                valid = positions < len(route_index.downstream_intermediates)
                if valid.any():
                    valid_positions = np.flatnonzero(valid)
                    valid[valid_positions] = (
                        route_index.downstream_intermediates[positions[valid_positions]]
                        == unique_posts[valid_positions]
                    )
                    for local, position in enumerate(positions):
                        if not valid[local]:
                            continue
                        start = int(route_index.downstream_indptr[int(position)])
                        stop = int(route_index.downstream_indptr[int(position) + 1])
                        outgoing = route_index.downstream_edge_indices[start:stop]
                        downstream_effect[local] = float(
                            (graph.signed_synapse_counts[outgoing] * state.multiplier[outgoing]).sum()
                        ) if len(outgoing) else 0.0
            else:
                for local, intermediate in enumerate(unique_posts):
                    start, stop = int(graph.indptr[intermediate]), int(graph.indptr[intermediate + 1])
                    outgoing = np.arange(start, stop, dtype=np.int32)
                    outgoing = outgoing[output_mask[graph.post_indices[outgoing]]]
                    # Net anatomical signed influence, scaled by current learned
                    # strength; never pick an arbitrary first outgoing edge.
                    downstream_effect[local] = float(
                        (graph.signed_synapse_counts[outgoing] * state.multiplier[outgoing]).sum()
                    ) if len(outgoing) else 0.0
            effects = downstream_effect[inverse]
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

    def diagnose_activity_window(self, brain, fired_indices, output_context):
        """Read-only causal engagement summary for one already-simulated window."""
        np = brain.np
        graph, state = brain.connectome, brain.plasticity
        fired = np.unique(np.asarray(fired_indices, dtype=np.int32))
        chunks = [
            np.arange(int(graph.indptr[pre]), int(graph.indptr[pre + 1]), dtype=np.int32)
            for pre in fired if 0 <= int(pre) < graph.neuron_count
            and int(graph.indptr[pre + 1]) > int(graph.indptr[pre])
        ]
        candidates = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        result = {}
        for name, outputs in output_context.items():
            credit = self._unique_credit(
                np,
                self._route_credit_edges(
                    brain,
                    candidates,
                    outputs,
                    discover_upstream_from_candidates=True,
                ),
            )
            edges = credit.edges
            influence = credit.effective_influence
            if influence is None:
                influence = np.ones(len(edges), dtype=np.float32)
            effective_magnitude = (
                credit.weights * influence * np.abs(state.multiplier[edges])
                if len(edges) else np.empty(0, dtype=np.float32)
            )
            plastic = state.plastic_mask[edges] if len(edges) else np.empty(0, dtype=np.bool_)
            eligibility = state.eligibility[edges[plastic]] if len(edges) else np.empty(0, dtype=np.float32)
            presynaptic = self._pre_indices(np, graph, edges) if len(edges) else np.empty(0, dtype=np.int32)
            positive = credit.path_polarities > 0.0
            negative = credit.path_polarities < 0.0
            result[name] = {
                "_causal_edge_indices": edges,
                "_causal_edge_hops": credit.hops,
                "active_click_causal_presynaptic_neurons": int(len(np.unique(presynaptic))),
                "active_click_causal_edges": int(len(edges)),
                "active_plastic_click_causal_edges": int(plastic.sum()),
                "active_frozen_click_causal_edges": int((~plastic).sum()),
                "positive_effect_route_count": int(positive.sum()),
                "negative_effect_route_count": int(negative.sum()),
                "positive_effect_magnitude": float(effective_magnitude[positive].sum()),
                "negative_effect_magnitude": float(effective_magnitude[negative].sum()),
                "net_click_route_influence": float(
                    (credit.path_polarities * effective_magnitude).sum()
                ),
                "total_click_route_eligibility": float(eligibility.sum()),
                "mean_click_route_eligibility": float(eligibility.mean()) if len(eligibility) else 0.0,
                "eligible_click_route_edges": int((eligibility > 0.0).sum()),
            }
        return result

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

    def apply_learning_signal(
        self,
        brain,
        signal: LearningSignal,
        output_context,
        *,
        telemetry_observer=None,
        attribution_observer=None,
        timing_profiler=None,
        telemetry_level: str = "full",
    ) -> DirectionalUpdate:
        if telemetry_level not in {"full", "summary"}:
            raise ValueError("telemetry_level must be 'full' or 'summary'")
        # Diagnostic callbacks need the edge arrays regardless of the compact
        # ordinary-training mode.  The learning math below is shared exactly.
        materialize_edge_telemetry = telemetry_level == "full" or telemetry_observer is not None or attribution_observer is not None
        np = brain.np
        state = brain.plasticity
        if timing_profiler is not None:
            with timing_profiler.section("active_edge_collection_seconds"):
                active_edges = self._active_edges(brain)
        else:
            active_edges = self._active_edges(brain)
        total, sum_abs = 0, 0.0
        per_channel: dict[str, int] = {}
        changed_all = []
        updated_edge_hops: dict[int, int] = {} if materialize_edge_telemetry else {}
        hops: dict[int, int] = {}
        excitatory = inhibitory = consolidated = ambiguous = 0
        directional_credits = {}
        reward_credits = {}
        prospective = self.config.two_hop_credit_mode == "prospective_anatomical"
        channel_sum_abs_delta: dict[str, float] = {}
        channel_unique_edge_updates: dict[str, int] = {}
        channel_hop_counts: dict[str, dict[int, int]] = {}
        channel_edge_indices: dict[str, tuple[int, ...]] = {} if materialize_edge_telemetry else {}

        def directional_credit_for(name):
            nonlocal ambiguous
            if name not in directional_credits:
                outputs = output_context.get(name)
                if outputs is not None:
                    if timing_profiler is not None:
                        with timing_profiler.section("directional_credit_selection_seconds"):
                            directional_credits[name] = self._route_credit_edges(
                                brain,
                                active_edges,
                                outputs,
                                discover_upstream_from_candidates=prospective,
                                timing_profiler=timing_profiler,
                            )
                    else:
                        directional_credits[name] = self._route_credit_edges(
                            brain,
                            active_edges,
                            outputs,
                            discover_upstream_from_candidates=prospective,
                        )
                else:
                    directional_credits[name] = None
                if directional_credits[name] is not None:
                    ambiguous += directional_credits[name].ambiguous_path_edges_skipped
            return directional_credits[name]

        def reward_credit_for(name):
            if name not in reward_credits:
                outputs = output_context.get(name)
                reward_credits[name] = self._credit_edges(
                    brain, active_edges, outputs
                ) if outputs is not None else None
            return reward_credits[name]

        for name, direction in signal.nonzero_directions().items():
            credit = directional_credit_for(name)
            legacy_credit = (
                self._credit_edges(brain, active_edges, output_context[name])
                if attribution_observer is not None
                and prospective
                and name in output_context
                else credit
            )
            if credit is None or not len(credit.edges):
                per_channel[name] = 0
                if telemetry_observer is not None:
                    telemetry_observer(
                        name,
                        np.empty(0, dtype=np.int32),
                        np.empty(0, dtype=np.int8),
                        np.empty(0, dtype=np.float32),
                        np.empty(0, dtype=np.float32),
                        np.empty(0, dtype=np.float32),
                        float(direction),
                    )
                if attribution_observer is not None:
                    attribution_observer(
                        channel=name,
                        current_edge_indices=np.empty(0, dtype=np.int32),
                        current_hops=np.empty(0, dtype=np.int8),
                        legacy_edge_indices=(
                            np.asarray(legacy_credit.edges, dtype=np.int32).copy()
                            if legacy_credit is not None else np.empty(0, dtype=np.int32)
                        ),
                        legacy_hops=(
                            np.asarray(legacy_credit.hops, dtype=np.int8).copy()
                            if legacy_credit is not None else np.empty(0, dtype=np.int8)
                        ),
                        actual_deltas=np.empty(0, dtype=np.float32),
                        eligibility=np.empty(0, dtype=np.float32),
                        path_polarities=np.empty(0, dtype=np.float32),
                        requested_direction=float(direction),
                        brain=brain,
                    )
                continue
            edges = credit.edges
            factor = np.maximum(self.config.minimum_learning_factor, 1.0 - self.config.stability_protection * state.stability[edges])
            delta = self.config.learning_rate * float(direction) * credit.path_polarities * state.eligibility[edges] * factor * credit.weights
            if timing_profiler is not None:
                with timing_profiler.section("multiplier_update_seconds"):
                    old = state.multiplier[edges].copy()
                    state.multiplier[edges] = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
                    actual = state.multiplier[edges] - old
            else:
                old = state.multiplier[edges].copy()
                state.multiplier[edges] = np.clip(old + delta, state.config.min_multiplier, state.config.max_multiplier)
                actual = state.multiplier[edges] - old
            count = int(len(edges)); total += count; per_channel[name] = count
            channel_sum_abs_delta[name] = float(np.abs(actual).sum())
            changed_edges = np.unique(edges[np.abs(actual) > 0.0])
            channel_unique_edge_updates[name] = int(len(changed_edges))
            if materialize_edge_telemetry:
                channel_edge_indices[name] = tuple(int(edge) for edge in changed_edges)
            sum_abs += channel_sum_abs_delta[name]; changed_all.append(edges)
            if materialize_edge_telemetry:
                for edge, hop in zip(edges, credit.hops):
                    updated_edge_hops[int(edge)] = int(hop)
            for hop in np.unique(credit.hops):
                hops[int(hop)] = hops.get(int(hop), 0) + int((credit.hops == hop).sum())
            channel_hop_counts[name] = {
                int(hop): int((credit.hops == hop).sum())
                for hop in np.unique(credit.hops)
            }
            if telemetry_observer is not None:
                telemetry_observer(
                    name,
                    edges.copy(),
                    credit.hops.copy(),
                    actual.copy(),
                    state.eligibility[edges].copy(),
                    credit.path_polarities.copy(),
                    float(direction),
                )
            if attribution_observer is not None:
                attribution_observer(
                    channel=name,
                    current_edge_indices=edges.copy(),
                    current_hops=credit.hops.copy(),
                    legacy_edge_indices=np.asarray(legacy_credit.edges, dtype=np.int32).copy(),
                    legacy_hops=np.asarray(legacy_credit.hops, dtype=np.int8).copy(),
                    actual_deltas=actual.copy(),
                    eligibility=state.eligibility[edges].copy(),
                    path_polarities=credit.path_polarities.copy(),
                    requested_direction=float(direction),
                    brain=brain,
                )
            anatomical_sign = np.sign(brain.connectome.signed_synapse_counts[edges])
            excitatory += int((anatomical_sign > 0).sum())
            inhibitory += int((anatomical_sign < 0).sum())

        reinforced = []
        for name in signal.positive_reinforcements():
            # Reward consolidation remains on the legacy active-chain route;
            # the F.1B.3 switch is directional-learning-only.
            credit = reward_credit_for(name)
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

        unique_updated = (
            np.unique(np.concatenate(changed_all)).astype(np.int32, copy=False)
            if changed_all else np.empty(0, dtype=np.int32)
        )
        updated = unique_updated if materialize_edge_telemetry else np.empty(0, dtype=np.int32)
        if timing_profiler is not None:
            # Keep packaging visible in P.8 without including it in the
            # directional math timings.
            with timing_profiler.section("directional_telemetry_packaging_seconds"):
                result = DirectionalUpdate(
                    total, per_channel, sum_abs / total if total else 0.0, updated, hops,
                    excitatory, inhibitory, consolidated, int(len(unique_updated)), tuple(reinforced), ambiguous,
                    sum_abs, updated_edge_hops,
                    channel_sum_abs_delta,
                    channel_unique_edge_updates,
                    channel_hop_counts,
                    channel_edge_indices,
                    telemetry_level,
                )
            return result
        return DirectionalUpdate(
            total, per_channel, sum_abs / total if total else 0.0, updated, hops,
            excitatory, inhibitory, consolidated, int(len(unique_updated)), tuple(reinforced), ambiguous,
            sum_abs, updated_edge_hops,
            channel_sum_abs_delta,
            channel_unique_edge_updates,
            channel_hop_counts,
            channel_edge_indices,
            telemetry_level,
        )
