"""Explicit, fixed-budget reallocation among existing anatomical edges."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlasticityBudgetConfig:
    protected_stability: float = 0.75
    baseline_multiplier: float = 1.0


class PlasticityBudgetManager:
    """Separate generic allocation policy from mutable plastic state."""

    def __init__(self, config: PlasticityBudgetConfig | None = None) -> None:
        self.config = config or PlasticityBudgetConfig()

    def choose_retirement_candidates(self, state, count: int, *, exclude=()):
        if count < 0:
            raise ValueError("count must be >= 0")
        np = state.np
        excluded = set(int(x) for x in exclude)
        candidates = [
            int(edge) for edge in state.plastic_indices
            if int(edge) not in excluded
            and float(state.stability[int(edge)]) < self.config.protected_stability
        ]
        if not candidates or count == 0:
            return np.empty(0, dtype=np.int32)
        # Lower score means more replaceable. Stable/high-usage/strongly
        # modified memories naturally rank later and are protected by the
        # threshold above.
        candidates.sort(key=lambda edge: (
            float(state.usage_ema[edge])
            + float(state.stability[edge])
            + abs(float(state.multiplier[edge]) - self.config.baseline_multiplier),
            edge,
        ))
        return np.asarray(candidates[:count], dtype=np.int32)

    def reallocate(
        self,
        state,
        requested_edges,
        *,
        need_scores=None,
        count: int | None = None,
        exclude_retirement_edges=(),
    ) -> dict[str, object]:
        """Promote requested frozen edges while retiring safe donors atomically."""
        np = state.np
        requested = np.asarray(requested_edges, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("requested_edges must be one-dimensional")
        if len(requested) and (int(requested.min()) < 0 or int(requested.max()) >= state.edge_count):
            raise IndexError("requested edge index out of range")
        if len(requested) != len(np.unique(requested)):
            raise ValueError("requested_edges contains duplicates")
        if need_scores is None:
            scores = np.ones(len(requested), dtype=np.float32)
        else:
            scores = np.asarray(need_scores, dtype=np.float32)
            if scores.shape != requested.shape:
                raise ValueError("need_scores must match requested_edges")
        frozen = [(int(edge), float(score)) for edge, score in zip(requested, scores) if not state.plastic_mask[int(edge)]]
        frozen.sort(key=lambda item: (-item[1], item[0]))
        target_count = len(frozen) if count is None else int(count)
        if target_count < 0:
            raise ValueError("count must be >= 0")
        selected = frozen[:target_count]
        promote = np.asarray([edge for edge, _ in selected], dtype=np.int32)
        excluded = set(int(edge) for edge in exclude_retirement_edges)
        excluded.update(int(edge) for edge in promote)
        promote_set = set(int(edge) for edge in promote)
        protected_skipped = sum(
            1 for edge in state.plastic_indices
            if int(edge) not in promote_set
            and float(state.stability[int(edge)]) >= self.config.protected_stability
        )
        retire = self.choose_retirement_candidates(state, len(promote), exclude=excluded)
        if len(retire) != len(promote):
            raise ValueError("not enough safe plastic donor edges for requested promotion")
        result = state.exchange_plastic_edges(
            promote,
            retire,
            protected_stability=self.config.protected_stability,
        )
        result.update({
            "requested_edges": promote.copy(),
            "need_scores": np.asarray([score for _, score in selected], dtype=np.float32),
            "active_donors_excluded": int(len(set(int(edge) for edge in exclude_retirement_edges) & set(int(edge) for edge in state.plastic_indices))),
            "protected_donors_skipped": int(protected_skipped),
        })
        return result
