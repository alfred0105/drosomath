"""Conservative automatic allocation of fixed plastic capacity."""

from __future__ import annotations

from dataclasses import dataclass

from .plasticity_budget import PlasticityBudgetManager
from .plasticity_need import PlasticityNeedConfig, PlasticityNeedTracker


@dataclass(frozen=True, slots=True)
class PlasticityBudgetAdaptationConfig:
    enabled: bool = False
    max_promotions_per_event: int = 1
    need: PlasticityNeedConfig = PlasticityNeedConfig()


class PlasticityBudgetAdaptation:
    """Task-independent controller for missing-plastic-route evidence.

    It only observes failures with zero generic directional updates. The
    promoted edge is therefore not modified until a later activation records
    ordinary eligibility.
    """

    def __init__(self, *, config: PlasticityBudgetAdaptationConfig | None = None,
                 route_controller=None, budget_manager=None) -> None:
        self.config = config or PlasticityBudgetAdaptationConfig()
        if self.config.max_promotions_per_event < 1:
            raise ValueError("max_promotions_per_event must be >= 1")
        self.need_tracker = PlasticityNeedTracker(self.config.need)
        self.route_controller = route_controller
        self.budget_manager = budget_manager or PlasticityBudgetManager()

    @staticmethod
    def _edge_updates(update) -> int:
        if update is None:
            return 0
        if isinstance(update, dict):
            return int(update.get("edge_updates", 0))
        return int(getattr(update, "edge_updates", 0))

    @staticmethod
    def _directional_values(signal):
        directions = signal.nonzero_directions()
        return directions

    def observe_directional_failure(
        self,
        *,
        brain,
        signal,
        output_context,
        directional_update,
        legacy_rescue_used: bool = False,
    ) -> dict[str, object]:
        """Observe one generic event and optionally reallocate 1–N edges."""
        np = brain.np
        before = len(brain.plasticity.plastic_indices)
        self.need_tracker.advance(success=bool(signal.success))
        result: dict[str, object] = {
            "structural_need_candidates_observed": 0,
            "tracked_need_candidates": self.need_tracker.tracked_count,
            "ready_need_candidates": 0,
            "reallocation_triggered": False,
            "promoted_edges_count": 0,
            "retired_edges_count": 0,
            "promoted_edges": [],
            "retired_edges": [],
            "plastic_budget_before": before,
            "plastic_budget_after": before,
            "budget_delta": 0,
            "max_candidate_need": 0.0,
            "mean_promoted_need": 0.0,
            "protected_donors_skipped": 0,
            "active_donors_excluded": 0,
            "legacy_rescue_used": bool(legacy_rescue_used),
        }
        if signal.success:
            result["tracked_need_candidates"] = self.need_tracker.tracked_count
            return result
        # D.2 is deliberately a missing-route escape hatch, not a second
        # optimizer for ordinary imperfect learning.
        if not self.config.enabled or self._edge_updates(directional_update) != 0:
            result["tracked_need_candidates"] = self.need_tracker.tracked_count
            return result
        directions = self._directional_values(signal)
        if not directions or self.route_controller is None:
            return result

        candidate_edges = []
        candidate_weights = []
        for name, direction in directions.items():
            outputs = output_context.get(name)
            if outputs is None:
                continue
            credit = self.route_controller.structural_credit_edges(brain, outputs)
            if not len(credit.edges):
                continue
            # A candidate is useful when changing its multiplier in the
            # requested direction changes the output in the right direction.
            required = np.sign(float(direction) * credit.path_polarities)
            state = brain.plasticity
            movable = np.where(
                ((required > 0.0) & (state.multiplier[credit.edges] < state.config.max_multiplier - 1e-7))
                | ((required < 0.0) & (state.multiplier[credit.edges] > state.config.min_multiplier + 1e-7))
            )[0]
            if len(movable):
                candidate_edges.extend(int(edge) for edge in credit.edges[movable])
                candidate_weights.extend(
                    abs(float(direction)) * float(weight)
                    for weight in credit.weights[movable]
                )
        if candidate_edges:
            self.need_tracker.observe(candidate_edges, candidate_weights)
        result["structural_need_candidates_observed"] = len(set(candidate_edges))
        result["tracked_need_candidates"] = self.need_tracker.tracked_count
        records = self.need_tracker.records()
        result["max_candidate_need"] = max((record.need_score for record in records), default=0.0)
        ready_edges, ready_scores = self.need_tracker.ready_candidates()
        result["ready_need_candidates"] = int(len(ready_edges))
        if not len(ready_edges):
            return result

        # Never retire the route currently active in this trial. This includes
        # both ordinary eligible edges and the generic update's explicit set.
        protected = set(int(edge) for edge in self.route_controller._active_edges(brain))
        updated = getattr(directional_update, "updated_edge_indices", ())
        if isinstance(directional_update, dict):
            updated = directional_update.get("updated_edge_indices", ())
        protected.update(int(edge) for edge in updated)
        count = min(self.config.max_promotions_per_event, len(ready_edges))
        try:
            allocation = self.budget_manager.reallocate(
                brain.plasticity,
                ready_edges,
                need_scores=ready_scores,
                count=count,
                exclude_retirement_edges=protected,
            )
        except ValueError:
            # A conservative automatic feature must not destabilize a run if
            # the current budget has no safe donor. Evidence remains tracked.
            result["active_donors_excluded"] = len(protected)
            return result
        promoted = allocation.get("promoted_edges", allocation.get("requested_edges", ()))
        retired = allocation.get("retired_edges", ())
        self.need_tracker.discard(promoted)
        result.update({
            "structural_need_candidates_observed": len(set(candidate_edges)),
            "tracked_need_candidates": self.need_tracker.tracked_count,
            "reallocation_triggered": True,
            "promoted_edges_count": int(len(promoted)),
            "retired_edges_count": int(len(retired)),
            "promoted_edges": [int(edge) for edge in promoted],
            "retired_edges": [int(edge) for edge in retired],
            "plastic_budget_after": len(brain.plasticity.plastic_indices),
            "budget_delta": len(brain.plasticity.plastic_indices) - before,
            "mean_promoted_need": float(np.mean(allocation.get("need_scores", []))) if len(promoted) else 0.0,
            "protected_donors_skipped": int(allocation.get("protected_donors_skipped", 0)),
            "active_donors_excluded": int(allocation.get("active_donors_excluded", 0)),
        })
        return result
