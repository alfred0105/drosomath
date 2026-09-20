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
        self._promoted_route_status: dict[int, dict[str, bool]] = {}
        self._allocation_state: dict[int, str] = {}
        self._unique_promoted: set[int] = set()
        self._unique_retired: set[int] = set()
        self._promoted_then_retired = 0
        self._retired_then_promoted = 0
        self._reallocation_events = 0
        self._total_promoted = 0
        self._total_retired = 0
        self._reallocation_attempts = 0
        self._successful_reallocations = 0
        self._failed_no_safe_donor = 0
        self._protected_edges_retired = 0
        self._donor_stability: list[float] = []
        self._donor_usage: list[float] = []
        self._donor_multiplier_delta: list[float] = []
        self._promoted_later_activated = 0
        self._promoted_later_updated = 0

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

    def _observe_promoted_routes(self, brain, directional_update) -> tuple[int, int]:
        """Consume one-time later-activation/later-update observations."""
        updated = getattr(directional_update, "updated_edge_indices", ())
        if isinstance(directional_update, dict):
            updated = directional_update.get("updated_edge_indices", ())
        updated_set = {int(edge) for edge in updated}
        activated = updated_count = 0
        for edge, status in self._promoted_route_status.items():
            if not status["activated"] and (
                float(brain.plasticity.eligibility[edge]) > 0.0 or edge in updated_set
            ):
                status["activated"] = True
                activated += 1
                self._promoted_later_activated += 1
            if not status["updated"] and edge in updated_set:
                status["updated"] = True
                updated_count += 1
                self._promoted_later_updated += 1
        return activated, updated_count

    def telemetry(self) -> dict[str, object]:
        total_transitions = self._promoted_then_retired + self._retired_then_promoted
        donor_count = len(self._donor_stability)
        return {
            "need_candidates_observed": int(getattr(self, "_need_candidates_observed", 0)),
            "candidates_reaching_minimum_observations": int(getattr(self, "_minimum_observation_candidates", 0)),
            "candidates_reaching_promotion_threshold": int(getattr(self, "_threshold_candidates", 0)),
            "reallocation_attempts": int(self._reallocation_attempts),
            "successful_reallocations": int(self._successful_reallocations),
            "failed_reallocations_no_safe_donor": int(self._failed_no_safe_donor),
            "reallocation_events": int(self._reallocation_events),
            "reallocation_event_count": int(self._reallocation_events),
            "total_promoted_edges": int(self._total_promoted),
            "total_retired_edges": int(self._total_retired),
            "unique_promoted_edges": int(len(self._unique_promoted)),
            "unique_retired_edges": int(len(self._unique_retired)),
            "promoted_edges_later_activated": int(self._promoted_later_activated),
            "promoted_edges_later_updated": int(self._promoted_later_updated),
            "promotion_to_learning_rate": (
                self._promoted_later_updated / max(1, len(self._unique_promoted))
            ),
            "promoted_then_retired": int(self._promoted_then_retired),
            "retired_then_promoted": int(self._retired_then_promoted),
            "repeated_exchange_count": int(self._reallocation_events),
            "allocation_churn_rate": total_transitions / max(1, len(self._unique_promoted) + len(self._unique_retired)),
            "protected_edges_retired": int(self._protected_edges_retired),
            "mean_donor_stability": sum(self._donor_stability) / donor_count if donor_count else 0.0,
            "mean_donor_usage": sum(self._donor_usage) / donor_count if donor_count else 0.0,
            "mean_donor_abs_multiplier_delta": sum(self._donor_multiplier_delta) / donor_count if donor_count else 0.0,
            "promotion_observation_counts": list(getattr(self, "_promotion_observation_counts", [])),
            "promotion_need_scores": list(getattr(self, "_promotion_need_scores", [])),
        }

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
        later_activated, later_updated = self._observe_promoted_routes(brain, directional_update)
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
            "promoted_edges_later_activated": later_activated,
            "promoted_edges_later_updated": later_updated,
            "reallocation_attempts": 0,
            "successful_reallocations": 0,
            "failed_reallocations_no_safe_donor": 0,
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
        self._need_candidates_observed = getattr(self, "_need_candidates_observed", 0) + len(set(candidate_edges))
        result["tracked_need_candidates"] = self.need_tracker.tracked_count
        records = self.need_tracker.records()
        result["max_candidate_need"] = max((record.need_score for record in records), default=0.0)
        self._minimum_observation_candidates = getattr(self, "_minimum_observation_candidates", 0) + sum(
            record.observation_count >= self.config.need.minimum_observations for record in records
        )
        self._threshold_candidates = getattr(self, "_threshold_candidates", 0) + sum(
            record.need_score >= self.config.need.promotion_threshold for record in records
        )
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
        self._reallocation_attempts += 1
        result["reallocation_attempts"] = 1
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
            self._failed_no_safe_donor += 1
            result["failed_reallocations_no_safe_donor"] = 1
            return result
        promoted = allocation.get("promoted_edges", allocation.get("requested_edges", ()))
        retired = allocation.get("retired_edges", ())
        self._reallocation_events += 1
        self._successful_reallocations += 1
        self._total_promoted += len(promoted)
        self._total_retired += len(retired)
        result["successful_reallocations"] = 1
        self._unique_promoted.update(int(edge) for edge in promoted)
        self._unique_retired.update(int(edge) for edge in retired)
        self._promotion_observation_counts = getattr(self, "_promotion_observation_counts", []) + [
            int(next(record.observation_count for record in records if record.edge_index == int(edge)))
            for edge in promoted if any(record.edge_index == int(edge) for record in records)
        ]
        self._promotion_need_scores = getattr(self, "_promotion_need_scores", []) + [
            float(score) for score in allocation.get("need_scores", [])
        ]
        state = brain.plasticity
        for edge in retired:
            edge = int(edge)
            previous = self._allocation_state.get(edge)
            if previous == "promoted":
                self._promoted_then_retired += 1
            self._allocation_state[edge] = "retired"
            self._donor_stability.append(float(state.stability[edge]))
            self._donor_usage.append(float(state.usage_ema[edge]))
            self._donor_multiplier_delta.append(abs(float(state.multiplier[edge]) - self.budget_manager.config.baseline_multiplier))
            if float(state.stability[edge]) >= self.budget_manager.config.protected_stability:
                self._protected_edges_retired += 1
        for edge in promoted:
            edge = int(edge)
            if self._allocation_state.get(edge) == "retired":
                self._retired_then_promoted += 1
            self._allocation_state[edge] = "promoted"
            self._promoted_route_status.setdefault(edge, {"activated": False, "updated": False})
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
