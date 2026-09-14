from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Hashable, Iterable


@dataclass(slots=True)
class ModuleState:
    module_id: int
    neuron_ids: set[int] = field(default_factory=set)
    activity_ema: float = 0.0
    reward_ema: float = 0.0
    observations: int = 0


class AdaptiveModuleManager:
    """Role-free module accounting for emergent specialization.

    Modules receive numeric IDs only; no semantic labels such as math/language
    are assigned. Activity and reward statistics can later drive routing,
    rewiring and budget allocation. Optional context keys are opaque experiment
    identifiers used only to measure whether different modules specialize.
    """

    def __init__(
        self,
        neuron_ids: Iterable[int],
        *,
        module_count: int,
        activity_alpha: float = 0.05,
        reward_alpha: float = 0.05,
    ) -> None:
        ids = tuple(sorted(set(neuron_ids)))
        if not ids:
            raise ValueError("at least one neuron is required")
        if module_count < 1:
            raise ValueError("module_count must be >= 1")
        if module_count > len(ids):
            raise ValueError("module_count cannot exceed neuron count")
        if not 0.0 < activity_alpha <= 1.0:
            raise ValueError("activity_alpha must be in (0, 1]")
        if not 0.0 < reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be in (0, 1]")

        self.activity_alpha = activity_alpha
        self.reward_alpha = reward_alpha
        self.modules = {i: ModuleState(i) for i in range(module_count)}
        self._membership: dict[int, int] = {}
        self._context_activity: dict[Hashable, dict[int, float]] = {}

        # Deterministic, role-free initialization. It avoids encoding any task
        # semantics while keeping modules approximately balanced at startup.
        for index, neuron_id in enumerate(ids):
            module_id = index % module_count
            self.modules[module_id].neuron_ids.add(neuron_id)
            self._membership[neuron_id] = module_id

    def module_of(self, neuron_id: int) -> int:
        try:
            return self._membership[neuron_id]
        except KeyError as exc:
            raise KeyError(f"unknown neuron {neuron_id}") from exc

    def observe(
        self,
        fired: Iterable[int],
        *,
        reward: float | None = None,
        context: Hashable | None = None,
    ) -> None:
        counts = {module_id: 0 for module_id in self.modules}
        for neuron_id in fired:
            counts[self.module_of(neuron_id)] += 1

        for module_id, state in self.modules.items():
            size = max(1, len(state.neuron_ids))
            normalized_activity = counts[module_id] / size
            state.activity_ema += self.activity_alpha * (
                normalized_activity - state.activity_ema
            )
            if reward is not None and counts[module_id] > 0:
                state.reward_ema += self.reward_alpha * (reward - state.reward_ema)
            state.observations += 1

        if context is not None:
            context_scores = self._context_activity.setdefault(
                context,
                {module_id: 0.0 for module_id in self.modules},
            )
            for module_id, count in counts.items():
                size = max(1, len(self.modules[module_id].neuron_ids))
                value = count / size
                context_scores[module_id] += self.activity_alpha * (
                    value - context_scores[module_id]
                )

    def utility_scores(self, *, load_penalty: float = 0.1) -> dict[int, float]:
        """Score modules without assigning fixed functions.

        Reward contribution raises utility while a small activity penalty keeps
        one early-lucky module from monopolizing all future resources.
        """
        if load_penalty < 0.0:
            raise ValueError("load_penalty must be >= 0")
        return {
            module_id: state.reward_ema - load_penalty * state.activity_ema
            for module_id, state in self.modules.items()
        }

    def allocate_budget(
        self,
        total_budget: int,
        *,
        minimum_per_module: int = 0,
        demand_floor: float = 1e-6,
    ) -> dict[int, int]:
        """Allocate a fixed resource budget from learned demand, never creating more."""
        module_count = len(self.modules)
        if total_budget < module_count * minimum_per_module:
            raise ValueError("total_budget is below requested module minimums")
        if minimum_per_module < 0:
            raise ValueError("minimum_per_module must be >= 0")

        base = {module_id: minimum_per_module for module_id in self.modules}
        remaining = total_budget - module_count * minimum_per_module
        if remaining == 0:
            return base

        demand = {
            module_id: max(
                demand_floor,
                state.activity_ema * (1.0 + max(0.0, state.reward_ema)),
            )
            for module_id, state in self.modules.items()
        }
        total_demand = sum(demand.values())
        raw = {
            module_id: remaining * demand[module_id] / total_demand
            for module_id in self.modules
        }
        floors = {module_id: int(value) for module_id, value in raw.items()}
        for module_id, value in floors.items():
            base[module_id] += value

        leftovers = total_budget - sum(base.values())
        order = sorted(
            self.modules,
            key=lambda module_id: (-(raw[module_id] - floors[module_id]), module_id),
        )
        for module_id in order[:leftovers]:
            base[module_id] += 1
        return base

    def context_specialization(self, context: Hashable) -> float:
        """Return normalized concentration in [0, 1]; higher means more specialized."""
        scores = self._context_activity.get(context)
        if not scores:
            return 0.0
        values = [max(0.0, value) for value in scores.values()]
        total = sum(values)
        if total <= 0.0 or len(values) <= 1:
            return 0.0
        probabilities = [value / total for value in values if value > 0.0]
        entropy = -sum(p * log(p) for p in probabilities)
        max_entropy = log(len(values))
        return max(0.0, min(1.0, 1.0 - entropy / max_entropy))
