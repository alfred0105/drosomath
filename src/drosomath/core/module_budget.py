from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .modules import AdaptiveModuleManager
from .plasticity import PlasticityTracker
from .structural import StructuralPlasticityConfig
from .synapse import SynapseState


@dataclass(frozen=True, slots=True)
class ModuleBudgetPlan:
    desired: dict[int, int]
    current: dict[int, int]

    @property
    def surplus(self) -> dict[int, int]:
        return {
            module_id: max(0, self.current.get(module_id, 0) - target)
            for module_id, target in self.desired.items()
        }

    @property
    def deficit(self) -> dict[int, int]:
        return {
            module_id: max(0, target - self.current.get(module_id, 0))
            for module_id, target in self.desired.items()
        }


@dataclass(frozen=True, slots=True)
class ModuleBudgetRewireResult:
    pruned: tuple[tuple[int, int], ...]
    regrown: tuple[tuple[int, int], ...]
    before: ModuleBudgetPlan
    after: ModuleBudgetPlan

    @property
    def changed(self) -> int:
        return len(self.pruned)


class ModuleBudgetReallocator:
    """Move a fixed synapse budget toward modules with learned demand.

    A synapse is charged to the module containing its postsynaptic neuron. This
    gives every edge exactly one accounting owner and lets resources move between
    role-free modules without changing the total synapse count.
    """

    def __init__(
        self,
        modules: AdaptiveModuleManager,
        tracker: PlasticityTracker,
        *,
        structural_config: StructuralPlasticityConfig | None = None,
        minimum_per_module: int = 0,
    ) -> None:
        if minimum_per_module < 0:
            raise ValueError("minimum_per_module must be >= 0")
        self.modules = modules
        self.tracker = tracker
        self.config = structural_config or StructuralPlasticityConfig()
        self.minimum_per_module = minimum_per_module

    def plan(self) -> ModuleBudgetPlan:
        total = self.tracker.synapse_count
        desired = self.modules.allocate_budget(
            total,
            minimum_per_module=self.minimum_per_module,
        )
        current = {module_id: 0 for module_id in self.modules.modules}
        for synapse in self.tracker.synapses:
            if not synapse.alive:
                continue
            module_id = self.modules.module_of(synapse.post_id)
            current[module_id] += 1
        return ModuleBudgetPlan(desired=desired, current=current)

    def rewire(
        self,
        *,
        step: int,
        candidate_pairs: Iterable[tuple[int, int]],
        max_moves: int | None = None,
    ) -> ModuleBudgetRewireResult:
        if step < 0:
            raise ValueError("step must be >= 0")
        if max_moves is not None and max_moves < 0:
            raise ValueError("max_moves must be >= 0")

        before = self.plan()
        surplus = before.surplus.copy()
        deficit = before.deficit.copy()
        possible_moves = min(sum(surplus.values()), sum(deficit.values()))
        move_limit = self.config.max_rewire_per_cycle
        if max_moves is not None:
            move_limit = min(move_limit, max_moves)
        move_limit = min(move_limit, possible_moves)
        if move_limit <= 0:
            return ModuleBudgetRewireResult((), (), before, before)

        existing = {
            (synapse.pre_id, synapse.post_id)
            for synapse in self.tracker.synapses
            if synapse.alive
        }
        selected_regrow: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        remaining_deficit = deficit.copy()

        for pre_id, post_id in candidate_pairs:
            if len(selected_regrow) >= move_limit:
                break
            key = (pre_id, post_id)
            if pre_id == post_id or key in existing or key in seen:
                continue
            try:
                target_module = self.modules.module_of(post_id)
                self.modules.module_of(pre_id)
            except KeyError:
                continue
            if remaining_deficit.get(target_module, 0) <= 0:
                continue
            selected_regrow.append(key)
            seen.add(key)
            remaining_deficit[target_module] -= 1

        if not selected_regrow:
            return ModuleBudgetRewireResult((), (), before, before)

        eligible = [
            synapse
            for synapse in self.tracker.synapses
            if self._eligible_for_prune(synapse, step)
            and surplus.get(self.modules.module_of(synapse.post_id), 0) > 0
        ]
        eligible.sort(key=self._prune_sort_key)

        selected_prune: list[SynapseState] = []
        remaining_surplus = surplus.copy()
        for synapse in eligible:
            if len(selected_prune) >= len(selected_regrow):
                break
            module_id = self.modules.module_of(synapse.post_id)
            if remaining_surplus.get(module_id, 0) <= 0:
                continue
            selected_prune.append(synapse)
            remaining_surplus[module_id] -= 1

        selected_regrow = selected_regrow[: len(selected_prune)]
        if not selected_prune:
            return ModuleBudgetRewireResult((), (), before, before)

        pruned: list[tuple[int, int]] = []
        for synapse in selected_prune:
            key = (synapse.pre_id, synapse.post_id)
            removed = self.tracker.unregister_synapse(
                pre_id=synapse.pre_id,
                post_id=synapse.post_id,
            )
            if removed is None:
                continue
            removed.prune()
            pruned.append(key)

        selected_regrow = selected_regrow[: len(pruned)]
        for pre_id, post_id in selected_regrow:
            self.tracker.register_synapse(
                SynapseState(
                    pre_id=pre_id,
                    post_id=post_id,
                    weight=self.config.regrow_weight,
                )
            )

        after = self.plan()
        return ModuleBudgetRewireResult(
            pruned=tuple(pruned),
            regrown=tuple(selected_regrow),
            before=before,
            after=after,
        )

    def _eligible_for_prune(self, synapse: SynapseState, step: int) -> bool:
        if not synapse.alive:
            return False
        if synapse.stability >= self.config.protected_stability:
            return False
        if synapse.reward_ema > self.config.reward_threshold:
            return False
        if synapse.last_used_step < 0:
            return True
        return step - synapse.last_used_step >= self.config.stale_steps

    @staticmethod
    def _prune_sort_key(synapse: SynapseState) -> tuple[float, int, float, int]:
        return (
            synapse.reward_ema,
            synapse.usage_count,
            synapse.weight,
            synapse.last_used_step,
        )
