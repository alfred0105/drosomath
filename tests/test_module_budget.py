import unittest

from drosomath.core import (
    AdaptiveModuleManager,
    ModuleBudgetReallocator,
    PlasticityTracker,
    StructuralPlasticityConfig,
    SynapseState,
)


class ModuleBudgetReallocatorTests(unittest.TestCase):
    def _modules(self) -> AdaptiveModuleManager:
        modules = AdaptiveModuleManager([0, 1, 2, 3], module_count=2)
        # Role-free learned demand: module 1 currently contributes more useful
        # activity than module 0, so it should receive more of the fixed budget.
        modules.modules[0].activity_ema = 0.0
        modules.modules[0].reward_ema = 0.0
        modules.modules[1].activity_ema = 1.0
        modules.modules[1].reward_ema = 1.0
        return modules

    def _tracker(self) -> PlasticityTracker:
        # Module membership is round-robin: module 0={0,2}, module 1={1,3}.
        # Start with all four postsynaptic endpoints in module 0.
        return PlasticityTracker(
            [
                SynapseState(1, 0, 0.1, reward_ema=-0.2),
                SynapseState(3, 0, 0.1, reward_ema=-0.2),
                SynapseState(1, 2, 0.1, reward_ema=-0.2),
                SynapseState(3, 2, 0.1, reward_ema=-0.2),
            ]
        )

    def _reallocator(
        self,
        tracker: PlasticityTracker,
        modules: AdaptiveModuleManager,
        *,
        max_rewire: int = 2,
    ) -> ModuleBudgetReallocator:
        return ModuleBudgetReallocator(
            modules,
            tracker,
            structural_config=StructuralPlasticityConfig(
                min_age_cycles=1,
                stale_steps=0,
                reward_threshold=0.0,
                protected_stability=0.8,
                max_rewire_per_cycle=max_rewire,
                regrow_weight=0.05,
            ),
        )

    def test_plan_preserves_total_budget(self) -> None:
        tracker = self._tracker()
        reallocator = self._reallocator(tracker, self._modules())

        plan = reallocator.plan()

        self.assertEqual(sum(plan.desired.values()), tracker.synapse_count)
        self.assertEqual(sum(plan.current.values()), tracker.synapse_count)
        self.assertGreater(plan.deficit[1], 0)
        self.assertGreater(plan.surplus[0], 0)

    def test_rewire_moves_budget_toward_high_demand_module(self) -> None:
        tracker = self._tracker()
        modules = self._modules()
        reallocator = self._reallocator(tracker, modules, max_rewire=2)
        candidates = [(0, 1), (2, 1), (0, 3), (2, 3)]
        before_count = tracker.synapse_count

        result = reallocator.rewire(
            step=10,
            candidate_pairs=candidates,
        )

        self.assertEqual(result.changed, 2)
        self.assertEqual(tracker.synapse_count, before_count)
        self.assertLess(result.after.current[0], result.before.current[0])
        self.assertGreater(result.after.current[1], result.before.current[1])

    def test_stable_memory_synapse_is_not_reallocated(self) -> None:
        protected = SynapseState(
            1,
            0,
            0.1,
            stability=0.95,
            reward_ema=-1.0,
        )
        movable = SynapseState(3, 0, 0.1, reward_ema=-1.0)
        tracker = PlasticityTracker([protected, movable])
        modules = self._modules()
        reallocator = self._reallocator(tracker, modules, max_rewire=2)

        result = reallocator.rewire(
            step=10,
            candidate_pairs=[(0, 1), (2, 3)],
        )

        self.assertEqual(result.changed, 1)
        self.assertTrue(tracker.has_synapse(pre_id=1, post_id=0))
        self.assertFalse(protected.alive is False)
        self.assertFalse(movable.alive)

    def test_no_deficit_means_no_rewire(self) -> None:
        modules = AdaptiveModuleManager([0, 1], module_count=1)
        tracker = PlasticityTracker([SynapseState(0, 1, 0.1, reward_ema=-1.0)])
        reallocator = ModuleBudgetReallocator(
            modules,
            tracker,
            structural_config=StructuralPlasticityConfig(
                min_age_cycles=1,
                stale_steps=0,
                max_rewire_per_cycle=4,
            ),
        )

        result = reallocator.rewire(step=10, candidate_pairs=[(1, 0)])

        self.assertEqual(result.changed, 0)
        self.assertEqual(tracker.synapse_count, 1)


if __name__ == "__main__":
    unittest.main()
