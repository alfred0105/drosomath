import unittest

from drosomath.core import AdaptiveModuleManager


class AdaptiveModuleManagerTests(unittest.TestCase):
    def test_initial_partition_is_balanced_without_semantic_roles(self) -> None:
        manager = AdaptiveModuleManager(range(8), module_count=2)
        sizes = sorted(len(state.neuron_ids) for state in manager.modules.values())
        self.assertEqual(sizes, [4, 4])

    def test_context_specialization_rises_when_one_module_dominates(self) -> None:
        manager = AdaptiveModuleManager(range(4), module_count=2, activity_alpha=0.5)
        dominant_neurons = sorted(manager.modules[0].neuron_ids)
        for _ in range(8):
            manager.observe(dominant_neurons, reward=1.0, context="opaque-context")

        self.assertGreater(manager.context_specialization("opaque-context"), 0.5)

    def test_budget_is_fixed_and_shifts_toward_demand(self) -> None:
        manager = AdaptiveModuleManager(range(4), module_count=2, activity_alpha=0.5)
        active = sorted(manager.modules[0].neuron_ids)
        for _ in range(5):
            manager.observe(active, reward=1.0)

        allocation = manager.allocate_budget(100, minimum_per_module=10)
        self.assertEqual(sum(allocation.values()), 100)
        self.assertGreater(allocation[0], allocation[1])


if __name__ == "__main__":
    unittest.main()
