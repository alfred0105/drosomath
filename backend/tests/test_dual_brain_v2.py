from __future__ import annotations

import unittest

import numpy as np

from app.dual_brain_v2 import DualBrainV2, DualBrainV2Config


class DualBrainV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DualBrainV2Config(
            seed=7,
            neurons_per_module=24,
            active_neurons_per_module=6,
            token_dim=48,
            token_active=8,
            local_synapses_per_module=80,
            inter_module_synapses_per_brain=100,
            bridge_synapses=40,
            rewire_interval=4,
            rewire_fraction=0.10,
        )
        self.model = DualBrainV2(self.config)

    def test_fixed_synapse_budget(self) -> None:
        before = self.model.resource_report()["synapse_budget"]
        for i in range(12):
            value = i % 9
            self.model.step(
                [str(value), "NEXT"],
                value + 1,
                active_outputs=10,
                learn=True,
            )
        after = self.model.resource_report()["synapse_budget"]
        self.assertEqual(before, after)

    def test_local_edges_stay_inside_module(self) -> None:
        n = self.config.neurons_per_module
        for module, bank in enumerate(self.model.local_banks):
            self.assertTrue(np.all(bank.src // n == module))
            self.assertTrue(np.all(bank.dst // n == module))

    def test_inter_edges_cross_modules_but_not_brains(self) -> None:
        c = self.config
        n = c.neurons_per_module
        for brain, bank in enumerate(self.model.inter_banks):
            src_module = bank.src // n
            dst_module = bank.dst // n
            lo = brain * c.modules_per_brain
            hi = lo + c.modules_per_brain
            self.assertTrue(
                np.all((src_module >= lo) & (src_module < hi))
            )
            self.assertTrue(
                np.all((dst_module >= lo) & (dst_module < hi))
            )
            self.assertTrue(np.all(src_module != dst_module))

    def test_bridge_edges_always_cross_brains(self) -> None:
        c = self.config
        neurons_per_brain = (
            c.modules_per_brain * c.neurons_per_module
        )
        bank = self.model.bridge_bank
        self.assertTrue(
            np.all(
                bank.src // neurons_per_brain
                != bank.dst // neurons_per_brain
            )
        )
        for _ in range(8):
            self.model.step(
                ["3", "NEXT"],
                4,
                active_outputs=10,
                learn=True,
            )
        self.assertTrue(
            np.all(
                bank.src // neurons_per_brain
                != bank.dst // neurons_per_brain
            )
        )

    def test_task_name_is_telemetry_only(self) -> None:
        a = DualBrainV2(self.config)
        b = DualBrainV2(self.config)
        for _ in range(10):
            out_a = a.step(
                ["2", "NEXT"],
                3,
                active_outputs=10,
                learn=True,
                task_name="math",
            )
            out_b = b.step(
                ["2", "NEXT"],
                3,
                active_outputs=10,
                learn=True,
                task_name=None,
            )
            self.assertEqual(out_a["choice"], out_b["choice"])
        np.testing.assert_allclose(a.w_router, b.w_router)
        np.testing.assert_allclose(a.w_output, b.w_output)

    def test_topology_snapshot_is_visualization_ready(self) -> None:
        snapshot = self.model.topology_snapshot(
            max_edges_per_bank=5
        )
        self.assertEqual(
            len(snapshot["modules"]),
            self.config.total_modules,
        )
        self.assertEqual(
            snapshot["resources"]["bridge_budget"],
            self.config.bridge_synapses,
        )
        self.assertTrue(snapshot["banks"])
        self.assertTrue(
            all(
                len(bank["edges"]) <= 5
                for bank in snapshot["banks"]
            )
        )


if __name__ == "__main__":
    unittest.main()
