import unittest


class MushroomBodyCircuitTests(unittest.TestCase):
    def _connectome(self):
        import numpy as np
        from drosomath.flywire_real import FlyWireConnectome

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([100, 101, 102, 103, 104], dtype=np.int64),
            indptr=np.asarray([0, 1, 2, 3, 3, 3], dtype=np.int64),
            post_indices=np.asarray([1, 2, 3], dtype=np.int32),
            signed_synapse_counts=np.asarray([5.0, 7.0, 2.0], dtype=np.float32),
            outgoing_strength=np.asarray([5.0, 7.0, 2.0, 0.0, 0.0], dtype=np.float32),
        )
        connectome.metadata = {
            "type": np.asarray(["visual", "KC", "MBON", "DAN", "other"], dtype=object),
        }
        return connectome

    def test_roles_and_anatomical_scope(self) -> None:
        from drosomath.malecns.mushroom_body import MushroomBodyCircuit

        connectome = self._connectome()
        circuit = MushroomBodyCircuit.from_connectome(connectome)
        self.assertEqual(len(circuit.roles.kenyon_cells), 1)
        self.assertEqual(len(circuit.roles.mbons), 1)
        self.assertEqual(len(circuit.roles.dopamine_neurons), 1)
        self.assertEqual(int(circuit.eligible_edge_mask.sum()), 1)
        self.assertEqual(circuit.summary()["plasticity_scope"], "anatomical KC -> MBON edges only")

    def test_reward_changes_only_the_real_kc_to_mbon_edge(self) -> None:
        import numpy as np
        from drosomath.malecns.mushroom_body import MushroomBodyCircuit
        from drosomath.whole_brain import PlasticSparseFlyBrain, PlasticStateConfig, UsageRewardRule

        connectome = self._connectome()
        brain = PlasticSparseFlyBrain(
            connectome,
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0),
            seed=1,
        )
        circuit = MushroomBodyCircuit.from_connectome(connectome)
        circuit.attach(brain)
        circuit._brain._schedule_spike_outputs(np.asarray([1], dtype=np.int32))
        before = brain.plasticity.multiplier.copy()
        circuit.learn_from_reward(
            reward=1.0,
            rule=UsageRewardRule(learning_rate=0.5),
        )
        self.assertGreater(float(brain.plasticity.multiplier[1]), float(before[1]))
        self.assertAlmostEqual(float(brain.plasticity.multiplier[0]), float(before[0]))
        self.assertAlmostEqual(float(brain.plasticity.multiplier[2]), float(before[2]))


if __name__ == "__main__":
    unittest.main()
