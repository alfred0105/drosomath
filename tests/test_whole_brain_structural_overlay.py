import tempfile
import unittest
from pathlib import Path


class WholeBrainStructuralOverlayTests(unittest.TestCase):
    def _connectome(self):
        import numpy as np
        from drosomath.flywire_real import FlyWireConnectome

        # 0 -> {1,2}; 1 -> 3; 2 -> 4; 3 -> 4.
        # Two-hop closure can therefore grow 0 -> 3 or 0 -> 4 while keeping
        # the immutable anatomical arrays unchanged.
        return FlyWireConnectome(
            flywire_ids=np.asarray([100, 101, 102, 103, 104], dtype=np.int64),
            indptr=np.asarray([0, 2, 3, 4, 5, 5], dtype=np.int64),
            post_indices=np.asarray([1, 2, 3, 4, 4], dtype=np.int32),
            signed_synapse_counts=np.asarray([5, 4, 4, 4, 3], dtype=np.float32),
            outgoing_strength=np.asarray([9, 4, 4, 3, 0], dtype=np.float32),
        )

    def _brain(self):
        from drosomath.whole_brain import PlasticSparseFlyBrain, PlasticStateConfig

        return PlasticSparseFlyBrain(
            self._connectome(),
            plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=3),
            seed=3,
        )

    def _configure(self, brain):
        from drosomath.whole_brain import StructuralOverlayConfig

        return brain.configure_structural_plasticity(
            StructuralOverlayConfig(
                max_edges=4,
                rewire_per_cycle=2,
                candidate_pool_size=5,
                fanout_per_hop=2,
                min_age_cycles=1,
            )
        )

    def _seed_activity(self, overlay):
        import numpy as np

        overlay.record_firing(np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int32))
        overlay.learn_from_reward(1.0)

    def test_rewire_silences_donor_and_grows_two_hop_edge(self):
        brain = self._brain()
        overlay = self._configure(brain)
        original_posts = brain.connectome.post_indices.copy()
        original_edge_count = brain.connectome.edge_count
        self._seed_activity(overlay)

        result = overlay.rewire(cycle_label="test")

        self.assertGreater(result["added"], 0)
        self.assertEqual(brain.connectome.edge_count, original_edge_count)
        self.assertTrue((brain.connectome.post_indices == original_posts).all())
        self.assertEqual(overlay.edge_count, result["added"])
        donors = overlay.donor_edge[overlay.active]
        self.assertTrue(
            (brain.plasticity.multiplier[donors] == brain.plasticity.config.min_multiplier).all()
        )
        pairs = {
            (int(overlay.pre_index[s]), int(overlay.post_index[s]))
            for s in overlay.np.flatnonzero(overlay.active)
        }
        self.assertTrue(pairs & {(0, 3), (0, 4)})

    def test_stable_learned_edge_is_not_replaced(self):
        brain = self._brain()
        overlay = self._configure(brain)
        self._seed_activity(overlay)
        overlay.rewire(cycle_label="first")
        slots = overlay.np.flatnonzero(overlay.active)
        self.assertGreater(len(slots), 0)
        protected = int(slots[0])
        old_pair = (int(overlay.pre_index[protected]), int(overlay.post_index[protected]))
        overlay.stability[protected] = 0.95
        overlay.age_cycles[protected] = 5

        self._seed_activity(overlay)
        overlay.rewire(cycle_label="second")
        self.assertEqual(
            old_pair,
            (int(overlay.pre_index[protected]), int(overlay.post_index[protected])),
        )

    def test_checkpoint_round_trip_restores_structural_edges(self):
        from drosomath.malecns.checkpoint import (
            restore_learning_checkpoint,
            save_learning_checkpoint,
        )

        first = self._brain()
        overlay = self._configure(first)
        self._seed_activity(overlay)
        overlay.rewire(cycle_label="checkpoint")
        self.assertGreater(overlay.edge_count, 0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "brain.npz"
            saved = save_learning_checkpoint(path, brain=first, stage="concept")
            second = self._brain()
            restored = restore_learning_checkpoint(path, brain=second)

        self.assertEqual(saved["structural_edge_count"], overlay.edge_count)
        self.assertEqual(restored["structural_edge_count"], overlay.edge_count)
        self.assertIsNotNone(second.structural_overlay)
        self.assertEqual(second.structural_overlay.edge_count, overlay.edge_count)
        slots = second.structural_overlay.np.flatnonzero(second.structural_overlay.active)
        donors = second.structural_overlay.donor_edge[slots]
        self.assertTrue(
            (second.plasticity.multiplier[donors] == second.plasticity.config.min_multiplier).all()
        )


if __name__ == "__main__":
    unittest.main()
