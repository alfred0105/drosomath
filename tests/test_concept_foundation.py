import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "concept foundation tests need NumPy")
class ConceptFoundationTests(unittest.TestCase):
    def _graph(self):
        import numpy as np
        from drosomath.malecns.loader import MaleCNSConnectome

        n_visual = 220
        n_output = 48
        n = n_visual + n_output
        body_ids = np.arange(10_000, 10_000 + n, dtype=np.int64)
        superclass = np.asarray(
            ["visual_projection"] * n_visual + ["descending_neuron"] * n_output,
            dtype=object,
        )
        return MaleCNSConnectome(
            body_ids=body_ids,
            indptr=np.zeros(n + 1, dtype=np.int64),
            post_indices=np.empty(0, dtype=np.int32),
            synapse_counts=np.empty(0, dtype=np.int32),
            signed_synapse_counts=np.empty(0, dtype=np.float32),
            outgoing_strength=np.linspace(1, n, n, dtype=np.float32),
            presynaptic_sign=np.ones(n, dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine"] * n, dtype=object),
            metadata={"superclass": superclass},
            min_connection_synapses=5,
        )

    def _bundle(self):
        from drosomath.malecns.concept_foundation import (
            ConceptFoundationConfig,
            build_concept_foundation,
        )

        config = ConceptFoundationConfig(
            output_population_size=32,
            train_examples_per_label=12,
            heldout_examples_per_label=8,
            anchor_examples_per_label=2,
            validation_trials_per_label=4,
            stage_trials=8,
        )
        return build_concept_foundation(self._graph(), config=config), config

    def test_curriculum_starts_with_objects_not_math_symbols(self):
        bundle, _ = self._bundle()
        self.assertEqual(
            [task.name for task in bundle.tasks],
            ["object_presence", "single_vs_multiple", "latent_quantity_1_3"],
        )
        all_labels = {label for task in bundle.tasks for label in task.labels}
        self.assertFalse(any(label.startswith("SUM") for label in all_labels))
        self.assertFalse(any(label in {"LT", "EQ", "GT", "N1", "N2", "N3"} for label in all_labels))
        self.assertEqual(bundle.tasks[-1].labels, ("LATENT_A", "LATENT_B", "LATENT_C"))

    def test_virtual_positions_are_disjoint_train_and_heldout(self):
        bundle, _ = self._bundle()
        field = bundle.field
        self.assertTrue(set(field.train_positions).isdisjoint(field.heldout_positions))
        self.assertEqual(
            len(field.train_positions) + len(field.heldout_positions),
            field.width * field.height,
        )
        all_groups = [body_id for group in field.position_body_ids for body_id in group]
        self.assertEqual(len(all_groups), len(set(all_groups)))
        self.assertTrue(set(all_groups).isdisjoint(field.background_body_ids))
        self.assertFalse(bundle.provenance["biological_retinotopy_claimed"])

    def test_heldout_examples_use_unseen_position_neurons(self):
        bundle, _ = self._bundle()
        field = bundle.field
        train_neurons = {
            body_id
            for position in field.train_positions
            for body_id in field.position_body_ids[position]
        }
        heldout_neurons = {
            body_id
            for position in field.heldout_positions
            for body_id in field.position_body_ids[position]
        }
        self.assertTrue(train_neurons.isdisjoint(heldout_neurons))

        background = set(field.background_body_ids)
        latent = bundle.tasks[-1]
        for examples in latent.heldout_examples.values():
            for example in examples:
                stimulus_neurons = set(example) - background
                self.assertTrue(stimulus_neurons)
                self.assertTrue(stimulus_neurons <= heldout_neurons)
                self.assertTrue(stimulus_neurons.isdisjoint(train_neurons))

    def test_anchor_examples_are_small_fixed_reference_set(self):
        bundle, config = self._bundle()
        for task in bundle.tasks:
            for label in task.labels:
                self.assertEqual(len(task.anchors[label]), config.anchor_examples_per_label)
                self.assertGreater(len(task.train_examples[label]), len(task.anchors[label]))

    def test_foundation_gate_requires_unseen_position_transfer(self):
        from drosomath.malecns.concept_foundation import _foundation_gate

        names = ["object_presence", "single_vs_multiple", "latent_quantity_1_3"]
        thresholds = [0.80, 0.70, 0.50]
        stages = []
        final_tasks = {}
        for name, threshold in zip(names, thresholds):
            held = threshold + 0.05
            stages.append(
                {
                    "stage": name,
                    "after_train": {"accuracy": min(1.0, held + 0.05)},
                    "after_heldout": {
                        "accuracy": held,
                        "chance": 0.5 if name != "latent_quantity_1_3" else 1 / 3,
                    },
                }
            )
            final_tasks[name] = {"heldout_accuracy": held}
        retention = [{"after_stage": names[-1], "tasks": final_tasks}]
        gate = _foundation_gate(stages, retention)
        self.assertTrue(gate["passed"])

        stages[-1]["after_train"]["accuracy"] = 0.95
        stages[-1]["after_heldout"]["accuracy"] = 0.51
        gate = _foundation_gate(stages, retention)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["tasks"]["latent_quantity_1_3"]["generalizes_to_unseen_positions"])


if __name__ == "__main__":
    unittest.main()
