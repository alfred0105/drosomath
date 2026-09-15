import importlib.util
import tempfile
import unittest
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "MaleCNS v1 tests need NumPy")
class MaleCNSV1Tests(unittest.TestCase):
    def _graph(self, *, many=False):
        import numpy as np
        from drosomath.malecns.loader import MaleCNSConnectome

        if many:
            n_visual, n_output = 220, 40
            n = n_visual + n_output
            body_ids = np.arange(1000, 1000 + n, dtype=np.int64)
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

        return MaleCNSConnectome(
            body_ids=np.asarray([10, 20], dtype=np.int64),
            indptr=np.asarray([0, 1, 1], dtype=np.int64),
            post_indices=np.asarray([1], dtype=np.int32),
            synapse_counts=np.asarray([8], dtype=np.int32),
            signed_synapse_counts=np.asarray([8.0], dtype=np.float32),
            outgoing_strength=np.asarray([8.0, 0.0], dtype=np.float32),
            presynaptic_sign=np.asarray([1, 1], dtype=np.int8),
            consensus_nt=np.asarray(["acetylcholine", "acetylcholine"], dtype=object),
            metadata={"superclass": np.asarray(["visual_projection", "descending_neuron"], dtype=object)},
            min_connection_synapses=5,
        )

    def test_plastic_fraction_unlocks_monotonically(self):
        from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState

        state = SparsePlasticityState(10000, config=PlasticStateConfig(plastic_fraction=0.05, seed=4))
        first = state.plastic_mask.copy()
        state.set_plastic_fraction(0.20)
        self.assertTrue((~first | state.plastic_mask).all())
        self.assertGreater(state.plastic_edge_count, int(first.sum()))

    def test_checkpoint_round_trip_restores_learned_edges(self):
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.malecns.checkpoint import restore_learning_checkpoint, save_learning_checkpoint
        from drosomath.whole_brain import PlasticStateConfig

        graph = self._graph()
        a = PlasticMaleCNSBrain(graph, plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
        a.plasticity.multiplier[0] = 1.75
        a.plasticity.stability[0] = 0.42
        a.plasticity.usage_ema[0] = 0.6
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "brain.npz"
            save_learning_checkpoint(p, brain=a, completed_trials=64, stage="numerosity_1_4")
            b = PlasticMaleCNSBrain(graph, plasticity_config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
            info = restore_learning_checkpoint(p, brain=b)
        self.assertAlmostEqual(float(b.plasticity.multiplier[0]), 1.75, places=5)
        self.assertAlmostEqual(float(b.plasticity.stability[0]), 0.42, places=5)
        self.assertAlmostEqual(float(b.plasticity.usage_ema[0]), 0.6, places=5)
        self.assertEqual(info["completed_trials"], 64)
        self.assertEqual(info["stage"], "numerosity_1_4")

    def test_curriculum_has_variable_numerosity_and_math_stages(self):
        from drosomath.malecns.curriculum_v1 import CurriculumV1Config, build_curriculum

        graph = self._graph(many=True)
        tasks, output = build_curriculum(
            graph,
            config=CurriculumV1Config(output_population_size=16, visual_pool_size=192, token_size=6),
        )
        self.assertEqual([t.name for t in tasks], ["laterality", "numerosity_1_4", "compare_1_3", "addition_1_3"])
        self.assertEqual(len(output.body_ids), 16)
        numerosity = tasks[1]
        self.assertEqual(numerosity.labels, ("N1", "N2", "N3", "N4"))
        self.assertGreater(len(set(numerosity.exemplars["N2"])), 1)
        self.assertTrue(all(len(x) == 12 for x in numerosity.exemplars["N2"]))
        self.assertEqual(tasks[2].labels, ("LT", "EQ", "GT"))
        self.assertEqual(tasks[3].labels, ("SUM2", "SUM3", "SUM4", "SUM5", "SUM6"))

    def test_silent_output_is_not_misclassified_by_bias(self):
        from drosomath.malecns.brain import PlasticMaleCNSBrain
        from drosomath.malecns.output_readout import OutputPopulation, PopulationReadout
        from drosomath.malecns.output_session import MaleCNSOutputSession, NO_OUTPUT, OutputSessionConfig
        from drosomath.whole_brain import PlasticStateConfig

        graph = self._graph()
        graph.indptr[:] = 0
        graph.post_indices = graph.post_indices[:0]
        graph.synapse_counts = graph.synapse_counts[:0]
        graph.signed_synapse_counts = graph.signed_synapse_counts[:0]
        graph.outgoing_strength[:] = 0
        brain = PlasticMaleCNSBrain(graph, plasticity_config=PlasticStateConfig(plastic_fraction=1.0))
        output = OutputPopulation.from_body_ids(graph, [20])
        readout = PopulationReadout(output, ("A", "B"))
        readout.freeze()
        session = MaleCNSOutputSession(brain, readout, config=OutputSessionConfig(no_output_reward=-0.25))
        row = session.evaluate_trial(stimulus_body_ids=[10], target="A", duration_ms=1.0, stimulus_rate_hz=0.0)
        self.assertEqual(row["prediction"], NO_OUTPUT)
        self.assertFalse(row["correct"])
        self.assertTrue(row["silent"])

    def test_memory_v2_gate_requires_measurable_learning_and_retention(self):
        from drosomath.malecns.curriculum_v1 import CurriculumV1Config, build_curriculum
        from drosomath.malecns.curriculum_v2_memory import _phase1_gate

        tasks, _ = build_curriculum(
            self._graph(many=True),
            config=CurriculumV1Config(output_population_size=16, visual_pool_size=192, token_size=6),
        )
        stages = [
            {"stage": "laterality", "after_accuracy": 0.90},
            {"stage": "numerosity_1_4", "after_accuracy": 0.50},
            {"stage": "compare_1_3", "after_accuracy": 0.55},
            {"stage": "addition_1_3", "after_accuracy": 0.30},
        ]
        retention = [{
            "after_stage": "addition_1_3",
            "tasks": {
                "laterality": {"accuracy": 0.90},
                "numerosity_1_4": {"accuracy": 0.42},
                "compare_1_3": {"accuracy": 0.46},
                "addition_1_3": {"accuracy": 0.30},
            },
        }]
        gate = _phase1_gate(tasks, stages, retention)
        self.assertTrue(gate["passed"])
        retention[0]["tasks"]["compare_1_3"]["accuracy"] = 0.30
        gate = _phase1_gate(tasks, stages, retention)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["tasks"]["compare_1_3"]["passed"])


if __name__ == "__main__":
    unittest.main()
