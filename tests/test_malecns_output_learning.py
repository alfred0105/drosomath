import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional MaleCNS dependency")
class MaleCNSOutputLearningTests(unittest.TestCase):
    def test_population_readout_learns_two_output_patterns(self) -> None:
        import numpy as np

        from drosomath.malecns.output_readout import (
            OutputPopulation,
            OutputReadoutConfig,
            PopulationReadout,
        )

        population = OutputPopulation(
            body_ids=(101, 102),
            indices=np.asarray([0, 1], dtype=np.int32),
            source="test",
        )
        readout = PopulationReadout(
            population,
            labels=("A", "B"),
            config=OutputReadoutConfig(learning_rate=0.2, seed=0),
        )

        a = np.asarray([1.0, 0.0], dtype=np.float32)
        b = np.asarray([0.0, 1.0], dtype=np.float32)
        for _ in range(40):
            readout.train(a, target="A")
            readout.train(b, target="B")

        self.assertEqual(readout.predict(a)[0], "A")
        self.assertEqual(readout.predict(b)[0], "B")
        self.assertEqual(readout.train_steps, 80)

    def test_decoder_phase_does_not_record_brain_plasticity(self) -> None:
        import numpy as np

        from drosomath.flywire_real import FlyWireConnectome
        from drosomath.whole_brain import PlasticSparseFlyBrain
        from drosomath.malecns.output_readout import OutputPopulation, PopulationReadout
        from drosomath.malecns.output_session import MaleCNSOutputSession

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([101, 102], dtype=np.int64),
            indptr=np.asarray([0, 1, 1], dtype=np.int64),
            post_indices=np.asarray([1], dtype=np.int32),
            signed_synapse_counts=np.asarray([80.0], dtype=np.float32),
            outgoing_strength=np.asarray([80.0, 0.0], dtype=np.float32),
        )
        brain = PlasticSparseFlyBrain(connectome, usage_alpha=1.0)
        population = OutputPopulation(
            body_ids=(102,),
            indices=np.asarray([1], dtype=np.int32),
            source="test",
        )
        readout = PopulationReadout(population, labels=("A", "B"))
        session = MaleCNSOutputSession(brain, readout)

        before_multiplier = brain.plasticity.multiplier.copy()
        session.train_decoder_trial(
            stimulus_body_ids=[101],
            target="A",
            duration_ms=10.0,
            stimulus_rate_hz=5000.0,
        )

        self.assertTrue(np.array_equal(brain.plasticity.multiplier, before_multiplier))
        self.assertAlmostEqual(float(brain.plasticity.usage_ema[0]), 0.0)
        self.assertAlmostEqual(float(brain.plasticity.eligibility[0]), 0.0)

    def test_frozen_decoder_allows_reward_to_train_only_brain(self) -> None:
        import numpy as np

        from drosomath.flywire_real import FlyWireConnectome
        from drosomath.whole_brain import PlasticSparseFlyBrain, UsageRewardRule
        from drosomath.malecns.output_readout import OutputPopulation, PopulationReadout
        from drosomath.malecns.output_session import MaleCNSOutputSession

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([201, 202], dtype=np.int64),
            indptr=np.asarray([0, 1, 1], dtype=np.int64),
            post_indices=np.asarray([1], dtype=np.int32),
            signed_synapse_counts=np.asarray([80.0], dtype=np.float32),
            outgoing_strength=np.asarray([80.0, 0.0], dtype=np.float32),
        )
        brain = PlasticSparseFlyBrain(connectome, usage_alpha=1.0)
        population = OutputPopulation(
            body_ids=(202,),
            indices=np.asarray([1], dtype=np.int32),
            source="test",
        )
        readout = PopulationReadout(population, labels=("A", "B"))
        # Make the frozen decoder deterministically interpret this simple test as A.
        readout.bias[:] = np.asarray([2.0, -2.0], dtype=np.float32)
        readout.freeze()
        weights_before = readout.weights.copy()
        bias_before = readout.bias.copy()

        session = MaleCNSOutputSession(
            brain,
            readout,
            reward_rule=UsageRewardRule(learning_rate=0.1),
        )
        multiplier_before = float(brain.plasticity.multiplier[0])
        result = session.train_brain_trial(
            stimulus_body_ids=[201],
            target="A",
            duration_ms=10.0,
            stimulus_rate_hz=5000.0,
        )

        self.assertTrue(result.correct)
        self.assertEqual(result.reward, 1.0)
        self.assertGreater(float(brain.plasticity.multiplier[0]), multiplier_before)
        self.assertTrue(np.array_equal(readout.weights, weights_before))
        self.assertTrue(np.array_equal(readout.bias, bias_before))

    def test_brain_training_requires_frozen_decoder(self) -> None:
        import numpy as np

        from drosomath.flywire_real import FlyWireConnectome
        from drosomath.whole_brain import PlasticSparseFlyBrain
        from drosomath.malecns.output_readout import OutputPopulation, PopulationReadout
        from drosomath.malecns.output_session import MaleCNSOutputSession

        connectome = FlyWireConnectome(
            flywire_ids=np.asarray([1, 2], dtype=np.int64),
            indptr=np.asarray([0, 1, 1], dtype=np.int64),
            post_indices=np.asarray([1], dtype=np.int32),
            signed_synapse_counts=np.asarray([50.0], dtype=np.float32),
            outgoing_strength=np.asarray([50.0, 0.0], dtype=np.float32),
        )
        brain = PlasticSparseFlyBrain(connectome)
        population = OutputPopulation(
            body_ids=(2,),
            indices=np.asarray([1], dtype=np.int32),
            source="test",
        )
        session = MaleCNSOutputSession(
            brain,
            PopulationReadout(population, labels=("A", "B")),
        )

        with self.assertRaises(RuntimeError):
            session.train_brain_trial(
                stimulus_body_ids=[1],
                target="A",
                duration_ms=2.0,
                stimulus_rate_hz=1000.0,
            )


if __name__ == "__main__":
    unittest.main()
