import unittest
from types import SimpleNamespace
import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.whole_brain.directional_modulation import PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState


def fake_brain():
    # pre 0 has excitatory and inhibitory direct output edges; pre 1 is a
    # real upstream path into pre 0 for bounded two-hop tests.
    graph = SimpleNamespace(
        neuron_count=3,
        indptr=np.asarray([0, 2, 3, 3]),
        post_indices=np.asarray([2, 2, 0]),
        signed_synapse_counts=np.asarray([1.0, -1.0, 1.0]),
    )
    state = SparsePlasticityState(3, config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
    state.eligibility[:] = 1.0
    return SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic={0, 1})


class LearningSignalTests(unittest.TestCase):
    def test_signal_is_task_independent(self):
        signal = LearningSignal(reward=1.0, directional_error={"choice/left": 0.4})
        self.assertEqual(signal.nonzero_directions(), {"choice/left": 0.4})

    def test_zero_direction_is_ignored(self):
        signal = LearningSignal(reward=0.0, directional_error={"motor/click": 0.0})
        self.assertEqual(signal.nonzero_directions(), {})

    def test_directional_sign_and_consolidation(self):
        brain = fake_brain(); controller = PlasticityController()
        before = brain.plasticity.multiplier.copy()
        update = controller.apply_learning_signal(brain, LearningSignal(0.0, {"motor/click": 1.0}, success=True), {"motor/click": np.asarray([2])})
        self.assertGreater(brain.plasticity.multiplier[0], before[0])
        self.assertLess(brain.plasticity.multiplier[1], before[1])
        self.assertGreater(brain.plasticity.stability[0], 0.0)
        self.assertEqual(update.hop_counts[1], 2)
        self.assertEqual(update.hop_counts[2], 1)

    def test_zero_eligibility_cannot_change(self):
        brain = fake_brain(); brain.plasticity.eligibility[:] = 0.0
        before = brain.plasticity.multiplier.copy()
        update = PlasticityController().apply_learning_signal(brain, LearningSignal(0.0, {"motor/click": -1.0}), {"motor/click": np.asarray([2])})
        self.assertEqual(update.edge_updates, 0)
        np.testing.assert_array_equal(before, brain.plasticity.multiplier)


if __name__ == "__main__":
    unittest.main()
