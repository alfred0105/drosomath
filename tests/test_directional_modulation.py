import ast
import pathlib
import unittest
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.keyboard_learning import KeyboardNeuralSession
from drosomath.whole_brain.directional_modulation import PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState


def make_brain(edges, *, neuron_count=3, eligibility=1.0, plastic=True, recent=None):
    """Build a deterministic compact CSR brain from (pre, post, sign) rows."""
    ordered = sorted(edges, key=lambda item: item[0])
    indptr, posts, signs, cursor = [0], [], [], 0
    for pre in range(neuron_count):
        while cursor < len(ordered) and ordered[cursor][0] == pre:
            _, post, sign = ordered[cursor]
            posts.append(post); signs.append(float(sign)); cursor += 1
        indptr.append(len(posts))
    graph = SimpleNamespace(
        neuron_count=neuron_count,
        indptr=np.asarray(indptr, dtype=np.int32),
        post_indices=np.asarray(posts, dtype=np.int32),
        signed_synapse_counts=np.asarray(signs, dtype=np.float32),
    )
    state = SparsePlasticityState(len(posts), config=PlasticStateConfig(plastic_fraction=1.0, seed=1))
    state.eligibility[:] = eligibility
    if not plastic:
        state.plastic_mask[:] = False
    return SimpleNamespace(
        np=np, connectome=graph, plasticity=state,
        _recent_presynaptic=set(range(neuron_count)) if recent is None else set(recent),
    )


class LearningSignalTests(unittest.TestCase):
    def test_signal_is_task_independent(self):
        signal = LearningSignal(reward=1.0, directional_error={"choice/left": 0.4}, reinforcement={"choice/left": 1.0})
        self.assertEqual(signal.nonzero_directions(), {"choice/left": 0.4})
        self.assertEqual(signal.positive_reinforcements(), {"choice/left": 1.0})

    def test_generic_modules_do_not_import_keyboard(self):
        root = pathlib.Path(__file__).parents[1] / "src" / "drosomath"
        for path in (root / "learning_signal.py", root / "whole_brain" / "directional_modulation.py"):
            imports = [node.module or "" for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any("keyboard" in module for module in imports), path)


class DirectionalModulationTests(unittest.TestCase):
    context = {"motor/click": np.asarray([2], dtype=np.int32)}

    def apply(self, brain, direction=0.0, reinforcement=0.0, success=False):
        return PlasticityController().apply_learning_signal(
            brain,
            LearningSignal(0.0, {"motor/click": direction}, success=success, reinforcement={"motor/click": reinforcement}),
            self.context,
        )

    def test_one_hop_sign_rules(self):
        for sign, direction, expected in ((1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)):
            with self.subTest(sign=sign, direction=direction):
                brain = make_brain([(0, 2, sign)])
                before = float(brain.plasticity.multiplier[0])
                self.apply(brain, direction=direction)
                self.assertGreater((float(brain.plasticity.multiplier[0]) - before) * expected, 0.0)

    def test_two_hop_path_polarity_all_sign_combinations(self):
        for upstream, downstream, expected in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            with self.subTest(upstream=upstream, downstream=downstream):
                brain = make_brain([(0, 1, upstream), (1, 2, downstream)])
                before = float(brain.plasticity.multiplier[0])
                update = self.apply(brain, direction=1.0)
                self.assertGreater((float(brain.plasticity.multiplier[0]) - before) * expected, 0.0)
                self.assertEqual(update.hop_counts[2], 1)

    def test_mixed_downstream_uses_aggregate_effect(self):
        brain = make_brain([(0, 1, 1), (1, 2, 1), (1, 2, -3)])
        before = float(brain.plasticity.multiplier[0])
        self.apply(brain, direction=1.0)
        self.assertLess(float(brain.plasticity.multiplier[0]), before)

    def test_zero_net_downstream_is_skipped(self):
        brain = make_brain([(0, 1, 1), (1, 2, 1), (1, 2, -1)])
        before = brain.plasticity.multiplier.copy()
        update = self.apply(brain, direction=1.0)
        self.assertEqual(float(brain.plasticity.multiplier[0]), float(before[0]))
        self.assertGreaterEqual(update.ambiguous_path_edges_skipped, 1)

    def test_zero_eligibility_and_nonplastic_edges_do_not_change(self):
        for kwargs in ({"eligibility": 0.0}, {"plastic": False}):
            with self.subTest(kwargs=kwargs):
                brain = make_brain([(0, 2, 1)], **kwargs)
                before = brain.plasticity.multiplier.copy()
                update = self.apply(brain, direction=1.0)
                self.assertEqual(update.edge_updates, 0)
                np.testing.assert_array_equal(before, brain.plasticity.multiplier)

    def test_edge_outside_two_hops_is_not_changed(self):
        brain = make_brain([(0, 1, 1), (1, 2, 1), (2, 3, 1)], neuron_count=4)
        before = brain.plasticity.multiplier.copy()
        PlasticityController().apply_learning_signal(
            brain, LearningSignal(0.0, {"motor/click": 1.0}), {"motor/click": np.asarray([3])},
        )
        self.assertEqual(float(brain.plasticity.multiplier[0]), float(before[0]))
        self.assertNotEqual(float(brain.plasticity.multiplier[1]), float(before[1]))

    def test_successful_reinforcement_consolidates_without_directional_update(self):
        brain = make_brain([(0, 1, 1), (1, 2, 1)])
        before_multiplier = brain.plasticity.multiplier.copy()
        update = self.apply(brain, reinforcement=1.0, success=True)
        self.assertEqual(update.edge_updates, 0)
        self.assertEqual(update.reinforced_channels, ("motor/click",))
        self.assertGreater(update.consolidated_edges, 0)
        self.assertGreater(float(brain.plasticity.stability.mean()), 0.0)
        np.testing.assert_array_equal(before_multiplier, brain.plasticity.multiplier)

    def test_reinforcement_stability_is_small_gradual_and_success_only(self):
        brain = make_brain([(0, 2, 1)])
        first = self.apply(brain, reinforcement=1.0, success=True)
        one = float(brain.plasticity.stability[0])
        self.assertGreater(one, 0.0); self.assertLess(one, 0.1)
        self.apply(brain, reinforcement=1.0, success=True)
        self.assertGreater(float(brain.plasticity.stability[0]), one)
        failed = make_brain([(0, 2, 1)])
        update = self.apply(failed, reinforcement=0.0, success=False)
        self.assertEqual(update.consolidated_edges, 0)
        self.assertEqual(float(failed.plasticity.stability[0]), 0.0)
        self.assertGreater(first.consolidated_edges, 0)

    def test_high_stability_keeps_nonzero_learning_factor_and_is_deterministic(self):
        snapshots = []
        for _ in range(2):
            brain = make_brain([(0, 2, 1)])
            brain.plasticity.stability[:] = 1.0
            before = float(brain.plasticity.multiplier[0])
            self.apply(brain, direction=1.0)
            self.assertGreater(float(brain.plasticity.multiplier[0]), before)
            snapshots.append(brain.plasticity.multiplier.copy())
        np.testing.assert_array_equal(snapshots[0], snapshots[1])


class KeyboardAdapterTests(unittest.TestCase):
    def test_correct_click_produces_generic_reinforcement(self):
        session = SimpleNamespace(config=SimpleNamespace(click_only=True))
        signal = KeyboardNeuralSession._learning_signal_for_trial(session, reward=1.0, correct=True, current_deficit=0.3, click_gain=1.0)
        self.assertEqual(signal.nonzero_directions(), {})
        self.assertEqual(signal.positive_reinforcements(), {"motor/click": 1.0})

    def test_failed_click_produces_correction_without_reinforcement(self):
        session = SimpleNamespace(config=SimpleNamespace(click_only=True))
        signal = KeyboardNeuralSession._learning_signal_for_trial(session, reward=0.0, correct=False, current_deficit=0.3, click_gain=1.2)
        self.assertEqual(signal.nonzero_directions(), {"motor/click": 0.36})
        self.assertEqual(signal.positive_reinforcements(), {})


if __name__ == "__main__":
    unittest.main()
