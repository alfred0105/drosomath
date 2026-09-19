import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from drosomath.learning_signal import LearningSignal
from drosomath.malecns.keyboard_learning import KeyboardNeuralSession
from drosomath.whole_brain.directional_modulation import PlasticityController
from drosomath.whole_brain.plastic_state import PlasticStateConfig, SparsePlasticityState
from drosomath.whole_brain.usage_learning import RewardCredit, UsageRewardRule


def brain(edges, neuron_count=3, *, plastic_fraction=1.0, eligibility=1.0):
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
    state = SparsePlasticityState(len(posts), config=PlasticStateConfig(plastic_fraction=plastic_fraction, seed=3))
    state.eligibility[:] = eligibility
    state.usage_ema[:] = 1.0
    return SimpleNamespace(np=np, connectome=graph, plasticity=state, _recent_presynaptic=set(range(neuron_count)))


class RewardRuleTests(unittest.TestCase):
    def setUp(self):
        self.rule = UsageRewardRule(learning_rate=0.1)

    def test_no_credit_preserves_legacy_behavior(self):
        left, right = brain([(0, 2, 1), (1, 2, 1)]), brain([(0, 2, 1), (1, 2, 1)])
        a = self.rule.apply(left.plasticity, reward=1.0)
        b = self.rule.apply_recent_presynaptic(right.plasticity, reward=1.0, indptr=right.connectome.indptr, presynaptic_indices=[0, 1])
        self.assertEqual(a.edge_updates, b.edge_updates)
        np.testing.assert_allclose(left.plasticity.multiplier, right.plasticity.multiplier)

    def test_sparse_credit_only_changes_credited_edges(self):
        b = brain([(0, 2, 1), (0, 1, 1), (1, 2, 1)])
        before = b.plasticity.multiplier.copy()
        credit = RewardCredit(np.asarray([0], dtype=np.int32), np.asarray([1.0], dtype=np.float32))
        stats = self.rule.apply_recent_presynaptic(b.plasticity, reward=1.0, indptr=b.connectome.indptr, presynaptic_indices=[0, 1], reward_credit=credit)
        self.assertEqual(stats.edge_updates, 1)
        self.assertNotEqual(float(b.plasticity.multiplier[0]), float(before[0]))
        np.testing.assert_array_equal(b.plasticity.multiplier[1:], before[1:])
        self.assertEqual(stats.uncredited_edges_updated, 0)

    def test_uncredited_eligible_edge_gets_no_full_reward(self):
        b = brain([(0, 2, 1), (0, 1, 1)])
        credit = RewardCredit(np.asarray([0], dtype=np.int32), np.asarray([1.0], dtype=np.float32))
        self.rule.apply_recent_presynaptic(b.plasticity, reward=1.0, indptr=b.connectome.indptr, presynaptic_indices=[0], reward_credit=credit)
        self.assertEqual(float(b.plasticity.multiplier[1]), 1.0)

    def test_sparse_weight_scales_update(self):
        full = brain([(0, 2, 1)]); half = brain([(0, 2, 1)])
        self.rule.apply_recent_presynaptic(full.plasticity, reward=1.0, indptr=full.connectome.indptr, presynaptic_indices=[0], reward_credit=RewardCredit(np.array([0]), np.array([1.0])))
        self.rule.apply_recent_presynaptic(half.plasticity, reward=1.0, indptr=half.connectome.indptr, presynaptic_indices=[0], reward_credit=RewardCredit(np.array([0]), np.array([0.5])))
        self.assertAlmostEqual(float(half.plasticity.multiplier[0] - 1.0), 0.5 * float(full.plasticity.multiplier[0] - 1.0), places=6)

    def test_zero_eligibility_and_nonplastic_edges_do_not_update(self):
        for kwargs in ({"eligibility": 0.0}, {"plastic_fraction": 0.0}):
            b = brain([(0, 2, 1)], **kwargs)
            stats = self.rule.apply_recent_presynaptic(b.plasticity, reward=1.0, indptr=b.connectome.indptr, presynaptic_indices=[0], reward_credit=RewardCredit(np.array([0]), np.array([1.0])))
            self.assertEqual(stats.edge_updates, 0)


class RewardCreditBuilderTests(unittest.TestCase):
    context = {"motor/click": np.asarray([2], dtype=np.int32)}

    def build(self, edges, neuron_count=3, signal=None):
        b = brain(edges, neuron_count)
        signal = signal or LearningSignal(1.0, reinforcement={"motor/click": 1.0}, success=True)
        return b, PlasticityController().build_reward_credit(b, signal, self.context)

    def test_one_hop_positive_route_receives_credit(self):
        _, credit = self.build([(0, 2, 1)])
        np.testing.assert_array_equal(credit.edge_indices, [0])
        self.assertEqual(credit.aligned_one_hop_edges, 1)
        self.assertEqual(float(credit.weights[0]), 1.0)

    def test_two_hop_positive_route_has_decay(self):
        _, credit = self.build([(0, 1, 1), (1, 2, 1)])
        self.assertEqual(set(credit.edge_indices.tolist()), {0, 1})
        self.assertEqual(credit.aligned_two_hop_edges, 1)
        self.assertAlmostEqual(float(credit.weights[0]), 0.5, places=6)

    def test_negative_path_gets_no_positive_credit(self):
        _, credit = self.build([(0, 2, -1)])
        self.assertEqual(len(credit.edge_indices), 0)
        self.assertEqual(credit.unaligned_edges_skipped, 1)

    def test_ambiguous_path_gets_no_positive_credit(self):
        _, credit = self.build([(0, 1, 1), (1, 2, 1), (1, 2, -1)])
        # The direct positive edge remains valid; the ambiguous upstream edge
        # (0 -> 1) must not receive invented positive credit.
        self.assertNotIn(0, credit.edge_indices.tolist())
        self.assertGreaterEqual(credit.unaligned_edges_skipped, 1)

    def test_fixed_state_credit_selection_is_deterministic(self):
        a, x = self.build([(0, 1, 1), (1, 2, 1)])
        _, y = self.build([(0, 1, 1), (1, 2, 1)])
        np.testing.assert_array_equal(x.edge_indices, y.edge_indices)
        np.testing.assert_array_equal(x.weights, y.weights)
        self.assertEqual(a.plasticity.edge_count, 2)

    def test_credit_api_has_no_dense_edge_count_allocation(self):
        source = inspect.getsource(PlasticityController.build_reward_credit)
        self.assertNotIn("zeros(graph.edge_count", source)
        self.assertNotIn("zeros(state.edge_count", source)


class IntegrationRewardLocalityTests(unittest.TestCase):
    def test_keyboard_success_and_failure_build_different_generic_signals(self):
        session = SimpleNamespace(config=SimpleNamespace(click_only=True))
        success = KeyboardNeuralSession._learning_signal_for_trial(session, reward=1.0, correct=True, current_deficit=0.0, click_gain=1.0)
        failure = KeyboardNeuralSession._learning_signal_for_trial(session, reward=0.0, correct=False, current_deficit=0.5, click_gain=1.0)
        self.assertEqual(success.positive_reinforcements(), {"motor/click": 1.0})
        self.assertEqual(failure.positive_reinforcements(), {})

    def test_localized_reward_does_not_break_directional_correction(self):
        b = brain([(0, 2, 1)])
        controller = PlasticityController()
        before = float(b.plasticity.multiplier[0])
        update = controller.apply_learning_signal(b, LearningSignal(0.0, {"motor/click": 1.0}), {"motor/click": np.array([2])})
        self.assertGreater(update.edge_updates, 0)
        self.assertGreater(float(b.plasticity.multiplier[0]), before)

    def test_localized_reward_and_consolidation_are_not_double_stability_gain(self):
        b = brain([(0, 2, 1)])
        controller = PlasticityController()
        signal = LearningSignal(1.0, reinforcement={"motor/click": 1.0}, success=True)
        credit = controller.build_reward_credit(b, signal, {"motor/click": np.array([2])})
        UsageRewardRule(learning_rate=0.1).apply_recent_presynaptic(b.plasticity, reward=1.0, indptr=b.connectome.indptr, presynaptic_indices=[0], reward_credit=credit)
        update = controller.apply_learning_signal(b, signal, {"motor/click": np.array([2])})
        self.assertAlmostEqual(float(b.plasticity.stability[0]), 0.01, places=6)
        self.assertEqual(update.consolidated_edges, 1)

    def test_generic_reward_module_has_no_keyboard_import(self):
        from drosomath.whole_brain import usage_learning
        self.assertNotIn("keyboard", inspect.getsource(usage_learning).lower())

    def test_old_non_keyboard_callers_without_credit_still_work(self):
        b = brain([(0, 2, 1)])
        stats = UsageRewardRule().apply_recent_presynaptic(b.plasticity, reward=1.0, indptr=b.connectome.indptr, presynaptic_indices=[0])
        self.assertEqual(stats.edge_updates, 1)


if __name__ == "__main__":
    unittest.main()
