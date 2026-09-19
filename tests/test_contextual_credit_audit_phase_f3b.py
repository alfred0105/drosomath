from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    CONTEXTUAL_GRAMMAR,
    ContextualPredictionLearningSession,
    ContextualPredictionSession,
    ORDERED_PAIRS,
    PlasticMaleCNSBrain,
    SequenceLearningSession,
    SequenceMemoryConfig,
    SYMBOLS,
    SymbolInterface,
    SymbolInterfaceConfig,
    TwoCueSequenceSession,
    WorkingMemoryInterface,
    balanced_pair_schedule,
)
from drosomath.whole_brain import PlasticStateConfig
from run_contextual_credit_audit_phase_f3b import (
    _CreditRecorder,
    _audit_state,
    _conflict_metrics,
    _group_name,
    _jaccard,
    _mean_cancellation,
    _overlap_metrics,
    _representation_summary,
    _run_training_arm,
)
from test_working_memory_phase_f2a import make_connectome


class ContextualCreditAuditPhaseF3BTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.interface_config = SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7)
        cls.interface = SymbolInterface(cls.connectome, cls.interface_config)
        cls.wm = WorkingMemoryInterface(cls.connectome, cls.interface)

    def make_brain(self, seed=149):
        return PlasticMaleCNSBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        )

    def test_matched_arms_have_same_episodes_and_only_target_differs(self):
        schedule = balanced_pair_schedule(25, seed=149)
        self.assertEqual(len(schedule), 400)
        self.assertTrue(all(schedule.count(context) == 25 for context in ORDERED_PAIRS))
        recall = self.make_brain()
        prediction = self.make_brain()
        np.testing.assert_array_equal(recall.plasticity.plastic_mask, prediction.plasticity.plastic_mask)
        self.assertEqual(inspect.getsource(ContextualPredictionSession.run_trial).count("CONTEXTUAL_GRAMMAR"), 1)
        self.assertIn("target=result.target", inspect.getsource(ContextualPredictionLearningSession.train_trial))

    def test_stable_active_definition_is_three_of_four(self):
        repetitions = [{1, 2}, {1, 2}, {1, 2}, {1, 3}]
        counts = {value: sum(value in active for active in repetitions) for value in {1, 2, 3}}
        stable = {value for value, count in counts.items() if count >= 3}
        self.assertEqual(stable, {1, 2})
        self.assertNotIn(3, stable)

    def test_state_audit_returns_compact_phase_metrics(self):
        brain = self.make_brain()
        from drosomath.malecns.symbol_interface import _snapshot_persistent_state
        audit = _audit_state(self.connectome, self.wm, self.interface_config, 149, _snapshot_persistent_state(brain), brain.rng.bit_generator.state)
        self.assertEqual(set(audit["stable"]), {"first", "pre_go"})
        self.assertEqual(len(audit["stable"]["pre_go"]), 16)
        self.assertTrue(all("fingerprint" in row for rows in audit["metrics"]["first"].values() for row in rows))

    def test_context_similarity_grouping(self):
        self.assertEqual(_group_name(("A", "B"), ("A", "C")), "same_first_different_second")
        self.assertEqual(_group_name(("A", "B"), ("C", "B")), "same_second_different_first")
        self.assertEqual(_group_name(("A", "B"), ("C", "D")), "different_first_and_second")
        self.assertEqual(_jaccard({1, 2}, {2, 3}), 1 / 3)

    def test_representation_summary_has_strict_and_broad_pools(self):
        stable = {context: {index for index in range(8) if index % 4 == SYMBOLS.index(context[0])} | {16 + SYMBOLS.index(context[1])} for context in ORDERED_PAIRS}
        frequencies = {context: {value: 1.0 for value in values} for context, values in stable.items()}
        audit = {"stable": {"first": stable, "pre_go": stable}, "frequencies": {"first": frequencies, "pre_go": frequencies}, "metrics": {"first": {}, "pre_go": {}}}
        result = _representation_summary(audit)
        self.assertIn("same_first_different_second", result["pairwise_context_similarity"]["pre_go"])
        self.assertEqual(set(result["conjunctive_specific_pools"]["pre_go"]["per_context"]), {"".join(context) for context in ORDERED_PAIRS})
        self.assertIn("broader_conjunction_count", result["conjunctive_specific_pools"]["pre_go"]["per_context"]["AB"])

    def test_credit_sets_overlap_by_relationship(self):
        buckets = {context: {"positive_edges": {0, SYMBOLS.index(context[0])}, "negative_edges": set()} for context in ORDERED_PAIRS}
        same_first = _overlap_metrics(buckets, lambda a, b: a[0] == b[0] and a != b)
        same_second = _overlap_metrics(buckets, lambda a, b: a[1] == b[1] and a != b)
        self.assertGreaterEqual(same_first["pairs"], 1)
        self.assertGreaterEqual(same_second["pairs"], 1)

    def test_conflict_matrix_excludes_same_target_pairs(self):
        buckets = {context: {"positive_edges": {SYMBOLS.index(CONTEXTUAL_GRAMMAR[context])}} for context in ORDERED_PAIRS}
        result = _conflict_metrics(buckets)
        self.assertIn("same_first_different_second", result)
        self.assertIn("same_second_different_first", result)

    def test_sign_cancellation_tracks_positive_negative_requests(self):
        bucket = {"positive_edges": {1}, "negative_edges": {1, 2}, "channel_data": {"symbol/A": {"positive": {1}, "negative": {1, 2}, "signed": {1: 0.5, 2: -0.25}, "absolute": {1: 1.0, 2: 0.25}}}}
        result = _mean_cancellation([({"AB": bucket}, {}, {})])
        self.assertGreater(result["cancellation_fraction"], 0.0)
        self.assertEqual(result["per_output"]["A"]["intersection"], 1.0)

    def test_training_arm_reports_per_context_credit_and_fixed_budget(self):
        schedule = balanced_pair_schedule(1, seed=149)
        brain, session, per_context, phase, records, budget_start, budget_end = _run_training_arm(self.connectome, self.wm, self.interface_config, 149, schedule, False)
        self.assertEqual(len(records), 16)
        self.assertEqual(set(per_context), set(ORDERED_PAIRS))
        self.assertEqual(budget_start, budget_end)
        self.assertEqual(set(phase), set(ORDERED_PAIRS))

    def test_diagnostic_recorder_uses_attribution_observer_only(self):
        brain = self.make_brain()
        session = SequenceLearningSession(brain, self.wm, config=SequenceMemoryConfig())
        recorder = _CreditRecorder(session, ("A", "B"), {"eligible_after_first": set(), "newly_second": set(), "newly_go": set()})
        session.train_trial("A", "B", capture_phase_credit=True)
        self.assertTrue(hasattr(recorder, "events"))
        self.assertNotIn("classifier", inspect.getsource(ContextualPredictionLearningSession.train_trial).lower())

    def test_no_external_decoder_or_learning_parameter_change(self):
        source = inspect.getsource(ContextualPredictionLearningSession)
        self.assertNotIn("backprop", source.lower())
        self.assertNotIn("classifier", source.lower())
        self.assertEqual(SequenceMemoryConfig().directional_learning_rate, 0.02)
        self.assertEqual(SequenceMemoryConfig().reward_learning_rate, 0.02)


if __name__ == "__main__":
    unittest.main()
