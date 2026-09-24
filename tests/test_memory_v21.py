import unittest
from types import SimpleNamespace

import numpy as np

from drosomath.whole_brain import (
    AdaptiveReplayConfig,
    AdaptiveReplayScheduler,
    ConsolidationConfig,
    MemoryConsolidator,
)


class MemoryV21Tests(unittest.TestCase):
    def test_adaptive_replay_prefers_weaker_task(self) -> None:
        scheduler = AdaptiveReplayScheduler(
            AdaptiveReplayConfig(error_power=2.0, min_weight=0.01, ema_decay=0.8)
        )
        scheduler.set_accuracy(0, 0.98)
        scheduler.set_accuracy(1, 0.40)
        snap = scheduler.snapshot(2)["tasks"]
        p0 = snap[0]["probability"]
        p1 = snap[1]["probability"]
        self.assertGreater(p1, p0 * 10.0)

        rng = np.random.default_rng(3)
        picks = [scheduler.choose_prior_index(rng, 2) for _ in range(400)]
        self.assertGreater(picks.count(1), picks.count(0) * 5)

    def test_replay_ema_updates_after_result(self) -> None:
        scheduler = AdaptiveReplayScheduler(AdaptiveReplayConfig(ema_decay=0.5))
        scheduler.set_accuracy(0, 0.4)
        self.assertAlmostEqual(scheduler.update(0, correct=True), 0.7)
        self.assertAlmostEqual(scheduler.update(0, correct=False), 0.35)

    def test_stage_local_consolidation_ignores_old_stable_edge(self) -> None:
        state = SimpleNamespace(
            np=np,
            plastic_mask=np.array([True, True, True]),
            stability=np.array([0.90, 0.20, 0.10], dtype=np.float32),
            usage_ema=np.array([0.8, 0.8, 0.8], dtype=np.float32),
            multiplier=np.array([1.2, 1.1, 1.0], dtype=np.float32),
        )
        baseline = np.array([0.90, 0.10, 0.10], dtype=np.float32)
        consolidator = MemoryConsolidator(
            ConsolidationConfig(top_fraction=1.0, boost=0.25, min_stage_gain=1e-4)
        )
        old_first = float(state.stability[0])
        old_second = float(state.stability[1])
        report = consolidator.consolidate(state, baseline_stability=baseline)

        self.assertTrue(report["stage_local"])
        self.assertEqual(report["candidate_edges"], 1)
        self.assertEqual(report["consolidated_edges"], 1)
        self.assertAlmostEqual(float(state.stability[0]), old_first)
        self.assertGreater(float(state.stability[1]), old_second)

    def test_v21_module_imports(self) -> None:
        from drosomath.malecns.curriculum_v21_memory import MemoryV21Config

        config = MemoryV21Config(stage_trials=16, validation_trials_per_label=4)
        self.assertEqual(config.adaptive_error_power, 2.0)
        self.assertEqual(config.stage_trials, 16)


if __name__ == "__main__":
    unittest.main()
