from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from drosomath.flywire_real import FlyBrainParams
from drosomath.malecns import (
    DelayedCueSession,
    PlasticMaleCNSBrain,
    SYMBOLS,
    SymbolInterface,
    SymbolInterfaceConfig,
    WorkingMemoryInterface,
    WorkingMemoryLearningConfig,
)
from drosomath.whole_brain import PlasticStateConfig, TimingProfiler
from run_delay_generalization_phase_f2c import (
    EVALUATION_DELAYS_MS,
    _criterion,
    _memory_horizon,
    _neural_exact_replay,
    _retention_curve,
    _state_decay_curve,
)
from test_working_memory_phase_f2a import make_connectome


class RecordingBrain(PlasticMaleCNSBrain):
    def __init__(self, *args, **kwargs):
        self.calls = []
        super().__init__(*args, **kwargs)

    def step(self, *, stimulus_indices=None, stimulus_rate_hz=0.0):
        self.calls.append((None if stimulus_indices is None else tuple(stimulus_indices), float(stimulus_rate_hz)))
        return super().step(stimulus_indices=stimulus_indices, stimulus_rate_hz=stimulus_rate_hz)


class DelayGeneralizationPhaseF2CTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connectome = make_connectome()
        cls.interface_config = SymbolInterfaceConfig(sensory_population_size=8, output_population_size=8, seed=7)
        cls.interface = SymbolInterface(cls.connectome, cls.interface_config)
        cls.wm = WorkingMemoryInterface(cls.connectome, cls.interface)

    def make_brain(self, seed=101, brain_type=PlasticMaleCNSBrain):
        return brain_type(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=seed,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=seed),
        )

    def test_training_protocol_remains_at_20ms(self):
        config = WorkingMemoryLearningConfig()
        self.assertEqual(config.delay_ms, 20.0)
        self.assertEqual(config.training_trials, 800)

    def test_evaluation_delay_grid_is_exact(self):
        self.assertEqual(EVALUATION_DELAYS_MS, (0, 10, 20, 40, 80))

    def test_zero_delay_has_no_hidden_silent_step(self):
        brain = RecordingBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=101,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=101),
        )
        DelayedCueSession(brain, self.wm, delay_ms=0).run_trial("A")
        self.assertEqual(len(brain.calls), 200)
        self.assertTrue(all(rate > 0.0 for _, rate in brain.calls[:100]))
        self.assertTrue(all(rate > 0.0 for _, rate in brain.calls[100:]))

    def test_nonzero_delay_has_silent_external_input(self):
        brain = RecordingBrain(
            self.connectome,
            params=FlyBrainParams(dt_ms=0.2),
            seed=101,
            plasticity_config=PlasticStateConfig(plastic_fraction=0.05, seed=101),
        )
        DelayedCueSession(brain, self.wm, delay_ms=10).run_trial("A")
        self.assertEqual(len(brain.calls), 250)
        self.assertTrue(all(indices is None and rate == 0.0 for indices, rate in brain.calls[100:150]))

    def test_same_common_go_population_for_every_delay_and_cue(self):
        for cue in SYMBOLS:
            self.assertTrue(np.array_equal(self.wm.population_for_input("GO"), self.wm.go_population))
            self.assertFalse(np.array_equal(self.wm.population_for_input(cue), self.wm.go_population))

    def test_reset_removes_transient_state_only(self):
        brain = self.make_brain()
        session = DelayedCueSession(brain, self.wm, delay_ms=20)
        intact = session.run_trial("A", reset_before_go=False)
        reset = session.run_trial("A", reset_before_go=True)
        self.assertGreaterEqual(intact.pre_go_active_count, 0)
        self.assertEqual(reset.post_reset_active_count, 0)

    def test_evaluation_is_learning_disabled(self):
        session = DelayedCueSession(self.make_brain(), self.wm, delay_ms=80)
        self.assertFalse(session.learning_enabled)

    def test_normalized_retention_uses_each_seed_20ms_baseline(self):
        runs = {
            str(seed): {"evaluations": {
                "20": {"intact": {"accuracy": base, "target_minus_best_competitor_margin_hz": base * 10}},
                "40": {"intact": {"accuracy": base / 2, "target_minus_best_competitor_margin_hz": base * 5}},
                "0": {"intact": {"accuracy": base, "target_minus_best_competitor_margin_hz": base * 10}},
                "10": {"intact": {"accuracy": base, "target_minus_best_competitor_margin_hz": base * 10}},
                "80": {"intact": {"accuracy": base / 4, "target_minus_best_competitor_margin_hz": base * 2.5}},
            }} for seed, base in zip((101, 103, 107), (0.2, 0.4, 0.8))
        }
        curve = _retention_curve(runs, "accuracy")
        self.assertAlmostEqual(curve["101"]["40"], 0.5)
        self.assertAlmostEqual(curve["107"]["80"], 0.25)

    def test_memory_horizon_is_largest_tested_qualifying_delay(self):
        intact = {str(delay): value for delay, value in ((0, .9), (10, .9), (20, .8), (40, .7), (80, .6))}
        reset = {str(delay): value for delay, value in ((0, .1), (10, .1), (20, .1), (40, .1), (80, .1))}
        self.assertEqual(_memory_horizon(intact, reset), 40)

    def test_40ms_and_80ms_criteria_are_fixed(self):
        def row(acc):
            return {"accuracy": acc, "output_collapse": False}
        runs = {
            str(seed): {"evaluations": {
                str(delay): {"intact": row(.8 if delay == 20 else .6), "reset": row(.1)}
                for delay in EVALUATION_DELAYS_MS
            }} for seed in (101, 103, 107)
        }
        self.assertTrue(_criterion(runs, 40)["supported"])
        self.assertTrue(_criterion(runs, 80)["supported"])

    def test_activity_explosion_guard_uses_fourfold_ratio(self):
        # The helper receives compact summaries, so construct only the fields
        # consumed by the state-decay calculation.
        def arm(active, conductance):
            return {"state_memory": {
                "mean_pre_go_active_count": active,
                "mean_pre_go_membrane_norm": 1.0,
                "mean_pre_go_conductance_norm": conductance,
                "pairwise_pre_go_active_set_jaccard": {key: 0.0 for key in ("A|B", "A|C", "A|D", "B|C", "B|D", "C|D")},
                "pre_go_fingerprint_diversity": 4,
                "mean_post_reset_active_count": 0.0,
            }}
        runs = {
            str(seed): {"evaluations": {
                str(delay): {"intact": arm(10 if delay < 80 else 41, 2 if delay < 80 else 9), "reset": arm(0, 0)}
                for delay in EVALUATION_DELAYS_MS
            }} for seed in (101, 103, 107)
        }
        curve = _state_decay_curve(runs)
        self.assertGreater(curve["80"]["pre_go_active_count_ratio_vs_20ms"], 4.0)
        self.assertGreater(curve["80"]["pre_go_conductance_norm_ratio_vs_20ms"], 4.0)

    def test_neural_profiler_is_off_without_attachment(self):
        brain = self.make_brain()
        self.assertIsNone(brain._neural_timing_profiler)
        profiler = TimingProfiler(enabled=False)
        brain.configure_neural_timing(profiler)
        DelayedCueSession(brain, self.wm, delay_ms=0).run_trial("A")
        self.assertEqual(profiler.report()["seconds"], {})

    def test_neural_timing_sections_are_named_disjoint_phases(self):
        expected = {
            "stimulus_injection_seconds",
            "active_neuron_state_update_seconds",
            "synaptic_scheduling_seconds",
            "delay_ring_handling_seconds",
        }
        self.assertEqual(len(expected), 4)

    def test_neural_exact_replay_preserves_fired_state_and_rng(self):
        result = _neural_exact_replay(self.connectome, self.wm, self.interface_config)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["fired_indices_equal"])
        self.assertTrue(result["rng_equal"])


if __name__ == "__main__":
    unittest.main()
