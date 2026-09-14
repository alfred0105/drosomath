from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .numerosity_stage22c import Stage22CConfig, Stage22CExperiment


STAGE22D_TRAIN_PROFILES = (
    "train_brightness_weak",
    "train_brightness_medium",
    "train_brightness_strong",
)

STAGE22D_PROFILES = (
    "position_only",
    "brightness_only",
    "combined",
)


@dataclass(frozen=True)
class Stage22DConfig(Stage22CConfig):
    """Final 0-2 curriculum: progressively increase relative brightness asymmetry."""

    weak_gain_faint_range: tuple[float, float] = (0.78, 0.96)
    weak_gain_bright_range: tuple[float, float] = (1.04, 1.22)

    medium_gain_faint_range: tuple[float, float] = (0.62, 0.82)
    medium_gain_bright_range: tuple[float, float] = (1.18, 1.38)

    strong_gain_faint_range: tuple[float, float] = (0.50, 0.70)
    strong_gain_bright_range: tuple[float, float] = (1.30, 1.52)


class Stage22DExperiment(Stage22CExperiment):
    """Stage 2.2D learner with curriculum-trained brightness invariance.

    The visual front end is unchanged from Stage 2.2C. Only the sensory training
    distribution changes: the two-dot relative-brightness challenge is raised in
    three 20k phases. Position jitter, area variation, spacing variation, global
    energy controls, KC sparsification, and reward-only learning remain intact.

    No object count, peak count, component count, or target-derived numerosity
    feature is supplied to the learner.
    """

    CHECKPOINT_VERSION = 3

    def __init__(self, config: Stage22DConfig | None = None) -> None:
        super().__init__(config or Stage22DConfig())

    @staticmethod
    def _profile_flags(stimulus_profile: str) -> dict[str, bool]:
        if stimulus_profile in STAGE22D_TRAIN_PROFILES:
            return {
                "fractional_position": True,
                "novel_area": False,
                "close_spacing": False,
                "strong_brightness": False,
            }
        if stimulus_profile not in STAGE22D_PROFILES:
            raise ValueError(f"Unknown Stage 2.2D stimulus profile: {stimulus_profile}")
        return {
            "fractional_position": stimulus_profile in {"position_only", "combined"},
            "novel_area": stimulus_profile == "combined",
            "close_spacing": stimulus_profile == "combined",
            "strong_brightness": stimulus_profile in {"brightness_only", "combined"},
        }

    def _phase_ranges(self, stimulus_profile: str) -> tuple[tuple[float, float], tuple[float, float]]:
        c = self.config
        assert isinstance(c, Stage22DConfig)
        if stimulus_profile == "train_brightness_weak":
            return c.weak_gain_faint_range, c.weak_gain_bright_range
        if stimulus_profile == "train_brightness_medium":
            return c.medium_gain_faint_range, c.medium_gain_bright_range
        if stimulus_profile == "train_brightness_strong":
            return c.strong_gain_faint_range, c.strong_gain_bright_range
        if stimulus_profile == "position_only":
            # Position is the only held-out factor here; brightness stays inside
            # the hardest training envelope reached before frozen evaluation.
            return c.strong_gain_faint_range, c.strong_gain_bright_range
        raise ValueError(f"No training brightness ranges for {stimulus_profile}")

    def _raw_gains(self, numerosity: int, *, stimulus_profile: str) -> np.ndarray:
        c = self.config
        assert isinstance(c, Stage22DConfig)
        if numerosity <= 0:
            return np.zeros(0, dtype=np.float32)

        if stimulus_profile in {"brightness_only", "combined"} and numerosity == 2:
            gains = np.array(
                [
                    float(self.rng.uniform(*c.stage22c_ood_gain_faint_range)),
                    float(self.rng.uniform(*c.stage22c_ood_gain_bright_range)),
                ],
                dtype=np.float32,
            )
            self.rng.shuffle(gains)
            return gains

        if numerosity == 2:
            faint_range, bright_range = self._phase_ranges(stimulus_profile)
            gains = np.array(
                [
                    float(self.rng.uniform(*faint_range)),
                    float(self.rng.uniform(*bright_range)),
                ],
                dtype=np.float32,
            )
            self.rng.shuffle(gains)
            return gains

        # Global energy is normalized after rasterization, so this is nuisance
        # variation rather than a numerosity cue.
        return self.rng.uniform(0.50, 1.52, size=numerosity).astype(np.float32)

    @staticmethod
    def curriculum_phase(stimulus_profile: str) -> str:
        return {
            "train_brightness_weak": "weak_0_20k",
            "train_brightness_medium": "medium_20_40k",
            "train_brightness_strong": "strong_40_60k",
            "position_only": "frozen_position_ood",
            "brightness_only": "frozen_brightness_ood",
            "combined": "frozen_combined_ood",
        }[stimulus_profile]

    def _make_stimulus(
        self,
        numerosity: int,
        *,
        stimulus_profile: str = "train_brightness_weak",
    ) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
        stimulus, dots, controls = super()._make_stimulus(
            numerosity,
            stimulus_profile=stimulus_profile,
        )
        c = self.config
        assert isinstance(c, Stage22DConfig)
        controls.update(
            {
                "curriculum": "stage2_2d_progressive_brightness",
                "curriculum_phase": self.curriculum_phase(stimulus_profile),
                "progressive_brightness_training": stimulus_profile in STAGE22D_TRAIN_PROFILES,
                "extreme_brightness_ood": stimulus_profile in {"brightness_only", "combined"},
                "training_gain_envelopes": {
                    "weak": {
                        "faint": list(c.weak_gain_faint_range),
                        "bright": list(c.weak_gain_bright_range),
                    },
                    "medium": {
                        "faint": list(c.medium_gain_faint_range),
                        "bright": list(c.medium_gain_bright_range),
                    },
                    "strong": {
                        "faint": list(c.strong_gain_faint_range),
                        "bright": list(c.strong_gain_bright_range),
                    },
                },
            }
        )
        return stimulus, dots, controls

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        assert isinstance(c, Stage22DConfig)
        payload = super().config_dict()
        payload["substage"] = "2.2D_progressive_brightness_curriculum"
        payload.pop("stage_2_2c_training", None)
        payload.pop("stage_2_2c_profiles", None)
        payload["stage_2_2d_training"] = {
            "total_trials": 60_000,
            "phase_boundaries": [20_000, 40_000, 60_000],
            "phase_profiles": list(STAGE22D_TRAIN_PROFILES),
            "position_jitter": c.training_position_jitter,
            "total_area_range_identical_for_1_vs_2": list(c.training_total_area_range),
            "close_spacing_probability": c.training_close_spacing_probability,
            "close_spacing_range": list(c.training_close_spacing_range),
            "brightness_curriculum": {
                "weak": {
                    "faint": list(c.weak_gain_faint_range),
                    "bright": list(c.weak_gain_bright_range),
                },
                "medium": {
                    "faint": list(c.medium_gain_faint_range),
                    "bright": list(c.medium_gain_bright_range),
                },
                "strong": {
                    "faint": list(c.strong_gain_faint_range),
                    "bright": list(c.strong_gain_bright_range),
                },
            },
        }
        payload["stage_2_2d_profiles"] = {
            "position_only": {
                "held_out_half_grid_phase": True,
                "brightness": "within strongest training envelope",
            },
            "brightness_only": {
                "faint_gain_range": list(c.stage22c_ood_gain_faint_range),
                "bright_gain_range": list(c.stage22c_ood_gain_bright_range),
                "other_nuisance_factors": "training_distribution",
            },
            "combined": {
                "held_out_half_grid_phase": True,
                "small_total_area_range": list(c.ood_small_area_range),
                "large_total_area_range": list(c.ood_large_area_range),
                "two_dot_spacing_range": list(c.close_spacing_range),
                "faint_gain_range": list(c.stage22c_ood_gain_faint_range),
                "bright_gain_range": list(c.stage22c_ood_gain_bright_range),
            },
            "learning_enabled_during_ood": False,
        }
        payload["continuous_cue_controls"] = {
            "equalize_signal_energy_distribution_for_1_vs_2": True,
            "area_distribution_is_identical_for_1_vs_2_within_each_profile": True,
            "preprocessing_restores_global_L1_after_contrast_and_phase_pool": True,
            "explicit_object_counter": False,
            "explicit_peak_counter": False,
            "explicit_numerosity_feature": False,
        }
        return payload
