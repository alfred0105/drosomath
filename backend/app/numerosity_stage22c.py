from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .numerosity import NumerosityConfig, NumerosityExperiment


STAGE22C_TRAIN_PROFILE = "train_augmented"
STAGE22C_PROFILES = (
    "position_only",
    "brightness_only",
    "combined",
)


@dataclass(frozen=True)
class Stage22CConfig(NumerosityConfig):
    """Stage 2.2C curriculum parameters.

    Training sees moderate nuisance variation. Frozen OOD evaluation deliberately
    exceeds that training envelope. All non-zero classes still share the same
    total-signal-energy distribution, and total-area distributions are identical
    for 1 and 2 within each profile.
    """

    training_position_jitter: float = 0.40
    training_total_area_range: tuple[float, float] = (1.20, 1.68)
    training_close_spacing_probability: float = 0.25
    training_close_spacing_range: tuple[float, float] = (2.40, 3.20)
    training_gain_faint_range: tuple[float, float] = (0.78, 0.96)
    training_gain_bright_range: tuple[float, float] = (1.04, 1.22)
    training_single_gain_range: tuple[float, float] = (0.78, 1.22)

    # Deliberately stronger than the augmentation seen during learning.
    stage22c_ood_gain_faint_range: tuple[float, float] = (0.42, 0.60)
    stage22c_ood_gain_bright_range: tuple[float, float] = (1.40, 1.62)


class Stage22CExperiment(NumerosityExperiment):
    """Stage 2.2C learner: train invariance rather than hand-coding a counter.

    The Stage-2.2B early-vision front end is retained. The change is the sensory
    curriculum: learning now receives moderate continuous position jitter,
    moderate per-dot brightness asymmetry, modest total-area variation, and a
    mixture of ordinary/closer two-dot spacing. The target label is still used
    only to issue reward; no numerosity, peak count, or component count enters
    the visual representation.
    """

    CHECKPOINT_VERSION = 2

    def __init__(self, config: Stage22CConfig | None = None) -> None:
        super().__init__(config or Stage22CConfig())

    @staticmethod
    def _profile_flags(stimulus_profile: str) -> dict[str, bool]:
        if stimulus_profile == STAGE22C_TRAIN_PROFILE:
            return {
                "fractional_position": True,
                "novel_area": False,
                "close_spacing": False,
                "strong_brightness": False,
            }
        if stimulus_profile not in STAGE22C_PROFILES:
            raise ValueError(f"Unknown Stage 2.2C stimulus profile: {stimulus_profile}")
        return {
            "fractional_position": stimulus_profile in {"position_only", "combined"},
            "novel_area": stimulus_profile == "combined",
            "close_spacing": stimulus_profile == "combined",
            "strong_brightness": stimulus_profile in {"brightness_only", "combined"},
        }

    def _training_like_point(self) -> tuple[float, float]:
        c = self.config
        assert isinstance(c, Stage22CConfig)
        margin_low = 1.35
        margin_high = float(c.grid_size) - 2.35
        base_x = float(self.rng.integers(2, c.grid_size - 2))
        base_y = float(self.rng.integers(2, c.grid_size - 2))
        x = base_x + float(self.rng.uniform(-c.training_position_jitter, c.training_position_jitter))
        y = base_y + float(self.rng.uniform(-c.training_position_jitter, c.training_position_jitter))
        return (
            float(np.clip(x, margin_low, margin_high)),
            float(np.clip(y, margin_low, margin_high)),
        )

    def _held_out_half_grid_point(self) -> tuple[float, float]:
        c = self.config
        candidates = [float(i) + 0.5 for i in range(1, c.grid_size - 2)]
        return (
            float(self.rng.choice(candidates)),
            float(self.rng.choice(candidates)),
        )

    def _sample_positions(
        self,
        numerosity: int,
        *,
        stimulus_profile: str,
    ) -> list[tuple[float, float]]:
        c = self.config
        assert isinstance(c, Stage22CConfig)
        if numerosity <= 0:
            return []

        point_sampler = (
            self._held_out_half_grid_point
            if stimulus_profile in {"position_only", "combined"}
            else self._training_like_point
        )
        first = point_sampler()
        if numerosity == 1:
            return [first]

        force_close = stimulus_profile == "combined"
        mixed_close = stimulus_profile != "combined" and (
            float(self.rng.random()) < c.training_close_spacing_probability
        )
        use_close = force_close or mixed_close
        if force_close:
            low, high = c.close_spacing_range
        elif mixed_close:
            low, high = c.training_close_spacing_range
        else:
            low, high = 3.0, float(c.grid_size)

        second = first
        for _attempt in range(600):
            candidate = point_sampler()
            distance = math.dist(first, candidate)
            if low <= distance < high:
                second = candidate
                break

        if second == first:
            for _attempt in range(300):
                candidate = point_sampler()
                if math.dist(first, candidate) >= 2.0:
                    second = candidate
                    break

        return [first, second]

    def _controlled_radii(
        self,
        numerosity: int,
        *,
        stimulus_profile: str,
    ) -> tuple[np.ndarray, float]:
        c = self.config
        assert isinstance(c, Stage22CConfig)
        if numerosity <= 0:
            return np.zeros(0, dtype=np.float32), 0.0

        if stimulus_profile == "combined":
            area_range = (
                c.ood_small_area_range
                if bool(self.rng.integers(0, 2))
                else c.ood_large_area_range
            )
            total_area = float(self.rng.uniform(*area_range))
            raw = self.rng.uniform(0.78, 1.22, size=numerosity)
        else:
            # Same distribution for 1 and 2; only the partition over dots differs.
            total_area = float(self.rng.uniform(*c.training_total_area_range))
            raw = self.rng.uniform(0.84, 1.16, size=numerosity)

        scale = math.sqrt(total_area / float(np.sum(raw * raw)))
        return (raw * scale).astype(np.float32), total_area

    def _raw_gains(self, numerosity: int, *, stimulus_profile: str) -> np.ndarray:
        c = self.config
        assert isinstance(c, Stage22CConfig)
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
            # Moderate asymmetry is part of training and is also used by the
            # position-only test so position remains the only held-out factor.
            gains = np.array(
                [
                    float(self.rng.uniform(*c.training_gain_faint_range)),
                    float(self.rng.uniform(*c.training_gain_bright_range)),
                ],
                dtype=np.float32,
            )
            self.rng.shuffle(gains)
            return gains

        return self.rng.uniform(*c.training_single_gain_range, size=numerosity).astype(np.float32)

    def _make_stimulus(
        self,
        numerosity: int,
        *,
        stimulus_profile: str = STAGE22C_TRAIN_PROFILE,
    ) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
        stimulus, dots, controls = super()._make_stimulus(
            numerosity,
            stimulus_profile=stimulus_profile,
        )
        c = self.config
        assert isinstance(c, Stage22CConfig)
        min_distance = controls.get("min_pair_distance")
        controls.update(
            {
                "curriculum": "stage2_2c_augmented_invariance",
                "training_position_jitter": c.training_position_jitter,
                "training_area_range": list(c.training_total_area_range),
                "training_close_spacing_probability": c.training_close_spacing_probability,
                "moderate_brightness_training": stimulus_profile
                in {STAGE22C_TRAIN_PROFILE, "position_only"},
                "extreme_brightness_ood": stimulus_profile
                in {"brightness_only", "combined"},
                "close_spacing_sampled": (
                    bool(min_distance is not None and float(min_distance) < 3.0)
                ),
            }
        )
        return stimulus, dots, controls

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        assert isinstance(c, Stage22CConfig)
        payload = super().config_dict()
        payload["substage"] = "2.2C_augmented_invariance_training"
        payload.pop("stage_2_2b_profiles", None)
        payload["stage_2_2c_training"] = {
            "profile": STAGE22C_TRAIN_PROFILE,
            "continuous_position_jitter": c.training_position_jitter,
            "total_area_range_identical_for_1_vs_2": list(c.training_total_area_range),
            "close_spacing_probability": c.training_close_spacing_probability,
            "close_spacing_range": list(c.training_close_spacing_range),
            "two_dot_moderate_gain_ranges": {
                "faint": list(c.training_gain_faint_range),
                "bright": list(c.training_gain_bright_range),
            },
            "single_dot_gain_range": list(c.training_single_gain_range),
        }
        payload["stage_2_2c_profiles"] = {
            "position_only": {
                "held_out_half_grid_phase": True,
                "other_nuisance_factors": "training_distribution",
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
