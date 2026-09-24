from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Kept for compatibility with already-saved Stage 2.2A runs/code while the
# branch moves to the focused Stage 2.2B retest.
STAGE22A_PROFILES = (
    "position_only",
    "area_only",
    "spacing_only",
    "brightness_only",
    "combined",
)

STAGE22B_PROFILES = (
    "position_only",
    "brightness_only",
    "combined",
)


@dataclass(frozen=True)
class NumerosityConfig:
    seed: int = 7
    grid_size: int = 12
    n_kc: int = 512
    kc_active: int = 64
    input_fan_in_probability: float = 0.08
    learning_rate: float = 0.08
    output_decay: float = 0.99999
    policy_temperature: float = 1.0
    radius_jitter_min: float = 0.90
    radius_jitter_max: float = 1.10
    total_area_factor: float = 1.44
    signal_energy_min: float = 1.85
    signal_energy_max: float = 2.15
    pixel_noise_sd: float = 0.01
    translation_offsets: tuple[int, ...] = (-3, 0, 3)
    ood_small_area_range: tuple[float, float] = (0.88, 1.02)
    ood_large_area_range: tuple[float, float] = (1.90, 2.05)
    close_spacing_range: tuple[float, float] = (2.0, 3.0)
    strong_gain_faint_range: tuple[float, float] = (0.52, 0.72)
    strong_gain_bright_range: tuple[float, float] = (1.28, 1.48)

    # Stage 2.2B early-vision robustness. These operations never receive a
    # target label and never count components/peaks.
    contrast_mix: float = 0.30
    contrast_floor: float = 0.14
    phase_offsets: tuple[float, ...] = (-0.5, 0.0, 0.5)


class NumerosityExperiment:
    """Reward learner with a Stage-2.2B sub-pixel-stable visual front end.

    Stage 2.2A isolated the original OOD collapse and showed that half-grid
    positions were the dominant failure mode, with strong relative brightness a
    secondary problem. It also exposed a renderer confound: integer-center dots
    and half-grid dots were rasterized with different support rules.

    Stage 2.2B fixes that confound and strengthens early vision without giving
    the learner a numerosity answer:

    1. Every dot, integer or fractional, is sampled by the same continuous
       Gaussian rasterizer.
    2. A small binomial anti-alias filter reduces sampling-phase artifacts.
    3. Mild local divisive contrast normalization makes faint local structure
       less likely to disappear; global L1 energy is restored afterward.
    4. A local half-pixel phase envelope pools small retinal phase changes and
       again restores global L1 energy.
    5. The existing translated sparse receptor bank -> KC representation and
       reward-modulated choice learner remain unchanged in principle.

    No connected-component count, peak count, target-derived feature, or answer
    label is inserted into the encoder or the plasticity update.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, config: NumerosityConfig | None = None) -> None:
        self.config = config or NumerosityConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)
        self.n_input = c.grid_size * c.grid_size

        mask = self.rng.random((c.n_kc, self.n_input)) < c.input_fan_in_probability
        weights = self.rng.normal(0.60, 0.20, size=(c.n_kc, self.n_input))
        weights = np.maximum(weights, 0.05) * mask
        norm = np.sqrt(mask.sum(axis=1, keepdims=True))
        norm[norm == 0] = 1.0
        self.w_input = (weights / norm).astype(np.float32)

        self.w_output = self.rng.normal(0.0, 0.005, size=(3, c.n_kc)).astype(np.float32)
        self.output_bias = np.zeros(3, dtype=np.float32)

        pixel_y, pixel_x = np.mgrid[0 : c.grid_size, 0 : c.grid_size]
        self._pixel_x = pixel_x.astype(np.float32)
        self._pixel_y = pixel_y.astype(np.float32)
        self._blur_kernel = (
            np.array(
                [
                    [1.0, 2.0, 1.0],
                    [2.0, 4.0, 2.0],
                    [1.0, 2.0, 1.0],
                ],
                dtype=np.float32,
            )
            / 16.0
        )

    def save_checkpoint(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "metadata": metadata or {},
            "rng_state": self.rng.bit_generator.state,
        }
        np.savez_compressed(
            path,
            w_input=self.w_input,
            w_output=self.w_output,
            output_bias=self.output_bias,
            metadata_json=np.array(json.dumps(payload)),
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            if int(payload.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
                raise ValueError("Unsupported DrosoMath checkpoint version")
            self.w_input = data["w_input"].astype(np.float32, copy=True)
            self.w_output = data["w_output"].astype(np.float32, copy=True)
            self.output_bias = data["output_bias"].astype(np.float32, copy=True)
        self.rng.bit_generator.state = payload["rng_state"]
        return dict(payload.get("metadata") or {})

    @staticmethod
    def _profile_flags(stimulus_profile: str) -> dict[str, bool]:
        if stimulus_profile == "train":
            return {
                "fractional_position": False,
                "novel_area": False,
                "close_spacing": False,
                "strong_brightness": False,
            }
        if stimulus_profile not in STAGE22A_PROFILES:
            raise ValueError(f"Unknown stimulus profile: {stimulus_profile}")
        return {
            "fractional_position": stimulus_profile in {"position_only", "combined"},
            "novel_area": stimulus_profile in {"area_only", "combined"},
            "close_spacing": stimulus_profile in {"spacing_only", "combined"},
            "strong_brightness": stimulus_profile in {"brightness_only", "combined"},
        }

    def _sample_positions(
        self,
        numerosity: int,
        *,
        stimulus_profile: str,
    ) -> list[tuple[float, float]]:
        c = self.config
        flags = self._profile_flags(stimulus_profile)
        if numerosity <= 0:
            return []

        if flags["fractional_position"]:
            candidates = [float(i) + 0.5 for i in range(1, c.grid_size - 2)]
        else:
            candidates = [float(i) for i in range(1, c.grid_size - 1)]

        if numerosity == 1:
            return [
                (
                    float(self.rng.choice(candidates)),
                    float(self.rng.choice(candidates)),
                )
            ]

        first = (
            float(self.rng.choice(candidates)),
            float(self.rng.choice(candidates)),
        )
        second = first

        if flags["close_spacing"]:
            low, high = c.close_spacing_range
            for _attempt in range(500):
                candidate = (
                    float(self.rng.choice(candidates)),
                    float(self.rng.choice(candidates)),
                )
                distance = math.dist(first, candidate)
                if low <= distance < high:
                    second = candidate
                    break
        else:
            for _attempt in range(300):
                candidate = (
                    float(self.rng.choice(candidates)),
                    float(self.rng.choice(candidates)),
                )
                if math.dist(first, candidate) >= 3.0:
                    second = candidate
                    break

        if second == first:
            for _attempt in range(300):
                candidate = (
                    float(self.rng.choice(candidates)),
                    float(self.rng.choice(candidates)),
                )
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
        if numerosity <= 0:
            return np.zeros(0, dtype=np.float32), 0.0

        flags = self._profile_flags(stimulus_profile)
        if flags["novel_area"]:
            area_range = (
                c.ood_small_area_range
                if bool(self.rng.integers(0, 2))
                else c.ood_large_area_range
            )
            total_area = float(self.rng.uniform(*area_range))
            raw = self.rng.uniform(0.78, 1.22, size=numerosity)
        else:
            total_area = c.total_area_factor
            raw = self.rng.uniform(c.radius_jitter_min, c.radius_jitter_max, size=numerosity)

        scale = math.sqrt(total_area / float(np.sum(raw * raw)))
        return (raw * scale).astype(np.float32), total_area

    def _raw_gains(self, numerosity: int, *, stimulus_profile: str) -> np.ndarray:
        c = self.config
        if numerosity <= 0:
            return np.zeros(0, dtype=np.float32)

        flags = self._profile_flags(stimulus_profile)
        if flags["strong_brightness"] and numerosity == 2:
            faint = float(self.rng.uniform(*c.strong_gain_faint_range))
            bright = float(self.rng.uniform(*c.strong_gain_bright_range))
            raw_gains = np.array([faint, bright], dtype=np.float32)
            self.rng.shuffle(raw_gains)
            return raw_gains
        if flags["strong_brightness"]:
            return self.rng.uniform(0.52, 1.48, size=numerosity).astype(np.float32)
        return self.rng.uniform(0.90, 1.10, size=numerosity).astype(np.float32)

    def _make_stimulus(
        self,
        numerosity: int,
        *,
        stimulus_profile: str = "train",
    ) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
        flags = self._profile_flags(stimulus_profile)
        c = self.config
        grid = np.zeros((c.grid_size, c.grid_size), dtype=np.float32)
        positions = self._sample_positions(numerosity, stimulus_profile=stimulus_profile)
        radii, requested_area = self._controlled_radii(
            numerosity,
            stimulus_profile=stimulus_profile,
        )
        raw_gains = self._raw_gains(numerosity, stimulus_profile=stimulus_profile)
        dot_payload: list[dict[str, float]] = []

        # Same total-energy distribution for 1 and 2 in every profile.
        target_energy = (
            float(self.rng.uniform(c.signal_energy_min, c.signal_energy_max))
            if numerosity
            else 0.0
        )

        # One continuous rasterizer is used for integer and fractional centers.
        # This removes the Stage-2.2A 3x3-vs-4x4 renderer confound.
        for dot_index, ((x, y), radius) in enumerate(zip(positions, radii, strict=True)):
            gain = float(raw_gains[dot_index])
            sigma2 = max(0.25, 0.60 * float(radius) * float(radius))
            d2 = (self._pixel_x - float(x)) ** 2 + (self._pixel_y - float(y)) ** 2
            grid += (gain * np.exp(-(d2 / sigma2))).astype(np.float32)

        raw_energy = float(np.sum(grid))
        energy_scale = target_energy / raw_energy if raw_energy > 0 else 0.0
        grid *= energy_scale

        for dot_index, ((x, y), radius) in enumerate(zip(positions, radii, strict=True)):
            shown_gain = float(raw_gains[dot_index]) * energy_scale
            dot_payload.append(
                {
                    "x": round((x + 0.5) / c.grid_size, 4),
                    "y": round((y + 0.5) / c.grid_size, 4),
                    "r": round(0.060 * float(radius), 4),
                    "gain": round(shown_gain, 4),
                }
            )

        signal_energy = float(np.sum(grid))
        area_factor = float(np.sum(radii * radii)) if radii.size else 0.0
        min_pair_distance = (
            min(
                math.dist(positions[i], positions[j])
                for i in range(len(positions))
                for j in range(i + 1, len(positions))
            )
            if len(positions) >= 2
            else None
        )
        gain_ratio = (
            float(np.max(raw_gains) / max(float(np.min(raw_gains)), 1e-8))
            if len(raw_gains) >= 2
            else None
        )

        if c.pixel_noise_sd > 0:
            grid += self.rng.normal(
                0.0,
                c.pixel_noise_sd,
                size=grid.shape,
            ).astype(np.float32)
        np.clip(grid, 0.0, 1.0, out=grid)

        controls: dict[str, Any] = {
            "stimulus_profile": stimulus_profile,
            "signal_energy": round(signal_energy, 5),
            "area_factor": round(area_factor, 5),
            "requested_area_factor": round(requested_area, 5),
            "post_noise_energy": round(float(np.sum(grid)), 5),
            "min_pair_distance": round(min_pair_distance, 5) if min_pair_distance is not None else None,
            "gain_ratio": round(gain_ratio, 5) if gain_ratio is not None else None,
            "fractional_position": flags["fractional_position"],
            "novel_area": flags["novel_area"],
            "close_spacing": flags["close_spacing"],
            "strong_brightness": flags["strong_brightness"],
            "rasterizer": "continuous_gaussian_all_positions",
            "visual_preprocess": "antialias_divisive_contrast_phase_pool_v1",
        }
        return grid.reshape(-1), dot_payload, controls

    @staticmethod
    def _zero_fill_shift(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate a retinal image without wraparound."""
        height, width = grid.shape
        shifted = np.zeros_like(grid)

        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)

        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)

        if dst_y0 < dst_y1 and dst_x0 < dst_x1:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = grid[
                src_y0:src_y1,
                src_x0:src_x1,
            ]
        return shifted

    def _blur3(self, grid: np.ndarray) -> np.ndarray:
        padded = np.pad(grid, 1, mode="constant")
        result = np.zeros_like(grid)
        for ky in range(3):
            for kx in range(3):
                result += self._blur_kernel[ky, kx] * padded[
                    ky : ky + grid.shape[0],
                    kx : kx + grid.shape[1],
                ]
        return result

    @staticmethod
    def _restore_l1(reference: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        reference_sum = float(np.sum(reference))
        transformed_sum = float(np.sum(transformed))
        if reference_sum <= 0.0 or transformed_sum <= 1e-12:
            return transformed.astype(np.float32, copy=False)
        return (transformed * (reference_sum / transformed_sum)).astype(np.float32)

    def _half_phase_shift(self, grid: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Cheap bilinear half-pixel shift used only for local phase pooling."""
        shifted = grid
        if dx != 0.0:
            integer_dx = 1 if dx > 0 else -1
            shifted = 0.5 * shifted + 0.5 * self._zero_fill_shift(
                shifted,
                dx=integer_dx,
                dy=0,
            )
        if dy != 0.0:
            integer_dy = 1 if dy > 0 else -1
            shifted = 0.5 * shifted + 0.5 * self._zero_fill_shift(
                shifted,
                dx=0,
                dy=integer_dy,
            )
        return shifted.astype(np.float32, copy=False)

    def _preprocess_retina(self, grid: np.ndarray) -> np.ndarray:
        """Stage-2.2B anti-alias/contrast/phase front end with L1 preservation."""
        c = self.config

        # Mild anti-aliasing: binomial 3x3 low-pass, then restore total energy.
        antialiased = self._restore_l1(grid, self._blur3(grid))

        # Local divisive normalization. A mild mix avoids turning this into an
        # engineered object detector; L1 restoration prevents a new global-energy
        # shortcut between 1 and 2.
        local_rms = np.sqrt(self._blur3(antialiased * antialiased) + 1e-6)
        normalized = antialiased / (c.contrast_floor + local_rms)
        normalized = self._restore_l1(antialiased, normalized)
        contrast_stable = (
            (1.0 - c.contrast_mix) * antialiased + c.contrast_mix * normalized
        ).astype(np.float32)
        contrast_stable = self._restore_l1(antialiased, contrast_stable)

        # Local phase envelope. This is not a count: it only makes a feature at
        # x and x+0.5 produce a more similar retinal representation.
        phase_views = [
            self._half_phase_shift(contrast_stable, dx, dy)
            for dy in c.phase_offsets
            for dx in c.phase_offsets
        ]
        phase_pooled = np.max(np.stack(phase_views, axis=0), axis=0)
        return self._restore_l1(contrast_stable, phase_pooled)

    def _encode(self, stimulus: np.ndarray) -> np.ndarray:
        c = self.config
        grid = stimulus.reshape(c.grid_size, c.grid_size)
        grid = self._preprocess_retina(grid)

        receptor_views: list[np.ndarray] = []
        for dy in c.translation_offsets:
            for dx in c.translation_offsets:
                shifted = self._zero_fill_shift(grid, dx=dx, dy=dy)
                response = np.maximum(0.0, self.w_input @ shifted.reshape(-1))
                receptor_views.append(response)

        raw = np.max(np.stack(receptor_views, axis=0), axis=0)
        k = min(c.kc_active, c.n_kc)
        active_idx = np.argpartition(raw, -k)[-k:]
        activity = np.zeros(c.n_kc, dtype=np.float32)
        activity[active_idx] = raw[active_idx]
        return activity

    def _policy(self, kc_activity: np.ndarray) -> np.ndarray:
        c = self.config
        logits = (self.w_output @ kc_activity + self.output_bias) / c.policy_temperature
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        return exp / float(np.sum(exp))

    def step(
        self,
        trial: int,
        *,
        learn: bool = True,
        trial_kind: str = "train",
        stimulus_profile: str = "train",
    ) -> dict[str, Any]:
        c = self.config
        target = int(self.rng.integers(0, 3))
        stimulus, dots, controls = self._make_stimulus(
            target,
            stimulus_profile=stimulus_profile,
        )
        kc_activity = self._encode(stimulus)
        probabilities = self._policy(kc_activity)
        choice = int(self.rng.choice(3, p=probabilities))
        correct = choice == target
        reward = 1.0 if correct else -1.0

        eligibility = -probabilities.astype(np.float32)
        eligibility[choice] += 1.0
        delta = c.learning_rate * reward * eligibility[:, None] * kc_activity[None, :]

        if learn:
            self.w_output += delta
            self.output_bias += c.learning_rate * 0.05 * reward * eligibility
            self.w_output *= c.output_decay
        else:
            delta = np.zeros_like(delta)

        active_kc = np.flatnonzero(kc_activity > 0)
        max_kc = float(kc_activity[active_kc].max()) if active_kc.size else 1.0
        sparse_kc = [
            [int(i), round(float(kc_activity[i] / max(max_kc, 1e-8)), 3)]
            for i in active_kc
        ]

        changed = np.abs(delta) > 1e-8
        changed_values = np.abs(delta[changed])
        mean_delta = float(changed_values.mean()) if changed_values.size else 0.0
        entropy = -float(
            np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
        )

        return {
            "trial": trial,
            "trial_kind": trial_kind,
            "learning_enabled": learn,
            "target": target,
            "choice": choice,
            "correct": correct,
            "reward": reward,
            "stimulus": {
                "kind": "dots",
                "numerosity": target,
                "dots": dots,
                "controls": controls,
            },
            "policy": {
                "p0": round(float(probabilities[0]), 5),
                "p1": round(float(probabilities[1]), 5),
                "p2": round(float(probabilities[2]), 5),
                "entropy": round(entropy, 5),
            },
            "kc_activity": sparse_kc,
            "plasticity": {
                "mean_delta_w": round(mean_delta, 6),
                "active_synapses": int(changed.sum()),
            },
        }

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        offsets = list(c.translation_offsets)
        return {
            "stage": 2.2,
            "substage": "2.2B_robust_visual_encoder",
            "seed": c.seed,
            "grid_size": c.grid_size,
            "n_kc": c.n_kc,
            "kc_active": c.kc_active,
            "input_fan_in_probability": c.input_fan_in_probability,
            "learning_rate": c.learning_rate,
            "output_decay": c.output_decay,
            "policy_temperature": c.policy_temperature,
            "training_radius_jitter_range": [c.radius_jitter_min, c.radius_jitter_max],
            "training_total_area_factor_nonzero": c.total_area_factor,
            "signal_energy_range_nonzero": [c.signal_energy_min, c.signal_energy_max],
            "pixel_noise_sd": c.pixel_noise_sd,
            "visual_encoder": {
                "kind": "continuous_gaussian_antialias_contrast_phasepool_sparse_receptors",
                "rasterizer": "same_continuous_gaussian_for_integer_and_fractional_centers",
                "antialias_kernel": "3x3_binomial",
                "local_divisive_contrast_mix": c.contrast_mix,
                "local_divisive_contrast_floor": c.contrast_floor,
                "phase_offsets": list(c.phase_offsets),
                "phase_pool": "local_max_then_global_L1_restore",
                "translation_offsets_x": offsets,
                "translation_offsets_y": offsets,
                "translated_receptor_view_count": len(offsets) * len(offsets),
                "explicit_object_counter": False,
                "explicit_peak_counter": False,
                "explicit_numerosity_feature": False,
            },
            "stage_2_2b_profiles": {
                "position_only": {
                    "fractional_half_grid_positions": True,
                },
                "brightness_only": {
                    "faint_gain_range": list(c.strong_gain_faint_range),
                    "bright_gain_range": list(c.strong_gain_bright_range),
                },
                "combined": {
                    "fractional_half_grid_positions": True,
                    "small_total_area_range": list(c.ood_small_area_range),
                    "large_total_area_range": list(c.ood_large_area_range),
                    "two_dot_spacing_range": list(c.close_spacing_range),
                    "strong_relative_brightness": True,
                },
                "learning_enabled_during_ood": False,
            },
            "continuous_cue_controls": {
                "equalize_signal_energy_distribution_for_1_vs_2": True,
                "area_distribution_is_identical_for_1_vs_2_within_each_profile": True,
                "preprocessing_restores_global_L1_after_contrast_and_phase_pool": True,
            },
        }
