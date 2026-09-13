from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STAGE22A_PROFILES = (
    "position_only",
    "area_only",
    "spacing_only",
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


class NumerosityExperiment:
    """Reward learner used for Stage 2.1 acquisition and Stage 2.2A tests.

    Stage 2.2 showed that changing several nuisance factors at once causes a
    catastrophic 2->1 collapse. Stage 2.2A therefore isolates those factors.
    The same frozen Stage-2.1 learner can be evaluated on five profiles:

    - ``position_only``: held-out half-grid centers, otherwise training-like.
    - ``area_only``: held-out total-area bands, otherwise training-like.
    - ``spacing_only``: held-out close two-dot spacing, otherwise training-like.
    - ``brightness_only``: stronger relative two-dot brightness asymmetry.
    - ``combined``: all four nuisance transforms together.

    The integrated signal-energy distribution remains identical for 1- and
    2-dot stimuli in every profile. The encoder never receives the answer label
    and does not explicitly count connected components or peaks.
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

        for dot_index, ((x, y), radius) in enumerate(zip(positions, radii, strict=True)):
            gain = float(raw_gains[dot_index])
            sigma2 = max(0.25, 0.60 * float(radius) * float(radius))

            if not flags["fractional_position"]:
                cx, cy = int(x), int(y)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        value = gain * math.exp(-((dx * dx + dy * dy) / sigma2))
                        grid[cy + dy, cx + dx] += value
            else:
                x0, y0 = math.floor(x), math.floor(y)
                for px in range(x0 - 1, x0 + 3):
                    for py in range(y0 - 1, y0 + 3):
                        if not (0 <= px < c.grid_size and 0 <= py < c.grid_size):
                            continue
                        d2 = (float(px) - x) ** 2 + (float(py) - y) ** 2
                        grid[py, px] += gain * math.exp(-(d2 / sigma2))

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

    def _encode(self, stimulus: np.ndarray) -> np.ndarray:
        c = self.config
        grid = stimulus.reshape(c.grid_size, c.grid_size)

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
            "substage": "2.2A_factor_isolation",
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
                "kind": "fixed_sparse_receptors_with_translation_max_pooling",
                "translation_offsets_x": offsets,
                "translation_offsets_y": offsets,
                "view_count": len(offsets) * len(offsets),
                "explicit_object_counter": False,
                "explicit_peak_counter": False,
            },
            "stage_2_2a_profiles": {
                "position_only": {
                    "fractional_half_grid_positions": True,
                },
                "area_only": {
                    "small_total_area_range": list(c.ood_small_area_range),
                    "large_total_area_range": list(c.ood_large_area_range),
                },
                "spacing_only": {
                    "two_dot_spacing_range": list(c.close_spacing_range),
                },
                "brightness_only": {
                    "faint_gain_range": list(c.strong_gain_faint_range),
                    "bright_gain_range": list(c.strong_gain_bright_range),
                },
                "combined": {
                    "all_above_transforms": True,
                },
                "learning_enabled": False,
            },
            "continuous_cue_controls": {
                "equalize_signal_energy_distribution_for_1_vs_2": True,
                "area_distribution_is_identical_for_1_vs_2_within_each_profile": True,
            },
        }
