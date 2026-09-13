from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NumerosityConfig:
    seed: int = 7
    grid_size: int = 12
    n_kc: int = 512
    kc_active: int = 64
    input_fan_in_probability: float = 0.08
    learning_rate: float = 0.14
    output_decay: float = 0.99999
    policy_temperature: float = 1.0
    radius_jitter_min: float = 0.90
    radius_jitter_max: float = 1.10
    total_area_factor: float = 1.44
    signal_energy_min: float = 1.85
    signal_energy_max: float = 2.15
    pixel_noise_sd: float = 0.01


class NumerosityExperiment:
    """Stage-2 reward learner with continuous visual cues controlled.

    This is still a protocol-validation prototype, not the full FlyWire neural
    simulation. For non-zero stimuli, 1-dot and 2-dot displays are constructed
    so total dot area and integrated signal energy are matched up to small
    trial-to-trial jitter that is sampled independently of numerosity.

    Dot locations remain randomized. A one-dot display therefore tends to use a
    larger/brighter dot while a two-dot display uses two smaller/dimmer dots.
    The learner cannot solve 1 versus 2 simply by total brightness or total area.

    Learning remains reinforcement-only: the target class is used only to turn
    the sampled choice into +1/-1 reward. Probe trials can call step(...,
    learn=False), which evaluates the current policy without changing weights.
    """

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

    def _sample_positions(self, numerosity: int) -> list[tuple[int, int]]:
        c = self.config
        placed: list[tuple[int, int]] = []
        for _ in range(numerosity):
            x = y = 1
            for _attempt in range(200):
                x = int(self.rng.integers(1, c.grid_size - 1))
                y = int(self.rng.integers(1, c.grid_size - 1))
                if all((x - px) ** 2 + (y - py) ** 2 >= 9 for px, py in placed):
                    break
            placed.append((x, y))
        return placed

    def _controlled_radii(self, numerosity: int) -> np.ndarray:
        c = self.config
        if numerosity <= 0:
            return np.zeros(0, dtype=np.float32)
        raw = self.rng.uniform(c.radius_jitter_min, c.radius_jitter_max, size=numerosity)
        scale = math.sqrt(c.total_area_factor / float(np.sum(raw * raw)))
        return (raw * scale).astype(np.float32)

    def _make_stimulus(self, numerosity: int) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
        c = self.config
        grid = np.zeros((c.grid_size, c.grid_size), dtype=np.float32)
        positions = self._sample_positions(numerosity)
        radii = self._controlled_radii(numerosity)
        dot_payload: list[dict[str, float]] = []

        # Independent per-trial target energy: its distribution is identical for
        # 1-dot and 2-dot stimuli, so energy itself carries no count information.
        target_energy = float(self.rng.uniform(c.signal_energy_min, c.signal_energy_max)) if numerosity else 0.0
        raw_gains = self.rng.uniform(0.90, 1.10, size=numerosity) if numerosity else np.zeros(0)

        for dot_index, ((x, y), radius) in enumerate(zip(positions, radii, strict=True)):
            gain = float(raw_gains[dot_index])
            sigma2 = max(0.25, 0.60 * float(radius) * float(radius))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    value = gain * math.exp(-((dx * dx + dy * dy) / sigma2))
                    grid[y + dy, x + dx] += value

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

        if c.pixel_noise_sd > 0:
            grid += self.rng.normal(0.0, c.pixel_noise_sd, size=grid.shape).astype(np.float32)
        np.clip(grid, 0.0, 1.0, out=grid)

        controls = {
            "signal_energy": round(signal_energy, 5),
            "area_factor": round(area_factor, 5),
            "post_noise_energy": round(float(np.sum(grid)), 5),
        }
        return grid.reshape(-1), dot_payload, controls

    def _encode(self, stimulus: np.ndarray) -> np.ndarray:
        c = self.config
        raw = np.maximum(0.0, self.w_input @ stimulus)
        k = min(c.kc_active, c.n_kc)
        active_idx = np.argpartition(raw, -k)[-k:]
        activity = np.zeros(c.n_kc, dtype=np.float32)
        activity[active_idx] = raw[active_idx]

        # Broad-field channels are retained as realistic nuisance sensors, but
        # Stage 2 equalizes their signal for 1 vs 2. They mainly separate zero
        # (absence) from non-zero stimuli and cannot directly encode 1 versus 2.
        energy = float(np.sum(stimulus)) / max(c.signal_energy_max, 1e-6)
        activity[0] = max(float(activity[0]), energy)
        activity[1] = max(float(activity[1]), math.sqrt(max(0.0, energy)))
        return activity

    def _policy(self, kc_activity: np.ndarray) -> np.ndarray:
        c = self.config
        logits = (self.w_output @ kc_activity + self.output_bias) / c.policy_temperature
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        return exp / float(np.sum(exp))

    def step(self, trial: int, *, learn: bool = True, trial_kind: str = "train") -> dict[str, Any]:
        c = self.config
        target = int(self.rng.integers(0, 3))
        stimulus, dots, controls = self._make_stimulus(target)
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
            self.output_bias += c.learning_rate * 0.10 * reward * eligibility
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
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))

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
        return {
            "stage": 2,
            "seed": c.seed,
            "grid_size": c.grid_size,
            "n_kc": c.n_kc,
            "kc_active": c.kc_active,
            "input_fan_in_probability": c.input_fan_in_probability,
            "learning_rate": c.learning_rate,
            "output_decay": c.output_decay,
            "policy_temperature": c.policy_temperature,
            "radius_jitter_range": [c.radius_jitter_min, c.radius_jitter_max],
            "total_area_factor_nonzero": c.total_area_factor,
            "signal_energy_range_nonzero": [c.signal_energy_min, c.signal_energy_max],
            "pixel_noise_sd": c.pixel_noise_sd,
            "continuous_cue_controls": {
                "equalize_total_area_for_1_vs_2": True,
                "equalize_signal_energy_distribution_for_1_vs_2": True,
                "randomize_positions": True,
            },
        }
