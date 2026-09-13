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
    learning_rate: float = 0.18
    output_decay: float = 0.99999
    policy_temperature: float = 1.0
    dot_gain_min: float = 0.90
    dot_gain_max: float = 1.10
    dot_radius_min: float = 0.90
    dot_radius_max: float = 1.10
    pixel_noise_sd: float = 0.01


class NumerosityExperiment:
    """Reward-modulated 0/1/2-dot learner for DrosoMath Stage 1.

    This is a protocol-validation learner, not yet the whole FlyWire connectome.
    Random dot positions are projected through a fixed sparse expansion layer.
    Two broad-field sensory channels intentionally preserve continuous visual
    cues (total energy and a nonlinear transform) in Stage 1. Stage 2 will
    equalize those cues to test whether performance generalizes to numerosity
    itself rather than brightness/area.

    Only the sparse-code -> choice weights are plastic. The update is
    reinforcement-only: the target class is never inserted directly into the
    learning rule; the learner receives only its chosen action and +/- reward.
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

    def _make_stimulus(self, numerosity: int) -> tuple[np.ndarray, list[dict[str, float]]]:
        c = self.config
        grid = np.zeros((c.grid_size, c.grid_size), dtype=np.float32)
        placed: list[tuple[int, int]] = []
        dots: list[dict[str, float]] = []

        for _ in range(numerosity):
            x = y = 1
            for _attempt in range(100):
                x = int(self.rng.integers(1, c.grid_size - 1))
                y = int(self.rng.integers(1, c.grid_size - 1))
                if all((x - px) ** 2 + (y - py) ** 2 >= 9 for px, py in placed):
                    break
            placed.append((x, y))

            gain = float(self.rng.uniform(c.dot_gain_min, c.dot_gain_max))
            radius = float(self.rng.uniform(c.dot_radius_min, c.dot_radius_max))
            dots.append(
                {
                    "x": round((x + 0.5) / c.grid_size, 4),
                    "y": round((y + 0.5) / c.grid_size, 4),
                    "r": round(0.060 * radius, 4),
                    "gain": round(gain, 3),
                }
            )

            sigma2 = max(0.35, 0.60 * radius * radius)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    value = gain * math.exp(-((dx * dx + dy * dy) / sigma2))
                    grid[y + dy, x + dx] += value

        if c.pixel_noise_sd > 0:
            grid += self.rng.normal(0.0, c.pixel_noise_sd, size=grid.shape).astype(np.float32)
        np.clip(grid, 0.0, 1.0, out=grid)
        return grid.reshape(-1), dots

    def _encode(self, stimulus: np.ndarray) -> np.ndarray:
        c = self.config
        raw = np.maximum(0.0, self.w_input @ stimulus)
        k = min(c.kc_active, c.n_kc)
        active_idx = np.argpartition(raw, -k)[-k:]
        activity = np.zeros(c.n_kc, dtype=np.float32)
        activity[active_idx] = raw[active_idx]

        # Stage-1 broad-field visual channels. These do not contain the answer;
        # they expose sensory energy that naturally covaries with dot number.
        # Stage 2 will remove/equalize these continuous cues.
        energy = float(np.sum(stimulus)) / 5.0
        activity[0] = max(float(activity[0]), energy)
        activity[1] = max(float(activity[1]), min(4.0, energy * energy))
        return activity

    def _policy(self, kc_activity: np.ndarray) -> np.ndarray:
        c = self.config
        logits = (self.w_output @ kc_activity + self.output_bias) / c.policy_temperature
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        return exp / float(np.sum(exp))

    def step(self, trial: int) -> dict[str, Any]:
        c = self.config
        target = int(self.rng.integers(0, 3))
        stimulus, dots = self._make_stimulus(target)
        kc_activity = self._encode(stimulus)
        probabilities = self._policy(kc_activity)
        choice = int(self.rng.choice(3, p=probabilities))
        correct = choice == target
        reward = 1.0 if correct else -1.0

        # Reward-modulated policy/eligibility update. Only the sampled action and
        # reward enter the update; target is used solely to generate the reward.
        eligibility = -probabilities.astype(np.float32)
        eligibility[choice] += 1.0
        delta = c.learning_rate * reward * eligibility[:, None] * kc_activity[None, :]
        self.w_output += delta
        self.output_bias += c.learning_rate * 0.10 * reward * eligibility
        self.w_output *= c.output_decay

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
            "target": target,
            "choice": choice,
            "correct": correct,
            "reward": reward,
            "stimulus": {
                "kind": "dots",
                "numerosity": target,
                "dots": dots,
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
            "seed": c.seed,
            "grid_size": c.grid_size,
            "n_kc": c.n_kc,
            "kc_active": c.kc_active,
            "input_fan_in_probability": c.input_fan_in_probability,
            "learning_rate": c.learning_rate,
            "output_decay": c.output_decay,
            "policy_temperature": c.policy_temperature,
            "dot_gain_range": [c.dot_gain_min, c.dot_gain_max],
            "dot_radius_range": [c.dot_radius_min, c.dot_radius_max],
            "pixel_noise_sd": c.pixel_noise_sd,
            "broad_field_sensory_channels": 2,
            "stage1_continuous_cues_allowed": True,
        }
