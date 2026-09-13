from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DIGIT_TOKENS = tuple(str(i) for i in range(10))
SPECIAL_TOKENS = ("+", "=", "NEXT")
ALL_TOKENS = DIGIT_TOKENS + SPECIAL_TOKENS
N_OUTPUTS = 19

# Whole unordered pairs are held out from addition training. Every resulting sum
# still appears in at least one training pair, so the held-out test measures
# composition across known symbols/answers rather than unseen output classes.
HELD_OUT_UNORDERED_PAIRS = {
    (0, 4),
    (0, 7),
    (1, 5),
    (1, 8),
    (2, 6),
    (2, 9),
    (3, 7),
    (4, 8),
    (5, 9),
    (6, 8),
}


def _addition_splits() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    train: list[tuple[int, int]] = []
    commutativity: list[tuple[int, int]] = []
    held_out: list[tuple[int, int]] = []

    for a in range(10):
        for b in range(a, 10):
            if (a, b) in HELD_OUT_UNORDERED_PAIRS:
                held_out.append((a, b))
                if a != b:
                    held_out.append((b, a))
                continue

            if a == b:
                train.append((a, b))
                continue

            # Pick one orientation deterministically but not always smaller-first.
            # The opposite orientation is never rewarded during addition training.
            if ((a * 11 + b * 7) % 2) == 0:
                train.append((a, b))
                commutativity.append((b, a))
            else:
                train.append((b, a))
                commutativity.append((a, b))

    represented_sums = {a + b for a, b in train}
    if represented_sums != set(range(N_OUTPUTS)):
        missing = sorted(set(range(N_OUTPUTS)) - represented_sums)
        raise RuntimeError(f"Stage 3S addition split lost output classes: {missing}")

    return tuple(train), tuple(commutativity), tuple(held_out)


ADDITION_TRAIN_PAIRS, ADDITION_COMMUTATIVITY_PAIRS, ADDITION_HELD_OUT_PAIRS = _addition_splits()


@dataclass(frozen=True)
class SymbolicStage3SConfig:
    seed: int = 31
    token_dim: int = 128
    token_active: int = 16
    sequence_slots: int = 4
    n_kc: int = 1024
    kc_active: int = 96
    input_fan_in_probability: float = 0.07
    learning_rate: float = 0.055
    output_decay: float = 0.999995
    policy_temperature: float = 0.85


class SymbolicStage3SExperiment:
    """Strict symbolic arithmetic baseline with arbitrary sparse token codes.

    The learner never receives a scalar number value, magnitude feature, sum,
    distance-to-answer, or an engineered ordinal embedding. Each visible symbol
    is represented by a fixed random sparse code. The target integer is used only
    by the external reward rule after the learner has selected one of 19 actions.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, config: SymbolicStage3SConfig | None = None) -> None:
        self.config = config or SymbolicStage3SConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)

        self.token_codes: dict[str, np.ndarray] = {}
        for token in ALL_TOKENS:
            code = np.zeros(c.token_dim, dtype=np.float32)
            active = self.rng.choice(c.token_dim, size=c.token_active, replace=False)
            code[active] = 1.0 / np.sqrt(float(c.token_active))
            self.token_codes[token] = code

        self.n_input = c.token_dim * c.sequence_slots
        mask = self.rng.random((c.n_kc, self.n_input)) < c.input_fan_in_probability
        weights = self.rng.normal(0.60, 0.20, size=(c.n_kc, self.n_input))
        weights = np.maximum(weights, 0.05) * mask
        norm = np.sqrt(mask.sum(axis=1, keepdims=True))
        norm[norm == 0] = 1.0
        self.w_input = (weights / norm).astype(np.float32)

        self.w_output = self.rng.normal(0.0, 0.004, size=(N_OUTPUTS, c.n_kc)).astype(np.float32)
        self.output_bias = np.zeros(N_OUTPUTS, dtype=np.float32)

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
                raise ValueError("Unsupported Stage 3S checkpoint version")
            self.w_input = data["w_input"].astype(np.float32, copy=True)
            self.w_output = data["w_output"].astype(np.float32, copy=True)
            self.output_bias = data["output_bias"].astype(np.float32, copy=True)
        self.rng.bit_generator.state = payload["rng_state"]
        return dict(payload.get("metadata") or {})

    def _sequence_vector(self, tokens: Iterable[str]) -> np.ndarray:
        c = self.config
        token_list = list(tokens)
        if len(token_list) > c.sequence_slots:
            raise ValueError(f"Too many Stage 3S tokens: {token_list}")

        sequence = np.zeros(self.n_input, dtype=np.float32)
        for slot, token in enumerate(token_list):
            if token not in self.token_codes:
                raise ValueError(f"Unknown Stage 3S token: {token}")
            start = slot * c.token_dim
            sequence[start : start + c.token_dim] = self.token_codes[token]
        return sequence

    def _encode(self, tokens: Iterable[str]) -> np.ndarray:
        x = self._sequence_vector(tokens)
        raw = np.maximum(0.0, self.w_input @ x)
        k = min(self.config.kc_active, self.config.n_kc)
        active_idx = np.argpartition(raw, -k)[-k:]
        activity = np.zeros(self.config.n_kc, dtype=np.float32)
        activity[active_idx] = raw[active_idx]
        return activity

    def _policy(self, kc_activity: np.ndarray) -> np.ndarray:
        c = self.config
        logits = (self.w_output @ kc_activity + self.output_bias) / c.policy_temperature
        logits -= float(np.max(logits))
        exp = np.exp(logits)
        return exp / float(np.sum(exp))

    @staticmethod
    def task_example(task: str, *, a: int | None = None, b: int | None = None) -> tuple[list[str], int, str]:
        if task == "identity":
            if a is None or not 0 <= a <= 9:
                raise ValueError("identity requires a digit 0..9")
            return [str(a)], a, str(a)

        if task == "successor":
            if a is None or not 0 <= a <= 8:
                raise ValueError("successor requires a digit 0..8")
            return [str(a), "NEXT"], a + 1, f"NEXT({a})"

        if task == "addition":
            if a is None or b is None or not (0 <= a <= 9 and 0 <= b <= 9):
                raise ValueError("addition requires digits 0..9")
            return [str(a), "+", str(b), "="], a + b, f"{a} + {b} = ?"

        raise ValueError(f"Unknown Stage 3S task: {task}")

    def step(
        self,
        trial: int,
        *,
        task: str,
        learn: bool = True,
        a: int | None = None,
        b: int | None = None,
        pair_pool: tuple[tuple[int, int], ...] | None = None,
        trial_kind: str = "train",
    ) -> dict[str, Any]:
        if task == "identity" and a is None:
            a = int(self.rng.integers(0, 10))
        elif task == "successor" and a is None:
            a = int(self.rng.integers(0, 9))
        elif task == "addition" and (a is None or b is None):
            pool = pair_pool or ADDITION_TRAIN_PAIRS
            a, b = pool[int(self.rng.integers(0, len(pool)))]

        tokens, target, expression = self.task_example(task, a=a, b=b)
        kc_activity = self._encode(tokens)
        probabilities = self._policy(kc_activity)
        choice = int(self.rng.choice(N_OUTPUTS, p=probabilities))
        correct = choice == target
        reward = 1.0 if correct else -1.0

        eligibility = -probabilities.astype(np.float32)
        eligibility[choice] += 1.0
        delta = self.config.learning_rate * reward * eligibility[:, None] * kc_activity[None, :]

        if learn:
            self.w_output += delta
            self.output_bias += self.config.learning_rate * 0.03 * reward * eligibility
            self.w_output *= self.config.output_decay
        else:
            delta = np.zeros_like(delta)

        active_kc = np.flatnonzero(kc_activity > 0)
        max_kc = float(kc_activity[active_kc].max()) if active_kc.size else 1.0
        sparse_kc = [
            [int(i), round(float(kc_activity[i] / max(max_kc, 1e-8)), 3)]
            for i in active_kc
        ]
        top_idx = np.argsort(probabilities)[-5:][::-1]
        top_policy = [
            {"choice": int(i), "p": round(float(probabilities[i]), 6)}
            for i in top_idx
        ]

        changed = np.abs(delta) > 1e-8
        changed_values = np.abs(delta[changed])
        mean_delta = float(changed_values.mean()) if changed_values.size else 0.0
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))

        return {
            "trial": trial,
            "trial_kind": trial_kind,
            "learning_enabled": learn,
            "task": task,
            "tokens": tokens,
            "expression": expression,
            "operand_a": a,
            "operand_b": b,
            "target": target,
            "choice": choice,
            "correct": correct,
            "reward": reward,
            "policy": {
                "top": top_policy,
                "entropy": round(entropy, 6),
                "chosen_probability": round(float(probabilities[choice]), 6),
                "target_probability": round(float(probabilities[target]), 6),
            },
            "kc_activity": sparse_kc,
            "plasticity": {
                "mean_delta_w": round(mean_delta, 7),
                "active_synapses": int(changed.sum()),
            },
        }

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        return {
            "stage": "3S",
            "substage": "symbolic_number_arithmetic_baseline",
            "seed": c.seed,
            "symbols": list(ALL_TOKENS),
            "outputs": list(range(N_OUTPUTS)),
            "chance_accuracy": 1.0 / N_OUTPUTS,
            "token_encoding": {
                "kind": "fixed_random_sparse_code",
                "token_dim": c.token_dim,
                "active_bits": c.token_active,
                "sequence_slots": c.sequence_slots,
                "scalar_number_value_injected": False,
                "ordinal_embedding_injected": False,
                "sum_feature_injected": False,
                "distance_to_answer_injected": False,
            },
            "learner": {
                "kind": "fixed_sparse_token_projection_to_kc_then_reward_modulated_softmax",
                "n_kc": c.n_kc,
                "kc_active": c.kc_active,
                "input_fan_in_probability": c.input_fan_in_probability,
                "learning_rate": c.learning_rate,
                "output_decay": c.output_decay,
                "policy_temperature": c.policy_temperature,
            },
            "addition_split": {
                "train_pairs": [list(pair) for pair in ADDITION_TRAIN_PAIRS],
                "commutativity_reversed_pairs": [list(pair) for pair in ADDITION_COMMUTATIVITY_PAIRS],
                "held_out_unordered_pairs": [list(pair) for pair in sorted(HELD_OUT_UNORDERED_PAIRS)],
                "held_out_ordered_pairs": [list(pair) for pair in ADDITION_HELD_OUT_PAIRS],
                "every_output_sum_0_to_18_present_in_training": True,
            },
            "scientific_constraint": (
                "Targets are used only after action selection to issue reward. The learner input contains only arbitrary "
                "fixed sparse symbol codes and slot position; no numeric magnitude or arithmetic result is supplied."
            ),
        }
