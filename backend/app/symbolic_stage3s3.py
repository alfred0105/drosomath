from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .symbolic_stage3s import DIGIT_TOKENS, N_OUTPUTS
from .symbolic_stage3s2 import (
    ADDITION_HELDOUT_PAIRS,
    ADDITION_REVERSE_TEST_PAIRS,
    ADDITION_TRAIN_PAIRS,
    SMALL_ADDITION_PAIRS,
)

SPECIAL_TOKENS_3S3 = ("+", "=", "NEXT", "PREV")
ALL_TOKENS_3S3 = DIGIT_TOKENS + SPECIAL_TOKENS_3S3

NEXT2_TRAIN_STARTS = (0, 2, 3, 4, 6, 7)
NEXT2_HELDOUT_STARTS = (1, 5)
NEXT3_TRAIN_STARTS = (0, 1, 3, 4, 6)
NEXT3_HELDOUT_STARTS = (2, 5)
PREV2_TRAIN_STARTS = (2, 3, 5, 6, 7, 9)
PREV2_HELDOUT_STARTS = (4, 8)
PREV3_TRAIN_STARTS = (3, 4, 6, 7, 9)
PREV3_HELDOUT_STARTS = (5, 8)

BRIDGE_ADDITION_PAIRS = tuple(
    pair for pair in ADDITION_TRAIN_PAIRS if min(pair) <= 2
)


@dataclass(frozen=True)
class SymbolicStage3S3Config:
    seed: int = 67
    token_dim: int = 128
    token_active: int = 16
    n_state: int = 192
    state_active: int = 40
    recurrent_gain: float = 0.56
    operator_gain: float = 0.92
    state_carry: float = 0.34
    operator_token_gain: float = 0.42
    output_learning_rate: float = 0.040
    token_plasticity_rate: float = 0.0016
    recurrent_plasticity_rate: float = 0.0008
    operator_plasticity_rate: float = 0.0018
    output_decay: float = 0.9999985
    internal_decay: float = 0.9999998
    policy_temperature: float = 0.82
    token_weight_clip: float = 0.85
    recurrent_weight_clip: float = 0.60
    operator_weight_clip: float = 0.70
    correct_reward: float = 1.0
    incorrect_reward: float = -0.08


class SymbolicStage3S3Experiment:
    """Stage 3S.3: shared learned operator transitions plus scaffolded curriculum.

    Every visible symbol is still an arbitrary sparse token. NEXT and PREV are
    treated as operator *roles* by the architecture, but their numerical meaning
    is not hard-coded: their transition matrices are random at initialization
    and can only acquire useful behavior through scalar reinforcement.

    Guided multi-step examples contain explicit intermediate digit symbols during
    training (teacher-forcing scaffold). Autonomous and held-out tests never show
    those intermediate symbols. No scalar magnitude, ordinal coordinate, sum,
    answer-distance or commutativity feature is injected.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, config: SymbolicStage3S3Config | None = None) -> None:
        self.config = config or SymbolicStage3S3Config()
        c = self.config
        self.rng = np.random.default_rng(c.seed)

        self.token_codes: dict[str, np.ndarray] = {}
        for token in ALL_TOKENS_3S3:
            code = np.zeros(c.token_dim, dtype=np.float32)
            active = self.rng.choice(c.token_dim, size=c.token_active, replace=False)
            code[active] = 1.0 / np.sqrt(float(c.token_active))
            self.token_codes[token] = code

        self.w_token = self.rng.normal(
            0.0, 0.24 / np.sqrt(float(c.token_active)), size=(c.n_state, c.token_dim)
        ).astype(np.float32)
        self.w_rec = self.rng.normal(
            0.0, 0.14 / np.sqrt(float(c.state_active)), size=(c.n_state, c.n_state)
        ).astype(np.float32)
        self.w_next = self.rng.normal(
            0.0, 0.14 / np.sqrt(float(c.state_active)), size=(c.n_state, c.n_state)
        ).astype(np.float32)
        self.w_prev = self.rng.normal(
            0.0, 0.14 / np.sqrt(float(c.state_active)), size=(c.n_state, c.n_state)
        ).astype(np.float32)
        self.w_output = self.rng.normal(0.0, 0.004, size=(N_OUTPUTS, c.n_state)).astype(np.float32)
        self.output_bias = np.zeros(N_OUTPUTS, dtype=np.float32)

    @staticmethod
    def _operator_spec(task: str) -> tuple[str, int]:
        mapping = {
            "next": ("NEXT", 1),
            "next2": ("NEXT", 2),
            "next3": ("NEXT", 3),
            "guided_next2": ("NEXT", 2),
            "guided_next3": ("NEXT", 3),
            "prev": ("PREV", 1),
            "prev2": ("PREV", 2),
            "prev3": ("PREV", 3),
            "guided_prev2": ("PREV", 2),
            "guided_prev3": ("PREV", 3),
        }
        if task not in mapping:
            raise ValueError(task)
        return mapping[task]

    @classmethod
    def task_example(
        cls, task: str, *, a: int | None = None, b: int | None = None
    ) -> tuple[list[str], int, str]:
        if task == "identity":
            if a is None or not 0 <= a <= 9:
                raise ValueError("identity requires 0..9")
            return [str(a)], a, str(a)

        if task in {
            "next", "next2", "next3", "guided_next2", "guided_next3",
            "prev", "prev2", "prev3", "guided_prev2", "guided_prev3",
        }:
            op, steps = cls._operator_spec(task)
            delta = 1 if op == "NEXT" else -1
            if a is None or not (0 <= a + delta * steps <= 9):
                raise ValueError(f"{task} start is out of range")
            guided = task.startswith("guided_")
            tokens: list[str] = [str(a)]
            current = a
            for step in range(steps):
                tokens.append(op)
                current += delta
                if guided and step < steps - 1:
                    tokens.append(str(current))
            target = a + delta * steps
            label = f"{op}^{steps}({a})"
            if guided:
                label = f"guided {label}"
            return tokens, target, label

        if task == "addition":
            if a is None or b is None or not (0 <= a <= 9 and 0 <= b <= 9):
                raise ValueError("addition requires digits 0..9")
            return [str(a), "+", str(b), "="], a + b, f"{a} + {b} = ?"

        raise ValueError(f"Unknown Stage 3S.3 task: {task}")

    @staticmethod
    def active_output_count(task: str) -> int:
        return N_OUTPUTS if task == "addition" else 10

    def _sparsify(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = min(self.config.state_active, self.config.n_state)
        active_idx = np.argpartition(raw, -k)[-k:]
        h = np.zeros(self.config.n_state, dtype=np.float32)
        positive = np.maximum(raw[active_idx], 0.0)
        if not np.any(positive):
            positive = np.maximum(np.abs(raw[active_idx]), 1e-6)
        h[active_idx] = positive
        norm = float(np.linalg.norm(h))
        if norm > 1e-8:
            h /= norm
        return h, active_idx

    def _transition_matrix(self, token: str) -> tuple[np.ndarray, str, float]:
        c = self.config
        if token == "NEXT":
            return self.w_next, "next", c.operator_gain
        if token == "PREV":
            return self.w_prev, "prev", c.operator_gain
        return self.w_rec, "general", c.recurrent_gain

    def _encode_sequence(self, tokens: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        c = self.config
        h_prev = np.zeros(c.n_state, dtype=np.float32)
        traces: list[dict[str, Any]] = []
        for token in tokens:
            x = self.token_codes[token]
            matrix, kind, gain = self._transition_matrix(token)
            token_gain = c.operator_token_gain if kind in {"next", "prev"} else 1.0
            pre = token_gain * (self.w_token @ x) + gain * (matrix @ h_prev) + c.state_carry * h_prev
            raw = np.tanh(pre).astype(np.float32)
            h, active_idx = self._sparsify(raw)
            mask = np.zeros(c.n_state, dtype=np.float32)
            mask[active_idx] = 1.0
            traces.append({
                "x": x.copy(), "prev": h_prev.copy(), "mask": mask,
                "kind": kind, "gain": gain,
            })
            h_prev = h
        return h_prev, traces

    def _policy(self, state: np.ndarray, *, active_outputs: int) -> np.ndarray:
        logits = (self.w_output @ state + self.output_bias) / self.config.policy_temperature
        if active_outputs < N_OUTPUTS:
            logits[active_outputs:] = -1.0e9
        logits -= float(np.max(logits))
        exp = np.exp(logits)
        probs = exp / float(np.sum(exp))
        if active_outputs < N_OUTPUTS:
            probs[active_outputs:] = 0.0
            probs /= float(np.sum(probs))
        return probs.astype(np.float32)

    def _apply_internal_plasticity(
        self,
        traces: list[dict[str, Any]],
        feedback: np.ndarray,
        reward: float,
        *,
        token_scale: float,
        recurrent_scale: float,
        operator_scale: float,
    ) -> tuple[int, float]:
        c = self.config
        d_token = np.zeros_like(self.w_token)
        d_rec = np.zeros_like(self.w_rec)
        d_next = np.zeros_like(self.w_next)
        d_prev = np.zeros_like(self.w_prev)
        credit = feedback.astype(np.float32, copy=True)

        for trace in reversed(traces):
            local = credit * trace["mask"]
            scale = float(np.max(np.abs(local)))
            if scale > 1e-8:
                local /= scale

            if token_scale > 0:
                d_token += c.token_plasticity_rate * token_scale * reward * np.outer(local, trace["x"])

            kind = str(trace["kind"])
            prev = trace["prev"]
            if np.any(prev):
                if kind == "next" and operator_scale > 0:
                    d_next += c.operator_plasticity_rate * operator_scale * reward * np.outer(local, prev)
                    matrix = self.w_next
                elif kind == "prev" and operator_scale > 0:
                    d_prev += c.operator_plasticity_rate * operator_scale * reward * np.outer(local, prev)
                    matrix = self.w_prev
                else:
                    if recurrent_scale > 0:
                        d_rec += c.recurrent_plasticity_rate * recurrent_scale * reward * np.outer(local, prev)
                    matrix = self.w_rec
            else:
                matrix = self.w_rec

            credit = (c.state_carry * local + float(trace["gain"]) * (matrix.T @ local)).astype(np.float32)
            norm = float(np.linalg.norm(credit))
            if norm > 1.0:
                credit /= norm

        self.w_token += d_token
        self.w_rec += d_rec
        self.w_next += d_next
        self.w_prev += d_prev
        self.w_token *= c.internal_decay
        self.w_rec *= c.internal_decay
        self.w_next *= c.internal_decay
        self.w_prev *= c.internal_decay
        np.clip(self.w_token, -c.token_weight_clip, c.token_weight_clip, out=self.w_token)
        np.clip(self.w_rec, -c.recurrent_weight_clip, c.recurrent_weight_clip, out=self.w_rec)
        np.clip(self.w_next, -c.operator_weight_clip, c.operator_weight_clip, out=self.w_next)
        np.clip(self.w_prev, -c.operator_weight_clip, c.operator_weight_clip, out=self.w_prev)

        changed = np.concatenate((np.abs(d_token).ravel(), np.abs(d_rec).ravel(), np.abs(d_next).ravel(), np.abs(d_prev).ravel()))
        active = changed > 1e-9
        return int(active.sum()), float(changed[active].mean()) if np.any(active) else 0.0

    def _sample_start(self, task: str, pool: tuple[int, ...] | None) -> int:
        if pool:
            return int(pool[int(self.rng.integers(0, len(pool)))])
        if "prev3" in task:
            return int(self.rng.integers(3, 10))
        if "prev2" in task:
            return int(self.rng.integers(2, 10))
        if task == "prev":
            return int(self.rng.integers(1, 10))
        if "next3" in task:
            return int(self.rng.integers(0, 7))
        if "next2" in task:
            return int(self.rng.integers(0, 8))
        if task == "next":
            return int(self.rng.integers(0, 9))
        raise ValueError(task)

    def step(
        self,
        trial: int,
        *,
        task: str,
        learn: bool = True,
        a: int | None = None,
        b: int | None = None,
        pair_pool: tuple[tuple[int, int], ...] | None = None,
        start_pool: tuple[int, ...] | None = None,
        token_plasticity_scale: float = 1.0,
        recurrent_plasticity_scale: float = 1.0,
        operator_plasticity_scale: float = 1.0,
        output_plasticity_scale: float = 1.0,
        trial_kind: str = "train",
    ) -> dict[str, Any]:
        if task == "identity" and a is None:
            a = int(self.rng.integers(0, 10))
        elif task in {
            "next", "next2", "next3", "guided_next2", "guided_next3",
            "prev", "prev2", "prev3", "guided_prev2", "guided_prev3",
        } and a is None:
            a = self._sample_start(task, start_pool)
        elif task == "addition" and (a is None or b is None):
            pool = pair_pool or ADDITION_TRAIN_PAIRS
            a, b = pool[int(self.rng.integers(0, len(pool)))]

        tokens, target, expression = self.task_example(task, a=a, b=b)
        state, traces = self._encode_sequence(tokens)
        active_outputs = self.active_output_count(task)
        probs = self._policy(state, active_outputs=active_outputs)
        choice = int(self.rng.choice(N_OUTPUTS, p=probs))
        correct = choice == target
        reward = self.config.correct_reward if correct else self.config.incorrect_reward

        eligibility = -probs.astype(np.float32)
        eligibility[choice] += 1.0
        if active_outputs < N_OUTPUTS:
            eligibility[active_outputs:] = 0.0

        d_out = (
            self.config.output_learning_rate * output_plasticity_scale * reward
            * eligibility[:, None] * state[None, :]
        ).astype(np.float32)

        internal_active = 0
        internal_mean = 0.0
        if learn:
            feedback = self.w_output.T @ eligibility
            norm = float(np.linalg.norm(feedback))
            if norm > 1e-8:
                feedback /= norm
            internal_active, internal_mean = self._apply_internal_plasticity(
                traces, feedback, reward,
                token_scale=token_plasticity_scale,
                recurrent_scale=recurrent_plasticity_scale,
                operator_scale=operator_plasticity_scale,
            )
            self.w_output += d_out
            self.output_bias += self.config.output_learning_rate * 0.025 * output_plasticity_scale * reward * eligibility
            self.w_output *= self.config.output_decay
        else:
            d_out = np.zeros_like(d_out)

        state_idx = np.flatnonzero(state > 0)
        max_state = float(state[state_idx].max()) if state_idx.size else 1.0
        sparse_state = [[int(i), round(float(state[i] / max(max_state, 1e-8)), 3)] for i in state_idx]
        top_idx = np.argsort(probs)[-5:][::-1]
        top_policy = [{"choice": int(i), "p": round(float(probs[i]), 6)} for i in top_idx if probs[i] > 0]
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))

        out_mask = np.abs(d_out) > 1e-9
        out_active = int(out_mask.sum())
        out_mean = float(np.abs(d_out[out_mask]).mean()) if out_active else 0.0
        total_active = out_active + internal_active
        mean_delta = ((out_mean * out_active + internal_mean * internal_active) / total_active) if total_active else 0.0

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
            "active_output_count": active_outputs,
            "plasticity_scales": {
                "token": token_plasticity_scale,
                "recurrent": recurrent_plasticity_scale,
                "operator": operator_plasticity_scale,
                "output": output_plasticity_scale,
            },
            "policy": {
                "top": top_policy,
                "entropy": round(entropy, 6),
                "chosen_probability": round(float(probs[choice]), 6),
                "target_probability": round(float(probs[target]), 6),
            },
            "state_activity": sparse_state,
            "plasticity": {
                "mean_delta_w": round(mean_delta, 8),
                "active_synapses": total_active,
                "output_active_synapses": out_active,
                "internal_active_synapses": internal_active,
            },
        }

    def save_checkpoint(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "metadata": metadata or {},
            "rng_state": self.rng.bit_generator.state,
        }
        np.savez_compressed(
            path,
            w_token=self.w_token,
            w_rec=self.w_rec,
            w_next=self.w_next,
            w_prev=self.w_prev,
            w_output=self.w_output,
            output_bias=self.output_bias,
            metadata_json=np.array(json.dumps(payload)),
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            if int(payload.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
                raise ValueError("Unsupported Stage 3S.3 checkpoint version")
            self.w_token = data["w_token"].astype(np.float32, copy=True)
            self.w_rec = data["w_rec"].astype(np.float32, copy=True)
            self.w_next = data["w_next"].astype(np.float32, copy=True)
            self.w_prev = data["w_prev"].astype(np.float32, copy=True)
            self.w_output = data["w_output"].astype(np.float32, copy=True)
            self.output_bias = data["output_bias"].astype(np.float32, copy=True)
        self.rng.bit_generator.state = payload["rng_state"]
        return dict(payload.get("metadata") or {})

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        return {
            "stage": "3S.3",
            "substage": "600k_operator_transition_scaffold",
            "seed": c.seed,
            "symbols": list(ALL_TOKENS_3S3),
            "outputs": list(range(N_OUTPUTS)),
            "token_encoding": {
                "kind": "fixed_random_sparse_symbol_codes",
                "token_dim": c.token_dim,
                "active_bits": c.token_active,
                "scalar_number_value_injected": False,
                "ordinal_embedding_injected": False,
                "sum_feature_injected": False,
                "commutativity_feature_injected": False,
                "distance_to_answer_injected": False,
            },
            "operator_architecture": {
                "NEXT": "random_initialized_shared_plastic_transition_matrix",
                "PREV": "random_initialized_shared_plastic_transition_matrix",
                "numeric_transition_hardcoded": False,
                "guided_intermediate_symbols_training_only": True,
                "autonomous_tests_hide_intermediate_symbols": True,
            },
            "reward": {
                "correct": c.correct_reward,
                "incorrect": c.incorrect_reward,
            },
            "learner": {
                "kind": "shared_operator_transition_recurrent_sparse_associator",
                "n_state": c.n_state,
                "state_active": c.state_active,
                "recurrent_gain": c.recurrent_gain,
                "operator_gain": c.operator_gain,
                "output_learning_rate": c.output_learning_rate,
                "operator_plasticity_rate": c.operator_plasticity_rate,
            },
            "composition_split": {
                "next2_train": list(NEXT2_TRAIN_STARTS),
                "next2_heldout": list(NEXT2_HELDOUT_STARTS),
                "next3_train": list(NEXT3_TRAIN_STARTS),
                "next3_heldout": list(NEXT3_HELDOUT_STARTS),
                "prev2_train": list(PREV2_TRAIN_STARTS),
                "prev2_heldout": list(PREV2_HELDOUT_STARTS),
                "prev3_train": list(PREV3_TRAIN_STARTS),
                "prev3_heldout": list(PREV3_HELDOUT_STARTS),
            },
            "addition_split": {
                "train_pairs": [list(pair) for pair in ADDITION_TRAIN_PAIRS],
                "reverse_test_pairs": [list(pair) for pair in ADDITION_REVERSE_TEST_PAIRS],
                "heldout_pairs": [list(pair) for pair in ADDITION_HELDOUT_PAIRS],
                "bridge_pairs": [list(pair) for pair in BRIDGE_ADDITION_PAIRS],
                "small_pairs": [list(pair) for pair in SMALL_ADDITION_PAIRS],
            },
            "scientific_constraint": (
                "No numeric magnitude, sum, ordinal coordinate, answer distance or commutativity feature is supplied. "
                "NEXT/PREV are given operator roles but random transition matrices learn their meaning only from reward. "
                "Teacher-forced intermediate digit symbols appear only in scaffold training; frozen composition tests are autonomous."
            ),
        }
