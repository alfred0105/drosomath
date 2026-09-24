from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .symbolic_stage3s import (
    ADDITION_COMMUTATIVITY_PAIRS,
    ADDITION_HELD_OUT_PAIRS,
    ADDITION_TRAIN_PAIRS,
    ALL_TOKENS,
    DIGIT_TOKENS,
    N_OUTPUTS,
)


@dataclass(frozen=True)
class SymbolicStage3S1Config:
    seed: int = 41
    token_dim: int = 128
    token_active: int = 16
    n_state: int = 160
    state_active: int = 36
    recurrent_gain: float = 0.72
    state_carry: float = 0.38
    output_learning_rate: float = 0.045
    token_plasticity_rate: float = 0.0022
    recurrent_plasticity_rate: float = 0.0012
    output_decay: float = 0.999997
    internal_decay: float = 0.9999995
    policy_temperature: float = 0.82
    token_weight_clip: float = 0.85
    recurrent_weight_clip: float = 0.55


class SymbolicStage3S1Experiment:
    """Stage 3S.1: symbolic arithmetic with recurrent, reward-gated plastic state.

    Inputs remain arbitrary fixed sparse symbol codes. No scalar magnitude,
    ordinal coordinate, sum, answer feature, commutativity flag, or distance to
    answer is injected. The target is consulted only after action selection to
    turn the scalar reward on or off/sign it.

    Compared with Stage 3S, the same token projection is reused at every
    sequence position and a recurrent state integrates the token stream. Both
    token->state and recurrent state transitions are plastic through a
    reward-gated credit signal derived from the learner's own chosen action.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, config: SymbolicStage3S1Config | None = None) -> None:
        self.config = config or SymbolicStage3S1Config()
        c = self.config
        self.rng = np.random.default_rng(c.seed)

        self.token_codes: dict[str, np.ndarray] = {}
        for token in ALL_TOKENS:
            code = np.zeros(c.token_dim, dtype=np.float32)
            active = self.rng.choice(c.token_dim, size=c.token_active, replace=False)
            code[active] = 1.0 / np.sqrt(float(c.token_active))
            self.token_codes[token] = code

        self.w_token = self.rng.normal(
            0.0, 0.24 / np.sqrt(float(c.token_active)), size=(c.n_state, c.token_dim)
        ).astype(np.float32)
        self.w_rec = self.rng.normal(
            0.0, 0.16 / np.sqrt(float(c.state_active)), size=(c.n_state, c.n_state)
        ).astype(np.float32)
        self.w_output = self.rng.normal(0.0, 0.004, size=(N_OUTPUTS, c.n_state)).astype(np.float32)
        self.output_bias = np.zeros(N_OUTPUTS, dtype=np.float32)

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
        raise ValueError(f"Unknown Stage 3S.1 task: {task}")

    @staticmethod
    def active_output_count(task: str) -> int:
        # Task-level action gating prevents identity/successor rehearsal from
        # punishing arithmetic-only outputs 10..18 before/while addition learns.
        return 10 if task in {"identity", "successor"} else N_OUTPUTS

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

    def _encode_sequence(self, tokens: list[str]) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        c = self.config
        h_prev = np.zeros(c.n_state, dtype=np.float32)
        traces: list[dict[str, np.ndarray]] = []

        for token in tokens:
            x = self.token_codes[token]
            pre = (
                self.w_token @ x
                + c.recurrent_gain * (self.w_rec @ h_prev)
                + c.state_carry * h_prev
            )
            raw = np.tanh(pre).astype(np.float32)
            h, active_idx = self._sparsify(raw)
            mask = np.zeros(c.n_state, dtype=np.float32)
            mask[active_idx] = 1.0
            traces.append({"x": x.copy(), "prev": h_prev.copy(), "mask": mask})
            h_prev = h

        return h_prev, traces

    def _policy(self, state: np.ndarray, *, active_outputs: int) -> np.ndarray:
        c = self.config
        logits = (self.w_output @ state + self.output_bias) / c.policy_temperature
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
        traces: list[dict[str, np.ndarray]],
        feedback: np.ndarray,
        reward: float,
    ) -> tuple[int, float]:
        c = self.config
        delta_token = np.zeros_like(self.w_token)
        delta_rec = np.zeros_like(self.w_rec)
        credit = feedback.astype(np.float32, copy=True)

        for trace in reversed(traces):
            local = credit * trace["mask"]
            scale = float(np.max(np.abs(local)))
            if scale > 1e-8:
                local = local / scale
            delta_token += c.token_plasticity_rate * reward * np.outer(local, trace["x"])
            if np.any(trace["prev"]):
                delta_rec += c.recurrent_plasticity_rate * reward * np.outer(local, trace["prev"])

            # Approximate recurrent credit assignment. This uses only the
            # learner's own recurrent weights and chosen-action feedback.
            credit = (
                c.state_carry * local
                + c.recurrent_gain * (self.w_rec.T @ local)
            ).astype(np.float32)
            credit_norm = float(np.linalg.norm(credit))
            if credit_norm > 1.0:
                credit /= credit_norm

        self.w_token += delta_token
        self.w_rec += delta_rec
        self.w_token *= c.internal_decay
        self.w_rec *= c.internal_decay
        np.clip(self.w_token, -c.token_weight_clip, c.token_weight_clip, out=self.w_token)
        np.clip(self.w_rec, -c.recurrent_weight_clip, c.recurrent_weight_clip, out=self.w_rec)

        changed = np.concatenate(
            [np.abs(delta_token).ravel(), np.abs(delta_rec).ravel()]
        )
        active = changed > 1e-9
        return int(active.sum()), float(changed[active].mean()) if np.any(active) else 0.0

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
        state, traces = self._encode_sequence(tokens)
        active_outputs = self.active_output_count(task)
        probabilities = self._policy(state, active_outputs=active_outputs)
        choice = int(self.rng.choice(N_OUTPUTS, p=probabilities))
        correct = choice == target
        reward = 1.0 if correct else -1.0

        eligibility = -probabilities.astype(np.float32)
        eligibility[choice] += 1.0
        if active_outputs < N_OUTPUTS:
            eligibility[active_outputs:] = 0.0

        old_output = self.w_output.copy() if learn else self.w_output
        delta_output = (
            self.config.output_learning_rate
            * reward
            * eligibility[:, None]
            * state[None, :]
        ).astype(np.float32)

        internal_active = 0
        internal_mean = 0.0
        if learn:
            feedback = old_output.T @ eligibility
            fb_norm = float(np.linalg.norm(feedback))
            if fb_norm > 1e-8:
                feedback = feedback / fb_norm
            internal_active, internal_mean = self._apply_internal_plasticity(traces, feedback, reward)

            self.w_output += delta_output
            self.output_bias += self.config.output_learning_rate * 0.025 * reward * eligibility
            self.w_output *= self.config.output_decay
        else:
            delta_output = np.zeros_like(delta_output)

        state_active_idx = np.flatnonzero(state > 0)
        max_state = float(state[state_active_idx].max()) if state_active_idx.size else 1.0
        sparse_state = [
            [int(i), round(float(state[i] / max(max_state, 1e-8)), 3)]
            for i in state_active_idx
        ]
        top_idx = np.argsort(probabilities)[-5:][::-1]
        top_policy = [
            {"choice": int(i), "p": round(float(probabilities[i]), 6)}
            for i in top_idx
            if probabilities[i] > 0
        ]
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))

        output_changed = np.abs(delta_output) > 1e-9
        out_vals = np.abs(delta_output[output_changed])
        output_mean = float(out_vals.mean()) if out_vals.size else 0.0
        total_active = int(output_changed.sum()) + internal_active
        if total_active:
            mean_delta = (
                output_mean * int(output_changed.sum()) + internal_mean * internal_active
            ) / total_active
        else:
            mean_delta = 0.0

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
            "policy": {
                "top": top_policy,
                "entropy": round(entropy, 6),
                "chosen_probability": round(float(probabilities[choice]), 6),
                "target_probability": round(float(probabilities[target]), 6),
            },
            "state_activity": sparse_state,
            "plasticity": {
                "mean_delta_w": round(mean_delta, 8),
                "active_synapses": total_active,
                "output_active_synapses": int(output_changed.sum()),
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
            w_output=self.w_output,
            output_bias=self.output_bias,
            metadata_json=np.array(json.dumps(payload)),
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            if int(payload.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
                raise ValueError("Unsupported Stage 3S.1 checkpoint version")
            self.w_token = data["w_token"].astype(np.float32, copy=True)
            self.w_rec = data["w_rec"].astype(np.float32, copy=True)
            self.w_output = data["w_output"].astype(np.float32, copy=True)
            self.output_bias = data["output_bias"].astype(np.float32, copy=True)
        self.rng.bit_generator.state = payload["rng_state"]
        return dict(payload.get("metadata") or {})

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        return {
            "stage": "3S.1",
            "substage": "masked_outputs_recurrent_internal_plasticity",
            "seed": c.seed,
            "symbols": list(ALL_TOKENS),
            "outputs": list(range(N_OUTPUTS)),
            "token_encoding": {
                "kind": "fixed_random_sparse_code_shared_across_sequence_positions",
                "token_dim": c.token_dim,
                "active_bits": c.token_active,
                "scalar_number_value_injected": False,
                "ordinal_embedding_injected": False,
                "sum_feature_injected": False,
                "commutativity_feature_injected": False,
                "distance_to_answer_injected": False,
            },
            "action_gating": {
                "identity_successor_active_outputs": list(range(10)),
                "addition_active_outputs": list(range(N_OUTPUTS)),
                "purpose": "prevent arithmetic-only outputs 10..18 from being punished by pre-addition tasks",
            },
            "learner": {
                "kind": "shared_token_projection_recurrent_sparse_state_reward_gated_internal_and_output_plasticity",
                "n_state": c.n_state,
                "state_active": c.state_active,
                "recurrent_gain": c.recurrent_gain,
                "state_carry": c.state_carry,
                "output_learning_rate": c.output_learning_rate,
                "token_plasticity_rate": c.token_plasticity_rate,
                "recurrent_plasticity_rate": c.recurrent_plasticity_rate,
                "policy_temperature": c.policy_temperature,
            },
            "addition_split": {
                "train_pairs": [list(pair) for pair in ADDITION_TRAIN_PAIRS],
                "commutativity_reversed_pairs": [list(pair) for pair in ADDITION_COMMUTATIVITY_PAIRS],
                "held_out_ordered_pairs": [list(pair) for pair in ADDITION_HELD_OUT_PAIRS],
                "every_output_sum_0_to_18_present_in_training": True,
            },
            "scientific_constraint": (
                "No numeric magnitude or arithmetic result is supplied. Internal plasticity receives only a scalar "
                "reward after the learner chooses an action plus credit derived from its own state/readout weights."
            ),
        }


__all__ = [
    "SymbolicStage3S1Config",
    "SymbolicStage3S1Experiment",
    "ADDITION_TRAIN_PAIRS",
    "ADDITION_COMMUTATIVITY_PAIRS",
    "ADDITION_HELD_OUT_PAIRS",
    "N_OUTPUTS",
    "DIGIT_TOKENS",
]
