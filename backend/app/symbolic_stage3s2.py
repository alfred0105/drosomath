from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .symbolic_stage3s import ALL_TOKENS, HELD_OUT_UNORDERED_PAIRS, N_OUTPUTS


SUCCESSOR2_TRAIN_STARTS = (0, 2, 3, 4, 6, 7)
SUCCESSOR2_HELDOUT_STARTS = (1, 5)
SUCCESSOR3_TRAIN_STARTS = (0, 1, 3, 4, 6)
SUCCESSOR3_HELDOUT_STARTS = (2, 5)


def _build_addition_splits() -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """Build structured commutativity exposure without injecting the rule.

    Most non-held-out unordered pairs are shown in both orientations, a small
    deterministic subset is shown in only one orientation and tested reversed,
    and the original Stage 3S unordered holdout set is never rewarded at all.
    """
    train: list[tuple[int, int]] = []
    reverse_test: list[tuple[int, int]] = []
    heldout: list[tuple[int, int]] = []

    for a in range(10):
        for b in range(a, 10):
            if (a, b) in HELD_OUT_UNORDERED_PAIRS:
                heldout.append((a, b))
                if a != b:
                    heldout.append((b, a))
                continue

            if a == b:
                train.append((a, b))
                continue

            # A deterministic subset gets only one orientation during training.
            one_way = ((a * 13 + b * 17) % 7) in {0, 1}
            if one_way:
                if ((a + b) % 2) == 0:
                    train.append((a, b))
                    reverse_test.append((b, a))
                else:
                    train.append((b, a))
                    reverse_test.append((a, b))
            else:
                # Seeing both orders is experience, not an engineered
                # commutativity feature. The learner still receives only tokens
                # and scalar reward.
                train.extend(((a, b), (b, a)))

    if {a + b for a, b in train} != set(range(N_OUTPUTS)):
        raise RuntimeError("Stage 3S.2 addition split lost an output class")

    train_unordered = {tuple(sorted(pair)) for pair in train}
    heldout_unordered = {tuple(sorted(pair)) for pair in heldout}
    if not train_unordered.isdisjoint(heldout_unordered):
        raise RuntimeError("Stage 3S.2 train/heldout leakage")

    return tuple(train), tuple(reverse_test), tuple(heldout)


ADDITION_TRAIN_PAIRS, ADDITION_REVERSE_TEST_PAIRS, ADDITION_HELDOUT_PAIRS = _build_addition_splits()
MICRO_ADDITION_PAIRS = tuple(pair for pair in ADDITION_TRAIN_PAIRS if max(pair) <= 2)
SMALL_ADDITION_PAIRS = tuple(pair for pair in ADDITION_TRAIN_PAIRS if max(pair) <= 4)


@dataclass(frozen=True)
class SymbolicStage3S2Config:
    seed: int = 53
    token_dim: int = 128
    token_active: int = 16
    n_state: int = 192
    state_active: int = 40
    recurrent_gain: float = 0.70
    state_carry: float = 0.42
    output_learning_rate: float = 0.042
    token_plasticity_rate: float = 0.0018
    recurrent_plasticity_rate: float = 0.0010
    output_decay: float = 0.999998
    internal_decay: float = 0.9999997
    policy_temperature: float = 0.86
    token_weight_clip: float = 0.85
    recurrent_weight_clip: float = 0.55
    correct_reward: float = 1.0
    incorrect_reward: float = -0.20


class SymbolicStage3S2Experiment:
    """Stage 3S.2: consolidated recurrent symbolic curriculum.

    Visible symbols remain arbitrary fixed sparse codes. No numeric magnitude,
    ordinal coordinate, sum, answer hint, commutativity flag, or distance to the
    answer is supplied. Repeated NEXT operations are represented literally by
    repeated NEXT tokens. Targets are consulted only after action selection to
    issue scalar reinforcement.

    Plasticity multipliers are supplied by the curriculum so early symbol
    representations can consolidate while recurrent/output learning continues.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, config: SymbolicStage3S2Config | None = None) -> None:
        self.config = config or SymbolicStage3S2Config()
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
    def task_example(
        task: str, *, a: int | None = None, b: int | None = None
    ) -> tuple[list[str], int, str]:
        if task == "identity":
            if a is None or not 0 <= a <= 9:
                raise ValueError("identity requires a digit 0..9")
            return [str(a)], a, str(a)

        if task in {"successor", "successor2", "successor3"}:
            steps = {"successor": 1, "successor2": 2, "successor3": 3}[task]
            max_start = 9 - steps
            if a is None or not 0 <= a <= max_start:
                raise ValueError(f"{task} requires a digit 0..{max_start}")
            tokens = [str(a)] + ["NEXT"] * steps
            return tokens, a + steps, f"NEXT^{steps}({a})"

        if task == "addition":
            if a is None or b is None or not (0 <= a <= 9 and 0 <= b <= 9):
                raise ValueError("addition requires digits 0..9")
            return [str(a), "+", str(b), "="], a + b, f"{a} + {b} = ?"

        raise ValueError(f"Unknown Stage 3S.2 task: {task}")

    @staticmethod
    def active_output_count(task: str) -> int:
        return 10 if task in {"identity", "successor", "successor2", "successor3"} else N_OUTPUTS

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
            pre = self.w_token @ x + c.recurrent_gain * (self.w_rec @ h_prev) + c.state_carry * h_prev
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
        *,
        token_scale: float,
        recurrent_scale: float,
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

            if token_scale > 0:
                delta_token += (
                    c.token_plasticity_rate
                    * token_scale
                    * reward
                    * np.outer(local, trace["x"])
                )
            if recurrent_scale > 0 and np.any(trace["prev"]):
                delta_rec += (
                    c.recurrent_plasticity_rate
                    * recurrent_scale
                    * reward
                    * np.outer(local, trace["prev"])
                )

            credit = (c.state_carry * local + c.recurrent_gain * (self.w_rec.T @ local)).astype(np.float32)
            credit_norm = float(np.linalg.norm(credit))
            if credit_norm > 1.0:
                credit /= credit_norm

        self.w_token += delta_token
        self.w_rec += delta_rec
        self.w_token *= c.internal_decay
        self.w_rec *= c.internal_decay
        np.clip(self.w_token, -c.token_weight_clip, c.token_weight_clip, out=self.w_token)
        np.clip(self.w_rec, -c.recurrent_weight_clip, c.recurrent_weight_clip, out=self.w_rec)

        changed = np.concatenate((np.abs(delta_token).ravel(), np.abs(delta_rec).ravel()))
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
        start_pool: tuple[int, ...] | None = None,
        token_plasticity_scale: float = 1.0,
        recurrent_plasticity_scale: float = 1.0,
        output_plasticity_scale: float = 1.0,
        trial_kind: str = "train",
    ) -> dict[str, Any]:
        if task == "identity" and a is None:
            a = int(self.rng.integers(0, 10))
        elif task == "successor" and a is None:
            a = int(self.rng.integers(0, 9))
        elif task == "successor2" and a is None:
            pool = start_pool or SUCCESSOR2_TRAIN_STARTS
            a = int(pool[int(self.rng.integers(0, len(pool)))])
        elif task == "successor3" and a is None:
            pool = start_pool or SUCCESSOR3_TRAIN_STARTS
            a = int(pool[int(self.rng.integers(0, len(pool)))])
        elif task == "addition" and (a is None or b is None):
            pool = pair_pool or ADDITION_TRAIN_PAIRS
            a, b = pool[int(self.rng.integers(0, len(pool)))]

        tokens, target, expression = self.task_example(task, a=a, b=b)
        state, traces = self._encode_sequence(tokens)
        active_outputs = self.active_output_count(task)
        probabilities = self._policy(state, active_outputs=active_outputs)
        choice = int(self.rng.choice(N_OUTPUTS, p=probabilities))
        correct = choice == target
        reward = self.config.correct_reward if correct else self.config.incorrect_reward

        eligibility = -probabilities.astype(np.float32)
        eligibility[choice] += 1.0
        if active_outputs < N_OUTPUTS:
            eligibility[active_outputs:] = 0.0

        delta_output = (
            self.config.output_learning_rate
            * output_plasticity_scale
            * reward
            * eligibility[:, None]
            * state[None, :]
        ).astype(np.float32)

        internal_active = 0
        internal_mean = 0.0
        if learn:
            feedback = self.w_output.T @ eligibility
            fb_norm = float(np.linalg.norm(feedback))
            if fb_norm > 1e-8:
                feedback = feedback / fb_norm
            internal_active, internal_mean = self._apply_internal_plasticity(
                traces,
                feedback,
                reward,
                token_scale=token_plasticity_scale,
                recurrent_scale=recurrent_plasticity_scale,
            )

            self.w_output += delta_output
            self.output_bias += (
                self.config.output_learning_rate
                * 0.025
                * output_plasticity_scale
                * reward
                * eligibility
            )
            self.w_output *= self.config.output_decay
        else:
            delta_output = np.zeros_like(delta_output)

        state_idx = np.flatnonzero(state > 0)
        max_state = float(state[state_idx].max()) if state_idx.size else 1.0
        sparse_state = [
            [int(i), round(float(state[i] / max(max_state, 1e-8)), 3)]
            for i in state_idx
        ]
        top_idx = np.argsort(probabilities)[-5:][::-1]
        top_policy = [
            {"choice": int(i), "p": round(float(probabilities[i]), 6)}
            for i in top_idx
            if probabilities[i] > 0
        ]
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))

        out_changed = np.abs(delta_output) > 1e-9
        out_values = np.abs(delta_output[out_changed])
        out_mean = float(out_values.mean()) if out_values.size else 0.0
        out_active = int(out_changed.sum())
        total_active = out_active + internal_active
        mean_delta = (
            (out_mean * out_active + internal_mean * internal_active) / total_active
            if total_active
            else 0.0
        )

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
                "output": output_plasticity_scale,
            },
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
            w_output=self.w_output,
            output_bias=self.output_bias,
            metadata_json=np.array(json.dumps(payload)),
        )

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(str(data["metadata_json"].item()))
            if int(payload.get("checkpoint_version", -1)) != self.CHECKPOINT_VERSION:
                raise ValueError("Unsupported Stage 3S.2 checkpoint version")
            self.w_token = data["w_token"].astype(np.float32, copy=True)
            self.w_rec = data["w_rec"].astype(np.float32, copy=True)
            self.w_output = data["w_output"].astype(np.float32, copy=True)
            self.output_bias = data["output_bias"].astype(np.float32, copy=True)
        self.rng.bit_generator.state = payload["rng_state"]
        return dict(payload.get("metadata") or {})

    def config_dict(self) -> dict[str, Any]:
        c = self.config
        return {
            "stage": "3S.2",
            "substage": "360k_consolidated_multistep_curriculum",
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
            "reward": {
                "correct": c.correct_reward,
                "incorrect": c.incorrect_reward,
                "purpose": "reduce destructive interference from mostly-wrong early arithmetic exploration",
            },
            "learner": {
                "kind": "shared_token_projection_recurrent_sparse_state_with_consolidation_schedule",
                "n_state": c.n_state,
                "state_active": c.state_active,
                "recurrent_gain": c.recurrent_gain,
                "state_carry": c.state_carry,
                "output_learning_rate": c.output_learning_rate,
                "token_plasticity_rate": c.token_plasticity_rate,
                "recurrent_plasticity_rate": c.recurrent_plasticity_rate,
                "policy_temperature": c.policy_temperature,
            },
            "composition_split": {
                "successor2_train_starts": list(SUCCESSOR2_TRAIN_STARTS),
                "successor2_heldout_starts": list(SUCCESSOR2_HELDOUT_STARTS),
                "successor3_train_starts": list(SUCCESSOR3_TRAIN_STARTS),
                "successor3_heldout_starts": list(SUCCESSOR3_HELDOUT_STARTS),
            },
            "addition_split": {
                "train_pairs": [list(pair) for pair in ADDITION_TRAIN_PAIRS],
                "one_way_reverse_test_pairs": [list(pair) for pair in ADDITION_REVERSE_TEST_PAIRS],
                "heldout_ordered_pairs": [list(pair) for pair in ADDITION_HELDOUT_PAIRS],
                "micro_train_pairs": [list(pair) for pair in MICRO_ADDITION_PAIRS],
                "small_train_pairs": [list(pair) for pair in SMALL_ADDITION_PAIRS],
                "every_output_sum_0_to_18_present_in_full_training": True,
            },
            "scientific_constraint": (
                "Inputs contain only arbitrary symbol codes and repeated visible operator tokens. Numeric magnitude, "
                "sum, commutativity, ordinal coordinates and answer distance are never supplied. Targets are used only "
                "after a choice to issue scalar reinforcement."
            ),
        }
