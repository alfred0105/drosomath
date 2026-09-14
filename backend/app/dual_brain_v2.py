from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class DualBrainV2Config:
    """Resource-bounded dual-brain prototype.

    Defaults stay intentionally small enough for fast CPU experiments. Functional
    roles are not assigned to modules; routing and specialization must emerge
    from reward and activity.
    """

    seed: int = 20260914
    brains: int = 2
    modules_per_brain: int = 4
    neurons_per_module: int = 64
    active_neurons_per_module: int = 12
    token_dim: int = 128
    token_active: int = 16
    output_dim: int = 20
    router_top_k: int = 3
    gateway_fraction: float = 0.20

    local_synapses_per_module: int = 512
    inter_module_synapses_per_brain: int = 768
    bridge_synapses: int = 256

    state_carry: float = 0.30
    recurrent_gain: float = 0.75
    input_gain: float = 1.00
    dormant_gain: float = 0.18
    policy_temperature: float = 0.85
    router_temperature: float = 0.90
    load_balance_strength: float = 0.35

    output_learning_rate: float = 0.030
    input_learning_rate: float = 0.0008
    router_learning_rate: float = 0.0012
    synapse_learning_rate: float = 0.0010
    synapse_weight_decay: float = 2.0e-6
    consolidation_rate: float = 0.010
    consolidation_decay: float = 1.0e-4
    weight_clip: float = 0.90
    output_weight_clip: float = 1.20

    usage_ema_rate: float = 0.010
    reward_ema_rate: float = 0.010
    module_usage_ema_rate: float = 0.010

    rewire_interval: int = 250
    rewire_fraction: float = 0.025
    protected_stability: float = 0.80
    exploratory_regrowth_fraction: float = 0.20

    correct_reward: float = 1.0
    incorrect_reward: float = -0.08

    def __post_init__(self) -> None:
        if self.brains != 2:
            raise ValueError("DualBrainV2 currently requires exactly two brains")
        if not 1 <= self.router_top_k <= self.total_modules:
            raise ValueError("router_top_k must be within the module count")
        if not 0.0 < self.gateway_fraction <= 1.0:
            raise ValueError("gateway_fraction must be in (0, 1]")
        if not 0.0 <= self.rewire_fraction <= 1.0:
            raise ValueError("rewire_fraction must be in [0, 1]")

    @property
    def total_modules(self) -> int:
        return self.brains * self.modules_per_brain

    @property
    def total_neurons(self) -> int:
        return self.total_modules * self.neurons_per_module


EdgeSampler = Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]


@dataclass
class SparseSynapseBank:
    """Fixed-capacity edge-list synapse store with plasticity and rewiring."""

    name: str
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    usage_ema: np.ndarray
    reward_ema: np.ndarray
    stability: np.ndarray
    sampler: EdgeSampler = field(repr=False)
    rewires_total: int = 0

    @classmethod
    def create(
        cls,
        name: str,
        budget: int,
        sampler: EdgeSampler,
        rng: np.random.Generator,
        *,
        init_scale: float = 0.08,
    ) -> "SparseSynapseBank":
        if budget <= 0:
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float32)
            return cls(
                name,
                empty_i,
                empty_i.copy(),
                empty_f,
                empty_f.copy(),
                empty_f.copy(),
                empty_f.copy(),
                sampler,
            )
        src, dst = sampler(budget, rng)
        weight = rng.normal(0.0, init_scale, size=budget).astype(np.float32)
        zeros = np.zeros(budget, dtype=np.float32)
        return cls(
            name=name,
            src=src.astype(np.int32, copy=False),
            dst=dst.astype(np.int32, copy=False),
            weight=weight,
            usage_ema=zeros.copy(),
            reward_ema=zeros.copy(),
            stability=zeros.copy(),
            sampler=sampler,
        )

    @property
    def budget(self) -> int:
        return int(self.weight.size)

    @property
    def nbytes(self) -> int:
        return int(
            self.src.nbytes
            + self.dst.nbytes
            + self.weight.nbytes
            + self.usage_ema.nbytes
            + self.reward_ema.nbytes
            + self.stability.nbytes
        )

    def propagate(self, state: np.ndarray, total_neurons: int) -> np.ndarray:
        out = np.zeros(total_neurons, dtype=np.float32)
        if self.budget == 0:
            return out
        signal = self.weight * state[self.src]
        np.add.at(out, self.dst, signal)
        return out

    def apply_plasticity(
        self,
        pre_state: np.ndarray,
        post_state: np.ndarray,
        reward: float,
        config: DualBrainV2Config,
    ) -> None:
        if self.budget == 0:
            return
        signal = pre_state[self.src] * post_state[self.dst]
        activity = np.abs(signal).astype(np.float32)

        ur = config.usage_ema_rate
        rr = config.reward_ema_rate
        self.usage_ema *= 1.0 - ur
        self.usage_ema += ur * activity
        self.reward_ema *= 1.0 - rr
        self.reward_ema += rr * (reward * activity)

        # Repeated positively rewarded use consolidates the edge independently
        # from its raw weight. Consolidated edges decay less and resist pruning.
        positive = max(float(reward), 0.0)
        self.stability *= 1.0 - config.consolidation_decay
        self.stability += (
            config.consolidation_rate
            * positive
            * activity
            * (1.0 - self.stability)
        )
        np.clip(self.stability, 0.0, 1.0, out=self.stability)

        plastic_fraction = 1.0 - 0.85 * self.stability
        self.weight += (
            config.synapse_learning_rate * reward * signal * plastic_fraction
        )
        decay = config.synapse_weight_decay * (1.0 - self.stability)
        self.weight *= 1.0 - decay
        np.clip(
            self.weight,
            -config.weight_clip,
            config.weight_clip,
            out=self.weight,
        )

    def utility(self) -> np.ndarray:
        positive_reward = np.maximum(self.reward_ema, 0.0)
        return (
            self.usage_ema * (np.abs(self.weight) + 0.25 * self.stability)
            + 0.25 * positive_reward
        )

    def rewire(
        self,
        rng: np.random.Generator,
        config: DualBrainV2Config,
    ) -> int:
        if self.budget == 0 or config.rewire_fraction <= 0.0:
            return 0
        target = max(1, int(round(self.budget * config.rewire_fraction)))
        eligible = np.flatnonzero(self.stability < config.protected_stability)
        if eligible.size == 0:
            return 0
        target = min(target, int(eligible.size))
        util = self.utility()[eligible]

        explore_n = int(round(target * config.exploratory_regrowth_fraction))
        exploit_n = target - explore_n
        order = eligible[np.argsort(util)]
        chosen: list[int] = order[:exploit_n].tolist()
        if explore_n > 0:
            remaining = np.setdiff1d(
                eligible,
                np.asarray(chosen, dtype=np.int64),
                assume_unique=False,
            )
            if remaining.size > 0:
                extra = rng.choice(
                    remaining,
                    size=min(explore_n, remaining.size),
                    replace=False,
                )
                chosen.extend(int(x) for x in np.atleast_1d(extra))

        idx = np.asarray(chosen, dtype=np.int64)
        if idx.size == 0:
            return 0
        new_src, new_dst = self.sampler(int(idx.size), rng)
        self.src[idx] = new_src
        self.dst[idx] = new_dst
        self.weight[idx] = rng.normal(
            0.0,
            0.035,
            size=idx.size,
        ).astype(np.float32)
        self.usage_ema[idx] = 0.0
        self.reward_ema[idx] = 0.0
        self.stability[idx] = 0.0
        self.rewires_total += int(idx.size)
        return int(idx.size)

    def snapshot(self) -> dict[str, Any]:
        if self.budget == 0:
            return {
                "name": self.name,
                "budget": 0,
                "mean_abs_weight": 0.0,
                "mean_usage": 0.0,
                "mean_stability": 0.0,
                "consolidated_fraction": 0.0,
                "rewires_total": self.rewires_total,
            }
        return {
            "name": self.name,
            "budget": self.budget,
            "mean_abs_weight": float(np.mean(np.abs(self.weight))),
            "mean_usage": float(np.mean(self.usage_ema)),
            "mean_stability": float(np.mean(self.stability)),
            "consolidated_fraction": float(
                np.mean(self.stability >= 0.50)
            ),
            "rewires_total": self.rewires_total,
        }


class DualBrainV2:
    """Two resource-bounded brains whose module roles are learned, not assigned.

    Task names may be supplied for telemetry only. They are never used by the
    router, plasticity rules, or output learning.
    """

    CHECKPOINT_VERSION = 1

    def __init__(self, config: DualBrainV2Config | None = None) -> None:
        self.config = config or DualBrainV2Config()
        c = self.config
        self.rng = np.random.default_rng(c.seed)
        self.step_count = 0
        self.token_codes: dict[str, np.ndarray] = {}

        self.w_input = self.rng.normal(
            0.0,
            0.18 / np.sqrt(float(c.token_active)),
            size=(
                c.total_modules,
                c.neurons_per_module,
                c.token_dim,
            ),
        ).astype(np.float32)
        self.w_router = self.rng.normal(
            0.0,
            0.03,
            size=(c.total_modules, c.token_dim),
        ).astype(np.float32)
        self.w_output = self.rng.normal(
            0.0,
            0.01,
            size=(c.output_dim, c.total_neurons),
        ).astype(np.float32)
        self.output_bias = np.zeros(c.output_dim, dtype=np.float32)
        self.module_usage_ema = np.zeros(c.total_modules, dtype=np.float32)
        self.module_activity_ema = np.zeros(c.total_modules, dtype=np.float32)
        self.task_activity_ema: dict[str, np.ndarray] = {}

        self.local_banks: list[SparseSynapseBank] = []
        for module in range(c.total_modules):
            self.local_banks.append(
                SparseSynapseBank.create(
                    f"local_m{module}",
                    c.local_synapses_per_module,
                    self._make_local_sampler(module),
                    self.rng,
                )
            )

        self.inter_banks: list[SparseSynapseBank] = []
        for brain in range(c.brains):
            self.inter_banks.append(
                SparseSynapseBank.create(
                    f"inter_brain_{brain}",
                    c.inter_module_synapses_per_brain,
                    self._make_inter_sampler(brain),
                    self.rng,
                )
            )

        gateway_count = max(
            1,
            int(round(c.neurons_per_module * c.gateway_fraction)),
        )
        self.gateway_local_indices = np.sort(
            self.rng.choice(
                c.neurons_per_module,
                size=gateway_count,
                replace=False,
            )
        ).astype(np.int32)
        self.bridge_bank = SparseSynapseBank.create(
            "cross_brain_bridge",
            c.bridge_synapses,
            self._make_bridge_sampler(),
            self.rng,
        )

    def _module_bounds(self, module: int) -> tuple[int, int]:
        n = self.config.neurons_per_module
        start = module * n
        return start, start + n

    def _module_indices(self, module: int) -> np.ndarray:
        start, stop = self._module_bounds(module)
        return np.arange(start, stop, dtype=np.int32)

    def _brain_modules(self, brain: int) -> np.ndarray:
        c = self.config
        start = brain * c.modules_per_brain
        stop = start + c.modules_per_brain
        return np.arange(start, stop, dtype=np.int32)

    def _make_local_sampler(self, module: int) -> EdgeSampler:
        choices = self._module_indices(module)

        def sample(
            count: int,
            rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
            src = rng.choice(
                choices,
                size=count,
                replace=True,
            ).astype(np.int32)
            dst = rng.choice(
                choices,
                size=count,
                replace=True,
            ).astype(np.int32)
            same = src == dst
            if np.any(same) and choices.size > 1:
                dst[same] = choices[
                    (np.searchsorted(choices, dst[same]) + 1) % choices.size
                ]
            return src, dst

        return sample

    def _make_inter_sampler(self, brain: int) -> EdgeSampler:
        modules = self._brain_modules(brain)
        n = self.config.neurons_per_module

        def sample(
            count: int,
            rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
            src_module = rng.choice(modules, size=count, replace=True)
            offset = rng.integers(1, len(modules), size=count)
            local_pos = src_module - modules[0]
            dst_module = modules[(local_pos + offset) % len(modules)]
            src = src_module * n + rng.integers(0, n, size=count)
            dst = dst_module * n + rng.integers(0, n, size=count)
            return src.astype(np.int32), dst.astype(np.int32)

        return sample

    def _gateway_global_indices(self, brain: int) -> np.ndarray:
        indices: list[np.ndarray] = []
        n = self.config.neurons_per_module
        for module in self._brain_modules(brain):
            indices.append(module * n + self.gateway_local_indices)
        return np.concatenate(indices).astype(np.int32, copy=False)

    def _make_bridge_sampler(self) -> EdgeSampler:
        left = self._gateway_global_indices(0)
        right = self._gateway_global_indices(1)

        def sample(
            count: int,
            rng: np.random.Generator,
        ) -> tuple[np.ndarray, np.ndarray]:
            direction = rng.integers(0, 2, size=count, dtype=np.int8)
            src = np.empty(count, dtype=np.int32)
            dst = np.empty(count, dtype=np.int32)
            lr = direction == 0
            rl = ~lr
            if np.any(lr):
                src[lr] = rng.choice(
                    left,
                    size=int(lr.sum()),
                    replace=True,
                )
                dst[lr] = rng.choice(
                    right,
                    size=int(lr.sum()),
                    replace=True,
                )
            if np.any(rl):
                src[rl] = rng.choice(
                    right,
                    size=int(rl.sum()),
                    replace=True,
                )
                dst[rl] = rng.choice(
                    left,
                    size=int(rl.sum()),
                    replace=True,
                )
            return src, dst

        return sample

    def token_vector(self, token: str) -> np.ndarray:
        if token not in self.token_codes:
            c = self.config
            code = np.zeros(c.token_dim, dtype=np.float32)
            active = self.rng.choice(
                c.token_dim,
                size=c.token_active,
                replace=False,
            )
            code[active] = 1.0 / np.sqrt(float(c.token_active))
            self.token_codes[token] = code
        return self.token_codes[token]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - float(np.max(logits))
        exp = np.exp(z)
        total = float(np.sum(exp))
        if not np.isfinite(total) or total <= 0.0:
            return np.full_like(
                logits,
                1.0 / logits.size,
                dtype=np.float32,
            )
        return (exp / total).astype(np.float32)

    def _route(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        logits = (self.w_router @ x) / c.router_temperature
        logits -= c.load_balance_strength * self.module_usage_ema
        probs = self._softmax(logits.astype(np.float32))
        selected = self.rng.choice(
            c.total_modules,
            size=c.router_top_k,
            replace=False,
            p=probs,
        )
        mask = np.zeros(c.total_modules, dtype=np.float32)
        mask[selected] = 1.0
        rate = c.module_usage_ema_rate
        self.module_usage_ema *= 1.0 - rate
        self.module_usage_ema += rate * mask
        return mask, probs

    def _sparsify_module(self, values: np.ndarray) -> np.ndarray:
        k = min(self.config.active_neurons_per_module, values.size)
        if k == values.size:
            out = np.maximum(values, 0.0).astype(np.float32)
        else:
            idx = np.argpartition(values, -k)[-k:]
            out = np.zeros_like(values, dtype=np.float32)
            out[idx] = np.maximum(values[idx], 0.0)
        norm = float(np.linalg.norm(out))
        if norm > 1e-8:
            out /= norm
        return out

    def _recurrent_drive(self, state: np.ndarray) -> np.ndarray:
        c = self.config
        drive = np.zeros(c.total_neurons, dtype=np.float32)
        for bank in self.local_banks:
            drive += bank.propagate(state, c.total_neurons)
        for bank in self.inter_banks:
            drive += bank.propagate(state, c.total_neurons)
        drive += self.bridge_bank.propagate(
            state,
            c.total_neurons,
        )
        return drive

    def _encode_tokens(
        self,
        tokens: list[str],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        c = self.config
        state = np.zeros(c.total_neurons, dtype=np.float32)
        traces: list[dict[str, Any]] = []

        for token in tokens:
            x = self.token_vector(token)
            route_mask, route_probs = self._route(x)
            recurrent = self._recurrent_drive(state)
            next_state = np.zeros_like(state)

            for module in range(c.total_modules):
                start, stop = self._module_bounds(module)
                selected = route_mask[module] > 0.0
                input_drive = (
                    self.w_input[module] @ x
                    if selected
                    else 0.0
                )
                gain = 1.0 if selected else c.dormant_gain
                raw = np.tanh(
                    c.state_carry * state[start:stop]
                    + c.recurrent_gain * recurrent[start:stop]
                    + gain * c.input_gain * input_drive
                ).astype(np.float32)
                next_state[start:stop] = self._sparsify_module(raw)

            activity = np.asarray(
                [
                    np.linalg.norm(next_state[s:e])
                    for s, e in (
                        self._module_bounds(m)
                        for m in range(c.total_modules)
                    )
                ],
                dtype=np.float32,
            )
            ar = c.module_usage_ema_rate
            self.module_activity_ema *= 1.0 - ar
            self.module_activity_ema += ar * activity
            traces.append(
                {
                    "token": token,
                    "x": x.copy(),
                    "pre": state.copy(),
                    "post": next_state.copy(),
                    "route_mask": route_mask,
                    "route_probs": route_probs,
                    "module_activity": activity,
                }
            )
            state = next_state
        return state, traces

    def _policy(
        self,
        state: np.ndarray,
        active_outputs: int | None = None,
    ) -> np.ndarray:
        c = self.config
        active = (
            c.output_dim
            if active_outputs is None
            else int(active_outputs)
        )
        if not 1 <= active <= c.output_dim:
            raise ValueError("active_outputs must be within output_dim")
        logits = (
            self.w_output @ state + self.output_bias
        ) / c.policy_temperature
        if active < c.output_dim:
            logits[active:] = -1.0e9
        probs = self._softmax(logits.astype(np.float32))
        if active < c.output_dim:
            probs[active:] = 0.0
            probs /= float(np.sum(probs))
        return probs

    def _learn_router_and_inputs(
        self,
        traces: list[dict[str, Any]],
        reward: float,
    ) -> None:
        c = self.config
        for trace in traces:
            x = trace["x"]
            selected = trace["route_mask"]
            probs = trace["route_probs"]
            # Compact REINFORCE-style approximation for top-k routing.
            router_error = selected - c.router_top_k * probs
            self.w_router += (
                c.router_learning_rate
                * reward
                * router_error[:, None]
                * x[None, :]
            )

            post = trace["post"]
            for module in np.flatnonzero(selected > 0.0):
                start, stop = self._module_bounds(int(module))
                self.w_input[module] += (
                    c.input_learning_rate
                    * reward
                    * np.outer(post[start:stop], x)
                ).astype(np.float32)

    def _learn_synapses(
        self,
        traces: list[dict[str, Any]],
        reward: float,
    ) -> None:
        c = self.config
        for trace in traces:
            pre = trace["pre"]
            post = trace["post"]
            for bank in self.local_banks:
                bank.apply_plasticity(pre, post, reward, c)
            for bank in self.inter_banks:
                bank.apply_plasticity(pre, post, reward, c)
            self.bridge_bank.apply_plasticity(
                pre,
                post,
                reward,
                c,
            )

    def _rewire_if_due(self) -> dict[str, int]:
        c = self.config
        if (
            c.rewire_interval <= 0
            or self.step_count % c.rewire_interval != 0
        ):
            return {}
        changed: dict[str, int] = {}
        for bank in [
            *self.local_banks,
            *self.inter_banks,
            self.bridge_bank,
        ]:
            count = bank.rewire(self.rng, c)
            if count:
                changed[bank.name] = count
        return changed

    def step(
        self,
        tokens: list[str],
        target: int,
        *,
        learn: bool = True,
        active_outputs: int | None = None,
        task_name: str | None = None,
        sample_action: bool | None = None,
    ) -> dict[str, Any]:
        c = self.config
        if not tokens:
            raise ValueError("tokens must not be empty")
        active = (
            c.output_dim
            if active_outputs is None
            else int(active_outputs)
        )
        if not 0 <= int(target) < active:
            raise ValueError("target outside active output range")

        state, traces = self._encode_tokens(tokens)
        probs = self._policy(state, active_outputs=active)
        if sample_action is None:
            sample_action = learn
        choice = (
            int(self.rng.choice(c.output_dim, p=probs))
            if sample_action
            else int(np.argmax(probs))
        )
        correct = choice == int(target)
        reward = (
            c.correct_reward
            if correct
            else c.incorrect_reward
        )

        eligibility = -probs.astype(np.float32)
        eligibility[choice] += 1.0
        if active < c.output_dim:
            eligibility[active:] = 0.0

        if learn:
            self.w_output += (
                c.output_learning_rate
                * reward
                * eligibility[:, None]
                * state[None, :]
            ).astype(np.float32)
            self.output_bias += (
                0.20
                * c.output_learning_rate
                * reward
                * eligibility
            ).astype(np.float32)
            np.clip(
                self.w_output,
                -c.output_weight_clip,
                c.output_weight_clip,
                out=self.w_output,
            )
            self._learn_router_and_inputs(traces, reward)
            self._learn_synapses(traces, reward)

        self.step_count += 1
        rewired = self._rewire_if_due() if learn else {}

        mean_activity = np.mean(
            np.stack(
                [
                    trace["module_activity"]
                    for trace in traces
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
        if task_name:
            history = self.task_activity_ema.setdefault(
                task_name,
                np.zeros(c.total_modules, dtype=np.float32),
            )
            history *= 0.98
            history += 0.02 * mean_activity

        return {
            "step": self.step_count,
            "tokens": list(tokens),
            "target": int(target),
            "choice": choice,
            "correct": bool(correct),
            "reward": float(reward),
            "probabilities": probs.tolist(),
            "module_activity": mean_activity.tolist(),
            "module_usage": self.module_usage_ema.tolist(),
            "rewired": rewired,
            "bridge": self.bridge_bank.snapshot(),
        }

    def all_banks(self) -> list[SparseSynapseBank]:
        return [
            *self.local_banks,
            *self.inter_banks,
            self.bridge_bank,
        ]

    def resource_report(self) -> dict[str, Any]:
        c = self.config
        dense_bytes = int(
            self.w_input.nbytes
            + self.w_router.nbytes
            + self.w_output.nbytes
            + self.output_bias.nbytes
            + self.module_usage_ema.nbytes
            + self.module_activity_ema.nbytes
        )
        sparse_bytes = int(
            sum(bank.nbytes for bank in self.all_banks())
        )
        synapses = int(
            sum(bank.budget for bank in self.all_banks())
        )
        return {
            "brains": c.brains,
            "modules": c.total_modules,
            "neurons": c.total_neurons,
            "synapse_budget": synapses,
            "bridge_budget": self.bridge_bank.budget,
            "dense_state_bytes": dense_bytes,
            "sparse_synapse_bytes": sparse_bytes,
            "total_parameter_state_bytes": (
                dense_bytes + sparse_bytes
            ),
            "total_parameter_state_mib": (
                dense_bytes + sparse_bytes
            ) / (1024.0 * 1024.0),
        }

    def topology_snapshot(
        self,
        max_edges_per_bank: int = 64,
    ) -> dict[str, Any]:
        c = self.config
        modules = []
        for module in range(c.total_modules):
            brain = module // c.modules_per_brain
            modules.append(
                {
                    "module": module,
                    "brain": brain,
                    "usage": float(
                        self.module_usage_ema[module]
                    ),
                    "activity": float(
                        self.module_activity_ema[module]
                    ),
                    "start_neuron": self._module_bounds(module)[0],
                    "neuron_count": c.neurons_per_module,
                }
            )

        banks = []
        for bank in self.all_banks():
            if bank.budget <= max_edges_per_bank:
                idx = np.arange(bank.budget)
            else:
                util = bank.utility()
                half = max_edges_per_bank // 2
                strongest = np.argsort(util)[-half:]
                remaining = max_edges_per_bank - strongest.size
                random_idx = self.rng.choice(
                    bank.budget,
                    size=remaining,
                    replace=False,
                )
                idx = np.unique(
                    np.concatenate((strongest, random_idx))
                )[:max_edges_per_bank]
            banks.append(
                {
                    **bank.snapshot(),
                    "edges": [
                        {
                            "src": int(bank.src[i]),
                            "dst": int(bank.dst[i]),
                            "weight": float(bank.weight[i]),
                            "usage": float(bank.usage_ema[i]),
                            "stability": float(bank.stability[i]),
                        }
                        for i in idx
                    ],
                }
            )

        specialization = {
            task: values.tolist()
            for task, values in sorted(
                self.task_activity_ema.items()
            )
        }
        return {
            "step": self.step_count,
            "modules": modules,
            "banks": banks,
            "task_activity": specialization,
            "resources": self.resource_report(),
        }
