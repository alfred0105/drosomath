from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StructuralOverlayConfig:
    """Slow structural plasticity layered over immutable connectome anatomy.

    Each learned overlay edge consumes one donor anatomical edge by driving that
    donor's plastic multiplier to its configured minimum. The source connectome
    is never mutated, so anatomical and learned structure remain auditable.
    Candidate growth is constrained to two-hop closure in the anatomical graph.
    """

    enabled: bool = True
    max_edges: int = 2048
    rewire_per_cycle: int = 64
    candidate_pool_size: int = 48
    fanout_per_hop: int = 8
    initial_strength: float = 1.0
    min_strength: float = 0.10
    max_strength: float = 6.0
    learning_rate: float = 0.02
    usage_alpha: float = 0.05
    reward_ema_alpha: float = 0.10
    stability_gain: float = 0.02
    protected_stability: float = 0.75
    donor_protected_stability: float = 0.25
    min_age_cycles: int = 2

    def __post_init__(self) -> None:
        if self.max_edges < 0 or self.rewire_per_cycle < 0:
            raise ValueError("structural edge limits must be >= 0")
        if self.candidate_pool_size < 2 or self.fanout_per_hop < 1:
            raise ValueError("candidate pool/fanout are too small")
        if not 0.0 < self.initial_strength <= self.max_strength:
            raise ValueError("initial_strength must be in (0, max_strength]")
        if not 0.0 <= self.min_strength <= self.initial_strength:
            raise ValueError("min_strength must be in [0, initial_strength]")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")
        if not 0.0 < self.usage_alpha <= 1.0:
            raise ValueError("usage_alpha must be in (0, 1]")
        if not 0.0 < self.reward_ema_alpha <= 1.0:
            raise ValueError("reward_ema_alpha must be in (0, 1]")
        if self.stability_gain < 0.0:
            raise ValueError("stability_gain must be >= 0")
        if not 0.0 <= self.protected_stability <= 1.0:
            raise ValueError("protected_stability must be in [0, 1]")
        if not 0.0 <= self.donor_protected_stability <= 1.0:
            raise ValueError("donor_protected_stability must be in [0, 1]")
        if self.min_age_cycles < 0:
            raise ValueError("min_age_cycles must be >= 0")


class LearnedStructuralOverlay:
    """Fixed-budget learned edges on top of an immutable sparse connectome."""

    def __init__(self, connectome, plasticity, *, config: StructuralOverlayConfig | None = None) -> None:
        import numpy as np

        self.np = np
        self.connectome = connectome
        self.plasticity = plasticity
        self.config = config or StructuralOverlayConfig()
        c = self.config.max_edges
        n = int(connectome.neuron_count)

        self.active = np.zeros(c, dtype=np.bool_)
        self.pre_index = np.full(c, -1, dtype=np.int32)
        self.post_index = np.full(c, -1, dtype=np.int32)
        self.signed_strength = np.zeros(c, dtype=np.float32)
        self.usage_ema = np.zeros(c, dtype=np.float32)
        self.eligibility = np.zeros(c, dtype=np.float32)
        self.reward_ema = np.zeros(c, dtype=np.float32)
        self.stability = np.zeros(c, dtype=np.float32)
        self.age_cycles = np.zeros(c, dtype=np.int32)
        self.donor_edge = np.full(c, -1, dtype=np.int32)

        self.cycle_activity = np.zeros(n, dtype=np.float32)
        self.cycle_rewarded_activity = np.zeros(n, dtype=np.float32)
        self._trial_counts = np.zeros(n, dtype=np.int32)
        self._trial_touched: list[int] = []
        self._adjacency: dict[int, object] = {}
        self._cycles = 0
        self._total_regrown = 0
        self._total_replaced = 0
        self._rebuild_adjacency()

    @property
    def edge_count(self) -> int:
        return int(self.active.sum())

    def reset(self) -> None:
        donors = self.donor_edge[self.active]
        donors = donors[donors >= 0]
        if len(donors):
            self.plasticity.multiplier[donors] = 1.0
        self.active.fill(False)
        self.pre_index.fill(-1)
        self.post_index.fill(-1)
        self.signed_strength.fill(0.0)
        self.usage_ema.fill(0.0)
        self.eligibility.fill(0.0)
        self.reward_ema.fill(0.0)
        self.stability.fill(0.0)
        self.age_cycles.fill(0)
        self.donor_edge.fill(-1)
        self.cycle_activity.fill(0.0)
        self.cycle_rewarded_activity.fill(0.0)
        self._clear_trial_activity()
        self._cycles = 0
        self._total_regrown = 0
        self._total_replaced = 0
        self._rebuild_adjacency()

    def record_firing(self, fired_indices) -> None:
        if not self.config.enabled or len(fired_indices) == 0:
            return
        unique, counts = self.np.unique(fired_indices, return_counts=True)
        self.cycle_activity[unique] += counts.astype(self.np.float32, copy=False)
        for idx_raw, count_raw in zip(unique, counts):
            idx = int(idx_raw)
            if self._trial_counts[idx] == 0:
                self._trial_touched.append(idx)
            self._trial_counts[idx] += int(count_raw)

    def slots_for_pre(self, pre_index: int):
        slots = self._adjacency.get(int(pre_index))
        if slots is None:
            return self.np.empty(0, dtype=self.np.int32)
        return slots

    def record_use(self, slots) -> None:
        if len(slots) == 0:
            return
        u = self.usage_ema[slots]
        u += self.config.usage_alpha * (1.0 - u)
        self.eligibility[slots] += 1.0

    def learn_from_reward(self, reward: float) -> dict[str, object]:
        np = self.np
        slots = np.flatnonzero(self.active & (self.eligibility > 0.0))
        updated = 0
        if len(slots):
            mag = np.abs(self.signed_strength[slots])
            sign = np.where(self.signed_strength[slots] < 0.0, -1.0, 1.0)
            protection = 1.0 - 0.80 * self.stability[slots]
            delta = (
                self.config.learning_rate
                * float(reward)
                * np.maximum(self.usage_ema[slots], 0.05)
                * self.eligibility[slots]
                * protection
            )
            mag = np.clip(mag + delta, self.config.min_strength, self.config.max_strength)
            self.signed_strength[slots] = (sign * mag).astype(np.float32, copy=False)
            alpha = self.config.reward_ema_alpha
            self.reward_ema[slots] += alpha * (float(reward) - self.reward_ema[slots])
            if reward > 0.0:
                self.stability[slots] = np.clip(
                    self.stability[slots]
                    + self.config.stability_gain * self.usage_ema[slots],
                    0.0,
                    1.0,
                )
            updated = int(len(slots))

        if self._trial_touched:
            touched = np.asarray(self._trial_touched, dtype=np.int32)
            if reward > 0.0:
                counts = self._trial_counts[touched].astype(np.float32, copy=False)
                self.cycle_rewarded_activity[touched] += float(reward) * counts
        self.usage_ema[self.active] *= 0.995
        self.eligibility.fill(0.0)
        self._clear_trial_activity()
        return {"edge_updates": updated, "active_edges": self.edge_count}

    def rewire(self, *, cycle_label: str = "") -> dict[str, object]:
        """Run one slow structural cycle using two-hop anatomical closure."""
        np = self.np
        self._cycles += 1
        self.age_cycles[self.active] += 1
        if not self.config.enabled or self.config.rewire_per_cycle == 0 or self.config.max_edges == 0:
            self._reset_cycle_activity()
            return {"cycle": self._cycles, "label": cycle_label, "added": 0, "replaced": 0, "active_edges": self.edge_count}

        candidates = self._candidate_pairs()
        if not candidates:
            self._reset_cycle_activity()
            return {"cycle": self._cycles, "label": cycle_label, "added": 0, "replaced": 0, "active_edges": self.edge_count, "candidate_count": 0}

        free = np.flatnonzero(~self.active)
        add_n = min(len(free), len(candidates), self.config.rewire_per_cycle)
        donors = self._select_donor_edges(add_n)
        add_n = min(add_n, len(donors))

        added_rows = []
        for slot_raw, donor_raw, pair in zip(free[:add_n], donors[:add_n], candidates[:add_n]):
            slot = int(slot_raw)
            donor = int(donor_raw)
            pre, post = pair
            self._silence_donor(donor)
            self._write_slot(slot, pre, post, donor)
            added_rows.append((pre, post, donor))

        used_pairs = set(candidates[:add_n])
        remaining = [pair for pair in candidates[add_n:] if pair not in used_pairs]
        replace_budget = self.config.rewire_per_cycle - add_n
        replace_slots = self._replaceable_slots(replace_budget)
        replaced_rows = []
        for slot_raw, pair in zip(replace_slots, remaining):
            slot = int(slot_raw)
            old = (int(self.pre_index[slot]), int(self.post_index[slot]))
            donor = int(self.donor_edge[slot])
            self._write_slot(slot, pair[0], pair[1], donor)
            replaced_rows.append({"old": old, "new": pair, "donor_edge": donor})

        self._total_regrown += len(added_rows)
        self._total_replaced += len(replaced_rows)
        self._rebuild_adjacency()
        self._reset_cycle_activity()
        return {
            "cycle": self._cycles,
            "label": cycle_label,
            "candidate_count": len(candidates),
            "added": len(added_rows),
            "replaced": len(replaced_rows),
            "active_edges": self.edge_count,
            "added_preview": [
                {"pre_index": a, "post_index": b, "donor_edge": d}
                for a, b, d in added_rows[:8]
            ],
            "replaced_preview": replaced_rows[:8],
        }

    def summary(self) -> dict[str, object]:
        np = self.np
        slots = np.flatnonzero(self.active)
        return {
            "enabled": bool(self.config.enabled),
            "active_edges": int(len(slots)),
            "max_edges": int(self.config.max_edges),
            "cycles": int(self._cycles),
            "total_regrown": int(self._total_regrown),
            "total_replaced": int(self._total_replaced),
            "mean_abs_strength": float(np.mean(np.abs(self.signed_strength[slots]))) if len(slots) else 0.0,
            "mean_stability": float(np.mean(self.stability[slots])) if len(slots) else 0.0,
            "protected_edges": int(np.sum(self.stability[slots] >= self.config.protected_stability)) if len(slots) else 0,
            "donor_edges_silenced": int(len(slots)),
        }

    def checkpoint_payload(self, prefix: str = "structural__") -> dict[str, object]:
        np = self.np
        slots = np.flatnonzero(self.active).astype(np.int32, copy=False)
        payload: dict[str, object] = {
            prefix + "present": np.asarray([True], dtype=np.bool_),
            prefix + "slots": slots,
            prefix + "pre_index": self.pre_index[slots],
            prefix + "post_index": self.post_index[slots],
            prefix + "signed_strength": self.signed_strength[slots],
            prefix + "usage_ema": self.usage_ema[slots],
            prefix + "reward_ema": self.reward_ema[slots],
            prefix + "stability": self.stability[slots],
            prefix + "age_cycles": self.age_cycles[slots],
            prefix + "donor_edge": self.donor_edge[slots],
            prefix + "cycles": np.asarray([self._cycles], dtype=np.int64),
            prefix + "total_regrown": np.asarray([self._total_regrown], dtype=np.int64),
            prefix + "total_replaced": np.asarray([self._total_replaced], dtype=np.int64),
        }
        for key, value in asdict(self.config).items():
            payload[prefix + "config__" + key] = np.asarray([value])
        return payload

    def restore_from_checkpoint(self, data, prefix: str = "structural__") -> None:
        np = self.np
        self.active.fill(False)
        self.pre_index.fill(-1)
        self.post_index.fill(-1)
        self.signed_strength.fill(0.0)
        self.usage_ema.fill(0.0)
        self.eligibility.fill(0.0)
        self.reward_ema.fill(0.0)
        self.stability.fill(0.0)
        self.age_cycles.fill(0)
        self.donor_edge.fill(-1)
        slots = data[prefix + "slots"].astype(np.int64, copy=False)
        if len(slots) and int(slots.max()) >= self.config.max_edges:
            raise ValueError("structural checkpoint exceeds configured overlay capacity")
        self.active[slots] = True
        for name in ("pre_index", "post_index", "signed_strength", "usage_ema", "reward_ema", "stability", "age_cycles", "donor_edge"):
            getattr(self, name)[slots] = data[prefix + name]
        self._cycles = int(data[prefix + "cycles"][0]) if prefix + "cycles" in data else 0
        self._total_regrown = int(data[prefix + "total_regrown"][0]) if prefix + "total_regrown" in data else int(len(slots))
        self._total_replaced = int(data[prefix + "total_replaced"][0]) if prefix + "total_replaced" in data else 0
        donors = self.donor_edge[slots]
        donors = donors[donors >= 0]
        if len(donors):
            self.plasticity.multiplier[donors] = self.plasticity.config.min_multiplier
        self._rebuild_adjacency()

    def _candidate_pairs(self) -> list[tuple[int, int]]:
        np = self.np
        score = self.cycle_activity + 2.0 * self.cycle_rewarded_activity
        positive = np.flatnonzero(score > 0.0)
        if len(positive) < 2:
            return []
        pool_n = min(self.config.candidate_pool_size, len(positive))
        if len(positive) > pool_n:
            local = np.argpartition(score[positive], -pool_n)[-pool_n:]
            pool = positive[local]
        else:
            pool = positive
        pool = pool[np.argsort(score[pool])[::-1]]

        existing_overlay = {
            (int(self.pre_index[s]), int(self.post_index[s]))
            for s in np.flatnonzero(self.active)
        }
        ranked: dict[tuple[int, int], float] = {}
        for pre_raw in pool:
            pre = int(pre_raw)
            first = self._top_outgoing(pre, self.config.fanout_per_hop)
            for mid in first:
                second = self._top_outgoing(int(mid), self.config.fanout_per_hop)
                for post_raw in second:
                    post = int(post_raw)
                    if pre == post or (pre, post) in existing_overlay:
                        continue
                    if self._has_anatomical_edge(pre, post):
                        continue
                    pair_score = float(score[pre] + score[post] + 0.25 * score[int(mid)])
                    old = ranked.get((pre, post))
                    if old is None or pair_score > old:
                        ranked[(pre, post)] = pair_score
        ordered = sorted(ranked.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        return [pair for pair, _ in ordered]

    def _top_outgoing(self, pre: int, count: int):
        np = self.np
        start = int(self.connectome.indptr[pre])
        stop = int(self.connectome.indptr[pre + 1])
        if stop <= start:
            return np.empty(0, dtype=np.int32)
        posts = self.connectome.post_indices[start:stop]
        strength = np.abs(self.connectome.signed_synapse_counts[start:stop])
        k = min(count, len(posts))
        if len(posts) <= k:
            order = np.argsort(strength)[::-1]
        else:
            order = np.argpartition(strength, -k)[-k:]
            order = order[np.argsort(strength[order])[::-1]]
        return posts[order]

    def _has_anatomical_edge(self, pre: int, post: int) -> bool:
        start = int(self.connectome.indptr[pre])
        stop = int(self.connectome.indptr[pre + 1])
        if stop <= start:
            return False
        return bool(self.np.any(self.connectome.post_indices[start:stop] == int(post)))

    def _select_donor_edges(self, count: int):
        np = self.np
        if count <= 0:
            return np.empty(0, dtype=np.int32)
        used = self.donor_edge[self.active]
        used = used[used >= 0]
        eligible = self.plasticity.plastic_mask.copy()
        eligible &= self.plasticity.stability < self.config.donor_protected_stability
        eligible &= self.plasticity.multiplier > self.plasticity.config.min_multiplier + 1e-6
        if len(used):
            eligible[used] = False
        idx = np.flatnonzero(eligible)
        if len(idx) == 0:
            return np.empty(0, dtype=np.int32)
        score = (
            self.plasticity.usage_ema[idx]
            + 2.0 * self.plasticity.stability[idx]
            + 0.10 * np.abs(self.plasticity.multiplier[idx] - 1.0)
        )
        k = min(count, len(idx))
        if len(idx) <= k:
            order = np.argsort(score)
        else:
            order = np.argpartition(score, k - 1)[:k]
            order = order[np.argsort(score[order])]
        return idx[order].astype(np.int32, copy=False)

    def _replaceable_slots(self, count: int):
        np = self.np
        if count <= 0:
            return np.empty(0, dtype=np.int32)
        eligible = np.flatnonzero(
            self.active
            & (self.age_cycles >= self.config.min_age_cycles)
            & (self.stability < self.config.protected_stability)
        )
        if len(eligible) == 0:
            return np.empty(0, dtype=np.int32)
        score = (
            self.usage_ema[eligible]
            + 2.0 * np.maximum(0.0, self.reward_ema[eligible])
            + 4.0 * self.stability[eligible]
        )
        k = min(count, len(eligible))
        if len(eligible) <= k:
            order = np.argsort(score)
        else:
            order = np.argpartition(score, k - 1)[:k]
            order = order[np.argsort(score[order])]
        return eligible[order].astype(np.int32, copy=False)

    def _source_sign(self, pre: int) -> float:
        sign = getattr(self.connectome, "presynaptic_sign", None)
        if sign is not None and len(sign) == self.connectome.neuron_count:
            return -1.0 if int(sign[pre]) < 0 else 1.0
        start = int(self.connectome.indptr[pre])
        stop = int(self.connectome.indptr[pre + 1])
        if stop > start:
            total = float(self.np.sum(self.connectome.signed_synapse_counts[start:stop]))
            if total < 0.0:
                return -1.0
        return 1.0

    def _silence_donor(self, edge: int) -> None:
        self.plasticity.multiplier[int(edge)] = self.plasticity.config.min_multiplier
        self.plasticity.eligibility[int(edge)] = 0.0

    def _write_slot(self, slot: int, pre: int, post: int, donor: int) -> None:
        self.active[slot] = True
        self.pre_index[slot] = int(pre)
        self.post_index[slot] = int(post)
        self.signed_strength[slot] = self._source_sign(pre) * self.config.initial_strength
        self.usage_ema[slot] = 0.0
        self.eligibility[slot] = 0.0
        self.reward_ema[slot] = 0.0
        self.stability[slot] = 0.0
        self.age_cycles[slot] = 0
        self.donor_edge[slot] = int(donor)

    def _rebuild_adjacency(self) -> None:
        np = self.np
        adjacency: dict[int, list[int]] = {}
        for slot_raw in np.flatnonzero(self.active):
            slot = int(slot_raw)
            adjacency.setdefault(int(self.pre_index[slot]), []).append(slot)
        self._adjacency = {
            pre: np.asarray(slots, dtype=np.int32)
            for pre, slots in adjacency.items()
        }

    def _clear_trial_activity(self) -> None:
        if self._trial_touched:
            touched = self.np.asarray(self._trial_touched, dtype=self.np.int32)
            self._trial_counts[touched] = 0
            self._trial_touched.clear()

    def _reset_cycle_activity(self) -> None:
        self.cycle_activity.fill(0.0)
        self.cycle_rewarded_activity.fill(0.0)

