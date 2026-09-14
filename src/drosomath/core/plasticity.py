from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .synapse import SynapseState


@dataclass(frozen=True, slots=True)
class SynapseUse:
    step: int
    pre_id: int
    post_id: int


class PlasticityTracker:
    """Event-driven usage and delayed-reward tracker for plastic synapses.

    This phase intentionally does not change synaptic weights. It only records
    which existing synapses carried activity and assigns delayed reward credit
    to recently used connections. Weight learning comes in a later phase.
    """

    def __init__(
        self,
        synapses: Iterable[SynapseState] = (),
        *,
        reward_window: int = 32,
        reward_alpha: float = 0.05,
    ) -> None:
        if reward_window < 0:
            raise ValueError("reward_window must be >= 0")
        if not 0.0 < reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be in (0, 1]")

        self.reward_window = reward_window
        self.reward_alpha = reward_alpha
        self._synapses: dict[tuple[int, int], SynapseState] = {}
        self._recent: deque[SynapseUse] = deque()
        self._last_step = -1

        for synapse in synapses:
            self.register_synapse(synapse)

    def register_synapse(self, synapse: SynapseState) -> None:
        """Register one connection so future spike transfers can be tracked."""
        key = (synapse.pre_id, synapse.post_id)
        if key in self._synapses:
            raise ValueError(f"duplicate synapse {key}")
        self._synapses[key] = synapse

    def record_transfer(self, *, pre_id: int, post_id: int, step: int) -> bool:
        """Record one spike transfer through an existing live synapse."""
        self._check_step(step)
        self._evict_old(step)

        synapse = self._synapses.get((pre_id, post_id))
        if synapse is None or not synapse.alive:
            return False

        synapse.record_use(step=step)
        self._recent.append(
            SynapseUse(step=step, pre_id=pre_id, post_id=post_id)
        )
        return True

    def apply_reward(self, *, reward: float, step: int) -> int:
        """Assign delayed reward to unique synapses used inside the reward window."""
        self._check_step(step)
        self._evict_old(step)

        credited: set[tuple[int, int]] = set()
        for event in self._recent:
            key = (event.pre_id, event.post_id)
            if key in credited:
                continue

            synapse = self._synapses.get(key)
            if synapse is not None and synapse.alive:
                synapse.record_reward(
                    reward=reward,
                    reward_alpha=self.reward_alpha,
                )
                credited.add(key)

        return len(credited)

    def clear_recent(self) -> None:
        """Drop pending credit-assignment events without changing synapses."""
        self._recent.clear()

    def _check_step(self, step: int) -> None:
        if step < self._last_step:
            raise ValueError("step must be monotonic")
        self._last_step = step

    def _evict_old(self, step: int) -> None:
        cutoff = step - self.reward_window
        while self._recent and self._recent[0].step < cutoff:
            self._recent.popleft()
