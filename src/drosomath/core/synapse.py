from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SynapseState:
    """Minimal mutable state for one plastic synapse.

    This first version intentionally keeps the state compact. Structural
    rewiring, bridge synapses, and module-level specialization will be layered
    on later instead of being mixed into the initial implementation.
    """

    pre_id: int
    post_id: int
    weight: float
    stability: float = 0.0
    usage_count: int = 0
    reward_ema: float = 0.0
    age: int = 0
    last_used_step: int = -1
    alive: bool = True

    def record_use(
        self,
        *,
        step: int,
        reward: float = 0.0,
        reward_alpha: float = 0.05,
    ) -> None:
        """Record that this synapse participated in the current computation."""
        if not self.alive:
            return

        if not 0.0 < reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be in (0, 1]")

        self.usage_count += 1
        self.last_used_step = step
        self.reward_ema += reward_alpha * (reward - self.reward_ema)

    def tick(self) -> None:
        """Advance the synapse age by one structural-plasticity interval."""
        if self.alive:
            self.age += 1

    def decay_weight(self, rate: float, *, floor: float = 0.0) -> None:
        """Apply slow weight decay without deleting the synapse."""
        if not self.alive:
            return
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")

        self.weight = max(floor, self.weight * (1.0 - rate))

    def utility(self, *, usage_scale: float = 1.0) -> float:
        """Return a simple score that later pruning logic can build on.

        Usage says whether the connection is active, reward_ema says whether it
        is useful, and stability protects consolidated memories from aggressive
        pruning.
        """
        return (
            usage_scale * float(self.usage_count)
            + self.reward_ema
            + self.stability
        )

    def prune(self) -> None:
        """Mark this synapse inactive; storage reclamation happens elsewhere."""
        self.alive = False
