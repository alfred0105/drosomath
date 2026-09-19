"""Task-independent feedback passed from an environment to a plastic brain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class LearningSignal:
    """Value feedback plus requested change per named output channel.

    The signal deliberately contains no task label, synapse index, or anatomy.
    """

    reward: float
    directional_error: Mapping[str, float] = field(default_factory=dict)
    novelty: float = 0.0
    surprise: float = 0.0
    success: bool | None = None

    def nonzero_directions(self) -> dict[str, float]:
        return {str(name): float(value) for name, value in self.directional_error.items() if float(value) != 0.0}
