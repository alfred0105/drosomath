"""Generic transient presynaptic short-term depression (STD).

The mechanism is deliberately task-independent: it observes only a
presynaptic neuron's firing step and recovers its release factor lazily when
that neuron fires again.  No output, label, or curriculum state is referenced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresynapticDepressionConfig:
    enabled: bool = False
    recovery_tau_ms: float = 30.0
    depression_fraction: float = 0.15
    min_release_factor: float = 0.50

    def __post_init__(self) -> None:
        if self.recovery_tau_ms <= 0.0:
            raise ValueError("recovery_tau_ms must be > 0")
        if not 0.0 <= self.depression_fraction < 1.0:
            raise ValueError("depression_fraction must be in [0, 1)")
        if not 0.0 < self.min_release_factor <= 1.0:
            raise ValueError("min_release_factor must be in (0, 1]")


__all__ = ["PresynapticDepressionConfig"]
