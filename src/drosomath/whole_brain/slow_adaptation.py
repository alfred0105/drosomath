"""Optional generic transient spike-history adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SlowAdaptationConfig:
    """Configuration for non-persistent threshold adaptation."""

    enabled: bool = False
    tau_ms: float = 30.0
    spike_increment_mv: float = 0.75
    max_adaptation_mv: float = 3.0

    def __post_init__(self) -> None:
        if self.tau_ms <= 0.0:
            raise ValueError("tau_ms must be > 0")
        if self.spike_increment_mv < 0.0:
            raise ValueError("spike_increment_mv must be >= 0")
        if self.max_adaptation_mv < 0.0:
            raise ValueError("max_adaptation_mv must be >= 0")
        if self.spike_increment_mv > self.max_adaptation_mv:
            raise ValueError("spike_increment_mv must not exceed max_adaptation_mv")

    def decay_factor(self, dt_ms: float) -> float:
        if dt_ms <= 0.0:
            raise ValueError("dt_ms must be > 0")
        return float(math.exp(-float(dt_ms) / float(self.tau_ms)))


__all__ = ["SlowAdaptationConfig"]
