"""Task-independent transient local Hebbian binding configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransientHebbianBindingConfig:
    enabled: bool = False
    pre_trace_tau_ms: float = 20.0
    binding_tau_ms: float = 40.0
    binding_increment: float = 0.10
    max_binding_gain: float = 0.50

    def __post_init__(self) -> None:
        if self.pre_trace_tau_ms <= 0.0:
            raise ValueError("pre_trace_tau_ms must be > 0")
        if self.binding_tau_ms <= 0.0:
            raise ValueError("binding_tau_ms must be > 0")
        if self.binding_increment < 0.0:
            raise ValueError("binding_increment must be >= 0")
        if self.max_binding_gain < 0.0:
            raise ValueError("max_binding_gain must be >= 0")


__all__ = ["TransientHebbianBindingConfig"]
