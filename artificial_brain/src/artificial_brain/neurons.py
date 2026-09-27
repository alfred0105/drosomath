"""Vectorized Izhikevich neuron population."""

from typing import Dict, Optional, Tuple

import torch

from .device import resolve_device


PRESETS: Dict[str, Tuple[float, float, float, float]] = {
    "regular_spiking": (0.02, 0.2, -65.0, 8.0),
    "fast_spiking": (0.1, 0.2, -65.0, 2.0),
    "intrinsically_bursting": (0.02, 0.2, -55.0, 4.0),
}


class IzhikevichPopulation:
    """Tensor-backed population; all neuron updates are vectorized."""

    def __init__(self, size: int, device: Optional[str] = None,
                 neuron_types: Optional[torch.Tensor] = None, seed: int = 0):
        if size <= 0:
            raise ValueError("size must be positive")
        context = resolve_device(device)
        self.device, self.dtype, self.size = context.device, context.dtype, size
        types = torch.zeros(size, dtype=torch.long) if neuron_types is None else neuron_types.detach().cpu().long()
        if types.shape != (size,) or torch.any((types < 0) | (types >= len(PRESETS))):
            raise ValueError("neuron_types must have shape [size] and valid preset indices")
        table = torch.tensor(list(PRESETS.values()), dtype=self.dtype)
        params = table[types]
        self.a, self.b, self.c, self.d = (v.to(self.device) for v in params.unbind(dim=1))
        self.v = torch.full((size,), -65.0, dtype=self.dtype, device=self.device)
        self.u = self.b * self.v
        self.spike = torch.zeros(size, dtype=torch.bool, device=self.device)
        self.neuron_type = types.to(self.device)
        self.alive = torch.ones(size, dtype=torch.bool, device=self.device)

    @classmethod
    def with_preset(cls, size: int, preset: str, device: Optional[str] = None) -> "IzhikevichPopulation":
        if preset not in PRESETS:
            raise ValueError("unknown neuron preset: {}".format(preset))
        return cls(size, device=device, neuron_types=torch.full((size,), list(PRESETS).index(preset)))

    @torch.no_grad()
    def step(self, current: torch.Tensor, dt_ms: float = 1.0) -> torch.Tensor:
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        current = torch.as_tensor(current, dtype=self.dtype, device=self.device)
        if current.shape != (self.size,):
            raise ValueError("current must have shape [{}]".format(self.size))
        # Two half steps improve stability while preserving explicit millisecond dt.
        half = 0.5 * dt_ms
        for _ in range(2):
            self.v.add_(half * (0.04 * self.v.square() + 5.0 * self.v + 140.0 - self.u + current))
        self.u.add_(dt_ms * self.a * (self.b * self.v - self.u))
        self.spike.copy_((self.v >= 30.0) & self.alive)
        self.v.copy_(torch.where(self.spike, self.c, self.v))
        self.u.add_(self.spike.to(self.dtype) * self.d)
        self.v.copy_(torch.where(self.alive, self.v, torch.full_like(self.v, -65.0)))
        self.u.copy_(torch.where(self.alive, self.u, torch.zeros_like(self.u)))
        return self.spike

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().clone() for name in
                ("v", "u", "a", "b", "c", "d", "spike", "neuron_type", "alive")}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for name in ("v", "u", "a", "b", "c", "d", "spike", "neuron_type", "alive"):
            value = state[name].to(device=self.device, dtype=getattr(self, name).dtype)
            if value.shape != (self.size,):
                raise ValueError("invalid shape for neuron state {}".format(name))
            getattr(self, name).copy_(value)

    def reset_state(self) -> None:
        self.v.fill_(-65.0)
        self.u.copy_(self.b * self.v)
        self.spike.zero_()
