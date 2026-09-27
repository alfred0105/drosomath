"""Small recurrent spiking network with fixed interface populations."""

from typing import Dict, Optional

import torch

from .device import resolve_device
from .neurons import IzhikevichPopulation
from .synapses import SynapseEdges


class Brain:
    def __init__(self, input_neurons: int = 8, hidden_neurons: int = 256,
                 output_neurons: int = 2, initial_synapses: int = 5000,
                 dt_ms: float = 1.0, seed: int = 0, device: Optional[str] = None):
        if min(input_neurons, hidden_neurons, output_neurons) <= 0 or initial_synapses < 0:
            raise ValueError("population sizes must be positive and synapse count nonnegative")
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        context = resolve_device(device)
        self.device, self.dtype = context.device, context.dtype
        self.input_count, self.hidden_count, self.output_count = input_neurons, hidden_neurons, output_neurons
        self.num_neurons, self.dt_ms, self.seed = input_neurons + hidden_neurons + output_neurons, dt_ms, seed
        self.input_indices = torch.arange(input_neurons, device=self.device)
        self.hidden_indices = torch.arange(input_neurons, input_neurons + hidden_neurons, device=self.device)
        self.output_indices = torch.arange(input_neurons + hidden_neurons, self.num_neurons, device=self.device)
        self.generator = torch.Generator().manual_seed(seed)
        pre = torch.randint(self.num_neurons, (initial_synapses,), generator=self.generator)
        post = torch.randint(self.num_neurons, (initial_synapses,), generator=self.generator)
        magnitude = 0.5 + torch.rand(initial_synapses, generator=self.generator)
        signs = torch.where(pre < self.input_count + self.hidden_count, 1.0, -1.0)
        weights = magnitude * signs
        self.neurons = IzhikevichPopulation(self.num_neurons, device=str(self.device), seed=seed)
        self.synapses = SynapseEdges(pre, post, weights, self.num_neurons, device=str(self.device))
        self.step_count = 0
        self.rng_state = self.generator.get_state()
        self.config = {"input_neurons": input_neurons, "hidden_neurons": hidden_neurons,
                       "output_neurons": output_neurons, "initial_synapses": initial_synapses,
                       "dt_ms": dt_ms, "seed": seed}
        self._synaptic_current = torch.zeros(self.num_neurons, dtype=self.dtype, device=self.device)
        self._input_current = torch.zeros_like(self._synaptic_current)

    @torch.no_grad()
    def step(self, input_current: torch.Tensor) -> torch.Tensor:
        current = torch.as_tensor(input_current, dtype=self.dtype, device=self.device)
        if current.shape == (self.input_count,):
            self._input_current.zero_()
            self._input_current[self.input_indices] = current
        elif current.shape == (self.num_neurons,):
            self._input_current.copy_(current)
        else:
            raise ValueError("input_current must have shape [input_neurons] or [num_neurons]")
        self.synapses.propagate(self.neurons.spike, out=self._synaptic_current)
        self._synaptic_current.add_(self._input_current)
        self.neurons.step(self._synaptic_current, self.dt_ms)
        self.step_count += 1
        return self.get_output_spikes()

    def reset_state(self) -> None:
        self.neurons.reset_state()
        self._synaptic_current.zero_()
        self._input_current.zero_()
        self.step_count = 0

    def get_output_spikes(self) -> torch.Tensor:
        return self.neurons.spike[self.output_indices]

    def state_dict(self) -> Dict[str, object]:
        return {"neurons": self.neurons.state_dict(), "synapses": self.synapses.state_dict(),
                "input_indices": self.input_indices.detach().clone(),
                "hidden_indices": self.hidden_indices.detach().clone(),
                "output_indices": self.output_indices.detach().clone(), "step_count": self.step_count,
                "rng_state": self.generator.get_state().clone(), "config": dict(self.config)}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.neurons.load_state_dict(state["neurons"])
        for name in ("pre", "post", "weight", "delay_steps"):
            value = state["synapses"][name].to(device=self.device, dtype=getattr(self.synapses, name).dtype)
            if value.shape != getattr(self.synapses, name).shape:
                raise ValueError("checkpoint synapse shape mismatch")
            getattr(self.synapses, name).copy_(value)
        for name in ("input_indices", "hidden_indices", "output_indices"):
            value = state[name].to(device=self.device, dtype=torch.long)
            if not torch.equal(value, getattr(self, name)):
                raise ValueError("checkpoint {} do not match this brain's fixed interface".format(name))
        self.step_count = int(state["step_count"])
        self.config = dict(state.get("config", self.config))
        self.dt_ms = float(self.config.get("dt_ms", self.dt_ms))
        self.seed = int(self.config.get("seed", self.seed))
        self.generator.set_state(state["rng_state"].cpu())
        self.rng_state = self.generator.get_state()
