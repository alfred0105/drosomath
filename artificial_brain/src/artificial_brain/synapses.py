"""Sparse synapses represented as an edge list."""

from typing import Optional

import torch

from .device import resolve_device


class SynapseEdges:
    def __init__(self, pre: torch.Tensor, post: torch.Tensor, weight: torch.Tensor,
                 num_neurons: int, delay_steps: Optional[torch.Tensor] = None,
                 device: Optional[str] = None):
        context = resolve_device(device)
        self.device, self.dtype, self.num_neurons = context.device, context.dtype, num_neurons
        self.pre = torch.as_tensor(pre, dtype=torch.long, device=self.device)
        self.post = torch.as_tensor(post, dtype=torch.long, device=self.device)
        self.weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        if self.pre.ndim != 1 or self.post.shape != self.pre.shape or self.weight.shape != self.pre.shape:
            raise ValueError("pre, post, and weight must be equal-length vectors")
        if self.pre.numel() and (torch.any(self.pre < 0) or torch.any(self.pre >= num_neurons) or
                                torch.any(self.post < 0) or torch.any(self.post >= num_neurons)):
            raise ValueError("edge index outside population")
        self.delay_steps = (torch.zeros_like(self.pre) if delay_steps is None else
                            torch.as_tensor(delay_steps, dtype=torch.long, device=self.device))
        if self.delay_steps.shape != self.pre.shape or torch.any(self.delay_steps < 0):
            raise ValueError("delay_steps must be nonnegative and match edge count")

    @property
    def num_edges(self) -> int:
        return self.pre.numel()

    @torch.no_grad()
    def propagate(self, spikes: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        spikes = torch.as_tensor(spikes, device=self.device, dtype=torch.bool)
        if spikes.shape != (self.num_neurons,):
            raise ValueError("spikes must have shape [{}]".format(self.num_neurons))
        if out is None:
            out = torch.zeros(self.num_neurons, dtype=self.dtype, device=self.device)
        else:
            out.zero_()
        active = spikes[self.pre]
        out.index_add_(0, self.post[active], self.weight[active])
        return out

    def state_dict(self):
        return {name: getattr(self, name).detach().clone() for name in ("pre", "post", "weight", "delay_steps")}
