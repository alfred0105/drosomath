"""Portable spiking-network bootstrap based on Izhikevich neurons."""

from .brain import Brain
from .device import DeviceContext, resolve_device
from .neurons import IzhikevichPopulation
from .synapses import SynapseEdges

__all__ = ["Brain", "DeviceContext", "IzhikevichPopulation", "SynapseEdges", "resolve_device"]
