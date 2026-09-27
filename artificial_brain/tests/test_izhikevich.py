import torch

from artificial_brain.neurons import IzhikevichPopulation


def test_spike_and_reset():
    neurons = IzhikevichPopulation(1, device="cpu")
    neurons.v.fill_(29.0)
    neurons.u.zero_()
    spikes = neurons.step(torch.tensor([1000.0]), dt_ms=1.0)
    assert spikes.item()
    assert neurons.v.item() == neurons.c.item()
    assert neurons.u.item() > 0


def test_presets_have_distinct_firing_counts():
    types = torch.tensor([0, 1, 2])
    neurons = IzhikevichPopulation(3, device="cpu", neuron_types=types)
    counts = torch.zeros(3, dtype=torch.long)
    current = torch.full((3,), 10.0)
    for _ in range(500):
        counts += neurons.step(current).long()
    assert torch.all(counts > 0)
    assert len(set(counts.tolist())) > 1


def test_state_is_float32():
    neurons = IzhikevichPopulation(4, device="cpu")
    assert neurons.v.dtype == torch.float32
    assert neurons.u.dtype == torch.float32
