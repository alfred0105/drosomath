import torch

from artificial_brain.synapses import SynapseEdges


def test_propagation_and_inhibition():
    edges = SynapseEdges(torch.tensor([0, 0, 1]), torch.tensor([2, 3, 2]),
                         torch.tensor([2.0, -3.0, -1.0]), 4, device="cpu")
    current = edges.propagate(torch.tensor([True, True, False, False]))
    assert torch.equal(current, torch.tensor([0.0, 0.0, 1.0, -3.0]))


def test_empty_edges():
    edges = SynapseEdges(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                         torch.empty(0), 2, device="cpu")
    assert torch.equal(edges.propagate(torch.zeros(2, dtype=torch.bool)), torch.zeros(2))
