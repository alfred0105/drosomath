import torch

from artificial_brain.brain import Brain


def test_step_shapes_and_fixed_interfaces():
    brain = Brain(input_neurons=3, hidden_neurons=5, output_neurons=2,
                  initial_synapses=10, device="cpu", seed=7)
    before = (brain.input_indices.clone(), brain.hidden_indices.clone(), brain.output_indices.clone())
    output = brain.step(torch.ones(3))
    assert output.shape == (2,)
    assert output.dtype == torch.bool
    assert brain.neurons.v.shape == (10,)
    assert all(torch.equal(old, new) for old, new in zip(
        before, (brain.input_indices, brain.hidden_indices, brain.output_indices)))
    assert brain.step_count == 1


def test_brain_reproducible_initial_edges():
    first = Brain(hidden_neurons=4, initial_synapses=12, seed=42, device="cpu")
    second = Brain(hidden_neurons=4, initial_synapses=12, seed=42, device="cpu")
    assert torch.equal(first.synapses.pre, second.synapses.pre)
    assert torch.equal(first.synapses.post, second.synapses.post)
