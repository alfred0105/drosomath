import torch

from artificial_brain.brain import Brain
from artificial_brain.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    original = Brain(input_neurons=2, hidden_neurons=4, output_neurons=1,
                     initial_synapses=7, seed=5, device="cpu")
    original.step(torch.tensor([2.0, 3.0]))
    path = tmp_path / "brain.pt"
    save_checkpoint(original, path)
    restored = Brain(input_neurons=2, hidden_neurons=4, output_neurons=1,
                     initial_synapses=7, seed=5, device="cpu")
    load_checkpoint(restored, path, map_location="cpu")
    assert restored.step_count == original.step_count
    assert torch.equal(restored.neurons.v, original.neurons.v)
    assert torch.equal(restored.synapses.weight, original.synapses.weight)
    assert torch.equal(restored.output_indices, original.output_indices)
    assert restored.config == original.config
    assert torch.equal(restored.generator.get_state(), original.generator.get_state())
