# Artificial Developing Brain

Experimental project for a self-developing spiking neural network.

## Initial direction

- Neuron model: Izhikevich
- Fixed input/output neuron interfaces
- Plastic hidden recurrent network
- Sparse synapses
- STDP and reward-modulated learning
- Homeostasis
- Structural plasticity: synapse/neuron growth and pruning
- Optional simplified dendritic compartments
- PyTorch/CUDA-oriented implementation
- Reproducible checkpoints and tests

## Planned layout

- `src/` — implementation
- `tests/` — unit/integration tests
- `experiments/` — reproducible experiments
- `configs/` — experiment/model configuration
- `docs/` — design notes
