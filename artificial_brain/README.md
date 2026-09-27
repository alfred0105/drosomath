# Artificial Developing Brain — Phase 01

Phase 01 provides a portable PyTorch Izhikevich simulation core: a vectorized neuron population, sparse edge-list synapses, fixed input/output interfaces, state checkpoints, and CPU/MPS/CUDA device selection. Learning rules and structural plasticity are future work.

## Setup (macOS)

Python 3.12 is recommended. From this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The installed PyTorch build determines accelerator availability. The selector uses CUDA first, then MPS, then CPU; pass `device="cpu"`, `"mps"`, or `"cuda"` to override it. MPS is selected only when PyTorch reports it available.

## Run

```sh
pytest
python experiments/smoke_izhikevich.py
python experiments/benchmark_m5.py --quick
```

Run `python experiments/benchmark_m5.py` for all configured sizes. Results include measured device, wall time, throughput, and memory statistics where PyTorch exposes them.

## Phase 01 limits

The network uses zero-delay propagation (with a delay tensor reserved for extension), fixed membership for interface populations, and signed static weights. It does not implement STDP, reward learning, replay, homeostasis, neuron growth/death, or dendritic compartments. MPS support depends on the local PyTorch/macOS installation and must be verified by executing the MPS smoke test; device selection alone is not evidence of execution.
