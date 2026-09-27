# Codex Task — Phase 01: M5/MPS bootstrap + Izhikevich core

Repository: alfred0105/drosomath
Working branch: artificial-brain-dev
Project root: artificial_brain/

## Goal
Build the first runnable, testable foundation for an "artificial developing brain" on a MacBook Air M5 with 24 GB unified memory.

This phase is NOT about advanced learning yet. The goal is to establish a correct, portable simulation core that we can validate before adding STDP, reward modulation, memory replay, homeostasis, and structural growth.

## Non-negotiable architecture

1. Use PyTorch.
2. Support devices in this order:
   - CUDA if available
   - MPS if available
   - CPU fallback
3. Default dtype must be torch.float32.
4. Do not require float64.
5. Do not rely on CUDA-only APIs.
6. Avoid MPS-fragile sparse kernels for the first version. Represent synapses as edge lists and use basic tensor indexing / index_add_ style operations that are expected to work on CPU, MPS, and CUDA.
7. Input and output neurons are fixed interface neurons.
8. Hidden neurons are a recurrent population and will become plastic in later phases.
9. Neuron computation must be vectorized. Do not create one Python object per neuron for the simulation hot path.
10. Keep all new code inside artificial_brain/ except where packaging absolutely requires otherwise.

## Required project structure

Create or complete:

artificial_brain/
  pyproject.toml
  README.md
  src/
    artificial_brain/
      __init__.py
      device.py
      neurons.py
      synapses.py
      brain.py
      checkpoint.py
  tests/
    test_device.py
    test_izhikevich.py
    test_synapses.py
    test_brain_step.py
    test_checkpoint.py
  experiments/
    smoke_izhikevich.py
    benchmark_m5.py
  configs/
    macbook_m5.yaml
  checkpoints/
    .gitkeep

## Device layer

Implement a single device-selection utility.

Expected behavior:
- explicit device override supported: cpu, mps, cuda
- otherwise auto-select cuda -> mps -> cpu
- helpful error if requested device is unavailable
- expose selected torch.device and dtype
- float32 by default

No device-specific branching should leak throughout the rest of the code unless unavoidable.

## Izhikevich neuron population

Implement a vectorized population using tensors.

State per neuron:
- v
- u
- a
- b
- c
- d
- spike
- neuron_type
- alive

Use the standard Izhikevich dynamics:

dv/dt = 0.04*v^2 + 5*v + 140 - u + I
du/dt = a*(b*v - u)

When v >= 30 mV:
- mark spike
- v = c
- u = u + d

Use stable discrete stepping for dt in milliseconds. The implementation must make dt explicit.

Provide convenient constructors / parameter presets for at least:
- regular spiking
- fast spiking
- intrinsically bursting

Do not hard-code biological labels into the core update equation.

## Synapse representation

Do NOT use an N x N dense weight matrix.

Store only existing edges:
- pre: integer tensor [E]
- post: integer tensor [E]
- weight: float32 tensor [E]
- delay_steps: integer tensor [E] or a simple zero-delay initial implementation with a clean extension point

For phase 01, support excitatory and inhibitory weights by sign.

Implement propagation from presynaptic spikes to postsynaptic input current.

Prefer simple gather/indexing + index_add_ accumulation.

## Brain class

Create a Brain abstraction that owns:
- neuron population
- synapse edge list
- fixed input neuron indices
- hidden neuron indices
- fixed output neuron indices

Required methods:
- step(input_current)
- reset_state()
- get_output_spikes()
- state_dict()
- load_state_dict()

Input/output neurons should be fixed in membership, but their synaptic connectivity must not be conceptually frozen.

Initial default test config:
- input neurons: 8
- hidden neurons: 256
- output neurons: 2
- initial synapses: about 5000
- dt: 1.0 ms

Use a seeded RNG for reproducibility.

## Checkpointing

Implement checkpoint save/load using torch.save / torch.load.

Persist at minimum:
- neuron dynamic state
- neuron parameters
- synapse arrays
- input/hidden/output index assignments
- simulation step
- random seed / RNG state where practical
- config metadata

Checkpoint loading must support map_location so an MPS checkpoint can be loaded on CPU and vice versa.

## Config

Create configs/macbook_m5.yaml with conservative defaults for a MacBook Air M5 24 GB.

Include:
- device: auto
- dtype: float32
- dt_ms: 1.0
- input_neurons: 8
- hidden_neurons: 256
- output_neurons: 2
- initial_synapses: 5000
- seed
- benchmark durations / warmup

Do not assume the entire 24 GB is available to PyTorch.

## Experiments

### smoke_izhikevich.py
Demonstrate:
- device selected
- one or more known Izhikevich firing patterns
- spike count
- no NaNs / infs
- short run completes on CPU and on MPS when available

### benchmark_m5.py
Benchmark several sizes, e.g.:
- 256 hidden / 5k synapses
- 1k hidden / 50k synapses
- 10k hidden / 500k synapses

Measure:
- device
- neurons
- synapses
- simulation steps
- wall time
- steps/sec
- approximate memory statistics when available

If a scale fails due to memory or unsupported MPS behavior, report it cleanly rather than crashing the whole benchmark.

Do not over-optimize before measuring.

## Tests

At minimum verify:
1. device selection works
2. tensor dtype is float32
3. Izhikevich neuron spikes and resets correctly
4. presets produce distinct firing behavior under a controlled current
5. synaptic current reaches the correct postsynaptic indices
6. inhibitory weights subtract current
7. one Brain step has correct tensor shapes
8. fixed input/output index sets remain stable
9. checkpoint round trip preserves the model state
10. CPU tests pass without an Apple GPU

Use pytest.

## README updates

Document:
- what phase 01 implements
- setup on macOS
- Python version recommendation: 3.12
- creating .venv
- pip install commands
- how to run tests
- how to run smoke experiment
- how to run benchmark
- how MPS is selected
- limitations of phase 01

Keep the documentation concise and executable.

## Quality constraints

- Prefer simple, inspectable code.
- Type hints where useful.
- No unnecessary framework abstractions.
- Avoid per-neuron Python loops in the hot path.
- Avoid repeated tensor reallocation inside every timestep.
- Do not implement STDP, reward learning, replay, neuron birth/death, or dendritic compartments yet except for clean extension points.
- No fake benchmark numbers.
- Do not claim MPS was tested unless actually executed on MPS.

## Completion checklist

Before finishing:
1. Run pytest.
2. Run the CPU smoke test.
3. If running on Apple Silicon with MPS available, run the MPS smoke test.
4. Run at least the smallest benchmark.
5. Inspect git diff.
6. Commit all relevant files.
7. Push to origin branch: artificial-brain-dev.

Use a clear commit message such as:
feat(artificial-brain): bootstrap MPS Izhikevich simulation core

## Final response to the user

Report:
- commit SHA
- branch pushed
- files added/changed
- pytest result
- smoke-test result
- benchmark result
- whether MPS was actually available and tested
- any known limitations or next-step blockers

Do not merely say "done"; provide verifiable results.
