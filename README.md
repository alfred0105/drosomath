# DrosoMath

DrosoMath is an experimental platform for studying whether a Drosophila-connectome-inspired spiking system with reward learning, structural plasticity, continual memory, and emergent modular specialization can learn numerical concepts and later more abstract tasks.

## Current executable v0

The current branch contains a runnable reference simulator rather than only isolated plasticity primitives.

Implemented:

- leaky integrate-and-fire neuron/spike simulation
- delayed reward credit and bounded reward-modulated weight learning
- pair-based STDP with bounded weights
- fixed-budget synapse pruning/regrowth
- activity/reward-biased regrowth candidate generation
- memory consolidation through synaptic stability
- role-free internal modules with specialization metrics
- module-level fixed synapse-budget reallocation
- two independent brains connected by a sparse adaptive bridge
- fixed-budget bridge pruning/regrowth
- activity/reward-biased bridge candidate generation
- CSV connectome edge-list loading with subset/max-edge controls
- continual-learning retention/forgetting evaluation
- frozen-plasticity evaluation mode
- realtime local brain/learning dashboard
- GitHub Actions tests on Python 3.11 and 3.12

The current simulator is a correctness/reference backend for small-to-medium experiments. A large FlyWire-scale run will require a more compact sparse/tensor backend and profiling before claiming practical whole-brain performance.


## Connectome-backed learning

This branch adds a first learning path that uses the real annotated connectome rather than a symbolic substitute:

1. load the MaleCNS/FlyWire anatomical graph and neuron annotations;
2. identify Kenyon cells, MBONs, and dopamine-neuron annotations;
3. simulate LIF spikes through the anatomical graph;
4. restrict reward-modulated plasticity to real anatomical KC -> MBON edges;
5. report DAN activity while accepting the scalar reward only at trial end.

Non-KC -> MBON anatomical edges remain part of the fixed brain dynamics, but they are not modified by this learning rule. No target label, answer, or trainable external decoder is inserted into the neural state.

Example:

```python
from drosomath.malecns import MushroomBodyCircuit, PlasticMaleCNSBrain, load_malecns_v1

connectome = load_malecns_v1("data/malecns_v1", min_connection_synapses=5)
brain = PlasticMaleCNSBrain(connectome, seed=7)
circuit = MushroomBodyCircuit.from_connectome(connectome)
circuit.attach(brain)

report = circuit.run_trial(stimulus_body_ids=(123, 456), reward=1.0)
print(report["spikes"], report["learning"]["learning"])
```

This is a connectome-backed first learning path, not a claim that every biological mushroom-body detail or synaptic sign has been recovered. The loader preserves source annotations and the simulator keeps its assumptions auditable.


## Fast local workflow

Install once:

```powershell
python -m pip install -e .
```

Run the latest benchmark, save its JSON result, and push only that result so it can be inspected remotely:

```powershell
.\bench.ps1
```

Open the realtime dashboard with one command:

```powershell
.\live.ps1
```

`live.ps1` starts a local-only server, opens `http://127.0.0.1:8765/`, and streams simulation state without requiring a separate frontend build or external web service.

`run-mushroom-body.ps1` saves `results/latest_mushroom_body_trial.json` and pushes that result to the current Git branch by default, so later analysis can read it directly from GitHub. Use `-NoPush` for a local-only run.


The live dashboard shows:

- current firing neurons and membrane potential
- synaptic weight, stability, usage count, and reward EMA
- current task, expected output, latest prediction, and reward
- running accuracy, mean weight, mean stability, and STDP update count
- recent learning history
- structural rewiring events
- pause/resume, single-step, reset, and speed controls

The live demo is intentionally small and fast. It is an observability path for the reference simulator, not yet the FlyWire-scale 3D renderer.

## Other commands

```powershell
python -m unittest discover -s tests -v
drosomath-demo
drosomath-memory-demo
drosomath-ablation-demo
drosomath-live
```

`drosomath-demo` exercises spike propagation, reward learning, and structural rewiring.

`drosomath-memory-demo` runs a small sequential-memory experiment and prints retention, forgetting, and learned synapse statistics as JSON.

`drosomath-ablation-demo` compares learning variants under the same task sequence.

## Architecture

- **Reference simulator:** Python event-driven LIF network
- **Plasticity:** reward learning + optional STDP
- **Slow structural loop:** pruning/regrowth under fixed budgets
- **Memory:** stability-based consolidation
- **Organization:** role-free modules and two independent brains
- **Communication:** sparse adaptive inter-brain bridge
- **Data input:** CSV edge lists suitable for processed FlyWire/Codex exports
- **Evaluation:** continual-memory retention/forgetting metrics
- **Live transport:** local HTTP + Server-Sent Events using only Python's standard library
- **Live UI:** zero-build Canvas dashboard
- **Planned large visualization:** 3D topology/activity viewer for larger connectomes

## Development phases

1. compact mutable synapse state — done
2. event-driven usage tracking and delayed reward credit — done
3. bounded reward-modulated weight learning — done
4. fixed-budget pruning/regrowth — done
5. activity/reward-biased internal regrowth — done
6. executable neuron/spike simulator — done
7. connectome loader — done
8. independent Brain A / Brain B containers — done
9. sparse adaptive bridge — done
10. role-free modules and specialization metrics — done
11. memory consolidation and module synapse-budget reallocation — done
12. bridge structural plasticity and bridge candidate generation — done
13. STDP and frozen continual-memory evaluation — done
14. realtime local telemetry/dashboard — done for reference simulator; 3D large-graph view remains
15. large-connectome sparse backend and FlyWire-scale performance work — next
16. interference-heavy task curricula, controls, and quantitative experiments — next

## Scientific principle

External experiment code may present stimuli, read designated outputs, and deliver reward/punishment signals, but it should not solve the task for the simulated brain. Learning claims should be compared against frozen-plasticity, random-reward, shuffled-connectome, and simpler baseline networks.

The primary target metric is not only task accuracy. DrosoMath is intended to measure continual-learning retention, forgetting, adaptation speed, active-resource use, and performance per compute/memory budget.
