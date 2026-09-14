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
- telemetry snapshots
- GitHub Actions tests on Python 3.11 and 3.12

The current simulator is a correctness/reference backend for small-to-medium experiments. A large FlyWire-scale run will require a more compact sparse/tensor backend and profiling before claiming practical whole-brain performance.

## Install and run

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
drosomath-demo
drosomath-memory-demo
```

`drosomath-demo` exercises spike propagation, reward learning, and structural rewiring.

`drosomath-memory-demo` runs a small sequential-memory experiment and prints retention, forgetting, and learned synapse statistics as JSON.

## Architecture

- **Reference simulator:** Python event-driven LIF network
- **Plasticity:** reward learning + optional STDP
- **Slow structural loop:** pruning/regrowth under fixed budgets
- **Memory:** stability-based consolidation
- **Organization:** role-free modules and two independent brains
- **Communication:** sparse adaptive inter-brain bridge
- **Data input:** CSV edge lists suitable for processed FlyWire/Codex exports
- **Evaluation:** continual-memory retention/forgetting metrics
- **Planned live stack:** FastAPI/WebSocket + Vite/TypeScript/Three.js

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
14. live telemetry transport / 3D topology viewer — next
15. large-connectome sparse backend and FlyWire-scale performance work — next
16. real task curricula, controls, and quantitative experiments — next

## Scientific principle

External experiment code may present stimuli, read designated outputs, and deliver reward/punishment signals, but it should not solve the task for the simulated brain. Learning claims should be compared against frozen-plasticity, random-reward, shuffled-connectome, and simpler baseline networks.

The primary target metric is not only task accuracy. DrosoMath is intended to measure continual-learning retention, forgetting, adaptation speed, active-resource use, and performance per compute/memory budget.
