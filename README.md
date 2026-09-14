# DrosoMath

DrosoMath is an experimental platform for studying whether a Drosophila connectome model with biologically inspired plasticity can learn numerical concepts and, eventually, more abstract mathematical operations.

## Initial goals

1. Run a Drosophila connectome simulation backend.
2. Stream neural activity and plasticity telemetry in real time.
3. Visualize brain activity in an interactive 3D viewer.
4. Start with numerosity discrimination before arithmetic.
5. Evaluate generalization with held-out problems and control conditions.

## Planned architecture

- **Simulator:** Python / PyTorch
- **Backend:** FastAPI + WebSocket
- **Frontend:** Vite + TypeScript + Three.js
- **Data:** FlyWire / Codex-derived connectome data (not committed to the repository)

## Plasticity roadmap

Development is intentionally incremental so each mechanism can be tested before structural complexity is added.

- Phase 1: compact mutable synapse state
- Phase 2: event-driven usage tracking and delayed reward credit
- Phase 3: bounded reward-modulated weight strengthening/weakening
- Phase 4: fixed-budget pruning and synapse regrowth
- Phase 5: activity/reward-biased regrowth candidate generation
- Next: dual-brain containers, sparse adaptive bridges, self-specializing modules, budget reallocation, and live topology telemetry

Phase 4 keeps the registered synapse count constant during a rewiring cycle. Stale, low-reward connections can be removed only when replacement candidates are available, while high-stability connections are protected as a first approximation of consolidated memory.

Phase 5 biases replacement candidates toward neurons participating in recent, frequently used, positively rewarded paths. Candidate selection uses small top-k source/target pools instead of constructing a full neuron-by-neuron matrix, keeping the slow structural-plasticity loop scalable for large connectomes.

## Scientific principle

The external experiment code may present stimuli, read choices, and deliver reward/punishment signals, but it should not directly compute the answer for the simulated brain. Claims of learning must be tested against frozen-plasticity, random-reward, and shuffled-connectome controls.

## Status

Bootstrap phase. Core weight and structural plasticity primitives are being added and tested before the full simulator loop is introduced.
