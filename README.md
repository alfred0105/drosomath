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

## Scientific principle

The external experiment code may present stimuli, read choices, and deliver reward/punishment signals, but it should not directly compute the answer for the simulated brain. Claims of learning must be tested against frozen-plasticity, random-reward, and shuffled-connectome controls.

## Status

Bootstrap phase.
