# DrosoMath

DrosoMath is an experimental platform for studying whether a Drosophila connectome model with biologically inspired plasticity can learn numerical concepts and, eventually, more abstract mathematical operations.

## Initial goals

1. Run a Drosophila connectome simulation backend.
2. Stream neural activity and plasticity telemetry in real time.
3. Visualize brain activity in an interactive 3D viewer.
4. Start with numerosity discrimination before arithmetic.
5. Evaluate generalization with held-out problems and control conditions.

## Architecture

- **Simulator:** Python / PyTorch (planned next)
- **Backend:** FastAPI + WebSocket
- **Frontend:** Vite + TypeScript + Three.js
- **Data:** FlyWire / Codex-derived connectome data (not committed to the repository)

## Current bootstrap

The first scaffold uses **mock neural telemetry** so the realtime pipeline can be validated before integrating FlyWire data. It renders an interactive 3D point-cloud brain and streams changing activity at 10 Hz over WebSocket.

### Run the backend

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Linux / WSL:

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Run the frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the Vite URL (normally `http://localhost:5173`). Drag to rotate the brain, scroll to zoom, and watch neural activity change in real time.

## Scientific principle

The external experiment code may present stimuli, read choices, and deliver reward/punishment signals, but it should not directly compute the answer for the simulated brain. Claims of learning must be tested against frozen-plasticity, random-reward, and shuffled-connectome controls.

## Next milestones

- Replace the mock layout with FlyWire/Codex neuron coordinates.
- Add region/neuronal-class filtering and click-to-inspect.
- Add a simulator adapter for the FlyBrain connectome model.
- Stream firing, membrane-potential and plasticity telemetry separately.
- Implement the first numerosity-discrimination experiment.
- Add held-out generalization tests and control conditions.

## Status

Realtime 3D bootstrap in progress.
