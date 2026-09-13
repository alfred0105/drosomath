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
- **Data:** FlyWire / Codex FAFB v783 (kept local; not committed)

## Current state

The 3D viewer can now use **real FlyWire FAFB v783 soma coordinates** while neural activity is still mock telemetry. This intentionally separates anatomical-data integration from simulation integration so the UI pipeline can be verified before the whole-connectome LIF/plasticity engine is attached.

FAFB v783 contains 139,255 neurons, but `coordinates.csv.gz` contains soma positions for only the subset with available soma coordinates (roughly 23k). DrosoMath labels this honestly in the UI rather than inventing positions for the remaining neurons. A later milestone will add representative coordinates for non-soma cells from skeleton/synapse geometry.

The current mock outcome generator marks exactly 4 of every 5 trials correct, so its overall and sufficiently long rolling success rates converge to **80% by construction**. This is only a UI/metrics pipeline test and must not be interpreted as learning.

### Get the real FAFB v783 soma layout

From the repository root:

```powershell
python scripts/download_fafb783.py
```

The helper downloads the two small FAFB v783 static exports used by the viewer from FlyWire/Codex public storage and places them under `data/flywire/fafb783/`:

- `classification.csv.gz`
- `coordinates.csv.gz`

The `data/` directory is ignored by Git. If the helper URL is unavailable, download the same FAFB v783 files manually from the Codex Download Data page and place them in that directory. Follow the FlyWire/Codex data terms for any redistribution or publication; DrosoMath does not commit these data files.

You can use a different local dataset directory with:

```powershell
$env:DROSOMATH_FLYWIRE_DIR = "D:\path\to\fafb783"
```

Restart the backend after adding or changing the data files. `/api/layout` automatically uses real FlyWire soma coordinates when both files are present and otherwise falls back to mock geometry.

## Run the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Linux / WSL:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run the frontend

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open the Vite URL (normally `http://localhost:5173`). Drag to rotate the brain and scroll to zoom. With FAFB data installed the status line reports the real soma count and data source; neural flashes are still clearly marked as mock activity until the simulator adapter lands.

## Result logging

Each browser telemetry session creates a local result bundle under `runs/<run_id>/`:

```text
config.json
summary.json
metrics.csv
```

`summary.json` contains overall accuracy plus rolling 20/100/500-trial success rates. `metrics.csv` keeps the per-trial outcome history without storing high-volume per-neuron activity.

To publish only the newest run to the current Git branch so it can be reviewed from GitHub:

```powershell
cd C:\Projects\drosomath
.\scripts\publish_latest_run.ps1
```

## Scientific principle

The external experiment code may present stimuli, read choices, and deliver reward/punishment signals, but it should not directly compute the answer for the simulated brain. Claims of learning must be tested against frozen-plasticity, random-reward, and shuffled-connectome controls.

## Next milestones

- Add click-to-inspect for real FlyWire root IDs and cell classifications.
- Add representative positions for neurons without a soma coordinate.
- Load the FAFB v783 connection table and whole-connectome LIF engine.
- Replace mock telemetry with real spike and membrane-potential telemetry.
- Add KC/MBON/DAN plasticity telemetry.
- Implement the first numerosity-discrimination experiment.
- Add held-out generalization tests and control conditions.

## Status

Real FAFB soma geometry and persistent run logging integrated; connectome dynamics are the next step.
