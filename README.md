# DrosoMath

DrosoMath is an experimental platform for studying whether a Drosophila connectome model with biologically inspired plasticity can learn numerical concepts and, eventually, more abstract mathematical operations.

## Initial goals

1. Run a Drosophila connectome simulation backend.
2. Stream neural activity and plasticity telemetry in real time.
3. Visualize brain activity in an interactive 3D viewer.
4. Start with numerosity discrimination before arithmetic.
5. Evaluate generalization with held-out problems and control conditions.

## Architecture

- **Simulator:** Python / NumPy prototype now; whole-connectome PyTorch/LIF is the next integration stage
- **Backend:** FastAPI + WebSocket
- **Frontend:** Vite + TypeScript + Three.js
- **Data:** FlyWire / Codex FAFB v783 (kept local; not committed)

## Current state

The viewer loads the public **FlyWire Codex FAFB v783 coordinates for 139,255 neurons** and renders them as a realtime point cloud. DrosoMath conservatively calls these Codex coordinates rather than assuming every exported position is a biological soma coordinate.

The old fixed-80% mock outcome generator has been removed. The live backend now runs the first real learning protocol:

### Stage 1 — dots 0–2

- Target classes: `0`, `1`, `2` dots (chance accuracy = 33.3%).
- Dot positions are randomized every trial.
- The learner chooses among actions `0`, `1`, and `2` probabilistically.
- Correct choice gives `+1`; incorrect choice gives `-1`.
- The target label is **not inserted directly into the weight update**. Learning uses chosen action + reward.
- A fixed sparse expansion layer feeds plastic choice weights via a reward-modulated policy/eligibility update.
- Stage 1 intentionally allows continuous visual cues such as total visual energy to covary with dot count. This validates acquisition first.
- Stage 2 will equalize brightness/area cues and test whether the learned behavior generalizes to numerosity itself.

The 3D activity overlay during Stage 1 is a **display proxy mapped onto real FlyWire anatomical coordinates**, not a claim that those FlyWire neurons actually emitted those spikes. Whole-connectome neural dynamics remain a separate upcoming milestone.

## Get the FAFB v783 coordinates

From the repository root:

```powershell
python scripts/download_fafb783.py
```

The helper downloads the two FAFB v783 static exports currently used by the viewer into `data/flywire/fafb783/`:

- `classification.csv.gz`
- `coordinates.csv.gz`

The `data/` directory is ignored by Git. Follow the FlyWire/Codex data terms for redistribution or publication.

You can use a different local dataset directory with:

```powershell
$env:DROSOMATH_FLYWIRE_DIR = "D:\path\to\fafb783"
```

## Run the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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

Open `http://localhost:5173`. The HUD shows:

- randomized current dot stimulus
- model choice and correct target
- `P(0)`, `P(1)`, `P(2)` choice probabilities
- overall success rate
- rolling 20/100/500-trial success rates
- per-target accuracy for 0/1/2 dots
- plastic-weight update telemetry

Five training trials run for each 10 Hz UI frame, so the UI remains readable while the learner advances faster than the visualization refresh.

## Result logging

Each browser telemetry session creates a local result bundle under `runs/<run_id>/`:

```text
config.json
summary.json
metrics.csv
```

`summary.json` contains overall and rolling accuracy, accuracy by target, and a 3×3 confusion matrix (rows = target, columns = chosen answer). `metrics.csv` stores each training trial, including target, choice, reward, policy probabilities, and plasticity statistics.

To publish only the newest run to the current Git branch:

```powershell
cd C:\Projects\drosomath
.\scripts\publish_latest_run.ps1
```

After that, the run can be inspected directly from GitHub.

## Scientific principle

The external experiment code may present stimuli, read choices, and deliver reward/punishment signals, but it should not directly calculate the answer for the learner. Claims of abstract numerical learning require stronger controls than Stage 1, including equalized continuous cues, held-out stimulus distributions, frozen plasticity, random reward, and eventually shuffled-connectome controls.

## Next milestones

1. Run and analyze Stage-1 0/1/2-dot acquisition.
2. Add Stage-2 controlled-cue numerosity generalization.
3. Load FAFB v783 connectivity and cell-type data needed for real KC/MBON/DAN circuits.
4. Attach the numerosity protocol to whole-connectome LIF dynamics.
5. Replace the display-proxy activity overlay with actual simulated spike/membrane telemetry.
6. Expand the curriculum to 0–4 and then 0–9.
7. Learn dot ↔ Arabic numeral associations before arithmetic.

## Status

Live 0–2 dot reinforcement-learning protocol integrated with persistent run logging and realtime 3D visualization. Whole-connectome numerosity learning is the next scientific integration step.
