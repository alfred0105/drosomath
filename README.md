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

## Experimental Dual-Brain V2

The `feature/dual-brain-v2` branch adds a new fixed-resource architecture experiment without replacing the current Stage 3S.3 baseline.

- two interacting brains;
- four anonymous modules per brain by default;
- no pre-assigned math/language/memory roles;
- reward-modulated sparse routing;
- fixed-capacity local, inter-module, and cross-brain synapse banks;
- usage/reward/stability state for each sparse synapse;
- periodic pruning and regrowth without increasing the global synapse budget;
- long-term-memory consolidation through stability-protected connections;
- topology snapshots designed for later realtime visualization;
- sequential retention benchmark to measure catastrophic forgetting.

The architecture and experiment plan are documented in `docs/dual-brain-v2.md`.

Run the first continual-memory benchmark from `backend/`:

```powershell
python -m app.dual_brain_v2_benchmark --steps 3000 --eval-trials 300 --bridge 256
```

Run V2 invariants/tests:

```powershell
python -m unittest discover -s tests -v
```

V2 currently uses a small architecture-validation network. It is deliberately not yet presented as a 139k-neuron whole-FlyWire simulation. The next step is to validate retention, spontaneous specialization, bridge-budget effects, and structural plasticity before scaling the same interfaces to connectome-derived populations.

## Current state

The viewer loads the public **FlyWire Codex FAFB v783 coordinates for 139,255 neurons** and renders them as a realtime point cloud. DrosoMath conservatively calls these Codex coordinates rather than assuming every exported position is a biological soma coordinate.

The old fixed-80% mock outcome generator has been removed. The backend now runs a reward-modulated 0/1/2-dot learning protocol.

### Stage 1 — acquisition

Stage 1 established that the prototype learner can acquire the 0/1/2 classification task from reward. Dot positions were randomized, but continuous cues such as total visual energy were allowed to covary with dot count. Stage 1 therefore validated acquisition, not abstract numerosity.

### Stage 2 — controlled continuous cues

Stage 2 is now the active experiment.

- Target classes remain `0`, `1`, and `2` dots (chance accuracy = 33.3%).
- Dot positions are randomized every trial.
- For **1-dot versus 2-dot stimuli**, total dot area is matched.
- Integrated signal-energy distributions for **1 versus 2** are matched independently of numerosity.
- Consequently, one-dot trials tend to use one larger/brighter dot while two-dot trials use two smaller/dimmer dots.
- Every 10th trial is a **probe trial** with plasticity completely disabled.
- Probe accuracy is reported separately from the combined training stream.
- Correct choice gives `+1`; incorrect choice gives `-1`.
- The target label is never inserted directly into the weight update. Learning uses sampled action + reward.

Zero remains the visual absence condition, so Stage 2's strongest control is the **1-vs-2 distinction**. Later controls can make the background and temporal presentation more demanding.

The 3D activity overlay is still a **display proxy mapped onto real FlyWire anatomical coordinates**, not a claim that those FlyWire neurons actually emitted those spikes. Whole-connectome neural dynamics remain a separate upcoming milestone.

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

Open `http://localhost:5173`. The Stage-2 HUD shows:

- randomized current dot stimulus
- model choice and correct target
- train/probe trial type
- `P(0)`, `P(1)`, `P(2)` choice probabilities
- overall and rolling 20/100/500-trial accuracy
- held-out probe overall / recent 20 / recent 100 accuracy
- per-target accuracy for 0/1/2 dots
- plastic-weight update telemetry (`probe` explicitly shows no update)

Five trials run for each 10 Hz UI frame, so the learner advances faster than the visualization refresh.

## Result logging

Each browser telemetry session creates a local result bundle under `runs/<run_id>/`:

```text
config.json
summary.json
metrics.csv
```

Stage-2 `metrics.csv` stores each trial with:

- train vs probe type
- target / choice / reward
- policy probabilities
- total signal energy and area-control values
- overall + rolling accuracy
- separate probe accuracy
- per-target accuracy
- plasticity statistics

`summary.json` includes the aggregate confusion matrices and probe metrics for GitHub-side analysis.

To publish only the newest run to the current Git branch:

```powershell
cd C:\Projects\drosomath
.\scripts\publish_latest_run.ps1
```

After that, the run can be inspected directly from GitHub.

## Scientific principle

The external experiment code may present stimuli, read choices, and deliver reward/punishment signals, but it should not directly calculate the answer for the learner. Claims of abstract numerical learning require controlled continuous cues, probe trials without learning, held-out stimulus distributions, frozen-plasticity and random-reward controls, and eventually shuffled-connectome controls.

## Next milestones

1. Run and analyze Stage-2 controlled-cue acquisition and probe performance.
2. Add stronger held-out spatial/appearance distributions if Stage 2 succeeds.
3. Load FAFB v783 connectivity and cell-type data needed for real KC/MBON/DAN circuits.
4. Attach the numerosity protocol to whole-connectome LIF dynamics.
5. Replace the display-proxy activity overlay with actual simulated spike/membrane telemetry.
6. Expand the curriculum to 0–4 and then 0–9.
7. Learn dot ↔ Arabic numeral associations before arithmetic.

## Status

Stage-2 0/1/2-dot controlled-cue reinforcement learning with frozen probe trials is integrated. Whole-connectome numerosity learning remains the next major scientific integration step. Dual-Brain V2 is being developed as a parallel experimental architecture on its feature branch so the existing Stage 3S.3 baseline remains available for direct controls.
