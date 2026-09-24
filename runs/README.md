# DrosoMath runs

Each local experiment/browser telemetry session writes a small result bundle here:

```text
runs/<run_id>/
  config.json
  summary.json
  metrics.csv
```

- `config.json`: experiment/source/configuration metadata.
- `summary.json`: final overall and rolling success metrics.
- `metrics.csv`: per-trial outcome history and lightweight plasticity metrics.

Per-neuron activity is intentionally not committed here because it would grow too quickly. Raw/high-volume neural traces should live in ignored local storage or external artifacts.

## Share a run for review

From the repository root, run:

```powershell
.\scripts\publish_latest_run.ps1
```

This commits and pushes only the most recent run directory on the current Git branch. Review the generated files before publishing if the experiment contains anything you do not want in Git history.
