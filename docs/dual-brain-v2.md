# DrosoMath Dual-Brain V2

This document records the current architecture direction for the next DrosoMath experiment series.

## Goal

Start from a biologically inspired sparse recurrent learner, but do **not** pre-assign functions such as "math brain", "language brain", or "memory brain". Instead, give the system two interacting brains, anonymous submodules, fixed neural/synaptic budgets, reward-modulated plasticity, and sparse learnable bridges. Then measure whether specialization and memory retention emerge.

## Architecture

```text
                     symbolic / sensory input
                              |
                       learned router
                              |
             +----------------+----------------+
             |                                 |
        +----v-----+                      +----v-----+
        | Brain A  |                      | Brain B  |
        | A0 A1 A2 |<---- sparse -------->| B0 B1 B2 |
        | A3       |      bridge          | B3       |
        +----------+                      +----------+
             |                                 |
             +---------------+-----------------+
                             |
                           output
```

Each brain contains anonymous modules. A module's eventual role is inferred only after training from activation and routing telemetry; the learning rule does not receive task labels.

## Fixed resource budget

All recurrent synapse groups are stored as fixed-capacity edge lists:

- local module synapses,
- same-brain inter-module synapses,
- cross-brain bridge synapses.

Pruning/regrowth changes `src`, `dst`, and weight state **inside the existing array**. Training therefore does not continually allocate more synapses or silently grow memory use.

`DualBrainV2.resource_report()` reports the current fixed budgets and approximate parameter-state bytes.

## Synapse state

Every sparse synapse currently tracks:

- `src`
- `dst`
- `weight`
- `usage_ema`
- `reward_ema`
- `stability`

Repeated positively rewarded use raises `stability`. Stable edges decay more slowly and are harder to prune. This provides a first explicit long-term-memory consolidation mechanism rather than treating weight magnitude alone as memory importance.

## Fast and slow plasticity

Fast loop, every training trial:

1. route the input to a small subset of modules;
2. propagate sparse recurrent activity;
3. choose an action;
4. apply scalar reward;
5. update output, routing, input, and active synapse weights;
6. update usage/reward/stability telemetry.

Slow loop, every `rewire_interval` trials:

1. score low-stability edges by utility;
2. recycle a small low-utility fraction;
3. regrow legal edges inside the same fixed budget;
4. preserve some exploratory rewiring so the topology can discover new paths.

## Bridge design

The two brains are not densely connected. A random gateway subset of neurons is eligible for cross-brain connectivity, and the bridge has its own small fixed budget. Bridge edges can point in either direction and are plastic/re-wirable.

The first experimental sweep should compare multiple bridge budgets rather than assume a single optimum. Recommended relative conditions are:

- no bridge: ablation control;
- sparse bridge;
- balanced bridge;
- rich bridge.

The objective is to measure the trade-off between cooperation and spontaneous specialization.

## Self-specialization

A reward-modulated sparse router selects a small number of anonymous modules for each token. A homeostatic usage penalty discourages one module from monopolizing all inputs.

Task names, when supplied, are stored only in `task_activity_ema` for later analysis. They never enter routing, synaptic plasticity, or output learning. This lets us ask *after training* whether different task families caused different modules or brains to specialize.

## Memory benchmark

The first benchmark intentionally emphasizes continual learning rather than peak one-task accuracy.

```text
identity -> NEXT -> PREV
```

After each sequential training block, all tasks are re-evaluated without learning. The benchmark reports:

- accuracy after each phase;
- retention relative to each task's post-training peak;
- bridge usage/stability;
- module usage;
- task-conditioned activity telemetry;
- a sampled topology snapshot.

Run from `backend/`:

```powershell
python -m app.dual_brain_v2_benchmark --steps 3000 --eval-trials 300 --bridge 256
```

Ablation without structural rewiring:

```powershell
python -m app.dual_brain_v2_benchmark --steps 3000 --eval-trials 300 --bridge 256 --no-rewire
```

Run unit tests:

```powershell
python -m unittest discover -s tests -v
```

## Current scale

The default V2 core is deliberately a small architecture-validation model: 2 brains x 4 modules x 64 neurons. It is **not** yet the 139k-neuron FlyWire whole-connectome simulation.

This separation is intentional. We first need to verify that fixed-budget rewiring, bridge learning, consolidation, and self-specialization work and are measurable. Once those mechanisms are stable, the same sparse bank interfaces can be attached to progressively larger connectome-derived module populations.

## Next implementation stages

1. Run bridge-budget and rewiring ablations.
2. Improve consolidation until continual-learning retention is meaningfully better than the static baseline.
3. Add explicit sensory filtering / relevance gating.
4. Add checkpoint save/load so long-term-memory tests can span sessions.
5. Stream `topology_snapshot()` to the existing realtime frontend.
6. Visualize edge creation/removal, weight, stability, and module growth in real time.
7. Add neuron/synapse budget reallocation between modules without increasing the global budget.
8. Attach the V2 plasticity layer to connectome-derived FlyWire populations.
9. Add Korean-symbol and phoneme curricula only after retention and specialization controls pass.

## Scientific controls

Claims should compare against at least:

- bridge disabled;
- rewiring disabled;
- plasticity frozen;
- random reward;
- shuffled topology;
- equal-resource non-modular baseline.

The central V2 question is not whether a larger network can memorize more examples. It is whether **a fixed-resource, modular, continually rewiring system can learn new tasks while retaining old ones and spontaneously divide computation between its brains/modules.**
