# agi-core-codex

`agi-core-codex` is a clean-slate rebuild of the AGI prototype around two constraints:

1. The shared core must stay genuinely domain-generic.
2. Benchmark-specific machinery must live in explicit strategy plugins with measured attribution.

This repository starts with ARC as the first implemented domain and keeps room for Zork and later robotics without polluting the common core with ARC-shaped hooks.

## What exists today

- A small generic kernel built around `Environment`, `Grammar`, `Scorer`, `Memory`, and `Strategy`.
- A transfer-oriented hypothesis pipeline built around `TaskRepresentation`, `Hypothesis`,
  `HypothesisFamily`, `Verifier`, and compiled executable programs with stable IDs.
- Deterministic budgeted search with explicit failure handling.
- Immutable run manifests and an artifact index for reproducible experiments.
- ARC support implemented as a plugin layer with named strategies:
  - grammar primitives, including crop-to-content, gravity, targeted color swaps, and foreground recoloring
  - boolean halves cross-reference
  - directional ray extension and masked span fill
  - row/column decomposition via explicit row and column sorting rules
  - separator-grid row/column propagation
  - separator-based cross-reference and cell reductions
  - scale/tile/downscale ratio detection
  - template stamping
  - anchor-based motif completion
  - triomino corner completion inside 2x2 windows
  - collinear gap bridging with inferred fill markers
  - hole projection from interior zero markers along the short axis
  - solid rectangle extraction from noisy connected components
  - rectangular ring recoloring with task-inferred target colors
  - scaffold-driven column projection from 8-cell anchor components
  - zero-pattern propagation from a learned displacement vector
  - constant-output synthesis
  - consistent color-map synthesis
  - task-scoped absolute patch synthesis
- Four CLI profiles:
  - `baseline-core`
  - `arc-accuracy`
  - `arc-theory`
  - `arc-transfer`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m agi_core_codex arc-data discover --benchmark arc-agi-1 --split training
python -m agi_core_codex arc-data make-splits \
  --benchmark arc-agi-1 \
  --output-dir experiments/splits \
  --train-val-count 80 \
  --seed 7
python -m agi_core_codex arc-accuracy tune \
  --split-file experiments/splits/arc_agi_1_train_dev.json
python -m agi_core_codex arc-transfer tune \
  --split-file experiments/splits/arc_agi_1_train_val.json
```

## Current ARC-AGI-1 snapshot

Latest accepted checkpoint on `main`: March 18, 2026.

- Frozen `arc-accuracy` baseline:
  - `train-dev`: `42/320` exact on train examples, `47/320` exact on test examples
  - `train-val`: `36/80` exact on train examples, `36/80` exact on test examples
  - `public-eval`: `17/400` exact on test examples, which is `4.25%`
- First `arc-transfer` checkpoint:
  - `train-dev`: `15/320` exact on train examples
  - `train-val`: `4/80` exact on train examples

The baseline and transfer numbers are intentionally reported side by side. The transfer
track is a family-based redesign, not a stronger benchmark result yet. Public eval
remains checkpoint-only reporting, not a tuning signal.

## Design choices

- The core has no ARC-only escape hatches.
- Dynamic operators get stable semantic IDs.
- Failure is treated as failure, never silently mapped to identity.
- ARC recovery heuristics live as explicit strategies that can be ablated from the CLI.
- The transfer track uses five broad ARC hypothesis families instead of the one-task
  strategy zoo: global transforms, object transforms, relation propagation, template
  completion, and region routing.
- Public evaluation should be checkpoint-only; tuning happens on train-derived splits.

## Layout

- `src/agi_core_codex/core/`: generic kernel, manifests, memory, reusable strategies
- `src/agi_core_codex/domains/arc/`: ARC domain implementation and ARC-only strategies
- `experiments/splits/`: split policies and example split files
- `artifacts/`: immutable run outputs
- `tests/`: unit and regression tests

## Real-data workflow

Use `arc-data make-splits` against the ARC training directory to create deterministic
`train-dev` and `train-val` files. If the dataset is present in a common location,
the command auto-discovers it. In this workspace it can discover the sibling
read-only dataset under `~/github/agi-core/data/ARC-AGI/data/training`.

Once a split file has `benchmark` and `source_dataset_dir` metadata, `arc-accuracy`,
`arc-theory`, and `arc-transfer` can auto-resolve the dataset path, so `--dataset-dir`
becomes optional.
