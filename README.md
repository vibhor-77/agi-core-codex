# agi-core-codex

`agi-core-codex` is a clean-slate rebuild of the AGI prototype around two constraints:

1. The shared core must stay genuinely domain-generic.
2. Benchmark-specific machinery must live in explicit strategy plugins with measured attribution.

This repository starts with ARC as the first implemented domain and keeps room for Zork and later robotics without polluting the common core with ARC-shaped hooks.

## What exists today

- A small generic kernel built around `Environment`, `Grammar`, `Scorer`, `Memory`, and `Strategy`.
- Deterministic budgeted search with explicit failure handling.
- Immutable run manifests and an artifact index for reproducible experiments.
- ARC support implemented as a plugin layer with named strategies:
  - grammar primitives
  - boolean halves cross-reference
  - separator-based cross-reference
  - constant-output synthesis
  - consistent color-map synthesis
  - task-scoped absolute patch synthesis
- Three CLI profiles:
  - `baseline-core`
  - `arc-accuracy`
  - `arc-theory`

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
  --split-file experiments/splits/arc_train_dev.json
```

## Design choices

- The core has no ARC-only escape hatches.
- Dynamic operators get stable semantic IDs.
- Failure is treated as failure, never silently mapped to identity.
- ARC recovery heuristics live as explicit strategies that can be ablated from the CLI.
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

Once a split file has `benchmark` and `source_dataset_dir` metadata, `arc-accuracy`
and `arc-theory` can auto-resolve the dataset path, so `--dataset-dir` becomes optional.
