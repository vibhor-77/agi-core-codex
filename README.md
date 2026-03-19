# agi-core-codex

`agi-core-codex` now defaults to a minimal 4-pillars learner.

The active path is intentionally small:
- tiny seeded primitives
- generic AST composition
- staged wake/sleep memory
- round-based compounding
- synthetic curriculum before serious ARC work

The old solver stack still exists for comparison, but it is now explicitly archived behind `legacy` commands instead of being the default mental model of the repo.

## Active Idea

The active learner is trying to validate a stricter thesis than "build a stronger ARC solver":

1. Start from near-zero generic seeds.
2. Search for small programs that solve tasks.
3. Promote reusable subprograms into a committed library only after a sleep step.
4. Reuse that committed library in later rounds so capability grows by compounding, not by hand-authored tactics.

The active seeded grid vocabulary for v1 is deliberately tiny:
- unary seeds: `identity`, `flip_h`, `flip_v`, `transpose`, `crop_support`
- binary compositors: `chain`, `overlay`, `hcat`, `vcat`

No separator, motif, barrier, template, row/column, or object-specific ARC tactics are part of the default learner path.

## Active Surface

The default public learner surface is:
- [PrimitiveSpec](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/ops.py)
- [Program](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/core.py)
- [LearnerMemory](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/core.py)
- [WakeSleepLearner](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/core.py)
- [RoundSummary](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/core.py)

The main active runner is [run_minimal](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/runner.py).

## Default CLI

Use the minimal learner by default:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest

python -m agi_core_codex minimal tune \
  --domain synthetic-grid \
  --curriculum-tier pair \
  --rounds 2

python -m agi_core_codex minimal tune \
  --domain arc \
  --split-file experiments/splits/arc_minimal_smoke.json \
  --rounds 2
```

Archived solver commands are still available, but only through `legacy`:

```bash
python -m agi_core_codex legacy arc-accuracy tune --split-file experiments/splits/arc_agi_1_train_dev.json
python -m agi_core_codex legacy arc-transfer tune --split-file experiments/splits/arc_agi_1_train_val.json
python -m agi_core_codex legacy arc-bootstrap tune --split-file experiments/splits/arc_agi_1_train_val.json --rounds 2
```

For backward compatibility, direct old command names like `python -m agi_core_codex arc-data ...` still route to `legacy`.

## What The Rewrite Changed

- The default entrypoint is now the minimal learner, not the ARC strategy stack.
- The active curriculum starts with generated synthetic baby tasks in [domains.py](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal/domains.py).
- Wake can only reuse the committed library from prior rounds.
- Sleep promotes reusable subprograms using only exactness, mean accuracy, compression gain, reuse breadth, and failure behavior.
- `library_ref` wrappers are no longer promoted as if they were genuine abstractions.
- The public `agi_core_codex.core` surface now points at the minimal learner types.

## Current Evidence

This rewrite is about compounding evidence first, not top-line ARC score first.

Current verified signals:
- synthetic compounding smoke: round 2 beats round 1 and reuses committed library
- minimal ARC smoke: the active ARC path runs end to end without importing legacy ARC strategy modules
- focused verification after the rewrite: `73 passed`

The synthetic curriculum is the first gate. The graduation criteria before serious ARC emphasis are:
- round-to-round solve gain
- non-zero library reuse
- lower search cost per exact solve after promotion on at least one curriculum tier

## Layout

- [src/agi_core_codex/minimal](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/minimal): active minimal learner
- [src/agi_core_codex/legacy](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/legacy): archived command surface for older solver stacks
- [src/agi_core_codex/domains/arc](/Users/vibhorjain/github/agi-core-codex/src/agi_core_codex/domains/arc): ARC data types, loaders, scorers, and legacy solver implementations
- [experiments/splits](/Users/vibhorjain/github/agi-core-codex/experiments/splits): synthetic and ARC split files
- [tests](/Users/vibhorjain/github/agi-core-codex/tests): regression and smoke coverage

## Honest Status

The repo is much closer to the spirit of the 4 pillars now, but it is still early.

- The active learner is minimal enough to study compounding directly.
- The legacy ARC solvers are still in the repository because they remain useful baselines.
- Serious ARC performance is not yet the claim of the active path.
- The next work should improve generic sleep/promotion and curriculum design before reintroducing benchmark pressure.
