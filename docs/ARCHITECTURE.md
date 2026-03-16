# Architecture

## Core

The core exports five primary concepts:

- `Environment`: task identity and size normalization
- `Grammar`: deterministic primitive enumeration
- `Scorer`: domain-owned execution and scoring
- `Memory`: explicit reusable library storage
- `Strategy`: optional search operators layered on top of the kernel

The kernel never calls domain-specific methods such as object decomposition, cross-reference matching, or patch correction. Those belong in domain strategy plugins.

## ARC

ARC lives under `domains/arc/` and owns:

- task loading
- grid types and scoring
- primitive enumeration
- task-scoped synthesis strategies

ARC-specific search operators are named strategies with explicit attribution in manifests, so benchmark heuristics cannot hide behind the common core abstraction.

## Experiments

Every run writes:

- a manifest with code/environment fingerprints
- per-task best-program summaries
- per-strategy phase attribution
- aggregate metrics

This makes regression tracking possible without depending on ad hoc `runs/` directories.

