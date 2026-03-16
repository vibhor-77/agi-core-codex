from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_BENCHMARK_DIRS = {
    "arc-agi-1": "ARC-AGI",
    "arc-agi-2": "ARC-AGI-2",
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _candidate_paths_for_root(root: Path, benchmark: str, split: str) -> tuple[Path, ...]:
    benchmark_dir = _BENCHMARK_DIRS[benchmark]
    return (
        root / benchmark_dir / "data" / split,
        root / "data" / benchmark_dir / "data" / split,
        root / split,
        root,
    )


def _env_candidates(benchmark: str, split: str) -> tuple[Path, ...]:
    benchmark_key = benchmark.upper().replace("-", "_")
    names = (
        f"{benchmark_key}_{split.upper()}_DIR",
        f"{benchmark_key}_DATA_DIR",
    )
    paths = []
    for name in names:
        value = os.environ.get(name)
        if value:
            paths.append(Path(value).expanduser())
    return tuple(paths)


def validate_dataset_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.json"))


def candidate_dataset_paths(
    *,
    benchmark: str,
    split: str,
    search_roots: Iterable[Path] = (),
) -> tuple[Path, ...]:
    if benchmark not in _BENCHMARK_DIRS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    if split not in {"training", "evaluation"}:
        raise ValueError(f"unsupported ARC split: {split}")

    repo_root = repository_root()
    roots = (
        *_env_candidates(benchmark, split),
        *(Path(root).expanduser() for root in search_roots),
        repo_root,
        repo_root.parent,
        repo_root.parent / "agi-core",
        Path.home() / "github" / "agi-core-codex",
        Path.home() / "github" / "agi-core",
    )

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for candidate in _candidate_paths_for_root(root, benchmark, split):
            resolved = candidate.expanduser()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
    return tuple(candidates)


def discover_arc_dataset(
    *,
    benchmark: str,
    split: str,
    search_roots: Iterable[Path] = (),
) -> Path | None:
    for candidate in candidate_dataset_paths(
        benchmark=benchmark,
        split=split,
        search_roots=search_roots,
    ):
        if validate_dataset_dir(candidate):
            return candidate
    return None

