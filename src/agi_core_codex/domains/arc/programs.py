from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from agi_core_codex.core.hashing import stable_hash
from agi_core_codex.core.interfaces import CostModel, ProgramHandle
from agi_core_codex.domains.arc.types import Grid, freeze_grid, grid_to_lists


ArcProgramExecutor = Callable[[Grid], Grid]


def _looks_like_grid(value: Any) -> bool:
    if not isinstance(value, tuple):
        return False
    return all(isinstance(row, tuple) for row in value)


def _normalize_semantics(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_semantics(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_normalize_semantics(item) for item in value]
    if isinstance(value, list):
        return [_normalize_semantics(item) for item in value]
    if _looks_like_grid(value):
        return grid_to_lists(value)
    return value


def make_arc_program(
    *,
    name: str,
    semantics: Mapping[str, Any],
    executor: ArcProgramExecutor,
    complexity: int = 1,
) -> ProgramHandle:
    full_semantics = {
        "domain": "arc",
        "name": name,
        "semantics": _normalize_semantics(dict(semantics)),
    }

    def wrapped(grid: Grid) -> Grid:
        return freeze_grid(executor(freeze_grid(grid)))

    return ProgramHandle(
        id=stable_hash(full_semantics, namespace="arc.program"),
        name=name,
        domain="arc",
        executor=wrapped,
        cost=CostModel(complexity=complexity),
        semantics=full_semantics,
    )
