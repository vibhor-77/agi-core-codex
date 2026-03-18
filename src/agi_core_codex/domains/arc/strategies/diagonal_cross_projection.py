from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _project_diagonal_cross(grid: Grid) -> Grid:
    if not grid:
        return ()

    nonzero = [
        (row_index, col_index, value)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value != 0
    ]
    if len(nonzero) != 1:
        return grid

    seed_row, seed_col, color = nonzero[0]
    height, width = grid_shape(grid)
    rows = [[0 for _ in range(width)] for _ in range(height)]
    for row_index in range(height):
        for col_index in range(width):
            if abs(row_index - seed_row) == abs(col_index - seed_col):
                rows[row_index][col_index] = color
    return freeze_grid(rows)


def _emit_program(
    *,
    context: Any,
    strategy_name: str,
    name: str,
    semantics: dict[str, Any],
    executor: Callable[[Grid], Grid],
    complexity: int,
):
    program = make_arc_program(
        name=name,
        semantics=semantics,
        executor=executor,
        complexity=complexity,
    )
    return context.evaluate(program, strategy_name)


@dataclass(frozen=True)
class DiagonalCrossProjectionStrategy:
    name: str = "arc-diagonal-cross-projection"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        if any(grid_shape(example.input) != grid_shape(example.output) for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("diagonal cross projection only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _project_diagonal_cross(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("single-seed diagonal cross rule did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="project-diagonal-cross",
            semantics={"type": "diagonal_cross_projection"},
            executor=_project_diagonal_cross,
            complexity=3,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
