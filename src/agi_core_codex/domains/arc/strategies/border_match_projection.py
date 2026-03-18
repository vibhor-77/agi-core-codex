from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _frame_colors(grid: Grid) -> tuple[int, int, int, int] | None:
    height, width = grid_shape(grid)
    if height < 3 or width < 3:
        return None

    top = grid[0][1]
    bottom = grid[height - 1][1]
    left = grid[1][0]
    right = grid[1][width - 1]
    if 0 in (top, bottom, left, right):
        return None

    if any(grid[0][col_index] != top for col_index in range(1, width - 1)):
        return None
    if any(grid[height - 1][col_index] != bottom for col_index in range(1, width - 1)):
        return None
    if any(grid[row_index][0] != left for row_index in range(1, height - 1)):
        return None
    if any(grid[row_index][width - 1] != right for row_index in range(1, height - 1)):
        return None

    return top, bottom, left, right


def _project_border_matches(
    grid: Grid,
    *,
    top_color: int,
    bottom_color: int,
    left_color: int,
    right_color: int,
) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    rows = [[0 for _ in range(width)] for _ in range(height)]

    rows[0][0] = grid[0][0]
    rows[0][width - 1] = grid[0][width - 1]
    rows[height - 1][0] = grid[height - 1][0]
    rows[height - 1][width - 1] = grid[height - 1][width - 1]

    for col_index in range(1, width - 1):
        rows[0][col_index] = top_color
        rows[height - 1][col_index] = bottom_color
    for row_index in range(1, height - 1):
        rows[row_index][0] = left_color
        rows[row_index][width - 1] = right_color

    for row_index in range(1, height - 1):
        for col_index in range(1, width - 1):
            value = grid[row_index][col_index]
            if value == top_color:
                rows[1][col_index] = value
            if value == bottom_color:
                rows[height - 2][col_index] = value
            if value == left_color:
                rows[row_index][1] = value
            if value == right_color:
                rows[row_index][width - 2] = value

    return freeze_grid(rows)


def _project_dynamic_border_matches(grid: Grid) -> Grid:
    colors = _frame_colors(grid)
    if colors is None:
        return grid
    top_color, bottom_color, left_color, right_color = colors
    return _project_border_matches(
        grid,
        top_color=top_color,
        bottom_color=bottom_color,
        left_color=left_color,
        right_color=right_color,
    )


@dataclass(frozen=True)
class BorderMatchProjectionStrategy:
    name: str = "arc-border-match-projection"
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
                notes=("border match projection only applies to same-shape tasks",),
            )

        colors = _frame_colors(context.task.train[0].input)
        if colors is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("border match projection needs a four-sided colored frame",),
            )

        if any(_project_dynamic_border_matches(example.input) != example.output for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("border match projection did not match every train example",),
            )

        program = make_arc_program(
            name="project-border-matching-cells",
            semantics={"type": "project_border_matches"},
            executor=_project_dynamic_border_matches,
            complexity=2,
        )
        candidate = context.evaluate(program, self.name)
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
