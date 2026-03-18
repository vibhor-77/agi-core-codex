from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _paint_orth_diagonal_halos(grid: Grid) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index in range(width):
            value = grid[row_index][col_index]
            if value == 1:
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = row_index + delta_row
                    next_col = col_index + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width and rows[next_row][next_col] == 0:
                        rows[next_row][next_col] = 7
            elif value == 2:
                for delta_row, delta_col in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    next_row = row_index + delta_row
                    next_col = col_index + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width and rows[next_row][next_col] == 0:
                        rows[next_row][next_col] = 4
    return freeze_grid(rows)


@dataclass(frozen=True)
class OrthDiagonalHaloStrategy:
    name: str = "arc-orth-diagonal-halo"
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
                notes=("orth-diagonal halo only applies to same-shape tasks",),
            )

        if any(_paint_orth_diagonal_halos(example.input) != example.output for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("orth-diagonal halo did not match every train example",),
            )

        program = make_arc_program(
            name="paint-1-orth-7-and-2-diagonal-4",
            semantics={
                "type": "paint_orth_diagonal_halos",
                "orth_source_color": 1,
                "orth_fill_color": 7,
                "diag_source_color": 2,
                "diag_fill_color": 4,
            },
            executor=_paint_orth_diagonal_halos,
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
