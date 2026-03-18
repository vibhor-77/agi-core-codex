from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _find_barrier_row(grid: Grid, *, barrier_color: int) -> int | None:
    for row_index, row in enumerate(grid):
        if all(value == barrier_color for value in row):
            return row_index
    return None


def _fill_relative_to_barrier(
    grid: Grid,
    *,
    barrier_color: int,
    toward_barrier_color: int,
    away_from_barrier_color: int,
) -> Grid:
    if not grid:
        return ()

    height, _ = grid_shape(grid)
    barrier_row = _find_barrier_row(grid, barrier_color=barrier_color)
    if barrier_row is None:
        return grid

    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index, value in enumerate(grid[row_index]):
            if value == toward_barrier_color:
                if row_index < barrier_row:
                    for next_row in range(row_index + 1, barrier_row):
                        rows[next_row][col_index] = toward_barrier_color
                elif row_index > barrier_row:
                    for next_row in range(barrier_row + 1, row_index):
                        rows[next_row][col_index] = toward_barrier_color
            elif value == away_from_barrier_color:
                if row_index < barrier_row:
                    for next_row in range(0, row_index):
                        rows[next_row][col_index] = away_from_barrier_color
                elif row_index > barrier_row:
                    for next_row in range(row_index + 1, height):
                        rows[next_row][col_index] = away_from_barrier_color
    return freeze_grid(rows)


@dataclass(frozen=True)
class BarrierDirectionalFillStrategy:
    name: str = "arc-barrier-directional-fill"
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
                notes=("barrier directional fill only applies to same-shape tasks",),
            )

        first_example = context.task.train[0]
        input_colors = sorted({value for row in first_example.input for value in row if value != 0})
        if len(input_colors) != 3:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("barrier directional fill expects exactly three nonzero input colors",),
            )

        barrier_color = next(
            (
                color
                for color in input_colors
                if _find_barrier_row(first_example.input, barrier_color=color) is not None
            ),
            None,
        )
        if barrier_color is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("barrier directional fill needs a full-width barrier row",),
            )

        other_colors = [color for color in input_colors if color != barrier_color]
        executor = None
        chosen = None
        for toward_barrier_color in other_colors:
            away_from_barrier_color = next(
                color for color in other_colors if color != toward_barrier_color
            )
            candidate_executor = (
                lambda grid, bc=barrier_color, tc=toward_barrier_color, ac=away_from_barrier_color:
                _fill_relative_to_barrier(
                    grid,
                    barrier_color=bc,
                    toward_barrier_color=tc,
                    away_from_barrier_color=ac,
                )
            )
            if all(candidate_executor(example.input) == example.output for example in context.task.train):
                executor = candidate_executor
                chosen = (toward_barrier_color, away_from_barrier_color)
                break

        if executor is None or chosen is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("barrier directional fill did not match every train example",),
            )

        toward_barrier_color, away_from_barrier_color = chosen
        program = make_arc_program(
            name=(
                f"fill-{toward_barrier_color}-toward-barrier-"
                f"and-{away_from_barrier_color}-away-from-barrier"
            ),
            semantics={
                "type": "barrier_directional_fill",
                "barrier_color": barrier_color,
                "toward_barrier_color": toward_barrier_color,
                "away_from_barrier_color": away_from_barrier_color,
            },
            executor=executor,
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
