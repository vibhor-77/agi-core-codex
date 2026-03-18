from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _recolor_right_parity_columns(
    grid: Grid,
    *,
    source_color: int,
    target_color: int,
) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    occupied_cols = [
        col_index
        for col_index in range(width)
        if any(grid[row_index][col_index] == source_color for row_index in range(height))
    ]
    if not occupied_cols:
        return grid

    target_parity = occupied_cols[-1] % 2
    rows = [list(row) for row in grid]
    for col_index in occupied_cols:
        if col_index % 2 != target_parity:
            continue
        for row_index in range(height):
            if rows[row_index][col_index] == source_color:
                rows[row_index][col_index] = target_color
    return freeze_grid(rows)


@dataclass(frozen=True)
class RightParityColumnRecolorStrategy:
    name: str = "arc-right-parity-column-recolor"
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
                notes=("right-parity column recolor only applies to same-shape tasks",),
            )

        first_example = context.task.train[0]
        input_colors = sorted({value for row in first_example.input for value in row if value != 0})
        output_colors = sorted({value for row in first_example.output for value in row if value != 0})
        if len(input_colors) != 1 or len(output_colors) != 2 or input_colors[0] not in output_colors:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("right-parity column recolor expects one input color and one added output color",),
            )

        source_color = input_colors[0]
        target_color = next(color for color in output_colors if color != source_color)
        for example in context.task.train:
            if any(value not in (0, source_color) for row in example.input for value in row):
                return context.finish_strategy(
                    name=self.name,
                    status="not_applicable",
                    generated=0,
                    candidates=(),
                    notes=("right-parity column recolor only supports a single nonzero input color",),
                )
            if any(
                sum(
                    1
                    for row_index in range(len(example.input))
                    if example.input[row_index][col_index] == source_color
                )
                > 1
                for col_index in range(len(example.input[0]))
            ):
                return context.finish_strategy(
                    name=self.name,
                    status="not_applicable",
                    generated=0,
                    candidates=(),
                    notes=("right-parity column recolor expects at most one source cell per column",),
                )

        executor = lambda grid, sc=source_color, tc=target_color: _recolor_right_parity_columns(
            grid,
            source_color=sc,
            target_color=tc,
        )
        if any(executor(example.input) != example.output for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("right-parity column recolor did not match every train example",),
            )

        program = make_arc_program(
            name=f"recolor-right-parity-{source_color}-columns-to-{target_color}",
            semantics={
                "type": "recolor_right_parity_columns",
                "source_color": source_color,
                "target_color": target_color,
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
