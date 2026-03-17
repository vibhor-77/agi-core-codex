from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import background_color, connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _added_marker_color(input_grid: Grid, output_grid: Grid) -> int | None:
    colors = {
        output_grid[row_index][col_index]
        for row_index in range(len(input_grid))
        for col_index in range(len(input_grid[row_index]))
        if input_grid[row_index][col_index] != output_grid[row_index][col_index]
        and output_grid[row_index][col_index] != 0
    }
    if len(colors) != 1:
        return None
    return next(iter(colors))


def _place_bottom_center_markers(grid: Grid, marker_color: int) -> Grid:
    if not grid:
        return ()

    rows = [list(row) for row in grid]
    bottom_row = len(rows) - 1
    bg = background_color(grid)
    for component in connected_components(grid, bg_color=bg):
        _, col_start, _, col_end = component.bbox
        center_col = (col_start + col_end) // 2
        rows[bottom_row][center_col] = marker_color
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
class BottomCenterMarkerStrategy:
    name: str = "arc-bottom-center-marker"
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
                notes=("bottom-center markers only apply to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        marker_color = _added_marker_color(first_example.input, first_example.output)
        if marker_color is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("could not infer a single marker color from the first example",),
            )

        executor = lambda grid, marker_color=marker_color: _place_bottom_center_markers(grid, marker_color)
        if executor(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("bottom-center marker placement did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name=f"bottom-center-markers-{marker_color}",
            semantics={
                "type": "bottom_center_markers",
                "marker_color": marker_color,
            },
            executor=executor,
            complexity=4,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
