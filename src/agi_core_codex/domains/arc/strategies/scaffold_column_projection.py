from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _identify_scaffold(grid: Grid) -> tuple[int, tuple[tuple[int, int], ...], tuple[int, int, int, int]] | None:
    components = [component for component in connected_components(grid, bg_color=0) if component.size == 8]
    if len(components) != 1:
        return None
    component = components[0]
    return component.color, component.pixels, component.bbox


def _project_scaffold_columns(grid: Grid) -> Grid:
    if not grid:
        return ()

    scaffold = _identify_scaffold(grid)
    if scaffold is None:
        return grid

    object_color, pixels, bbox = scaffold
    colors = {value for row in grid for value in row if value != 0}
    if len(colors) != 2:
        return grid
    source_color = next(color for color in colors if color != object_color)

    height = len(grid)
    _, col_start, row_end, col_end = bbox
    column_rows: dict[int, list[int]] = {}
    for row_index, col_index in pixels:
        column_rows.setdefault(col_index, []).append(row_index)

    selected_cols = sorted(
        {
            col_index
            for row_index in range(row_end + 1, height)
            for col_index in range(col_start, col_end + 1)
            if grid[row_index][col_index] == source_color
        }
    )
    if not selected_cols or any(col_index not in column_rows for col_index in selected_cols):
        return grid

    rows = [list(row) for row in grid]
    changed = False
    for col_index in selected_cols:
        start_row = max(column_rows[col_index]) + 1
        for row_index in range(start_row, height):
            if rows[row_index][col_index] == 0:
                rows[row_index][col_index] = source_color
                changed = True

    return freeze_grid(rows) if changed else grid


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
class ScaffoldColumnProjectionStrategy:
    name: str = "arc-scaffold-column-projection"
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
                notes=("scaffold column projection only applies to same-shape tasks",),
            )

        first_example = context.task.train[0]
        if _project_scaffold_columns(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("scaffold column projection did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="project-columns-from-scaffold",
            semantics={"type": "scaffold_column_projection"},
            executor=_project_scaffold_columns,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
