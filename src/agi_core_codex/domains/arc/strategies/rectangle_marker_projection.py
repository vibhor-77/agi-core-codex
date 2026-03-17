from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import background_color, connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _project_markers_into_rectangle(grid: Grid) -> Grid:
    if not grid:
        return ()

    bg = background_color(grid)
    components = connected_components(grid, bg_color=bg)
    if not components:
        return grid

    target = max(components, key=lambda component: (component.size, component.bbox))
    row_start, col_start, row_end, col_end = target.bbox
    rows = [list(row) for row in grid]

    for component in components:
        if component == target or component.size != 1:
            continue
        marker_row, marker_col = component.pixels[0]
        color = component.color
        if row_start <= marker_row <= row_end and marker_col < col_start:
            rows[marker_row][col_start] = color
        elif row_start <= marker_row <= row_end and marker_col > col_end:
            rows[marker_row][col_end] = color
        elif col_start <= marker_col <= col_end and marker_row < row_start:
            rows[row_start][marker_col] = color
        elif col_start <= marker_col <= col_end and marker_row > row_end:
            rows[row_end][marker_col] = color

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
class RectangleMarkerProjectionStrategy:
    name: str = "arc-rectangle-marker-projection"
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
                notes=("rectangle marker projection only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _project_markers_into_rectangle(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("marker projection into dominant rectangle did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="project-markers-into-rectangle",
            semantics={"type": "rectangle_marker_projection"},
            executor=_project_markers_into_rectangle,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
