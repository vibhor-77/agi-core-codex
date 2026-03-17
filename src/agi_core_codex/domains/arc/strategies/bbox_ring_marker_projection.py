from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import background_color, connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _project_markers_to_bbox_ring(grid: Grid) -> Grid:
    if not grid:
        return ()

    bg = background_color(grid)
    components = connected_components(grid, bg_color=bg)
    if not components:
        return grid

    target_components = [component for component in components if component.size > 1]
    if len(target_components) != 1:
        return grid

    target = target_components[0]
    row_start, col_start, row_end, col_end = target.bbox
    rows = [list(row) for row in grid]
    changed = False

    for component in components:
        if component == target or component.size != 1:
            continue
        marker_row, marker_col = component.pixels[0]
        projected_row = (
            row_start - 1
            if marker_row < row_start
            else row_end + 1 if marker_row > row_end else marker_row
        )
        projected_col = (
            col_start - 1
            if marker_col < col_start
            else col_end + 1 if marker_col > col_end else marker_col
        )
        if not (0 <= projected_row < len(rows) and 0 <= projected_col < len(rows[0])):
            continue
        rows[marker_row][marker_col] = bg
        rows[projected_row][projected_col] = component.color
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
class BboxRingMarkerProjectionStrategy:
    name: str = "arc-bbox-ring-marker-projection"
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
                notes=("bbox ring projection only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _project_markers_to_bbox_ring(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("marker projection onto the bbox ring did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="project-markers-to-bbox-ring",
            semantics={"type": "bbox_ring_marker_projection"},
            executor=_project_markers_to_bbox_ring,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
