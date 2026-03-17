from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _infer_bbox_recolor_params(input_grid: Grid, output_grid: Grid) -> tuple[int, int, int] | None:
    if grid_shape(input_grid) != grid_shape(output_grid):
        return None

    changes = [
        (row_index, col_index, input_grid[row_index][col_index], output_grid[row_index][col_index])
        for row_index in range(len(input_grid))
        for col_index in range(len(input_grid[row_index]))
        if input_grid[row_index][col_index] != output_grid[row_index][col_index]
    ]
    if not changes:
        return None

    source_colors = {source for _, _, source, _ in changes}
    replacement_colors = {replacement for _, _, _, replacement in changes}
    if len(source_colors) != 1 or len(replacement_colors) != 1:
        return None

    source_color = next(iter(source_colors))
    replacement_color = next(iter(replacement_colors))
    changed_positions = {(row_index, col_index) for row_index, col_index, _, _ in changes}
    candidate_region_colors = sorted(
        {
            cell
            for row in input_grid
            for cell in row
            if cell not in (0, source_color)
        }
    )

    for region_color in candidate_region_colors:
        positions = [
            (row_index, col_index)
            for row_index, row in enumerate(input_grid)
            for col_index, value in enumerate(row)
            if value == region_color
        ]
        if not positions:
            continue
        row_start = min(row_index for row_index, _ in positions)
        col_start = min(col_index for _, col_index in positions)
        row_end = max(row_index for row_index, _ in positions)
        col_end = max(col_index for _, col_index in positions)
        predicted_changes = {
            (row_index, col_index)
            for row_index in range(row_start, row_end + 1)
            for col_index in range(col_start, col_end + 1)
            if input_grid[row_index][col_index] == source_color
        }
        if predicted_changes == changed_positions:
            return source_color, replacement_color, region_color
    return None


def _recolor_inside_region_bbox(
    grid: Grid,
    *,
    source_color: int,
    replacement_color: int,
    region_color: int,
) -> Grid:
    positions = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value == region_color
    ]
    if not positions:
        return grid

    row_start = min(row_index for row_index, _ in positions)
    col_start = min(col_index for _, col_index in positions)
    row_end = max(row_index for row_index, _ in positions)
    col_end = max(col_index for _, col_index in positions)
    rows = [list(row) for row in grid]
    changed = False
    for row_index in range(row_start, row_end + 1):
        for col_index in range(col_start, col_end + 1):
            if rows[row_index][col_index] == source_color:
                rows[row_index][col_index] = replacement_color
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
class BboxRecolorStrategy:
    name: str = "arc-bbox-recolor"
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
                notes=("bbox recolor only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        params = _infer_bbox_recolor_params(first_example.input, first_example.output)
        if params is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("could not infer a bbox recolor rule from the first example",),
            )

        source_color, replacement_color, region_color = params
        executor = lambda grid, params=params: _recolor_inside_region_bbox(
            grid,
            source_color=params[0],
            replacement_color=params[1],
            region_color=params[2],
        )
        if executor(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("bbox recolor replay did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name=f"bbox-recolor-{source_color}-to-{replacement_color}-inside-{region_color}",
            semantics={
                "type": "bbox_recolor",
                "source_color": source_color,
                "replacement_color": replacement_color,
                "region_color": region_color,
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
