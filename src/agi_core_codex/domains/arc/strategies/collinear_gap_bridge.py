from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _infer_bridge_spec(task: ArcTask) -> tuple[str, int, int] | None:
    axes: set[str] = set()
    source_colors: set[int] = set()
    fill_colors: set[int] = set()

    for example in task.train:
        input_grid = example.input
        output_grid = example.output
        if grid_shape(input_grid) != grid_shape(output_grid):
            return None

        height = len(input_grid)
        width = len(input_grid[0])
        diffs = [
            (row_index, col_index)
            for row_index in range(height)
            for col_index in range(width)
            if input_grid[row_index][col_index] != output_grid[row_index][col_index]
        ]
        if not diffs:
            return None

        horizontal_matches: set[tuple[int, int]] = set()
        vertical_matches: set[tuple[int, int]] = set()
        horizontal_sources: set[int] = set()
        vertical_sources: set[int] = set()
        local_fill_colors: set[int] = set()

        for row_index, col_index in diffs:
            if input_grid[row_index][col_index] != 0:
                return None
            local_fill_colors.add(output_grid[row_index][col_index])

            if (
                0 < col_index < width - 1
                and input_grid[row_index][col_index - 1] == input_grid[row_index][col_index + 1] != 0
            ):
                horizontal_matches.add((row_index, col_index))
                horizontal_sources.add(input_grid[row_index][col_index - 1])
            if (
                0 < row_index < height - 1
                and input_grid[row_index - 1][col_index] == input_grid[row_index + 1][col_index] != 0
            ):
                vertical_matches.add((row_index, col_index))
                vertical_sources.add(input_grid[row_index - 1][col_index])

        axis = None
        source_color = None
        if horizontal_matches == set(diffs) and len(horizontal_sources) == 1:
            axis = "horizontal"
            source_color = next(iter(horizontal_sources))
        if vertical_matches == set(diffs) and len(vertical_sources) == 1:
            if axis is not None:
                return None
            axis = "vertical"
            source_color = next(iter(vertical_sources))

        if axis is None or source_color is None or len(local_fill_colors) != 1:
            return None

        axes.add(axis)
        source_colors.add(source_color)
        fill_colors.update(local_fill_colors)

    if len(axes) != 1 or len(source_colors) != 1 or len(fill_colors) != 1:
        return None

    return next(iter(axes)), next(iter(source_colors)), next(iter(fill_colors))


def _bridge_collinear_gaps(
    grid: Grid,
    *,
    axis: str,
    source_color: int,
    fill_color: int,
) -> Grid:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    rows = [list(row) for row in grid]
    changed = False
    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] != 0:
                continue

            if (
                axis == "horizontal"
                and 0 < col_index < width - 1
                and grid[row_index][col_index - 1] == grid[row_index][col_index + 1] == source_color
            ):
                rows[row_index][col_index] = fill_color
                changed = True
            if (
                axis == "vertical"
                and 0 < row_index < height - 1
                and grid[row_index - 1][col_index] == grid[row_index + 1][col_index] == source_color
            ):
                rows[row_index][col_index] = fill_color
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
class CollinearGapBridgeStrategy:
    name: str = "arc-collinear-gap-bridge"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        spec = _infer_bridge_spec(context.task)
        if spec is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("collinear gap bridge pattern not detected",),
            )

        axis, source_color, fill_color = spec
        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name=f"bridge-{axis}-gaps-{source_color}-with-{fill_color}",
            semantics={
                "type": "collinear_gap_bridge",
                "axis": axis,
                "source_color": source_color,
                "fill_color": fill_color,
            },
            executor=lambda grid: _bridge_collinear_gaps(
                grid,
                axis=axis,
                source_color=source_color,
                fill_color=fill_color,
            ),
            complexity=4,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
