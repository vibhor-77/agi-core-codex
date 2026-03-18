from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _infer_fill_spec(task: ArcTask) -> tuple[int, int] | None:
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

        local_source_colors: set[int] = set()
        local_fill_colors: set[int] = set()
        for row_index, col_index in diffs:
            if input_grid[row_index][col_index] != 0:
                return None
            local_fill_colors.add(output_grid[row_index][col_index])

            matches: set[int] = set()
            for row_start in (row_index - 1, row_index):
                for col_start in (col_index - 1, col_index):
                    if row_start < 0 or col_start < 0:
                        continue
                    if row_start + 1 >= height or col_start + 1 >= width:
                        continue

                    cells = [
                        (scan_row, scan_col, input_grid[scan_row][scan_col])
                        for scan_row in range(row_start, row_start + 2)
                        for scan_col in range(col_start, col_start + 2)
                    ]
                    nonzero = [
                        value
                        for scan_row, scan_col, value in cells
                        if (scan_row, scan_col) != (row_index, col_index) and value != 0
                    ]
                    if len(nonzero) == 3 and len(set(nonzero)) == 1:
                        matches.add(nonzero[0])

            if len(matches) != 1:
                return None
            local_source_colors.update(matches)

        if len(local_source_colors) != 1 or len(local_fill_colors) != 1:
            return None
        source_colors.update(local_source_colors)
        fill_colors.update(local_fill_colors)

    if len(source_colors) != 1 or len(fill_colors) != 1:
        return None

    return next(iter(source_colors)), next(iter(fill_colors))


def _fill_triomino_corners(grid: Grid, *, source_color: int, fill_color: int) -> Grid:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    rows = [list(row) for row in grid]
    changed = False
    for row_start in range(height - 1):
        for col_start in range(width - 1):
            window = [
                (row_index, col_index, grid[row_index][col_index])
                for row_index in range(row_start, row_start + 2)
                for col_index in range(col_start, col_start + 2)
            ]
            source_count = sum(1 for _, _, value in window if value == source_color)
            zero_count = sum(1 for _, _, value in window if value == 0)
            if source_count != 3 or zero_count != 1:
                continue
            for row_index, col_index, value in window:
                if value == 0:
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
class TriominoCornerFillStrategy:
    name: str = "arc-triomino-corner-fill"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        spec = _infer_fill_spec(context.task)
        if spec is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("triomino corner fill pattern not detected",),
            )

        source_color, fill_color = spec
        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name=f"fill-triomino-corners-{source_color}-with-{fill_color}",
            semantics={
                "type": "triomino_corner_fill",
                "source_color": source_color,
                "fill_color": fill_color,
            },
            executor=lambda grid: _fill_triomino_corners(
                grid,
                source_color=source_color,
                fill_color=fill_color,
            ),
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
