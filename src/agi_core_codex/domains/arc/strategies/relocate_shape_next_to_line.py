from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [row_index for row_index, _ in cells]
    cols = [col_index for _, col_index in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_full_line(
    grid: Grid,
    *,
    color: int,
) -> tuple[str, int, int, int] | None:
    height, width = grid_shape(grid)
    cells = tuple(
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value == color
    )
    if not cells:
        return None

    row_start, col_start, row_end, col_end = _bbox(cells)
    if row_start == row_end and len(cells) == width and col_start == 0 and col_end == width - 1:
        return ("horizontal", row_start, col_start, col_end)
    if col_start == col_end and len(cells) == height and row_start == 0 and row_end == height - 1:
        return ("vertical", col_start, row_start, row_end)
    return None


def _relocate_shape_next_to_line(
    grid: Grid,
    *,
    line_color: int,
    shape_color: int,
    separator_color: int,
) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    line = _detect_full_line(grid, color=line_color)
    shape_pixels = tuple(
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value == shape_color
    )
    if line is None or not shape_pixels:
        return grid

    shape_row_start, shape_col_start, shape_row_end, shape_col_end = _bbox(shape_pixels)
    rows = [[0 for _ in range(width)] for _ in range(height)]

    orientation, line_index, span_start, span_end = line
    if orientation == "horizontal":
        line_row = line_index
        for col_index in range(span_start, span_end + 1):
            rows[line_row][col_index] = line_color

        shape_height = shape_row_end - shape_row_start + 1
        if shape_row_end < line_row:
            new_row_end = line_row - 1
            new_row_start = new_row_end - shape_height + 1
            separator_row = new_row_start - 1
        else:
            new_row_start = line_row + 1
            new_row_end = new_row_start + shape_height - 1
            separator_row = new_row_end + 1
        if not (0 <= new_row_start <= new_row_end < height and 0 <= separator_row < height):
            return grid

        col_shift = span_start - shape_col_start
        for row_index, col_index in shape_pixels:
            target_row = new_row_start + (row_index - shape_row_start)
            target_col = col_index + col_shift
            if not (0 <= target_row < height and 0 <= target_col < width):
                return grid
            rows[target_row][target_col] = shape_color
        for col_index in range(span_start, span_end + 1):
            rows[separator_row][col_index] = separator_color
    else:
        line_col = line_index
        for row_index in range(span_start, span_end + 1):
            rows[row_index][line_col] = line_color

        shape_width = shape_col_end - shape_col_start + 1
        if shape_col_end < line_col:
            new_col_end = line_col - 1
            new_col_start = new_col_end - shape_width + 1
            separator_col = new_col_start - 1
        else:
            new_col_start = line_col + 1
            new_col_end = new_col_start + shape_width - 1
            separator_col = new_col_end + 1
        if not (0 <= new_col_start <= new_col_end < width and 0 <= separator_col < width):
            return grid

        row_shift = span_start - shape_row_start
        for row_index, col_index in shape_pixels:
            target_row = row_index + row_shift
            target_col = new_col_start + (col_index - shape_col_start)
            if not (0 <= target_row < height and 0 <= target_col < width):
                return grid
            rows[target_row][target_col] = shape_color
        for row_index in range(span_start, span_end + 1):
            rows[row_index][separator_col] = separator_color

    return freeze_grid(rows)


@dataclass(frozen=True)
class RelocateShapeNextToLineStrategy:
    name: str = "arc-relocate-shape-next-to-line"
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
                notes=("shape relocation only applies to same-shape tasks",),
            )

        first_example = context.task.train[0]
        input_colors = sorted({value for row in first_example.input for value in row if value != 0})
        output_colors = sorted({value for row in first_example.output for value in row if value != 0})
        if len(input_colors) != 2 or len(output_colors) != 3:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("shape relocation expects two input colors and one added separator color",),
            )

        line_color = next(
            (
                color
                for color in input_colors
                if _detect_full_line(first_example.input, color=color) is not None
            ),
            None,
        )
        if line_color is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("shape relocation needs a full-row or full-column anchor line",),
            )

        shape_color = next(color for color in input_colors if color != line_color)
        separator_colors = [color for color in output_colors if color not in input_colors]
        if len(separator_colors) != 1:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("shape relocation needs exactly one new separator color",),
            )
        separator_color = separator_colors[0]

        executor = lambda grid, lc=line_color, sc=shape_color, sep=separator_color: _relocate_shape_next_to_line(
            grid,
            line_color=lc,
            shape_color=sc,
            separator_color=sep,
        )
        if any(executor(example.input) != example.output for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("shape relocation did not match every train example",),
            )

        program = make_arc_program(
            name=f"relocate-shape-{shape_color}-next-to-line-{line_color}-with-{separator_color}",
            semantics={
                "type": "relocate_shape_next_to_line",
                "line_color": line_color,
                "shape_color": shape_color,
                "separator_color": separator_color,
            },
            executor=executor,
            complexity=3,
        )
        candidate = context.evaluate(program, self.name)
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
