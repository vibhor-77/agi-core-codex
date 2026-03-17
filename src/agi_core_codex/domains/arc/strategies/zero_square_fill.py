from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _infer_fill_color(input_grid: Grid, output_grid: Grid) -> int | None:
    if grid_shape(input_grid) != grid_shape(output_grid):
        return None

    changes = [
        (input_grid[row_index][col_index], output_grid[row_index][col_index])
        for row_index in range(len(input_grid))
        for col_index in range(len(input_grid[row_index]))
        if input_grid[row_index][col_index] != output_grid[row_index][col_index]
    ]
    if not changes:
        return None

    if any(source != 0 for source, _ in changes):
        return None
    fill_colors = {replacement for _, replacement in changes}
    if len(fill_colors) != 1:
        return None
    return next(iter(fill_colors))


def _fill_zero_component_squares(grid: Grid, fill_color: int) -> Grid:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    rows = [list(row) for row in grid]
    visited: set[tuple[int, int]] = set()
    changed = False

    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] != 0 or (row_index, col_index) in visited:
                continue

            stack = [(row_index, col_index)]
            visited.add((row_index, col_index))
            component_cells = []
            while stack:
                current_row, current_col = stack.pop()
                component_cells.append((current_row, current_col))
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if grid[next_row][next_col] != 0 or (next_row, next_col) in visited:
                        continue
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

            component_set = set(component_cells)
            candidates = []
            for candidate_row, candidate_col in component_cells:
                block = {
                    (candidate_row + delta_row, candidate_col + delta_col)
                    for delta_row in range(3)
                    for delta_col in range(3)
                }
                if all(
                    0 <= block_row < height
                    and 0 <= block_col < width
                    and (block_row, block_col) in component_set
                    for block_row, block_col in block
                ):
                    candidates.append((candidate_row, candidate_col))

            if not candidates:
                continue

            fill_row, fill_col = min(candidates)
            for delta_row in range(3):
                for delta_col in range(3):
                    rows[fill_row + delta_row][fill_col + delta_col] = fill_color
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
class ZeroSquareFillStrategy:
    name: str = "arc-zero-square-fill"
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
                notes=("zero square fill only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        fill_color = _infer_fill_color(first_example.input, first_example.output)
        if fill_color is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("could not infer a single zero-to-color fill rule from the first example",),
            )

        executor = lambda grid, fill_color=fill_color: _fill_zero_component_squares(grid, fill_color)
        if executor(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("zero component square fill did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name=f"fill-zero-component-squares-{fill_color}",
            semantics={
                "type": "zero_square_fill",
                "fill_color": fill_color,
            },
            executor=executor,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
