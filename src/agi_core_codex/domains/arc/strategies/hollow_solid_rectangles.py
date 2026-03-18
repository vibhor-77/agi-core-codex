from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _hollow_solid_rectangles(grid: Grid) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    seen: set[tuple[int, int]] = set()
    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index in range(width):
            color = grid[row_index][col_index]
            if color == 0 or (row_index, col_index) in seen:
                continue

            stack = [(row_index, col_index)]
            seen.add((row_index, col_index))
            pixels = []
            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row, current_col))
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in seen or grid[next_row][next_col] != color:
                        continue
                    seen.add((next_row, next_col))
                    stack.append((next_row, next_col))

            component_rows = [value for value, _ in pixels]
            component_cols = [value for _, value in pixels]
            row_start = min(component_rows)
            row_end = max(component_rows)
            col_start = min(component_cols)
            col_end = max(component_cols)
            if (row_end - row_start + 1) < 3 or (col_end - col_start + 1) < 3:
                continue
            if (row_end - row_start + 1) * (col_end - col_start + 1) != len(pixels):
                continue

            for inner_row in range(row_start + 1, row_end):
                for inner_col in range(col_start + 1, col_end):
                    rows[inner_row][inner_col] = 0

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
class HollowSolidRectanglesStrategy:
    name: str = "arc-hollow-solid-rectangles"
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
                notes=("hollow solid rectangles only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _hollow_solid_rectangles(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("solid-rectangle hollowing did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="hollow-solid-rectangles",
            semantics={"type": "hollow_solid_rectangles"},
            executor=_hollow_solid_rectangles,
            complexity=4,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
