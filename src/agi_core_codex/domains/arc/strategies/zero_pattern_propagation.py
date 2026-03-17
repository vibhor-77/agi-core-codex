from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _background_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0]


def _propagate_zero_pattern(grid: Grid) -> Grid:
    if not grid:
        return ()

    background = _background_color(grid)
    special_cells = [
        (row_index, col_index, value)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value not in (background, 0)
    ]
    zero_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value == 0
    ]
    if len(special_cells) != 1 or not zero_cells:
        return grid

    special_row, special_col, special_color = special_cells[0]
    reference_row, reference_col = max(
        zero_cells,
        key=lambda cell: (
            abs(cell[0] - special_row) + abs(cell[1] - special_col),
            -cell[0],
            -cell[1],
        ),
    )
    step_row = special_row - reference_row
    step_col = special_col - reference_col
    if step_row == 0 and step_col == 0:
        return grid

    height = len(grid)
    width = len(grid[0])
    rows = [list(row) for row in grid]
    changed = False
    multiplier = 1
    while True:
        translated = []
        for zero_row, zero_col in zero_cells:
            next_row = zero_row + step_row * multiplier
            next_col = zero_col + step_col * multiplier
            if 0 <= next_row < height and 0 <= next_col < width:
                translated.append((next_row, next_col))
        if not translated:
            break
        for next_row, next_col in translated:
            if rows[next_row][next_col] == background:
                rows[next_row][next_col] = special_color
                changed = True
        multiplier += 1

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
class ZeroPatternPropagationStrategy:
    name: str = "arc-zero-pattern-propagation"
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
                notes=("zero pattern propagation only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _propagate_zero_pattern(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("zero pattern propagation did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="propagate-zero-pattern",
            semantics={"type": "zero_pattern_propagation"},
            executor=_propagate_zero_pattern,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
