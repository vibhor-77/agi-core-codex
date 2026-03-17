from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import (
    find_uniform_col_separators,
    find_uniform_row_separators,
    flatten_cells,
    intersect_separators,
    merge_cells_by_separators,
    split_grid_by_separators,
)
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, grid_shape


def _is_empty(cell: Grid) -> bool:
    return all(value == 0 for row in cell for value in row)


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


def _propagate_matching_cells(
    grid: Grid,
    row_separators: tuple[int, ...],
    col_separators: tuple[int, ...],
) -> Grid:
    cells = split_grid_by_separators(grid, row_separators, col_separators)
    if not cells or not cells[0]:
        return grid

    result = [list(row) for row in cells]
    pattern_positions: dict[Grid, list[tuple[int, int]]] = {}
    for row_index, col_index, cell in flatten_cells(cells):
        if _is_empty(cell):
            continue
        pattern_positions.setdefault(cell, []).append((row_index, col_index))

    for pattern, positions in pattern_positions.items():
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for row_index, col_index in positions:
            by_row.setdefault(row_index, []).append(col_index)
            by_col.setdefault(col_index, []).append(row_index)

        for row_index, cols in by_row.items():
            if len(cols) < 2:
                continue
            left = min(cols)
            right = max(cols)
            for col_index in range(left, right + 1):
                if _is_empty(cells[row_index][col_index]):
                    result[row_index][col_index] = pattern

        for col_index, rows in by_col.items():
            if len(rows) < 2:
                continue
            top = min(rows)
            bottom = max(rows)
            for row_index in range(top, bottom + 1):
                if _is_empty(cells[row_index][col_index]):
                    result[row_index][col_index] = pattern

    merged_cells = tuple(tuple(row) for row in result)
    return merge_cells_by_separators(grid, row_separators, col_separators, merged_cells)


@dataclass(frozen=True)
class SeparatorPropagationStrategy:
    name: str = "arc-separator-propagation"
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
                notes=("separator propagation only applies to same-shape input/output tasks",),
            )

        row_separators = intersect_separators(
            find_uniform_row_separators(example.input)
            for example in context.task.train
        )
        col_separators = intersect_separators(
            find_uniform_col_separators(example.input)
            for example in context.task.train
        )
        if not row_separators and not col_separators:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("no shared separator structure found across training inputs",),
            )

        first_example = context.task.train[0]
        if _propagate_matching_cells(first_example.input, row_separators, col_separators) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("row/column cell propagation did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="cross-ref-separator-propagation",
            semantics={
                "type": "separator_propagation",
                "row_separators": row_separators,
                "col_separators": col_separators,
            },
            executor=lambda grid, row_separators=row_separators, col_separators=col_separators: _propagate_matching_cells(
                grid,
                row_separators,
                col_separators,
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
