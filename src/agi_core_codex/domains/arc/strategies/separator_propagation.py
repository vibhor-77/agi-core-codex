from __future__ import annotations

from collections import Counter, deque
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
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


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


def _separator_color(
    grid: Grid,
    row_separators: tuple[int, ...],
    col_separators: tuple[int, ...],
) -> int:
    samples: list[int] = []
    for row_index in row_separators:
        samples.extend(value for value in grid[row_index] if value != 0)
    for col_index in col_separators:
        samples.extend(
            grid[row_index][col_index]
            for row_index in range(len(grid))
            if grid[row_index][col_index] != 0
        )
    return Counter(samples).most_common(1)[0][0] if samples else 0


def _payload_color(cell: Grid, scaffold_color: int) -> int | None:
    colors = {
        value
        for row in cell
        for value in row
        if value not in (0, scaffold_color)
    }
    if len(colors) != 1:
        return None
    return next(iter(colors))


def _supports_payload_color(cell: Grid, scaffold_color: int, color: int) -> bool:
    return all(
        value in (0, scaffold_color, color)
        for row in cell
        for value in row
    )


def _fill_topleft_region(cell: Grid, color: int) -> Grid:
    if not cell or not cell[0]:
        return cell

    rows = [list(row) for row in cell]
    height = len(rows)
    width = len(rows[0])
    starts: list[tuple[int, int]] = []

    for col_index in range(width):
        if rows[0][col_index] == 0:
            starts.append((0, col_index))
        else:
            break
    for row_index in range(height):
        if rows[row_index][0] == 0:
            starts.append((row_index, 0))
        else:
            break

    queue = deque(dict.fromkeys(starts))
    seen = set(queue)
    while queue:
        row_index, col_index = queue.popleft()
        rows[row_index][col_index] = color
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = row_index + delta_row
            next_col = col_index + delta_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if (next_row, next_col) in seen or rows[next_row][next_col] != 0:
                continue
            seen.add((next_row, next_col))
            queue.append((next_row, next_col))

    return freeze_grid(rows)


def _propagate_payload_color_cells(
    grid: Grid,
    row_separators: tuple[int, ...],
    col_separators: tuple[int, ...],
) -> Grid:
    cells = split_grid_by_separators(grid, row_separators, col_separators)
    if not cells or not cells[0]:
        return grid

    scaffold_color = _separator_color(grid, row_separators, col_separators)
    result = [list(row) for row in cells]
    payload_positions: dict[int, list[tuple[int, int]]] = {}
    for row_index, col_index, cell in flatten_cells(cells):
        color = _payload_color(cell, scaffold_color)
        if color is None:
            continue
        payload_positions.setdefault(color, []).append((row_index, col_index))

    for color, positions in payload_positions.items():
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for row_index, col_index in positions:
            by_row.setdefault(row_index, []).append(col_index)
            by_col.setdefault(col_index, []).append(row_index)

        for row_index, cols in by_row.items():
            if len(cols) < 2:
                continue
            for col_index in range(min(cols), max(cols) + 1):
                cell = cells[row_index][col_index]
                if _supports_payload_color(cell, scaffold_color, color):
                    result[row_index][col_index] = _fill_topleft_region(cell, color)

        for col_index, rows in by_col.items():
            if len(rows) < 2:
                continue
            for row_index in range(min(rows), max(rows) + 1):
                cell = cells[row_index][col_index]
                if _supports_payload_color(cell, scaffold_color, color):
                    result[row_index][col_index] = _fill_topleft_region(cell, color)

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
        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []

        exact_match_executor = (
            lambda grid, row_separators=row_separators, col_separators=col_separators: _propagate_matching_cells(
                grid,
                row_separators,
                col_separators,
            )
        )
        if exact_match_executor(first_example.input) == first_example.output:
            candidate_specs.append(
                (
                    "cross-ref-separator-propagation",
                    {
                        "type": "separator_propagation",
                        "variant": "exact_cell_match",
                        "row_separators": row_separators,
                        "col_separators": col_separators,
                    },
                    exact_match_executor,
                    5,
                )
            )

        payload_executor = (
            lambda grid, row_separators=row_separators, col_separators=col_separators: _propagate_payload_color_cells(
                grid,
                row_separators,
                col_separators,
            )
        )
        if payload_executor(first_example.input) == first_example.output:
            candidate_specs.append(
                (
                    "cross-ref-separator-payload-propagation",
                    {
                        "type": "separator_propagation",
                        "variant": "payload_color",
                        "row_separators": row_separators,
                        "col_separators": col_separators,
                    },
                    payload_executor,
                    6,
                )
            )

        if not candidate_specs:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("no separator propagation rule matched the first example",),
            )

        candidates = []
        generated = 0
        for name, semantics, executor, complexity in candidate_specs:
            generated += 1
            evaluation = _emit_program(
                context=context,
                strategy_name=self.name,
                name=name,
                semantics=semantics,
                executor=executor,
                complexity=complexity,
            )
            if evaluation is None:
                break
            candidates.append(evaluation)

        return context.finish_strategy(
            name=self.name,
            status="ok" if candidates else "budget_exhausted",
            generated=generated if candidates else 0,
            candidates=candidates,
        )
