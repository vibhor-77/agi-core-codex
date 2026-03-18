from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _zero_block_families(grid: Grid) -> tuple[tuple[tuple[int, int], ...], ...]:
    height, width = grid_shape(grid)
    blocks = {
        (row_index, col_index)
        for row_index in range(height - 1)
        for col_index in range(width - 1)
        if all(grid[row_index + delta_row][col_index + delta_col] == 0 for delta_row in (0, 1) for delta_col in (0, 1))
    }

    seen: set[tuple[int, int]] = set()
    families: list[tuple[tuple[int, int], ...]] = []
    for block in sorted(blocks):
        if block in seen:
            continue
        stack = [block]
        seen.add(block)
        family = []
        while stack:
            row_index, col_index = stack.pop()
            family.append((row_index, col_index))
            for next_block in (
                (row_index - 1, col_index),
                (row_index + 1, col_index),
                (row_index, col_index - 1),
                (row_index, col_index + 1),
            ):
                if next_block in blocks and next_block not in seen:
                    seen.add(next_block)
                    stack.append(next_block)
        families.append(tuple(sorted(family)))
    return tuple(families)


def _family_cover(family: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cover = {
        (row_index + delta_row, col_index + delta_col)
        for row_index, col_index in family
        for delta_row in (0, 1)
        for delta_col in (0, 1)
    }
    return tuple(sorted(cover))


def _largest_zero_rectangle(
    grid: Grid,
    cover: tuple[tuple[int, int], ...],
) -> tuple[int, int, int, int] | None:
    if not cover:
        return None

    cover_set = set(cover)
    row_values = sorted({row_index for row_index, _ in cover})
    col_values = sorted({col_index for _, col_index in cover})
    best: tuple[int, int, int, int, int, int, int, int, int] | None = None
    for row_start in row_values:
        for row_end in row_values:
            if row_end < row_start:
                continue
            for col_start in col_values:
                for col_end in col_values:
                    if col_end < col_start:
                        continue

                    rectangle = [
                        (row_index, col_index)
                        for row_index in range(row_start, row_end + 1)
                        for col_index in range(col_start, col_end + 1)
                    ]
                    if not all((row_index, col_index) in cover_set and grid[row_index][col_index] == 0 for row_index, col_index in rectangle):
                        continue

                    area = len(rectangle)
                    candidate = (
                        area,
                        -(row_end - row_start + 1),
                        -(col_end - col_start + 1),
                        -row_start,
                        -col_start,
                        row_start,
                        row_end,
                        col_start,
                        col_end,
                    )
                    if best is None or candidate > best:
                        best = candidate

    if best is None:
        return None
    _, _, _, _, _, row_start, row_end, col_start, col_end = best
    return (row_start, row_end, col_start, col_end)


def _fill_zero_rectangle_families(grid: Grid) -> Grid:
    if not grid:
        return ()

    rows = [list(row) for row in grid]
    for family in _zero_block_families(grid):
        rectangle = _largest_zero_rectangle(grid, _family_cover(family))
        if rectangle is None:
            continue
        row_start, row_end, col_start, col_end = rectangle
        for row_index in range(row_start, row_end + 1):
            for col_index in range(col_start, col_end + 1):
                rows[row_index][col_index] = 2
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
class ZeroRectangleFamilyFillStrategy:
    name: str = "arc-zero-rectangle-family-fill"
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
                notes=("zero rectangle family fill only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _fill_zero_rectangle_families(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("largest zero-rectangle family fill did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="fill-largest-zero-rectangle-families",
            semantics={"type": "zero_rectangle_family_fill", "fill_color": 2},
            executor=_fill_zero_rectangle_families,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
