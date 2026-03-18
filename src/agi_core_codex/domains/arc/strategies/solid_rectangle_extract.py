from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _component_mask(grid: Grid, *, color: int, bbox: tuple[int, int, int, int]) -> list[list[int]]:
    row_start, col_start, row_end, col_end = bbox
    return [
        [1 if grid[row_index][col_index] == color else 0 for col_index in range(col_start, col_end + 1)]
        for row_index in range(row_start, row_end + 1)
    ]


def _maximal_solid_rectangles(mask: list[list[int]]) -> tuple[tuple[int, int, int, int], ...]:
    if not mask or not mask[0]:
        return ()

    height = len(mask)
    width = len(mask[0])
    rectangles: list[tuple[int, int, int, int]] = []
    for top in range(height):
        common = mask[top][:]
        for bottom in range(top, height):
            if bottom > top:
                common = [left & right for left, right in zip(common, mask[bottom])]
            if bottom - top + 1 < 2:
                continue

            run_start: int | None = None
            for col_index in range(width + 1):
                value = common[col_index] if col_index < width else 0
                if value:
                    if run_start is None:
                        run_start = col_index
                    continue
                if run_start is not None and col_index - run_start >= 2:
                    rectangles.append((top, run_start, bottom, col_index - 1))
                run_start = None

    maximal = []
    for rect in rectangles:
        if any(
            other != rect
            and other[0] <= rect[0]
            and other[1] <= rect[1]
            and other[2] >= rect[2]
            and other[3] >= rect[3]
            for other in rectangles
        ):
            continue
        maximal.append(rect)
    return tuple(maximal)


def _extract_solid_rectangles(grid: Grid) -> Grid:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    rows = [[0] * width for _ in range(height)]
    changed = False
    for component in connected_components(grid, bg_color=0):
        row_start, col_start, row_end, col_end = component.bbox
        if row_end - row_start + 1 < 2 or col_end - col_start + 1 < 2:
            continue

        mask = _component_mask(grid, color=component.color, bbox=component.bbox)
        rectangles = _maximal_solid_rectangles(mask)
        for top, left, bottom, right in rectangles:
            for row_index in range(row_start + top, row_start + bottom + 1):
                for col_index in range(col_start + left, col_start + right + 1):
                    rows[row_index][col_index] = component.color

        if rectangles:
            for row_index, col_index in component.pixels:
                if rows[row_index][col_index] != grid[row_index][col_index]:
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
class SolidRectangleExtractStrategy:
    name: str = "arc-solid-rectangle-extract"
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
                notes=("solid rectangle extraction only applies to same-shape tasks",),
            )

        first_example = context.task.train[0]
        if _extract_solid_rectangles(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("solid rectangle extraction did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="keep-max-solid-rectangles",
            semantics={"type": "solid_rectangle_extract"},
            executor=_extract_solid_rectangles,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
