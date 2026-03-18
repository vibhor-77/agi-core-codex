from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _one_components(grid: Grid) -> tuple[tuple[tuple[int, int], ...], ...]:
    height, width = grid_shape(grid)
    visited: set[tuple[int, int]] = set()
    components: list[tuple[tuple[int, int], ...]] = []
    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] != 1 or (row_index, col_index) in visited:
                continue
            queue = deque([(row_index, col_index)])
            visited.add((row_index, col_index))
            pixels: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                pixels.append((current_row, current_col))
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in visited or grid[next_row][next_col] != 1:
                        continue
                    visited.add((next_row, next_col))
                    queue.append((next_row, next_col))
            components.append(tuple(sorted(pixels)))
    return tuple(components)


def _fill_open_frames(grid: Grid) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    rows = [list(row) for row in grid]

    for component in _one_components(grid):
        component_rows = [row_index for row_index, _ in component]
        component_cols = [col_index for _, col_index in component]
        row_start = min(component_rows)
        col_start = min(component_cols)
        row_end = max(component_rows)
        col_end = max(component_cols)
        if row_end - row_start < 2 or col_end - col_start < 2 or row_start == 0:
            continue

        pixels = set(component)
        if any((row_end, col_index) not in pixels for col_index in range(col_start, col_end + 1)):
            continue
        if any((row_index, col_start) not in pixels for row_index in range(row_start, row_end + 1)):
            continue
        if any((row_index, col_end) not in pixels for row_index in range(row_start, row_end + 1)):
            continue
        if (row_start, col_start) not in pixels or (row_start, col_end) not in pixels:
            continue

        top_gap = [col_index for col_index in range(col_start + 1, col_end) if (row_start, col_index) not in pixels]
        if not top_gap:
            continue

        marker_cells = [
            (row_index, col_index, grid[row_index][col_index])
            for row_index in range(row_start, row_end + 1)
            for col_index in range(col_start, col_end + 1)
            if grid[row_index][col_index] not in (0, 1)
        ]
        if len(marker_cells) != 1:
            continue

        _, _, marker_color = marker_cells[0]
        for col_index in range(col_start, col_end + 1):
            rows[row_start - 1][col_index] = marker_color
        for row_index in range(row_start, row_end):
            for col_index in range(col_start + 1, col_end):
                if grid[row_index][col_index] != 1:
                    rows[row_index][col_index] = marker_color

    return freeze_grid(rows)


@dataclass(frozen=True)
class OpenFrameFillStrategy:
    name: str = "arc-open-frame-fill"
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
                notes=("open-frame fill only applies to same-shape tasks",),
            )

        if any(_fill_open_frames(example.input) != example.output for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("open-frame fill did not match every train example",),
            )

        program = make_arc_program(
            name="fill-open-frames-with-marker-color",
            semantics={"type": "fill_open_frames"},
            executor=_fill_open_frames,
            complexity=2,
        )
        candidate = context.evaluate(program, self.name)
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
