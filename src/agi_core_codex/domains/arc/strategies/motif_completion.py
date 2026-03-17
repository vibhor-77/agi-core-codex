from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _multicolor_components8(grid: Grid) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[tuple[tuple[int, int, int], ...]] = []

    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] == 0 or (row_index, col_index) in visited:
                continue

            stack = [(row_index, col_index)]
            visited.add((row_index, col_index))
            cells = []
            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col, grid[current_row][current_col]))
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if grid[next_row][next_col] == 0 or (next_row, next_col) in visited:
                            continue
                        visited.add((next_row, next_col))
                        stack.append((next_row, next_col))
            components.append(tuple(sorted(cells)))

    return tuple(components)


def _template_specs(
    grid: Grid,
) -> tuple[
    tuple[
        tuple[tuple[int, int, int], ...],
        tuple[int, int, int],
        int,
        int,
        int,
    ],
    ...,
]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    specs = []
    for component in _multicolor_components8(grid):
        if len(component) < 3:
            continue

        row_start = min(row_index for row_index, _, _ in component)
        col_start = min(col_index for _, col_index, _ in component)
        relative = tuple(
            sorted(
                (row_index - row_start, col_index - col_start, color)
                for row_index, col_index, color in component
            )
        )
        if relative in seen:
            continue

        counts = Counter(color for _, _, color in relative)
        unique_colors = [color for color, count in counts.items() if count == 1]
        if len(unique_colors) != 1:
            continue

        anchor_color = unique_colors[0]
        anchor = next(cell for cell in relative if cell[2] == anchor_color)
        max_row = max(row_index for row_index, _, _ in relative)
        max_col = max(col_index for _, col_index, _ in relative)
        specs.append((relative, anchor, anchor_color, max_row, max_col))
        seen.add(relative)

    return tuple(specs)


def _complete_motifs(grid: Grid) -> Grid:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    result = [list(row) for row in grid]

    for relative, anchor, anchor_color, max_row, max_col in _template_specs(grid):
        relative_set = set(relative)
        anchor_row, anchor_col, _ = anchor

        anchor_positions = [
            (row_index, col_index)
            for row_index in range(height)
            for col_index in range(width)
            if grid[row_index][col_index] == anchor_color
        ]
        for row_index, col_index in anchor_positions:
            start_row = row_index - anchor_row
            start_col = col_index - anchor_col
            if not (
                0 <= start_row
                and 0 <= start_col
                and start_row + max_row < height
                and start_col + max_col < width
            ):
                continue

            observed = []
            valid = True
            for scan_row in range(start_row, start_row + max_row + 1):
                for scan_col in range(start_col, start_col + max_col + 1):
                    value = grid[scan_row][scan_col]
                    if value == 0:
                        continue
                    relative_cell = (scan_row - start_row, scan_col - start_col, value)
                    if relative_cell not in relative_set:
                        valid = False
                        break
                    observed.append(relative_cell)
                if not valid:
                    break
            if not valid or len(observed) == len(relative):
                continue

            missing = [
                (start_row + rel_row, start_col + rel_col, color)
                for rel_row, rel_col, color in relative
                if (rel_row, rel_col, color) not in observed
            ]
            if any(
                result[missing_row][missing_col] not in (0, color)
                for missing_row, missing_col, color in missing
            ):
                continue
            for missing_row, missing_col, color in missing:
                result[missing_row][missing_col] = color

        non_anchor = [cell for cell in relative if cell != anchor]
        for start_row in range(height - max_row):
            for start_col in range(width - max_col):
                if grid[start_row + anchor_row][start_col + anchor_col] != 0:
                    continue

                valid = True
                for rel_row, rel_col, color in non_anchor:
                    if grid[start_row + rel_row][start_col + rel_col] != color:
                        valid = False
                        break
                if not valid:
                    continue

                for scan_row in range(start_row, start_row + max_row + 1):
                    for scan_col in range(start_col, start_col + max_col + 1):
                        value = grid[scan_row][scan_col]
                        if value == 0:
                            continue
                        relative_cell = (scan_row - start_row, scan_col - start_col, value)
                        if relative_cell not in relative_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue

                result[start_row + anchor_row][start_col + anchor_col] = anchor_color

    return freeze_grid(result)


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
class MotifCompletionStrategy:
    name: str = "arc-motif-completion"
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
                notes=("motif completion only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _complete_motifs(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("anchor-based motif completion did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="complete-anchor-motifs",
            semantics={"type": "motif_completion"},
            executor=_complete_motifs,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
