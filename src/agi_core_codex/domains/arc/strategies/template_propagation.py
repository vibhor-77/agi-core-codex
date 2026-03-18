from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _component_bboxes(
    grid: Grid,
) -> tuple[tuple[int, tuple[tuple[int, int], ...], tuple[int, int, int, int]], ...]:
    if not grid:
        return ()

    height = len(grid)
    width = len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[tuple[int, tuple[tuple[int, int], ...], tuple[int, int, int, int]]] = []
    for row_index in range(height):
        for col_index in range(width):
            color = grid[row_index][col_index]
            if color == 0 or (row_index, col_index) in visited:
                continue

            stack = [(row_index, col_index)]
            pixels = []
            visited.add((row_index, col_index))
            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row, current_col))
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in visited:
                        continue
                    if grid[next_row][next_col] != color:
                        continue
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

            rows = [row for row, _ in pixels]
            cols = [col for _, col in pixels]
            components.append(
                (
                    color,
                    tuple(sorted(pixels)),
                    (min(rows), min(cols), max(rows), max(cols)),
                )
            )
    return tuple(components)


def _bbox(pixels: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in pixels]
    cols = [col for _, col in pixels]
    return (min(rows), min(cols), max(rows), max(cols))


def _subset_offsets(
    seed_pixels: tuple[tuple[int, int], ...],
    template_relative: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    template_set = set(template_relative)
    offsets: list[tuple[int, int]] = []
    for seed_row, seed_col in seed_pixels:
        for template_row, template_col in template_relative:
            offset = (seed_row - template_row, seed_col - template_col)
            if all((row - offset[0], col - offset[1]) in template_set for row, col in seed_pixels):
                if offset not in offsets:
                    offsets.append(offset)
    return tuple(offsets)


def _propagate_largest_template(grid: Grid) -> Grid:
    if not grid:
        return ()

    counts = Counter(cell for row in grid for cell in row if cell != 0)
    if len(counts) < 2:
        return grid

    components = _component_bboxes(grid)
    template_color = counts.most_common(1)[0][0]
    template_pixels = tuple(
        sorted(
            pixel
            for color, pixels, _ in components
            if color == template_color
            for pixel in pixels
        )
    )
    if not template_pixels:
        return grid

    template_bbox = _bbox(template_pixels)
    template_row0, template_col0, template_row1, template_col1 = template_bbox
    template_relative = tuple(
        sorted((row - template_row0, col - template_col0) for row, col in template_pixels)
    )
    template_height = template_row1 - template_row0 + 1
    template_width = template_col1 - template_col0 + 1
    template_center = (
        (template_row0 + template_row1) / 2,
        (template_col0 + template_col1) / 2,
    )

    height = len(grid)
    width = len(grid[0])
    rows = [list(row) for row in grid]
    colors = sorted({color for color, _, _ in components if color != template_color})
    for color in colors:
        color_components = [
            (pixels, component_bbox)
            for component_color, pixels, component_bbox in components
            if component_color == color
        ]
        if not color_components:
            continue

        union_pixels = tuple(sorted(pixel for pixels, _ in color_components for pixel in pixels))
        seeds: list[tuple[tuple[int, int], ...], tuple[int, int, int, int]]
        if len(color_components) > 1 and _subset_offsets(union_pixels, template_relative):
            seeds = [(union_pixels, _bbox(union_pixels))]
        else:
            seeds = color_components

        for seed_pixels, seed_bbox in seeds:
            offsets = _subset_offsets(seed_pixels, template_relative)
            if not offsets:
                continue

            seed_row0, seed_col0, seed_row1, seed_col1 = seed_bbox
            seed_center = ((seed_row0 + seed_row1) / 2, (seed_col0 + seed_col1) / 2)
            delta_row = 0 if seed_center[0] == template_center[0] else (1 if seed_center[0] > template_center[0] else -1)
            delta_col = 0 if seed_center[1] == template_center[1] else (1 if seed_center[1] > template_center[1] else -1)
            if delta_row == 0 and delta_col == 0:
                continue

            step = (
                delta_row * (template_height + 1) if delta_row else 0,
                delta_col * (template_width + 1) if delta_col else 0,
            )

            def _score(offset: tuple[int, int]) -> float:
                center = (offset[0] + (template_height - 1) / 2, offset[1] + (template_width - 1) / 2)
                return ((center[0] - template_center[0]) * delta_row) + (
                    (center[1] - template_center[1]) * delta_col
                )

            current = max(offsets, key=_score)
            while True:
                projected = [(current[0] + row, current[1] + col) for row, col in template_relative]
                in_bounds = [
                    (row_index, col_index)
                    for row_index, col_index in projected
                    if 0 <= row_index < height and 0 <= col_index < width
                ]
                if not in_bounds:
                    break
                for row_index, col_index in in_bounds:
                    rows[row_index][col_index] = color
                current = (current[0] + step[0], current[1] + step[1])

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
class TemplatePropagationStrategy:
    name: str = "arc-template-propagation"
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
                notes=("template propagation only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _propagate_largest_template(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("largest-template propagation did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="propagate-largest-template",
            semantics={"type": "template_propagation", "template": "largest-nonzero-color"},
            executor=_propagate_largest_template,
            complexity=6,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
