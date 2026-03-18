from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _infer_ring_recolor_spec(task: ArcTask) -> tuple[int, int] | None:
    source_colors: set[int] = set()
    fill_colors: set[int] = set()
    for example in task.train:
        if grid_shape(example.input) != grid_shape(example.output):
            return None
        height = len(example.input)
        width = len(example.input[0])
        for row_index in range(height):
            for col_index in range(width):
                if example.input[row_index][col_index] != example.output[row_index][col_index]:
                    source_colors.add(example.input[row_index][col_index])
                    fill_colors.add(example.output[row_index][col_index])

    if len(source_colors) != 1 or len(fill_colors) != 1:
        return None
    return next(iter(source_colors)), next(iter(fill_colors))


def _is_rectangular_ring(
    grid: Grid,
    *,
    color: int,
    bbox: tuple[int, int, int, int],
) -> bool:
    row_start, col_start, row_end, col_end = bbox
    if row_end - row_start + 1 < 3 or col_end - col_start + 1 < 3:
        return False

    for row_index in range(row_start, row_end + 1):
        for col_index in range(col_start, col_end + 1):
            value = grid[row_index][col_index]
            is_border = (
                row_index in (row_start, row_end)
                or col_index in (col_start, col_end)
            )
            if is_border and value != color:
                return False
            if not is_border and value != 0:
                return False
    return True


def _recolor_rectangular_rings(
    grid: Grid,
    *,
    source_color: int,
    fill_color: int,
) -> Grid:
    if not grid:
        return ()

    rows = [list(row) for row in grid]
    changed = False
    for component in connected_components(grid, bg_color=0):
        if component.color != source_color:
            continue
        if not _is_rectangular_ring(grid, color=source_color, bbox=component.bbox):
            continue
        for row_index, col_index in component.pixels:
            rows[row_index][col_index] = fill_color
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
class RectangularRingRecolorStrategy:
    name: str = "arc-rectangular-ring-recolor"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        spec = _infer_ring_recolor_spec(context.task)
        if spec is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("rectangular ring recolor pattern not detected",),
            )

        source_color, fill_color = spec
        first_example = context.task.train[0]
        if _recolor_rectangular_rings(
            first_example.input,
            source_color=source_color,
            fill_color=fill_color,
        ) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("rectangular ring recolor did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name=f"recolor-rectangular-rings-{source_color}-to-{fill_color}",
            semantics={
                "type": "rectangular_ring_recolor",
                "source_color": source_color,
                "fill_color": fill_color,
            },
            executor=lambda grid: _recolor_rectangular_rings(
                grid,
                source_color=source_color,
                fill_color=fill_color,
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
