from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _project_holes_along_short_axis(grid: Grid) -> Grid:
    if not grid:
        return ()

    rows = [list(row) for row in grid]
    changed = False
    for component in connected_components(grid, bg_color=0):
        row_start, col_start, row_end, col_end = component.bbox
        height = row_end - row_start + 1
        width = col_end - col_start + 1
        if height == width:
            continue

        holes: list[tuple[int, int]] = []
        valid = True
        for row_index in range(row_start, row_end + 1):
            for col_index in range(col_start, col_end + 1):
                value = grid[row_index][col_index]
                if value == 0:
                    holes.append((row_index, col_index))
                elif value != component.color:
                    valid = False
                    break
            if not valid:
                break
        if not valid or not holes:
            continue

        if width > height:
            for hole_col in sorted({col_index for _, col_index in holes}):
                for row_index in range(row_start, row_end + 1):
                    if rows[row_index][hole_col] != 0:
                        rows[row_index][hole_col] = 0
                        changed = True
        else:
            for hole_row in sorted({row_index for row_index, _ in holes}):
                for col_index in range(col_start, col_end + 1):
                    if rows[hole_row][col_index] != 0:
                        rows[hole_row][col_index] = 0
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
class HoleProjectionStrategy:
    name: str = "arc-hole-projection"
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
                notes=("hole projection only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _project_holes_along_short_axis(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("short-axis hole projection did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="project-holes-along-short-axis",
            semantics={"type": "hole_projection"},
            executor=_project_holes_along_short_axis,
            complexity=5,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
