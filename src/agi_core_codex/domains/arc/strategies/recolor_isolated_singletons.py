from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _recolor_isolated_twos(grid: Grid) -> Grid:
    if not grid:
        return ()

    height, width = grid_shape(grid)
    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] != 2:
                continue
            neighbors = 0
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row_index + delta_row
                next_col = col_index + delta_col
                if 0 <= next_row < height and 0 <= next_col < width and grid[next_row][next_col] == 2:
                    neighbors += 1
            if neighbors == 0:
                rows[row_index][col_index] = 1
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
class RecolorIsolatedSingletonsStrategy:
    name: str = "arc-recolor-isolated-singletons"
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
                notes=("isolated singleton recolor only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _recolor_isolated_twos(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("isolated singleton recolor did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="recolor-isolated-twos-to-ones",
            semantics={"type": "recolor_isolated_twos", "source_color": 2, "target_color": 1},
            executor=_recolor_isolated_twos,
            complexity=2,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
