from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _recolor_alternating_diagonal_cells(grid: Grid) -> Grid:
    if not grid:
        return ()

    nonzero = [
        (row_index, col_index, value)
        for row_index, row in enumerate(grid)
        for col_index, value in enumerate(row)
        if value != 0
    ]
    colors = {value for _, _, value in nonzero}
    if len(colors) != 1:
        return grid

    positions = {(row_index, col_index) for row_index, col_index, _ in nonzero}
    rows = [list(row) for row in grid]
    for row_index, col_index in sorted(positions):
        if (row_index - 1, col_index - 1) in positions:
            continue

        chain_row = row_index
        chain_col = col_index
        chain_index = 0
        while (chain_row, chain_col) in positions:
            if chain_index % 2 == 1:
                rows[chain_row][chain_col] = 4
            chain_row += 1
            chain_col += 1
            chain_index += 1

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
class AlternatingDiagonalRecolorStrategy:
    name: str = "arc-alternating-diagonal-recolor"
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
                notes=("alternating diagonal recolor only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        if _recolor_alternating_diagonal_cells(first_example.input) != first_example.output:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("alternating down-right diagonal recolor did not match the first example",),
            )

        candidate = _emit_program(
            context=context,
            strategy_name=self.name,
            name="recolor-odd-diagonal-chain-cells",
            semantics={"type": "alternating_diagonal_recolor", "replacement_color": 4},
            executor=_recolor_alternating_diagonal_cells,
            complexity=3,
        )
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )
