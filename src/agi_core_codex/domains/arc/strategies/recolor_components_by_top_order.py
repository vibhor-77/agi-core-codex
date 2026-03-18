from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _recolor_components_by_top_order(grid: Grid) -> Grid:
    if not grid:
        return ()

    components = connected_components(grid, bg_color=0)
    if not components:
        return grid

    if len({component.color for component in components}) != 1 or len(components) > 9:
        return grid

    ordered_components = sorted(
        components,
        key=lambda component: (component.bbox[0], component.bbox[1], component.bbox[2], component.bbox[3]),
    )
    rows = [list(row) for row in grid]
    for color_index, component in enumerate(ordered_components, start=1):
        for row_index, col_index in component.pixels:
            rows[row_index][col_index] = color_index
    return freeze_grid(rows)


@dataclass(frozen=True)
class RecolorComponentsByTopOrderStrategy:
    name: str = "arc-recolor-components-by-top-order"
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
                notes=("component top-order recolor only applies to same-shape tasks",),
            )

        if any(
            _recolor_components_by_top_order(example.input) != example.output
            for example in context.task.train
        ):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("component top-order recolor did not match every train example",),
            )

        program = make_arc_program(
            name="recolor-components-by-top-order",
            semantics={"type": "recolor_components_by_top_order"},
            executor=_recolor_components_by_top_order,
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
