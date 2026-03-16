from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, freeze_grid, grid_shape


def _infer_mapping(task: ArcTask) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    saw_change = False
    for example in task.train:
        if grid_shape(example.input) != grid_shape(example.output):
            return None
        for input_row, output_row in zip(example.input, example.output, strict=True):
            for input_cell, output_cell in zip(input_row, output_row, strict=True):
                current = mapping.get(input_cell)
                if current is None:
                    mapping[input_cell] = output_cell
                elif current != output_cell:
                    return None
                if input_cell != output_cell:
                    saw_change = True
    if not saw_change:
        return None
    return mapping


def _apply_mapping(grid, mapping: dict[int, int]):
    return freeze_grid(
        tuple(mapping.get(cell, cell) for cell in row)
        for row in grid
    )


@dataclass(frozen=True)
class ColorMapStrategy:
    name: str = "arc-color-map"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        mapping = _infer_mapping(context.task)
        if mapping is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("no consistent input-output color map was found",),
            )

        program = make_arc_program(
            name="consistent-color-map",
            semantics={"type": "color_map", "mapping": mapping},
            executor=lambda grid, mapping=mapping: _apply_mapping(grid, mapping),
            complexity=max(2, len(mapping)),
        )
        candidate = context.evaluate(program, self.name)
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )

