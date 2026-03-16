from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, freeze_grid, grid_shape


def _infer_absolute_patch(task: ArcTask) -> tuple[tuple[int, int, int], ...] | None:
    reference: tuple[tuple[int, int, int], ...] | None = None
    for example in task.train:
        if grid_shape(example.input) != grid_shape(example.output):
            return None
        edits: list[tuple[int, int, int]] = []
        for row_index, (input_row, output_row) in enumerate(zip(example.input, example.output, strict=True)):
            for col_index, (input_cell, output_cell) in enumerate(zip(input_row, output_row, strict=True)):
                if input_cell != output_cell:
                    edits.append((row_index, col_index, output_cell))
        edits_tuple = tuple(edits)
        if reference is None:
            reference = edits_tuple
            continue
        if reference != edits_tuple:
            return None
    if not reference:
        return None
    return reference


def _apply_patch(grid, patch: tuple[tuple[int, int, int], ...]):
    rows = [list(row) for row in grid]
    for row_index, col_index, value in patch:
        if row_index >= len(rows) or col_index >= len(rows[row_index]):
            raise ValueError("patch index out of bounds")
        rows[row_index][col_index] = value
    return freeze_grid(rows)


@dataclass(frozen=True)
class AbsolutePatchStrategy:
    name: str = "arc-absolute-patch"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        patch = _infer_absolute_patch(context.task)
        if patch is None:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("no task-stable absolute patch was found",),
            )

        program = make_arc_program(
            name="absolute-patch",
            semantics={
                "type": "absolute_patch",
                "task_id": context.task.task_id,
                "patch": patch,
            },
            executor=lambda grid, patch=patch: _apply_patch(grid, patch),
            complexity=max(2, len(patch) + 1),
        )
        candidate = context.evaluate(program, self.name)
        candidates = [] if candidate is None else [candidate]
        return context.finish_strategy(
            name=self.name,
            status="ok" if candidate is not None else "budget_exhausted",
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )

