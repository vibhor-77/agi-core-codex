from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, grid_cell_count


@dataclass(frozen=True)
class ConstantOutputStrategy:
    name: str = "arc-constant-output"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        outputs = {example.output for example in context.task.train}
        if len(outputs) != 1:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("training outputs are not constant",),
            )

        output = next(iter(outputs))
        program = make_arc_program(
            name="constant-output",
            semantics={"type": "constant_output", "grid": output},
            executor=lambda grid, output=output: output,
            complexity=max(2, grid_cell_count(output)),
        )
        candidate = context.evaluate(program, self.name)
        candidates = [] if candidate is None else [candidate]
        status = "ok" if candidate is not None else "budget_exhausted"
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=1 if candidate is not None else 0,
            candidates=candidates,
        )

