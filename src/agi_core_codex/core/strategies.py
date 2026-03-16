from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import CostModel, StrategyResult


@dataclass(frozen=True)
class GrammarPrimitiveStrategy:
    name: str = "grammar-primitives"
    domain: str = "*"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return True

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(
                name=self.name,
                status="budget_exhausted",
                notes=("budget exhausted before grammar enumeration",),
            )

        candidates = []
        generated = 0
        for program in context.grammar.enumerate_primitives(context.task):
            generated += 1
            evaluation = context.evaluate(program, self.name)
            if evaluation is None:
                break
            candidates.append(evaluation)

        return context.finish_strategy(
            name=self.name,
            status="ok" if candidates else "no_candidates",
            generated=generated,
            candidates=candidates,
        )


@dataclass(frozen=True)
class LibraryReplayStrategy:
    name: str = "library-replay"
    domain: str = "*"
    recall_limit: int = 8
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return True

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(
                name=self.name,
                status="budget_exhausted",
                notes=("budget exhausted before library replay",),
            )

        entries = context.memory.recall(context.domain, limit=self.recall_limit)
        candidates = []
        for entry in entries:
            evaluation = context.evaluate(
                entry.program,
                self.name,
                metadata={"origin_task_key": entry.origin_task_key},
            )
            if evaluation is None:
                break
            candidates.append(evaluation)

        return context.finish_strategy(
            name=self.name,
            status="ok" if candidates else "no_library_hits",
            generated=len(entries),
            candidates=candidates,
        )

