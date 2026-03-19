from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.interfaces import (
    CostModel,
    HypothesisFamily,
    StrategyResult,
    Verifier,
)


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


@dataclass(frozen=True)
class HypothesisFamilyStrategy:
    family: HypothesisFamily
    verifier: Verifier

    @property
    def name(self) -> str:
        return f"transfer:{self.family.name}"

    @property
    def domain(self) -> str:
        return self.family.domain

    @property
    def cost_model(self) -> CostModel:
        return self.family.cost_model

    def applies(self, task: Any) -> bool:
        return True

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        representation = self.family.build_representation(context.task, context.environment)
        hypotheses = tuple(self.family.propose(context.task, representation))
        candidates = []
        notes: list[str] = []

        for hypothesis in hypotheses:
            verification = self.verifier.verify(context.task, hypothesis, representation)
            if verification.compiled_program is None:
                if verification.failure_reason is not None:
                    notes.append(f"{hypothesis.id}:{verification.failure_reason}")
                continue

            metadata = dict(verification.compiled_program.metadata)
            metadata.update(
                {
                    "family_name": verification.compiled_program.family_name,
                    "representation_summary": verification.compiled_program.representation_summary,
                    "genericity_score": verification.compiled_program.genericity_score,
                    "transfer_proxy_score": verification.compiled_program.transfer_proxy_score,
                    "hypothesis_id": hypothesis.id,
                    "hypothesis_description": hypothesis.description,
                    "verification_fail_reason": verification.failure_reason,
                }
            )
            evaluation = context.evaluate(
                verification.compiled_program.handle,
                self.name,
                metadata=metadata,
            )
            if evaluation is None:
                break
            candidates.append(evaluation)

        status = "ok" if candidates else "no_candidates"
        if hypotheses and not candidates and not notes:
            status = "verified_no_matches"
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=len(hypotheses),
            candidates=candidates,
            notes=tuple(notes[:8]),
        )
