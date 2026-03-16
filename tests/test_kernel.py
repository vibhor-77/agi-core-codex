from __future__ import annotations

from dataclasses import dataclass, field

from agi_core_codex.core.interfaces import CostModel, ProgramHandle, SearchBudget
from agi_core_codex.core.kernel import SearchKernel
from agi_core_codex.core.memory import InMemoryLibrary
from agi_core_codex.core.strategies import GrammarPrimitiveStrategy
from agi_core_codex.domains.arc.environment import ArcEnvironment
from agi_core_codex.domains.arc.scorer import ArcScorer
from agi_core_codex.domains.arc.types import ArcExample, ArcTask, ArcTestCase, freeze_grid


def _identity_task() -> ArcTask:
    return ArcTask(
        task_id="identity",
        train=(
            ArcExample(input=freeze_grid([[1, 0], [0, 1]]), output=freeze_grid([[1, 0], [0, 1]])),
        ),
        test=(ArcTestCase(input=freeze_grid([[2, 2], [2, 2]]), output=freeze_grid([[2, 2], [2, 2]])),),
    )


class SingleProgramGrammar:
    domain = "arc"

    def __init__(self, program: ProgramHandle) -> None:
        self._program = program

    def primitive_count(self, task=None) -> int:
        return 1

    def enumerate_primitives(self, task):
        return (self._program,)


@dataclass(frozen=True)
class ForeignStrategy:
    name: str = "foreign"
    domain: str = "zork"
    cost_model: CostModel = field(default_factory=CostModel.zero)

    def applies(self, task) -> bool:
        return True

    def run(self, context):
        raise AssertionError("domain-mismatched strategy should not run")


def test_domain_mismatch_strategy_is_skipped() -> None:
    identity = ProgramHandle(
        id="identity",
        name="identity",
        domain="arc",
        executor=lambda grid: grid,
        cost=CostModel(complexity=1),
    )
    kernel = SearchKernel((ForeignStrategy(), GrammarPrimitiveStrategy()))
    report = kernel.run(
        task=_identity_task(),
        environment=ArcEnvironment(),
        grammar=SingleProgramGrammar(identity),
        scorer=ArcScorer(),
        memory=InMemoryLibrary(),
        budget=SearchBudget(max_evaluations=3, max_cell_evaluations=30),
        seed=0,
    )

    assert report.strategy_results[0].status == "domain_mismatch"
    assert report.best_candidate is not None
    assert report.best_candidate.score.train_exact is True


def test_explicit_failures_do_not_fall_back_to_identity() -> None:
    failing = ProgramHandle(
        id="explode",
        name="explode",
        domain="arc",
        executor=lambda grid: (_ for _ in ()).throw(ValueError("boom")),
        cost=CostModel(complexity=1),
    )
    kernel = SearchKernel((GrammarPrimitiveStrategy(),))
    report = kernel.run(
        task=_identity_task(),
        environment=ArcEnvironment(),
        grammar=SingleProgramGrammar(failing),
        scorer=ArcScorer(),
        memory=InMemoryLibrary(),
        budget=SearchBudget(max_evaluations=3, max_cell_evaluations=30),
        seed=0,
    )

    assert report.best_candidate is not None
    assert report.best_candidate.score.train_exact is False
    assert report.best_candidate.score.train_accuracy == 0.0
    assert report.best_candidate.score.failure_count > 0


def test_budget_accounting_caps_evaluations() -> None:
    programs = tuple(
        ProgramHandle(
            id=f"identity-{index}",
            name=f"identity-{index}",
            domain="arc",
            executor=lambda grid: grid,
            cost=CostModel(complexity=1),
        )
        for index in range(5)
    )

    class ManyProgramGrammar:
        domain = "arc"

        def primitive_count(self, task=None) -> int:
            return len(programs)

        def enumerate_primitives(self, task):
            return programs

    kernel = SearchKernel((GrammarPrimitiveStrategy(),))
    report = kernel.run(
        task=_identity_task(),
        environment=ArcEnvironment(),
        grammar=ManyProgramGrammar(),
        scorer=ArcScorer(),
        memory=InMemoryLibrary(),
        budget=SearchBudget(max_evaluations=1, max_cell_evaluations=100),
        seed=0,
    )

    grammar_result = report.strategy_results[0]
    assert grammar_result.evaluated == 1
    assert report.budget_used.evaluations == 1

