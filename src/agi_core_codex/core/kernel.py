from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from agi_core_codex.core.interfaces import (
    CandidateEvaluation,
    CostModel,
    Environment,
    Grammar,
    Memory,
    ProgramHandle,
    Scorer,
    ScoreBreakdown,
    SearchBudget,
    SearchReport,
    Strategy,
    StrategyResult,
)
from agi_core_codex.core.memory import LibraryEntry


def _rank_candidate(candidate: CandidateEvaluation) -> tuple[Any, ...]:
    score = candidate.score
    test_accuracy = score.test_accuracy if score.test_accuracy is not None else -1.0
    test_exact = int(score.test_exact is True)
    return (
        int(score.train_exact),
        round(score.train_accuracy, 12),
        test_exact,
        round(test_accuracy, 12),
        -score.failure_count,
        -candidate.program.cost.complexity,
        -candidate.program.cost.evaluations,
        -candidate.program.cost.cell_evaluations,
        candidate.program.id,
    )


@dataclass
class SearchContext:
    domain: str
    task: Any
    environment: Environment
    grammar: Grammar
    scorer: Scorer
    memory: Memory
    budget: SearchBudget
    seed: int
    budget_used: CostModel = field(default_factory=CostModel.zero)
    _strategy_base_costs: dict[str, CostModel] = field(default_factory=dict)

    def start_strategy(self, strategy_name: str, cost_model: CostModel) -> bool:
        if not self._can_consume(cost_model):
            return False
        self.budget_used = self.budget_used + cost_model
        self._strategy_base_costs[strategy_name] = cost_model
        return True

    def evaluate(
        self,
        program: ProgramHandle,
        strategy_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> CandidateEvaluation | None:
        if self.budget.max_program_complexity is not None:
            if program.cost.complexity > self.budget.max_program_complexity:
                return None

        eval_cost = CostModel(
            complexity=program.cost.complexity,
            evaluations=1,
            cell_evaluations=self.environment.task_size(self.task),
        )
        if not self._can_consume(eval_cost):
            return None

        self.budget_used = self.budget_used + eval_cost
        try:
            score = self.scorer.evaluate_program(self.task, program)
        except Exception as exc:  # pragma: no cover - defensive fallback
            score = ScoreBreakdown(
                train_exact=False,
                train_accuracy=0.0,
                example_accuracies=(),
                failure_count=1,
                notes=(f"scorer_error:{type(exc).__name__}:{exc}",),
            )

        return CandidateEvaluation(
            program=program,
            strategy_name=strategy_name,
            score=score,
            consumed=eval_cost,
            metadata=dict(metadata or {}),
        )

    def empty_strategy_result(
        self,
        *,
        name: str,
        status: str,
        notes: Sequence[str] = (),
        errors: Sequence[str] = (),
    ) -> StrategyResult:
        return StrategyResult(
            name=name,
            domain=self.domain,
            status=status,
            generated=0,
            evaluated=0,
            consumed=self._strategy_base_costs.get(name, CostModel.zero()),
            candidates=(),
            errors=tuple(errors),
            notes=tuple(notes),
        )

    def finish_strategy(
        self,
        *,
        name: str,
        status: str,
        generated: int,
        candidates: Sequence[CandidateEvaluation],
        notes: Sequence[str] = (),
        errors: Sequence[str] = (),
    ) -> StrategyResult:
        base_cost = self._strategy_base_costs.get(name, CostModel.zero())
        eval_cost = CostModel.zero()
        for candidate in candidates:
            eval_cost = eval_cost + candidate.consumed
        return StrategyResult(
            name=name,
            domain=self.domain,
            status=status,
            generated=generated,
            evaluated=len(candidates),
            consumed=base_cost + eval_cost,
            candidates=tuple(candidates),
            errors=tuple(errors),
            notes=tuple(notes),
        )

    def _can_consume(self, cost: CostModel) -> bool:
        next_evaluations = self.budget_used.evaluations + cost.evaluations
        next_cells = self.budget_used.cell_evaluations + cost.cell_evaluations
        if self.budget.max_evaluations is not None and next_evaluations > self.budget.max_evaluations:
            return False
        if self.budget.max_cell_evaluations is not None and next_cells > self.budget.max_cell_evaluations:
            return False
        return True


class SearchKernel:
    def __init__(self, strategies: Sequence[Strategy]) -> None:
        self._strategies = tuple(strategies)

    def run(
        self,
        *,
        task: Any,
        environment: Environment,
        grammar: Grammar,
        scorer: Scorer,
        memory: Memory,
        budget: SearchBudget,
        seed: int,
    ) -> SearchReport:
        context = SearchContext(
            domain=environment.domain,
            task=task,
            environment=environment,
            grammar=grammar,
            scorer=scorer,
            memory=memory,
            budget=budget,
            seed=seed,
        )

        strategy_results: list[StrategyResult] = []
        all_candidates: list[CandidateEvaluation] = []
        for strategy in self._strategies:
            if strategy.domain not in {"*", environment.domain}:
                strategy_results.append(
                    context.empty_strategy_result(
                        name=strategy.name,
                        status="domain_mismatch",
                        notes=(f"strategy domain {strategy.domain} does not match {environment.domain}",),
                    )
                )
                continue
            if not strategy.applies(task):
                strategy_results.append(
                    context.empty_strategy_result(
                        name=strategy.name,
                        status="not_applicable",
                        notes=("strategy declined the task",),
                    )
                )
                continue

            try:
                result = strategy.run(context)
            except Exception as exc:  # pragma: no cover - defensive fallback
                result = context.empty_strategy_result(
                    name=strategy.name,
                    status="error",
                    errors=(f"{type(exc).__name__}:{exc}",),
                )
            strategy_results.append(result)
            all_candidates.extend(result.candidates)

        best_candidate = None
        if all_candidates:
            best_candidate = sorted(all_candidates, key=_rank_candidate, reverse=True)[0]
            if best_candidate.score.train_exact:
                memory.store(
                    environment.domain,
                    LibraryEntry(
                        program=best_candidate.program,
                        train_accuracy=best_candidate.score.train_accuracy,
                        origin_task_key=environment.task_key(task),
                        metadata={"strategy_name": best_candidate.strategy_name},
                    ),
                )

        return SearchReport(
            domain=environment.domain,
            task_key=environment.task_key(task),
            seed=seed,
            primitive_count=grammar.primitive_count(task),
            budget=budget,
            budget_used=context.budget_used,
            strategy_results=tuple(strategy_results),
            best_candidate=best_candidate,
        )

