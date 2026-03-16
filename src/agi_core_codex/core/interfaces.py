from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


ProgramExecutor = Callable[[Any], Any]


@dataclass(frozen=True)
class CostModel:
    complexity: int = 0
    evaluations: int = 0
    cell_evaluations: int = 0

    def __add__(self, other: "CostModel") -> "CostModel":
        return CostModel(
            complexity=self.complexity + other.complexity,
            evaluations=self.evaluations + other.evaluations,
            cell_evaluations=self.cell_evaluations + other.cell_evaluations,
        )

    @classmethod
    def zero(cls) -> "CostModel":
        return cls()

    def to_dict(self) -> dict[str, int]:
        return {
            "complexity": self.complexity,
            "evaluations": self.evaluations,
            "cell_evaluations": self.cell_evaluations,
        }


@dataclass(frozen=True)
class SearchBudget:
    max_evaluations: int | None = None
    max_cell_evaluations: int | None = None
    max_program_complexity: int | None = None


@dataclass(frozen=True)
class ProgramHandle:
    id: str
    name: str
    domain: str
    executor: ProgramExecutor = field(repr=False, compare=False)
    cost: CostModel = field(default_factory=CostModel.zero)
    semantics: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ScoreBreakdown:
    train_exact: bool
    train_accuracy: float
    example_accuracies: tuple[float, ...]
    failure_count: int = 0
    test_exact: bool | None = None
    test_accuracy: float | None = None
    test_predictions: tuple[Any, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateEvaluation:
    program: ProgramHandle
    strategy_name: str
    score: ScoreBreakdown
    consumed: CostModel
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyResult:
    name: str
    domain: str
    status: str
    generated: int
    evaluated: int
    consumed: CostModel
    candidates: tuple[CandidateEvaluation, ...] = ()
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchReport:
    domain: str
    task_key: str
    seed: int
    primitive_count: int
    budget: SearchBudget
    budget_used: CostModel
    strategy_results: tuple[StrategyResult, ...]
    best_candidate: CandidateEvaluation | None


@runtime_checkable
class Environment(Protocol):
    domain: str

    def task_key(self, task: Any) -> str:
        ...

    def task_size(self, task: Any) -> int:
        ...


@runtime_checkable
class Grammar(Protocol):
    domain: str

    def primitive_count(self, task: Any | None = None) -> int:
        ...

    def enumerate_primitives(self, task: Any) -> Sequence[ProgramHandle]:
        ...


@runtime_checkable
class Scorer(Protocol):
    domain: str

    def evaluate_program(self, task: Any, program: ProgramHandle) -> ScoreBreakdown:
        ...


@runtime_checkable
class Memory(Protocol):
    def store(self, domain: str, entry: Any) -> None:
        ...

    def recall(self, domain: str, limit: int | None = None) -> Sequence[Any]:
        ...


@runtime_checkable
class Strategy(Protocol):
    name: str
    domain: str
    cost_model: CostModel

    def applies(self, task: Any) -> bool:
        ...

    def run(self, context: Any) -> StrategyResult:
        ...

