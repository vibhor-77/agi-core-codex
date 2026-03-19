from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agi_core_codex.core.hashing import stable_hash
from agi_core_codex.core.interfaces import (
    CostModel,
    HypothesisFamily,
    ProgramHandle,
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


def _compose_program_chain(programs: tuple[ProgramHandle, ...], *, domain: str) -> ProgramHandle:
    semantics = {
        "type": "sequential_composition",
        "domain": domain,
        "children": tuple(program.id for program in programs),
    }

    def executor(value: Any) -> Any:
        current = value
        for program in programs:
            current = program.executor(current)
        return current

    return ProgramHandle(
        id=stable_hash(semantics, namespace="core.program"),
        name="-then-".join(program.name for program in programs),
        domain=domain,
        executor=executor,
        cost=CostModel(
            complexity=sum(program.cost.complexity for program in programs) + max(len(programs) - 1, 0)
        ),
        semantics=semantics,
    )


@dataclass(frozen=True)
class EnumerativeCompositionStrategy:
    name: str = "bootstrap-composition"
    domain: str = "*"
    max_depth: int = 2
    beam_width: int = 8
    recall_limit: int = 16
    require_library_component: bool = False
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return True

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        seed_programs = tuple(context.grammar.enumerate_primitives(context.task))
        library_entries = tuple(context.memory.recall(context.domain, limit=self.recall_limit))
        library_ids = {entry.program.id for entry in library_entries}
        frontier: list[tuple[ProgramHandle, bool]] = [(program, False) for program in seed_programs]
        frontier.extend((entry.program, True) for entry in library_entries)

        generated = 0
        candidates = []
        depth_frontier: list[tuple[ProgramHandle, bool]] = list(frontier)
        seen_program_ids: set[str] = set()

        for program, uses_library in depth_frontier:
            if program.id in seen_program_ids:
                continue
            seen_program_ids.add(program.id)
            generated += 1
            evaluation = context.evaluate(
                program,
                self.name,
                metadata={
                    "depth": 1,
                    "uses_library": uses_library or program.id in library_ids,
                },
            )
            if evaluation is None:
                return context.finish_strategy(
                    name=self.name,
                    status="budget_exhausted",
                    generated=generated,
                    candidates=candidates,
                    notes=(f"seed_programs={len(seed_programs)}", f"library_programs={len(library_entries)}"),
                )
            candidates.append(evaluation)

        ranked_frontier = sorted(candidates, key=lambda candidate: candidate.score.train_accuracy, reverse=True)[
            : self.beam_width
        ]
        current_frontier = [
            (candidate.program, bool(candidate.metadata.get("uses_library")))
            for candidate in ranked_frontier
        ]

        for depth in range(2, self.max_depth + 1):
            next_frontier: list[tuple[ProgramHandle, bool]] = []
            for prefix_program, prefix_uses_library in current_frontier:
                for suffix_program, suffix_uses_library in frontier:
                    uses_library = prefix_uses_library or suffix_uses_library
                    if self.require_library_component and not uses_library:
                        continue
                    composed = _compose_program_chain(
                        (prefix_program, suffix_program),
                        domain=context.domain,
                    )
                    if composed.id in seen_program_ids:
                        continue
                    seen_program_ids.add(composed.id)
                    generated += 1
                    evaluation = context.evaluate(
                        composed,
                        self.name,
                        metadata={"depth": depth, "uses_library": uses_library},
                    )
                    if evaluation is None:
                        return context.finish_strategy(
                            name=self.name,
                            status="budget_exhausted",
                            generated=generated,
                            candidates=candidates,
                            notes=(
                                f"seed_programs={len(seed_programs)}",
                                f"library_programs={len(library_entries)}",
                                f"max_depth={self.max_depth}",
                            ),
                        )
                    candidates.append(evaluation)
                    next_frontier.append((evaluation.program, uses_library))

            ranked_candidates = sorted(
                [candidate for candidate in candidates if candidate.metadata.get("depth") == depth],
                key=lambda candidate: candidate.score.train_accuracy,
                reverse=True,
            )[: self.beam_width]
            current_frontier = [
                (candidate.program, bool(candidate.metadata.get("uses_library")))
                for candidate in ranked_candidates
            ]
            if not current_frontier:
                break

        return context.finish_strategy(
            name=self.name,
            status="ok" if candidates else "no_candidates",
            generated=generated,
            candidates=candidates,
            notes=(
                f"seed_programs={len(seed_programs)}",
                f"library_programs={len(library_entries)}",
                f"max_depth={self.max_depth}",
                (
                    "compositions require at least one library component"
                    if self.require_library_component
                    else "compositions may use only seeded programs"
                ),
            ),
        )
