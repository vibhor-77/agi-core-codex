from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from agi_core_codex.core.hashing import stable_hash
from agi_core_codex.minimal.ops import Grid, PrimitiveSpec, compositor_specs, grid_accuracy


@dataclass(frozen=True)
class GridExample:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class GridTestCase:
    input: Grid
    output: Grid | None = None


@dataclass(frozen=True)
class GridTask:
    task_id: str
    train: tuple[GridExample, ...]
    test: tuple[GridTestCase, ...]


@dataclass(frozen=True)
class Program:
    id: str
    name: str
    kind: str
    complexity: int
    children: tuple["Program", ...] = ()
    primitive_name: str | None = None
    executor: Callable[[Grid], Grid] = field(repr=False, compare=False, default=lambda grid: grid)

    def execute(self, grid: Grid) -> Grid:
        return self.executor(grid)

    def contains_library_reference(self) -> bool:
        return self.kind == "library_ref" or any(child.contains_library_reference() for child in self.children)


@dataclass(frozen=True)
class LibraryEntry:
    program: Program
    promotion_score: tuple[float, int, int]
    promoted_round: int
    source_task_keys: tuple[str, ...]
    reuse_count: int = 0


@dataclass
class LearnerMemory:
    committed: dict[str, LibraryEntry] = field(default_factory=dict)
    pending: dict[str, LibraryEntry] = field(default_factory=dict)

    def committed_programs(self) -> tuple[Program, ...]:
        return tuple(entry.program for entry in self.committed.values())

    def stage(self, entry: LibraryEntry) -> None:
        if entry.program.id in self.committed or entry.program.id in self.pending:
            return
        self.pending[entry.program.id] = entry

    def commit(self) -> int:
        pending = tuple(self.pending.values())
        for entry in pending:
            self.committed[entry.program.id] = entry
        self.pending.clear()
        return len(pending)

    def increment_reuse(self, program_ids: Iterable[str]) -> None:
        for program_id in program_ids:
            if program_id in self.committed:
                entry = self.committed[program_id]
                self.committed[program_id] = LibraryEntry(
                    program=entry.program,
                    promotion_score=entry.promotion_score,
                    promoted_round=entry.promoted_round,
                    source_task_keys=entry.source_task_keys,
                    reuse_count=entry.reuse_count + 1,
                )

    def reused_program_count(self) -> int:
        return sum(1 for entry in self.committed.values() if entry.reuse_count > 0)

    def total_reuse_count(self) -> int:
        return sum(entry.reuse_count for entry in self.committed.values())


@dataclass(frozen=True)
class CandidateScore:
    train_exact: bool
    train_accuracy: float
    failure_count: int
    test_exact: bool | None = None
    test_accuracy: float | None = None


@dataclass(frozen=True)
class CandidateResult:
    program: Program
    score: CandidateScore
    evaluation_index: int
    uses_committed_library: bool


@dataclass(frozen=True)
class TaskRun:
    task_key: str
    best: CandidateResult | None
    evaluated_count: int
    candidates: tuple[CandidateResult, ...]


@dataclass(frozen=True)
class RoundSummary:
    round_index: int
    solved_train: int
    task_count: int
    library_size: int
    promoted_count: int
    library_reuse_count: int
    evaluated_candidates: int
    search_cost_per_exact: float


def _program_rank_key(candidate: CandidateResult) -> tuple[object, ...]:
    score = candidate.score
    test_accuracy = score.test_accuracy if score.test_accuracy is not None else -1.0
    test_exact = int(score.test_exact is True)
    return (
        int(score.train_exact),
        round(score.train_accuracy, 12),
        test_exact,
        round(test_accuracy, 12),
        -score.failure_count,
        -candidate.program.complexity,
        candidate.program.id,
    )


def _evaluate_program(task: GridTask, program: Program) -> CandidateScore:
    train_scores: list[float] = []
    test_scores: list[float] = []
    failures = 0
    for example in task.train:
        try:
            predicted = program.execute(example.input)
        except Exception:
            failures += 1
            train_scores.append(0.0)
            continue
        train_scores.append(grid_accuracy(example.output, predicted))

    for case in task.test:
        try:
            predicted = program.execute(case.input)
        except Exception:
            failures += 1
            predicted = ()
        if case.output is not None:
            test_scores.append(grid_accuracy(case.output, predicted))

    train_accuracy = sum(train_scores) / len(train_scores) if train_scores else 0.0
    train_exact = bool(train_scores) and all(score == 1.0 for score in train_scores) and failures == 0
    if not test_scores:
        return CandidateScore(
            train_exact=train_exact,
            train_accuracy=train_accuracy,
            failure_count=failures,
        )
    test_accuracy = sum(test_scores) / len(test_scores)
    return CandidateScore(
        train_exact=train_exact,
        train_accuracy=train_accuracy,
        failure_count=failures,
        test_exact=all(score == 1.0 for score in test_scores) and failures == 0,
        test_accuracy=test_accuracy,
    )


def make_leaf_program(spec: PrimitiveSpec) -> Program:
    if spec.unary is None:
        raise ValueError(f"primitive {spec.name} is not unary")
    semantics = {"type": "primitive", "name": spec.name}
    return Program(
        id=stable_hash(semantics, namespace="minimal.program"),
        name=spec.name,
        kind="primitive",
        primitive_name=spec.name,
        complexity=spec.complexity,
        executor=spec.unary,
    )


def wrap_library_program(program: Program) -> Program:
    semantics = {"type": "library_ref", "program_id": program.id}
    return Program(
        id=stable_hash(semantics, namespace="minimal.program"),
        name=f"lib:{program.name}",
        kind="library_ref",
        primitive_name=program.primitive_name,
        complexity=program.complexity,
        children=(program,),
        executor=program.executor,
    )


def compose_program(spec: PrimitiveSpec, left: Program, right: Program) -> Program:
    semantics = {
        "type": "compose",
        "name": spec.name,
        "left": left.id,
        "right": right.id,
    }

    def executor(grid: Grid) -> Grid:
        if spec.name == "chain":
            return right.execute(left.execute(grid))
        if spec.binary is None:
            raise ValueError(f"compositor {spec.name} is missing a binary executor")
        return spec.binary(left.execute(grid), right.execute(grid))

    return Program(
        id=stable_hash(semantics, namespace="minimal.program"),
        name=f"{left.name}-{spec.name}-{right.name}",
        kind="composition",
        primitive_name=spec.name,
        complexity=left.complexity + right.complexity + spec.complexity,
        children=(left, right),
        executor=executor,
    )


_COMPOSITOR_SPECS = {spec.name: spec for spec in compositor_specs()}


def _canonicalize_program(program: Program) -> Program:
    if program.kind == "library_ref":
        return _canonicalize_program(program.children[0])
    if program.kind != "composition":
        return program
    left = _canonicalize_program(program.children[0])
    right = _canonicalize_program(program.children[1])
    spec = _COMPOSITOR_SPECS[program.primitive_name or ""]
    return compose_program(spec, left, right)


def iter_subprograms(program: Program) -> Iterable[Program]:
    yield program
    for child in program.children:
        yield from iter_subprograms(child)


class WakeSleepLearner:
    def __init__(
        self,
        *,
        unary_primitives: Sequence[PrimitiveSpec],
        binary_compositors: Sequence[PrimitiveSpec],
    ) -> None:
        self._unary_primitives = tuple(unary_primitives)
        self._binary_compositors = tuple(binary_compositors)

    @staticmethod
    def _entry_priority(entry: LibraryEntry) -> tuple[object, ...]:
        exact_or_accuracy, task_count, compression_gain = entry.promotion_score
        return (
            round(exact_or_accuracy, 12),
            entry.reuse_count,
            compression_gain,
            task_count,
            entry.program.complexity,
            entry.program.id,
        )

    def _ordered_library_refs(
        self,
        memory: LearnerMemory,
        *,
        round_index: int,
    ) -> tuple[Program, ...]:
        entries = self._search_library_entries(memory, round_index=round_index)
        if not entries:
            return ()
        ordered_entries = sorted(self._ordered_library_entries(entries), key=self._entry_priority, reverse=True)
        return tuple(wrap_library_program(entry.program) for entry in ordered_entries)

    def _ordered_library_entries(
        self,
        entries: Sequence[LibraryEntry],
    ) -> tuple[LibraryEntry, ...]:
        return tuple(sorted(entries, key=self._entry_priority, reverse=True))

    @staticmethod
    def _max_program_complexity(round_index: int) -> int:
        return 3 + (2 * round_index)

    def _frontier_library_programs(
        self,
        memory: LearnerMemory,
        *,
        round_index: int,
    ) -> tuple[Program, ...]:
        entries = self._ordered_library_entries(self._search_library_entries(memory, round_index=round_index))
        if not entries:
            return ()
        exact_entries = [entry for entry in entries if entry.promotion_score[0] >= 1.0]
        partial_entries = [
            entry
            for entry in entries
            if entry.promotion_score[0] < 1.0 and entry.promotion_score[2] > 0
        ]
        exact_limit = 2 if round_index <= 1 else 4
        partial_limit = 0 if round_index <= 1 else 2
        selected: list[LibraryEntry] = []
        seen_ids: set[str] = set()
        for entry in exact_entries[:exact_limit] + partial_entries[:partial_limit]:
            if entry.program.id in seen_ids:
                continue
            seen_ids.add(entry.program.id)
            selected.append(entry)
        return tuple(wrap_library_program(entry.program) for entry in selected)

    def _search_library_entries(
        self,
        memory: LearnerMemory,
        *,
        round_index: int,
    ) -> tuple[LibraryEntry, ...]:
        entries = tuple(memory.committed.values())
        if round_index <= 0:
            return entries
        filtered: list[LibraryEntry] = []
        for entry in entries:
            if entry.program.kind != "primitive":
                filtered.append(entry)
                continue
            if entry.reuse_count > 0 or entry.promotion_score[2] > 0:
                filtered.append(entry)
        return tuple(filtered)

    def run_round(
        self,
        *,
        tasks: Sequence[GridTask],
        memory: LearnerMemory,
        round_index: int,
    ) -> tuple[tuple[TaskRun, ...], RoundSummary]:
        task_runs: list[TaskRun] = []
        total_evaluated = 0
        library_reuse_count = 0

        for task in tasks:
            run = self._run_task(task=task, memory=memory, round_index=round_index)
            task_runs.append(run)
            total_evaluated += run.evaluated_count
            if run.best is not None and run.best.uses_committed_library:
                library_reuse_count += 1
                reused_ids = {
                    subprogram.children[0].id
                    for subprogram in iter_subprograms(run.best.program)
                    if subprogram.kind == "library_ref"
                }
                memory.increment_reuse(reused_ids)

        promoted = self.sleep(task_runs=tuple(task_runs), memory=memory, round_index=round_index)
        committed = memory.commit()
        solved_train = sum(1 for run in task_runs if run.best is not None and run.best.score.train_exact)
        search_cost_per_exact = (
            total_evaluated / solved_train
            if solved_train
            else float(total_evaluated)
        )
        summary = RoundSummary(
            round_index=round_index,
            solved_train=solved_train,
            task_count=len(task_runs),
            library_size=len(memory.committed),
            promoted_count=committed if committed else promoted,
            library_reuse_count=library_reuse_count,
            evaluated_candidates=total_evaluated,
            search_cost_per_exact=search_cost_per_exact,
        )
        return tuple(task_runs), summary

    def _run_task(self, *, task: GridTask, memory: LearnerMemory, round_index: int) -> TaskRun:
        seed_programs = tuple(make_leaf_program(spec) for spec in self._unary_primitives)
        library_programs = self._ordered_library_refs(memory, round_index=round_index)
        frontier_library_programs = self._frontier_library_programs(memory, round_index=round_index)
        max_program_complexity = self._max_program_complexity(round_index)

        ordered_candidates: list[Program] = []
        seen_program_ids: set[str] = set()

        def append_candidate(program: Program) -> None:
            if program.complexity > max_program_complexity:
                return
            if program.kind == "composition" and program.primitive_name == "chain":
                left, right = program.children
                if left.primitive_name == "identity" or right.primitive_name == "identity":
                    return
            if program.id in seen_program_ids:
                return
            seen_program_ids.add(program.id)
            ordered_candidates.append(program)

        for program in seed_programs:
            append_candidate(program)

        for program in frontier_library_programs:
            append_candidate(program)

        if round_index > 0 and library_programs:
            left_pool = tuple(program for program in frontier_library_programs if program.primitive_name != "identity")
            if not left_pool:
                left_pool = frontier_library_programs
            right_pool = tuple(program for program in seed_programs if program.primitive_name != "identity") + left_pool
            early_compositors = (
                tuple(spec for spec in self._binary_compositors if spec.name == "chain")
                if round_index == 1
                else self._binary_compositors
            )
            for compositor in early_compositors:
                for left in left_pool:
                    for right in right_pool:
                        append_candidate(compose_program(compositor, left, right))

        for program in library_programs:
            append_candidate(program)

        if round_index > 0 and library_programs:
            full_leaf_pool = library_programs + seed_programs
            for compositor in self._binary_compositors:
                if compositor.name == "chain":
                    for left in library_programs:
                        for right in full_leaf_pool:
                            append_candidate(compose_program(compositor, left, right))
                    for left in seed_programs:
                        for right in library_programs:
                            append_candidate(compose_program(compositor, left, right))
                    continue
                for left in library_programs:
                    for right in full_leaf_pool:
                        append_candidate(compose_program(compositor, left, right))
                for left in seed_programs:
                    for right in library_programs:
                        append_candidate(compose_program(compositor, left, right))

        candidate_results: list[CandidateResult] = []
        for evaluation_index, program in enumerate(ordered_candidates, start=1):
            score = _evaluate_program(task, program)
            candidate = CandidateResult(
                program=program,
                score=score,
                evaluation_index=evaluation_index,
                uses_committed_library=program.contains_library_reference(),
            )
            candidate_results.append(candidate)
            if score.train_exact:
                break

        best = sorted(candidate_results, key=_program_rank_key, reverse=True)[0] if candidate_results else None
        return TaskRun(
            task_key=task.task_id,
            best=best,
            evaluated_count=len(candidate_results),
            candidates=tuple(candidate_results),
        )

    def sleep(
        self,
        *,
        task_runs: Sequence[TaskRun],
        memory: LearnerMemory,
        round_index: int,
        promotion_limit: int = 16,
        partial_limit: int = 3,
    ) -> int:
        subtree_occurrences: dict[str, list[tuple[Program, str, float, bool]]] = defaultdict(list)
        for run in task_runs:
            ranked = sorted(run.candidates, key=_program_rank_key, reverse=True)[:partial_limit]
            for candidate in ranked:
                for subtree in iter_subprograms(candidate.program):
                    if subtree.kind == "library_ref":
                        continue
                    canonical_subtree = _canonicalize_program(subtree)
                    subtree_occurrences[canonical_subtree.id].append(
                        (
                            canonical_subtree,
                            run.task_key,
                            candidate.score.train_accuracy,
                            candidate.score.train_exact,
                        )
                    )

        scored: list[tuple[tuple[float, int, int], Program, tuple[str, ...]]] = []
        for program_id, occurrences in subtree_occurrences.items():
            program = occurrences[0][0]
            task_keys = tuple(sorted({task_key for _, task_key, _, _ in occurrences}))
            mean_accuracy = sum(score for _, _, score, _ in occurrences) / len(occurrences)
            exact_hits = sum(1 for _, _, _, is_exact in occurrences if is_exact)
            compression_gain = len(occurrences) * max(program.complexity - 1, 0)
            if exact_hits == 0 and mean_accuracy <= 0.0:
                continue
            promotion_score = (1.0 if exact_hits else mean_accuracy, len(task_keys), compression_gain)
            scored.append((promotion_score, program, task_keys))

        scored.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[1].id), reverse=True)
        promotions = 0
        for promotion_score, program, task_keys in scored:
            memory.stage(
                LibraryEntry(
                    program=program,
                    promotion_score=promotion_score,
                    promoted_round=round_index,
                    source_task_keys=task_keys,
                )
            )
            promotions += 1
            if promotions >= promotion_limit:
                break
        return promotions


def aggregate_round_metrics(round_summaries: Sequence[RoundSummary]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for summary in round_summaries:
        prefix = f"round_{summary.round_index + 1}"
        metrics[f"{prefix}_solved_train"] = summary.solved_train
        metrics[f"{prefix}_library_size"] = summary.library_size
        metrics[f"{prefix}_promoted_count"] = summary.promoted_count
        metrics[f"{prefix}_library_reuse_count"] = summary.library_reuse_count
        metrics[f"{prefix}_evaluated_candidates"] = summary.evaluated_candidates
        metrics[f"{prefix}_search_cost_per_exact"] = round(summary.search_cost_per_exact, 6)
    if round_summaries:
        metrics["round_count"] = len(round_summaries)
        metrics["final_library_size"] = round_summaries[-1].library_size
    return metrics
