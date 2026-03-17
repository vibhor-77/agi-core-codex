from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid


def _line_nonzero_count(line: tuple[int, ...]) -> int:
    return sum(1 for cell in line if cell != 0)


def _line_sum(line: tuple[int, ...]) -> int:
    return sum(line)


def _line_max(line: tuple[int, ...]) -> int:
    return max(line, default=0)


def _line_unique_colors(line: tuple[int, ...]) -> int:
    return len(set(line))


LINE_KEYS: tuple[tuple[str, Callable[[tuple[int, ...]], int]], ...] = (
    ("nonzero-count", _line_nonzero_count),
    ("sum", _line_sum),
    ("max", _line_max),
    ("unique-colors", _line_unique_colors),
)


def _sort_rows(grid: Grid, key_fn: Callable[[tuple[int, ...]], int], *, reverse: bool) -> Grid:
    return freeze_grid(sorted(grid, key=key_fn, reverse=reverse))


def _sort_columns(grid: Grid, key_fn: Callable[[tuple[int, ...]], int], *, reverse: bool) -> Grid:
    if not grid:
        return ()
    columns = list(zip(*grid, strict=True))
    ordered = sorted(columns, key=key_fn, reverse=reverse)
    return freeze_grid(zip(*ordered, strict=True))


def _emit_program(
    *,
    context: Any,
    strategy_name: str,
    name: str,
    semantics: dict[str, Any],
    executor: Callable[[Grid], Grid],
    complexity: int,
):
    program = make_arc_program(
        name=name,
        semantics=semantics,
        executor=executor,
        complexity=complexity,
    )
    return context.evaluate(program, strategy_name)


@dataclass(frozen=True)
class RowColumnDecompositionStrategy:
    name: str = "arc-row-column-decomposition"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        if not all(
            len(example.input) == len(example.output)
            and (len(example.input[0]) if example.input else 0)
            == (len(example.output[0]) if example.output else 0)
            for example in context.task.train
        ):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("row/column sorting only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []
        sorters = (
            ("rows", _sort_rows),
            ("columns", _sort_columns),
        )

        for axis_name, sorter in sorters:
            for key_name, key_fn in LINE_KEYS:
                for direction_name, reverse in (("asc", False), ("desc", True)):
                    base_output = sorter(first_example.input, key_fn, reverse=reverse)
                    if base_output != first_example.output:
                        continue
                    candidate_specs.append(
                        (
                            f"sort-{axis_name}-by-{key_name}-{direction_name}",
                            {
                                "type": "row_column_sort",
                                "axis": axis_name,
                                "key": key_name,
                                "order": direction_name,
                            },
                            lambda grid, sorter=sorter, key_fn=key_fn, reverse=reverse: sorter(
                                grid,
                                key_fn,
                                reverse=reverse,
                            ),
                            4,
                        )
                    )

        candidates = []
        generated = 0
        for name, semantics, executor, complexity in candidate_specs:
            generated += 1
            evaluation = _emit_program(
                context=context,
                strategy_name=self.name,
                name=name,
                semantics=semantics,
                executor=executor,
                complexity=complexity,
            )
            if evaluation is None:
                break
            candidates.append(evaluation)

        status = "ok" if candidates else "not_applicable"
        notes = () if candidates else ("no consistent row/column sorting rule matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
