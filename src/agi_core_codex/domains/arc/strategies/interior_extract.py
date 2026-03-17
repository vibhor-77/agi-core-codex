from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import (
    extract_enclosed_interior,
    nonzero_mask,
    recolor_nonzero,
    single_nonzero_color,
)
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, grid_shape


def _variants_for_first_example(
    base_output: Grid,
    expected_output: Grid,
) -> tuple[tuple[str, int | None], ...]:
    variants: list[tuple[str, int | None]] = []
    if base_output == expected_output:
        variants.append(("raw", None))
    target_color = single_nonzero_color(expected_output)
    if target_color is not None and nonzero_mask(base_output) == nonzero_mask(expected_output):
        variants.append(("recolor_nonzero", target_color))
    return tuple(variants)


def _apply_variant(grid: Grid, variant_name: str, target_color: int | None) -> Grid:
    if variant_name == "raw":
        return grid
    if variant_name == "recolor_nonzero" and target_color is not None:
        return recolor_nonzero(grid, target_color)
    raise ValueError(f"unknown variant: {variant_name}")


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
class InteriorExtractStrategy:
    name: str = "arc-interior-extract"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        if any(grid_shape(example.input) != grid_shape(example.output) for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("interior extraction only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        base_output = extract_enclosed_interior(first_example.input)
        if not any(cell != 0 for row in base_output for cell in row):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("no enclosed interior found in the first training example",),
            )

        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []
        for variant_name, target_color in _variants_for_first_example(base_output, first_example.output):
            def _make_executor(variant_name=variant_name, target_color=target_color):
                def executor(grid: Grid) -> Grid:
                    interior = extract_enclosed_interior(grid)
                    return _apply_variant(interior, variant_name, target_color)

                return executor

            name = "extract-enclosed-interior"
            if variant_name == "recolor_nonzero":
                name += f"-recolor-{target_color}"
            candidate_specs.append(
                (
                    name,
                    {
                        "type": "interior_extract",
                        "variant": variant_name,
                        "target_color": target_color,
                    },
                    _make_executor(),
                    3 + (1 if variant_name == "recolor_nonzero" else 0),
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
        notes = () if candidates else ("no interior extraction rule matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
