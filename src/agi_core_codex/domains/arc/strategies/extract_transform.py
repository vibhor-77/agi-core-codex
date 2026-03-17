from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import nonzero_mask, recolor_nonzero, single_nonzero_color
from agi_core_codex.domains.arc.grammar import (
    _crop_nonzero,
    _extract_largest_cc,
    _extract_unique_color_region,
    _flip_horizontal,
    _mirror_tile_both,
    _mirror_tile_horizontal,
    _mirror_tile_vertical,
    _rotate_180,
    _rotate_tile_clockwise,
    _tile_horizontal,
    _transpose,
)
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid


def _variants_for_first_example(base_output: Grid, expected_output: Grid) -> tuple[tuple[str, int | None], ...]:
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
class ExtractTransformStrategy:
    name: str = "arc-extract-transform"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        stage_one = (
            ("crop-nonzero", _crop_nonzero, 2),
            ("extract-largest-cc", _extract_largest_cc, 3),
            ("extract-unique-color-region", _extract_unique_color_region, 3),
        )
        stage_two = (
            ("flip-horizontal", _flip_horizontal, 1),
            ("rotate-180", _rotate_180, 1),
            ("transpose", _transpose, 1),
            ("tile-horizontal", _tile_horizontal, 3),
            ("mirror-tile-horizontal", _mirror_tile_horizontal, 3),
            ("mirror-tile-vertical", _mirror_tile_vertical, 3),
            ("mirror-tile-both", _mirror_tile_both, 3),
            ("rotate-tile-clockwise", _rotate_tile_clockwise, 4),
        )

        first_example = context.task.train[0]
        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []

        for stage_one_name, stage_one_fn, stage_one_cost in stage_one:
            for stage_two_name, stage_two_fn, stage_two_cost in stage_two:
                try:
                    base_output = stage_two_fn(stage_one_fn(first_example.input))
                except Exception:
                    continue
                for variant_name, target_color in _variants_for_first_example(
                    base_output,
                    first_example.output,
                ):
                    def _make_executor(
                        stage_one_fn=stage_one_fn,
                        stage_two_fn=stage_two_fn,
                        variant_name=variant_name,
                        target_color=target_color,
                    ):
                        def executor(grid: Grid) -> Grid:
                            return _apply_variant(
                                stage_two_fn(stage_one_fn(grid)),
                                variant_name,
                                target_color,
                            )

                        return executor

                    name = f"{stage_one_name}-then-{stage_two_name}"
                    if variant_name == "recolor_nonzero":
                        name += f"-recolor-{target_color}"
                    candidate_specs.append(
                        (
                            name,
                            {
                                "type": "extract_transform",
                                "first": stage_one_name,
                                "second": stage_two_name,
                                "variant": variant_name,
                                "target_color": target_color,
                            },
                            _make_executor(),
                            stage_one_cost + stage_two_cost + (1 if variant_name == "recolor_nonzero" else 0),
                        )
                    )

        candidates = []
        generated = 0
        seen_names: set[str] = set()
        for name, semantics, executor, complexity in candidate_specs:
            if name in seen_names:
                continue
            seen_names.add(name)
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
        notes = () if candidates else ("no extractor/transform composition matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
