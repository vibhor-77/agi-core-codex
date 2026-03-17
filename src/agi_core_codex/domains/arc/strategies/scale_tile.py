from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _scale_grid(grid: Grid, factor: int) -> Grid:
    rows = []
    for row in grid:
        expanded_row = []
        for cell in row:
            expanded_row.extend([cell] * factor)
        for _ in range(factor):
            rows.append(tuple(expanded_row))
    return tuple(rows)


def _tile_grid(grid: Grid, factor: int) -> Grid:
    if not grid:
        return ()
    tiled_rows = [tuple(row) * factor for row in grid]
    return tuple(tiled_rows * factor)


def _downscale_grid(grid: Grid, factor: int) -> Grid:
    height, width = grid_shape(grid)
    if height % factor != 0 or width % factor != 0:
        raise ValueError("grid cannot be downscaled by the requested factor")
    rows = []
    for row_start in range(0, height, factor):
        out_row = []
        for col_start in range(0, width, factor):
            block = [
                grid[row_index][col_index]
                for row_index in range(row_start, row_start + factor)
                for col_index in range(col_start, col_start + factor)
            ]
            out_row.append(Counter(block).most_common(1)[0][0])
        rows.append(tuple(out_row))
    return tuple(rows)


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
class ScaleTileStrategy:
    name: str = "arc-scale-tile"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []
        first_example = context.task.train[0]
        input_height, input_width = grid_shape(first_example.input)
        output_height, output_width = grid_shape(first_example.output)

        if input_height and input_width and output_height and output_width:
            if all(
                grid_shape(example.output)[0] == grid_shape(example.input)[0] * (output_height // input_height)
                and grid_shape(example.output)[1] == grid_shape(example.input)[1] * (output_width // input_width)
                and output_height % input_height == 0
                and output_width % input_width == 0
                for example in context.task.train
            ):
                h_ratio = output_height // input_height
                w_ratio = output_width // input_width
                if h_ratio == w_ratio and h_ratio >= 2:
                    factor = h_ratio
                    for op_name, op in (
                        ("scale", _scale_grid),
                        ("tile", _tile_grid),
                    ):
                        if op(first_example.input, factor) == first_example.output:
                            candidate_specs.append(
                                (
                                    f"cross-ref-{op_name}-{factor}x",
                                    {
                                        "type": "cross_reference_scale_tile",
                                        "operation": op_name,
                                        "factor": factor,
                                    },
                                    lambda grid, op=op, factor=factor: op(grid, factor),
                                    max(2, factor),
                                )
                            )

            if all(
                grid_shape(example.input)[0] == grid_shape(example.output)[0] * (input_height // output_height)
                and grid_shape(example.input)[1] == grid_shape(example.output)[1] * (input_width // output_width)
                and input_height % output_height == 0
                and input_width % output_width == 0
                for example in context.task.train
            ):
                h_ratio = input_height // output_height
                w_ratio = input_width // output_width
                if h_ratio == w_ratio and h_ratio >= 2:
                    factor = h_ratio
                    if _downscale_grid(first_example.input, factor) == first_example.output:
                        candidate_specs.append(
                            (
                                f"cross-ref-downscale-{factor}x",
                                {
                                    "type": "cross_reference_scale_tile",
                                    "operation": "downscale",
                                    "factor": factor,
                                },
                                lambda grid, factor=factor: _downscale_grid(grid, factor),
                                max(2, factor),
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
        notes = () if candidates else ("no consistent scale/tile/downscale ratio rule matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
