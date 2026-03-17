from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid


def _extend_rays(grid: Grid, direction: str, *, seed_color: int | None = None) -> Grid:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    rows = [list(row) for row in grid]
    deltas = {
        "right": (0, 1),
        "left": (0, -1),
        "down": (1, 0),
        "up": (-1, 0),
    }
    delta_row, delta_col = deltas[direction]

    for row_index in range(height):
        for col_index in range(width):
            color = grid[row_index][col_index]
            if color == 0:
                continue
            if seed_color is not None and color != seed_color:
                continue
            next_row = row_index + delta_row
            next_col = col_index + delta_col
            while 0 <= next_row < height and 0 <= next_col < width and grid[next_row][next_col] == 0:
                rows[next_row][next_col] = color
                next_row += delta_row
                next_col += delta_col
    return freeze_grid(rows)


def _mask_overlap(primary: Grid, secondary: Grid) -> Grid:
    return tuple(
        tuple(
            primary_cell if primary_cell != 0 and secondary_cell != 0 else 0
            for primary_cell, secondary_cell in zip(primary_row, secondary_row, strict=True)
        )
        for primary_row, secondary_row in zip(primary, secondary, strict=True)
    )


def _mask_color_overlaps(grid: Grid, primary_direction: str, secondary_direction: str) -> Grid:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    rows = [[0 for _ in range(width)] for _ in range(height)]
    colors = sorted({cell for row in grid for cell in row if cell != 0})
    for color in colors:
        primary = _extend_rays(grid, primary_direction, seed_color=color)
        secondary = _extend_rays(grid, secondary_direction, seed_color=color)
        overlap = _mask_overlap(primary, secondary)
        for row_index, row in enumerate(overlap):
            for col_index, value in enumerate(row):
                if value != 0:
                    rows[row_index][col_index] = value
    return freeze_grid(rows)


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
class RayExtensionStrategy:
    name: str = "arc-ray-extension"
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
                notes=("ray extension only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []

        for direction in ("right", "left", "down", "up"):
            if _extend_rays(first_example.input, direction) == first_example.output:
                candidate_specs.append(
                    (
                        f"extend-rays-{direction}",
                        {
                            "type": "ray_extension",
                            "mode": "single_direction",
                            "direction": direction,
                        },
                        lambda grid, direction=direction: _extend_rays(grid, direction),
                        3,
                    )
                )

        for primary_direction, secondary_direction in (
            ("right", "left"),
            ("left", "right"),
            ("down", "up"),
            ("up", "down"),
        ):
            if _mask_color_overlaps(first_example.input, primary_direction, secondary_direction) == first_example.output:
                candidate_specs.append(
                    (
                        f"mask-extend-rays-{primary_direction}-with-{secondary_direction}",
                        {
                            "type": "ray_extension",
                            "mode": "masked_overlap",
                            "primary_direction": primary_direction,
                            "secondary_direction": secondary_direction,
                        },
                        lambda grid, primary_direction=primary_direction, secondary_direction=secondary_direction: _mask_color_overlaps(
                            grid,
                            primary_direction,
                            secondary_direction,
                        ),
                        5,
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
        notes = () if candidates else ("no consistent ray-extension rule matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
