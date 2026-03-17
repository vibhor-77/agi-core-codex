from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import (
    cells_share_shape,
    find_uniform_col_separators,
    find_uniform_row_separators,
    flatten_cells,
    intersect_separators,
    nonzero_mask,
    recolor_nonzero,
    single_nonzero_color,
    split_grid_by_separators,
)
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, grid_shape


def _op_or(a: Grid, b: Grid) -> Grid:
    return tuple(
        tuple(left if left != 0 else right for left, right in zip(a_row, b_row, strict=True))
        for a_row, b_row in zip(a, b, strict=True)
    )


def _op_and(a: Grid, b: Grid) -> Grid:
    return tuple(
        tuple(left if right != 0 else 0 for left, right in zip(a_row, b_row, strict=True))
        for a_row, b_row in zip(a, b, strict=True)
    )


def _op_xor(a: Grid, b: Grid) -> Grid:
    return tuple(
        tuple(int((left != 0) ^ (right != 0)) for left, right in zip(a_row, b_row, strict=True))
        for a_row, b_row in zip(a, b, strict=True)
    )


def _op_a_minus_b(a: Grid, b: Grid) -> Grid:
    return tuple(
        tuple(left if right == 0 else 0 for left, right in zip(a_row, b_row, strict=True))
        for a_row, b_row in zip(a, b, strict=True)
    )


def _op_b_minus_a(a: Grid, b: Grid) -> Grid:
    return tuple(
        tuple(right if left == 0 else 0 for left, right in zip(a_row, b_row, strict=True))
        for a_row, b_row in zip(a, b, strict=True)
    )


def _reduce_or(cells: tuple[Grid, ...]) -> Grid:
    height, width = grid_shape(cells[0])
    rows = [[0 for _ in range(width)] for _ in range(height)]
    for cell in cells:
        for row_index, row in enumerate(cell):
            for col_index, value in enumerate(row):
                if value != 0:
                    rows[row_index][col_index] = value
    return tuple(tuple(row) for row in rows)


def _reduce_majority(cells: tuple[Grid, ...]) -> Grid:
    height, width = grid_shape(cells[0])
    threshold = len(cells) // 2
    rows = []
    for row_index in range(height):
        row = []
        for col_index in range(width):
            values = [
                cell[row_index][col_index]
                for cell in cells
                if cell[row_index][col_index] != 0
            ]
            if len(values) > threshold:
                row.append(Counter(values).most_common(1)[0][0])
            else:
                row.append(0)
        rows.append(tuple(row))
    return tuple(rows)


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
class BooleanHalvesStrategy:
    name: str = "arc-boolean-halves"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        first_example = context.task.train[0]
        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []

        orientations: list[tuple[str, Callable[[Grid], tuple[Grid, Grid]]]] = []
        if all(
            len(example.input) == len(example.output) * 2
            and (len(example.input[0]) if example.input else 0)
            == (len(example.output[0]) if example.output else 0)
            for example in context.task.train
        ):
            orientations.append(
                (
                    "top_bottom",
                    lambda grid: (
                        grid[: len(grid) // 2],
                        grid[len(grid) // 2 :],
                    ),
                )
            )
        if all(
            len(example.input) == len(example.output)
            and (len(example.input[0]) if example.input else 0)
            == (len(example.output[0]) if example.output else 0) * 2
            for example in context.task.train
        ):
            orientations.append(
                (
                    "left_right",
                    lambda grid: (
                        tuple(row[: len(row) // 2] for row in grid),
                        tuple(row[len(row) // 2 :] for row in grid),
                    ),
                )
            )

        operations = (
            ("or", _op_or),
            ("and", _op_and),
            ("xor", _op_xor),
            ("a_minus_b", _op_a_minus_b),
            ("b_minus_a", _op_b_minus_a),
        )

        for orientation_name, splitter in orientations:
            left, right = splitter(first_example.input)
            for op_name, operation in operations:
                base_output = operation(left, right)
                for variant_name, target_color in _variants_for_first_example(
                    base_output,
                    first_example.output,
                ):
                    def _make_executor(
                        splitter=splitter,
                        operation=operation,
                        variant_name=variant_name,
                        target_color=target_color,
                    ):
                        def executor(grid: Grid) -> Grid:
                            left, right = splitter(grid)
                            return _apply_variant(operation(left, right), variant_name, target_color)
                        return executor

                    program_name = f"cross-ref-{orientation_name}-{op_name}"
                    if variant_name == "recolor_nonzero":
                        program_name += f"-recolor-{target_color}"
                    candidate_specs.append(
                        (
                            program_name,
                            {
                                "type": "cross_reference_boolean_halves",
                                "orientation": orientation_name,
                                "operation": op_name,
                                "variant": variant_name,
                                "target_color": target_color,
                            },
                            _make_executor(),
                            4 if variant_name == "raw" else 5,
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
        notes = () if candidates else ("no consistent boolean-halves program matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )


@dataclass(frozen=True)
class SeparatorCrossReferenceStrategy:
    name: str = "arc-separator-cross-reference"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        row_separators = intersect_separators(
            find_uniform_row_separators(example.input)
            for example in context.task.train
        )
        col_separators = intersect_separators(
            find_uniform_col_separators(example.input)
            for example in context.task.train
        )
        if not row_separators and not col_separators:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("no shared separator structure found across training inputs",),
            )

        first_example = context.task.train[0]
        first_cells = split_grid_by_separators(first_example.input, row_separators, col_separators)
        flat_cells = flatten_cells(first_cells)
        if not flat_cells:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("separator structure did not yield any non-empty cells",),
            )

        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []
        for row_index, col_index, cell in flat_cells:
            for variant_name, target_color in _variants_for_first_example(cell, first_example.output):
                def _make_extract_executor(
                    row_index=row_index,
                    col_index=col_index,
                    row_separators=row_separators,
                    col_separators=col_separators,
                    variant_name=variant_name,
                    target_color=target_color,
                ):
                    def executor(grid: Grid) -> Grid:
                        cells = split_grid_by_separators(grid, row_separators, col_separators)
                        cell = cells[row_index][col_index]
                        return _apply_variant(cell, variant_name, target_color)
                    return executor

                name = f"cross-ref-cell-{row_index}-{col_index}"
                if variant_name == "recolor_nonzero":
                    name += f"-recolor-{target_color}"
                candidate_specs.append(
                    (
                        name,
                        {
                            "type": "cross_reference_extract_cell",
                            "row_index": row_index,
                            "col_index": col_index,
                            "row_separators": row_separators,
                            "col_separators": col_separators,
                            "variant": variant_name,
                            "target_color": target_color,
                        },
                        _make_extract_executor(),
                        5 if variant_name == "raw" else 6,
                    )
                )

        if cells_share_shape(first_cells):
            output_shape = grid_shape(first_example.output)
            operations = (
                ("or", _op_or),
                ("and", _op_and),
                ("xor", _op_xor),
                ("a_minus_b", _op_a_minus_b),
            )
            for left_row, left_col, left_cell in flat_cells:
                for right_row, right_col, right_cell in flat_cells:
                    if (left_row, left_col) == (right_row, right_col):
                        continue
                    if grid_shape(left_cell) != output_shape or grid_shape(right_cell) != output_shape:
                        continue
                    for op_name, operation in operations:
                        base_output = operation(left_cell, right_cell)
                        for variant_name, target_color in _variants_for_first_example(
                            base_output,
                            first_example.output,
                        ):
                            def _make_binary_executor(
                                left_row=left_row,
                                left_col=left_col,
                                right_row=right_row,
                                right_col=right_col,
                                row_separators=row_separators,
                                col_separators=col_separators,
                                operation=operation,
                                variant_name=variant_name,
                                target_color=target_color,
                            ):
                                def executor(grid: Grid) -> Grid:
                                    cells = split_grid_by_separators(grid, row_separators, col_separators)
                                    left_cell = cells[left_row][left_col]
                                    right_cell = cells[right_row][right_col]
                                    return _apply_variant(
                                        operation(left_cell, right_cell),
                                        variant_name,
                                        target_color,
                                    )
                                return executor

                            name = (
                                f"cross-ref-cell-{left_row}-{left_col}-{op_name}"
                                f"-cell-{right_row}-{right_col}"
                            )
                            if variant_name == "recolor_nonzero":
                                name += f"-recolor-{target_color}"
                            candidate_specs.append(
                                (
                                    name,
                                    {
                                        "type": "cross_reference_cell_binary",
                                        "left": (left_row, left_col),
                                        "right": (right_row, right_col),
                                        "operation": op_name,
                                        "row_separators": row_separators,
                                        "col_separators": col_separators,
                                        "variant": variant_name,
                                        "target_color": target_color,
                                    },
                                    _make_binary_executor(),
                                    6 if variant_name == "raw" else 7,
                                )
                            )

            if len(flat_cells) >= 2:
                reductions = (
                    ("or_reduce", _reduce_or),
                    ("majority_reduce", _reduce_majority),
                )
                reduction_cells = tuple(cell for _, _, cell in flat_cells)
                if grid_shape(reduction_cells[0]) == output_shape:
                    for reduction_name, reduction in reductions:
                        base_output = reduction(reduction_cells)
                        for variant_name, target_color in _variants_for_first_example(
                            base_output,
                            first_example.output,
                        ):
                            def _make_reduction_executor(
                                row_separators=row_separators,
                                col_separators=col_separators,
                                reduction=reduction,
                                variant_name=variant_name,
                                target_color=target_color,
                            ):
                                def executor(grid: Grid) -> Grid:
                                    cells = split_grid_by_separators(grid, row_separators, col_separators)
                                    flat_cells = tuple(cell for _, _, cell in flatten_cells(cells))
                                    return _apply_variant(
                                        reduction(flat_cells),
                                        variant_name,
                                        target_color,
                                    )
                                return executor

                            name = f"cross-ref-{reduction_name}-cells"
                            if variant_name == "recolor_nonzero":
                                name += f"-recolor-{target_color}"
                            candidate_specs.append(
                                (
                                    name,
                                    {
                                        "type": "cross_reference_cell_reduction",
                                        "reduction": reduction_name,
                                        "row_separators": row_separators,
                                        "col_separators": col_separators,
                                        "variant": variant_name,
                                        "target_color": target_color,
                                    },
                                    _make_reduction_executor(),
                                    7 if variant_name == "raw" else 8,
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
        notes = () if candidates else ("separator structure found, but no cross-reference program matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
