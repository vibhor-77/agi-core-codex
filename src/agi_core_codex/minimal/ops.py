from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


Grid = tuple[tuple[int, ...], ...]
UnaryGridOp = Callable[[Grid], Grid]
BinaryGridOp = Callable[[Grid, Grid], Grid]


def freeze_grid(grid) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def grid_shape(grid: Grid) -> tuple[int, int]:
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


def grid_accuracy(expected: Grid, actual: Grid) -> float:
    if grid_shape(expected) != grid_shape(actual):
        return 0.0
    total = len(expected) * len(expected[0]) if expected else 0
    if total == 0:
        return 1.0
    correct = 0
    for expected_row, actual_row in zip(expected, actual, strict=True):
        for expected_cell, actual_cell in zip(expected_row, actual_row, strict=True):
            if expected_cell == actual_cell:
                correct += 1
    return correct / total


def identity(grid: Grid) -> Grid:
    return grid


def flip_h(grid: Grid) -> Grid:
    return freeze_grid(row[::-1] for row in grid)


def flip_v(grid: Grid) -> Grid:
    return freeze_grid(grid[::-1])


def transpose(grid: Grid) -> Grid:
    if not grid:
        return ()
    return freeze_grid(zip(*grid))


def crop_support(grid: Grid) -> Grid:
    coords = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell != 0
    ]
    if not coords:
        return ()
    rows = [row_index for row_index, _ in coords]
    cols = [col_index for _, col_index in coords]
    row_start, row_end = min(rows), max(rows)
    col_start, col_end = min(cols), max(cols)
    return freeze_grid(
        row[col_start : col_end + 1]
        for row in grid[row_start : row_end + 1]
    )


def overlay(left: Grid, right: Grid) -> Grid:
    if grid_shape(left) != grid_shape(right):
        raise ValueError("overlay requires matching shapes")
    return freeze_grid(
        tuple(right_cell if right_cell != 0 else left_cell for left_cell, right_cell in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def hcat(left: Grid, right: Grid) -> Grid:
    if not left or not right:
        raise ValueError("hcat requires non-empty grids")
    if len(left) != len(right):
        raise ValueError("hcat requires matching heights")
    return freeze_grid(
        tuple(left_row) + tuple(right_row)
        for left_row, right_row in zip(left, right, strict=True)
    )


def vcat(left: Grid, right: Grid) -> Grid:
    if not left or not right:
        raise ValueError("vcat requires non-empty grids")
    if len(left[0]) != len(right[0]):
        raise ValueError("vcat requires matching widths")
    return freeze_grid(tuple(left) + tuple(right))


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    arity: int
    complexity: int
    unary: UnaryGridOp | None = None
    binary: BinaryGridOp | None = None


def unary_seed_specs() -> tuple[PrimitiveSpec, ...]:
    return (
        PrimitiveSpec("identity", arity=1, complexity=1, unary=identity),
        PrimitiveSpec("flip_h", arity=1, complexity=1, unary=flip_h),
        PrimitiveSpec("flip_v", arity=1, complexity=1, unary=flip_v),
        PrimitiveSpec("transpose", arity=1, complexity=1, unary=transpose),
        PrimitiveSpec("crop_support", arity=1, complexity=2, unary=crop_support),
    )


def compositor_specs() -> tuple[PrimitiveSpec, ...]:
    return (
        PrimitiveSpec("chain", arity=2, complexity=1),
        PrimitiveSpec("overlay", arity=2, complexity=1, binary=overlay),
        PrimitiveSpec("hcat", arity=2, complexity=1, binary=hcat),
        PrimitiveSpec("vcat", arity=2, complexity=1, binary=vcat),
    )
