from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Grid = tuple[tuple[int, ...], ...]


def freeze_grid(grid: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def grid_to_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def grid_shape(grid: Grid) -> tuple[int, int]:
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


def grid_cell_count(grid: Grid) -> int:
    rows, cols = grid_shape(grid)
    return rows * cols


def grid_accuracy(expected: Grid, actual: Grid) -> float:
    if grid_shape(expected) != grid_shape(actual):
        return 0.0
    total = grid_cell_count(expected)
    if total == 0:
        return 1.0
    correct = 0
    for expected_row, actual_row in zip(expected, actual, strict=True):
        for expected_cell, actual_cell in zip(expected_row, actual_row, strict=True):
            if expected_cell == actual_cell:
                correct += 1
    return correct / total


def task_colors(grids: Iterable[Grid]) -> tuple[int, ...]:
    colors = {cell for grid in grids for row in grid for cell in row}
    return tuple(sorted(colors))


@dataclass(frozen=True)
class ArcExample:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTestCase:
    input: Grid
    output: Grid | None = None


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[ArcExample, ...]
    test: tuple[ArcTestCase, ...]

