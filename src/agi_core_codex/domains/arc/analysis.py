from __future__ import annotations

from collections.abc import Iterable, Sequence

from agi_core_codex.domains.arc.types import Grid, freeze_grid, grid_shape


def nonzero_mask(grid: Grid) -> Grid:
    return freeze_grid(
        tuple(1 if cell != 0 else 0 for cell in row)
        for row in grid
    )


def single_nonzero_color(grid: Grid) -> int | None:
    colors = {cell for row in grid for cell in row if cell != 0}
    if len(colors) != 1:
        return None
    return next(iter(colors))


def recolor_nonzero(grid: Grid, color: int) -> Grid:
    return freeze_grid(
        tuple(color if cell != 0 else 0 for cell in row)
        for row in grid
    )


def find_uniform_row_separators(grid: Grid) -> tuple[int, ...]:
    separators = []
    for row_index, row in enumerate(grid):
        if row and len(set(row)) == 1 and row[0] != 0:
            separators.append(row_index)
    return tuple(separators)


def find_uniform_col_separators(grid: Grid) -> tuple[int, ...]:
    height, width = grid_shape(grid)
    separators = []
    for col_index in range(width):
        column = [grid[row_index][col_index] for row_index in range(height)]
        if column and len(set(column)) == 1 and column[0] != 0:
            separators.append(col_index)
    return tuple(separators)


def intersect_separators(separator_sets: Iterable[Sequence[int]]) -> tuple[int, ...]:
    normalized = [set(separator_set) for separator_set in separator_sets]
    if not normalized:
        return ()
    shared = set.intersection(*normalized)
    return tuple(sorted(shared))


def _segments(length: int, separators: Sequence[int]) -> tuple[tuple[int, int], ...]:
    start = 0
    segments = []
    for separator in sorted(separators):
        if start < separator:
            segments.append((start, separator))
        start = separator + 1
    if start < length:
        segments.append((start, length))
    return tuple(segments)


def split_grid_by_separators(
    grid: Grid,
    row_separators: Sequence[int],
    col_separators: Sequence[int],
) -> tuple[tuple[Grid, ...], ...]:
    height, width = grid_shape(grid)
    row_segments = _segments(height, row_separators)
    col_segments = _segments(width, col_separators)
    cells = []
    for row_start, row_end in row_segments:
        row_cells = []
        for col_start, col_end in col_segments:
            row_cells.append(
                freeze_grid(
                    grid[row_index][col_start:col_end]
                    for row_index in range(row_start, row_end)
                )
            )
        cells.append(tuple(row_cells))
    return tuple(cells)


def flatten_cells(cells: tuple[tuple[Grid, ...], ...]) -> tuple[tuple[int, int, Grid], ...]:
    return tuple(
        (row_index, col_index, cell)
        for row_index, row in enumerate(cells)
        for col_index, cell in enumerate(row)
    )


def cells_share_shape(cells: tuple[tuple[Grid, ...], ...]) -> bool:
    flat = flatten_cells(cells)
    if not flat:
        return False
    expected_shape = grid_shape(flat[0][2])
    return all(grid_shape(cell) == expected_shape for _, _, cell in flat)

