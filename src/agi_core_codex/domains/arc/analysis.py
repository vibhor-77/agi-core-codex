from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections import Counter
from dataclasses import dataclass

from agi_core_codex.domains.arc.types import Grid, freeze_grid, grid_shape


@dataclass(frozen=True)
class Component:
    color: int
    pixels: tuple[tuple[int, int], ...]
    size: int
    bbox: tuple[int, int, int, int]


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


def fill_enclosed(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    nonzero = [
        grid[row_index][col_index]
        for row_index in range(height)
        for col_index in range(width)
        if grid[row_index][col_index] != 0
    ]
    if not nonzero:
        return grid

    fill_color = Counter(nonzero).most_common(1)[0][0]
    reachable: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = []
    for row_index in range(height):
        for col_index in (0, width - 1):
            if grid[row_index][col_index] == 0 and (row_index, col_index) not in reachable:
                reachable.add((row_index, col_index))
                stack.append((row_index, col_index))
    for col_index in range(width):
        for row_index in (0, height - 1):
            if grid[row_index][col_index] == 0 and (row_index, col_index) not in reachable:
                reachable.add((row_index, col_index))
                stack.append((row_index, col_index))

    while stack:
        current_row, current_col = stack.pop()
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row = current_row + delta_row
            next_col = current_col + delta_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if grid[next_row][next_col] != 0 or (next_row, next_col) in reachable:
                continue
            reachable.add((next_row, next_col))
            stack.append((next_row, next_col))

    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] == 0 and (row_index, col_index) not in reachable:
                rows[row_index][col_index] = fill_color
    return freeze_grid(rows)


def extract_enclosed_interior(grid: Grid) -> Grid:
    filled = fill_enclosed(grid)
    return freeze_grid(
        tuple(
            0 if grid[row_index][col_index] != 0 else filled[row_index][col_index]
            for col_index in range(len(row))
        )
        for row_index, row in enumerate(grid)
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


def merge_cells_by_separators(
    base_grid: Grid,
    row_separators: Sequence[int],
    col_separators: Sequence[int],
    cells: tuple[tuple[Grid, ...], ...],
) -> Grid:
    height, width = grid_shape(base_grid)
    row_segments = _segments(height, row_separators)
    col_segments = _segments(width, col_separators)
    rows = [list(row) for row in base_grid]
    for row_index, (row_start, row_end) in enumerate(row_segments):
        for col_index, (col_start, col_end) in enumerate(col_segments):
            cell = cells[row_index][col_index]
            for local_row, global_row in enumerate(range(row_start, row_end)):
                for local_col, global_col in enumerate(range(col_start, col_end)):
                    rows[global_row][global_col] = cell[local_row][local_col]
    return freeze_grid(rows)


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


def background_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    return counts.most_common(1)[0][0]


def connected_components(grid: Grid, *, bg_color: int | None = None) -> tuple[Component, ...]:
    height, width = grid_shape(grid)
    if bg_color is None:
        bg_color = background_color(grid)

    visited: set[tuple[int, int]] = set()
    components: list[Component] = []
    for row_index in range(height):
        for col_index in range(width):
            color = grid[row_index][col_index]
            if color == bg_color or (row_index, col_index) in visited:
                continue

            stack = [(row_index, col_index)]
            pixels = []
            visited.add((row_index, col_index))
            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row, current_col))
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in visited:
                        continue
                    if grid[next_row][next_col] != color:
                        continue
                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

            rows = [row for row, _ in pixels]
            cols = [col for _, col in pixels]
            components.append(
                Component(
                    color=color,
                    pixels=tuple(sorted(pixels)),
                    size=len(pixels),
                    bbox=(min(rows), min(cols), max(rows), max(cols)),
                )
            )
    return tuple(components)
