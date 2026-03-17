from __future__ import annotations

from typing import Iterable

from agi_core_codex.core.interfaces import ProgramHandle
from agi_core_codex.domains.arc.analysis import connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, task_colors


def _rotate_90(grid: Grid) -> Grid:
    if not grid:
        return ()
    return freeze_grid(zip(*grid[::-1]))


def _rotate_180(grid: Grid) -> Grid:
    return freeze_grid(row[::-1] for row in grid[::-1])


def _rotate_270(grid: Grid) -> Grid:
    if not grid:
        return ()
    return freeze_grid(list(zip(*grid))[::-1])


def _flip_horizontal(grid: Grid) -> Grid:
    return freeze_grid(row[::-1] for row in grid)


def _flip_vertical(grid: Grid) -> Grid:
    return freeze_grid(grid[::-1])


def _transpose(grid: Grid) -> Grid:
    if not grid:
        return ()
    return freeze_grid(zip(*grid))


def _replace_color(grid: Grid, source: int, target: int) -> Grid:
    return freeze_grid(
        tuple(target if cell == source else cell for cell in row)
        for row in grid
    )


def _swap_colors(grid: Grid, first: int, second: int) -> Grid:
    return freeze_grid(
        tuple(
            second if cell == first else first if cell == second else cell
            for cell in row
        )
        for row in grid
    )


def _recolor_foreground(grid: Grid, color: int) -> Grid:
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    background = max(counts.items(), key=lambda item: item[1])[0]
    return freeze_grid(
        tuple(color if cell != background else cell for cell in row)
        for row in grid
    )


def _crop_nonzero(grid: Grid) -> Grid:
    coords = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell != 0
    ]
    if not coords:
        return ()
    rows = [row for row, _ in coords]
    cols = [col for _, col in coords]
    row_start, row_end = min(rows), max(rows)
    col_start, col_end = min(cols), max(cols)
    return freeze_grid(
        row[col_start : col_end + 1]
        for row in grid[row_start : row_end + 1]
    )


def _gravity_down(grid: Grid) -> Grid:
    if not grid:
        return ()
    height = len(grid)
    width = len(grid[0])
    rows = [[0 for _ in range(width)] for _ in range(height)]
    for col_index in range(width):
        nonzero = [grid[row_index][col_index] for row_index in range(height) if grid[row_index][col_index] != 0]
        for offset, value in enumerate(nonzero):
            rows[height - len(nonzero) + offset][col_index] = value
    return freeze_grid(rows)


def _gravity_up(grid: Grid) -> Grid:
    if not grid:
        return ()
    height = len(grid)
    width = len(grid[0])
    rows = [[0 for _ in range(width)] for _ in range(height)]
    for col_index in range(width):
        nonzero = [grid[row_index][col_index] for row_index in range(height) if grid[row_index][col_index] != 0]
        for offset, value in enumerate(nonzero):
            rows[offset][col_index] = value
    return freeze_grid(rows)


def _mirror_tile_horizontal(grid: Grid) -> Grid:
    return freeze_grid(tuple(row) + tuple(reversed(row)) for row in grid)


def _mirror_tile_vertical(grid: Grid) -> Grid:
    return freeze_grid(list(grid) + list(reversed(grid)))


def _tile_horizontal(grid: Grid) -> Grid:
    return freeze_grid(tuple(row) + tuple(row) for row in grid)


def _mirror_tile_both(grid: Grid) -> Grid:
    top = [tuple(row) + tuple(reversed(row)) for row in grid]
    bottom = [tuple(row) + tuple(reversed(row)) for row in reversed(grid)]
    return freeze_grid(top + bottom)


def _rotate_tile_clockwise(grid: Grid) -> Grid:
    if not grid:
        return ()
    height = len(grid)
    width = len(grid[0])
    if height != width:
        return grid
    rotate_90 = freeze_grid(
        tuple(grid[height - 1 - col_index][row_index] for col_index in range(height))
        for row_index in range(width)
    )
    rotate_270 = freeze_grid(
        tuple(grid[col_index][width - 1 - row_index] for col_index in range(height))
        for row_index in range(width)
    )
    rotate_180 = freeze_grid(row[::-1] for row in grid[::-1])
    top = [tuple(grid[row_index]) + tuple(rotate_90[row_index]) for row_index in range(height)]
    bottom = [tuple(rotate_270[row_index]) + tuple(rotate_180[row_index]) for row_index in range(height)]
    return freeze_grid(top + bottom)


def _inpaint_by_symmetry(grid: Grid) -> Grid:
    if not grid:
        return ()
    colors = {cell for row in grid for cell in row if cell != 0}
    if not colors:
        return grid

    height = len(grid)
    width = len(grid[0])
    best_result: Grid | None = None
    best_score = (height * width, height * width)

    for mask_color in colors:
        rows = [list(row) for row in grid]
        for _ in range(4):
            changed = False
            for row_index in range(height):
                for col_index in range(width):
                    if rows[row_index][col_index] != mask_color:
                        continue
                    for mirror_row, mirror_col in (
                        (row_index, width - 1 - col_index),
                        (height - 1 - row_index, col_index),
                        (height - 1 - row_index, width - 1 - col_index),
                    ):
                        if rows[mirror_row][mirror_col] != mask_color:
                            rows[row_index][col_index] = rows[mirror_row][mirror_col]
                            changed = True
                            break
                    else:
                        if height == width and rows[col_index][row_index] != mask_color:
                            rows[row_index][col_index] = rows[col_index][row_index]
                            changed = True
            if not changed:
                break

        remaining = sum(1 for row in rows for cell in row if cell == mask_color)
        non_mask_changes = sum(
            1
            for row_index, row in enumerate(rows)
            for col_index, cell in enumerate(row)
            if grid[row_index][col_index] != mask_color and cell != grid[row_index][col_index]
        )
        score = (remaining, non_mask_changes)
        if score < best_score:
            best_score = score
            best_result = freeze_grid(rows)

    return best_result if best_result is not None else grid


def _extract_largest_cc(grid: Grid) -> Grid:
    components = connected_components(grid)
    if not components:
        return grid
    component = max(components, key=lambda item: item.size)
    row_start, col_start, row_end, col_end = component.bbox
    return freeze_grid(
        row[col_start : col_end + 1]
        for row in grid[row_start : row_end + 1]
    )


def _extract_unique_color_region(grid: Grid) -> Grid:
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            if cell == 0:
                continue
            counts[cell] = counts.get(cell, 0) + 1
    if not counts:
        return grid
    minimum = min(counts.values())
    target = min(color for color, count in counts.items() if count == minimum)
    positions = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell == target
    ]
    row_indices = [row_index for row_index, _ in positions]
    col_indices = [col_index for _, col_index in positions]
    row_start, row_end = min(row_indices), max(row_indices)
    col_start, col_end = min(col_indices), max(col_indices)
    return freeze_grid(
        row[col_start : col_end + 1]
        for row in grid[row_start : row_end + 1]
    )


def _inpaint_periodic(grid: Grid) -> Grid:
    if not grid:
        return ()
    height = len(grid)
    width = len(grid[0])
    if not any(cell == 0 for row in grid for cell in row):
        return grid

    for period_height in range(1, height + 1):
        for period_width in range(1, width + 1):
            tile: list[list[int | None]] = [
                [None for _ in range(period_width)]
                for _ in range(period_height)
            ]
            consistent = True
            for row_index in range(height):
                if not consistent:
                    break
                for col_index in range(width):
                    value = grid[row_index][col_index]
                    if value == 0:
                        continue
                    tile_row = row_index % period_height
                    tile_col = col_index % period_width
                    seen = tile[tile_row][tile_col]
                    if seen is None:
                        tile[tile_row][tile_col] = value
                    elif seen != value:
                        consistent = False
                        break
            if not consistent:
                continue
            if any(
                tile[tile_row][tile_col] is None
                for tile_row in range(period_height)
                for tile_col in range(period_width)
            ):
                continue

            rows = [list(row) for row in grid]
            for row_index in range(height):
                for col_index in range(width):
                    if rows[row_index][col_index] == 0:
                        rows[row_index][col_index] = tile[row_index % period_height][col_index % period_width] or 0
            return freeze_grid(rows)

    return grid


class ArcGrammar:
    domain = "arc"

    def primitive_count(self, task: ArcTask | None = None) -> int:
        if task is None:
            return 7
        return len(self.enumerate_primitives(task))

    def enumerate_primitives(self, task: ArcTask) -> tuple[ProgramHandle, ...]:
        programs = list(self._static_primitives())
        programs.append(
            make_arc_program(
                name="crop-nonzero",
                semantics={"type": "crop_nonzero"},
                executor=_crop_nonzero,
                complexity=2,
            )
        )
        programs.append(
            make_arc_program(
                name="gravity-down",
                semantics={"type": "gravity_down"},
                executor=_gravity_down,
                complexity=2,
            )
        )
        programs.append(
            make_arc_program(
                name="gravity-up",
                semantics={"type": "gravity_up"},
                executor=_gravity_up,
                complexity=2,
            )
        )
        programs.append(
            make_arc_program(
                name="mirror-tile-horizontal",
                semantics={"type": "mirror_tile_horizontal"},
                executor=_mirror_tile_horizontal,
                complexity=3,
            )
        )
        programs.append(
            make_arc_program(
                name="mirror-tile-vertical",
                semantics={"type": "mirror_tile_vertical"},
                executor=_mirror_tile_vertical,
                complexity=3,
            )
        )
        programs.append(
            make_arc_program(
                name="tile-horizontal",
                semantics={"type": "tile_horizontal"},
                executor=_tile_horizontal,
                complexity=3,
            )
        )
        programs.append(
            make_arc_program(
                name="mirror-tile-both",
                semantics={"type": "mirror_tile_both"},
                executor=_mirror_tile_both,
                complexity=3,
            )
        )
        programs.append(
            make_arc_program(
                name="rotate-tile-clockwise",
                semantics={"type": "rotate_tile_clockwise"},
                executor=_rotate_tile_clockwise,
                complexity=4,
            )
        )
        programs.append(
            make_arc_program(
                name="inpaint-by-symmetry",
                semantics={"type": "inpaint_by_symmetry"},
                executor=_inpaint_by_symmetry,
                complexity=4,
            )
        )
        programs.append(
            make_arc_program(
                name="extract-largest-cc",
                semantics={"type": "extract_largest_cc"},
                executor=_extract_largest_cc,
                complexity=3,
            )
        )
        programs.append(
            make_arc_program(
                name="extract-unique-color-region",
                semantics={"type": "extract_unique_color_region"},
                executor=_extract_unique_color_region,
                complexity=3,
            )
        )
        programs.append(
            make_arc_program(
                name="inpaint-periodic",
                semantics={"type": "inpaint_periodic"},
                executor=_inpaint_periodic,
                complexity=4,
            )
        )
        colors = task_colors(
            [example.input for example in task.train]
            + [example.output for example in task.train]
        )
        for source in colors:
            for target in colors:
                if source == target:
                    continue
                programs.append(
                    make_arc_program(
                        name=f"replace-color-{source}-to-{target}",
                        semantics={"type": "replace_color", "source": source, "target": target},
                        executor=lambda grid, source=source, target=target: _replace_color(
                            grid,
                            source,
                            target,
                        ),
                        complexity=2,
                    )
                )
        for index, first in enumerate(colors):
            for second in colors[index + 1 :]:
                programs.append(
                    make_arc_program(
                        name=f"swap-colors-{first}-with-{second}",
                        semantics={"type": "swap_colors", "first": first, "second": second},
                        executor=lambda grid, first=first, second=second: _swap_colors(
                            grid,
                            first,
                            second,
                        ),
                        complexity=2,
                    )
                )
        for color in colors:
            programs.append(
                make_arc_program(
                    name=f"recolor-foreground-{color}",
                    semantics={"type": "recolor_foreground", "color": color},
                    executor=lambda grid, color=color: _recolor_foreground(grid, color),
                    complexity=2,
                )
            )

        return tuple(sorted(programs, key=lambda program: (program.cost.complexity, program.name, program.id)))

    def _static_primitives(self) -> Iterable[ProgramHandle]:
        return (
            make_arc_program(
                name="identity",
                semantics={"type": "identity"},
                executor=lambda grid: grid,
            ),
            make_arc_program(
                name="rotate-90",
                semantics={"type": "rotate_90"},
                executor=_rotate_90,
            ),
            make_arc_program(
                name="rotate-180",
                semantics={"type": "rotate_180"},
                executor=_rotate_180,
            ),
            make_arc_program(
                name="rotate-270",
                semantics={"type": "rotate_270"},
                executor=_rotate_270,
            ),
            make_arc_program(
                name="flip-horizontal",
                semantics={"type": "flip_horizontal"},
                executor=_flip_horizontal,
            ),
            make_arc_program(
                name="flip-vertical",
                semantics={"type": "flip_vertical"},
                executor=_flip_vertical,
            ),
            make_arc_program(
                name="transpose",
                semantics={"type": "transpose"},
                executor=_transpose,
            ),
        )
