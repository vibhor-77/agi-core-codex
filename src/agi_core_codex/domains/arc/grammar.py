from __future__ import annotations

from typing import Iterable

from agi_core_codex.core.interfaces import ProgramHandle
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
