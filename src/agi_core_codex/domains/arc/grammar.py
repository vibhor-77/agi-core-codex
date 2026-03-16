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


class ArcGrammar:
    domain = "arc"

    def primitive_count(self, task: ArcTask | None = None) -> int:
        if task is None:
            return 7
        return len(self.enumerate_primitives(task))

    def enumerate_primitives(self, task: ArcTask) -> tuple[ProgramHandle, ...]:
        programs = list(self._static_primitives())
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
