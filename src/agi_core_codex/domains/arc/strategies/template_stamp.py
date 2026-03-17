from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agi_core_codex.core.interfaces import CostModel, StrategyResult
from agi_core_codex.domains.arc.analysis import background_color, connected_components
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


def _build_template(component) -> Grid:
    row0, col0, row1, col1 = component.bbox
    height = row1 - row0 + 1
    width = col1 - col0 + 1
    rows = [[0] * width for _ in range(height)]
    for row, col in component.pixels:
        rows[row - row0][col - col0] = component.color
    return freeze_grid(rows)


def _shape_signature(component) -> tuple[tuple[int, int], ...]:
    row0, col0, _, _ = component.bbox
    return tuple(sorted((row - row0, col - col0) for row, col in component.pixels))


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
class TemplateStampStrategy:
    name: str = "arc-template-stamp"
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def applies(self, task: Any) -> bool:
        return isinstance(task, ArcTask) and bool(task.train)

    def run(self, context: Any) -> StrategyResult:
        if not context.start_strategy(self.name, self.cost_model):
            return context.empty_strategy_result(name=self.name, status="budget_exhausted")

        if any(grid_shape(example.input) != grid_shape(example.output) for example in context.task.train):
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("template stamping only applies to same-shape input/output tasks",),
            )

        first_example = context.task.train[0]
        bg = background_color(first_example.input)
        components = sorted(
            connected_components(first_example.input, bg_color=bg),
            key=lambda component: component.size,
        )
        if len(components) < 2:
            return context.finish_strategy(
                name=self.name,
                status="not_applicable",
                generated=0,
                candidates=(),
                notes=("need at least a template component and a marker color",),
            )

        candidate_specs: list[tuple[str, dict[str, Any], Callable[[Grid], Grid], int]] = []
        for template_component in components[: min(3, len(components))]:
            if template_component.size > 25:
                continue
            template_signature = _shape_signature(template_component)
            template = _build_template(template_component)
            template_height, template_width = grid_shape(template)

            def _make_executor(
                template_signature=template_signature,
                bg=bg,
                template_height=template_height,
                template_width=template_width,
            ):
                def executor(grid: Grid) -> Grid:
                    components = connected_components(grid, bg_color=bg)
                    matching_components = [
                        component
                        for component in components
                        if component.size == len(template_signature)
                        and _shape_signature(component) == template_signature
                    ]
                    if not matching_components:
                        return grid
                    template_component = sorted(
                        matching_components,
                        key=lambda component: component.bbox,
                    )[0]
                    template = _build_template(template_component)

                    marker_components = [
                        component
                        for component in components
                        if component.size == 1 and component.pixels != template_component.pixels
                    ]
                    rows = [list(row) for row in grid]
                    markers = [component.pixels[0] for component in marker_components]
                    for row_index, col_index in markers:
                        rows[row_index][col_index] = bg
                    for marker_row, marker_col in markers:
                        start_row = marker_row - template_height // 2
                        start_col = marker_col - template_width // 2
                        for template_row in range(template_height):
                            for template_col in range(template_width):
                                value = template[template_row][template_col]
                                if value == 0:
                                    continue
                                out_row = start_row + template_row
                                out_col = start_col + template_col
                                if 0 <= out_row < len(rows) and 0 <= out_col < len(rows[out_row]):
                                    rows[out_row][out_col] = value
                    return freeze_grid(rows)

                return executor

            executor = _make_executor()
            if executor(first_example.input) != first_example.output:
                continue
            candidate_specs.append(
                (
                    f"template-stamp-shape-{len(template_signature)}",
                    {
                        "type": "template_stamp",
                        "template_shape": template_signature,
                        "background_color": bg,
                    },
                    executor,
                    max(3, template_component.size),
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
        notes = () if candidates else ("no template/marker stamping rule matched the first example",)
        return context.finish_strategy(
            name=self.name,
            status=status,
            generated=generated,
            candidates=candidates,
            notes=notes,
        )
