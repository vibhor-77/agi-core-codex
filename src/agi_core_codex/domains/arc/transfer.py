from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from agi_core_codex.core.hashing import stable_hash
from agi_core_codex.core.interfaces import (
    CompiledProgram,
    CostModel,
    Hypothesis,
    TaskRepresentation,
    VerificationResult,
)
from agi_core_codex.core.strategies import HypothesisFamilyStrategy
from agi_core_codex.domains.arc.analysis import (
    background_color,
    connected_components,
    fill_enclosed,
    find_uniform_col_separators,
    find_uniform_row_separators,
)
from agi_core_codex.domains.arc.analyze import _object_local_transform
from agi_core_codex.domains.arc.grammar import (
    _crop_nonzero,
    _extract_largest_cc,
    _extract_unique_color_region,
    _flip_horizontal,
    _flip_vertical,
    _mirror_tile_both,
    _mirror_tile_horizontal,
    _mirror_tile_vertical,
    _rotate_180,
)
from agi_core_codex.domains.arc.programs import ArcProgramExecutor, make_arc_program
from agi_core_codex.domains.arc.scorer import ArcScorer
from agi_core_codex.domains.arc.strategies.barrier_directional_fill import _fill_relative_to_barrier
from agi_core_codex.domains.arc.strategies.motif_completion import _complete_motifs
from agi_core_codex.domains.arc.strategies.separator_propagation import _propagate_matching_cells
from agi_core_codex.domains.arc.types import ArcTask, Grid, grid_shape


def _grid_columns(grid: Grid) -> tuple[tuple[int, ...], ...]:
    if not grid:
        return ()
    return tuple(tuple(row[col_index] for row in grid) for col_index in range(len(grid[0])))


def _axis_symmetry(grid: Grid) -> tuple[str, ...]:
    axes: list[str] = []
    if grid and grid == _flip_horizontal(grid):
        axes.append("vertical")
    if grid and grid == _flip_vertical(grid):
        axes.append("horizontal")
    if grid and len(grid) == len(grid[0]) and grid == tuple(zip(*grid)):
        axes.append("main_diagonal")
    return tuple(axes)


def _smallest_period(sequence: Sequence[object]) -> int | None:
    if not sequence:
        return None
    for period in range(1, len(sequence) + 1):
        if all(sequence[index] == sequence[index % period] for index in range(len(sequence))):
            return period
    return None


def _shape_signature(component) -> tuple[tuple[int, int], ...]:
    row_start, col_start, _, _ = component.bbox
    return tuple(sorted((row - row_start, col - col_start) for row, col in component.pixels))


def _repeated_template_count(grid: Grid) -> int:
    components = connected_components(grid)
    if not components:
        return 0
    counts = Counter((_shape_signature(component), component.color) for component in components)
    return sum(1 for count in counts.values() if count > 1)


def _dominant_nonzero_color(task: ArcTask) -> int | None:
    counts: Counter[int] = Counter()
    for example in task.train:
        for row in example.input:
            counts.update(cell for cell in row if cell != 0)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _rarest_nonzero_color(task: ArcTask) -> int | None:
    counts: Counter[int] = Counter()
    for example in task.train:
        for row in example.input:
            counts.update(cell for cell in row if cell != 0)
    if not counts:
        return None
    minimum = min(counts.values())
    return min(color for color, count in counts.items() if count == minimum)


def build_arc_task_representation(task: ArcTask) -> TaskRepresentation:
    first_input = task.train[0].input
    first_output = task.train[0].output
    same_shape = all(grid_shape(example.input) == grid_shape(example.output) for example in task.train)
    components_per_example = tuple(len(connected_components(example.input)) for example in task.train)
    component_count_max = max(components_per_example, default=0)
    repeated_templates = _repeated_template_count(first_input)
    row_separators = find_uniform_row_separators(first_input)
    col_separators = find_uniform_col_separators(first_input)
    row_period = _smallest_period(first_input)
    col_period = _smallest_period(_grid_columns(first_input))
    axes = _axis_symmetry(first_input)
    barrier_colors = tuple(
        sorted(
            {
                row[0]
                for row in first_input
                if row and row[0] != 0 and all(value == row[0] for value in row)
            }
        )
    )
    background = background_color(first_input)
    dominant = _dominant_nonzero_color(task)
    rarest = _rarest_nonzero_color(task)
    summary_parts = [
        "same-shape" if same_shape else "shape-change",
        f"components<={component_count_max}",
        f"templates={repeated_templates}",
        f"row-seps={len(row_separators)}",
        f"col-seps={len(col_separators)}",
        f"symmetry={','.join(axes) if axes else 'none'}",
        f"period={row_period or 0}x{col_period or 0}",
        f"barriers={','.join(str(color) for color in barrier_colors) if barrier_colors else 'none'}",
    ]
    return TaskRepresentation(
        domain="arc",
        task_key=task.task_id,
        summary="; ".join(summary_parts),
        features={
            "same_shape": same_shape,
            "input_shape": grid_shape(first_input),
            "output_shape": grid_shape(first_output),
            "component_counts": components_per_example,
            "component_count_max": component_count_max,
            "row_separators": row_separators,
            "col_separators": col_separators,
            "separator_count": len(row_separators) + len(col_separators),
            "dominant_nonzero_color": dominant,
            "rarest_nonzero_color": rarest,
            "symmetry_axes": axes,
            "row_period": row_period,
            "col_period": col_period,
            "repeated_templates": repeated_templates,
            "barrier_colors": barrier_colors,
            "background_color": background,
        },
    )


@dataclass(frozen=True)
class ArcTransferFamilyBase:
    name: str
    domain: str = "arc"
    cost_model: CostModel = field(default_factory=lambda: CostModel(complexity=1))

    def build_representation(self, task: Any, environment: Any) -> TaskRepresentation:
        if not isinstance(task, ArcTask):
            raise TypeError("arc transfer families only support ArcTask inputs")
        return build_arc_task_representation(task)


def _make_hypothesis(
    *,
    family_name: str,
    description: str,
    parameters: Mapping[str, Any],
    complexity: int,
) -> Hypothesis:
    return Hypothesis(
        id=stable_hash(
            {
                "family_name": family_name,
                "description": description,
                "parameters": dict(parameters),
            },
            namespace="arc.transfer.hypothesis",
        ),
        family_name=family_name,
        description=description,
        parameters=dict(parameters),
        cost=CostModel(complexity=complexity),
    )


@dataclass(frozen=True)
class GlobalTransformFamily(ArcTransferFamilyBase):
    name: str = "global-transform"

    def propose(self, task: Any, representation: TaskRepresentation) -> Sequence[Hypothesis]:
        same_shape = bool(representation.features.get("same_shape"))
        hypotheses: list[Hypothesis] = []
        if same_shape:
            for executor_key, complexity in (
                ("flip_horizontal", 1),
                ("flip_vertical", 1),
                ("rotate_180", 1),
            ):
                hypotheses.append(
                    _make_hypothesis(
                        family_name=self.name,
                        description=f"global:{executor_key}",
                        parameters={"kind": "global_transform", "executor_key": executor_key},
                        complexity=complexity,
                    )
                )
        for executor_key in ("mirror_tile_horizontal", "mirror_tile_vertical", "mirror_tile_both"):
            hypotheses.append(
                _make_hypothesis(
                    family_name=self.name,
                    description=f"global:{executor_key}",
                    parameters={"kind": "global_transform", "executor_key": executor_key},
                    complexity=2,
                )
            )
        hypotheses.append(
            _make_hypothesis(
                family_name=self.name,
                description="global:crop_nonzero",
                parameters={"kind": "global_transform", "executor_key": "crop_nonzero"},
                complexity=2,
            )
        )
        return tuple(hypotheses)


@dataclass(frozen=True)
class ObjectTransformFamily(ArcTransferFamilyBase):
    name: str = "object-transform"

    def propose(self, task: Any, representation: TaskRepresentation) -> Sequence[Hypothesis]:
        hypotheses = [
            _make_hypothesis(
                family_name=self.name,
                description="object:extract-largest-cc",
                parameters={"kind": "object_transform", "executor_key": "extract_largest_cc"},
                complexity=2,
            ),
            _make_hypothesis(
                family_name=self.name,
                description="object:extract-unique-color-region",
                parameters={"kind": "object_transform", "executor_key": "extract_unique_color_region"},
                complexity=2,
            ),
        ]
        if int(representation.features.get("component_count_max", 0)) > 0:
            for transform_key in ("flip_horizontal", "flip_vertical", "rotate_180"):
                for largest_only in (False, True):
                    suffix = "largest" if largest_only else "all"
                    hypotheses.append(
                        _make_hypothesis(
                            family_name=self.name,
                            description=f"object-local:{transform_key}:{suffix}",
                            parameters={
                                "kind": "object_local_transform",
                                "transform_key": transform_key,
                                "largest_only": largest_only,
                            },
                            complexity=3,
                        )
                    )
        return tuple(hypotheses)


@dataclass(frozen=True)
class RelationPropagationFamily(ArcTransferFamilyBase):
    name: str = "relation-propagation"

    def propose(self, task: Any, representation: TaskRepresentation) -> Sequence[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        if int(representation.features.get("separator_count", 0)) > 0:
            hypotheses.append(
                _make_hypothesis(
                    family_name=self.name,
                    description="relation:separator-propagation",
                    parameters={"kind": "relation_transform", "executor_key": "separator_propagation"},
                    complexity=3,
                )
            )
        if representation.features.get("barrier_colors"):
            hypotheses.append(
                _make_hypothesis(
                    family_name=self.name,
                    description="relation:barrier-directional-fill",
                    parameters={"kind": "relation_transform", "executor_key": "barrier_directional_fill"},
                    complexity=3,
                )
            )
        return tuple(hypotheses)


@dataclass(frozen=True)
class TemplateCompletionFamily(ArcTransferFamilyBase):
    name: str = "template-completion"

    def propose(self, task: Any, representation: TaskRepresentation) -> Sequence[Hypothesis]:
        if not representation.features.get("same_shape"):
            return ()
        return (
            _make_hypothesis(
                family_name=self.name,
                description="template:motif-completion",
                parameters={"kind": "template_transform", "executor_key": "motif_completion"},
                complexity=3,
            ),
        )


@dataclass(frozen=True)
class RegionRoutingFamily(ArcTransferFamilyBase):
    name: str = "region-routing"

    def propose(self, task: Any, representation: TaskRepresentation) -> Sequence[Hypothesis]:
        return (
            _make_hypothesis(
                family_name=self.name,
                description="region:fill-enclosed",
                parameters={"kind": "region_transform", "executor_key": "fill_enclosed"},
                complexity=2,
            ),
        )


_GLOBAL_EXECUTORS: dict[str, tuple[str, ArcProgramExecutor, int]] = {
    "flip_horizontal": ("flip-horizontal", _flip_horizontal, 1),
    "flip_vertical": ("flip-vertical", _flip_vertical, 1),
    "rotate_180": ("rotate-180", _rotate_180, 1),
    "mirror_tile_horizontal": ("mirror-tile-horizontal", _mirror_tile_horizontal, 2),
    "mirror_tile_vertical": ("mirror-tile-vertical", _mirror_tile_vertical, 2),
    "mirror_tile_both": ("mirror-tile-both", _mirror_tile_both, 2),
    "crop_nonzero": ("crop-nonzero", _crop_nonzero, 2),
}

_OBJECT_EXECUTORS: dict[str, tuple[str, ArcProgramExecutor, int]] = {
    "extract_largest_cc": ("extract-largest-cc", _extract_largest_cc, 2),
    "extract_unique_color_region": ("extract-unique-color-region", _extract_unique_color_region, 2),
}


def _compile_object_local_transform(hypothesis: Hypothesis) -> tuple[str, ArcProgramExecutor, int]:
    transform_key = str(hypothesis.parameters["transform_key"])
    largest_only = bool(hypothesis.parameters["largest_only"])
    _, transform, _ = _GLOBAL_EXECUTORS[transform_key]
    suffix = "largest" if largest_only else "all"
    return (
        f"object-local-{transform_key.replace('_', '-')}-{suffix}",
        lambda grid, transform=transform, largest_only=largest_only: _object_local_transform(
            grid,
            transform,
            largest_only=largest_only,
        ),
        3,
    )


def _compile_separator_propagation(task: ArcTask) -> tuple[str, ArcProgramExecutor, int] | None:
    row_separators = find_uniform_row_separators(task.train[0].input)
    col_separators = find_uniform_col_separators(task.train[0].input)
    if not row_separators and not col_separators:
        return None
    return (
        "separator-propagation",
        lambda grid, row_separators=row_separators, col_separators=col_separators: _propagate_matching_cells(
            grid,
            row_separators=row_separators,
            col_separators=col_separators,
        ),
        3,
    )


def _compile_barrier_fill(task: ArcTask) -> tuple[str, ArcProgramExecutor, int] | None:
    first_input = task.train[0].input
    barrier_colors = sorted(
        {
            row[0]
            for row in first_input
            if row and row[0] != 0 and all(value == row[0] for value in row)
        }
    )
    if not barrier_colors:
        return None
    barrier_color = barrier_colors[0]
    nonzero_colors = sorted({cell for row in first_input for cell in row if cell not in {0, barrier_color}})
    if len(nonzero_colors) < 2:
        return None

    for toward_color in nonzero_colors:
        for away_color in nonzero_colors:
            if toward_color == away_color:
                continue
            executor = lambda grid, bc=barrier_color, tc=toward_color, ac=away_color: _fill_relative_to_barrier(
                grid,
                barrier_color=bc,
                toward_barrier_color=tc,
                away_from_barrier_color=ac,
            )
            if all(executor(example.input) == example.output for example in task.train):
                return (
                    f"fill-{toward_color}-toward-barrier-and-{away_color}-away-from-barrier",
                    executor,
                    3,
                )
    return None


def _compile_hypothesis(task: ArcTask, hypothesis: Hypothesis) -> tuple[str, ArcProgramExecutor, int] | None:
    kind = str(hypothesis.parameters.get("kind"))
    if kind == "global_transform":
        return _GLOBAL_EXECUTORS[str(hypothesis.parameters["executor_key"])]
    if kind == "object_transform":
        return _OBJECT_EXECUTORS[str(hypothesis.parameters["executor_key"])]
    if kind == "object_local_transform":
        return _compile_object_local_transform(hypothesis)
    if kind == "relation_transform":
        executor_key = str(hypothesis.parameters["executor_key"])
        if executor_key == "separator_propagation":
            return _compile_separator_propagation(task)
        if executor_key == "barrier_directional_fill":
            return _compile_barrier_fill(task)
    if kind == "template_transform":
        return ("motif-completion", _complete_motifs, 3)
    if kind == "region_transform":
        return ("fill-enclosed", fill_enclosed, 2)
    return None


def _genericity_score(train_accuracy: float, example_accuracies: Sequence[float], complexity: int) -> float:
    if not example_accuracies:
        return 0.0
    consistency = 1.0 - (max(example_accuracies) - min(example_accuracies))
    score = 0.7 * train_accuracy + 0.25 * consistency - 0.03 * complexity
    return round(max(0.0, min(1.0, score)), 4)


def _transfer_proxy_score(
    *,
    train_accuracy: float,
    genericity_score: float,
    representation: TaskRepresentation,
    failure_reason: str | None,
) -> float:
    structure_signal = 0.0
    if int(representation.features.get("separator_count", 0)) > 0:
        structure_signal += 0.08
    if int(representation.features.get("repeated_templates", 0)) > 0:
        structure_signal += 0.08
    if int(representation.features.get("component_count_max", 0)) > 1:
        structure_signal += 0.04
    if failure_reason == "shape_mismatch":
        structure_signal -= 0.08
    score = 0.55 * genericity_score + 0.35 * train_accuracy + structure_signal
    return round(max(0.0, min(1.0, score)), 4)


def _failure_reason(task: ArcTask, executor: ArcProgramExecutor, train_accuracy: float) -> str | None:
    if train_accuracy >= 1.0:
        return None
    shape_mismatch = False
    try:
        for example in task.train:
            predicted = executor(example.input)
            if grid_shape(predicted) != grid_shape(example.output):
                shape_mismatch = True
                break
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"execution_error:{type(exc).__name__}"
    if shape_mismatch:
        return "shape_mismatch"
    if train_accuracy >= 0.95:
        return "near_exact"
    if train_accuracy >= 0.75:
        return "near_miss"
    return "train_mismatch"


class ArcTransferVerifier:
    domain = "arc"

    def __init__(self) -> None:
        self._scorer = ArcScorer()

    def verify(
        self,
        task: Any,
        hypothesis: Hypothesis,
        representation: TaskRepresentation,
    ) -> VerificationResult:
        if not isinstance(task, ArcTask):
            raise TypeError("arc transfer verifier only supports ArcTask inputs")

        compiled = _compile_hypothesis(task, hypothesis)
        if compiled is None:
            return VerificationResult(
                hypothesis=hypothesis,
                score=None,
                compiled_program=None,
                failure_reason="not_compilable",
            )

        name, executor, complexity = compiled
        handle = make_arc_program(
            name=name,
            semantics={
                "type": "arc_transfer",
                "family_name": hypothesis.family_name,
                "hypothesis_id": hypothesis.id,
                "parameters": dict(hypothesis.parameters),
            },
            executor=executor,
            complexity=complexity,
        )
        score = self._scorer.evaluate_program(task, handle)
        failure_reason = _failure_reason(task, handle.executor, score.train_accuracy)
        genericity_score = _genericity_score(
            score.train_accuracy,
            score.example_accuracies,
            handle.cost.complexity,
        )
        transfer_proxy_score = _transfer_proxy_score(
            train_accuracy=score.train_accuracy,
            genericity_score=genericity_score,
            representation=representation,
            failure_reason=failure_reason,
        )
        return VerificationResult(
            hypothesis=hypothesis,
            score=score,
            compiled_program=CompiledProgram(
                hypothesis=hypothesis,
                handle=handle,
                family_name=hypothesis.family_name,
                representation_summary=representation.summary,
                genericity_score=genericity_score,
                transfer_proxy_score=transfer_proxy_score,
                metadata={
                    "family_name": hypothesis.family_name,
                    "genericity_score": genericity_score,
                    "transfer_proxy_score": transfer_proxy_score,
                },
            ),
            failure_reason=failure_reason,
            notes=score.notes,
        )


def build_arc_transfer_profile(
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> tuple[HypothesisFamilyStrategy, ...]:
    verifier = ArcTransferVerifier()
    strategies = (
        HypothesisFamilyStrategy(GlobalTransformFamily(), verifier),
        HypothesisFamilyStrategy(ObjectTransformFamily(), verifier),
        HypothesisFamilyStrategy(RelationPropagationFamily(), verifier),
        HypothesisFamilyStrategy(TemplateCompletionFamily(), verifier),
        HypothesisFamilyStrategy(RegionRoutingFamily(), verifier),
    )
    selected = strategies
    include_set = tuple(include)
    if include_set:
        unknown = [name for name in include_set if name not in {strategy.name for strategy in strategies}]
        if unknown:
            raise ValueError(f"unknown transfer strategies: {', '.join(sorted(unknown))}")
        selected = tuple(strategy for strategy in selected if strategy.name in include_set)
    if exclude:
        exclude_set = set(exclude)
        selected = tuple(strategy for strategy in selected if strategy.name not in exclude_set)
    if not selected:
        raise ValueError("arc-transfer has no remaining strategies after filtering")
    return selected
