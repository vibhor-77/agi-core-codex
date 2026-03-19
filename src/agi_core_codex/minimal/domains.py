from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

from agi_core_codex.minimal.core import GridExample, GridTask, GridTestCase
from agi_core_codex.minimal.ops import (
    Grid,
    compositor_specs,
    freeze_grid,
    unary_seed_specs,
)


def _grid(rows: Sequence[Sequence[int]]) -> Grid:
    return freeze_grid(rows)


@dataclass(frozen=True)
class SyntheticCurriculum:
    tier: str
    tasks: tuple[GridTask, ...]


@dataclass(frozen=True)
class CurriculumProgramSpec:
    name: str
    children: tuple["CurriculumProgramSpec", ...] = ()


@dataclass(frozen=True)
class CurriculumTaskSpec:
    task_id: str
    program: CurriculumProgramSpec
    train_input: Grid
    test_input: Grid


def _seed(name: str) -> CurriculumProgramSpec:
    return CurriculumProgramSpec(name=name)


def _node(name: str, *children: CurriculumProgramSpec) -> CurriculumProgramSpec:
    return CurriculumProgramSpec(name=name, children=tuple(children))


_UNARY_SEEDS = {spec.name: spec.unary for spec in unary_seed_specs()}
_BINARY_COMPOSITORS = {spec.name: spec.binary for spec in compositor_specs()}


def _execute_program_spec(spec: CurriculumProgramSpec, grid: Grid) -> Grid:
    if not spec.children:
        executor = _UNARY_SEEDS[spec.name]
        if executor is None:
            raise ValueError(f"seed {spec.name} is missing a unary executor")
        return executor(grid)
    if spec.name == "chain":
        left, right = spec.children
        return _execute_program_spec(right, _execute_program_spec(left, grid))
    if len(spec.children) != 2:
        raise ValueError(f"compositor {spec.name} requires exactly two children")
    left = _execute_program_spec(spec.children[0], grid)
    right = _execute_program_spec(spec.children[1], grid)
    executor = _BINARY_COMPOSITORS[spec.name]
    if executor is None:
        raise ValueError(f"compositor {spec.name} is missing a binary executor")
    return executor(left, right)


def _materialize_task(spec: CurriculumTaskSpec) -> GridTask:
    return GridTask(
        task_id=spec.task_id,
        train=(
            GridExample(
                input=spec.train_input,
                output=_execute_program_spec(spec.program, spec.train_input),
            ),
        ),
        test=(
            GridTestCase(
                input=spec.test_input,
                output=_execute_program_spec(spec.program, spec.test_input),
            ),
        ),
    )


def build_synthetic_curriculum(tier: str = "all") -> SyntheticCurriculum:
    leaf_tasks = tuple(
        _materialize_task(spec)
        for spec in (
            CurriculumTaskSpec(
                task_id="seed_identity",
                program=_seed("identity"),
                train_input=_grid(((1, 0), (0, 2))),
                test_input=_grid(((3, 0), (0, 4))),
            ),
            CurriculumTaskSpec(
                task_id="seed_flip_h",
                program=_seed("flip_h"),
                train_input=_grid(((1, 2, 0),)),
                test_input=_grid(((3, 4, 0),)),
            ),
            CurriculumTaskSpec(
                task_id="seed_crop_support",
                program=_seed("crop_support"),
                train_input=_grid(((0, 0, 0), (0, 5, 0), (0, 0, 0))),
                test_input=_grid(((0, 0, 0, 0), (0, 6, 7, 0), (0, 0, 0, 0))),
            ),
        )
    )

    pair_tasks = tuple(
        _materialize_task(spec)
        for spec in (
            CurriculumTaskSpec(
                task_id="pair_crop_flip_h",
                program=_node("chain", _seed("crop_support"), _seed("flip_h")),
                train_input=_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 5), (0, 0, 0, 0))),
                test_input=_grid(((0, 0, 0, 0), (0, 4, 5, 0), (0, 6, 7, 0), (0, 0, 0, 0))),
            ),
            CurriculumTaskSpec(
                task_id="pair_crop_flip_v",
                program=_node("chain", _seed("crop_support"), _seed("flip_v")),
                train_input=_grid(((0, 0, 0), (0, 8, 0), (0, 9, 0), (0, 0, 0))),
                test_input=_grid(((0, 0, 0), (0, 2, 3), (0, 4, 5), (0, 0, 0))),
            ),
            CurriculumTaskSpec(
                task_id="pair_crop_transpose",
                program=_node("chain", _seed("crop_support"), _seed("transpose")),
                train_input=_grid(((0, 0, 0, 0), (0, 1, 2, 3), (0, 4, 5, 6), (0, 0, 0, 0))),
                test_input=_grid(((0, 0, 0, 0), (0, 6, 7, 8), (0, 9, 1, 2), (0, 0, 0, 0))),
            ),
        )
    )

    triple_tasks = tuple(
        _materialize_task(spec)
        for spec in (
            CurriculumTaskSpec(
                task_id="triple_hcat_crop_flip_h",
                program=_node(
                    "hcat",
                    _seed("crop_support"),
                    _node("chain", _seed("crop_support"), _seed("flip_h")),
                ),
                train_input=_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 0), (0, 0, 0, 0))),
                test_input=_grid(((0, 0, 0, 0), (0, 5, 6, 0), (0, 7, 8, 0), (0, 0, 0, 0))),
            ),
            CurriculumTaskSpec(
                task_id="triple_vcat_crop_flip_v",
                program=_node(
                    "vcat",
                    _seed("crop_support"),
                    _node("chain", _seed("crop_support"), _seed("flip_v")),
                ),
                train_input=_grid(((0, 0, 0), (0, 1, 2), (0, 3, 4), (0, 0, 0))),
                test_input=_grid(((0, 0, 0), (0, 5, 6), (0, 7, 8), (0, 0, 0))),
            ),
            CurriculumTaskSpec(
                task_id="triple_pair_transpose_flip_h",
                program=_node(
                    "chain",
                    _node("chain", _seed("crop_support"), _seed("transpose")),
                    _seed("flip_h"),
                ),
                train_input=_grid(((0, 0, 0, 0), (0, 1, 2, 3), (0, 4, 5, 6), (0, 0, 0, 0))),
                test_input=_grid(((0, 0, 0, 0), (0, 7, 8, 9), (0, 3, 4, 5), (0, 0, 0, 0))),
            ),
        )
    )

    tiers = {
        "single": leaf_tasks,
        "pair": leaf_tasks + pair_tasks,
        "triple": leaf_tasks + pair_tasks + triple_tasks,
        "all": leaf_tasks + pair_tasks + triple_tasks,
    }
    if tier not in tiers:
        raise ValueError(f"unknown synthetic tier: {tier}")
    return SyntheticCurriculum(tier=tier, tasks=tiers[tier])


def load_arc_tasks_minimal(
    *,
    split_file: Path,
    dataset_dir: Path,
    limit: int | None = None,
) -> tuple[GridTask, ...]:
    payload = json.loads(split_file.read_text())
    task_ids = tuple(str(task_id) for task_id in payload.get("task_ids", []))
    if task_ids:
        paths = [dataset_dir / f"{task_id}.json" for task_id in task_ids]
    else:
        paths = sorted(dataset_dir.glob("*.json"))

    tasks: list[GridTask] = []
    for path in paths:
        raw = json.loads(path.read_text())
        train = tuple(
            GridExample(
                input=freeze_grid(example["input"]),
                output=freeze_grid(example["output"]),
            )
            for example in raw["train"]
        )
        test = tuple(
            GridTestCase(
                input=freeze_grid(example["input"]),
                output=freeze_grid(example["output"]) if "output" in example else None,
            )
            for example in raw["test"]
        )
        tasks.append(GridTask(task_id=path.stem, train=train, test=test))
    if limit is not None:
        return tuple(tasks[:limit])
    return tuple(tasks)
