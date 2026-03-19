from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

from agi_core_codex.minimal.core import GridExample, GridTask, GridTestCase
from agi_core_codex.minimal.ops import (
    Grid,
    crop_support,
    flip_h,
    flip_v,
    freeze_grid,
    hcat,
    identity,
    transpose,
)


def _grid(rows: Sequence[Sequence[int]]) -> Grid:
    return freeze_grid(rows)


@dataclass(frozen=True)
class SyntheticCurriculum:
    tier: str
    tasks: tuple[GridTask, ...]


def build_synthetic_curriculum(tier: str = "all") -> SyntheticCurriculum:
    leaf_tasks = (
        GridTask(
            task_id="seed_identity",
            train=(
                GridExample(input=_grid(((1, 0), (0, 2))), output=identity(_grid(((1, 0), (0, 2))))),
            ),
            test=(
                GridTestCase(input=_grid(((3, 0), (0, 4))), output=_grid(((3, 0), (0, 4)))),
            ),
        ),
        GridTask(
            task_id="seed_flip_h",
            train=(
                GridExample(input=_grid(((1, 2, 0),)), output=flip_h(_grid(((1, 2, 0),)))),
            ),
            test=(
                GridTestCase(input=_grid(((3, 4, 0),)), output=_grid(((0, 4, 3),))),
            ),
        ),
        GridTask(
            task_id="seed_crop_support",
            train=(
                GridExample(
                    input=_grid(((0, 0, 0), (0, 5, 0), (0, 0, 0))),
                    output=_grid(((5,),)),
                ),
            ),
            test=(
                GridTestCase(
                    input=_grid(((0, 0, 0, 0), (0, 6, 7, 0), (0, 0, 0, 0))),
                    output=_grid(((6, 7),)),
                ),
            ),
        ),
    )

    pair_tasks = (
        GridTask(
            task_id="pair_crop_flip_h",
            train=(
                GridExample(
                    input=_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 5), (0, 0, 0, 0))),
                    output=flip_h(crop_support(_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 5), (0, 0, 0, 0))))),
                ),
            ),
            test=(
                GridTestCase(
                    input=_grid(((0, 0, 0, 0), (0, 4, 5, 0), (0, 6, 7, 0), (0, 0, 0, 0))),
                    output=_grid(((5, 4), (7, 6))),
                ),
            ),
        ),
        GridTask(
            task_id="pair_crop_flip_v",
            train=(
                GridExample(
                    input=_grid(((0, 0, 0), (0, 8, 0), (0, 9, 0), (0, 0, 0))),
                    output=flip_v(crop_support(_grid(((0, 0, 0), (0, 8, 0), (0, 9, 0), (0, 0, 0))))),
                ),
            ),
            test=(
                GridTestCase(
                    input=_grid(((0, 0, 0), (0, 2, 3), (0, 4, 5), (0, 0, 0))),
                    output=_grid(((4, 5), (2, 3))),
                ),
            ),
        ),
    )

    triple_tasks = (
        GridTask(
            task_id="triple_hcat_crop_flip",
            train=(
                GridExample(
                    input=_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 0), (0, 0, 0, 0))),
                    output=hcat(
                        crop_support(_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 0), (0, 0, 0, 0)))),
                        flip_h(crop_support(_grid(((0, 0, 0, 0), (0, 1, 2, 0), (0, 3, 4, 0), (0, 0, 0, 0))))),
                    ),
                ),
            ),
            test=(
                GridTestCase(
                    input=_grid(((0, 0, 0, 0), (0, 5, 6, 0), (0, 7, 8, 0), (0, 0, 0, 0))),
                    output=_grid(((5, 6, 6, 5), (7, 8, 8, 7))),
                ),
            ),
        ),
        GridTask(
            task_id="triple_chain_transpose_crop",
            train=(
                GridExample(
                    input=_grid(((0, 0, 0), (0, 1, 2), (0, 3, 4), (0, 0, 0))),
                    output=transpose(crop_support(_grid(((0, 0, 0), (0, 1, 2), (0, 3, 4), (0, 0, 0))))),
                ),
            ),
            test=(
                GridTestCase(
                    input=_grid(((0, 0, 0), (0, 6, 7), (0, 8, 9), (0, 0, 0))),
                    output=_grid(((6, 8), (7, 9))),
                ),
            ),
        ),
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
