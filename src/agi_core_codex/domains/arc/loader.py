from __future__ import annotations

import json
from pathlib import Path

from agi_core_codex.domains.arc.types import ArcExample, ArcTask, ArcTestCase, freeze_grid


def load_split_payload(split_file: Path) -> dict[str, object]:
    return json.loads(split_file.read_text())


def load_split_ids(split_file: Path) -> tuple[str, ...]:
    payload = load_split_payload(split_file)
    task_ids = payload.get("task_ids", [])
    return tuple(str(task_id) for task_id in task_ids)


def load_arc_task(path: Path) -> ArcTask:
    payload = json.loads(path.read_text())
    train = tuple(
        ArcExample(
            input=freeze_grid(example["input"]),
            output=freeze_grid(example["output"]),
        )
        for example in payload["train"]
    )
    test = tuple(
        ArcTestCase(
            input=freeze_grid(case["input"]),
            output=freeze_grid(case["output"]) if "output" in case else None,
        )
        for case in payload["test"]
    )
    return ArcTask(task_id=path.stem, train=train, test=test)


def load_arc_tasks(
    dataset_dir: Path,
    *,
    split_ids: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[ArcTask, ...]:
    if split_ids is None:
        paths = sorted(dataset_dir.glob("*.json"))
    else:
        paths = [dataset_dir / f"{task_id}.json" for task_id in split_ids]

    tasks = tuple(load_arc_task(path) for path in paths)
    if limit is not None:
        return tasks[:limit]
    return tasks
