from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def list_task_ids(dataset_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in dataset_dir.glob("*.json")))


def partition_train_tasks(
    task_ids: Iterable[str],
    *,
    seed: int,
    train_val_count: int | None = None,
    train_val_fraction: float = 0.2,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ids = sorted(str(task_id) for task_id in task_ids)
    if not ids:
        raise ValueError("cannot build splits from an empty task set")

    if train_val_count is None:
        train_val_count = max(1, round(len(ids) * train_val_fraction))
    if train_val_count <= 0 or train_val_count >= len(ids):
        raise ValueError("train_val_count must be between 1 and len(task_ids)-1")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    train_val_ids = tuple(sorted(shuffled[:train_val_count]))
    train_dev_ids = tuple(sorted(shuffled[train_val_count:]))
    return train_dev_ids, train_val_ids


def _split_payload(
    *,
    benchmark: str,
    split_kind: str,
    dataset_dir: Path,
    seed: int,
    task_ids: tuple[str, ...],
    train_total_count: int,
    train_val_count: int,
) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "split_kind": split_kind,
        "source_dataset_dir": str(dataset_dir),
        "seed": seed,
        "train_total_count": train_total_count,
        "train_val_count": train_val_count,
        "task_ids": list(task_ids),
    }


def write_train_splits(
    *,
    benchmark: str,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    train_val_count: int | None = None,
    train_val_fraction: float = 0.2,
    prefix: str | None = None,
) -> tuple[Path, Path]:
    task_ids = list_task_ids(dataset_dir)
    train_dev_ids, train_val_ids = partition_train_tasks(
        task_ids,
        seed=seed,
        train_val_count=train_val_count,
        train_val_fraction=train_val_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = prefix or benchmark.replace("-", "_")
    train_dev_path = output_dir / f"{file_prefix}_train_dev.json"
    train_val_path = output_dir / f"{file_prefix}_train_val.json"

    train_total_count = len(task_ids)
    held_out_count = len(train_val_ids)
    train_dev_path.write_text(
        json.dumps(
            _split_payload(
                benchmark=benchmark,
                split_kind="train-dev",
                dataset_dir=dataset_dir,
                seed=seed,
                task_ids=train_dev_ids,
                train_total_count=train_total_count,
                train_val_count=held_out_count,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    train_val_path.write_text(
        json.dumps(
            _split_payload(
                benchmark=benchmark,
                split_kind="train-val",
                dataset_dir=dataset_dir,
                seed=seed,
                task_ids=train_val_ids,
                train_total_count=train_total_count,
                train_val_count=held_out_count,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return train_dev_path, train_val_path

