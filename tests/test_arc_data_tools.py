from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agi_core_codex.core.manifests import load_manifest
from agi_core_codex.domains.arc.discovery import discover_arc_dataset
from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile
from agi_core_codex.domains.arc.splits import partition_train_tasks, write_train_splits


def test_discover_arc_dataset_finds_training_dir_from_search_root(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ARC-AGI" / "data" / "training"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "abc123.json").write_text('{"train": [], "test": []}\n')

    discovered = discover_arc_dataset(
        benchmark="arc-agi-1",
        split="training",
        search_roots=(tmp_path,),
    )
    assert discovered == dataset_dir


def test_partition_train_tasks_is_deterministic() -> None:
    task_ids = ("a", "b", "c", "d", "e")
    first = partition_train_tasks(task_ids, seed=3, train_val_count=2)
    second = partition_train_tasks(task_ids, seed=3, train_val_count=2)
    assert first == second
    assert set(first[0]).isdisjoint(set(first[1]))


def test_write_train_splits_emits_metadata(
    arc_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    train_dev_path, train_val_path = write_train_splits(
        benchmark="arc-agi-1",
        dataset_dir=arc_fixture_dir,
        output_dir=tmp_path,
        seed=5,
        train_val_count=2,
    )

    train_dev = json.loads(train_dev_path.read_text())
    train_val = json.loads(train_val_path.read_text())
    assert train_dev["split_kind"] == "train-dev"
    assert train_val["split_kind"] == "train-val"
    assert train_dev["train_total_count"] == 48
    assert train_val["train_val_count"] == 2
    assert set(train_dev["task_ids"]).isdisjoint(set(train_val["task_ids"]))


def test_arc_data_make_splits_cli(
    repo_root: Path,
    arc_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agi_core_codex",
            "arc-data",
            "make-splits",
            "--benchmark",
            "arc-agi-1",
            "--dataset-dir",
            str(arc_fixture_dir),
            "--output-dir",
            str(tmp_path),
            "--train-val-count",
            "2",
            "--seed",
            "5",
        ],
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )

    assert "train_dev=" in result.stdout
    assert (tmp_path / "arc_agi_1_train_dev.json").exists()
    assert (tmp_path / "arc_agi_1_train_val.json").exists()


def test_arc_profile_can_resolve_dataset_dir_from_split_metadata(
    arc_fixture_dir: Path,
    tmp_path: Path,
) -> None:
    train_dev_path, _ = write_train_splits(
        benchmark="arc-agi-1",
        dataset_dir=arc_fixture_dir,
        output_dir=tmp_path / "splits",
        seed=5,
        train_val_count=2,
    )

    manifest, manifest_path = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=None,
            split_file=train_dev_path,
            output_root=tmp_path / "artifacts",
            seed=5,
            limit=2,
        )
    )

    assert manifest.task_count == 2
    payload = load_manifest(manifest_path)
    assert any(
        note == f"dataset_dir={arc_fixture_dir}"
        for note in payload["notes"]
    )
