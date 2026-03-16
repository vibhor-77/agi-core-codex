from __future__ import annotations

from pathlib import Path

from agi_core_codex.core.manifests import load_manifest
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile


def _comparable_manifest(manifest) -> dict:
    payload = {
        "profile": manifest.profile,
        "mode": manifest.mode,
        "split": manifest.split,
        "strategy_set": manifest.strategy_set,
        "task_count": manifest.task_count,
        "metrics": manifest.metrics_as_dict(),
        "tasks": [
            {
                "task_key": task.task_key,
                "solved_train": task.solved_train,
                "solved_test": task.solved_test,
                "best_program_name": task.best_program_name,
                "best_strategy": task.best_strategy,
                "train_accuracy": task.train_accuracy,
                "test_accuracy": task.test_accuracy,
            }
            for task in manifest.tasks
        ],
    }
    return payload


def test_stable_program_ids_change_with_semantics() -> None:
    first = make_arc_program(
        name="absolute-patch",
        semantics={"type": "absolute_patch", "task_id": "a", "patch": ((0, 0, 9),)},
        executor=lambda grid: grid,
    )
    second = make_arc_program(
        name="absolute-patch",
        semantics={"type": "absolute_patch", "task_id": "a", "patch": ((0, 1, 9),)},
        executor=lambda grid: grid,
    )
    assert first.id != second.id


def test_arc_accuracy_solves_fixture_validation_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, manifest_path = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_train_val.json",
            output_root=tmp_path / "artifacts",
            seed=7,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"]
    assert metrics["solved_test"] == metrics["test_eligible_count"]
    assert manifest_path.exists()
    payload = load_manifest(manifest_path)
    assert payload["profile"] == "arc-accuracy"


def test_runs_are_deterministic_for_fixed_seed(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    first, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-theory",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_train_dev.json",
            output_root=tmp_path / "first",
            seed=13,
        )
    )
    second, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-theory",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_train_dev.json",
            output_root=tmp_path / "second",
            seed=13,
        )
    )

    assert _comparable_manifest(first) == _comparable_manifest(second)

