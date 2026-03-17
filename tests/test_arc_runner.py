from __future__ import annotations

from pathlib import Path

from agi_core_codex.core.manifests import load_manifest
from agi_core_codex.domains.arc.profiles import build_arc_profile, profile_strategy_names
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


def test_profile_strategy_filtering_preserves_explicit_ablation_controls() -> None:
    selected = build_arc_profile(
        "arc-accuracy",
        include=("arc-boolean-halves", "arc-separator-cross-reference"),
        exclude=("arc-separator-cross-reference",),
    )
    assert tuple(strategy.name for strategy in selected) == ("arc-boolean-halves",)
    assert "arc-boolean-halves" in profile_strategy_names("arc-accuracy")


def test_cross_reference_strategies_solve_recovery_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_cross_reference_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=11,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 4
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 4
    best_strategies = {task.task_key: task.best_strategy for task in manifest.tasks}
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert best_strategies["boolean_halves_recolor"] == "arc-boolean-halves"
    assert best_strategies["separator_extract"] == "arc-separator-cross-reference"
    assert best_strategies["separator_or_reduce"] == "arc-separator-cross-reference"
    assert best_strategies["separator_majority_reduce"] == "arc-separator-cross-reference"
    assert best_programs["separator_or_reduce"] == "cross-ref-or_reduce-cells"
    assert best_programs["separator_majority_reduce"] == "cross-ref-majority_reduce-cells"


def test_excluding_boolean_halves_strategy_breaks_that_recovery_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_boolean_halves_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=19,
            exclude_strategies=("arc-boolean-halves",),
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == 0
    assert manifest.strategy_set == (
        "arc-row-column-decomposition",
        "arc-separator-propagation",
        "arc-separator-cross-reference",
        "arc-scale-tile",
        "arc-template-stamp",
        "grammar-primitives",
        "arc-constant-output",
        "arc-color-map",
        "arc-absolute-patch",
    )


def test_scale_tile_strategy_solves_ratio_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_scale_tile_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=23,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 3
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 3
    assert {task.best_strategy for task in manifest.tasks} == {"arc-scale-tile"}


def test_row_column_strategy_solves_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_row_column_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=27,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 2
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 2
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"arc-row-column-decomposition"}
    assert best_programs["sort_rows_by_nonzero"] == "sort-rows-by-nonzero-count-desc"
    assert best_programs["sort_columns_by_sum"] == "sort-columns-by-sum-asc"


def test_template_stamp_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_template_stamp_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=29,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-template-stamp"


def test_separator_propagation_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_separator_propagation_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=31,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-separator-propagation"
