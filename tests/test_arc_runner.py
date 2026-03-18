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


def test_baseline_core_color_primitives_solve_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_color_grammar_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=5,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 2
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 2
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"grammar-primitives"}
    assert best_programs["recolor_foreground"] == "recolor-foreground-9"
    assert best_programs["swap_colors"] == "swap-colors-2-with-6"


def test_baseline_core_crop_primitive_solves_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_crop_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=6,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "grammar-primitives"
    assert manifest.tasks[0].best_program_name == "crop-nonzero"


def test_baseline_core_gravity_primitives_solve_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_gravity_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=8,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 2
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 2
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"grammar-primitives"}
    assert best_programs["gravity_down"] == "gravity-down"
    assert best_programs["gravity_up"] == "gravity-up"


def test_baseline_core_tile_family_primitives_solve_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_tile_family_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=37,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 3
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 3
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"grammar-primitives"}
    assert best_programs["tile_horizontal"] == "tile-horizontal"
    assert best_programs["mirror_tile_horizontal"] == "mirror-tile-horizontal"
    assert best_programs["mirror_tile_vertical"] == "mirror-tile-vertical"


def test_baseline_core_extraction_and_inpaint_primitives_solve_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_extraction_inpaint_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=43,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 3
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 3
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"grammar-primitives"}
    assert best_programs["extract_largest_cc"] == "extract-largest-cc"
    assert best_programs["extract_unique_color_region"] == "extract-unique-color-region"
    assert best_programs["inpaint_periodic"] == "inpaint-periodic"


def test_baseline_core_symmetry_tile_primitives_solve_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_symmetry_tile_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=47,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 3
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 3
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"grammar-primitives"}
    assert best_programs["mirror_tile_both"] == "mirror-tile-both"
    assert best_programs["rotate_tile_clockwise"] == "rotate-tile-clockwise"
    assert best_programs["inpaint_by_symmetry"] == "inpaint-by-symmetry"


def test_extract_transform_strategy_solves_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_extract_transform_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=53,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 2
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 2
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"arc-extract-transform"}
    assert best_programs["crop_tile_horizontal"] == "crop-nonzero-then-tile-horizontal"
    assert best_programs["crop_flip_horizontal"] == "crop-nonzero-then-flip-horizontal"


def test_interior_extract_strategy_solves_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_interior_extract_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=59,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert {task.best_strategy for task in manifest.tasks} == {"arc-interior-extract"}
    assert manifest.tasks[0].best_program_name == "extract-enclosed-interior-recolor-3"


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


def test_alternating_diagonal_recolor_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_alternating_diagonal_recolor_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=151,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-alternating-diagonal-recolor"
    assert manifest.tasks[0].best_program_name == "recolor-odd-diagonal-chain-cells"


def test_recolor_isolated_singletons_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_recolor_isolated_singletons_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=167,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-recolor-isolated-singletons"
    assert manifest.tasks[0].best_program_name == "recolor-isolated-twos-to-ones"


def test_recolor_components_by_top_order_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_recolor_components_by_top_order_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=173,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-recolor-components-by-top-order"
    assert manifest.tasks[0].best_program_name == "recolor-components-by-top-order"


def test_relocate_shape_next_to_line_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_relocate_shape_next_to_line_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=179,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-relocate-shape-next-to-line"
    assert manifest.tasks[0].best_program_name == "relocate-shape-3-next-to-line-2-with-8"


def test_zero_rectangle_family_fill_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_zero_rectangle_family_fill_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=157,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-zero-rectangle-family-fill"
    assert manifest.tasks[0].best_program_name == "fill-largest-zero-rectangle-families"


def test_hollow_solid_rectangles_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_hollow_solid_rectangles_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=163,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-hollow-solid-rectangles"
    assert manifest.tasks[0].best_program_name == "hollow-solid-rectangles"


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
        "arc-alternating-diagonal-recolor",
        "arc-recolor-components-by-top-order",
        "arc-recolor-isolated-singletons",
        "arc-relocate-shape-next-to-line",
        "arc-ray-extension",
        "arc-row-column-decomposition",
        "arc-separator-propagation",
        "arc-separator-cross-reference",
        "arc-scale-tile",
        "arc-template-stamp",
        "arc-template-propagation",
        "arc-diagonal-cross-projection",
        "arc-zero-rectangle-family-fill",
        "arc-zero-pattern-propagation",
        "arc-zero-square-fill",
        "arc-extract-transform",
        "arc-hole-projection",
        "arc-hollow-solid-rectangles",
        "arc-triomino-corner-fill",
        "arc-collinear-gap-bridge",
        "arc-solid-rectangle-extract",
        "arc-rectangular-ring-recolor",
        "arc-scaffold-column-projection",
        "arc-interior-extract",
        "arc-motif-completion",
        "arc-bottom-center-marker",
        "arc-bbox-recolor",
        "arc-bbox-ring-marker-projection",
        "arc-rectangle-marker-projection",
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


def test_ray_extension_strategy_solves_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_ray_extension_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=41,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 2
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 2
    best_programs = {task.task_key: task.best_program_name for task in manifest.tasks}
    assert {task.best_strategy for task in manifest.tasks} == {"arc-ray-extension"}
    assert best_programs["extend_rays_down"] == "extend-rays-down"
    assert best_programs["horizontal_span_fill"] == "mask-extend-rays-left-with-right"


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


def test_template_propagation_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_template_propagation_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=137,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-template-propagation"
    assert manifest.tasks[0].best_program_name == "propagate-largest-template"


def test_diagonal_cross_projection_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_diagonal_cross_projection_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=149,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-diagonal-cross-projection"
    assert manifest.tasks[0].best_program_name == "project-diagonal-cross"


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
    assert manifest.tasks[0].best_program_name == "cross-ref-separator-propagation"


def test_separator_payload_propagation_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_separator_payload_propagation_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=61,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-separator-propagation"
    assert manifest.tasks[0].best_program_name == "cross-ref-separator-payload-propagation"


def test_bottom_center_marker_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_bottom_center_markers_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=67,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-bottom-center-marker"
    assert manifest.tasks[0].best_program_name == "bottom-center-markers-4"


def test_rectangle_marker_projection_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_rectangle_marker_projection_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=71,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-rectangle-marker-projection"
    assert manifest.tasks[0].best_program_name == "project-markers-into-rectangle"


def test_bbox_ring_marker_projection_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_bbox_ring_marker_projection_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=79,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-bbox-ring-marker-projection"
    assert manifest.tasks[0].best_program_name == "project-markers-to-bbox-ring"


def test_bbox_recolor_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_bbox_recolor_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=83,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-bbox-recolor"
    assert manifest.tasks[0].best_program_name == "bbox-recolor-1-to-3-inside-8"


def test_zero_square_fill_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_zero_square_fill_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=89,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-zero-square-fill"
    assert manifest.tasks[0].best_program_name == "fill-zero-component-squares-1"


def test_motif_completion_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_motif_completion_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=97,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-motif-completion"
    assert manifest.tasks[0].best_program_name == "complete-anchor-motifs"


def test_zero_pattern_propagation_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_zero_pattern_propagation_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=101,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-zero-pattern-propagation"
    assert manifest.tasks[0].best_program_name == "propagate-zero-pattern"


def test_hole_projection_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_hole_projection_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=103,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-hole-projection"
    assert manifest.tasks[0].best_program_name == "project-holes-along-short-axis"


def test_triomino_corner_fill_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_triomino_corner_fill_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=107,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-triomino-corner-fill"
    assert manifest.tasks[0].best_program_name == "fill-triomino-corners-8-with-1"


def test_collinear_gap_bridge_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_collinear_gap_bridge_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=109,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-collinear-gap-bridge"
    assert manifest.tasks[0].best_program_name == "bridge-horizontal-gaps-1-with-2"


def test_solid_rectangle_extract_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_solid_rectangle_extract_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=113,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-solid-rectangle-extract"
    assert manifest.tasks[0].best_program_name == "keep-max-solid-rectangles"


def test_rectangular_ring_recolor_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_rectangular_ring_recolor_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=127,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-rectangular-ring-recolor"
    assert manifest.tasks[0].best_program_name == "recolor-rectangular-rings-1-to-3"


def test_scaffold_column_projection_strategy_solves_smoke_task(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-accuracy",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_scaffold_column_projection_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=131,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 1
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 1
    assert manifest.tasks[0].best_strategy == "arc-scaffold-column-projection"
    assert manifest.tasks[0].best_program_name == "project-columns-from-scaffold"
