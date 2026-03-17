from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agi_core_codex.domains.arc.analyze import classify_failures, scan_primitive_candidates
from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile


def test_scan_primitives_is_deterministic_for_fixed_fixture_splits(
    arc_fixture_dir: Path,
    repo_root: Path,
) -> None:
    first = scan_primitive_candidates(
        dataset_dir=arc_fixture_dir,
        train_dev_split_file=repo_root / "experiments" / "splits" / "arc_analysis_scan_train_dev.json",
        train_val_split_file=repo_root / "experiments" / "splits" / "arc_analysis_scan_train_val.json",
        include_candidates=("fill-enclosed", "gravity-left", "trim-rows"),
    )
    second = scan_primitive_candidates(
        dataset_dir=arc_fixture_dir,
        train_dev_split_file=repo_root / "experiments" / "splits" / "arc_analysis_scan_train_dev.json",
        train_val_split_file=repo_root / "experiments" / "splits" / "arc_analysis_scan_train_val.json",
        include_candidates=("fill-enclosed", "gravity-left", "trim-rows"),
    )

    assert first == second
    assert [result.name for result in first] == [
        "fill-enclosed",
        "gravity-left",
        "trim-rows",
    ]
    assert first[0].train_val.exact_solves == 1
    assert first[1].train_dev.exact_solves == 1
    assert first[2].train_dev.exact_solves == 0


def test_arc_analyze_scan_primitives_cli_emits_ranked_json(
    repo_root: Path,
    arc_fixture_dir: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agi_core_codex",
            "arc-analyze",
            "scan-primitives",
            "--dataset-dir",
            str(arc_fixture_dir),
            "--train-dev-split-file",
            str(repo_root / "experiments" / "splits" / "arc_analysis_scan_train_dev.json"),
            "--train-val-split-file",
            str(repo_root / "experiments" / "splits" / "arc_analysis_scan_train_val.json"),
            "--candidate",
            "fill-enclosed",
            "--candidate",
            "gravity-left",
            "--candidate",
            "trim-rows",
            "--format",
            "json",
            "--top",
            "2",
        ],
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert [item["name"] for item in payload] == ["fill-enclosed", "gravity-left"]


def test_classify_failures_is_deterministic_for_fixed_manifest(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, manifest_path = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_analysis_failure_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=17,
        )
    )

    assert manifest.metrics_as_dict()["solved_train"] == 0

    first = classify_failures(manifest_path=manifest_path)
    second = classify_failures(manifest_path=manifest_path)

    assert first == second
    assert first.unsolved_task_count == 2
    assert first.shape_buckets == {"same_shape": 2}
    assert first.separator_buckets == {"grid": 1, "none": 1}
    assert first.ray_span_buckets == {"mixed_span": 1, "row_span": 1}
    assert [task.task_key for task in first.tasks] == [
        "horizontal_span_fill",
        "separator_propagation",
    ]


def test_arc_analyze_classify_failures_cli_emits_bucket_summary(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    _, manifest_path = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_analysis_failure_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=17,
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agi_core_codex",
            "arc-analyze",
            "classify-failures",
            "--manifest-path",
            str(manifest_path),
            "--format",
            "json",
        ],
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["unsolved_task_count"] == 2
    assert payload["separator_buckets"] == {"grid": 1, "none": 1}
