from __future__ import annotations

from pathlib import Path

from agi_core_codex.domains.arc.loader import load_arc_task
from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile
from agi_core_codex.domains.arc.transfer import (
    ArcTransferVerifier,
    GlobalTransformFamily,
    build_arc_task_representation,
    build_arc_transfer_profile,
)
from agi_core_codex.domains.arc.transfer_analysis import (
    cluster_transfer_failures,
    diff_transfer_manifests,
)


def test_arc_transfer_profile_exposes_five_family_strategies() -> None:
    strategies = build_arc_transfer_profile()
    assert [strategy.name for strategy in strategies] == [
        "transfer:global-transform",
        "transfer:object-transform",
        "transfer:relation-propagation",
        "transfer:template-completion",
        "transfer:region-routing",
    ]


def test_arc_transfer_representation_is_deterministic(arc_fixture_dir: Path) -> None:
    task = load_arc_task(arc_fixture_dir / "separator_propagation.json")
    first = build_arc_task_representation(task)
    second = build_arc_task_representation(task)

    assert first == second
    assert first.features["separator_count"] > 0
    assert "row-seps=" in first.summary


def test_arc_transfer_hypothesis_generation_is_deterministic(arc_fixture_dir: Path) -> None:
    task = load_arc_task(arc_fixture_dir / "mirror_tile_horizontal.json")
    family = GlobalTransformFamily()
    representation = family.build_representation(task, environment=None)

    first = tuple(hypothesis.id for hypothesis in family.propose(task, representation))
    second = tuple(hypothesis.id for hypothesis in family.propose(task, representation))
    descriptions = tuple(hypothesis.description for hypothesis in family.propose(task, representation))

    assert first == second
    assert "global:mirror_tile_horizontal" in descriptions


def test_arc_transfer_verifier_compiles_stable_program_ids(arc_fixture_dir: Path) -> None:
    task = load_arc_task(arc_fixture_dir / "mirror_tile_horizontal.json")
    family = GlobalTransformFamily()
    representation = build_arc_task_representation(task)
    hypothesis = next(
        hypothesis
        for hypothesis in family.propose(task, representation)
        if hypothesis.parameters.get("executor_key") == "mirror_tile_horizontal"
    )
    verifier = ArcTransferVerifier()

    first = verifier.verify(task, hypothesis, representation)
    second = verifier.verify(task, hypothesis, representation)

    assert first.compiled_program is not None
    assert second.compiled_program is not None
    assert first.compiled_program.handle.id == second.compiled_program.handle.id
    assert first.compiled_program.handle.cost.complexity == 2
    assert first.score is not None and first.score.train_exact
    assert first.failure_reason is None


def test_arc_transfer_solves_seed_smoke_split(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-transfer",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_transfer_seed_smoke.json",
            output_root=tmp_path / "artifacts",
            seed=101,
        )
    )

    metrics = manifest.metrics_as_dict()
    assert metrics["solved_train"] == metrics["task_count"] == 5
    assert metrics["solved_test"] == metrics["test_eligible_count"] == 5
    by_task = {task.task_key: task for task in manifest.tasks}
    assert by_task["mirror_tile_horizontal"].family_name == "global-transform"
    assert by_task["extract_largest_cc"].family_name == "object-transform"
    assert by_task["separator_propagation"].family_name == "relation-propagation"
    assert by_task["motif_completion"].family_name == "template-completion"
    assert by_task["fill_enclosed"].family_name == "region-routing"


def test_arc_transfer_diff_and_cluster_reports(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    baseline_manifest, baseline_path = run_arc_profile(
        ArcRunOptions(
            profile="baseline-core",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_transfer_seed_smoke.json",
            output_root=tmp_path / "baseline-artifacts",
            seed=102,
        )
    )
    transfer_manifest, transfer_path = run_arc_profile(
        ArcRunOptions(
            profile="arc-transfer",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_transfer_seed_smoke.json",
            output_root=tmp_path / "transfer-artifacts",
            seed=102,
        )
    )

    diff = diff_transfer_manifests(
        baseline_manifest_path=baseline_path,
        candidate_manifest_path=transfer_path,
    )
    assert baseline_manifest.profile == "baseline-core"
    assert transfer_manifest.profile == "arc-transfer"
    assert diff.task_count == 5
    assert diff.gained >= 3

    cluster_manifest, cluster_manifest_path = run_arc_profile(
        ArcRunOptions(
            profile="arc-transfer",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_transfer_cluster_smoke.json",
            output_root=tmp_path / "cluster-artifacts",
            seed=103,
        )
    )
    report = cluster_transfer_failures(
        manifest_path=cluster_manifest_path,
        dataset_dir=arc_fixture_dir,
    )
    assert cluster_manifest.profile == "arc-transfer"
    assert report.unsolved_task_count == 1
    assert report.tasks[0].task_key == "color_remap"
    assert report.tasks[0].failure_reason in {"near_miss", "train_mismatch", "shape_mismatch"}
