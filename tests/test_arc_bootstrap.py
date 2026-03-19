from __future__ import annotations

from pathlib import Path

from agi_core_codex.domains.arc.bootstrap import ArcBootstrapGrammar
from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile


def test_arc_bootstrap_uses_tiny_seed_grammar() -> None:
    grammar = ArcBootstrapGrammar()
    primitives = grammar.enumerate_primitives(task=None)

    assert [program.name for program in primitives] == [
        "identity",
        "flip-horizontal",
        "flip-vertical",
        "rotate-180",
        "transpose",
        "crop-nonzero",
        "fill-enclosed",
    ]


def test_arc_bootstrap_compounds_across_rounds(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    first_manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-bootstrap",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_bootstrap_compound_smoke.json",
            output_root=tmp_path / "round-one",
            rounds=1,
            seed=13,
        )
    )
    second_manifest, _ = run_arc_profile(
        ArcRunOptions(
            profile="arc-bootstrap",
            mode="tune",
            dataset_dir=arc_fixture_dir,
            split_file=repo_root / "experiments" / "splits" / "arc_bootstrap_compound_smoke.json",
            output_root=tmp_path / "round-two",
            rounds=2,
            seed=13,
        )
    )

    first_metrics = first_manifest.metrics_as_dict()
    second_metrics = second_manifest.metrics_as_dict()

    assert first_metrics["solved_train"] == 2
    assert second_metrics["solved_train"] == 3
    assert second_metrics["round_1_solved_train"] == 2
    assert second_metrics["round_2_solved_train"] == 3
    assert second_metrics["final_library_size"] >= 2

    best_programs = {task.task_key: task.best_program_name for task in second_manifest.tasks}
    assert best_programs["crop_flip_horizontal"] in {
        "crop-nonzero-then-flip-horizontal",
        "flip-horizontal-then-crop-nonzero",
    }
