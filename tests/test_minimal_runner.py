from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agi_core_codex.minimal.runner import MinimalRunOptions, run_minimal


def test_minimal_synthetic_compounds_across_rounds(tmp_path: Path) -> None:
    round_one, _ = run_minimal(
        MinimalRunOptions(
            domain="synthetic-grid",
            mode="tune",
            output_root=tmp_path / "round-one",
            rounds=1,
            curriculum_tier="pair",
        )
    )
    round_two, _ = run_minimal(
        MinimalRunOptions(
            domain="synthetic-grid",
            mode="tune",
            output_root=tmp_path / "round-two",
            rounds=2,
            curriculum_tier="pair",
        )
    )
    first = round_one.metrics_as_dict()
    second = round_two.metrics_as_dict()
    assert second["solved_train"] > first["solved_train"]
    assert second["round_2_library_reuse_count"] > 0
    assert second["round_2_search_cost_per_exact"] < first["round_1_search_cost_per_exact"]


def test_minimal_arc_smoke_runs_without_loading_legacy_strategies(
    arc_fixture_dir: Path,
    repo_root: Path,
    tmp_path: Path,
) -> None:
    script = f"""
import json, sys
from pathlib import Path
from agi_core_codex.minimal.runner import MinimalRunOptions, run_minimal
manifest, _ = run_minimal(
    MinimalRunOptions(
        domain="arc",
        mode="tune",
        output_root=Path({str(tmp_path / "arc-minimal")!r}),
        rounds=2,
        split_file=Path({str(repo_root / "experiments" / "splits" / "arc_minimal_smoke.json")!r}),
        dataset_dir=Path({str(arc_fixture_dir)!r}),
    )
)
print(json.dumps({{
    "metrics": manifest.metrics_as_dict(),
    "has_legacy_strategies": any(name.startswith("agi_core_codex.domains.arc.strategies") for name in sys.modules),
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    metrics = payload["metrics"]
    assert metrics["round_1_solved_train"] == 1
    assert metrics["round_2_solved_train"] == 2
    assert metrics["final_library_size"] > 0
    assert metrics["solved_train"] == 2
    assert payload["has_legacy_strategies"] is False


def test_legacy_cli_still_runs_bootstrap_smoke(arc_fixture_dir: Path, repo_root: Path, tmp_path: Path) -> None:
    from agi_core_codex.cli.main import main

    result = main(
        [
            "legacy",
            "arc-bootstrap",
            "tune",
            "--dataset-dir",
            str(arc_fixture_dir),
            "--split-file",
            str(repo_root / "experiments" / "splits" / "arc_bootstrap_compound_smoke.json"),
            "--output-root",
            str(tmp_path / "legacy-bootstrap"),
            "--rounds",
            "2",
        ]
    )
    assert result == 0
