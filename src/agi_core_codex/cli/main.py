from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from agi_core_codex.legacy.cli import main as legacy_main
from agi_core_codex.minimal.runner import MinimalRunOptions, run_minimal


_LEGACY_ALIASES = {
    "arc-data",
    "arc-analyze",
    "baseline-core",
    "arc-accuracy",
    "arc-theory",
    "arc-transfer",
    "arc-bootstrap",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agi-core-codex")
    subparsers = parser.add_subparsers(dest="command", required=True)

    minimal_parser = subparsers.add_parser("minimal")
    minimal_modes = minimal_parser.add_subparsers(dest="mode", required=True)
    for mode in ("tune", "score"):
        mode_parser = minimal_modes.add_parser(mode)
        mode_parser.add_argument("--domain", choices=("synthetic-grid", "arc"), required=True)
        mode_parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
        mode_parser.add_argument("--rounds", type=int, default=3)
        mode_parser.add_argument("--seed", type=int, default=0)
        mode_parser.add_argument("--limit", type=int)
        mode_parser.add_argument("--curriculum-tier", choices=("single", "pair", "triple", "all"), default="all")
        mode_parser.add_argument("--split-file", type=Path)
        mode_parser.add_argument("--dataset-dir", type=Path)
        mode_parser.add_argument("--benchmark", choices=("arc-agi-1", "arc-agi-2"), default="arc-agi-1")
        mode_parser.add_argument("--dataset-split", choices=("training", "evaluation"), default="training")

    legacy_parser = subparsers.add_parser("legacy")
    legacy_parser.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, Sequence) and argv:
        if argv[0] in _LEGACY_ALIASES:
            return legacy_main(list(argv))

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "legacy":
        return legacy_main(args.args)

    manifest, manifest_path = run_minimal(
        MinimalRunOptions(
            domain=args.domain,
            mode=args.mode,
            output_root=args.output_root,
            rounds=args.rounds,
            seed=args.seed,
            split_file=args.split_file,
            dataset_dir=args.dataset_dir,
            benchmark=args.benchmark,
            dataset_split=args.dataset_split,
            curriculum_tier=args.curriculum_tier,
            limit=args.limit,
        )
    )
    metrics = manifest.metrics_as_dict()
    graduation_fragment = ""
    if manifest.domain == "synthetic-grid" and "graduation_ready_for_arc" in metrics:
        graduation_fragment = (
            f" graduation_ready_for_arc={metrics['graduation_ready_for_arc']}"
            f" graduation_solve_gain={metrics['graduation_solve_gain']}"
        )
    print(
        f"profile={manifest.profile} domain={manifest.domain} mode={manifest.mode} split={manifest.split} "
        f"solved_train={metrics['solved_train']}/{metrics['task_count']} "
        f"train_exact_accuracy={metrics['train_exact_accuracy']:.3f} "
        f"manifest={manifest_path}"
        f"{graduation_fragment}"
    )
    return 0
