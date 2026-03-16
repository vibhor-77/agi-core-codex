from __future__ import annotations

import argparse
from pathlib import Path

from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile


PROFILES = ("baseline-core", "arc-accuracy", "arc-theory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agi-core-codex")
    subparsers = parser.add_subparsers(dest="profile", required=True)

    for profile in PROFILES:
        profile_parser = subparsers.add_parser(profile)
        mode_parsers = profile_parser.add_subparsers(dest="mode", required=True)
        for mode in ("tune", "score"):
            mode_parser = mode_parsers.add_parser(mode)
            mode_parser.add_argument("--dataset-dir", type=Path, required=True)
            mode_parser.add_argument("--split-file", type=Path, required=True)
            mode_parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
            mode_parser.add_argument("--seed", type=int, default=0)
            mode_parser.add_argument("--limit", type=int)
            mode_parser.add_argument(
                "--include-strategy",
                action="append",
                default=[],
                help="Restrict the profile to a subset of its strategies. Repeatable.",
            )
            mode_parser.add_argument(
                "--exclude-strategy",
                action="append",
                default=[],
                help="Drop one or more strategies from the selected profile. Repeatable.",
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    manifest, manifest_path = run_arc_profile(
        ArcRunOptions(
            profile=args.profile,
            mode=args.mode,
            dataset_dir=args.dataset_dir,
            split_file=args.split_file,
            output_root=args.output_root,
            seed=args.seed,
            limit=args.limit,
            include_strategies=tuple(args.include_strategy),
            exclude_strategies=tuple(args.exclude_strategy),
        )
    )
    metrics = manifest.metrics_as_dict()
    print(
        f"profile={manifest.profile} mode={manifest.mode} split={manifest.split} "
        f"solved_train={metrics['solved_train']}/{metrics['task_count']} "
        f"train_exact_accuracy={metrics['train_exact_accuracy']:.3f} "
        f"manifest={manifest_path}"
    )
    return 0
