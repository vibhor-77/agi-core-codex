from __future__ import annotations

import argparse
from pathlib import Path

from agi_core_codex.domains.arc.discovery import discover_arc_dataset
from agi_core_codex.domains.arc.runner import ArcRunOptions, run_arc_profile
from agi_core_codex.domains.arc.splits import write_train_splits


PROFILES = ("baseline-core", "arc-accuracy", "arc-theory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agi-core-codex")
    subparsers = parser.add_subparsers(dest="profile", required=True)

    arc_data_parser = subparsers.add_parser("arc-data")
    arc_data_subparsers = arc_data_parser.add_subparsers(dest="arc_data_command", required=True)

    discover_parser = arc_data_subparsers.add_parser("discover")
    discover_parser.add_argument("--benchmark", choices=("arc-agi-1", "arc-agi-2"), required=True)
    discover_parser.add_argument("--split", choices=("training", "evaluation"), required=True)

    make_splits_parser = arc_data_subparsers.add_parser("make-splits")
    make_splits_parser.add_argument(
        "--benchmark",
        choices=("arc-agi-1", "arc-agi-2"),
        required=True,
    )
    make_splits_parser.add_argument("--dataset-dir", type=Path)
    make_splits_parser.add_argument("--output-dir", type=Path, required=True)
    make_splits_parser.add_argument("--seed", type=int, default=0)
    make_splits_parser.add_argument("--train-val-count", type=int)
    make_splits_parser.add_argument("--train-val-fraction", type=float, default=0.2)
    make_splits_parser.add_argument("--prefix", type=str)

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

    if args.profile == "arc-data":
        if args.arc_data_command == "discover":
            discovered = discover_arc_dataset(benchmark=args.benchmark, split=args.split)
            if discovered is None:
                print("not-found")
                return 1
            print(discovered)
            return 0

        if args.arc_data_command == "make-splits":
            dataset_dir = args.dataset_dir
            if dataset_dir is None:
                discovered = discover_arc_dataset(benchmark=args.benchmark, split="training")
                if discovered is None:
                    parser.error(
                        "could not auto-discover ARC training data; pass --dataset-dir explicitly"
                    )
                dataset_dir = discovered

            train_dev_path, train_val_path = write_train_splits(
                benchmark=args.benchmark,
                dataset_dir=dataset_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                train_val_count=args.train_val_count,
                train_val_fraction=args.train_val_fraction,
                prefix=args.prefix,
            )
            print(
                f"benchmark={args.benchmark} dataset_dir={dataset_dir} "
                f"train_dev={train_dev_path} train_val={train_val_path}"
            )
            return 0

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
