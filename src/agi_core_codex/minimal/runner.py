from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agi_core_codex.core.hashing import stable_hash
from agi_core_codex.core.manifests import (
    ArtifactIndexEntry,
    MetricRecord,
    PhaseRecord,
    RunManifest,
    TaskRecord,
    update_index,
    write_manifest,
)
from agi_core_codex.minimal.core import (
    CandidateResult,
    GridTask,
    LearnerMemory,
    RoundSummary,
    WakeSleepLearner,
    aggregate_round_metrics,
)
from agi_core_codex.minimal.domains import build_synthetic_curriculum, load_arc_tasks_minimal
from agi_core_codex.minimal.ops import compositor_specs, unary_seed_specs


@dataclass(frozen=True)
class MinimalRunOptions:
    domain: str
    mode: str
    output_root: Path
    rounds: int = 3
    seed: int = 0
    split_file: Path | None = None
    dataset_dir: Path | None = None
    benchmark: str = "arc-agi-1"
    dataset_split: str = "training"
    curriculum_tier: str = "all"
    limit: int | None = None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_hash(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "untracked"


def _env_hash() -> str:
    return stable_hash(
        {"python_version": sys.version, "platform": platform.platform()},
        namespace="environment",
    )


def _validate_dataset_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.json"))


def _discover_arc_dataset(*, benchmark: str, split: str) -> Path | None:
    benchmark_dir = {"arc-agi-1": "ARC-AGI", "arc-agi-2": "ARC-AGI-2"}[benchmark]
    repo_root = _repository_root()
    roots = (
        Path(os.environ.get(f"{benchmark.upper().replace('-', '_')}_{split.upper()}_DIR", "")),
        repo_root,
        repo_root.parent,
        repo_root.parent / "agi-core",
        Path.home() / "github" / "agi-core",
        Path.home() / "github" / "agi-core-codex",
    )
    candidates = []
    for root in roots:
        if str(root) == ".":
            continue
        candidates.extend(
            (
                root / benchmark_dir / "data" / split,
                root / "data" / benchmark_dir / "data" / split,
                root / split,
                root,
            )
        )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _validate_dataset_dir(resolved):
            return resolved
    return None


def _resolve_arc_dataset_dir(options: MinimalRunOptions) -> Path:
    if options.dataset_dir is not None:
        if not _validate_dataset_dir(options.dataset_dir):
            raise ValueError(f"invalid ARC dataset directory: {options.dataset_dir}")
        return options.dataset_dir
    if options.split_file is None:
        raise ValueError("arc minimal runs require --split-file")
    payload = None
    if options.split_file.exists():
        payload = __import__("json").loads(options.split_file.read_text())
    if isinstance(payload, dict):
        source = payload.get("source_dataset_dir")
        if isinstance(source, str):
            source_path = Path(source)
            if _validate_dataset_dir(source_path):
                return source_path
        benchmark = payload.get("benchmark")
        if isinstance(benchmark, str):
            discovered = _discover_arc_dataset(benchmark=benchmark, split=options.dataset_split)
            if discovered is not None:
                return discovered
    discovered = _discover_arc_dataset(benchmark=options.benchmark, split=options.dataset_split)
    if discovered is None:
        raise ValueError("could not auto-discover ARC dataset; pass --dataset-dir")
    return discovered


def _load_tasks(options: MinimalRunOptions) -> tuple[GridTask, ...]:
    if options.domain == "synthetic-grid":
        curriculum = build_synthetic_curriculum(options.curriculum_tier)
        tasks = curriculum.tasks
        return tasks[: options.limit] if options.limit is not None else tasks
    if options.domain == "arc":
        if options.split_file is None:
            raise ValueError("arc runs require --split-file")
        dataset_dir = _resolve_arc_dataset_dir(options)
        return load_arc_tasks_minimal(
            split_file=options.split_file,
            dataset_dir=dataset_dir,
            limit=options.limit,
        )
    raise ValueError(f"unsupported minimal domain: {options.domain}")


def _phase_records(round_summaries: tuple[RoundSummary, ...]) -> tuple[PhaseRecord, ...]:
    return tuple(
        PhaseRecord(
            name=f"round-{summary.round_index + 1}",
            status="ok",
            generated=summary.evaluated_candidates,
            evaluated=summary.evaluated_candidates,
            consumed={"evaluations": summary.evaluated_candidates},
            best_program_id=None,
            best_train_accuracy=None,
        )
        for summary in round_summaries
    )


def _task_record(task_run, round_summaries: tuple[RoundSummary, ...]) -> TaskRecord:
    best = task_run.best
    return TaskRecord(
        task_key=task_run.task_key,
        solved_train=bool(best and best.score.train_exact),
        solved_test=best.score.test_exact if best else None,
        best_program_id=best.program.id if best else None,
        best_program_name=best.program.name if best else None,
        best_strategy="minimal-wake-sleep",
        train_accuracy=best.score.train_accuracy if best else 0.0,
        test_accuracy=best.score.test_accuracy if best else None,
        failure_count=best.score.failure_count if best else 0,
        budget_used={"evaluations": task_run.evaluated_count},
        test_verification_status="available_and_checked" if best and best.score.test_accuracy is not None else "unavailable",
        phase_records=_phase_records(round_summaries),
    )


def _aggregate_metrics(task_records: tuple[TaskRecord, ...], round_summaries: tuple[RoundSummary, ...]) -> tuple[MetricRecord, ...]:
    task_count = len(task_records)
    solved_train = sum(1 for record in task_records if record.solved_train)
    train_accuracy = (
        sum(record.train_accuracy for record in task_records) / task_count
        if task_count
        else 0.0
    )
    test_eligible = [record for record in task_records if record.solved_test is not None]
    solved_test = sum(1 for record in test_eligible if record.solved_test)
    metrics = [
        MetricRecord("task_count", task_count, higher_is_better=False),
        MetricRecord("solved_train", solved_train),
        MetricRecord("train_exact_accuracy", solved_train / task_count if task_count else 0.0),
        MetricRecord("mean_train_accuracy", train_accuracy),
        MetricRecord("test_eligible_count", len(test_eligible), higher_is_better=False),
        MetricRecord("solved_test", solved_test),
        MetricRecord("test_exact_accuracy", solved_test / len(test_eligible) if test_eligible else 0.0),
    ]
    for name, value in aggregate_round_metrics(round_summaries).items():
        metrics.append(MetricRecord(name, value))
    return tuple(metrics)


def run_minimal(options: MinimalRunOptions) -> tuple[RunManifest, Path]:
    tasks = _load_tasks(options)
    learner = WakeSleepLearner(
        unary_primitives=unary_seed_specs(),
        binary_compositors=compositor_specs(),
    )
    memory = LearnerMemory()
    final_task_runs = ()
    round_summaries: list[RoundSummary] = []

    for round_index in range(options.rounds):
        task_runs, summary = learner.run_round(
            tasks=tasks,
            memory=memory,
            round_index=round_index,
        )
        final_task_runs = task_runs
        round_summaries.append(summary)

    round_summary_tuple = tuple(round_summaries)
    task_records = tuple(_task_record(task_run, round_summary_tuple) for task_run in final_task_runs)
    metrics = _aggregate_metrics(task_records, round_summary_tuple)

    created_at = datetime.now(timezone.utc).isoformat()
    run_id = stable_hash(
        {
            "created_at": created_at,
            "domain": options.domain,
            "mode": options.mode,
            "rounds": options.rounds,
            "split_file": str(options.split_file) if options.split_file else None,
            "tier": options.curriculum_tier,
            "seed": options.seed,
        },
        namespace="run",
    )[:16]

    notes = [
        "active minimal learner path",
        "compounding evidence prioritized over benchmark tactics",
        f"domain={options.domain}",
        f"rounds={options.rounds}",
        f"seed={options.seed}",
    ]
    if options.domain == "synthetic-grid":
        notes.append(f"curriculum_tier={options.curriculum_tier}")
    if options.split_file is not None:
        notes.append(f"split_file={options.split_file}")

    manifest = RunManifest(
        run_id=run_id,
        created_at=created_at,
        profile="minimal",
        mode=options.mode,
        domain=options.domain,
        split=options.split_file.stem if options.split_file else options.curriculum_tier,
        split_path=str(options.split_file) if options.split_file else "",
        code_hash=_code_hash(_repository_root()),
        env_hash=_env_hash(),
        python_version=sys.version.split()[0],
        primitive_count=len(unary_seed_specs()) + len(compositor_specs()),
        strategy_set=("minimal-wake-sleep",),
        seed=options.seed,
        task_count=len(task_records),
        metrics=metrics,
        tasks=task_records,
        notes=tuple(notes),
    )
    manifest_path = write_manifest(manifest, options.output_root)
    update_index(
        options.output_root,
        ArtifactIndexEntry(
            run_id=manifest.run_id,
            created_at=manifest.created_at,
            profile=manifest.profile,
            mode=manifest.mode,
            domain=manifest.domain,
            split=manifest.split,
            manifest_path=str(manifest_path),
            metrics=manifest.metrics,
        ),
    )
    return manifest, manifest_path
