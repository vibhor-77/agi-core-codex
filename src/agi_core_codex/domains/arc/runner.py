from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agi_core_codex.core.hashing import stable_hash
from agi_core_codex.core.interfaces import SearchBudget
from agi_core_codex.core.kernel import SearchKernel
from agi_core_codex.core.manifests import (
    ArtifactIndexEntry,
    MetricRecord,
    PhaseRecord,
    RunManifest,
    TaskRecord,
    update_index,
    write_manifest,
)
from agi_core_codex.core.memory import InMemoryLibrary, StagedMemory
from agi_core_codex.domains.arc.bootstrap import (
    ArcBootstrapGrammar,
    BootstrapRoundSummary,
    bootstrap_round_metrics,
    build_arc_bootstrap_strategies,
    sleep_promote_bootstrap_candidates,
)
from agi_core_codex.domains.arc.discovery import discover_arc_dataset, validate_dataset_dir
from agi_core_codex.domains.arc.environment import ArcEnvironment
from agi_core_codex.domains.arc.grammar import ArcGrammar
from agi_core_codex.domains.arc.loader import load_arc_tasks, load_split_ids, load_split_payload
from agi_core_codex.domains.arc.profiles import build_arc_profile
from agi_core_codex.domains.arc.scorer import ArcScorer
from agi_core_codex.domains.arc.transfer import build_arc_transfer_profile


@dataclass(frozen=True)
class ArcRunOptions:
    profile: str
    mode: str
    dataset_dir: Path | None
    split_file: Path
    output_root: Path
    seed: int = 0
    limit: int | None = None
    include_strategies: tuple[str, ...] = ()
    exclude_strategies: tuple[str, ...] = ()
    benchmark: str | None = None
    dataset_split: str = "training"
    rounds: int = 1


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


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
    payload = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    return stable_hash(payload, namespace="environment")


def _build_budget(profile: str, primitive_count: int, task_cells: int) -> SearchBudget:
    bonus = {
        "baseline-core": 0,
        "arc-accuracy": 12,
        "arc-theory": 16,
        "arc-transfer": 8,
        "arc-bootstrap": 24,
    }[profile]
    multiplier = {
        "baseline-core": 1,
        "arc-accuracy": 1,
        "arc-theory": 2,
        "arc-transfer": 1,
        "arc-bootstrap": 4,
    }[profile]
    max_evaluations = max(primitive_count * multiplier + bonus, 8)
    return SearchBudget(
        max_evaluations=max_evaluations,
        max_cell_evaluations=max_evaluations * max(task_cells, 1),
        max_program_complexity=max(primitive_count, 12),
    )


def _phase_records(search_report) -> tuple[PhaseRecord, ...]:
    records = []
    for result in search_report.strategy_results:
        best_train_accuracy = None
        best_program_id = None
        if result.candidates:
            top = sorted(
                result.candidates,
                key=lambda candidate: (
                    candidate.score.train_accuracy,
                    -candidate.score.failure_count,
                    candidate.program.id,
                ),
                reverse=True,
            )[0]
            best_train_accuracy = top.score.train_accuracy
            best_program_id = top.program.id
        records.append(
            PhaseRecord(
                name=result.name,
                status=result.status,
                generated=result.generated,
                evaluated=result.evaluated,
                consumed=result.consumed.to_dict(),
                best_program_id=best_program_id,
                best_train_accuracy=best_train_accuracy,
            )
        )
    return tuple(records)


def _task_record(search_report) -> TaskRecord:
    best = search_report.best_candidate
    test_verification_status = "not_checked"
    if best is not None:
        if best.score.test_accuracy is None:
            test_verification_status = "unavailable"
        else:
            test_verification_status = "available_and_checked"
    metadata = dict(best.metadata) if best is not None else {}

    return TaskRecord(
        task_key=search_report.task_key,
        solved_train=bool(best and best.score.train_exact),
        solved_test=best.score.test_exact if best else None,
        best_program_id=best.program.id if best else None,
        best_program_name=best.program.name if best else None,
        best_strategy=best.strategy_name if best else None,
        train_accuracy=best.score.train_accuracy if best else 0.0,
        test_accuracy=best.score.test_accuracy if best else None,
        failure_count=best.score.failure_count if best else 0,
        budget_used=search_report.budget_used.to_dict(),
        test_verification_status=test_verification_status,
        phase_records=_phase_records(search_report),
        family_name=metadata.get("family_name"),
        representation_summary=metadata.get("representation_summary"),
        verification_fail_reason=metadata.get("verification_fail_reason"),
        genericity_score=metadata.get("genericity_score"),
        transfer_proxy_score=metadata.get("transfer_proxy_score"),
    )


def _aggregate_metrics(task_records: tuple[TaskRecord, ...]) -> tuple[MetricRecord, ...]:
    task_count = len(task_records)
    solved_train = sum(1 for record in task_records if record.solved_train)
    train_accuracy_mean = (
        sum(record.train_accuracy for record in task_records) / task_count if task_count else 0.0
    )

    test_eligible = [record for record in task_records if record.solved_test is not None]
    solved_test = sum(1 for record in test_eligible if record.solved_test)
    test_accuracy_mean = (
        sum((record.test_accuracy or 0.0) for record in test_eligible) / len(test_eligible)
        if test_eligible
        else 0.0
    )

    transfer_scores = [
        record.transfer_proxy_score
        for record in task_records
        if record.transfer_proxy_score is not None
    ]
    genericity_scores = [
        record.genericity_score
        for record in task_records
        if record.genericity_score is not None
    ]

    metrics = [
        MetricRecord("task_count", task_count, higher_is_better=False),
        MetricRecord("solved_train", solved_train),
        MetricRecord("train_exact_accuracy", solved_train / task_count if task_count else 0.0),
        MetricRecord("mean_train_accuracy", train_accuracy_mean),
        MetricRecord("test_eligible_count", len(test_eligible), higher_is_better=False),
        MetricRecord("solved_test", solved_test),
        MetricRecord(
            "test_exact_accuracy",
            solved_test / len(test_eligible) if test_eligible else 0.0,
        ),
        MetricRecord("mean_test_accuracy", test_accuracy_mean),
    ]
    if genericity_scores:
        metrics.append(
            MetricRecord(
                "mean_genericity_score",
                sum(genericity_scores) / len(genericity_scores),
            )
        )
    if transfer_scores:
        metrics.append(
            MetricRecord(
                "mean_transfer_proxy_score",
                sum(transfer_scores) / len(transfer_scores),
            )
        )
    return tuple(metrics)


def _resolve_dataset_dir(options: ArcRunOptions) -> Path:
    if options.dataset_dir is not None:
        return options.dataset_dir

    split_payload = load_split_payload(options.split_file)
    source_dataset_dir = split_payload.get("source_dataset_dir")
    if isinstance(source_dataset_dir, str):
        source_path = Path(source_dataset_dir)
        if validate_dataset_dir(source_path):
            return source_path

    benchmark = options.benchmark or split_payload.get("benchmark")
    if not isinstance(benchmark, str):
        raise ValueError(
            "dataset directory was not provided and split metadata does not contain a usable benchmark"
        )

    discovered = discover_arc_dataset(benchmark=benchmark, split=options.dataset_split)
    if discovered is None:
        raise ValueError(
            f"could not discover dataset for benchmark={benchmark} split={options.dataset_split}"
        )
    return discovered


def run_arc_profile(options: ArcRunOptions) -> tuple[RunManifest, Path]:
    split_ids = load_split_ids(options.split_file)
    dataset_dir = _resolve_dataset_dir(options)
    tasks = load_arc_tasks(dataset_dir, split_ids=split_ids, limit=options.limit)

    if options.profile == "arc-bootstrap":
        return _run_arc_bootstrap_profile(options, tasks, dataset_dir)

    environment = ArcEnvironment()
    grammar = ArcGrammar()
    scorer = ArcScorer()
    memory = InMemoryLibrary()
    if options.profile == "arc-transfer":
        strategies = build_arc_transfer_profile(
            include=options.include_strategies,
            exclude=options.exclude_strategies,
        )
    else:
        strategies = build_arc_profile(
            options.profile,
            include=options.include_strategies,
            exclude=options.exclude_strategies,
        )
    kernel = SearchKernel(strategies)

    task_records = []
    for task in tasks:
        primitive_count = grammar.primitive_count(task)
        budget = _build_budget(options.profile, primitive_count, environment.task_size(task))
        report = kernel.run(
            task=task,
            environment=environment,
            grammar=grammar,
            scorer=scorer,
            memory=memory,
            budget=budget,
            seed=options.seed,
        )
        task_records.append(_task_record(report))

    metrics = _aggregate_metrics(tuple(task_records))
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = stable_hash(
        {
            "created_at": created_at,
            "profile": options.profile,
            "mode": options.mode,
            "split_file": str(options.split_file),
            "seed": options.seed,
        },
        namespace="run",
    )[:16]

    manifest = RunManifest(
        run_id=run_id,
        created_at=created_at,
        profile=options.profile,
        mode=options.mode,
        domain="arc",
        split=options.split_file.stem,
        split_path=str(options.split_file),
        code_hash=_code_hash(_repository_root()),
        env_hash=_env_hash(),
        python_version=sys.version.split()[0],
        primitive_count=max((ArcGrammar().primitive_count(task) for task in tasks), default=0),
        strategy_set=tuple(strategy.name for strategy in strategies),
        seed=options.seed,
        task_count=len(task_records),
        metrics=metrics,
        tasks=tuple(task_records),
        notes=tuple(
            note
            for note in (
                "public eval should remain checkpoint-only",
                (
                    "transfer track is family-based and does not inherit the baseline strategy zoo"
                    if options.profile == "arc-transfer"
                    else None
                ),
                f"mode={options.mode}",
                f"dataset_dir={dataset_dir}",
                (
                    f"include_strategies={','.join(options.include_strategies)}"
                    if options.include_strategies
                    else None
                ),
                (
                    f"exclude_strategies={','.join(options.exclude_strategies)}"
                    if options.exclude_strategies
                    else None
                ),
            )
            if note is not None
        ),
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


def _run_arc_bootstrap_profile(
    options: ArcRunOptions,
    tasks,
    dataset_dir: Path,
) -> tuple[RunManifest, Path]:
    environment = ArcEnvironment()
    grammar = ArcBootstrapGrammar()
    scorer = ArcScorer()
    memory = StagedMemory()

    final_task_records = ()
    round_summaries: list[BootstrapRoundSummary] = []
    for round_index in range(options.rounds):
        strategies = build_arc_bootstrap_strategies(round_index=round_index)
        kernel = SearchKernel(strategies)
        reports = []
        task_records = []
        for task in tasks:
            primitive_count = grammar.primitive_count(task)
            budget = _build_budget(options.profile, primitive_count, environment.task_size(task))
            report = kernel.run(
                task=task,
                environment=environment,
                grammar=grammar,
                scorer=scorer,
                memory=memory,
                budget=budget,
                seed=options.seed,
            )
            reports.append(report)
            task_records.append(_task_record(report))

        sleep_promote_bootstrap_candidates(reports=reports, memory=memory)
        committed_entries = memory.commit("arc")
        solved_train = sum(1 for record in task_records if record.solved_train)
        round_summaries.append(
            BootstrapRoundSummary(
                round_index=round_index,
                solved_train=solved_train,
                task_count=len(task_records),
                library_size=memory.size("arc"),
                committed_entries=committed_entries,
            )
        )
        final_task_records = tuple(task_records)

    metrics = _aggregate_metrics(final_task_records) + bootstrap_round_metrics(round_summaries)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = stable_hash(
        {
            "created_at": created_at,
            "profile": options.profile,
            "mode": options.mode,
            "split_file": str(options.split_file),
            "seed": options.seed,
            "rounds": options.rounds,
        },
        namespace="run",
    )[:16]

    manifest = RunManifest(
        run_id=run_id,
        created_at=created_at,
        profile=options.profile,
        mode=options.mode,
        domain="arc",
        split=options.split_file.stem,
        split_path=str(options.split_file),
        code_hash=_code_hash(_repository_root()),
        env_hash=_env_hash(),
        python_version=sys.version.split()[0],
        primitive_count=grammar.primitive_count(None),
        strategy_set=tuple(strategy.name for strategy in build_arc_bootstrap_strategies(round_index=0)),
        seed=options.seed,
        task_count=len(final_task_records),
        metrics=metrics,
        tasks=final_task_records,
        notes=tuple(
            note
            for note in (
                "bootstrap track uses staged wake/sleep memory",
                "public eval should remain checkpoint-only",
                f"mode={options.mode}",
                f"dataset_dir={dataset_dir}",
                f"rounds={options.rounds}",
                "tiny seed grammar + generic sequential composition only",
            )
            if note is not None
        ),
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
