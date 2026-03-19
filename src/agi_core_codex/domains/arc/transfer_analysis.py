from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agi_core_codex.core.manifests import load_manifest
from agi_core_codex.domains.arc.discovery import discover_arc_dataset, validate_dataset_dir
from agi_core_codex.domains.arc.loader import load_arc_tasks, load_split_ids, load_split_payload
from agi_core_codex.domains.arc.transfer import build_arc_task_representation


@dataclass(frozen=True)
class TransferTaskDiff:
    task_key: str
    status: str
    baseline_solved_train: bool
    candidate_solved_train: bool
    baseline_strategy: str | None
    candidate_strategy: str | None
    baseline_family_name: str | None
    candidate_family_name: str | None


@dataclass(frozen=True)
class TransferDiffReport:
    baseline_manifest_path: str
    candidate_manifest_path: str
    task_count: int
    gained: int
    lost: int
    shifted_best_strategy: int
    unchanged: int
    tasks: tuple[TransferTaskDiff, ...]


@dataclass(frozen=True)
class TransferFailureTask:
    task_key: str
    best_strategy: str | None
    family_name: str | None
    train_accuracy: float
    representation_cluster: str
    failure_reason: str


@dataclass(frozen=True)
class TransferFailureClusterReport:
    manifest_path: str
    split: str
    unsolved_task_count: int
    representation_clusters: dict[str, int]
    failure_reasons: dict[str, int]
    family_counts: dict[str, int]
    tasks: tuple[TransferFailureTask, ...]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _task_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(task["task_key"]): task for task in manifest.get("tasks", [])}


def diff_transfer_manifests(
    *,
    baseline_manifest_path: Path,
    candidate_manifest_path: Path,
) -> TransferDiffReport:
    baseline = load_manifest(baseline_manifest_path)
    candidate = load_manifest(candidate_manifest_path)
    baseline_tasks = _task_lookup(baseline)
    candidate_tasks = _task_lookup(candidate)
    task_keys = tuple(sorted(set(baseline_tasks) | set(candidate_tasks)))

    diffs: list[TransferTaskDiff] = []
    gained = 0
    lost = 0
    shifted = 0
    unchanged = 0
    for task_key in task_keys:
        before = baseline_tasks.get(task_key, {})
        after = candidate_tasks.get(task_key, {})
        before_solved = bool(before.get("solved_train"))
        after_solved = bool(after.get("solved_train"))
        if not before_solved and after_solved:
            status = "gained"
            gained += 1
        elif before_solved and not after_solved:
            status = "lost"
            lost += 1
        elif before.get("best_strategy") != after.get("best_strategy"):
            status = "shifted_best_strategy"
            shifted += 1
        else:
            status = "unchanged"
            unchanged += 1
        diffs.append(
            TransferTaskDiff(
                task_key=task_key,
                status=status,
                baseline_solved_train=before_solved,
                candidate_solved_train=after_solved,
                baseline_strategy=before.get("best_strategy"),
                candidate_strategy=after.get("best_strategy"),
                baseline_family_name=before.get("family_name"),
                candidate_family_name=after.get("family_name"),
            )
        )

    return TransferDiffReport(
        baseline_manifest_path=str(baseline_manifest_path),
        candidate_manifest_path=str(candidate_manifest_path),
        task_count=len(task_keys),
        gained=gained,
        lost=lost,
        shifted_best_strategy=shifted,
        unchanged=unchanged,
        tasks=tuple(diffs),
    )


def format_transfer_diff(report: TransferDiffReport, *, output_format: str = "text") -> str:
    if output_format == "json":
        return json.dumps(_jsonable(report), indent=2, sort_keys=True)

    lines = [
        f"baseline={report.baseline_manifest_path}",
        f"candidate={report.candidate_manifest_path}",
        f"task_count={report.task_count}",
        f"gained={report.gained}",
        f"lost={report.lost}",
        f"shifted_best_strategy={report.shifted_best_strategy}",
        f"unchanged={report.unchanged}",
    ]
    if report.tasks:
        lines.append("tasks:")
        for task in report.tasks:
            lines.append(
                "  "
                f"{task.task_key}: {task.status} "
                f"baseline={task.baseline_strategy or 'none'} "
                f"candidate={task.candidate_strategy or 'none'}"
            )
    return "\n".join(lines)


def _resolve_dataset_dir(dataset_dir: Path | None, manifest: dict[str, Any]) -> Path:
    if dataset_dir is not None:
        if not validate_dataset_dir(dataset_dir):
            raise ValueError(f"invalid dataset directory: {dataset_dir}")
        return dataset_dir

    split_path = Path(str(manifest["split_path"]))
    split_payload = load_split_payload(split_path)
    source_dataset_dir = split_payload.get("source_dataset_dir")
    if isinstance(source_dataset_dir, str):
        source_path = Path(source_dataset_dir)
        if validate_dataset_dir(source_path):
            return source_path

    split_benchmark = split_payload.get("benchmark")
    if isinstance(split_benchmark, str):
        discovered = discover_arc_dataset(benchmark=split_benchmark, split="training")
        if discovered is not None:
            return discovered
    raise ValueError("could not resolve ARC dataset directory for transfer failure clustering")


def _representation_cluster(summary_features: dict[str, Any]) -> str:
    if int(summary_features.get("separator_count", 0)) > 0:
        return "separator"
    if int(summary_features.get("repeated_templates", 0)) > 0:
        return "template"
    if int(summary_features.get("component_count_max", 0)) > 1:
        return "multi_object"
    if summary_features.get("symmetry_axes"):
        return "symmetry"
    if not summary_features.get("same_shape", True):
        return "shape_change"
    return "other"


def cluster_transfer_failures(
    *,
    manifest_path: Path,
    dataset_dir: Path | None = None,
) -> TransferFailureClusterReport:
    manifest = load_manifest(manifest_path)
    resolved_dataset_dir = _resolve_dataset_dir(dataset_dir, manifest)
    split_ids = load_split_ids(Path(str(manifest["split_path"])))
    tasks = load_arc_tasks(resolved_dataset_dir, split_ids=split_ids)
    task_map = {task.task_id: task for task in tasks}

    classifications: list[TransferFailureTask] = []
    representation_clusters: dict[str, int] = {}
    failure_reasons: dict[str, int] = {}
    family_counts: dict[str, int] = {}

    for task_record in manifest.get("tasks", []):
        if task_record.get("solved_train"):
            continue
        task_key = str(task_record["task_key"])
        task = task_map[task_key]
        representation = build_arc_task_representation(task)
        cluster = _representation_cluster(dict(representation.features))
        failure_reason = str(task_record.get("verification_fail_reason") or "unknown")
        family_name = task_record.get("family_name")

        representation_clusters[cluster] = representation_clusters.get(cluster, 0) + 1
        failure_reasons[failure_reason] = failure_reasons.get(failure_reason, 0) + 1
        family_key = str(family_name or "none")
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        classifications.append(
            TransferFailureTask(
                task_key=task_key,
                best_strategy=task_record.get("best_strategy"),
                family_name=family_name,
                train_accuracy=float(task_record.get("train_accuracy", 0.0)),
                representation_cluster=cluster,
                failure_reason=failure_reason,
            )
        )

    classifications.sort(key=lambda item: (-item.train_accuracy, item.task_key))
    return TransferFailureClusterReport(
        manifest_path=str(manifest_path),
        split=str(manifest["split"]),
        unsolved_task_count=len(classifications),
        representation_clusters=dict(sorted(representation_clusters.items())),
        failure_reasons=dict(sorted(failure_reasons.items())),
        family_counts=dict(sorted(family_counts.items())),
        tasks=tuple(classifications),
    )


def format_transfer_failure_clusters(
    report: TransferFailureClusterReport,
    *,
    output_format: str = "text",
) -> str:
    if output_format == "json":
        return json.dumps(_jsonable(report), indent=2, sort_keys=True)

    lines = [
        f"manifest={report.manifest_path}",
        f"split={report.split}",
        f"unsolved_tasks={report.unsolved_task_count}",
        f"representation_clusters={report.representation_clusters}",
        f"failure_reasons={report.failure_reasons}",
        f"family_counts={report.family_counts}",
    ]
    if report.tasks:
        lines.append("tasks:")
        for task in report.tasks:
            lines.append(
                "  "
                f"{task.task_key}: cluster={task.representation_cluster} "
                f"reason={task.failure_reason} family={task.family_name or 'none'} "
                f"train_accuracy={task.train_accuracy:.4f}"
            )
    return "\n".join(lines)
