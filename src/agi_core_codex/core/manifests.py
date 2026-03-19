from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricRecord:
    name: str
    value: float | int | str | bool
    higher_is_better: bool = True


@dataclass(frozen=True)
class PhaseRecord:
    name: str
    status: str
    generated: int
    evaluated: int
    consumed: dict[str, int]
    best_program_id: str | None = None
    best_train_accuracy: float | None = None


@dataclass(frozen=True)
class TaskRecord:
    task_key: str
    solved_train: bool
    solved_test: bool | None
    best_program_id: str | None
    best_program_name: str | None
    best_strategy: str | None
    train_accuracy: float
    test_accuracy: float | None
    failure_count: int
    budget_used: dict[str, int]
    test_verification_status: str
    phase_records: tuple[PhaseRecord, ...]
    family_name: str | None = None
    representation_summary: str | None = None
    verification_fail_reason: str | None = None
    genericity_score: float | None = None
    transfer_proxy_score: float | None = None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    profile: str
    mode: str
    domain: str
    split: str
    split_path: str
    code_hash: str
    env_hash: str
    python_version: str
    primitive_count: int
    strategy_set: tuple[str, ...]
    seed: int
    task_count: int
    metrics: tuple[MetricRecord, ...]
    tasks: tuple[TaskRecord, ...]
    notes: tuple[str, ...] = ()

    def metrics_as_dict(self) -> dict[str, float | int | str | bool]:
        return {metric.name: metric.value for metric in self.metrics}


@dataclass(frozen=True)
class ArtifactIndexEntry:
    run_id: str
    created_at: str
    profile: str
    mode: str
    domain: str
    split: str
    manifest_path: str
    metrics: tuple[MetricRecord, ...]


@dataclass(frozen=True)
class ArtifactIndex:
    entries: tuple[ArtifactIndexEntry, ...]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_manifest(manifest: RunManifest, output_root: Path) -> Path:
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{manifest.run_id}.json"
    path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def update_index(output_root: Path, entry: ArtifactIndexEntry) -> Path:
    index_path = output_root / "index.json"
    existing_entries: list[dict[str, Any]] = []
    if index_path.exists():
        existing_payload = json.loads(index_path.read_text())
        existing_entries = list(existing_payload.get("entries", []))

    filtered = [item for item in existing_entries if item.get("run_id") != entry.run_id]
    filtered.append(_jsonable(entry))
    filtered.sort(key=lambda item: (item["created_at"], item["run_id"]))
    index_path.write_text(json.dumps({"entries": filtered}, indent=2, sort_keys=True) + "\n")
    return index_path
