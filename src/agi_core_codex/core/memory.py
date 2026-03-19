from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from agi_core_codex.core.interfaces import ProgramHandle


@dataclass(frozen=True)
class LibraryEntry:
    program: ProgramHandle
    train_accuracy: float
    origin_task_key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InMemoryLibrary:
    def __init__(self) -> None:
        self._by_domain: dict[str, list[LibraryEntry]] = {}
        self._seen_ids: dict[str, set[str]] = {}

    def store(self, domain: str, entry: LibraryEntry) -> None:
        seen = self._seen_ids.setdefault(domain, set())
        if entry.program.id in seen:
            return
        seen.add(entry.program.id)
        self._by_domain.setdefault(domain, []).append(entry)

    def recall(self, domain: str, limit: int | None = None) -> tuple[LibraryEntry, ...]:
        entries = tuple(self._by_domain.get(domain, ()))
        if limit is None:
            return entries
        return entries[:limit]


class StagedMemory:
    def __init__(self) -> None:
        self._committed = InMemoryLibrary()
        self._pending_by_domain: dict[str, list[LibraryEntry]] = {}
        self._pending_ids: dict[str, set[str]] = {}

    def store(self, domain: str, entry: LibraryEntry) -> None:
        committed_ids = self._committed._seen_ids.setdefault(domain, set())
        pending_ids = self._pending_ids.setdefault(domain, set())
        if entry.program.id in committed_ids or entry.program.id in pending_ids:
            return
        pending_ids.add(entry.program.id)
        self._pending_by_domain.setdefault(domain, []).append(entry)

    def recall(self, domain: str, limit: int | None = None) -> tuple[LibraryEntry, ...]:
        return self._committed.recall(domain, limit=limit)

    def commit(self, domain: str) -> int:
        pending = tuple(self._pending_by_domain.get(domain, ()))
        for entry in pending:
            self._committed.store(domain, entry)
        self._pending_by_domain[domain] = []
        self._pending_ids[domain] = set()
        return len(pending)

    def size(self, domain: str) -> int:
        return len(self._committed.recall(domain))
