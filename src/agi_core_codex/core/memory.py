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

