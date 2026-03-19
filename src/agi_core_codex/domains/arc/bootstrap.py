from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agi_core_codex.core.interfaces import CostModel, ProgramHandle
from agi_core_codex.core.manifests import MetricRecord
from agi_core_codex.core.memory import LibraryEntry, StagedMemory
from agi_core_codex.core.strategies import EnumerativeCompositionStrategy
from agi_core_codex.domains.arc.analysis import fill_enclosed
from agi_core_codex.domains.arc.grammar import (
    _crop_nonzero,
    _flip_horizontal,
    _flip_vertical,
    _rotate_180,
    _transpose,
)
from agi_core_codex.domains.arc.programs import make_arc_program


def _identity(grid):
    return grid


@dataclass(frozen=True)
class ArcBootstrapGrammar:
    domain: str = "arc"

    def primitive_count(self, task: Any | None = None) -> int:
        return len(self.enumerate_primitives(task))

    def enumerate_primitives(self, task: Any) -> Sequence[ProgramHandle]:
        return (
            make_arc_program(
                name="identity",
                semantics={"type": "bootstrap_seed", "seed_name": "identity"},
                executor=_identity,
                complexity=1,
            ),
            make_arc_program(
                name="flip-horizontal",
                semantics={"type": "bootstrap_seed", "seed_name": "flip_horizontal"},
                executor=_flip_horizontal,
                complexity=1,
            ),
            make_arc_program(
                name="flip-vertical",
                semantics={"type": "bootstrap_seed", "seed_name": "flip_vertical"},
                executor=_flip_vertical,
                complexity=1,
            ),
            make_arc_program(
                name="rotate-180",
                semantics={"type": "bootstrap_seed", "seed_name": "rotate_180"},
                executor=_rotate_180,
                complexity=1,
            ),
            make_arc_program(
                name="transpose",
                semantics={"type": "bootstrap_seed", "seed_name": "transpose"},
                executor=_transpose,
                complexity=1,
            ),
            make_arc_program(
                name="crop-nonzero",
                semantics={"type": "bootstrap_seed", "seed_name": "crop_nonzero"},
                executor=_crop_nonzero,
                complexity=2,
            ),
            make_arc_program(
                name="fill-enclosed",
                semantics={"type": "bootstrap_seed", "seed_name": "fill_enclosed"},
                executor=fill_enclosed,
                complexity=2,
            ),
        )


@dataclass(frozen=True)
class BootstrapRoundSummary:
    round_index: int
    solved_train: int
    task_count: int
    library_size: int
    committed_entries: int


def build_arc_bootstrap_strategies(*, round_index: int) -> tuple[EnumerativeCompositionStrategy, ...]:
    return (
        EnumerativeCompositionStrategy(
            max_depth=1 if round_index == 0 else 2,
            beam_width=8,
            recall_limit=16,
            require_library_component=round_index > 0,
            cost_model=CostModel(complexity=1),
        ),
    )


def sleep_promote_bootstrap_candidates(
    *,
    reports: Sequence[Any],
    memory: StagedMemory,
    near_miss_threshold: float = 0.95,
    max_promotions: int = 32,
) -> int:
    ranked: list[tuple[str, float, str, Any]] = []
    for report in reports:
        for strategy_result in report.strategy_results:
            for candidate in strategy_result.candidates:
                if candidate.program.cost.complexity <= 1:
                    continue
                if candidate.score.train_accuracy < near_miss_threshold:
                    continue
                ranked.append(
                    (
                        candidate.program.id,
                        candidate.score.train_accuracy,
                        report.task_key,
                        candidate,
                    )
                )

    ranked.sort(key=lambda item: (item[1], -item[3].program.cost.complexity, item[0]), reverse=True)
    promotions = 0
    seen_ids: set[str] = set()
    for _, train_accuracy, task_key, candidate in ranked:
        if candidate.program.id in seen_ids:
            continue
        seen_ids.add(candidate.program.id)
        memory.store(
            "arc",
            LibraryEntry(
                program=candidate.program,
                train_accuracy=train_accuracy,
                origin_task_key=task_key,
                metadata={
                    "promotion_source": "bootstrap_sleep",
                    "strategy_name": candidate.strategy_name,
                },
            ),
        )
        promotions += 1
        if promotions >= max_promotions:
            break
    return promotions


def bootstrap_round_metrics(round_summaries: Sequence[BootstrapRoundSummary]) -> tuple[MetricRecord, ...]:
    metrics: list[MetricRecord] = []
    for summary in round_summaries:
        suffix = f"round_{summary.round_index + 1}"
        metrics.append(MetricRecord(f"{suffix}_solved_train", summary.solved_train))
        metrics.append(
            MetricRecord(
                f"{suffix}_train_exact_accuracy",
                summary.solved_train / summary.task_count if summary.task_count else 0.0,
            )
        )
        metrics.append(MetricRecord(f"{suffix}_library_size", summary.library_size))
        metrics.append(
            MetricRecord(f"{suffix}_committed_entries", summary.committed_entries)
        )
    if round_summaries:
        metrics.append(MetricRecord("round_count", len(round_summaries), higher_is_better=False))
        metrics.append(MetricRecord("final_library_size", round_summaries[-1].library_size))
    return tuple(metrics)
