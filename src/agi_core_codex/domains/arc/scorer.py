from __future__ import annotations

from agi_core_codex.core.interfaces import ProgramHandle, ScoreBreakdown
from agi_core_codex.domains.arc.types import ArcTask, freeze_grid, grid_accuracy


class ArcScorer:
    domain = "arc"

    def evaluate_program(self, task: ArcTask, program: ProgramHandle) -> ScoreBreakdown:
        train_accuracies: list[float] = []
        test_accuracies: list[float] = []
        test_predictions = []
        notes: list[str] = []
        failures = 0

        for example in task.train:
            try:
                predicted = freeze_grid(program.executor(example.input))
            except Exception as exc:
                failures += 1
                train_accuracies.append(0.0)
                notes.append(f"train_error:{type(exc).__name__}:{exc}")
                continue
            train_accuracies.append(grid_accuracy(example.output, predicted))

        for case in task.test:
            try:
                predicted = freeze_grid(program.executor(case.input))
            except Exception as exc:
                failures += 1
                predicted = ()
                notes.append(f"test_error:{type(exc).__name__}:{exc}")
            test_predictions.append(predicted)
            if case.output is not None:
                test_accuracies.append(grid_accuracy(case.output, predicted))

        train_exact = bool(train_accuracies) and all(score == 1.0 for score in train_accuracies) and failures == 0
        test_exact = None
        test_accuracy = None
        if test_accuracies:
            test_exact = all(score == 1.0 for score in test_accuracies) and failures == 0
            test_accuracy = sum(test_accuracies) / len(test_accuracies)

        return ScoreBreakdown(
            train_exact=train_exact,
            train_accuracy=sum(train_accuracies) / len(train_accuracies),
            example_accuracies=tuple(train_accuracies),
            failure_count=failures,
            test_exact=test_exact,
            test_accuracy=test_accuracy,
            test_predictions=tuple(test_predictions),
            notes=tuple(notes),
        )

