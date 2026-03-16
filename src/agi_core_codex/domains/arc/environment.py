from __future__ import annotations

from agi_core_codex.domains.arc.types import ArcTask, grid_cell_count


class ArcEnvironment:
    domain = "arc"

    def task_key(self, task: ArcTask) -> str:
        return task.task_id

    def task_size(self, task: ArcTask) -> int:
        total = 0
        for example in task.train:
            total += grid_cell_count(example.input)
        for case in task.test:
            total += grid_cell_count(case.input)
        return max(total, 1)

