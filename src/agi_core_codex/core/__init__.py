"""Active minimal learner surface.

The older strategy-oriented kernel remains in the repository for legacy
comparisons, but the default public core now points at the minimal
4-pillars learner primitives.
"""

from typing import Any

__all__ = [
    "LearnerMemory",
    "PrimitiveSpec",
    "Program",
    "RoundSummary",
    "WakeSleepLearner",
]


def __getattr__(name: str) -> Any:
    if name == "PrimitiveSpec":
        from agi_core_codex.minimal.ops import PrimitiveSpec

        return PrimitiveSpec
    if name in {"LearnerMemory", "Program", "RoundSummary", "WakeSleepLearner"}:
        from agi_core_codex.minimal.core import (
            LearnerMemory,
            Program,
            RoundSummary,
            WakeSleepLearner,
        )

        return {
            "LearnerMemory": LearnerMemory,
            "Program": Program,
            "RoundSummary": RoundSummary,
            "WakeSleepLearner": WakeSleepLearner,
        }[name]
    raise AttributeError(name)
