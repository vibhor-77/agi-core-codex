from __future__ import annotations

from agi_core_codex.core.strategies import GrammarPrimitiveStrategy, LibraryReplayStrategy
from agi_core_codex.domains.arc.strategies import (
    AbsolutePatchStrategy,
    ColorMapStrategy,
    ConstantOutputStrategy,
)


def build_arc_profile(profile: str):
    if profile == "baseline-core":
        return (GrammarPrimitiveStrategy(),)
    if profile == "arc-accuracy":
        return (
            GrammarPrimitiveStrategy(),
            ConstantOutputStrategy(),
            ColorMapStrategy(),
            AbsolutePatchStrategy(),
        )
    if profile == "arc-theory":
        return (
            LibraryReplayStrategy(),
            GrammarPrimitiveStrategy(),
            ConstantOutputStrategy(),
            ColorMapStrategy(),
            AbsolutePatchStrategy(),
        )
    raise ValueError(f"unknown profile: {profile}")

