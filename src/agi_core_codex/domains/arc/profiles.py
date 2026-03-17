from __future__ import annotations

from collections.abc import Sequence

from agi_core_codex.core.strategies import GrammarPrimitiveStrategy, LibraryReplayStrategy
from agi_core_codex.domains.arc.strategies import (
    AbsolutePatchStrategy,
    BooleanHalvesStrategy,
    ColorMapStrategy,
    ConstantOutputStrategy,
    RowColumnDecompositionStrategy,
    ScaleTileStrategy,
    SeparatorPropagationStrategy,
    SeparatorCrossReferenceStrategy,
    TemplateStampStrategy,
)


def _instantiate(name: str):
    registry = {
        "library-replay": LibraryReplayStrategy,
        "grammar-primitives": GrammarPrimitiveStrategy,
        "arc-boolean-halves": BooleanHalvesStrategy,
        "arc-row-column-decomposition": RowColumnDecompositionStrategy,
        "arc-separator-cross-reference": SeparatorCrossReferenceStrategy,
        "arc-separator-propagation": SeparatorPropagationStrategy,
        "arc-scale-tile": ScaleTileStrategy,
        "arc-template-stamp": TemplateStampStrategy,
        "arc-constant-output": ConstantOutputStrategy,
        "arc-color-map": ColorMapStrategy,
        "arc-absolute-patch": AbsolutePatchStrategy,
    }
    try:
        return registry[name]()
    except KeyError as exc:  # pragma: no cover - defensive validation
        raise ValueError(f"unknown strategy: {name}") from exc


def profile_strategy_names(profile: str) -> tuple[str, ...]:
    if profile == "baseline-core":
        return ("grammar-primitives",)
    if profile == "arc-accuracy":
        return (
            "arc-boolean-halves",
            "arc-row-column-decomposition",
            "arc-separator-propagation",
            "arc-separator-cross-reference",
            "arc-scale-tile",
            "arc-template-stamp",
            "grammar-primitives",
            "arc-constant-output",
            "arc-color-map",
            "arc-absolute-patch",
        )
    if profile == "arc-theory":
        return (
            "library-replay",
            "arc-boolean-halves",
            "arc-row-column-decomposition",
            "arc-separator-propagation",
            "arc-separator-cross-reference",
            "arc-scale-tile",
            "arc-template-stamp",
            "grammar-primitives",
            "arc-constant-output",
            "arc-color-map",
            "arc-absolute-patch",
        )
    raise ValueError(f"unknown profile: {profile}")


def build_arc_profile(
    profile: str,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
):
    selected = profile_strategy_names(profile)
    include_set = tuple(include)
    if include_set:
        unknown = [name for name in include_set if name not in selected]
        if unknown:
            raise ValueError(
                f"strategies not available in profile {profile}: {', '.join(sorted(unknown))}"
            )
        selected = tuple(name for name in selected if name in include_set)
    if exclude:
        selected = tuple(name for name in selected if name not in set(exclude))
    if not selected:
        raise ValueError(f"profile {profile} has no remaining strategies after filtering")
    return tuple(_instantiate(name) for name in selected)
