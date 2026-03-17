from __future__ import annotations

from collections.abc import Sequence

from agi_core_codex.core.strategies import GrammarPrimitiveStrategy, LibraryReplayStrategy
from agi_core_codex.domains.arc.strategies import (
    AbsolutePatchStrategy,
    BboxRecolorStrategy,
    BboxRingMarkerProjectionStrategy,
    BottomCenterMarkerStrategy,
    BooleanHalvesStrategy,
    ColorMapStrategy,
    ConstantOutputStrategy,
    ExtractTransformStrategy,
    InteriorExtractStrategy,
    RayExtensionStrategy,
    RectangleMarkerProjectionStrategy,
    RowColumnDecompositionStrategy,
    ScaleTileStrategy,
    SeparatorPropagationStrategy,
    SeparatorCrossReferenceStrategy,
    TemplateStampStrategy,
    ZeroSquareFillStrategy,
)


def _instantiate(name: str):
    registry = {
        "library-replay": LibraryReplayStrategy,
        "grammar-primitives": GrammarPrimitiveStrategy,
        "arc-boolean-halves": BooleanHalvesStrategy,
        "arc-bbox-recolor": BboxRecolorStrategy,
        "arc-bbox-ring-marker-projection": BboxRingMarkerProjectionStrategy,
        "arc-bottom-center-marker": BottomCenterMarkerStrategy,
        "arc-extract-transform": ExtractTransformStrategy,
        "arc-interior-extract": InteriorExtractStrategy,
        "arc-ray-extension": RayExtensionStrategy,
        "arc-rectangle-marker-projection": RectangleMarkerProjectionStrategy,
        "arc-row-column-decomposition": RowColumnDecompositionStrategy,
        "arc-separator-cross-reference": SeparatorCrossReferenceStrategy,
        "arc-separator-propagation": SeparatorPropagationStrategy,
        "arc-scale-tile": ScaleTileStrategy,
        "arc-template-stamp": TemplateStampStrategy,
        "arc-zero-square-fill": ZeroSquareFillStrategy,
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
            "arc-ray-extension",
            "arc-row-column-decomposition",
            "arc-separator-propagation",
            "arc-separator-cross-reference",
            "arc-scale-tile",
            "arc-template-stamp",
            "arc-zero-square-fill",
            "arc-extract-transform",
            "arc-interior-extract",
            "arc-bottom-center-marker",
            "arc-bbox-recolor",
            "arc-bbox-ring-marker-projection",
            "arc-rectangle-marker-projection",
            "grammar-primitives",
            "arc-constant-output",
            "arc-color-map",
            "arc-absolute-patch",
        )
    if profile == "arc-theory":
        return (
            "library-replay",
            "arc-boolean-halves",
            "arc-ray-extension",
            "arc-row-column-decomposition",
            "arc-separator-propagation",
            "arc-separator-cross-reference",
            "arc-scale-tile",
            "arc-template-stamp",
            "arc-zero-square-fill",
            "arc-extract-transform",
            "arc-interior-extract",
            "arc-bottom-center-marker",
            "arc-bbox-recolor",
            "arc-bbox-ring-marker-projection",
            "arc-rectangle-marker-projection",
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
