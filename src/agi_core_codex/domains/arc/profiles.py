from __future__ import annotations

from collections.abc import Sequence

from agi_core_codex.core.strategies import GrammarPrimitiveStrategy, LibraryReplayStrategy
from agi_core_codex.domains.arc.strategies import (
    AbsolutePatchStrategy,
    BboxRecolorStrategy,
    BboxRingMarkerProjectionStrategy,
    BottomCenterMarkerStrategy,
    BooleanHalvesStrategy,
    CollinearGapBridgeStrategy,
    ColorMapStrategy,
    ConstantOutputStrategy,
    ExtractTransformStrategy,
    HoleProjectionStrategy,
    InteriorExtractStrategy,
    MotifCompletionStrategy,
    RayExtensionStrategy,
    RectangularRingRecolorStrategy,
    RectangleMarkerProjectionStrategy,
    RowColumnDecompositionStrategy,
    ScaleTileStrategy,
    SeparatorPropagationStrategy,
    SeparatorCrossReferenceStrategy,
    SolidRectangleExtractStrategy,
    TemplateStampStrategy,
    TriominoCornerFillStrategy,
    ZeroPatternPropagationStrategy,
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
        "arc-collinear-gap-bridge": CollinearGapBridgeStrategy,
        "arc-extract-transform": ExtractTransformStrategy,
        "arc-hole-projection": HoleProjectionStrategy,
        "arc-interior-extract": InteriorExtractStrategy,
        "arc-motif-completion": MotifCompletionStrategy,
        "arc-ray-extension": RayExtensionStrategy,
        "arc-rectangular-ring-recolor": RectangularRingRecolorStrategy,
        "arc-rectangle-marker-projection": RectangleMarkerProjectionStrategy,
        "arc-row-column-decomposition": RowColumnDecompositionStrategy,
        "arc-separator-cross-reference": SeparatorCrossReferenceStrategy,
        "arc-separator-propagation": SeparatorPropagationStrategy,
        "arc-scale-tile": ScaleTileStrategy,
        "arc-template-stamp": TemplateStampStrategy,
        "arc-solid-rectangle-extract": SolidRectangleExtractStrategy,
        "arc-triomino-corner-fill": TriominoCornerFillStrategy,
        "arc-zero-pattern-propagation": ZeroPatternPropagationStrategy,
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
            "arc-zero-pattern-propagation",
            "arc-zero-square-fill",
            "arc-extract-transform",
            "arc-hole-projection",
            "arc-triomino-corner-fill",
            "arc-collinear-gap-bridge",
            "arc-solid-rectangle-extract",
            "arc-rectangular-ring-recolor",
            "arc-interior-extract",
            "arc-motif-completion",
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
            "arc-zero-pattern-propagation",
            "arc-zero-square-fill",
            "arc-extract-transform",
            "arc-hole-projection",
            "arc-triomino-corner-fill",
            "arc-collinear-gap-bridge",
            "arc-solid-rectangle-extract",
            "arc-rectangular-ring-recolor",
            "arc-interior-extract",
            "arc-motif-completion",
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
