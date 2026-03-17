from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from agi_core_codex.core.manifests import load_manifest
from agi_core_codex.domains.arc.analysis import (
    background_color,
    connected_components,
    fill_enclosed,
    find_uniform_col_separators,
    find_uniform_row_separators,
    intersect_separators,
)
from agi_core_codex.domains.arc.discovery import discover_arc_dataset, validate_dataset_dir
from agi_core_codex.domains.arc.loader import load_arc_tasks, load_split_ids, load_split_payload
from agi_core_codex.domains.arc.programs import make_arc_program
from agi_core_codex.domains.arc.scorer import ArcScorer
from agi_core_codex.domains.arc.types import ArcTask, Grid, freeze_grid, grid_shape


ArcExecutor = Callable[[Grid], Grid]


@dataclass(frozen=True)
class CandidateDefinition:
    name: str
    family: str
    complexity_cost: int
    implemented: bool
    executor: ArcExecutor


@dataclass(frozen=True)
class SplitCandidateMetrics:
    exact_solves: int
    exact_gains: int
    mean_accuracy: float
    unsolved_mean_accuracy_lift: float | None
    solved_task_ids: tuple[str, ...]
    exact_gain_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateScanResult:
    name: str
    family: str
    complexity_cost: int
    implemented: bool
    train_dev: SplitCandidateMetrics
    train_val: SplitCandidateMetrics


@dataclass(frozen=True)
class FailureTaskClassification:
    task_key: str
    best_strategy: str | None
    best_program_name: str | None
    train_accuracy: float
    test_accuracy: float | None
    shape_bucket: str
    separator_bucket: str
    object_bucket: str
    ray_span_bucket: str
    near_miss_bucket: str


@dataclass(frozen=True)
class FailureClassificationReport:
    manifest_path: str
    split: str
    unsolved_task_count: int
    shape_buckets: dict[str, int]
    separator_buckets: dict[str, int]
    object_buckets: dict[str, int]
    ray_span_buckets: dict[str, int]
    near_miss_buckets: dict[str, int]
    tasks: tuple[FailureTaskClassification, ...]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve_dataset_dir(dataset_dir: Path | None, split_file: Path) -> Path:
    if dataset_dir is not None:
        if not validate_dataset_dir(dataset_dir):
            raise ValueError(f"invalid dataset directory: {dataset_dir}")
        return dataset_dir

    split_payload = load_split_payload(split_file)
    source_dataset_dir = split_payload.get("source_dataset_dir")
    if isinstance(source_dataset_dir, str):
        source_path = Path(source_dataset_dir)
        if validate_dataset_dir(source_path):
            return source_path

    benchmark = split_payload.get("benchmark")
    if isinstance(benchmark, str):
        discovered = discover_arc_dataset(benchmark=benchmark, split="training")
        if discovered is not None:
            return discovered

    raise ValueError(
        f"could not resolve dataset directory for split {split_file}; pass --dataset-dir explicitly"
    )


def _non_background_counts(grid: Grid) -> Counter[int]:
    background = background_color(grid) if grid else 0
    return Counter(cell for row in grid for cell in row if cell != background)


def _dominant_non_background_color(grid: Grid) -> int:
    counts = _non_background_counts(grid)
    if not counts:
        return background_color(grid)
    return counts.most_common(1)[0][0]


def _rarest_non_background_color(grid: Grid) -> int:
    counts = _non_background_counts(grid)
    if not counts:
        return background_color(grid)
    minimum = min(counts.values())
    return min(color for color, count in counts.items() if count == minimum)


def _second_non_background_color(grid: Grid) -> int:
    counts = _non_background_counts(grid)
    if not counts:
        return background_color(grid)
    ranked = counts.most_common()
    return ranked[1][0] if len(ranked) > 1 else ranked[0][0]


def _keep_color(grid: Grid, color: int) -> Grid:
    return freeze_grid(tuple(cell if cell == color else 0 for cell in row) for row in grid)


def _erase_color(grid: Grid, color: int) -> Grid:
    return freeze_grid(tuple(0 if cell == color else cell for cell in row) for row in grid)


def _fill_background_with(grid: Grid, color: int) -> Grid:
    background = background_color(grid)
    return freeze_grid(tuple(color if cell == background else cell for cell in row) for row in grid)


def _trim_rows(grid: Grid) -> Grid:
    if not grid:
        return ()
    top = 0
    while top < len(grid) and all(cell == 0 for cell in grid[top]):
        top += 1
    bottom = len(grid) - 1
    while bottom >= top and all(cell == 0 for cell in grid[bottom]):
        bottom -= 1
    return freeze_grid(grid[top : bottom + 1]) if top <= bottom else ()


def _trim_cols(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    left = 0
    while left < width and all(grid[row_index][left] == 0 for row_index in range(height)):
        left += 1
    right = width - 1
    while right >= left and all(grid[row_index][right] == 0 for row_index in range(height)):
        right -= 1
    return freeze_grid(row[left : right + 1] for row in grid) if left <= right else ()


def _crop_half_top(grid: Grid) -> Grid:
    return freeze_grid(grid[: len(grid) // 2])


def _crop_half_bottom(grid: Grid) -> Grid:
    return freeze_grid(grid[len(grid) // 2 :])


def _crop_half_left(grid: Grid) -> Grid:
    midpoint = len(grid[0]) // 2 if grid else 0
    return freeze_grid(row[:midpoint] for row in grid)


def _crop_half_right(grid: Grid) -> Grid:
    midpoint = len(grid[0]) // 2 if grid else 0
    return freeze_grid(row[midpoint:] for row in grid)


def _binarize(grid: Grid) -> Grid:
    return freeze_grid(tuple(0 if cell == 0 else 1 for cell in row) for row in grid)


def _invert_colors(grid: Grid) -> Grid:
    return freeze_grid(tuple(9 - cell for cell in row) for row in grid)


def _gravity_left(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    rows = [[0 for _ in range(width)] for _ in range(height)]
    for row_index, row in enumerate(grid):
        nonzero = [cell for cell in row if cell != 0]
        for offset, value in enumerate(nonzero):
            rows[row_index][offset] = value
    return freeze_grid(rows)


def _gravity_right(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    rows = [[0 for _ in range(width)] for _ in range(height)]
    for row_index, row in enumerate(grid):
        nonzero = [cell for cell in row if cell != 0]
        for offset, value in enumerate(nonzero):
            rows[row_index][width - len(nonzero) + offset] = value
    return freeze_grid(rows)


def _dilate(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] != 0:
                continue
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row_index + delta_row
                next_col = col_index + delta_col
                if 0 <= next_row < height and 0 <= next_col < width and grid[next_row][next_col] != 0:
                    rows[row_index][col_index] = grid[next_row][next_col]
                    break
    return freeze_grid(rows)


def _erode(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    rows = [list(row) for row in grid]
    for row_index in range(height):
        for col_index in range(width):
            if grid[row_index][col_index] == 0:
                continue
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row_index + delta_row
                next_col = col_index + delta_col
                if not (0 <= next_row < height and 0 <= next_col < width):
                    rows[row_index][col_index] = 0
                    break
                if grid[next_row][next_col] == 0:
                    rows[row_index][col_index] = 0
                    break
    return freeze_grid(rows)


def _rotate_180(grid: Grid) -> Grid:
    return freeze_grid(row[::-1] for row in grid[::-1])


def _flip_horizontal(grid: Grid) -> Grid:
    return freeze_grid(row[::-1] for row in grid)


def _flip_vertical(grid: Grid) -> Grid:
    return freeze_grid(grid[::-1])


def _transpose(grid: Grid) -> Grid:
    return freeze_grid(zip(*grid)) if grid else ()


def _object_local_transform(grid: Grid, transform: ArcExecutor, *, largest_only: bool) -> Grid:
    if not grid:
        return ()
    components = connected_components(grid)
    background = background_color(grid)
    if not components:
        return grid

    target_pixels: set[tuple[int, int]] = set()
    if largest_only:
        ordered = sorted(
            components,
            key=lambda component: (-component.size, component.bbox, component.color),
        )
        target_pixels = set(ordered[0].pixels)

    rows = [[background for _ in range(len(grid[0]))] for _ in range(len(grid))]
    for component in components:
        row_start, col_start, row_end, col_end = component.bbox
        box = [
            [background for _ in range(col_end - col_start + 1)]
            for _ in range(row_end - row_start + 1)
        ]
        for pixel_row, pixel_col in component.pixels:
            box[pixel_row - row_start][pixel_col - col_start] = component.color
        box_grid = freeze_grid(box)
        if not largest_only or component.pixels[0] in target_pixels:
            transformed = transform(box_grid)
            if grid_shape(transformed) != grid_shape(box_grid):
                raise ValueError("same-bbox object transform changed component shape")
        else:
            transformed = box_grid
        for local_row, row in enumerate(transformed):
            for local_col, value in enumerate(row):
                if value != background:
                    rows[row_start + local_row][col_start + local_col] = value
    return freeze_grid(rows)


def _sort_rows_by_nonzero(grid: Grid) -> Grid:
    return freeze_grid(sorted(grid, key=lambda row: sum(1 for cell in row if cell != 0)))


def _sort_cols_by_nonzero(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    columns = [[grid[row_index][col_index] for row_index in range(height)] for col_index in range(width)]
    columns.sort(key=lambda column: sum(1 for value in column if value != 0))
    return freeze_grid(
        tuple(columns[col_index][row_index] for col_index in range(width))
        for row_index in range(height)
    )


def _extract_largest_cc(grid: Grid) -> Grid:
    components = connected_components(grid)
    if not components:
        return grid
    component = max(components, key=lambda item: item.size)
    row_start, col_start, row_end, col_end = component.bbox
    return freeze_grid(
        row[col_start : col_end + 1]
        for row in grid[row_start : row_end + 1]
    )


def _extract_unique_color_region(grid: Grid) -> Grid:
    rarest = _rarest_non_background_color(grid)
    positions = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell == rarest
    ]
    if not positions:
        return grid
    row_indices = [row_index for row_index, _ in positions]
    col_indices = [col_index for _, col_index in positions]
    row_start, row_end = min(row_indices), max(row_indices)
    col_start, col_end = min(col_indices), max(col_indices)
    return freeze_grid(
        row[col_start : col_end + 1]
        for row in grid[row_start : row_end + 1]
    )


def _mirror_tile_horizontal(grid: Grid) -> Grid:
    return freeze_grid(tuple(row) + tuple(reversed(row)) for row in grid)


def _mirror_tile_vertical(grid: Grid) -> Grid:
    return freeze_grid(list(grid) + list(reversed(grid)))


def _mirror_tile_both(grid: Grid) -> Grid:
    top = [tuple(row) + tuple(reversed(row)) for row in grid]
    bottom = [tuple(row) + tuple(reversed(row)) for row in reversed(grid)]
    return freeze_grid(top + bottom)


def _tile_horizontal(grid: Grid) -> Grid:
    return freeze_grid(tuple(row) + tuple(row) for row in grid)


def _rotate_tile_clockwise(grid: Grid) -> Grid:
    height, width = grid_shape(grid)
    if height == 0 or width == 0 or height != width:
        return grid
    rotate_90 = freeze_grid(tuple(grid[height - 1 - col_index][row_index] for col_index in range(height)) for row_index in range(width))
    rotate_270 = freeze_grid(tuple(grid[col_index][width - 1 - row_index] for col_index in range(height)) for row_index in range(width))
    top = [tuple(grid[row_index]) + tuple(rotate_90[row_index]) for row_index in range(height)]
    bottom = [tuple(rotate_270[row_index]) + tuple(_rotate_180(grid)[row_index]) for row_index in range(height)]
    return freeze_grid(top + bottom)


def _inpaint_by_symmetry(grid: Grid) -> Grid:
    if not grid:
        return ()
    colors = {cell for row in grid for cell in row if cell != 0}
    if not colors:
        return grid

    height, width = grid_shape(grid)
    best_result: Grid | None = None
    best_score = (height * width, height * width)

    for mask_color in colors:
        rows = [list(row) for row in grid]
        for _ in range(4):
            changed = False
            for row_index in range(height):
                for col_index in range(width):
                    if rows[row_index][col_index] != mask_color:
                        continue
                    for mirror_row, mirror_col in (
                        (row_index, width - 1 - col_index),
                        (height - 1 - row_index, col_index),
                        (height - 1 - row_index, width - 1 - col_index),
                    ):
                        if rows[mirror_row][mirror_col] != mask_color:
                            rows[row_index][col_index] = rows[mirror_row][mirror_col]
                            changed = True
                            break
                    else:
                        if height == width and rows[col_index][row_index] != mask_color:
                            rows[row_index][col_index] = rows[col_index][row_index]
                            changed = True
            if not changed:
                break

        remaining = sum(
            1
            for row in rows
            for cell in row
            if cell == mask_color
        )
        non_mask_changes = sum(
            1
            for row_index, row in enumerate(rows)
            for col_index, cell in enumerate(row)
            if grid[row_index][col_index] != mask_color and cell != grid[row_index][col_index]
        )
        score = (remaining, non_mask_changes)
        if score < best_score:
            best_score = score
            best_result = freeze_grid(rows)

    return best_result if best_result is not None else grid


def _inpaint_periodic(grid: Grid) -> Grid:
    if not grid:
        return ()
    height, width = grid_shape(grid)
    if not any(cell == 0 for row in grid for cell in row):
        return grid

    for period_height in range(1, height + 1):
        for period_width in range(1, width + 1):
            tile: list[list[int | None]] = [
                [None for _ in range(period_width)]
                for _ in range(period_height)
            ]
            consistent = True
            for row_index in range(height):
                if not consistent:
                    break
                for col_index in range(width):
                    value = grid[row_index][col_index]
                    if value == 0:
                        continue
                    tile_row = row_index % period_height
                    tile_col = col_index % period_width
                    seen = tile[tile_row][tile_col]
                    if seen is None:
                        tile[tile_row][tile_col] = value
                    elif seen != value:
                        consistent = False
                        break
            if not consistent:
                continue
            if any(
                tile[tile_row][tile_col] is None
                for tile_row in range(period_height)
                for tile_col in range(period_width)
            ):
                continue

            rows = [list(row) for row in grid]
            for row_index in range(height):
                for col_index in range(width):
                    if rows[row_index][col_index] == 0:
                        rows[row_index][col_index] = tile[row_index % period_height][col_index % period_width] or 0
            return freeze_grid(rows)

    return grid


def candidate_catalog() -> tuple[CandidateDefinition, ...]:
    return (
        CandidateDefinition("trim-rows", "old_unary", 2, False, _trim_rows),
        CandidateDefinition("trim-cols", "old_unary", 2, False, _trim_cols),
        CandidateDefinition("crop-half-top", "old_unary", 2, False, _crop_half_top),
        CandidateDefinition("crop-half-bottom", "old_unary", 2, False, _crop_half_bottom),
        CandidateDefinition("crop-half-left", "old_unary", 2, False, _crop_half_left),
        CandidateDefinition("crop-half-right", "old_unary", 2, False, _crop_half_right),
        CandidateDefinition("binarize", "old_unary", 2, False, _binarize),
        CandidateDefinition("invert-colors", "old_unary", 2, False, _invert_colors),
        CandidateDefinition("gravity-left", "old_unary", 2, False, _gravity_left),
        CandidateDefinition("gravity-right", "old_unary", 2, False, _gravity_right),
        CandidateDefinition("fill-enclosed", "old_unary", 2, False, fill_enclosed),
        CandidateDefinition("dilate", "old_unary", 2, False, _dilate),
        CandidateDefinition("erode", "old_unary", 2, False, _erode),
        CandidateDefinition(
            "sort-rows-by-nonzero",
            "old_unary",
            2,
            False,
            _sort_rows_by_nonzero,
        ),
        CandidateDefinition(
            "sort-cols-by-nonzero",
            "old_unary",
            2,
            False,
            _sort_cols_by_nonzero,
        ),
        CandidateDefinition(
            "extract-largest-cc",
            "old_unary",
            3,
            True,
            _extract_largest_cc,
        ),
        CandidateDefinition(
            "extract-unique-color-region",
            "old_unary",
            3,
            True,
            _extract_unique_color_region,
        ),
        CandidateDefinition(
            "mirror-tile-horizontal",
            "old_unary",
            3,
            True,
            _mirror_tile_horizontal,
        ),
        CandidateDefinition(
            "mirror-tile-vertical",
            "old_unary",
            3,
            True,
            _mirror_tile_vertical,
        ),
        CandidateDefinition(
            "mirror-tile-both",
            "old_unary",
            3,
            True,
            _mirror_tile_both,
        ),
        CandidateDefinition(
            "tile-horizontal",
            "old_unary",
            3,
            True,
            _tile_horizontal,
        ),
        CandidateDefinition(
            "rotate-tile-clockwise",
            "old_unary",
            4,
            True,
            _rotate_tile_clockwise,
        ),
        CandidateDefinition(
            "inpaint-by-symmetry",
            "old_unary",
            4,
            True,
            _inpaint_by_symmetry,
        ),
        CandidateDefinition(
            "inpaint-periodic",
            "old_unary",
            4,
            True,
            _inpaint_periodic,
        ),
        CandidateDefinition(
            "keep-dominant-color",
            "color_macro",
            3,
            False,
            lambda grid: _keep_color(grid, _dominant_non_background_color(grid)),
        ),
        CandidateDefinition(
            "keep-rarest-color",
            "color_macro",
            3,
            False,
            lambda grid: _keep_color(grid, _rarest_non_background_color(grid)),
        ),
        CandidateDefinition(
            "keep-second-color",
            "color_macro",
            3,
            False,
            lambda grid: _keep_color(grid, _second_non_background_color(grid)),
        ),
        CandidateDefinition(
            "erase-dominant-color",
            "color_macro",
            3,
            False,
            lambda grid: _erase_color(grid, _dominant_non_background_color(grid)),
        ),
        CandidateDefinition(
            "erase-rarest-color",
            "color_macro",
            3,
            False,
            lambda grid: _erase_color(grid, _rarest_non_background_color(grid)),
        ),
        CandidateDefinition(
            "erase-second-color",
            "color_macro",
            3,
            False,
            lambda grid: _erase_color(grid, _second_non_background_color(grid)),
        ),
        CandidateDefinition(
            "fill-bg-with-dominant-color",
            "color_macro",
            3,
            False,
            lambda grid: _fill_background_with(grid, _dominant_non_background_color(grid)),
        ),
        CandidateDefinition(
            "fill-bg-with-rarest-color",
            "color_macro",
            3,
            False,
            lambda grid: _fill_background_with(grid, _rarest_non_background_color(grid)),
        ),
        CandidateDefinition(
            "fill-bg-with-second-color",
            "color_macro",
            3,
            False,
            lambda grid: _fill_background_with(grid, _second_non_background_color(grid)),
        ),
        CandidateDefinition(
            "object-all-rotate-180",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _rotate_180, largest_only=False),
        ),
        CandidateDefinition(
            "object-all-flip-horizontal",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _flip_horizontal, largest_only=False),
        ),
        CandidateDefinition(
            "object-all-flip-vertical",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _flip_vertical, largest_only=False),
        ),
        CandidateDefinition(
            "object-all-transpose",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _transpose, largest_only=False),
        ),
        CandidateDefinition(
            "object-largest-rotate-180",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _rotate_180, largest_only=True),
        ),
        CandidateDefinition(
            "object-largest-flip-horizontal",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _flip_horizontal, largest_only=True),
        ),
        CandidateDefinition(
            "object-largest-flip-vertical",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _flip_vertical, largest_only=True),
        ),
        CandidateDefinition(
            "object-largest-transpose",
            "object_local",
            4,
            False,
            lambda grid: _object_local_transform(grid, _transpose, largest_only=True),
        ),
    )


def _score_candidate_on_tasks(
    candidate: CandidateDefinition,
    tasks: tuple[ArcTask, ...],
    *,
    reference_manifest: dict[str, Any] | None,
) -> SplitCandidateMetrics:
    scorer = ArcScorer()
    program = make_arc_program(
        name=candidate.name,
        semantics={
            "type": "analysis_candidate",
            "candidate": candidate.name,
            "family": candidate.family,
        },
        executor=candidate.executor,
        complexity=candidate.complexity_cost,
    )

    solved_task_ids: list[str] = []
    exact_gain_task_ids: list[str] = []
    accuracies: list[float] = []
    unsolved_deltas: list[float] = []
    reference_tasks = {
        task["task_key"]: task
        for task in reference_manifest.get("tasks", [])
    } if reference_manifest is not None else {}

    for task in tasks:
        score = scorer.evaluate_program(task, program)
        accuracy = score.test_accuracy if score.test_accuracy is not None else score.train_accuracy
        accuracies.append(accuracy)
        exact = score.test_exact if score.test_exact is not None else score.train_exact
        if exact:
            solved_task_ids.append(task.task_id)

        if reference_manifest is not None:
            baseline = reference_tasks.get(task.task_id)
            if baseline is None:
                continue
            baseline_exact = baseline["solved_test"] if baseline["solved_test"] is not None else baseline["solved_train"]
            if exact and not baseline_exact:
                exact_gain_task_ids.append(task.task_id)
            if baseline_exact:
                continue
            baseline_accuracy = (
                baseline["test_accuracy"]
                if baseline["test_accuracy"] is not None
                else baseline["train_accuracy"]
            )
            unsolved_deltas.append(accuracy - float(baseline_accuracy))

    return SplitCandidateMetrics(
        exact_solves=len(solved_task_ids),
        exact_gains=len(exact_gain_task_ids) if reference_manifest is not None else len(solved_task_ids),
        mean_accuracy=sum(accuracies) / len(accuracies) if accuracies else 0.0,
        unsolved_mean_accuracy_lift=(
            sum(unsolved_deltas) / len(unsolved_deltas)
            if unsolved_deltas
            else None
        ),
        solved_task_ids=tuple(sorted(solved_task_ids)),
        exact_gain_task_ids=(
            tuple(sorted(exact_gain_task_ids))
            if reference_manifest is not None
            else tuple(sorted(solved_task_ids))
        ),
    )


def scan_primitive_candidates(
    *,
    dataset_dir: Path | None,
    train_dev_split_file: Path,
    train_val_split_file: Path,
    include_candidates: tuple[str, ...] = (),
    exclude_candidates: tuple[str, ...] = (),
    reference_train_dev_manifest: Path | None = None,
    reference_train_val_manifest: Path | None = None,
    only_unimplemented: bool = False,
) -> tuple[CandidateScanResult, ...]:
    catalog = {candidate.name: candidate for candidate in candidate_catalog()}
    if include_candidates:
        selected = {name: catalog[name] for name in include_candidates}
    else:
        selected = dict(catalog)
    for name in exclude_candidates:
        selected.pop(name, None)
    if only_unimplemented:
        selected = {
            name: candidate
            for name, candidate in selected.items()
            if not candidate.implemented
        }

    train_dev_dataset_dir = _resolve_dataset_dir(dataset_dir, train_dev_split_file)
    train_val_dataset_dir = _resolve_dataset_dir(dataset_dir, train_val_split_file)
    train_dev_tasks = load_arc_tasks(
        train_dev_dataset_dir,
        split_ids=load_split_ids(train_dev_split_file),
    )
    train_val_tasks = load_arc_tasks(
        train_val_dataset_dir,
        split_ids=load_split_ids(train_val_split_file),
    )

    reference_train_dev = (
        load_manifest(reference_train_dev_manifest)
        if reference_train_dev_manifest is not None
        else None
    )
    reference_train_val = (
        load_manifest(reference_train_val_manifest)
        if reference_train_val_manifest is not None
        else None
    )

    results = []
    for candidate in selected.values():
        results.append(
            CandidateScanResult(
                name=candidate.name,
                family=candidate.family,
                complexity_cost=candidate.complexity_cost,
                implemented=candidate.implemented,
                train_dev=_score_candidate_on_tasks(
                    candidate,
                    train_dev_tasks,
                    reference_manifest=reference_train_dev,
                ),
                train_val=_score_candidate_on_tasks(
                    candidate,
                    train_val_tasks,
                    reference_manifest=reference_train_val,
                ),
            )
        )

    def _sort_key(result: CandidateScanResult) -> tuple[float, ...] | tuple[float, str]:
        holdout_lift = result.train_val.unsolved_mean_accuracy_lift
        holdout_signal = holdout_lift if holdout_lift is not None else result.train_val.mean_accuracy
        return (
            -result.train_val.exact_gains,
            -result.train_dev.exact_gains,
            -holdout_signal,
            result.complexity_cost,
            result.name,
        )

    return tuple(sorted(results, key=_sort_key))


def format_scan_results(
    results: tuple[CandidateScanResult, ...],
    *,
    output_format: str,
    top: int | None = None,
) -> str:
    visible = results if top is None else results[:top]
    if output_format == "json":
        return json.dumps(_jsonable(visible), indent=2, sort_keys=True)

    lines = []
    for index, result in enumerate(visible, start=1):
        holdout_signal = (
            result.train_val.unsolved_mean_accuracy_lift
            if result.train_val.unsolved_mean_accuracy_lift is not None
            else result.train_val.mean_accuracy
        )
        lines.append(
            f"{index}. {result.name} family={result.family} complexity={result.complexity_cost} "
            f"train_val_exact={result.train_val.exact_solves} train_val_gain={result.train_val.exact_gains} "
            f"train_dev_exact={result.train_dev.exact_solves} train_dev_gain={result.train_dev.exact_gains} "
            f"train_val_signal={holdout_signal:.4f}"
        )
        lines.append(
            f"   train_val_tasks={','.join(result.train_val.solved_task_ids) or '-'} "
            f"train_val_gain_tasks={','.join(result.train_val.exact_gain_task_ids) or '-'}"
        )
        lines.append(
            f"   train_dev_tasks={','.join(result.train_dev.solved_task_ids) or '-'} "
            f"train_dev_gain_tasks={','.join(result.train_dev.exact_gain_task_ids) or '-'}"
        )
    return "\n".join(lines)


def _load_tasks_from_manifest(
    manifest_payload: dict[str, Any],
    dataset_dir: Path | None,
) -> tuple[ArcTask, ...]:
    if dataset_dir is None:
        for note in manifest_payload.get("notes", []):
            if isinstance(note, str) and note.startswith("dataset_dir="):
                candidate = Path(note.split("=", 1)[1])
                if validate_dataset_dir(candidate):
                    dataset_dir = candidate
                    break
    if dataset_dir is None:
        split_path = manifest_payload.get("split_path")
        if isinstance(split_path, str):
            dataset_dir = _resolve_dataset_dir(None, Path(split_path))
    if dataset_dir is None:
        raise ValueError("could not resolve dataset directory for manifest")

    task_ids = tuple(task["task_key"] for task in manifest_payload.get("tasks", []))
    return load_arc_tasks(dataset_dir, split_ids=task_ids)


def _separator_bucket(task: ArcTask) -> str:
    row_separators = intersect_separators(
        find_uniform_row_separators(example.input)
        for example in task.train
    )
    col_separators = intersect_separators(
        find_uniform_col_separators(example.input)
        for example in task.train
    )
    if row_separators and col_separators:
        return "grid"
    if row_separators:
        return "rows_only"
    if col_separators:
        return "cols_only"
    return "none"


def _object_bucket(task: ArcTask) -> str:
    component_counts = [len(connected_components(example.input)) for example in task.train]
    if max(component_counts, default=0) == 0:
        return "none"
    if max(component_counts) == 1 and min(component_counts) == 1:
        return "single_object"

    same_shape = all(grid_shape(example.input) == grid_shape(example.output) for example in task.train)
    if same_shape:
        bbox_preserving = True
        for example in task.train:
            input_components = connected_components(example.input)
            output_components = connected_components(example.output)
            if len(input_components) != len(output_components):
                bbox_preserving = False
                break
            input_bboxes = sorted(component.bbox for component in input_components)
            output_bboxes = sorted(component.bbox for component in output_components)
            if input_bboxes != output_bboxes:
                bbox_preserving = False
                break
        if bbox_preserving and max(component_counts) > 1:
            return "bbox_preserving_objects"

    for example in task.train:
        input_components = connected_components(example.input)
        non_background = sum(component.size for component in input_components)
        if len(input_components) > 1 and non_background > 0:
            largest = max(component.size for component in input_components)
            if largest / non_background >= 0.6:
                return "largest_object_salient"
    return "multi_object"


def _detect_row_span(example_input: Grid, example_output: Grid) -> bool:
    if grid_shape(example_input) != grid_shape(example_output):
        return False
    height, width = grid_shape(example_input)
    for row_index in range(height):
        for color in {cell for cell in example_input[row_index] if cell != 0}:
            seed_cols = [col_index for col_index, cell in enumerate(example_input[row_index]) if cell == color]
            if len(seed_cols) < 2:
                continue
            left = min(seed_cols)
            right = max(seed_cols)
            if right - left <= 1:
                continue
            if any(example_input[row_index][col_index] == 0 and example_output[row_index][col_index] == color for col_index in range(left + 1, right)):
                return True
    return False


def _detect_col_span(example_input: Grid, example_output: Grid) -> bool:
    if grid_shape(example_input) != grid_shape(example_output):
        return False
    height, width = grid_shape(example_input)
    for col_index in range(width):
        column_colors = {example_input[row_index][col_index] for row_index in range(height) if example_input[row_index][col_index] != 0}
        for color in column_colors:
            seed_rows = [row_index for row_index in range(height) if example_input[row_index][col_index] == color]
            if len(seed_rows) < 2:
                continue
            top = min(seed_rows)
            bottom = max(seed_rows)
            if bottom - top <= 1:
                continue
            if any(example_input[row_index][col_index] == 0 and example_output[row_index][col_index] == color for row_index in range(top + 1, bottom)):
                return True
    return False


def _detect_ray_growth(example_input: Grid, example_output: Grid) -> bool:
    if grid_shape(example_input) != grid_shape(example_output):
        return False
    height, width = grid_shape(example_input)
    for row_index in range(height):
        for col_index in range(width):
            output_color = example_output[row_index][col_index]
            if example_input[row_index][col_index] != 0 or output_color == 0:
                continue
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                current_row = row_index - delta_row
                current_col = col_index - delta_col
                seen_gap = False
                while 0 <= current_row < height and 0 <= current_col < width:
                    current_value = example_input[current_row][current_col]
                    if current_value == output_color:
                        if seen_gap:
                            return True
                        break
                    if current_value != 0:
                        break
                    seen_gap = True
                    current_row -= delta_row
                    current_col -= delta_col
    return False


def _ray_span_bucket(task: ArcTask) -> str:
    row_span = any(_detect_row_span(example.input, example.output) for example in task.train)
    col_span = any(_detect_col_span(example.input, example.output) for example in task.train)
    if row_span and col_span:
        return "mixed_span"
    if row_span:
        return "row_span"
    if col_span:
        return "col_span"
    if any(_detect_ray_growth(example.input, example.output) for example in task.train):
        return "ray_like"
    return "none"


def _near_miss_bucket(train_accuracy: float) -> str:
    if train_accuracy >= 0.95:
        return "0.95-0.99"
    if train_accuracy >= 0.80:
        return "0.80-0.94"
    if train_accuracy >= 0.50:
        return "0.50-0.79"
    return "<0.50"


def classify_failures(
    *,
    manifest_path: Path,
    dataset_dir: Path | None = None,
) -> FailureClassificationReport:
    manifest_payload = load_manifest(manifest_path)
    tasks = _load_tasks_from_manifest(manifest_payload, dataset_dir)
    task_map = {task.task_id: task for task in tasks}

    records: list[FailureTaskClassification] = []
    for task_record in manifest_payload.get("tasks", []):
        solved = task_record["solved_test"] if task_record["solved_test"] is not None else task_record["solved_train"]
        if solved:
            continue
        task = task_map[task_record["task_key"]]
        shape_bucket = (
            "shape_change"
            if any(grid_shape(example.input) != grid_shape(example.output) for example in task.train)
            else "same_shape"
        )
        records.append(
            FailureTaskClassification(
                task_key=task.task_id,
                best_strategy=task_record["best_strategy"],
                best_program_name=task_record["best_program_name"],
                train_accuracy=float(task_record["train_accuracy"]),
                test_accuracy=(
                    float(task_record["test_accuracy"])
                    if task_record["test_accuracy"] is not None
                    else None
                ),
                shape_bucket=shape_bucket,
                separator_bucket=_separator_bucket(task),
                object_bucket=_object_bucket(task),
                ray_span_bucket=_ray_span_bucket(task),
                near_miss_bucket=_near_miss_bucket(float(task_record["train_accuracy"])),
            )
        )

    shape_buckets = Counter(record.shape_bucket for record in records)
    separator_buckets = Counter(record.separator_bucket for record in records)
    object_buckets = Counter(record.object_bucket for record in records)
    ray_span_buckets = Counter(record.ray_span_bucket for record in records)
    near_miss_buckets = Counter(record.near_miss_bucket for record in records)

    return FailureClassificationReport(
        manifest_path=str(manifest_path),
        split=str(manifest_payload.get("split", "")),
        unsolved_task_count=len(records),
        shape_buckets=dict(sorted(shape_buckets.items())),
        separator_buckets=dict(sorted(separator_buckets.items())),
        object_buckets=dict(sorted(object_buckets.items())),
        ray_span_buckets=dict(sorted(ray_span_buckets.items())),
        near_miss_buckets=dict(sorted(near_miss_buckets.items())),
        tasks=tuple(sorted(records, key=lambda record: record.task_key)),
    )


def format_failure_report(report: FailureClassificationReport, *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(_jsonable(report), indent=2, sort_keys=True)

    lines = [
        f"unsolved_tasks={report.unsolved_task_count} split={report.split}",
        f"shape_buckets={report.shape_buckets}",
        f"separator_buckets={report.separator_buckets}",
        f"object_buckets={report.object_buckets}",
        f"ray_span_buckets={report.ray_span_buckets}",
        f"near_miss_buckets={report.near_miss_buckets}",
    ]
    for task in report.tasks:
        lines.append(
            f"{task.task_key} shape={task.shape_bucket} separators={task.separator_bucket} "
            f"objects={task.object_bucket} ray_span={task.ray_span_bucket} "
            f"near_miss={task.near_miss_bucket} best={task.best_strategy}:{task.best_program_name}"
        )
    return "\n".join(lines)
