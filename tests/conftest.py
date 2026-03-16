from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def arc_fixture_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "arc"

