"""Shared fixtures.

`src` goes on the path so the suite runs against a checkout without an install
step; CI additionally installs the package and runs the same tests against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXAMPLES = ROOT / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def demo_suite_path() -> Path:
    return EXAMPLES / "suites" / "demo.yaml"


@pytest.fixture
def tmp_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


@pytest.fixture
def tmp_cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite"


def write_task(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path
