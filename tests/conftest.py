"""Shared fixtures. Real-data tests read fitacf files placed in ``tests/data``."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
FITACF_GLOBS = ("*.fitacf", "*.fitacf.bz2", "*.fitacf.gz")


@pytest.fixture(scope="session")
def fitacf_files() -> list[Path]:
    """Sorted fitacf files under tests/data; skips the test when none exist."""
    files = sorted(p for g in FITACF_GLOBS for p in DATA_DIR.rglob(g))
    if not files:
        pytest.skip("no .fitacf files in tests/data")
    return files
