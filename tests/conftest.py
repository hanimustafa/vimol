import os
import sys

import pytest

# conftest.py is imported by pytest before test_vimol.py's own path fixup
# runs, and an unrelated editable install elsewhere on this machine can
# otherwise shadow this checkout's vimol package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vimol import theme as _theme


@pytest.fixture(autouse=True)
def _isolate_theme_cache(tmp_path, monkeypatch):
    """Every Viewer construction touches theme.read_cached()/write_cached()
    (the frame-0 guess and _apply_probe_theme's write-through) -- point both
    at a throwaway, per-test path so no test run ever reads or writes the
    real ~/.vimol-theme on the machine running the suite."""
    monkeypatch.setattr(_theme, "_CACHE_PATH", str(tmp_path / "vimol-theme-test-cache"))
