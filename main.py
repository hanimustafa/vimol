#!/usr/bin/env python3
"""main.app — the driver application for the mviewer library.

This is a thin launcher so the viewer runs straight from a checkout without
installation:

    ./main.py examples/c60.xyz
    ./main.py examples/benzene.xyz --style spacefill --spin
    ./main.py examples/c60.xyz --render out.png

For an installed copy, use the `mviewer` console command instead (see
pyproject.toml). The real logic lives in the embeddable library at
src/mviewer/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mviewer.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
