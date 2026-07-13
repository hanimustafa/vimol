#!/usr/bin/env python3
"""main.app — open a molecule full-screen and drive it with the mouse.

Deliberately tiny: every bit of real logic lives in the embeddable library
(src/mviewer/). This launcher just resolves a file and hands off to the
full-screen viewer, which gives you mouse rotate (drag), pan (right/middle
drag), zoom (wheel) and hover-to-identify.

    ./main.py                     # opens the bundled C60 demo
    ./main.py path/to/mol.pdb
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import mviewer  # noqa: E402

_DEFAULT = os.path.join(os.path.dirname(__file__), "examples", "c60.xyz")

if __name__ == "__main__":
    try:
        mviewer.view(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT)
    except (RuntimeError, OSError, ValueError) as e:
        # e.g. run in a pipe / non-terminal; batch rendering lives in `mviewer`
        raise SystemExit(f"mviewer: {e}\nFor stills or flags use: python -m mviewer --help")
