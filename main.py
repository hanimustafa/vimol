#!/usr/bin/env python3
"""main.app — open a molecule full-screen and drive it with the mouse.

Deliberately tiny: every bit of real logic lives in the embeddable library
(src/mviewer/). This launcher just resolves a file and hands off to the
full-screen viewer, which gives you mouse rotate (drag), pan (right/middle
drag), zoom (wheel) and hover-to-identify.

    ./main.py                     # opens the bundled C60 demo
    ./main.py path/to/mol.pdb
    ./main.py --cpu path/to/mol.pdb   # force the numpy CPU raycaster
    ./main.py --gpu path/to/mol.pdb   # force the OpenGL backend (needs mviewer[gl])

For every other flag (--style, --size, --render, ...) use the full CLI:
python -m mviewer --help
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import mviewer  # noqa: E402

_DEFAULT = os.path.join(os.path.dirname(__file__), "examples", "c60.xyz")

if __name__ == "__main__":
    argv = sys.argv[1:]
    backend = "auto"
    if "--cpu" in argv:
        argv.remove("--cpu")
        backend = "cpu"
    elif "--gpu" in argv:
        argv.remove("--gpu")
        backend = "gl"
    path = argv[0] if argv else _DEFAULT
    try:
        mviewer.view(path, backend=backend)
    except (RuntimeError, OSError, ValueError) as e:
        # e.g. run in a pipe / non-terminal; batch rendering lives in `mviewer`
        raise SystemExit(f"mviewer: {e}\nFor stills or flags use: python -m mviewer --help")
