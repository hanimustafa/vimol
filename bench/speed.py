"""Head-to-head frame timing: pure-numpy raycaster vs the numba kernel.

Both paths are bit-identical (tests/test_render_fast.py); this script only
measures speed. Needs the [fast] extra for the numba column.

    python bench/speed.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vimol import _render_fast as _fast
from vimol import parsers
from vimol.bonds import ensure_bonds
from vimol.camera import Camera
from vimol.render import Renderer, Style

EX = os.path.join(os.path.dirname(__file__), "..", "examples")

mol = parsers.load(os.path.join(EX, "c60.xyz"))
ensure_bonds(mol)
print(f"molecule: {mol.n_atoms} atoms, {len(mol.bonds)} bonds")

if _fast.available():
    t0 = time.perf_counter()
    _fast._warm()                      # synchronous JIT compile
    print(f"numba compile: {time.perf_counter() - t0:.1f} s, ready={_fast.ready()}")
else:
    print("numba not installed -- numpy column only (pip install vimol[fast])")
HAVE_FAST = _fast.ready()
_real_ready = _fast.ready


def bench(W, H, ss, transparent, fast, n=8):
    cam = Camera(center=mol.centroid(), extent=mol.radius_of_gyration_extent())
    cam.fit(W * ss, H * ss, mol.radius_of_gyration_extent() + 0.5)
    st = Style(transparent=transparent)
    _fast.ready = _real_ready if fast else (lambda: False)
    try:
        r = Renderer(W * ss, H * ss)
        r.render(mol, cam, st)         # warmup
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            r.render(mol, cam, st)
            ts.append(time.perf_counter() - t0)
    finally:
        _fast.ready = _real_ready
    return np.median(ts) * 1000


variants = [("numpy", False)] + ([("numba", True)] if HAVE_FAST else [])
for W, H, ss, label in [(2560, 1440, 1, "interactive 2560x1440 ss1"),
                        (2560, 1440, 2, "settle      2560x1440 ss2"),
                        (1280, 800, 1, "small       1280x800  ss1")]:
    for transparent in (False, True):
        tag = "rgba" if transparent else "rgb "
        line = f"{label} {tag}:"
        base = None
        for name, fast in variants:
            med = bench(W, H, ss, transparent, fast)
            base = base or med
            line += f"  {name} {med:7.1f} ms ({base / med:5.2f}x)"
        print(line)
