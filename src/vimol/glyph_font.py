"""A stroked capital letter for each one-character amino acid code.

Each glyph is a handful of polylines in a unit box -- x and y both run 0..1,
y downward from the cap line -- and the rasterizer inks every pixel within
:data:`STROKE` of a stroke. That keeps the letters smooth and evenly weighted
at any size, where a bitmap font of a sensible size would read as pixel art
beside the shaded solids it sits on.

Hand-drawn rather than taken from a font file: vimol ships a stdlib PNG
encoder and no Pillow (see the README), and twenty capitals is a smaller thing
to carry than a font dependency. ``X`` is the fallback for a residue whose name
has no one-letter code.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Cap width as a fraction of cap height, and stroke half-width in cap heights.
ASPECT = 0.72
STROKE = 0.085

Polyline = List[Tuple[float, float]]

_O: Polyline = [(0.50, 0.02), (0.86, 0.20), (0.98, 0.50), (0.86, 0.80),
                (0.50, 0.98), (0.14, 0.80), (0.02, 0.50), (0.14, 0.20), (0.50, 0.02)]
_C: Polyline = [(0.94, 0.22), (0.72, 0.04), (0.40, 0.03), (0.10, 0.22),
                (0.02, 0.50), (0.10, 0.78), (0.40, 0.97), (0.72, 0.96), (0.94, 0.78)]
_P_BOWL: Polyline = [(0.14, 0.02), (0.60, 0.02), (0.88, 0.16), (0.88, 0.38),
                     (0.60, 0.52), (0.14, 0.52)]

_STROKES: Dict[str, List[Polyline]] = {
    "A": [[(0.03, 1.0), (0.50, 0.0), (0.97, 1.0)], [(0.20, 0.62), (0.80, 0.62)]],
    "C": [_C],
    "D": [[(0.14, 0.02), (0.14, 0.98)],
          [(0.14, 0.02), (0.56, 0.02), (0.88, 0.26), (0.88, 0.74), (0.56, 0.98), (0.14, 0.98)]],
    "E": [[(0.90, 0.02), (0.13, 0.02), (0.13, 0.98), (0.90, 0.98)],
          [(0.13, 0.50), (0.74, 0.50)]],
    "F": [[(0.90, 0.02), (0.13, 0.02), (0.13, 0.98)], [(0.13, 0.50), (0.74, 0.50)]],
    "G": [_C + [(0.94, 0.56), (0.56, 0.56)]],
    "H": [[(0.13, 0.02), (0.13, 0.98)], [(0.87, 0.02), (0.87, 0.98)],
          [(0.13, 0.50), (0.87, 0.50)]],
    "I": [[(0.50, 0.02), (0.50, 0.98)], [(0.20, 0.02), (0.80, 0.02)],
          [(0.20, 0.98), (0.80, 0.98)]],
    "K": [[(0.15, 0.02), (0.15, 0.98)], [(0.90, 0.02), (0.15, 0.56)],
          [(0.42, 0.38), (0.92, 0.98)]],
    "L": [[(0.16, 0.02), (0.16, 0.98), (0.88, 0.98)]],
    "M": [[(0.06, 0.98), (0.06, 0.02), (0.50, 0.66), (0.94, 0.02), (0.94, 0.98)]],
    "N": [[(0.14, 0.98), (0.14, 0.02), (0.86, 0.98), (0.86, 0.02)]],
    "O": [_O],
    "P": [[(0.14, 0.98), (0.14, 0.02)], _P_BOWL],
    "Q": [_O, [(0.58, 0.70), (0.96, 1.02)]],
    "R": [[(0.14, 0.98), (0.14, 0.02)], _P_BOWL, [(0.50, 0.52), (0.90, 0.98)]],
    "S": [[(0.92, 0.18), (0.68, 0.03), (0.32, 0.03), (0.10, 0.20), (0.16, 0.42),
           (0.50, 0.50), (0.84, 0.58), (0.90, 0.80), (0.68, 0.97), (0.32, 0.97),
           (0.08, 0.82)]],
    "T": [[(0.50, 0.02), (0.50, 0.98)], [(0.05, 0.02), (0.95, 0.02)]],
    "U": [[(0.13, 0.02), (0.13, 0.66), (0.40, 0.97), (0.60, 0.97), (0.87, 0.66),
           (0.87, 0.02)]],
    "V": [[(0.04, 0.02), (0.50, 0.98), (0.96, 0.02)]],
    "W": [[(0.02, 0.02), (0.26, 0.98), (0.50, 0.34), (0.74, 0.98), (0.98, 0.02)]],
    "X": [[(0.08, 0.02), (0.92, 0.98)], [(0.92, 0.02), (0.08, 0.98)]],
    "Y": [[(0.06, 0.02), (0.50, 0.50), (0.94, 0.02)], [(0.50, 0.50), (0.50, 0.98)]],
}

# (S, 2, 2) arrays of segment endpoints, x already scaled into cap-height units
# so a distance in this space is the same in both directions.
_CACHE: Dict[str, np.ndarray] = {}


CELL_W, CELL_H = 64, 88          # one glyph's tile in the atlas, ~ASPECT
_ATLAS_COLS = 6
_ATLAS: Optional[Tuple[np.ndarray, Dict[str, Tuple[float, float, float, float]]]] = None


def atlas() -> Tuple[np.ndarray, Dict[str, Tuple[float, float, float, float]]]:
    """A coverage bitmap of every glyph, plus each one's box in it.

    The GPU path stamps letters as textured quads: one upload, one sample per
    fragment, and the coverage doubles as the blend that mixes ink into the
    surface the letter sits on -- which is what makes it read as printed on the
    tablet rather than floating in front of it. Anti-aliased here rather than
    left to the sampler, so a letter is clean even where the quad is small.

    Returned as ``(image, boxes)`` with ``boxes[char] = (u0, v0, u1, v1)``.
    """
    global _ATLAS
    if _ATLAS is not None:
        return _ATLAS

    chars = sorted(_STROKES)
    rows = (len(chars) + _ATLAS_COLS - 1) // _ATLAS_COLS
    width, height = _ATLAS_COLS * CELL_W, rows * CELL_H
    image = np.zeros((height, width), np.float32)
    boxes = {}

    # Cell coordinates in cap heights, both axes to the same scale, with a
    # half-stroke margin so the ink never reaches the tile edge and bleeds into
    # its neighbour under bilinear sampling.
    margin = STROKE
    u = np.linspace(-margin, ASPECT + margin, CELL_W, dtype=np.float32)
    v = np.linspace(-margin, 1.0 + margin, CELL_H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    # One pixel, in the same units -- the width the edge is feathered over.
    feather = float((v[1] - v[0]) * 0.75)

    for k, char in enumerate(chars):
        row, col = divmod(k, _ATLAS_COLS)
        near = np.full_like(uu, 1e9)
        for (ax, ay), (bx, by) in segments(char):
            dx, dy = float(bx - ax), float(by - ay)
            span = dx * dx + dy * dy
            du, dv = uu - ax, vv - ay
            if span > 1e-12:
                t = np.clip(du * (dx / span) + dv * (dy / span), 0.0, 1.0)
                du = du - t * dx
                dv = dv - t * dy
            np.minimum(near, np.hypot(du, dv), out=near)
        coverage = np.clip((STROKE - near) / max(feather, 1e-6) + 0.5, 0.0, 1.0)
        y0, x0 = row * CELL_H, col * CELL_W
        image[y0:y0 + CELL_H, x0:x0 + CELL_W] = coverage
        boxes[char] = (x0 / width, y0 / height,
                       (x0 + CELL_W) / width, (y0 + CELL_H) / height)

    _ATLAS = ((image * 255.0).astype(np.uint8), boxes)
    return _ATLAS


def atlas_box(char: str) -> Tuple[float, float, float, float]:
    """The atlas box for *char*, falling back to ``X``."""
    _image, boxes = atlas()
    char = (char or "X").upper()[:1]
    return boxes.get(char, boxes["X"])


def segments(char: str) -> np.ndarray:
    """(S, 2, 2) segment endpoints for *char*: ``[[x0, y0], [x1, y1]]`` each.

    x is pre-multiplied by :data:`ASPECT`, so both axes are in cap heights and
    a single stroke half-width applies to every direction.
    """
    char = (char or "X").upper()[:1]
    cached = _CACHE.get(char)
    if cached is None:
        rows = []
        for line in _STROKES.get(char) or _STROKES["X"]:
            for a, b in zip(line, line[1:]):
                rows.append([[a[0] * ASPECT, a[1]], [b[0] * ASPECT, b[1]]])
        cached = _CACHE[char] = np.asarray(rows, np.float32)
    return cached
