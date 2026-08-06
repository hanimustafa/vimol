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
ASPECT = 0.70
STROKE = 0.062
# Half-length of the tick added across each free stroke end. Serifs are derived
# rather than drawn: every terminal gets one, perpendicular to the stroke it
# finishes. It is what stops a run of capitals reading as marker pen, and doing
# it by rule keeps thirty-three glyphs consistent with each other.
SERIF = 0.085

Polyline = List[Tuple[float, float]]

def _arc(start: float, end: float, steps: int = 18, rx: float = 0.48,
         ry: float = 0.48, cx: float = 0.50, cy: float = 0.50) -> "Polyline":
    """A run of an ellipse, in the unit box. Angles in turns, 0 at three
    o'clock, running clockwise in the y-down box these glyphs live in.

    Round letters are generated rather than hand-listed: at eight or nine
    points an O reads as a polygon once a tablet fills a quarter of the frame,
    and the fix is more points, not better-chosen ones.
    """
    import math
    t = [start + (end - start) * i / (steps - 1) for i in range(steps)]
    return [(cx + rx * math.cos(2 * math.pi * a),
             cy + ry * math.sin(2 * math.pi * a)) for a in t]


_O: Polyline = _arc(0.0, 1.0)
_C: Polyline = _arc(0.08, 0.92)
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
    # The bar hangs off the lower end of the arc, so the arc is run in the
    # direction that finishes there.
    "G": [_arc(0.92, 0.08) + [(0.92, 0.55), (0.56, 0.55)]],
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
    # Digits, for the residue number that trails the code.
    "0": [_arc(0.0, 1.0, rx=0.40)],
    "1": [[(0.24, 0.20), (0.50, 0.02), (0.50, 0.98)], [(0.24, 0.98), (0.78, 0.98)]],
    "2": [[(0.12, 0.22), (0.32, 0.03), (0.66, 0.04), (0.86, 0.24), (0.82, 0.46),
           (0.14, 0.98), (0.90, 0.98)]],
    "3": [[(0.12, 0.16), (0.38, 0.02), (0.70, 0.05), (0.84, 0.25), (0.60, 0.46)],
          [(0.60, 0.46), (0.86, 0.64), (0.74, 0.92), (0.40, 0.98), (0.12, 0.86)]],
    "4": [[(0.70, 0.98), (0.70, 0.02), (0.10, 0.70), (0.92, 0.70)]],
    "5": [[(0.84, 0.03), (0.24, 0.03), (0.18, 0.42), (0.46, 0.34), (0.76, 0.46),
           (0.86, 0.70), (0.70, 0.93), (0.36, 0.98), (0.12, 0.86)]],
    "6": [_arc(0.0, 1.0, steps=13, rx=0.34, ry=0.27, cy=0.71),
          [(0.16, 0.71), (0.28, 0.24), (0.54, 0.03), (0.82, 0.03)]],
    "7": [[(0.08, 0.03), (0.90, 0.03), (0.38, 0.98)]],
    "8": [_arc(0.0, 1.0, steps=13, rx=0.30, ry=0.24, cy=0.26),
          _arc(0.0, 1.0, steps=13, rx=0.36, ry=0.26, cy=0.74)],
    "9": [_arc(0.0, 1.0, steps=13, rx=0.34, ry=0.27, cy=0.29),
          [(0.84, 0.29), (0.72, 0.76), (0.46, 0.97), (0.18, 0.97)]],
}

# The residue number rides beside the one-letter code, smaller and sharing its
# baseline, the way a subscript does.
NUMBER_SCALE = 0.60
NUMBER_GAP = 0.09           # cap heights between the code and the number


def layout(code: str, number: str, size: float):
    """Place a residue's code and number: ``(char, dx, dy, height)`` each.

    ``dx``/``dy`` are offsets from the centre of the whole run, in world units,
    with ``dy`` pointing the same way the glyph's own down does. Both renderers
    lay a label out through this, so the terminal and the GPU put the same text
    in the same place.
    """
    number = "".join(c for c in str(number) if c.isdigit())
    small = size * NUMBER_SCALE
    code_w = size * ASPECT
    digit_w = small * ASPECT
    gap = size * NUMBER_GAP if number else 0.0
    total = code_w + gap + digit_w * len(number)
    x = -total * 0.5
    out = [(code[:1] or "X", x + code_w * 0.5, 0.0, size)]
    x += code_w + gap
    # Sharing the code's baseline means the smaller glyph sits lower by half
    # the difference in height.
    drop = (size - small) * 0.5
    for digit in number:
        out.append((digit, x + digit_w * 0.5, drop, small))
        x += digit_w
    return out

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


def run_width(code: str, number: str, size: float) -> float:
    """How wide the whole code-plus-number run is, in world units."""
    run = layout(code, number, size)
    left = min(dx - h * ASPECT * 0.5 for _c, dx, _dy, h in run)
    right = max(dx + h * ASPECT * 0.5 for _c, dx, _dy, h in run)
    return right - left


def atlas_box(char: str) -> Tuple[float, float, float, float]:
    """The atlas box for *char*, falling back to ``X``."""
    _image, boxes = atlas()
    char = (char or "X").upper()[:1]
    return boxes.get(char, boxes["X"])


def _serif(a, b):
    """A tick across the free end *b* of the stroke that runs from *a*."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return None
    # Perpendicular to the stroke, so a vertical stem gets a horizontal foot.
    px, py = -dy / length * SERIF, dx / length * SERIF
    return [[b[0] - px, b[1] - py], [b[0] + px, b[1] + py]]


def segments(char: str) -> np.ndarray:
    """(S, 2, 2) segment endpoints for *char*: ``[[x0, y0], [x1, y1]]`` each.

    x is pre-multiplied by :data:`ASPECT`, so both axes are in cap heights and
    a single stroke half-width applies to every direction. Serifs are added at
    every free terminal; a closed polyline (an O, the bowl of a 6) has no free
    end and so gets none.
    """
    char = (char or "X").upper()[:1]
    cached = _CACHE.get(char)
    if cached is None:
        rows = []
        for line in _STROKES.get(char) or _STROKES["X"]:
            for a, b in zip(line, line[1:]):
                rows.append([list(a), list(b)])
            if tuple(line[0]) != tuple(line[-1]):
                for a, b in ((line[1], line[0]), (line[-2], line[-1])):
                    tick = _serif(a, b)
                    if tick is not None:
                        rows.append(tick)
        scaled = np.asarray(rows, np.float32)
        scaled[:, :, 0] *= ASPECT
        cached = _CACHE[char] = scaled
    return cached
