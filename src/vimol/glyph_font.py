"""Real letterforms for the one-letter codes and the residue numbers.

The outlines live in :mod:`glyph_outlines`, baked out of DejaVu Sans by
``scripts/build_glyph_outlines.py``. Baking them means vimol carries no font
machinery -- no fontTools, no freetype, no Pillow (see the README) -- and draws
the same letterforms everywhere rather than whatever the host has installed.

DejaVu rather than Helvetica: Helvetica is proprietary and its outlines are not
ours to redistribute, while DejaVu ships under the permissive Bitstream Vera
license, which sits fine alongside vimol's MIT.

Both renderers work in *cell* coordinates: each glyph occupies a box of
``glyph_box(char)`` cap heights, with the glyph's own advance width plus a
margin across and the cap height plus a margin down. The GPU samples that box
out of the atlas; the raycaster tests the same box against the outlines
directly. So a letter is the same shape and the same size either way.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .glyph_outlines import ADVANCE, OUTLINES

# Slack around the advance box, in cap heights: enough for Q's tail and for the
# atlas edge never to clip a stroke.
MARGIN = 0.14
# Samples per axis when filling the atlas. The coverage this averages is what
# makes a letter's edge smooth at any size it is later drawn.
OVERSAMPLE = 3

CELL_H = 96                      # atlas rows per glyph
_ATLAS_COLS = 6


def glyph_box(char: str) -> Tuple[float, float]:
    """(width, height) of *char*'s cell, in cap heights."""
    char = _key(char)
    return ADVANCE[char] + 2 * MARGIN, 1.0 + 2 * MARGIN


def _key(char: str) -> str:
    char = (char or "X").upper()[:1]
    return char if char in OUTLINES else "X"


_CONTOURS: Dict[str, List[np.ndarray]] = {}


def contours(char: str) -> List[np.ndarray]:
    """*char*'s closed contours, in its cell's 0..1 coordinates.

    Used by the raycaster, which fills them with a crossing test per pixel
    rather than sampling a texture.
    """
    char = _key(char)
    cached = _CONTOURS.get(char)
    if cached is None:
        w, h = glyph_box(char)
        cached = _CONTOURS[char] = [
            np.array([((x + MARGIN) / w, (y + MARGIN) / h) for x, y in contour],
                     np.float64)
            for contour in OUTLINES[char]]
    return cached


def _fill(char: str, width: int, height: int) -> np.ndarray:
    """Coverage of *char* over a width x height grid of its cell.

    Even-odd crossing count, oversampled and averaged: for every edge, a pixel
    centre is inside if a ray cast from it crosses that edge an odd number of
    times. TrueType contours wind consistently, so even-odd and non-zero agree
    here and the parity test is the cheaper of the two.
    """
    n = OVERSAMPLE
    xs = (np.arange(width * n) + 0.5) / (width * n)
    ys = (np.arange(height * n) + 0.5) / (height * n)
    u = xs[None, :]
    v = ys[:, None]
    inside = np.zeros((height * n, width * n), bool)
    for contour in contours(char):
        a = contour
        b = np.roll(contour, -1, axis=0)
        for (x0, y0), (x1, y1) in zip(a, b):
            if y0 == y1:
                continue
            straddles = (y0 > v) != (y1 > v)
            crossing = x0 + (v - y0) * ((x1 - x0) / (y1 - y0))
            inside ^= straddles & (u < crossing)
    # Average each pixel's subsamples back down.
    return inside.reshape(height, n, width, n).mean(axis=(1, 3))


_ATLAS: Optional[Tuple[np.ndarray, Dict[str, Tuple[float, float, float, float]]]] = None


def atlas() -> Tuple[np.ndarray, Dict[str, Tuple[float, float, float, float]]]:
    """A coverage bitmap of every glyph, plus each one's box in it.

    The GPU stamps letters as textured quads: one upload, one sample per
    fragment, and the coverage doubles as the blend that mixes ink into the
    surface the letter sits on -- which is what makes it read as printed on a
    tablet rather than floating in front of it.
    """
    global _ATLAS
    if _ATLAS is not None:
        return _ATLAS

    chars = sorted(OUTLINES)
    # One cell width for the whole sheet, wide enough for the widest glyph, so
    # the grid stays regular; narrower glyphs simply use less of their cell.
    height = glyph_box(chars[0])[1]
    cell_w = int(round(CELL_H * max(glyph_box(c)[0] for c in chars) / height))
    rows = (len(chars) + _ATLAS_COLS - 1) // _ATLAS_COLS
    image = np.zeros((rows * CELL_H, _ATLAS_COLS * cell_w), np.float32)
    boxes = {}
    for k, char in enumerate(chars):
        row, col = divmod(k, _ATLAS_COLS)
        w, h = glyph_box(char)
        # Each glyph fills only as much of its cell as its own advance needs,
        # so a narrow digit is not stretched to a wide letter's width.
        used = min(cell_w, max(1, int(round(CELL_H * w / h))))
        y0, x0 = row * CELL_H, col * cell_w
        image[y0:y0 + CELL_H, x0:x0 + used] = _fill(char, used, CELL_H)
        boxes[char] = (x0 / image.shape[1], y0 / image.shape[0],
                       (x0 + used) / image.shape[1], (y0 + CELL_H) / image.shape[0])
    _ATLAS = ((image * 255.0).astype(np.uint8), boxes)
    return _ATLAS


def atlas_box(char: str) -> Tuple[float, float, float, float]:
    """The atlas box for *char*, falling back to ``X``."""
    _image, boxes = atlas()
    return boxes[_key(char)]


# The residue number rides beside the one-letter code, smaller and sharing its
# baseline, the way a subscript does.
NUMBER_SCALE = 0.60
NUMBER_GAP = 0.06           # cap heights between the code and the number


def layout(code: str, number: str, size: float):
    """Place a residue's code and number: ``(char, dx, dy, height)`` each.

    ``dx``/``dy`` are offsets from the centre of the whole run, in world units,
    with ``dy`` pointing the same way the glyph's own down does. Both renderers
    lay a label out through this, so the terminal and the GPU put the same text
    in the same place -- and both then size each quad from ``glyph_box``, so a
    narrow digit is not stretched to a wide letter's width.
    """
    number = "".join(c for c in str(number) if c.isdigit())
    small = size * NUMBER_SCALE
    widths = [(code[:1] or "X", size)] + [(d, small) for d in number]
    advances = [ADVANCE[_key(c)] * h for c, h in widths]
    gap = size * NUMBER_GAP if number else 0.0
    total = sum(advances) + gap
    x = -total * 0.5
    out = []
    for k, ((char, height), advance) in enumerate(zip(widths, advances)):
        if k == 1:
            x += gap
        # Sharing the code's baseline means a smaller glyph sits lower by half
        # the difference in height.
        out.append((char, x + advance * 0.5, (size - height) * 0.5, height))
        x += advance
    return out


def run_width(code: str, number: str, size: float) -> float:
    """How wide the whole code-plus-number run is, in world units."""
    run = layout(code, number, size)
    left = min(dx - glyph_box(c)[0] * h * 0.5 for c, dx, _dy, h in run)
    right = max(dx + glyph_box(c)[0] * h * 0.5 for c, dx, _dy, h in run)
    return right - left
