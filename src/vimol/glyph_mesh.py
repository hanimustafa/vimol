"""Triangle meshes for the glyph skin's ribbon and tablets.

The CPU raycaster draws those two as sets of half-spaces and fakes a smooth
ribbon with interpolated normals. The GPU can have the real thing: a swept tube
with a rounded cross-section, and tablets with chamfered rims, both with genuine
per-vertex normals. That is where the difference between a diagram and a
rendering actually lives -- a faceted box with clever shading still shows its
silhouette, and a chamfer catches a highlight along its edge that nothing else
reproduces.

The ribbon and the tablets are built once per molecule in world space and cached
with the rest of the glyph scene; the per-frame cost is one rotation of the
vertices and normals into view space. The letters are the exception and are
rebuilt every frame -- see :func:`letters` for why.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .glyph_font import ASPECT, atlas_box, layout

# Points per quarter-turn of the ribbon's rounded corners. Four is enough that
# the highlight running along the edge reads as curved at any terminal size.
CORNER_STEPS = 4
RIBBON_CORNER = 0.45        # corner radius as a fraction of the half-thickness
TABLET_CHAMFER = 0.055      # angstrom cut off the tablet's face/rim edge
LETTER_LIFT = 0.006         # angstrom the letter quad floats over its face
NO_UV = (-1.0, -1.0)        # a vertex that samples no glyph
# A letter printed onto a tablet face blends ink into the face it sits on, so
# the whole quad is surface and only the strokes darken. One floating in front
# of a rounded volume has no face to blend with -- its quad must vanish except
# where the ink is. The shader tells them apart by this offset on u, which
# keeps both in one mesh and one draw call.
CUTOUT = 2.0
# How square a tablet has to be to the camera before its letter is printed onto
# the face rather than floated in front of it.
FACE_ON_MIN = 0.45


def _unit(v, fallback):
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / n if n > 1e-9 else np.asarray(fallback, float)


@dataclass
class GlyphMesh:
    """An indexed triangle mesh in world space, plus per-vertex appearance."""
    vertices: np.ndarray       # (V, 3)
    normals: np.ndarray        # (V, 3)
    colors: np.ndarray         # (V, 3)
    flat: np.ndarray           # (V,)
    uv: np.ndarray             # (V, 2)
    indices: np.ndarray        # (I,)

    @staticmethod
    def empty() -> "GlyphMesh":
        return GlyphMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)),
                         np.zeros((0,)), np.zeros((0, 2)), np.zeros((0,), np.int64))


class MeshBuilder:
    """Accumulates triangles, then freezes them into a :class:`GlyphMesh`."""

    def __init__(self) -> None:
        self._pos: List[np.ndarray] = []
        self._nrm: List[np.ndarray] = []
        self._col: List[np.ndarray] = []
        self._flat: List[np.ndarray] = []
        self._uv: List[np.ndarray] = []
        self._idx: List[np.ndarray] = []
        self._n = 0

    def _add(self, pos, nrm, uv, color, flat) -> int:
        base = self._n
        count = len(pos)
        self._pos.append(np.asarray(pos, float).reshape(count, 3))
        self._nrm.append(np.asarray(nrm, float).reshape(count, 3))
        self._uv.append(np.asarray(uv, float).reshape(count, 2))
        self._col.append(np.tile(np.asarray(color, float), (count, 1)))
        self._flat.append(np.full(count, 1.0 if flat else 0.0))
        self._n += count
        return base

    def strip(self, rings_pos: np.ndarray, rings_nrm: np.ndarray, color, flat) -> None:
        """Sweep a closed cross-section along a path.

        ``rings_pos``/``rings_nrm`` are ``(steps, ring, 3)``: one closed loop of
        points per step along the path, already positioned in world space. Every
        neighbouring pair of loops becomes a band of quads.
        """
        steps, ring = rings_pos.shape[0], rings_pos.shape[1]
        if steps < 2 or ring < 3:
            return
        base = self._add(rings_pos.reshape(-1, 3), rings_nrm.reshape(-1, 3),
                         np.tile(NO_UV, (steps * ring, 1)), color, flat)
        s = np.arange(steps - 1)[:, None]
        k = np.arange(ring)[None, :]
        a = base + s * ring + k
        b = base + s * ring + (k + 1) % ring
        c = a + ring
        d = b + ring
        self._idx.append(np.stack([a, b, d, a, d, c], axis=-1).ravel())

    def cap(self, loop: np.ndarray, normal, color, flat) -> None:
        """Close a convex loop with a triangle fan facing *normal*."""
        n = len(loop)
        if n < 3:
            return
        base = self._add(loop, np.tile(normal, (n, 1)), np.tile(NO_UV, (n, 1)),
                         color, flat)
        k = np.arange(1, n - 1)
        self._idx.append(np.stack([np.full(n - 2, base), base + k, base + k + 1],
                                  axis=-1).ravel())

    def quad(self, corners: Sequence[np.ndarray], normal, uv, color, flat) -> None:
        """Two triangles over four corners, in order."""
        base = self._add(np.asarray(corners, float), np.tile(normal, (4, 1)),
                         uv, color, flat)
        self._idx.append(np.array([base, base + 1, base + 2,
                                   base, base + 2, base + 3], np.int64))

    def freeze(self) -> GlyphMesh:
        if not self._pos:
            return GlyphMesh.empty()
        return GlyphMesh(
            vertices=np.concatenate(self._pos),
            normals=np.concatenate(self._nrm),
            colors=np.concatenate(self._col),
            flat=np.concatenate(self._flat),
            uv=np.concatenate(self._uv),
            indices=np.concatenate(self._idx).astype(np.int64),
        )


def rounded_section(half_width: float, half_thickness: float):
    """A rounded-rectangle cross-section: local 2-D points and their normals.

    The corners are quarter arcs, so a point on one carries the arc's own
    outward normal and the flat runs between two arc ends inherit the flat
    normal at each end. That gives a ribbon with a soft edge that takes a
    highlight along its length -- the single detail that most separates a
    rendered ribbon from an extruded rectangle.
    """
    r = min(half_thickness * RIBBON_CORNER, half_width * 0.35)
    ax, ay = half_width - r, half_thickness - r
    centers = [(ax, ay), (-ax, ay), (-ax, -ay), (ax, -ay)]
    starts = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    pts, nrm = [], []
    for (cx, cy), start in zip(centers, starts):
        phi = start + np.linspace(0.0, 0.5 * np.pi, CORNER_STEPS)
        cos, sin = np.cos(phi), np.sin(phi)
        pts.append(np.column_stack([cx + r * cos, cy + r * sin]))
        nrm.append(np.column_stack([cos, sin]))
    return np.concatenate(pts), np.concatenate(nrm)


def ribbon(builder: MeshBuilder, path: np.ndarray, sides: np.ndarray,
           half_width: float, half_thickness: float, color, flat) -> None:
    """Sweep the ribbon's cross-section along a Cα spline."""
    if len(path) < 2:
        return
    tangents = np.gradient(path, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    # Orthogonalize the side vectors against the tangent, then re-derive up, so
    # the frame stays right-handed even where the spline turns sharply.
    side = sides - tangents * np.einsum("ij,ij->i", sides, tangents)[:, None]
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    up = np.cross(tangents, side)

    section, section_n = rounded_section(half_width, half_thickness)
    # (steps, ring, 3): each ring point is centre + x*side + y*up.
    pos = (path[:, None, :]
           + section[None, :, 0, None] * side[:, None, :]
           + section[None, :, 1, None] * up[:, None, :])
    nrm = (section_n[None, :, 0, None] * side[:, None, :]
           + section_n[None, :, 1, None] * up[:, None, :])
    builder.strip(pos, nrm, color, flat)
    # Close both ends so a chain terminus is a solid stop, not a hollow pipe.
    builder.cap(pos[0][::-1], -tangents[0], color, flat)
    builder.cap(pos[-1], tangents[-1], color, flat)


def tablet(builder: MeshBuilder, center: np.ndarray, e1: np.ndarray, e2: np.ndarray,
           normal: np.ndarray, loop: np.ndarray, half_thickness: float,
           color, flat) -> None:
    """A chamfered prism over a convex outline.

    *loop* is the outline in the tablet's own plane, counter-clockwise. The
    chamfer is cut inward from the face and the rim rather than rounded outward,
    so the solid never grows past the bounds the camera was framed on.
    """
    def to3(points2d, height):
        return (center[None, :] + points2d[:, 0, None] * e1[None, :]
                + points2d[:, 1, None] * e2[None, :] + height * normal[None, :])

    chamfer = min(TABLET_CHAMFER, half_thickness * 0.7)
    # Shrink the outline toward its centroid for the inset face ring; for a
    # convex polygon that is close enough to a true offset at this scale.
    middle = loop.mean(axis=0)
    span = float(np.linalg.norm(loop - middle, axis=1).mean())
    inset = middle + (loop - middle) * max(0.0, 1.0 - chamfer / max(span, 1e-6))

    rim_h = half_thickness - chamfer
    bevel = float(np.sqrt(0.5))
    outward = loop - middle
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
    out3 = outward[:, 0, None] * e1[None, :] + outward[:, 1, None] * e2[None, :]

    for sign in (1.0, -1.0):
        face = to3(inset, sign * half_thickness)
        edge = to3(loop, sign * rim_h)
        builder.cap(face if sign > 0 else face[::-1], sign * normal, color, flat)
        builder.strip(np.stack([face, edge]),
                      np.stack([np.tile(sign * normal, (len(face), 1)),
                                out3 * bevel + sign * normal * bevel]),
                      color, flat)
    builder.strip(np.stack([to3(loop, rim_h), to3(loop, -rim_h)]),
                  np.stack([out3, out3]), color, flat)


def label(builder: MeshBuilder, center: np.ndarray, right: np.ndarray,
          down: np.ndarray, normal: np.ndarray, code: str, number: str,
          size: float, color, flat, cutout: bool = False) -> None:
    """A residue's code and number, as one textured quad per glyph."""
    for char, dx, dy, height in layout(code, number, size):
        at = center + right * dx + down * dy
        u0, v0, u1, v1 = atlas_box(char)
        if cutout:
            u0, u1 = u0 + CUTOUT, u1 + CUTOUT
        half_w = height * ASPECT * 0.5
        half_h = height * 0.5
        corners = [at - right * half_w - down * half_h,
                   at + right * half_w - down * half_h,
                   at + right * half_w + down * half_h,
                   at - right * half_w + down * half_h]
        builder.quad(corners, normal, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)],
                     color, flat)


def letters(scene, rotation: np.ndarray) -> GlyphMesh:
    """Every letter, placed for this frame's camera orientation.

    Letters are the one part of the skin that cannot be cached with the rest of
    the geometry, because both of the things that make one readable depend on
    where the camera is. A letter on a tablet is printed onto the face and
    stands upright *on screen* -- pinning its rotation to the tablet's own axes
    instead would leave it lying at whatever angle the ring happens to sit at.
    A letter on a rounded volume has no face to print on, so it is squared to
    the viewer and cut out of its quad. A tablet turned nearly edge-on falls
    back to the second: there is no face left to print on, and no reason to
    prefer a label you cannot read to one you can.
    """
    view_right, view_down, toward = rotation[0], -rotation[1], rotation[2]
    builder = MeshBuilder()
    for k, char in enumerate(scene.label_char):
        size = float(scene.label_size[k])
        flat = bool(scene.label_flat[k])
        facing = float(np.dot(scene.label_normal[k], toward))
        if scene.label_on_tablet[k] and abs(facing) >= FACE_ON_MIN:
            face = scene.label_normal[k] * (1.0 if facing > 0 else -1.0)
            # Screen-down dropped into the tablet's plane: the letter lies on
            # the face and still stands upright to the reader.
            down = _unit(view_down - face * float(np.dot(view_down, face)),
                         view_right)
            label(builder,
                  scene.label_center[k] + face * float(scene.label_offset[k]),
                  np.cross(face, down), down, face, char, scene.label_number[k],
                  size, scene.label_surface[k], flat)
        else:
            # Lifted toward the camera by the reach of the solid it names, so
            # the shape it belongs to cannot swallow it.
            label(builder,
                  scene.label_center[k] + toward * float(scene.label_bias[k]),
                  view_right, view_down, toward, char, scene.label_number[k],
                  size, scene.label_color[k], flat, cutout=True)
    return builder.freeze()
