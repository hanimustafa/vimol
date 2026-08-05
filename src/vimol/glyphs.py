"""Build the glyph skin's geometry: ribbon, plates, volumes, letters, nodes.

This module turns a protein into drawable primitives and nothing else -- it
never touches a framebuffer. Everything it emits is in world coordinates, so
the renderer only has to rotate it into view space.

The four kinds of primitive it produces:

``spheres``     the rounded side-chain volumes and the hydrogen-bonding nodes
``cylinders``   the Cα-to-glyph sticks and the hydrogen-bond links
``polyhedra``   ribbon segments and aromatic ring plates, as sets of half-spaces
``labels``      screen-aligned one-letter codes

A convex solid is stored as a center plus planes ``n·(p − center) ≤ d``. Keeping
the offsets relative to a center makes them invariant under the camera's
rotation, so a frame only has to rotate the normals -- one matmul for every
plane in the scene at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .molecule import Molecule
from .residues import (BACKBONE, AROMATIC_RINGS, Residue, chain_runs,
                       hbond_role, protein_residues)


# -- tunables ------------------------------------------------------------
RIBBON_WIDTH = 1.6          # angstrom, full width across the ribbon
RIBBON_THICKNESS = 0.26     # angstrom, full thickness
RIBBON_SAMPLES = 8          # spline samples per residue
PLATE_INFLATE = 0.30        # angstrom the ring hull is pushed out by
PLATE_THICKNESS = 0.22      # angstrom, full thickness
BLOB_RADIUS = 0.85          # angstrom; comfortably over half a 1.5 A bond, so
                            # bonded atoms' spheres merge into one solid
GLYCINE_RADIUS = 0.55       # glycine has no side chain, so its marker is small
LETTER_SIZE = 1.0           # angstrom, cap height of the one-letter code
STICK_RADIUS = 0.10
NODE_RADIUS = 0.13
NODE_SPLIT = 0.16           # see `_node_positions`
LINK_RADIUS = 0.05
LINK_MAX_DISTANCE = 3.35    # angstrom, N···O
LINK_MIN_SEPARATION = 2     # residues apart, so i/i+1 neighbours don't count


@dataclass(frozen=True)
class Palette:
    ribbon: Tuple[float, float, float]
    plate: Tuple[float, float, float]
    volume: Tuple[float, float, float]
    stick: Tuple[float, float, float]
    ink: Tuple[float, float, float]
    donor: Tuple[float, float, float]
    acceptor: Tuple[float, float, float]
    link: Tuple[float, float, float]


# The letters are always dark, so both palettes keep the side-chain solids
# light; only the ribbon and the node colours flip with the background.
LIGHT = Palette(
    ribbon=(0.17, 0.17, 0.19), plate=(0.70, 0.50, 0.16),
    volume=(0.90, 0.87, 0.80), stick=(0.32, 0.33, 0.35),
    ink=(0.09, 0.09, 0.11), donor=(0.16, 0.36, 0.72),
    acceptor=(0.66, 0.12, 0.12), link=(0.80, 0.66, 0.42),
)
DARK = Palette(
    ribbon=(0.80, 0.82, 0.87), plate=(0.85, 0.62, 0.22),
    volume=(0.88, 0.86, 0.80), stick=(0.62, 0.64, 0.70),
    ink=(0.09, 0.09, 0.11), donor=(0.38, 0.62, 0.98),
    acceptor=(0.92, 0.32, 0.30), link=(0.88, 0.72, 0.42),
)


def palette(theme: str) -> Palette:
    return LIGHT if str(theme).lower() == "light" else DARK


@dataclass
class GlyphScene:
    """Drawable primitives for one molecule, in world coordinates."""
    sphere_center: np.ndarray      # (S, 3)
    sphere_radius: np.ndarray      # (S,)
    sphere_color: np.ndarray       # (S, 3)
    cyl_a: np.ndarray              # (C, 3)
    cyl_b: np.ndarray              # (C, 3)
    cyl_radius: np.ndarray         # (C,)
    cyl_color: np.ndarray          # (C, 3)
    poly_center: np.ndarray        # (P, 3)
    poly_color: np.ndarray         # (P, 3)
    poly_axes: np.ndarray          # (P, 3, 3) orthonormal rows of the enclosing box
    poly_half: np.ndarray          # (P, 3) half-extent along each of those axes
    poly_slice: np.ndarray         # (P + 1,) int, CSR bounds into the plane arrays
    plane_normal: np.ndarray       # (M, 3) unit, world space
    plane_offset: np.ndarray       # (M,)
    label_center: np.ndarray       # (L, 3)
    label_size: np.ndarray         # (L,) glyph height in angstrom
    label_bias: np.ndarray         # (L,) angstrom to lift the letter toward the camera
    label_color: np.ndarray        # (L, 3)
    label_char: List[str]

    def reach_from(self, center) -> float:
        """How far from *center* the drawn geometry actually gets.

        This is what the camera has to frame, and an atom radius is no guide
        to it: a ribbon runs between the Cα positions, a plate stands off past
        its own ring, and a letter is a billboard half its cap height wide.
        """
        center = np.asarray(center, float)
        spans = []
        for points, pad in ((self.sphere_center, self.sphere_radius),
                            (self.poly_center, np.linalg.norm(self.poly_half, axis=1)),
                            (self.label_center, self.label_size * 0.5),
                            (self.cyl_a, self.cyl_radius),
                            (self.cyl_b, self.cyl_radius)):
            if len(points):
                spans.append(np.linalg.norm(points - center, axis=1) + pad)
        return float(np.concatenate(spans).max()) if spans else 0.0


class _Builder:
    """Accumulates primitives, then freezes them into a GlyphScene."""

    def __init__(self, pal: Palette):
        self.pal = pal
        self.spheres: List[Tuple[np.ndarray, float, Tuple[float, float, float]]] = []
        self.cylinders: List[Tuple[np.ndarray, np.ndarray, float, Tuple[float, float, float]]] = []
        self.poly_center: List[np.ndarray] = []
        self.poly_color: List[Tuple[float, float, float]] = []
        self.poly_axes: List[np.ndarray] = []
        self.poly_half: List[np.ndarray] = []
        self.poly_counts: List[int] = []
        self.normals: List[np.ndarray] = []
        self.offsets: List[np.ndarray] = []
        self.labels: List[Tuple[np.ndarray, str, float, float, Tuple[float, float, float]]] = []

    def sphere(self, center, radius, color) -> None:
        self.spheres.append((np.asarray(center, float), float(radius), color))

    def cylinder(self, a, b, radius, color) -> None:
        self.cylinders.append((np.asarray(a, float), np.asarray(b, float), float(radius), color))

    def solid(self, center, normals, offsets, color, axes, half) -> None:
        """Add a convex solid, plus an oriented box that encloses it.

        A set of half-spaces does not hand over a bounding box, and the
        rasterizer needs one; only the geometry that produced the planes knows
        how far the solid reaches. *axes* (3 orthonormal rows) and *half* (a
        half-extent along each) let the renderer project a tight screen
        rectangle instead of the square a bounding sphere would give -- for a
        ribbon segment, which is a thin slab, that square is several times the
        pixels the segment actually covers, and every one of them gets shaded.
        """
        normals = np.asarray(normals, float)
        self.poly_center.append(np.asarray(center, float))
        self.poly_color.append(color)
        self.poly_axes.append(np.asarray(axes, float))
        self.poly_half.append(np.asarray(half, float))
        self.poly_counts.append(len(normals))
        self.normals.append(normals)
        self.offsets.append(np.asarray(offsets, float))

    def label(self, center, char, size, bias, color) -> None:
        self.labels.append((np.asarray(center, float), char, float(size), float(bias), color))

    def freeze(self) -> GlyphScene:
        def stack(rows, width, dtype=float):
            if not rows:
                return np.zeros((0, width) if width else (0,), dtype)
            return np.asarray(rows, dtype)

        counts = np.asarray(self.poly_counts, np.int64)
        slices = np.zeros(len(counts) + 1, np.int64)
        np.cumsum(counts, out=slices[1:])
        return GlyphScene(
            sphere_center=stack([s[0] for s in self.spheres], 3),
            sphere_radius=stack([s[1] for s in self.spheres], 0),
            sphere_color=stack([s[2] for s in self.spheres], 3),
            cyl_a=stack([c[0] for c in self.cylinders], 3),
            cyl_b=stack([c[1] for c in self.cylinders], 3),
            cyl_radius=stack([c[2] for c in self.cylinders], 0),
            cyl_color=stack([c[3] for c in self.cylinders], 3),
            poly_center=stack(self.poly_center, 3),
            poly_color=stack(self.poly_color, 3),
            poly_axes=(np.asarray(self.poly_axes, float) if self.poly_axes
                       else np.zeros((0, 3, 3))),
            poly_half=stack(self.poly_half, 3),
            poly_slice=slices,
            plane_normal=(np.concatenate(self.normals) if self.normals
                          else np.zeros((0, 3))),
            plane_offset=(np.concatenate(self.offsets) if self.offsets
                          else np.zeros((0,))),
            label_center=stack([l[0] for l in self.labels], 3),
            label_size=stack([l[2] for l in self.labels], 0),
            label_bias=stack([l[3] for l in self.labels], 0),
            label_color=stack([l[4] for l in self.labels], 3),
            label_char=[l[1] for l in self.labels],
        )


# -- small geometry helpers ----------------------------------------------

def _unit(v: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / n if n > 1e-9 else np.asarray(fallback, float)


def box_planes(u: np.ndarray, v: np.ndarray, w: np.ndarray,
               half: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """The six half-spaces of a box with orthonormal axes u/v/w.

    Returned relative to the box center, so the caller supplies that
    separately and the offsets survive any rotation of the camera.
    """
    normals = np.array([u, -u, v, -v, w, -w], float)
    offsets = np.array([half[0], half[0], half[1], half[1], half[2], half[2]], float)
    return normals, offsets


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Indices of the convex hull of 2-D *points*, counter-clockwise.

    Andrew's monotone chain: sort, then sweep the lower and upper chains
    popping any vertex that would make a clockwise turn. scipy has this, but
    scipy is an optional extra here (``vimol[align]``), and the skin must not
    depend on whether the user installed it.
    """
    n = len(points)
    if n < 3:
        return np.arange(n)
    order = np.lexsort((points[:, 1], points[:, 0]))

    def half(seq):
        out: List[int] = []
        for i in seq:
            while len(out) >= 2:
                a, b = points[out[-2]], points[out[-1]]
                if np.cross(b - a, points[i] - a) > 1e-12:
                    break
                out.pop()
            out.append(int(i))
        return out

    lower = half(order)
    upper = half(order[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=np.int64)


def _edge_intersections(normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Corners of the 2-D convex region ``n·p <= d``, from consecutive edges.

    The edges arrive in hull order, so each neighbouring pair meets at one
    corner; a pair that is parallel has no corner and is skipped.
    """
    following = np.roll(np.arange(len(normals)), -1)
    a = np.stack([normals, normals[following]], axis=1)          # (E, 2, 2)
    b = np.stack([offsets, offsets[following]], axis=1)          # (E, 2)
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    ok = np.abs(det) > 1e-9
    if not ok.any():
        return np.zeros((1, 2))
    a, b, det = a[ok], b[ok], det[ok]
    x = (b[:, 0] * a[:, 1, 1] - b[:, 1] * a[:, 0, 1]) / det
    y = (a[:, 0, 0] * b[:, 1] - a[:, 1, 0] * b[:, 0]) / det
    return np.column_stack([x, y])


def plane_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(center, normal, e1, e2) of the best-fit plane through *points*.

    The normal is the least-significant right singular vector of the centered
    coordinates -- the direction of least spread, which for a set of ring
    atoms is the ring normal.
    """
    center = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - center, full_matrices=True)
    normal = _unit(vt[2])
    e1 = _unit(vt[0])
    e2 = _unit(np.cross(normal, e1))
    return center, normal, e1, e2


def _catmull_rom(control: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Sample a Catmull-Rom spline through every row of *control*.

    Endpoints are duplicated so the curve starts and ends exactly on the first
    and last control point rather than overshooting past them.
    """
    n = len(control)
    if n < 2:
        return control.copy()
    padded = np.vstack([control[0], control, control[-1]])
    out = []
    for j in range(n - 1):
        p0, p1, p2, p3 = padded[j], padded[j + 1], padded[j + 2], padded[j + 3]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(control[-1][None, :])
    return np.vstack(out)


def virtual_cbeta(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Where glycine's Cβ would sit, from its backbone alone.

    The standard tetrahedral reconstruction: the three coefficients place the
    fourth substituent on Cα at the right bond length and angles given the N
    and C directions.
    """
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


# -- the pieces of the skin ----------------------------------------------

def _ribbon_frames(run: Sequence[Residue], positions: np.ndarray):
    """Guide points and side vectors along one continuous chain.

    Carson-Bugg: between consecutive Cα atoms the peptide plane fixes a natural
    "side" direction, ``normalize((CA(i+1) − CA(i)) × (O(i) − CA(i)))``. That
    direction flips by nearly 180° from one residue to the next in a β-strand
    (the carbonyls alternate), so each is negated when it opposes its
    predecessor -- without that the ribbon corkscrews once per residue.
    """
    centers: List[np.ndarray] = []
    sides: List[np.ndarray] = []
    previous: Optional[np.ndarray] = None
    for i in range(len(run) - 1):
        ca_i = run[i].index("CA")
        ca_j = run[i + 1].index("CA")
        o_i = run[i].index("O")
        if ca_i is None or ca_j is None:
            continue
        a = positions[ca_j] - positions[ca_i]
        if o_i is None:
            side = previous if previous is not None else _unit(np.cross(a, (0.0, 0.0, 1.0)))
        else:
            side = _unit(np.cross(a, positions[o_i] - positions[ca_i]))
        if previous is not None and float(np.dot(side, previous)) < 0.0:
            side = -side
        previous = side
        centers.append(0.5 * (positions[ca_i] + positions[ca_j]))
        sides.append(side)

    if not centers:
        return None
    # Run the ribbon out to the first and last Cα rather than stopping at the
    # midpoints, so a chain does not visibly lose half a residue at each end.
    first_ca = run[0].index("CA")
    last_ca = run[-1].index("CA")
    if first_ca is not None:
        centers.insert(0, positions[first_ca])
        sides.insert(0, sides[0])
    if last_ca is not None:
        centers.append(positions[last_ca])
        sides.append(sides[-1])
    return np.asarray(centers), np.asarray(sides)


def _mitred_segment(a: np.ndarray, b: np.ndarray, side: np.ndarray,
                    cap_start: np.ndarray, cap_end: np.ndarray,
                    half_w: float, half_t: float):
    """One ribbon segment from *a* to *b* as a set of half-spaces.

    The two end caps are *mitred*: their normals are the averaged tangents at
    the joints rather than this segment's own tangent, so a segment and its
    neighbour end on the identical plane. Square caps leave a wedge-shaped
    notch on the outside of every bend, which at six segments per residue turns
    the ribbon into a row of loose slats.
    """
    length = float(np.linalg.norm(b - a))
    if length < 1e-6:
        return None
    tangent = (b - a) / length
    side = side - tangent * float(np.dot(side, tangent))
    side = _unit(side, fallback=np.cross(tangent, (0.0, 0.0, 1.0)))
    up = _unit(np.cross(tangent, side))
    center = 0.5 * (a + b)

    normals = [side, -side, up, -up]
    offsets = [half_w, half_w, half_t, half_t]
    caps = []
    for cap, through, sign in ((cap_start, a, -1.0), (cap_end, b, 1.0)):
        n = _unit(cap, fallback=tangent) * sign
        # A joint sharper than about 60 degrees would mitre to a plane nearly
        # parallel to the ribbon, which is unbounded rather than merely ugly;
        # there, square off instead and accept a notch a spline this dense
        # never actually produces.
        if float(np.dot(n, tangent)) * sign < 0.5:
            n = tangent * sign
        caps.append((n, float(np.dot(n, through - center))))
    normals += [c[0] for c in caps]
    offsets += [c[1] for c in caps]

    # Walk the eight corners, each the intersection of a cap plane with one
    # (side, up) edge, to find how far the solid runs along the tangent. A
    # mitred corner overhangs half the segment length, so the enclosing box
    # cannot just be the nominal one.
    along = length * 0.5
    for s in (half_w, -half_w):
        for u in (half_t, -half_t):
            o = side * s + up * u
            for n, d in caps:
                denom = float(np.dot(n, tangent))
                if abs(denom) < 1e-6:
                    continue
                along = max(along, abs((d - float(np.dot(n, o))) / denom))
    axes = np.array([tangent, side, up])
    return (center, np.array(normals), np.array(offsets),
            axes, np.array([along, half_w, half_t]))


def _add_ribbon(builder: _Builder, run: Sequence[Residue], positions: np.ndarray) -> None:
    frames = _ribbon_frames(run, positions)
    if frames is None:
        return
    centers, sides = frames
    path = _catmull_rom(centers, RIBBON_SAMPLES)
    side_path = _catmull_rom(sides, RIBBON_SAMPLES)
    if len(path) < 2:
        return
    half_w, half_t = RIBBON_WIDTH * 0.5, RIBBON_THICKNESS * 0.5

    tangents = np.diff(path, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-9)
    # Joint j sits between segments j-1 and j; its mitre normal is the mean of
    # their tangents, and the two open ends just use the tangent they have.
    joints = np.vstack([tangents[0], tangents[:-1] + tangents[1:], tangents[-1]])

    for k in range(len(path) - 1):
        seg = _mitred_segment(path[k], path[k + 1],
                              side_path[k] + side_path[k + 1],
                              joints[k], joints[k + 1], half_w, half_t)
        if seg is None:
            continue
        center, normals, offsets, axes, half = seg
        builder.solid(center, normals, offsets, builder.pal.ribbon, axes, half)


def _add_plate(builder: _Builder, res: Residue, positions: np.ndarray):
    """An extruded ring-shaped plate for an aromatic side chain.

    Returns ``(center, reach)`` -- the glyph's anchor and how far the solid
    extends from it -- or None when the ring atoms are missing and the caller
    should fall back to a rounded volume.
    """
    names = AROMATIC_RINGS[res.name]
    idx = [res.atoms[n] for n in names if n in res.atoms]
    if len(idx) < 3:
        return None
    ring = positions[idx]
    center, normal, e1, e2 = plane_frame(ring)
    flat = np.column_stack([(ring - center) @ e1, (ring - center) @ e2])
    hull = convex_hull_2d(flat)
    if len(hull) < 3:
        return None

    loop = flat[hull]
    edges = np.roll(loop, -1, axis=0) - loop
    # The hull is counter-clockwise, so (dy, -dx) points out of it.
    out2d = np.column_stack([edges[:, 1], -edges[:, 0]])
    lengths = np.linalg.norm(out2d, axis=1)
    keep = lengths > 1e-9
    out2d = out2d[keep] / lengths[keep, None]
    side_normals = out2d[:, 0:1] * e1 + out2d[:, 1:2] * e2
    side_offsets = np.einsum("ij,ij->i", out2d, loop[keep]) + PLATE_INFLATE

    normals = np.vstack([side_normals, normal, -normal])
    offsets = np.concatenate([side_offsets,
                              [PLATE_THICKNESS * 0.5, PLATE_THICKNESS * 0.5]])
    # The inflated hull vertices sit further out than any edge plane's offset,
    # so bound on the vertices themselves.
    # Bound on the inflated outline's own corners, not on the ring atoms plus
    # the inflation: pushing two edges out and intersecting them moves a sharp
    # corner further than the inflation itself, and a box that assumed
    # otherwise would slice the plate's points off at its edges.
    corners = _edge_intersections(out2d, side_offsets)
    half = np.array([np.abs(corners[:, 0]).max(), np.abs(corners[:, 1]).max(),
                     PLATE_THICKNESS * 0.5])
    builder.solid(center, normals, offsets, builder.pal.plate,
                  np.array([e1, e2, normal]), half)
    return center, float(np.linalg.norm(half))


def _add_volume(builder: _Builder, res: Residue, positions: np.ndarray):
    """A rounded volume built from the side chain's own atom positions.

    A sphere per heavy atom, sized so that atoms a bond length apart merge into
    one smooth solid -- which makes the shape literally the geometry of the
    side chain rather than a stand-in for it.

    Returns ``(anchor, reach)`` like :func:`_add_plate`.
    """
    side = res.side_chain_indices()
    if side:
        for i in side:
            builder.sphere(positions[i], BLOB_RADIUS, builder.pal.volume)
        anchor = positions[side].mean(axis=0)
        reach = float(np.linalg.norm(positions[side] - anchor, axis=1).max()) + BLOB_RADIUS
        return anchor, reach

    # Glycine: nothing past Cα, so mark where its Cβ would be.
    n, ca, c = res.index("N"), res.index("CA"), res.index("C")
    if n is not None and ca is not None and c is not None:
        anchor = virtual_cbeta(positions[n], positions[ca], positions[c])
    else:
        anchor = positions[res.atoms["CA"]] + np.array([0.0, 0.0, 1.2])
    builder.sphere(anchor, GLYCINE_RADIUS, builder.pal.volume)
    return anchor, GLYCINE_RADIUS


def _node_positions(res: Residue, atom: int, role: str,
                    positions: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """Where a residue's hydrogen-bonding node(s) for one atom go.

    A pure donor or acceptor sits exactly on the atom's file coordinates. An
    atom that does both -- a hydroxyl, a histidine ring nitrogen, a thiol --
    cannot show two colours in one place, so its acceptor node keeps the exact
    coordinates and its donor node is set one node-width out along the atom's
    own bond axis.
    """
    p = positions[atom]
    if role != "both":
        return [(p, role)]
    others = [i for i in res.atoms.values() if i != atom]
    if others:
        d = positions[others] - p
        axis = _unit(-d[int(np.argmin(np.linalg.norm(d, axis=1)))])
    else:
        axis = np.array([0.0, 0.0, 1.0])
    return [(p, "acceptor"), (p + axis * NODE_SPLIT, "donor")]


def _add_nodes(builder: _Builder, residues: Sequence[Residue],
               positions: np.ndarray) -> None:
    for res in residues:
        for name, atom in res.atoms.items():
            role = hbond_role(res.name, name)
            if role is None:
                continue
            for point, kind in _node_positions(res, atom, role, positions):
                color = builder.pal.donor if kind == "donor" else builder.pal.acceptor
                builder.sphere(point, NODE_RADIUS, color)


def _add_links(builder: _Builder, residues: Sequence[Residue],
               positions: np.ndarray) -> None:
    """Hairlines between backbone amides that are close enough to be bonded.

    Distance alone, on the heavy atoms -- the files this draws usually have no
    hydrogens, so there is no N–H···O angle to test. Neighbouring residues are
    excluded because their backbone N and O are within range no matter what
    the structure is doing.
    """
    donors = [(k, res.atoms["N"]) for k, res in enumerate(residues)
              if "N" in res.atoms and hbond_role(res.name, "N") == "donor"]
    acceptors = [(k, res.atoms["O"]) for k, res in enumerate(residues)
                 if "O" in res.atoms]
    if not donors or not acceptors:
        return
    d_idx = np.array([a for _k, a in donors])
    a_idx = np.array([a for _k, a in acceptors])
    d_res = np.array([k for k, _a in donors])
    a_res = np.array([k for k, _a in acceptors])
    dist = np.linalg.norm(positions[d_idx][:, None, :] - positions[a_idx][None, :, :], axis=2)
    close = (dist < LINK_MAX_DISTANCE) & (
        np.abs(d_res[:, None] - a_res[None, :]) >= LINK_MIN_SEPARATION)
    for i, j in zip(*np.nonzero(close)):
        builder.cylinder(positions[d_idx[i]], positions[a_idx[j]],
                         LINK_RADIUS, builder.pal.link)


# -- entry point ----------------------------------------------------------

_CACHE: List[Tuple[tuple, GlyphScene]] = []
_CACHE_LIMIT = 4


def build_scene(molecule: Molecule, theme: str = "dark") -> Optional[GlyphScene]:
    """Glyph geometry for *molecule*, or None if it is not a protein.

    None means "this skin has nothing to say about this molecule" -- a file
    with no residue names (xyz, mol, sdf) or with no amino acids in it. The
    renderer falls back to ball-and-stick on None rather than drawing an empty
    frame.
    """
    residues = protein_residues(molecule)
    if not residues:
        return None

    positions = molecule.positions
    pal = palette(theme)
    builder = _Builder(pal)

    for run in chain_runs(residues, positions):
        _add_ribbon(builder, run, positions)

    for res in residues:
        ca = res.atoms["CA"]
        shape = _add_plate(builder, res, positions) if res.is_aromatic else None
        if shape is None:
            shape = _add_volume(builder, res, positions)
        anchor, reach = shape
        builder.cylinder(positions[ca], anchor, STICK_RADIUS, pal.stick)
        # The letter is lifted by the whole reach of its own solid, so it lands
        # on the near surface from every camera angle. Biasing by a thickness
        # or a single sphere radius is not enough: a tilted plate and a long
        # side chain both put geometry well in front of their own anchor, and
        # the letter comes out chewed into an unreadable shape.
        builder.label(anchor, res.letter, LETTER_SIZE, reach + 0.05, pal.ink)

    _add_nodes(builder, residues, positions)
    _add_links(builder, residues, positions)
    return builder.freeze()


def _cache_key(molecule: Molecule, theme: str) -> tuple:
    p = molecule.positions
    # Cheap and specific: identity plus a checksum of the coordinates, so an
    # edit or a new frame invalidates but a pure camera move does not.
    return (id(molecule), molecule.n_atoms, str(theme),
            float(p.sum()) if p.size else 0.0,
            float(p[-1, 0]) if p.size else 0.0)


def cached_scene(molecule: Molecule, theme: str = "dark") -> Optional[GlyphScene]:
    """:func:`build_scene`, memoized across frames.

    A spin or an orbit re-renders the same geometry many times a second;
    rebuilding splines and hulls each time would dominate the frame.
    """
    key = _cache_key(molecule, theme)
    for k, scene in _CACHE:
        if k == key:
            return scene
    scene = build_scene(molecule, theme)
    _CACHE.append((key, scene))
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.pop(0)
    return scene
