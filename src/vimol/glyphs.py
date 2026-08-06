"""Build the geometry behind the two protein styles: `ribbon` and `glyph`.

This module turns a protein into drawable primitives and nothing else -- it
never touches a framebuffer. Everything it emits is in world coordinates, so
the renderer only has to rotate it into view space. ``ribbon_only`` stops after
the backbone; the rest is what the lettered skin adds to it.

What it produces:

``spheres``     side-chain volumes, Cα/Cβ beads, and atoms drawn as themselves
``cylinders``   the rods out to each glyph, and real bonds
``polyhedra``   ribbon segments and ring tablets, as sets of half-spaces
``labels``      a residue's code and number, printed onto one of its faces
``mesh``        the same ribbon and tablets as triangles, for the GPU

A convex solid is stored as a center plus planes ``n·(p − center) ≤ d``. Keeping
the offsets relative to a center makes them invariant under the camera's
rotation, so a frame only has to rotate the normals -- one matmul for every
plane in the scene at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import glyph_font, glyph_mesh
from .bonds import perceive_bonds
from .molecule import Molecule
from .residues import (RING_ATOMS, Residue, chain_runs, protein_residues,
                       residue_class)


# -- tunables ------------------------------------------------------------
RIBBON_WIDTH = 2.0          # angstrom, full width across the ribbon
RIBBON_THICKNESS = 0.26     # angstrom, full thickness
RIBBON_SAMPLES = 16         # spline samples per residue. A helix turns every
                            # 3.6 residues, so a coarse sampling reproduces its
                            # coil as a polygon -- the kinks read as crinkle
                            # even though the path through the Ca atoms is
                            # exactly right.
# Passes of neighbour-averaging over the ribbon's side vectors. The path stays
# on the Cα atoms; this is only the ribbon's rotation about its own length. Per
# peptide plane that rotation swings by about a hundred degrees along a helix,
# and following it literally is what makes a ribbon read as crinkled -- the
# turns are real but they are not what anyone is looking at.
SIDE_SMOOTHING = 6
PLATE_INFLATE = 0.30        # angstrom the ring hull is pushed out by
PLATE_THICKNESS = 0.42      # angstrom, full thickness
BLOB_RADIUS = 0.92          # angstrom; well over half a 1.5 A bond, so bonded
                            # atoms' spheres run together with a soft waist --
                            # a snowman rather than a string of beads
GLYCINE_RADIUS = 0.55       # glycine has no side chain, so its marker is small
LETTER_SIZE = 0.78          # angstrom, cap height of the one-letter code
# Fraction of a sphere's radius the whole label run may span once wrapped onto
# it. Past about this the text reaches the horizon and folds out of sight.
LETTER_SPAN = 1.45
# The bare ribbon's colour: PyMOL's cartoon green, which is what a reader of
# protein pictures expects a backbone trace to look like.
RIBBON_GREEN = (0.13, 0.78, 0.15)
BEAD_RADIUS = 0.20          # the Cβ bead the rod passes through
# The Cα bead is bigger than the ribbon is thick, so it swells out of the
# backbone instead of sitting on it: the rod then grows out of the ribbon
# rather than being planted in it.
BASE_RADIUS = 0.32
ROD_RADIUS = 0.075
JOIN_OVERLAP = 0.04     # angstrom each ribbon segment runs past its joint
# `box_planes` emits side, -side, up, -up: the third plane is the first of the
# two large faces, which are the ones a ribbon shades smoothly across.
SMOOTH_FACE = 2


@dataclass(frozen=True)
class Palette:
    ribbon: Tuple[float, float, float]
    plate: Tuple[float, float, float]
    volume: Tuple[float, float, float]
    ink: Tuple[float, float, float]
    # Cα/Cβ beads and the rods between them, keyed by residue class, so a
    # glance separates the acidic from the aromatic from the merely greasy.
    beads: Dict[str, Tuple[float, float, float]]


# The letters are always dark, so both palettes keep the side-chain solids
# light; only the ribbon flips with the background. Atoms drawn as themselves
# keep their element colours and are not in here.
LIGHT = Palette(
    # Warm graphite rather than black: it still reads as the one dark mass in
    # the picture, but its shading has somewhere to go.
    ribbon=(0.16, 0.16, 0.18),
    # Brass and bone. The two solids are the subject, so they take the only
    # two saturated-ish values in the scheme and sit a clear step apart.
    plate=(0.74, 0.54, 0.23), volume=(0.89, 0.86, 0.79),
    ink=(0.11, 0.11, 0.13),
    # The beads and rods are fittings, not subject. They share one lightness so
    # they read as a family, and stay muted enough not to compete with the
    # tablets -- but light enough to be visible where they land, which is on
    # the graphite ribbon.
    beads={"aromatic": (0.56, 0.43, 0.22), "acidic": (0.58, 0.31, 0.27),
           "basic": (0.33, 0.44, 0.63), "polar": (0.29, 0.49, 0.45),
           "hydrophobic": (0.45, 0.42, 0.38)},
)
DARK = Palette(
    ribbon=(0.78, 0.80, 0.84),
    plate=(0.82, 0.60, 0.26), volume=(0.87, 0.84, 0.77),
    ink=(0.11, 0.11, 0.13),
    beads={"aromatic": (0.80, 0.62, 0.32), "acidic": (0.80, 0.44, 0.38),
           "basic": (0.48, 0.62, 0.88), "polar": (0.40, 0.68, 0.62),
           "hydrophobic": (0.66, 0.62, 0.56)},
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
    cyl_color: np.ndarray          # (C, 3) colour at the `a` end
    cyl_color_b: np.ndarray        # (C, 3) colour at the `b` end; a bond splits at the middle
    cyl_flat: np.ndarray           # (C,) bool
    sphere_flat: np.ndarray        # (S,) bool
    poly_center: np.ndarray        # (P, 3)
    poly_color: np.ndarray         # (P, 3)
    poly_flat: np.ndarray          # (P,) bool
    poly_axes: np.ndarray          # (P, 3, 3) orthonormal rows of the enclosing box
    poly_half: np.ndarray          # (P, 3) half-extent along each of those axes
    poly_smooth: np.ndarray        # (P, 2, 3) joint normals to shade between
    poly_smooth_span: np.ndarray   # (P,) half-length to interpolate those over
    poly_smooth_face: np.ndarray   # (P,) index of the first smooth-shaded plane, or -1
    poly_slice: np.ndarray         # (P + 1,) int, CSR bounds into the plane arrays
    plane_normal: np.ndarray       # (M, 3) unit, world space
    plane_offset: np.ndarray       # (M,)
    label_center: np.ndarray       # (L, 3)
    label_size: np.ndarray         # (L,) glyph height in angstrom
    label_bias: np.ndarray         # (L,) angstrom to lift the letter toward the camera
    label_color: np.ndarray        # (L, 3)
    label_flat: np.ndarray         # (L,) bool
    label_char: List[str]
    label_number: List[str]        # residue number, drawn small beside the code
    # True where the letter is already printed onto a tablet face in `mesh`.
    # The raycaster billboards every label; the GPU billboards only these.
    label_on_tablet: np.ndarray    # (L,) bool
    label_normal: np.ndarray       # (L, 3) outward normal of the face it is on
    label_down: np.ndarray         # (L, 3) in-plane direction the letter stands on
    label_offset: np.ndarray       # (L,) how far off the face to print it
    # Colour of the surface the letter is printed on. A printed letter blends
    # ink into its face, so the quad has to carry that face's colour; the ink
    # itself is a uniform. Getting this wrong paints the whole quad solid.
    label_surface: np.ndarray      # (L, 3)
    # The same ribbon and tablets as real geometry, for the GPU path. The
    # raycaster ignores it and intersects the half-spaces above instead.
    mesh: "glyph_mesh.GlyphMesh"

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
        self.poly_flat: List[bool] = []
        self.poly_axes: List[np.ndarray] = []
        self.poly_half: List[np.ndarray] = []
        self.poly_smooth: List[Tuple[np.ndarray, np.ndarray]] = []
        self.poly_smooth_span: List[float] = []
        self.poly_smooth_face: List[int] = []
        self.poly_counts: List[int] = []
        self.normals: List[np.ndarray] = []
        self.offsets: List[np.ndarray] = []
        self.labels: List[Tuple[np.ndarray, str, float, float, Tuple[float, float, float]]] = []
        self.mesh = glyph_mesh.MeshBuilder()

    def sphere(self, center, radius, color, flat: bool = False) -> None:
        self.spheres.append((np.asarray(center, float), float(radius), color, flat))

    def cylinder(self, a, b, radius, color, color_b=None, flat: bool = False) -> None:
        self.cylinders.append((np.asarray(a, float), np.asarray(b, float),
                               float(radius), color, color if color_b is None else color_b,
                               flat))

    def solid(self, center, normals, offsets, color, axes, half,
              flat: bool = False, smooth=None) -> None:
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
        self.poly_flat.append(flat)
        self.poly_axes.append(np.asarray(axes, float))
        self.poly_half.append(np.asarray(half, float))
        self.poly_counts.append(len(normals))
        self.normals.append(normals)
        self.offsets.append(np.asarray(offsets, float))
        # `smooth` is (n_start, n_end, half_length): the pair of joint normals
        # to interpolate between, and the half-length to interpolate over. The
        # faces it applies to are the third and fourth planes, the two large
        # ones (`box_planes` order: side, -side, up, -up).
        #
        # The half-length is NOT the enclosing box's half-extent along the same
        # axis -- a mitred corner overhangs, so that one is larger. Scaling by
        # it would leave the interpolation short of 0 and 1 at the real ends,
        # the normals either side of a joint would disagree, and every joint
        # would draw itself as a thin dark seam across the ribbon.
        self.poly_smooth.append(smooth[:2] if smooth is not None
                                else (np.zeros(3), np.zeros(3)))
        self.poly_smooth_span.append(smooth[2] if smooth is not None else 1.0)
        self.poly_smooth_face.append(SMOOTH_FACE if smooth is not None else -1)

    def label(self, center, char, number, size, bias, color, flat: bool = False,
              on_tablet: bool = False, normal=None, down=None,
              offset: float = 0.0, surface=None) -> None:
        self.labels.append((np.asarray(center, float), char, float(size), float(bias),
                            color, flat, on_tablet,
                            np.zeros(3) if normal is None else np.asarray(normal, float),
                            np.zeros(3) if down is None else np.asarray(down, float),
                            float(offset),
                            np.asarray(color if surface is None else surface, float),
                            str(number)))

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
            sphere_flat=stack([s[3] for s in self.spheres], 0, bool),
            cyl_a=stack([c[0] for c in self.cylinders], 3),
            cyl_b=stack([c[1] for c in self.cylinders], 3),
            cyl_radius=stack([c[2] for c in self.cylinders], 0),
            cyl_color=stack([c[3] for c in self.cylinders], 3),
            cyl_color_b=stack([c[4] for c in self.cylinders], 3),
            cyl_flat=stack([c[5] for c in self.cylinders], 0, bool),
            poly_center=stack(self.poly_center, 3),
            poly_color=stack(self.poly_color, 3),
            poly_flat=stack(self.poly_flat, 0, bool),
            poly_axes=(np.asarray(self.poly_axes, float) if self.poly_axes
                       else np.zeros((0, 3, 3))),
            poly_half=stack(self.poly_half, 3),
            poly_smooth=(np.asarray(self.poly_smooth, float) if self.poly_smooth
                         else np.zeros((0, 2, 3))),
            poly_smooth_span=stack(self.poly_smooth_span, 0),
            poly_smooth_face=stack(self.poly_smooth_face, 0, np.int64),
            poly_slice=slices,
            plane_normal=(np.concatenate(self.normals) if self.normals
                          else np.zeros((0, 3))),
            plane_offset=(np.concatenate(self.offsets) if self.offsets
                          else np.zeros((0,))),
            label_center=stack([l[0] for l in self.labels], 3),
            label_size=stack([l[2] for l in self.labels], 0),
            label_bias=stack([l[3] for l in self.labels], 0),
            label_color=stack([l[4] for l in self.labels], 3),
            label_flat=stack([l[5] for l in self.labels], 0, bool),
            label_char=[l[1] for l in self.labels],
            label_number=[l[11] for l in self.labels],
            label_on_tablet=np.array([l[6] for l in self.labels], bool),
            label_normal=stack([l[7] for l in self.labels], 3),
            label_down=stack([l[8] for l in self.labels], 3),
            label_offset=stack([l[9] for l in self.labels], 0),
            label_surface=stack([l[10] for l in self.labels], 3),
            mesh=self.mesh.freeze(),
        )


# -- small geometry helpers ----------------------------------------------

def _in_plane(v, normal):
    """*v* with its component along *normal* removed, unit -- None if that
    leaves nothing, i.e. *v* was along the normal to begin with."""
    flat = np.asarray(v, float) - np.asarray(normal, float) * float(
        np.dot(v, normal))
    length = float(np.linalg.norm(flat))
    return flat / length if length > 1e-6 else None


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

    The guide points are the Cα atoms themselves. Routing the spline through
    the peptide-plane midpoints instead would smooth it slightly more, but it
    lifts the ribbon off every Cα by up to an angstrom, and each residue's
    glyph link has to start exactly where its Cα is -- otherwise the link
    visibly launches from beside the ribbon rather than out of it.

    Carson-Bugg: between consecutive Cα atoms the peptide plane fixes a natural
    "side" direction -- the carbonyl, squared up against the chain. That
    direction flips by nearly 180° from one residue to the next in a β-strand
    (the carbonyls alternate), so each is negated when it opposes its
    predecessor -- without that the ribbon corkscrews once per residue. Each
    residue then averages the peptide planes on either side of it.
    """
    peptide: List[np.ndarray] = []
    previous: Optional[np.ndarray] = None
    for i in range(len(run) - 1):
        ca_i, ca_j = run[i].index("CA"), run[i + 1].index("CA")
        o_i = run[i].index("O")
        a = positions[ca_j] - positions[ca_i]
        if o_i is None:
            side = previous if previous is not None else _unit(np.cross(a, (0.0, 0.0, 1.0)))
        else:
            # The width runs along the carbonyl, squared up against the chain --
            # Carson-Bugg's D = (A x B) x A, which reduces to the component of
            # B perpendicular to A. Using A x B itself, the peptide plane's
            # *normal*, turns the ribbon through ninety degrees: a helix then
            # presents its edge to the outside and reads as a narrow coil
            # instead of the flat spiral a cartoon should show.
            b = positions[o_i] - positions[ca_i]
            side = _unit(b - a * (float(np.dot(a, b)) / float(np.dot(a, a))))
        if previous is not None and float(np.dot(side, previous)) < 0.0:
            side = -side
        previous = side
        peptide.append(side)

    if not peptide:
        return None
    centers = np.array([positions[r.atoms["CA"]] for r in run])
    sides = np.array([_unit(sum(peptide[j] for j in (i - 1, i)
                                if 0 <= j < len(peptide)))
                      for i in range(len(run))])
    return centers, _smooth_sides(sides)


def _smooth_sides(sides: np.ndarray) -> np.ndarray:
    """Relax the ribbon's twist without moving where the ribbon goes.

    A few passes of a three-tap average, renormalized each time, with the two
    ends held. The guide points are untouched, so the ribbon still runs through
    every Cα -- only how fast it rolls about its own length changes.
    """
    if len(sides) < 3:
        return sides
    out = sides.astype(float).copy()
    for _ in range(SIDE_SMOOTHING):
        blended = out.copy()
        blended[1:-1] = out[:-2] + 2.0 * out[1:-1] + out[2:]
        norm = np.linalg.norm(blended, axis=1, keepdims=True)
        out = np.where(norm > 1e-9, blended / np.maximum(norm, 1e-9), out)
    return out


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
        # Push each cap a hair past the joint so neighbouring segments overlap.
        # Ending exactly on the shared plane is right geometrically and wrong
        # on screen: at the joint pixels the two solids tie on depth, and
        # whichever cap wins is a rectangle seen edge-on, so every joint draws
        # itself as a thin dark line across the ribbon. Buried inside the
        # neighbour, the caps never win.
        caps.append((n, float(np.dot(n, through - center)) + JOIN_OVERLAP))
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
            axes, np.array([along, half_w, half_t]), length * 0.5)


def _add_ribbon(builder: _Builder, run: Sequence[Residue], positions: np.ndarray,
                color, flat: bool) -> None:
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

    # Each segment's own face normal, and the average of the two meeting at
    # each joint. Shading interpolates between the joint normals across a
    # segment, so it is continuous where segments meet and the ribbon reads as
    # one curved band instead of a run of flat facets catching the light in
    # steps. This is vertex-normal averaging; the geometry stays faceted, only
    # the shading is smooth.
    # The cross product drops the tangent component, so this is exactly the
    # `up` axis _mitred_segment derives for the same segment.
    faces = np.array([_unit(np.cross(tangents[k], side_path[k] + side_path[k + 1]))
                      for k in range(len(tangents))])
    joint_faces = np.vstack([faces[0], faces[:-1] + faces[1:], faces[-1]])
    joint_faces /= np.maximum(np.linalg.norm(joint_faces, axis=1, keepdims=True), 1e-9)

    for k in range(len(path) - 1):
        seg = _mitred_segment(path[k], path[k + 1],
                              side_path[k] + side_path[k + 1],
                              joints[k], joints[k + 1], half_w, half_t)
        if seg is None:
            continue
        center, normals, offsets, axes, half, half_length = seg
        builder.solid(center, normals, offsets, color, axes, half, flat=flat,
                      smooth=(joint_faces[k], joint_faces[k + 1], half_length))

    glyph_mesh.ribbon(builder.mesh, path, side_path, half_w, half_t, color, flat)


def _add_plate(builder: _Builder, res: Residue, positions: np.ndarray,
               color, flat: bool):
    """An extruded tablet cut to the shape of a side chain's ring.

    Returns ``(anchor, reach, normal, label_at)`` -- where the rod ends, how far
    the solid extends, its face normal, and the point on it a letter should be
    centred on -- or None when the ring atoms are missing and the caller should
    fall back to a rounded volume.
    """
    names = RING_ATOMS[res.name]
    idx = [res.atoms[n] for n in names if n in res.atoms]
    if len(idx) < 3:
        return None
    ring = positions[idx]
    center, normal, e1, e2 = plane_frame(ring)
    # Proline's ring closes back onto the backbone, so its true centroid lands
    # on the ribbon and the tablet would be half-buried in it with nowhere for
    # the rod to go. Slide the tablet out to the centre of the side chain
    # proper, keeping the ring's real outline and its real plane.
    carbons = res.side_chain_carbons()
    if not res.is_aromatic and carbons:
        center = positions[carbons].mean(axis=0)
    planar = np.column_stack([(ring - center) @ e1, (ring - center) @ e2])
    hull = convex_hull_2d(planar)
    if len(hull) < 3:
        return None

    loop = planar[hull]
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
    builder.solid(center, normals, offsets, color,
                  np.array([e1, e2, normal]), half, flat=flat)
    glyph_mesh.tablet(builder.mesh, center, e1, e2, normal, corners,
                      PLATE_THICKNESS * 0.5, color, flat)
    # The letter belongs at the middle of the plaque, which is the middle of the
    # outline -- not at `center`, which is only the origin the planes are
    # measured from and, for proline, is deliberately off to one side.
    mid = corners.mean(axis=0)
    label_at = center + mid[0] * e1 + mid[1] * e2
    return center, float(np.linalg.norm(half)), normal, label_at


def _add_volume(builder: _Builder, res: Residue, positions: np.ndarray,
                color, flat: bool):
    """A rounded volume built from the side chain's own carbon positions.

    A sphere per carbon, sized so that atoms a bond length apart merge into one
    smooth solid -- which makes the shape literally the geometry of the side
    chain rather than a stand-in for it. Only the carbons: the nitrogens,
    oxygens and sulfurs hanging off the skeleton are drawn as themselves, and
    swallowing them into the blob would be the one thing that hides them.

    Returns ``(anchor, reach, spheres)`` -- the glyph's anchor, how far the
    solid extends from it, and the spheres themselves, which the letter needs:
    a blob is not a ball centred on its anchor, so a letter placed from the
    anchor sinks into whichever lobe happens to lie in front of it.
    """
    side = res.side_chain_carbons()
    if side:
        for i in side:
            builder.sphere(positions[i], BLOB_RADIUS, color, flat)
        anchor = positions[side].mean(axis=0)
        reach = float(np.linalg.norm(positions[side] - anchor, axis=1).max()) + BLOB_RADIUS
        return anchor, reach, [(positions[i], BLOB_RADIUS) for i in side]

    # Glycine: nothing past Cα, so mark where its Cβ would be.
    n, ca, c = res.index("N"), res.index("CA"), res.index("C")
    if n is not None and ca is not None and c is not None:
        anchor = virtual_cbeta(positions[n], positions[ca], positions[c])
    else:
        anchor = positions[res.atoms["CA"]] + np.array([0.0, 0.0, 1.2])
    builder.sphere(anchor, GLYCINE_RADIUS, color, flat)
    return anchor, GLYCINE_RADIUS, [(anchor, GLYCINE_RADIUS)]


def _add_atoms(builder: _Builder, residues: Sequence[Residue], molecule: Molecule,
               bonds, colors: np.ndarray, radii: np.ndarray, flat: np.ndarray,
               bond_radius: float) -> None:
    """Draw every side-chain atom the glyphs do not stand for as itself.

    A ribbon stands for the backbone and a glyph solid for its side chain's
    carbon skeleton. What is left over -- a carboxylate, a hydroxyl, the indole
    N–H, a thiol -- is drawn as a real atom in its element colour, bonded to
    what it is bonded to, at the exact coordinates in the file. Those groups
    are where the chemistry is, and an abstract shape is the wrong thing to put
    over them.

    Nothing tests whether an atom is inside its own solid: the ones a shape
    swallows are hidden by the z-buffer, which is cheaper and more honest than
    a rule about which groups matter.
    """
    symbols = molecule.symbols
    polar = {i for res in residues for i in res.side_chain_polar()}
    if not polar:
        return
    # Hydrogens only where they say something the heavy atom does not: on a
    # hydroxyl or an amide, which is what makes it a donor. A structure solved
    # by NMR carries every C–H as well, and drawing those buries the skin under
    # a hundred white spheres -- they belong to the carbon skeleton the glyph
    # already stands for.
    drawn = set(polar)
    for a, b, _order in bonds:
        for h, heavy in ((a, b), (b, a)):
            if symbols[h].strip().upper() == "H" and heavy in polar:
                drawn.add(h)

    for i in sorted(drawn):
        builder.sphere(molecule.positions[i], float(radii[i]), colors[i], bool(flat[i]))
    for a, b, _order in bonds:
        # A bond is worth drawing when it holds a drawn atom on: to another
        # drawn atom, or back to the carbon skeleton it hangs off.
        both = a in drawn and b in drawn
        anchored = ((a in polar and symbols[b].strip().upper() == "C")
                    or (b in polar and symbols[a].strip().upper() == "C"))
        if not (both or anchored):
            continue
        builder.cylinder(molecule.positions[a], molecule.positions[b], bond_radius,
                         colors[a], colors[b], bool(flat[a]))


def _add_link_to_glyph(builder: _Builder, res: Residue, positions: np.ndarray,
                       anchor: np.ndarray, color, flat: bool) -> None:
    """Cα bead → rod → Cβ bead → rod → glyph.

    The Cβ is a bead on the way out rather than a point the rod passes through:
    it is a real atom of the residue and reads better shown than skewered.
    """
    ca = positions[res.atoms["CA"]]
    # Wider than the ribbon is thick, so it reads as a swelling of the backbone
    # the rod grows out of rather than a bead parked on top of it.
    builder.sphere(ca, BASE_RADIUS, color, flat)
    cb = res.index("CB")
    waypoints = [ca] if cb is None else [ca, positions[cb]]
    if cb is not None:
        builder.sphere(positions[cb], BEAD_RADIUS, color, flat)
    for start, end in zip(waypoints, waypoints[1:] + [anchor]):
        if float(np.linalg.norm(end - start)) > 1e-6:
            builder.cylinder(start, end, ROD_RADIUS, color, flat=flat)


# -- entry point ----------------------------------------------------------

_CACHE: List[Tuple[tuple, GlyphScene]] = []
_CACHE_LIMIT = 4


def build_scene(molecule: Molecule, theme: str = "dark", *,
                atom_colors=None, atom_radii=None, flat_mask=None,
                bond_radius: float = 0.10,
                ribbon_only: bool = False) -> Optional[GlyphScene]:
    """Glyph geometry for *molecule*, or None if it is not a protein.

    None means "this skin has nothing to say about this molecule" -- nothing in
    it that walks like a peptide. A file with no names at all still works: the
    backbone and the residue identities are inferred from elements and
    connectivity (see ``residues.infer_residues``). The renderer falls back to
    ball-and-stick on None rather than drawing an empty frame.

    ``atom_colors``/``atom_radii`` are how atoms drawn as themselves get their
    element colouring, so the skin inherits the light/dark palette and any hover
    tint without knowing about either. ``flat_mask`` marks the atoms of overlaid
    structures other than the main one: those residues render in their entry's
    flat tint, exactly as the other representations do, so an overlay stays
    readable as "this one is the subject, those are the comparison".
    """
    residues = protein_residues(molecule)
    if not residues:
        return None

    positions = molecule.positions
    colors = (np.asarray(atom_colors, float) if atom_colors is not None
              else molecule.element_colors())
    radii = (np.asarray(atom_radii, float) if atom_radii is not None
             else molecule.vdw_radii() * 0.25)
    flat = (np.asarray(flat_mask, bool) if flat_mask is not None
            else np.zeros(molecule.n_atoms, bool))
    pal = palette(theme)
    builder = _Builder(pal)

    def tint(res: Residue, own):
        """A tinted overlay entry paints every glyph in its own flat colour;
        the main frame keeps the skin's palette."""
        ca = res.atoms["CA"]
        return (colors[ca], True) if flat[ca] else (own, False)

    for run in chain_runs(residues, positions):
        color, is_flat = tint(run[0], RIBBON_GREEN if ribbon_only else pal.ribbon)
        _add_ribbon(builder, run, positions, color, is_flat)

    if ribbon_only:
        # The backbone trace on its own: no side chains, no letters, nothing
        # else to read. Everything below is what the glyph skin adds to it.
        return builder.freeze()

    for res in residues:
        ca = res.atoms["CA"]
        solid_color, is_flat = tint(res, pal.plate if res.is_aromatic else pal.volume)
        size = LETTER_SIZE if res.is_ring else LETTER_SIZE * 0.85
        shape = (_add_plate(builder, res, positions, solid_color, is_flat)
                 if res.is_ring else None)
        on_tablet = shape is not None
        if on_tablet:
            anchor, reach, face, label_at = shape
        else:
            anchor, reach, blobs = _add_volume(builder, res, positions,
                                               tint(res, pal.volume)[0], is_flat)
            face = label_at = None
        bead_color, _ = tint(res, pal.beads[residue_class(res.letter)])
        _add_link_to_glyph(builder, res, positions, anchor, bead_color, is_flat)

        # The letter is printed onto one flat face of the residue and stays
        # there: turn the structure and it foreshortens, then goes away, the
        # same as any other marking on a real object would. A tablet's face is
        # its own plane; a rounded volume has none, so it gets the plane facing
        # away from the backbone -- which is the side you can see when the
        # residue is pointing at you.
        curve = 0.0
        if on_tablet:
            normal, offset = face, PLATE_THICKNESS * 0.5 + glyph_mesh.LETTER_LIFT
            anchor = label_at
        else:
            normal = _unit(anchor - positions[ca], (0.0, 0.0, 1.0))
            # Print on the lobe that sticks out furthest that way, and wrap the
            # letter around it. Measuring from the anchor instead buries the
            # letter in whichever lobe happens to lie in front of it -- and on
            # glycine, whose anchor *is* its only sphere, inside that.
            centre, curve = max(blobs, key=lambda b: float(np.dot(b[0], normal))
                                + b[1])
            anchor = centre
            offset = curve + glyph_mesh.LETTER_LIFT
            # Wrapped text that reaches past the ball's horizon folds under
            # and disappears, so the whole run -- code and number together --
            # has to fit inside it. Scale on the run's width, not the cap
            # height: "G10" is more than twice as wide as it is tall, which is
            # exactly the case that overflowed a small glycine marker.
            span = glyph_font.run_width(res.letter, res.key[1], size)
            size *= min(1.0, curve * LETTER_SPAN / max(span, 1e-6))
        # Stand the letter on the stem where there is one to stand on; a volume
        # puts its stem straight through the face, so those line up along the
        # chain instead.
        down = _in_plane(positions[ca] - anchor, normal)
        if down is None:
            axis = positions[res.index("C", "CA")] - positions[res.index("N", "CA")]
            down = _in_plane(axis, normal)
        if down is None:
            down = _unit(np.cross(normal, (0.0, 0.0, 1.0)))
        # A tablet is printed on both faces -- one letter per side, each
        # handed for the side it is read from -- so you can read it from
        # wherever you are without it ever leaving the surface. A volume has
        # only the one face, pointing away from the backbone.
        faces = ((1.0, -1.0) if on_tablet else (1.0,))
        for side in faces:
            glyph_mesh.label(builder.mesh, anchor + normal * (side * offset),
                             np.cross(normal * side, down), down, normal * side,
                             res.letter, res.key[1], size, solid_color, is_flat,
                             curve=curve)
        builder.label(anchor, res.letter, res.key[1], size, reach + 0.05,
                      pal.ink, is_flat, on_tablet=on_tablet, normal=normal,
                      down=down, offset=offset, surface=solid_color)

    # Which hydrogens to keep and which bonds to draw both need connectivity.
    # The render path always perceives before drawing, but a caller building a
    # scene straight from a freshly parsed molecule has none yet, and would
    # silently get no bonds and no hydroxyl hydrogens.
    bonds = molecule.bonds or perceive_bonds(molecule)
    _add_atoms(builder, residues, molecule, bonds, colors, radii, flat, bond_radius)
    return builder.freeze()


def glyph_scene_for(molecule: Molecule, style) -> Optional[GlyphScene]:
    """The cached glyph scene a :class:`render.Style` asks for.

    Every caller goes through here -- the renderer, the camera's fit, and the
    widget's "is there anything to draw?" check. They must agree on the whole
    argument list or one of them populates the cache with a scene the others
    would not have built, and whichever asks first wins.
    """
    return cached_scene(
        molecule, style.glyph_theme,
        ribbon_only=style.representation == "ribbon",
        atom_colors=(style.color_override if style.color_override is not None
                     else molecule.element_colors()),
        atom_radii=molecule.vdw_radii() * style.atom_scale,
        flat_mask=style.flat_mask,
        bond_radius=style.bond_radius,
    )


def _digest(array) -> tuple:
    """A cheap, order-sensitive summary of an array for the cache key."""
    if array is None:
        return ()
    a = np.asarray(array, float)
    return (a.shape, float(a.sum()), float(a.ravel()[-1]) if a.size else 0.0)


def _cache_key(molecule: Molecule, theme: str, atom_colors, flat_mask,
               ribbon_only: bool) -> tuple:
    p = molecule.positions
    # Cheap and specific: identity plus a checksum of the coordinates, so an
    # edit or a new frame invalidates but a pure camera move does not. Colour
    # and flatness are geometry inputs too now -- marking a structure into an
    # overlay changes neither the atom count nor a coordinate, so without them
    # the cache would keep serving the untinted scene.
    return (id(molecule), molecule.n_atoms, str(theme),
            float(p.sum()) if p.size else 0.0,
            float(p[-1, 0]) if p.size else 0.0,
            _digest(atom_colors), _digest(flat_mask), bool(ribbon_only))


def cached_scene(molecule: Molecule, theme: str = "dark", *,
                 atom_colors=None, atom_radii=None, flat_mask=None,
                 bond_radius: float = 0.10,
                 ribbon_only: bool = False) -> Optional[GlyphScene]:
    """:func:`build_scene`, memoized across frames.

    A spin or an orbit re-renders the same geometry many times a second;
    rebuilding splines and hulls each time would dominate the frame.
    """
    key = _cache_key(molecule, theme, atom_colors, flat_mask, ribbon_only)
    for k, scene in _CACHE:
        if k == key:
            return scene
    scene = build_scene(molecule, theme, atom_colors=atom_colors,
                        atom_radii=atom_radii, flat_mask=flat_mask,
                        bond_radius=bond_radius, ribbon_only=ribbon_only)
    _CACHE.append((key, scene))
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.pop(0)
    return scene
