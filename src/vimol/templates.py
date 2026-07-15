"""Geometry templates for interactive molecule building.

For each ``(element, valence)`` combination we record the *ideal* local
arrangement of bonds around a central atom -- tetrahedral for four-coordinate
carbon, pyramidal for three-coordinate nitrogen, bent for two-coordinate
oxygen, and so on. Each arrangement is a small set of unit bond directions in a
canonical local frame.

The editor uses these to grow chemically-sensible fragments: place a central
atom, orient its first direction toward the atom it attaches to, and cap the
remaining directions with hydrogens.

Everything here is pure geometry (unit vectors and rotations); actual atom
placement, bond lengths and molecule mutation live in :mod:`vimol.editor`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import rotation_from_axis_angle
from .elements import normalize_symbol as _normalize_symbol


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perpendicular(a: np.ndarray) -> np.ndarray:
    """Return an arbitrary unit vector perpendicular to *a*."""
    a = _normalize(a)
    # cross with whichever cardinal axis is least aligned with a
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _normalize(np.cross(a, ref))


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector *a* onto unit vector *b*."""
    a = _normalize(a)
    b = _normalize(b)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-9:                       # parallel or antiparallel
        if c > 0:
            return np.eye(3)
        return rotation_from_axis_angle(_perpendicular(a), np.pi)
    return rotation_from_axis_angle(v / s, np.arctan2(s, c))


# -- canonical direction sets (unit vectors) ------------------------------
_TETRAHEDRAL = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
]) / np.sqrt(3.0)

_S3 = np.sqrt(3.0) / 2.0
_TRIGONAL = np.array([
    [1.0, 0.0, 0.0],
    [-0.5, _S3, 0.0],
    [-0.5, -_S3, 0.0],
])

_LINEAR = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
])

_TERMINAL = np.array([[0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class AtomTemplate:
    """The ideal local geometry for one ``(element, valence)`` combination.

    ``directions`` is an ``(valence, 3)`` array of unit bond directions in a
    canonical frame. ``directions[0]`` is the *attachment slot* -- the site
    oriented toward the parent atom when the fragment is grown onto an existing
    molecule; ``directions[1:]`` are the remaining open sites that get capped
    (with :attr:`cap`, hydrogen by default). When a fragment is born in free
    space (e.g. a lone methane), every direction is used.
    """

    element: str
    valence: int
    geometry: str
    directions: np.ndarray
    cap: str = "H"
    hybridization: str = ""          # "sp3" / "sp2" / "sp"; "" when N/A (e.g. H, halogens)

    def label(self) -> str:
        """A short human label, e.g. 'tetrahedral · sp3 · 4 bonds'."""
        bonds = f"{self.valence} bond" + ("s" if self.valence != 1 else "")
        parts = [self.geometry]
        if self.hybridization:
            parts.append(self.hybridization)
        parts.append(bonds)
        return " · ".join(parts)

    def free_directions(self) -> np.ndarray:
        """All bond directions in the canonical frame (for a free-standing atom)."""
        return self.directions

    def open_directions(self, attach_dir: np.ndarray) -> np.ndarray:
        """Open-site directions when the attachment slot points along *attach_dir*.

        *attach_dir* is the world-space direction from the new central atom
        toward the atom it bonds to. Returns the ``valence - 1`` remaining bond
        directions, rotated into world space, ready to receive caps.
        """
        if self.valence <= 1:
            return np.zeros((0, 3))
        R = rotation_between(self.directions[0], _normalize(attach_dir))
        return self.directions[1:] @ R.T


# -- registry: hybridization/valence options per element ------------------
# Mirrors the set of atom-type choices a builder like GaussView offers for each
# element: the standard VSEPR hybridization states (sp3 tetrahedral, sp2
# trigonal, sp linear, and the lone-pair-reduced pyramidal/bent variants). The
# order here is the order the geometry picker lists them in (most bonds first).
TEMPLATES = {
    # Carbon: sp3 / sp2 / sp
    ("C", 4): AtomTemplate("C", 4, "tetrahedral", _TETRAHEDRAL, hybridization="sp3"),
    ("C", 3): AtomTemplate("C", 3, "trigonal", _TRIGONAL, hybridization="sp2"),
    ("C", 2): AtomTemplate("C", 2, "linear", _LINEAR, hybridization="sp"),
    # Nitrogen: ammonium sp3 / amine sp3 / imine sp2 / nitrile sp
    ("N", 4): AtomTemplate("N", 4, "tetrahedral", _TETRAHEDRAL, hybridization="sp3"),
    ("N", 3): AtomTemplate("N", 3, "pyramidal", _TETRAHEDRAL[:3], hybridization="sp3"),
    ("N", 2): AtomTemplate("N", 2, "bent", _TRIGONAL[:2], hybridization="sp2"),
    ("N", 1): AtomTemplate("N", 1, "terminal", _TERMINAL, hybridization="sp"),
    # Oxygen: ether/hydroxyl sp3 / carbonyl-like sp2 / terminal
    ("O", 2): AtomTemplate("O", 2, "bent", _TETRAHEDRAL[:2], hybridization="sp3"),
    ("O", 1): AtomTemplate("O", 1, "terminal", _TERMINAL, hybridization="sp2"),
    # Boron: sp2 / sp3
    ("B", 3): AtomTemplate("B", 3, "trigonal", _TRIGONAL, hybridization="sp2"),
    ("B", 4): AtomTemplate("B", 4, "tetrahedral", _TETRAHEDRAL, hybridization="sp3"),
    # Heavier main-group with a couple of common valences
    ("P", 3): AtomTemplate("P", 3, "pyramidal", _TETRAHEDRAL[:3], hybridization="sp3"),
    ("S", 2): AtomTemplate("S", 2, "bent", _TETRAHEDRAL[:2], hybridization="sp3"),
    # Monovalent (halogens, hydrogen): a single terminal bond
    ("H", 1): AtomTemplate("H", 1, "terminal", _TERMINAL),
    ("F", 1): AtomTemplate("F", 1, "terminal", _TERMINAL),
    ("Cl", 1): AtomTemplate("Cl", 1, "terminal", _TERMINAL),
    ("Br", 1): AtomTemplate("Br", 1, "terminal", _TERMINAL),
    ("I", 1): AtomTemplate("I", 1, "terminal", _TERMINAL),
}

# Typical valence used to pick a default template when none is specified.
_DEFAULT_VALENCE = {"C": 4, "N": 3, "O": 2, "B": 3, "P": 3, "S": 2,
                    "H": 1, "F": 1, "Cl": 1, "Br": 1, "I": 1}


def default_template(element: str = "C") -> AtomTemplate:
    """The default building template for *element* (its most common valence).

    Falls back to a tetrahedral, carbon-like template for elements without a
    dedicated entry, so an unknown element still grows sensibly.
    """
    element = _normalize_symbol(element)
    val = _DEFAULT_VALENCE.get(element, 4)
    tmpl = TEMPLATES.get((element, val))
    if tmpl is not None:
        return tmpl
    # synthesize a tetrahedral fallback for the requested element
    return AtomTemplate(element, 4, "tetrahedral", _TETRAHEDRAL, hybridization="sp3")


def options_for(element: str):
    """The geometry/hybridization templates offered for *element*.

    Returns every registered template for the element, most bonds first (the
    order the picker lists them). Elements with no dedicated entry fall back to
    a single default template, so the picker always has at least one option.
    """
    element = _normalize_symbol(element)
    opts = [t for (el, _val), t in TEMPLATES.items() if el == element]
    if not opts:
        return [default_template(element)]
    opts.sort(key=lambda t: -t.valence)
    return opts


def free_direction(neighbor_dirs) -> np.ndarray:
    """A sensible open bond direction on an atom that already has neighbors.

    *neighbor_dirs* is a sequence of unit vectors pointing from the atom toward
    its existing neighbors. The result points away from their average (VSEPR-ish
    "as far from everything else as possible"); with no neighbors it defaults to
    +z, and for a symmetric arrangement (e.g. two opposed bonds) it picks a
    perpendicular direction rather than collapsing to zero.
    """
    dirs = [np.asarray(d, float) for d in neighbor_dirs]
    if not dirs:
        return np.array([0.0, 0.0, 1.0])
    s = np.sum([_normalize(d) for d in dirs], axis=0)
    n = float(np.linalg.norm(s))
    if n < 1e-6:
        return _perpendicular(dirs[0])
    return -s / n
