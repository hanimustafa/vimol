"""The core molecular data model.

A :class:`Molecule` stores atoms as parallel numpy arrays (positions, colors,
radii) plus a bond list. Keeping everything as numpy arrays lets the renderer
operate on the whole structure vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from . import elements


@dataclass
class VectorField:
    """A set of per-atom vectors (e.g. dipoles, forces) drawn as arrows.

    ``vectors`` is atom-aligned with :attr:`Molecule.positions` -- an atom
    with a near-zero vector simply has no arrow drawn. Color is the only
    thing distinguishing one field from another (this layer has no notion
    of what the vectors represent); magnitudes are never normalized, so
    arrow length is ``|vector| * scale``.
    """
    vectors: np.ndarray                              # (N, 3) angstrom, atom-aligned
    color: Tuple[float, float, float] = (0.15, 0.95, 0.85)
    scale: float = 1.0                                # arrow length = |vector| * scale
    radius: float = 0.05                              # shaft radius, angstrom
    head_scale: float = 2.5                           # head base radius = radius * head_scale
    head_length_frac: float = 0.25                    # fraction of arrow length that's the head


@dataclass
class Molecule:
    """A set of atoms and the bonds between them.

    Attributes
    ----------
    symbols:   list of element symbols, length N
    positions: (N, 3) float array, angstrom
    bonds:     list of (i, j, order) tuples, i < j
    name:      free-form label
    atom_names / atom_keys / atom_resnames: optional atom-aligned PDB metadata
    """

    symbols: List[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    bonds: List[Tuple[int, int, int]] = field(default_factory=list)
    name: str = ""
    vector_fields: List[VectorField] = field(default_factory=list)
    # Explicit bonds (e.g. drawn by option-drag) that survive re-perception
    # even beyond the auto distance threshold; (i, j, order) with i < j, like
    # `bonds`. `editor._reperceive` unions these into `bonds` on every rebuild.
    manual_bonds: List[Tuple[int, int, int]] = field(default_factory=list)
    # Indices of atoms created by editing since load (or since the last
    # `editor.cleanup` accepted the relaxed geometry). `editor.py` marks every
    # atom it creates; `editor.cleanup` reads this to weight its relaxation
    # and clears it once the geometry is "accepted".
    new_atoms: Set[int] = field(default_factory=set)
    # Optional PDB metadata, atom-aligned when present. Other formats leave
    # these empty because element + coordinates cannot identify CA/CB roles.
    atom_names: List[str] = field(default_factory=list)
    atom_is_hetatm: List[bool] = field(default_factory=list)
    atom_keys: List[str] = field(default_factory=list)
    # Residue name (PDB columns 18-20), atom-aligned when present. Kept apart
    # from `atom_keys` on purpose: align.py and StructureSet match structures on
    # that key's exact format, so it must not grow a field.
    atom_resnames: List[str] = field(default_factory=list)
    # True once distance-based perception has run for the current geometry.
    # A non-empty `bonds` already implies that, so this exists purely for the
    # molecules that legitimately have none within the cutoff -- noble gases,
    # isolated ions, a fragmented frame. Perception is deferred to draw time
    # (see StructureSet.composite), so without this they would be re-perceived
    # on every redraw, forever, to always produce the same empty list.
    # Deliberately outside identity and repr: whether a molecule happens to
    # have been drawn yet says nothing about what molecule it is.
    bonds_perceived: bool = field(default=False, compare=False, repr=False)

    # -- construction -----------------------------------------------------
    def add_atom(self, symbol: str, x: float, y: float, z: float) -> int:
        idx = len(self.symbols)
        self.symbols.append(elements.normalize_symbol(symbol))
        if self.atom_names:
            self.atom_names.append("")
        if self.atom_is_hetatm:
            self.atom_is_hetatm.append(False)
        if self.atom_keys:
            self.atom_keys.append("")
        if self.atom_resnames:
            self.atom_resnames.append("")
        self.positions = np.vstack([self.positions, [x, y, z]]) if len(self.positions) else np.array([[x, y, z]], float)
        return idx

    def add_bond(self, i: int, j: int, order: int = 1) -> None:
        if i > j:
            i, j = j, i
        self.bonds.append((i, j, order))

    def add_vector_field(self, vectors, color=(0.15, 0.95, 0.85), scale: float = 1.0,
                         radius: float = 0.05, head_scale: float = 2.5,
                         head_length_frac: float = 0.25) -> VectorField:
        """Attach a set of per-atom vectors (e.g. dipoles) to be drawn as arrows."""
        vectors = np.asarray(vectors, np.float64)
        if vectors.shape != (self.n_atoms, 3):
            raise ValueError(
                f"vectors must have shape ({self.n_atoms}, 3), got {vectors.shape}")
        vf = VectorField(vectors=vectors, color=tuple(color), scale=scale, radius=radius,
                         head_scale=head_scale, head_length_frac=head_length_frac)
        self.vector_fields.append(vf)
        return vf

    # -- derived quantities ----------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def centroid(self) -> np.ndarray:
        if self.n_atoms == 0:
            return np.zeros(3)
        return self.positions.mean(axis=0)

    def radius_of_gyration_extent(self) -> float:
        """A robust size estimate: max distance of any atom from the centroid."""
        if self.n_atoms == 0:
            return 1.0
        d = np.linalg.norm(self.positions - self.centroid(), axis=1)
        return float(max(d.max(), 1e-3))

    def vector_extent(self) -> float:
        """Max centroid-to-arrow-tip distance across all vector fields, 0 if none.

        Mirrors :meth:`radius_of_gyration_extent` so the camera's auto-fit can
        pad for long arrows the same way it pads for atom radii.
        """
        if not self.vector_fields or self.n_atoms == 0:
            return 0.0
        c = self.centroid()
        best = 0.0
        for vf in self.vector_fields:
            # A field can go stale if atoms are added after it was attached;
            # skip it here exactly as build_arrow_geometry does (rather than
            # letting the (N+1,3)+(N,3) broadcast raise mid-render).
            if np.asarray(vf.vectors).shape != (self.n_atoms, 3):
                continue
            tips = self.positions + vf.vectors * vf.scale
            # pad by the cone-head base radius so a fat arrowhead at the edge
            # of the scene isn't clipped by auto-fit.
            d = np.linalg.norm(tips - c, axis=1) + vf.radius * vf.head_scale
            best = max(best, float(d.max()))
        return best

    def element_colors(self) -> np.ndarray:
        return np.array([elements.element_color(s) for s in self.symbols], np.float64)

    def covalent_radii(self) -> np.ndarray:
        return np.array([elements.covalent_radius(s) for s in self.symbols], np.float64)

    def vdw_radii(self) -> np.ndarray:
        return np.array([elements.vdw_radius(s) for s in self.symbols], np.float64)

    def atomic_numbers(self) -> np.ndarray:
        return np.array([elements.symbol_to_z(s) for s in self.symbols], np.int32)

    def recenter(self) -> "Molecule":
        """Translate so the centroid sits at the origin (in place)."""
        if self.n_atoms:
            self.positions = self.positions - self.centroid()
        return self

    def formula(self) -> str:
        """Hill-system molecular formula, e.g. 'C8H10N4O2'."""
        from collections import Counter
        c = Counter(self.symbols)
        parts = []
        for el in ("C", "H"):
            if c.get(el):
                parts.append(el + (str(c[el]) if c[el] > 1 else ""))
                del c[el]
        for el in sorted(c):
            parts.append(el + (str(c[el]) if c[el] > 1 else ""))
        return "".join(parts)

    def __repr__(self) -> str:
        return f"<Molecule {self.name!r} atoms={self.n_atoms} bonds={len(self.bonds)} formula={self.formula()}>"
