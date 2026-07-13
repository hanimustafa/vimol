"""The core molecular data model.

A :class:`Molecule` stores atoms as parallel numpy arrays (positions, colors,
radii) plus a bond list. Keeping everything as numpy arrays lets the renderer
operate on the whole structure vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import elements


@dataclass
class Molecule:
    """A set of atoms and the bonds between them.

    Attributes
    ----------
    symbols:   list of element symbols, length N
    positions: (N, 3) float array, angstrom
    bonds:     list of (i, j, order) tuples, i < j
    name:      free-form label
    """

    symbols: List[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    bonds: List[Tuple[int, int, int]] = field(default_factory=list)
    name: str = ""

    # -- construction -----------------------------------------------------
    def add_atom(self, symbol: str, x: float, y: float, z: float) -> int:
        idx = len(self.symbols)
        self.symbols.append(elements.normalize_symbol(symbol))
        self.positions = np.vstack([self.positions, [x, y, z]]) if len(self.positions) else np.array([[x, y, z]], float)
        return idx

    def add_bond(self, i: int, j: int, order: int = 1) -> None:
        if i > j:
            i, j = j, i
        self.bonds.append((i, j, order))

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
