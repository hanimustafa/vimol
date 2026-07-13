"""Distance-based bond perception.

Two atoms are bonded when their separation is below the sum of covalent radii
plus a tolerance. A uniform grid (spatial hash) keeps this near-linear so it
scales to large structures instead of being O(N^2).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .molecule import Molecule


def perceive_bonds(mol: Molecule, tolerance: float = 0.45, max_bonds_per_atom: int = 8) -> List[Tuple[int, int, int]]:
    """Return a bond list inferred from interatomic distances.

    tolerance is added to the sum of covalent radii (angstrom).
    """
    n = mol.n_atoms
    if n < 2:
        return []
    pos = mol.positions
    cov = mol.covalent_radii()
    max_r = float(cov.max())
    cell = max(2 * max_r + tolerance, 1.0)

    # Spatial hash: map each atom to an integer grid cell.
    mins = pos.min(axis=0)
    ijk = np.floor((pos - mins) / cell).astype(np.int64)
    buckets: dict = {}
    for a in range(n):
        buckets.setdefault(tuple(ijk[a]), []).append(a)

    bonds: List[Tuple[int, int, int]] = []
    counts = np.zeros(n, dtype=np.int32)
    seen = set()
    neigh = [(-1, 0, 1)] * 3
    from itertools import product
    offsets = list(product(*neigh))

    for a in range(n):
        base = ijk[a]
        cand: List[int] = []
        for off in offsets:
            key = (base[0] + off[0], base[1] + off[1], base[2] + off[2])
            b = buckets.get(key)
            if b:
                cand.extend(b)
        pa = pos[a]
        ra = cov[a]
        for b in cand:
            if b <= a:
                continue
            cutoff = ra + cov[b] + tolerance
            d = pa - pos[b]
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= cutoff * cutoff:
                # Reject absurdly short contacts (overlapping/duplicate atoms).
                dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                if dist2 < 0.16:  # < 0.4 A
                    continue
                if counts[a] >= max_bonds_per_atom or counts[b] >= max_bonds_per_atom:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                bonds.append((a, b, 1))
                counts[a] += 1
                counts[b] += 1
    return bonds


def ensure_bonds(mol: Molecule, tolerance: float = 0.45) -> Molecule:
    """Populate mol.bonds if empty. Returns the molecule for chaining."""
    if not mol.bonds and mol.n_atoms > 1:
        mol.bonds = perceive_bonds(mol, tolerance=tolerance)
    return mol
