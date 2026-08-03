"""Distance-based bond perception.

Two atoms are bonded when their separation is below the sum of covalent radii
plus a tolerance. That test is the same for every pair, so it runs as array
work: a block of squared distances compared against a block of cutoffs, with
the pairs read straight out of the boolean result.

This used to be a uniform grid (spatial hash) to keep the pair search
near-linear rather than O(N^2). The asymptotics were real but irrelevant at
the sizes that actually hurt -- the grid was walked by a Python loop doing
scalar arithmetic per candidate pair, and at N=334 that cost ~8.8ms against
~0.9MB of distance matrix. Over an 803-frame trajectory it was ~7s of startup,
essentially all of it. Array work on the whole block is far cheaper up to
sizes where memory, not time, becomes the limit -- and _CHUNK_BYTES keeps that
in hand.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .molecule import Molecule

# Rejects overlapping/duplicate atoms: closer than 0.4 A is not a bond.
_MIN_DIST2 = 0.16
# Peak size of the transient coordinate-difference block. The rows of that
# block are what bounds memory: a full N x N x 3 float32 difference array is
# 4.8 GB at N=20000 (three times the distance matrix it reduces to, which is
# the part that surprises), so the pass is split into row chunks that each
# stay under this. Chunking costs nothing measurable -- the work per chunk is
# still one vectorized pass -- and it means a large structure degrades in
# speed rather than dying in the allocator.
_CHUNK_BYTES = 64 << 20


def _chunk_rows(n: int) -> int:
    """How many rows of the difference block to build at once, for *n* atoms."""
    per_row = max(n * 3 * 4, 1)          # xyz float32 deltas against all atoms
    return max(1, min(n, _CHUNK_BYTES // per_row))


def perceive_bonds(mol: Molecule, tolerance: float = 0.45, max_bonds_per_atom: int = 8) -> List[Tuple[int, int, int]]:
    """Return a bond list inferred from interatomic distances.

    tolerance is added to the sum of covalent radii (angstrom).
    """
    n = mol.n_atoms
    if n < 2:
        return []
    # float32, deliberately. It halves float64's footprint at no cost in
    # correctness, while float16 -- which would halve it again -- is both
    # wrong and slower here: its ~3 significant digits make the result depend
    # on where the molecule sits in space (a structure translated to +5000 A
    # loses every one of its bonds, silently), and numpy has no native
    # float16 arithmetic, so it upconverts and runs ~3x slower than float32.
    pos = np.ascontiguousarray(mol.positions, dtype=np.float32)
    cov = np.asarray(mol.covalent_radii(), dtype=np.float32)
    tol = np.float32(tolerance)

    rows = _chunk_rows(n)
    found: List[np.ndarray] = []
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        diff = pos[start:stop, None, :] - pos[None, :, :]
        # einsum sums the squares in one pass, without materializing diff**2.
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        cutoff = cov[start:stop, None] + cov[None, :] + tol
        close = (dist2 <= cutoff * cutoff) & (dist2 >= np.float32(_MIN_DIST2))
        # Each pair once: column j must sit past this row's global index, so
        # the kept triangle starts one past where this chunk begins.
        close = np.triu(close, start + 1)
        local_i, j = np.nonzero(close)
        if local_i.size:
            found.append(np.stack((local_i + start, j), axis=1))

    if not found:
        return []
    pairs = np.concatenate(found)
    # Ascending (a, b) so the cap below resolves crowding deterministically.
    # The grid version resolved it in bucket-visit order, which was stable but
    # arbitrary; either way the cap only engages on inputs no real structure
    # produces (more than max_bonds_per_atom neighbours inside the cutoff).
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]

    bonds: List[Tuple[int, int, int]] = []
    counts = np.zeros(n, dtype=np.int32)
    for a, b in pairs.tolist():
        if counts[a] >= max_bonds_per_atom or counts[b] >= max_bonds_per_atom:
            continue
        bonds.append((a, b, 1))
        counts[a] += 1
        counts[b] += 1
    return bonds


def ensure_bonds(mol: Molecule, tolerance: float = 0.45) -> Molecule:
    """Populate mol.bonds if empty. Returns the molecule for chaining."""
    if not mol.bonds and mol.n_atoms > 1:
        mol.bonds = perceive_bonds(mol, tolerance=tolerance)
    return mol
