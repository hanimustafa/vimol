"""MDL MOL (V2000) and SDF parser (SDF = one or more MOL records)."""
from __future__ import annotations

from typing import List

import numpy as np

from ..molecule import Molecule
from .. import elements


def _parse_one(block: str) -> Molecule | None:
    lines = block.splitlines()
    if len(lines) < 4:
        return None
    name = lines[0].strip()
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
        n_bonds = int(counts[3:6])
    except ValueError:
        return None
    symbols: List[str] = []
    coords: List[list] = []
    for k in range(n_atoms):
        row = lines[4 + k]
        x = float(row[0:10])
        y = float(row[10:20])
        z = float(row[20:30])
        sym = row[31:34].strip()
        symbols.append(elements.normalize_symbol(sym))
        coords.append([x, y, z])
    bonds: List[tuple] = []
    for k in range(n_bonds):
        row = lines[4 + n_atoms + k]
        a = int(row[0:3]) - 1
        b = int(row[3:6]) - 1
        order = int(row[6:9]) if row[6:9].strip() else 1
        lo, hi = (a, b) if a < b else (b, a)
        bonds.append((lo, hi, order))
    return Molecule(
        symbols=symbols,
        positions=np.array(coords, float) if coords else np.zeros((0, 3)),
        bonds=bonds,
        name=name[:60],
    )


def parse(text: str) -> List[Molecule]:
    mols: List[Molecule] = []
    for block in text.split("$$$$"):
        if not block.strip():
            continue
        m = _parse_one(block)
        if m is not None and m.n_atoms:
            mols.append(m)
    return mols
