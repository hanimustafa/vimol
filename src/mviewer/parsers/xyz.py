"""XYZ / extended-XYZ parser (supports multi-frame trajectories)."""
from __future__ import annotations

from typing import List

import numpy as np

from ..molecule import Molecule
from .. import elements


def parse(text: str) -> List[Molecule]:
    lines = text.splitlines()
    mols: List[Molecule] = []
    i = 0
    n = len(lines)
    while i < n:
        # Skip blank lines between frames.
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        try:
            count = int(lines[i].strip().split()[0])
        except (ValueError, IndexError):
            break
        comment = lines[i + 1] if i + 1 < n else ""
        symbols: List[str] = []
        coords: List[list] = []
        for k in range(count):
            row = i + 2 + k
            if row >= n:
                break
            parts = lines[row].split()
            if len(parts) < 4:
                continue
            sym = parts[0]
            # Extended XYZ may use atomic numbers instead of symbols.
            if sym.isdigit():
                sym = elements.SYMBOLS[int(sym)] if 0 < int(sym) < len(elements.SYMBOLS) else "X"
            symbols.append(elements.normalize_symbol(sym))
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        mol = Molecule(
            symbols=symbols,
            positions=np.array(coords, float) if coords else np.zeros((0, 3)),
            name=comment.strip()[:60],
        )
        mols.append(mol)
        i += 2 + count
    return mols
