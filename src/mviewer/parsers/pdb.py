"""Minimal but practical PDB parser.

Handles ATOM/HETATM records with fixed-column layout, CONECT bonds, TER, and
MODEL/ENDMDL multi-model files. Element is taken from columns 77-78 when
present, otherwise guessed from the atom name.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..molecule import Molecule
from .. import elements


def _guess_element(atom_name: str, col_element: str) -> str:
    col_element = col_element.strip()
    if col_element:
        return elements.normalize_symbol(col_element)
    name = atom_name.strip()
    # PDB atom names: element is usually the leading alphabetic chunk; names
    # starting in column 13 with a digit are hydrogens/remoteness-indexed.
    if not name:
        return "X"
    if name[0].isdigit():
        name = name[1:]
    # Two-letter elements only for the known set to avoid e.g. 'CA' (calcium)
    # vs 'CA' (alpha carbon) confusion -> default to single letter for protein.
    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return "X"
    return elements.normalize_symbol(letters[0])


def parse(text: str) -> List[Molecule]:
    models: List[Molecule] = []
    cur = Molecule()
    serial_to_index: dict = {}
    have_atoms = False

    def flush():
        nonlocal cur, serial_to_index, have_atoms
        if have_atoms:
            models.append(cur)
        cur = Molecule()
        serial_to_index = {}
        have_atoms = False

    for line in text.splitlines():
        rec = line[:6].strip()
        if rec in ("ATOM", "HETATM"):
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = len(cur.symbols)
            atom_name = line[12:16]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            col_el = line[76:78] if len(line) >= 78 else ""
            el = _guess_element(atom_name, col_el)
            idx = len(cur.symbols)
            cur.symbols.append(el)
            serial_to_index[serial] = idx
            cur.positions = (
                np.vstack([cur.positions, [x, y, z]]) if len(cur.positions) else np.array([[x, y, z]], float)
            )
            have_atoms = True
        elif rec == "CONECT":
            try:
                a = serial_to_index.get(int(line[6:11]))
            except ValueError:
                continue
            if a is None:
                continue
            for c in (11, 16, 21, 26):
                frag = line[c:c + 5].strip()
                if not frag:
                    continue
                try:
                    b = serial_to_index.get(int(frag))
                except ValueError:
                    continue
                if b is not None and a != b:
                    lo, hi = (a, b) if a < b else (b, a)
                    if (lo, hi, 1) not in cur.bonds:
                        cur.bonds.append((lo, hi, 1))
        elif rec == "ENDMDL":
            flush()
        elif rec == "END":
            flush()

    flush()
    if not models:
        models.append(cur)
    return models
