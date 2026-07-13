"""Molecular file-format parsers and a format-dispatching loader."""
from __future__ import annotations

import os
from typing import List

from ..molecule import Molecule
from . import xyz as _xyz
from . import pdb as _pdb
from . import mol as _mol

_EXT_PARSERS = {
    ".xyz": _xyz.parse,
    ".pdb": _pdb.parse,
    ".ent": _pdb.parse,
    ".mol": _mol.parse,
    ".sdf": _mol.parse,
    ".mdl": _mol.parse,
}

SUPPORTED_EXTENSIONS = tuple(sorted(_EXT_PARSERS))


def load(path: str, fmt: str | None = None) -> Molecule:
    """Load a single molecule from a file, dispatching on extension (or *fmt*)."""
    mols = load_all(path, fmt=fmt)
    if not mols:
        raise ValueError(f"No molecules found in {path!r}")
    return mols[0]


def load_all(path: str, fmt: str | None = None) -> List[Molecule]:
    """Load every model/frame from a file (e.g. multi-record SDF, NMR PDB)."""
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
    else:
        ext = fmt if fmt.startswith(".") else "." + fmt.lower()
    parser = _EXT_PARSERS.get(ext)
    if parser is None:
        raise ValueError(
            f"Unsupported format {ext!r}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    mols = parser(text)
    if isinstance(mols, Molecule):
        mols = [mols]
    for m in mols:
        if not m.name:
            m.name = os.path.splitext(os.path.basename(path))[0]
    return mols


def loads(text: str, fmt: str) -> Molecule:
    """Parse a molecule from an in-memory string given an explicit format."""
    ext = fmt if fmt.startswith(".") else "." + fmt.lower()
    parser = _EXT_PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported format {fmt!r}")
    mols = parser(text)
    if isinstance(mols, Molecule):
        return mols
    return mols[0]
