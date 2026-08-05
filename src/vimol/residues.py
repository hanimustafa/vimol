"""What the glyph skin needs to know about amino acids.

Everything here is table lookup on PDB names -- three-letter residue names and
four-character atom names -- so it works on the overwhelmingly common case of a
crystal structure with no hydrogens in the file. Nothing infers chemistry from
geometry; ``glyphs.py`` does the geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .molecule import Molecule


ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common variants that are still the same residue for drawing purposes.
    "MSE": "M", "SEC": "U", "PYL": "O", "HIE": "H", "HID": "H", "HIP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

BACKBONE = frozenset(("N", "CA", "C", "O", "OXT"))

# Ring atoms of the aromatic side chains, in no particular order -- the plate
# builder fits a plane to them and takes their convex hull, so ordering and
# fusion (tryptophan's two fused rings) both come out of the hull.
AROMATIC_RINGS: Dict[str, Tuple[str, ...]] = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

# Side-chain hydrogen-bonding roles. A residue's backbone N donates and its
# backbone O accepts; those are handled separately since they apply to (almost)
# every residue. "both" covers the hydroxyls, the histidine ring nitrogens
# (either tautomer is plausible without hydrogens in the file) and thiols.
_SIDE_CHAIN_ROLES: Dict[str, Dict[str, str]] = {
    "ARG": {"NE": "donor", "NH1": "donor", "NH2": "donor"},
    "LYS": {"NZ": "donor"},
    "TRP": {"NE1": "donor"},
    "ASN": {"ND2": "donor", "OD1": "acceptor"},
    "GLN": {"NE2": "donor", "OE1": "acceptor"},
    "ASP": {"OD1": "acceptor", "OD2": "acceptor"},
    "GLU": {"OE1": "acceptor", "OE2": "acceptor"},
    "MET": {"SD": "acceptor"},
    "MSE": {"SE": "acceptor"},
    "SER": {"OG": "both"},
    "THR": {"OG1": "both"},
    "TYR": {"OH": "both"},
    "CYS": {"SG": "both"},
    "HIS": {"ND1": "both", "NE2": "both"},
    "HIE": {"ND1": "both", "NE2": "both"},
    "HID": {"ND1": "both", "NE2": "both"},
    "HIP": {"ND1": "both", "NE2": "both"},
}


def one_letter(resname: str) -> str:
    """One-character code for a three-letter residue name, ``X`` if unknown."""
    return ONE_LETTER.get(resname.strip().upper(), "X")


def is_aromatic(resname: str) -> bool:
    return resname.strip().upper() in AROMATIC_RINGS


def hbond_role(resname: str, atom_name: str) -> Optional[str]:
    """``"donor"``, ``"acceptor"``, ``"both"`` or None for one named atom.

    Proline's backbone nitrogen is in the ring and carries no hydrogen, so it
    is the one backbone N that does not donate.
    """
    resname = resname.strip().upper()
    atom_name = atom_name.strip().upper()
    if atom_name == "N":
        return None if resname == "PRO" else "donor"
    if atom_name in ("O", "OXT"):
        return "acceptor"
    return _SIDE_CHAIN_ROLES.get(resname, {}).get(atom_name)


@dataclass
class Residue:
    """One amino acid's atoms, gathered by PDB identity."""
    key: Tuple[str, str, str]        # (chain, residue number, insertion code)
    name: str                        # three-letter, upper case
    letter: str                      # one-letter code, "X" when unknown
    atoms: Dict[str, int] = field(default_factory=dict)   # atom name -> atom index

    def index(self, *names: str) -> Optional[int]:
        """Index of the first of *names* this residue actually has."""
        for n in names:
            if n in self.atoms:
                return self.atoms[n]
        return None

    def side_chain_indices(self) -> List[int]:
        """Heavy side-chain atoms: everything that is not backbone."""
        return [i for name, i in self.atoms.items() if name not in BACKBONE]

    @property
    def is_aromatic(self) -> bool:
        return self.name in AROMATIC_RINGS


def _identity(key: str) -> Optional[Tuple[str, str, str, str, str]]:
    """(record, chain, resseq, icode, altloc) out of a Molecule.atom_keys entry."""
    parts = key.split("|")
    if len(parts) != 6:
        return None
    rec, chain, resseq, icode, _name, alt = parts
    return rec, chain, resseq, icode, alt


def protein_residues(molecule: Molecule) -> List[Residue]:
    """Group a molecule's atoms into amino acid residues, in file order.

    Returns an empty list unless the molecule carries the PDB metadata this
    needs: without residue names there is no letter to draw and no way to tell
    an aromatic ring from a straight chain, and without atom keys there is no
    way to tell where one residue stops and the next begins.

    HETATM records, waters and anything without a Cα are skipped, as are the
    B and later alternate conformations -- drawing two overlapping glyphs for
    one residue would read as a rendering fault, not as disorder.
    """
    n = molecule.n_atoms
    names, keys = molecule.atom_names, molecule.atom_keys
    resnames = molecule.atom_resnames
    if not (len(names) == len(keys) == len(resnames) == n) or n == 0:
        return []

    by_key: Dict[Tuple[str, str, str], Residue] = {}
    order: List[Tuple[str, str, str]] = []
    for i in range(n):
        if molecule.symbols[i].strip().upper() == "H":
            continue
        ident = _identity(keys[i])
        if ident is None:
            continue
        rec, chain, resseq, icode, alt = ident
        if rec != "ATOM" or alt not in ("", "A"):
            continue
        resname = resnames[i].strip().upper()
        if resname not in ONE_LETTER:
            continue
        key = (chain, resseq, icode)
        res = by_key.get(key)
        if res is None:
            res = by_key[key] = Residue(key=key, name=resname, letter=one_letter(resname))
            order.append(key)
        # First occurrence wins, so a stray duplicate atom name cannot move a
        # glyph off the coordinates the rest of the residue agrees on.
        res.atoms.setdefault(names[i].strip().upper(), i)

    return [by_key[k] for k in order if "CA" in by_key[k].atoms]


def chain_runs(residues: Sequence[Residue], positions: np.ndarray,
               max_peptide_bond: float = 2.0) -> List[List[Residue]]:
    """Split residues into runs joined by real peptide bonds.

    A ribbon must not leap across a chain break or between chains, and residue
    numbering alone does not say where the breaks are -- a gap in numbering can
    still be a continuous chain, and consecutive numbers can still be far
    apart. So the split is on the actual C(i)–N(i+1) distance, with the chain
    identifier as a hard boundary.
    """
    runs: List[List[Residue]] = []
    current: List[Residue] = []
    for res in residues:
        if current:
            prev = current[-1]
            c = prev.index("C")
            nitrogen = res.index("N")
            broken = (prev.key[0] != res.key[0] or c is None or nitrogen is None
                      or float(np.linalg.norm(positions[c] - positions[nitrogen]))
                      > max_peptide_bond)
            if broken:
                runs.append(current)
                current = []
        current.append(res)
    if current:
        runs.append(current)
    return runs
