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


# Broad side-chain classes, used only to tint a residue's Cα/Cβ beads and rods
# so a glance across the structure separates the acidic from the aromatic from
# the merely greasy.
_CLASSES = {
    "aromatic": "FWYH",
    "acidic": "DE",
    "basic": "KR",
    "polar": "STNQC",
}
CLASS_OF = {letter: name for name, letters in _CLASSES.items() for letter in letters}


def residue_class(letter: str) -> str:
    return CLASS_OF.get(letter.upper(), "hydrophobic")


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
    elements: Dict[str, str] = field(default_factory=dict)  # atom name -> element

    def index(self, *names: str) -> Optional[int]:
        """Index of the first of *names* this residue actually has."""
        for n in names:
            if n in self.atoms:
                return self.atoms[n]
        return None

    def side_chain_indices(self) -> List[int]:
        """Heavy side-chain atoms: everything that is not backbone."""
        return [i for name, i in self.atoms.items()
                if name not in BACKBONE and self.elements.get(name) != "H"]

    def side_chain_carbons(self) -> List[int]:
        """The side chain's carbon skeleton -- what a glyph solid stands for.

        Everything else the residue owns is drawn as itself, so the carbons are
        the only atoms a shape has to swallow.
        """
        return [i for name, i in self.atoms.items()
                if name not in BACKBONE and self.elements.get(name) == "C"]

    def side_chain_polar(self) -> List[int]:
        """Side-chain heavy atoms that are not carbon: N, O, S.

        The backbone amide is deliberately not here. The ribbon stands for the
        whole backbone, N and carbonyl O included, and drawing those as atoms
        stipples a blue and red dot onto every residue of it.
        """
        return [i for name, i in self.atoms.items()
                if name not in BACKBONE and self.elements.get(name) not in ("C", "H")]

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

    Residues are runs of consecutive atoms sharing an identity, not a lookup
    keyed on that identity. A PDB writes them contiguously either way, and the
    run rule is what keeps two overlaid copies of the same file apart: they
    repeat every chain/number pair, and a dictionary would fold the second copy
    into the first and leave it with no glyphs at all.
    """
    n = molecule.n_atoms
    names, keys = molecule.atom_names, molecule.atom_keys
    resnames = molecule.atom_resnames
    if not (len(names) == len(keys) == len(resnames) == n) or n == 0:
        return []

    out: List[Residue] = []
    current: Optional[Tuple[str, str, str]] = None
    for i in range(n):
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
        if key != current:
            current = key
            out.append(Residue(key=key, name=resname, letter=one_letter(resname)))
        # First occurrence wins, so a stray duplicate atom name cannot move a
        # glyph off the coordinates the rest of the residue agrees on.
        name = names[i].strip().upper()
        if name not in out[-1].atoms:
            out[-1].atoms[name] = i
            out[-1].elements[name] = molecule.symbols[i].strip().upper()

    return [r for r in out if "CA" in r.atoms]


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
