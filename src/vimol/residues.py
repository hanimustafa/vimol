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

# Proline is the one residue whose side chain closes a ring without being
# aromatic, and it closes it back onto the backbone -- so its ring is N, Cα and
# the three side-chain carbons. It gets a tablet cut to that pentagon like the
# aromatics do, in the aliphatic colour rather than the aromatic one.
PROLINE_RING = ("N", "CA", "CB", "CG", "CD")

RING_ATOMS: Dict[str, Tuple[str, ...]] = dict(AROMATIC_RINGS)
RING_ATOMS["PRO"] = PROLINE_RING

# Every side chain's atoms, in PDB names. Used to name and identify a residue
# in a file that has no names at all: PDB side-chain names encode the element
# and the Greek letter, and the Greek letter *is* the bond distance from Cα, so
# each name here doubles as an (element, depth) fact. The multiset of those
# facts is unique across the twenty, which is what makes the identification a
# lookup rather than a search.
SIDE_CHAIN_ATOMS: Dict[str, Tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
}

# beta, gamma, delta, epsilon, zeta, eta -- one bond further from Cα each.
_GREEK = {"B": 1, "G": 2, "D": 3, "E": 4, "Z": 5, "H": 6}


def _atom_fact(name: str) -> Tuple[int, str]:
    """(bond distance from Cα, element) for a PDB side-chain atom name."""
    return _GREEK.get(name[1], 0), name[0]


def _signature(facts) -> Tuple[Tuple[int, str], ...]:
    return tuple(sorted(facts))


_BY_SIGNATURE = {_signature(_atom_fact(n) for n in names): resname
                 for resname, names in SIDE_CHAIN_ATOMS.items()}

# Side-chain hydrogen-bonding roles. Nothing draws from this at the moment:
# the skin used to mark donors and acceptors with coloured nodes, then with
# hairlines between backbone pairs, and both were dropped -- the atoms
# themselves are drawn now, in their own element colours, which says the same
# thing without a second vocabulary. Kept because the table is the part that
# would be tedious to rebuild if the interactions are ever drawn again.
#
# A residue's backbone N donates and its backbone O accepts; those are handled
# separately since they apply to (almost) every residue. "both" covers the
# hydroxyls, the histidine ring nitrogens (either tautomer is plausible without
# hydrogens in the file) and thiols.
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

    @property
    def is_ring(self) -> bool:
        """Whether this residue's glyph is a tablet cut to a ring outline."""
        return self.name in RING_ATOMS


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
        # No names to read -- infer the whole thing from the structure.
        return infer_residues(molecule) if n else []

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


def infer_residues(molecule: Molecule) -> List[Residue]:
    """Read a protein out of elements and connectivity, with no names to help.

    An xyz file carries neither residue names nor atom names, so both have to
    come from the structure. The backbone comes from ``select.peptide_motifs``,
    the same detector the selection presets use. Each Cα then gets its side
    chain walked outward, breadth first, which gives every atom its bond
    distance from the Cα -- and that distance is exactly what a PDB name's
    Greek letter records. So the walk yields the same (element, depth) facts a
    named file would, the multiset identifies the residue against
    :data:`SIDE_CHAIN_ATOMS`, and pairing the two off in the same order hands
    every atom the name it would have had in a PDB.

    The result is ordinary :class:`Residue` objects, indistinguishable from the
    parsed ones, so everything downstream is unchanged.
    """
    from .bonds import perceive_bonds
    from .select import peptide_motifs

    runs = peptide_motifs(molecule)
    if not runs:
        return []
    symbols = [s.strip().upper() for s in molecule.symbols]
    neighbors: List[List[int]] = [[] for _ in range(molecule.n_atoms)]
    for i, j, _order in (molecule.bonds or perceive_bonds(molecule)):
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Every motif's backbone, so a side-chain walk cannot leak along the chain
    # into the next residue by way of a mis-perceived bond.
    spine = {i for run in runs for motif in run for i in motif[:3]}

    out: List[Residue] = []
    number = 0
    for chain, run in enumerate(runs):
        for nitrogen, alpha, carbonyl, oxygen in run:
            number += 1
            backbone = set(spine)
            if oxygen is not None:
                backbone.add(oxygen)
            side = _walk_side_chain(alpha, backbone, neighbors, symbols)
            name = _BY_SIGNATURE.get(_signature((d, symbols[i]) for i, d in side))
            res = Residue(key=(str(chain), str(number), ""),
                          name=name or "UNK", letter=one_letter(name or "UNK"))
            for atom_name, index in (("N", nitrogen), ("CA", alpha), ("C", carbonyl),
                                     ("O", oxygen)):
                if index is not None:
                    res.atoms[atom_name] = index
                    res.elements[atom_name] = symbols[index]
            if name:
                # Both lists in the same (depth, element) order, so equivalent
                # atoms pair up and a ring ends up with the exact names
                # RING_ATOMS looks for.
                canonical = sorted(SIDE_CHAIN_ATOMS[name], key=_atom_fact)
                found = sorted(side, key=lambda p: (p[1], symbols[p[0]]))
                for atom_name, (index, _depth) in zip(canonical, found):
                    res.atoms[atom_name] = index
                    res.elements[atom_name] = symbols[index]
            out.append(res)
    return out


def _walk_side_chain(alpha: int, backbone, neighbors, symbols):
    """[(atom, bond distance from Cα)] for the heavy atoms hanging off *alpha*.

    Breadth first, so the distance is the shortest path -- which is what makes
    it line up with the Greek letters. Proline's ring closes back onto the
    backbone nitrogen; the walk simply stops there, leaving the three carbons
    that make it distinguishable.

    Disulfides are the one covalent bond between two side chains a protein
    routinely has, and crossing one merges both cysteines into a single
    unrecognizable fragment. No residue has two sulfurs of its own, so refusing
    to step from one to another costs nothing and stops exactly that.
    """
    found, seen, frontier, depth = [], set(backbone), [alpha], 0
    while frontier:
        depth += 1
        nxt = []
        for atom in frontier:
            for j in neighbors[atom]:
                if j in seen or symbols[j] == "H":
                    continue
                if symbols[atom] == "S" and symbols[j] == "S":
                    continue
                seen.add(j)
                found.append((j, depth))
                nxt.append(j)
        frontier = nxt
    return found


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
