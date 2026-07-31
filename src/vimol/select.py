"""Fast, stateless atom-selection helpers."""
from __future__ import annotations

import numpy as np

from .molecule import Molecule
from .bonds import perceive_bonds


_PEPTIDE_BACKBONE = frozenset(("N", "CA", "C"))
# Long enough to tolerate ordinary carboxylate C--O distances, but shorter
# than a C--OH single bond (about 1.43 A). This matters for hydrogen-free
# threonine/serine: their terminal O must not make Cbeta look carbonyl-like.
_CARBONYL_CO_MAX2 = 1.36 ** 2


def _topology_backbone(molecule: Molecule, include_beta_carbon: bool) -> np.ndarray:
    """Infer N-Cα-C(-O) motifs from elements and connectivity in O(N+E)."""
    n = molecule.n_atoms
    if n < 4:
        return np.empty(0, dtype=np.int64)
    bonds = molecule.bonds or perceive_bonds(molecule)
    neighbors = [[] for _ in range(n)]
    bond_orders = {}
    for i, j, order in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
        bond_orders[(min(i, j), max(i, j))] = order
    symbols = molecule.symbols

    def heavy_degree(i: int) -> int:
        return sum(symbols[j] != "H" for j in neighbors[i])

    def carbonyl_oxygens(carbon: int):
        out = []
        for oxygen in neighbors[carbon]:
            if symbols[oxygen] != "O" or heavy_degree(oxygen) != 1:
                continue
            order = bond_orders.get((min(carbon, oxygen), max(carbon, oxygen)), 1)
            delta = molecule.positions[oxygen] - molecule.positions[carbon]
            if order >= 2 or float(np.dot(delta, delta)) <= _CARBONYL_CO_MAX2:
                out.append(oxygen)
        return out

    motifs = []
    for carbonyl, symbol in enumerate(symbols):
        if symbol != "C":
            continue
        oxygens = carbonyl_oxygens(carbonyl)
        if not oxygens:
            continue
        # The Cα is the carbonyl carbon's carbon neighbor that itself touches
        # nitrogen. This tiny condition rejects common side-chain amides:
        # their adjacent CB/CD carbon is not N-bound.
        alpha_candidates = [j for j in neighbors[carbonyl]
                            if symbols[j] == "C"
                            and any(symbols[k] == "N" for k in neighbors[j])]
        if not alpha_candidates:
            continue
        # Bond perception can occasionally add a longer C(carbonyl)-Cβ edge.
        # The true Cα is the closest legal candidate; selecting every legal
        # candidate is what made the Backbone preset leak into Cβ.
        alpha = min(
            alpha_candidates,
            key=lambda j: float(np.dot(molecule.positions[j] - molecule.positions[carbonyl],
                                      molecule.positions[j] - molecule.positions[carbonyl])))
        nitrogens = [j for j in neighbors[alpha] if symbols[j] == "N"]
        if nitrogens:
            nitrogen = min(
                nitrogens,
                key=lambda j: float(np.dot(molecule.positions[j] - molecule.positions[alpha],
                                          molecule.positions[j] - molecule.positions[alpha])))
            motifs.append((nitrogen, alpha, carbonyl))

    # A real peptide is a continuous ...C(=O)-N-CA-C(=O)-N... walk. Local
    # N-CA-C=O lookalikes can occur in caps and side chains; when a chain is
    # present, retain only motifs participating in at least one peptide bond.
    # A lone motif is still useful for a single amino-acid file.
    linked = set()
    motif_by_nitrogen = {}
    for i, (nitrogen, _alpha, _carbonyl) in enumerate(motifs):
        motif_by_nitrogen.setdefault(nitrogen, []).append(i)
    for i, (_nitrogen, _alpha, carbonyl) in enumerate(motifs):
        for neighbor in neighbors[carbonyl]:
            for j in motif_by_nitrogen.get(neighbor, ()):
                if i != j:
                    linked.update((i, j))
    kept = linked if linked else ({0} if len(motifs) == 1 else set())

    selected = set()
    for i in kept:
        nitrogen, alpha, carbonyl = motifs[i]
        # Backbone here deliberately means the peptide path. Carbonyl O is a
        # branch off that path and must not participate in an RMSD subset.
        selected.update((nitrogen, alpha, carbonyl))
        if include_beta_carbon:
            sidechain = [j for j in neighbors[alpha]
                         if symbols[j] == "C" and j != carbonyl]
            if sidechain:
                beta = min(
                    sidechain,
                    key=lambda j: float(np.dot(molecule.positions[j] - molecule.positions[alpha],
                                              molecule.positions[j] - molecule.positions[alpha])))
                selected.add(beta)
    return np.asarray(sorted(selected), dtype=np.int64)


def peptide_backbone(molecule: Molecule, include_beta_carbon: bool = False) -> np.ndarray:
    """Return peptide-path ``N``, ``CA``, ``C`` indices, optionally ``CB``.

    PDB ``ATOM`` names are exact and preferred. For unnamed formats, a single
    bond-graph pass recognizes the local ``N-Cα-C(-O)`` motif. An empty result
    means no credible peptide-backbone motif was found.
    """
    names = molecule.atom_names
    if len(names) == molecule.n_atoms:
        allowed = _PEPTIDE_BACKBONE | ({"CB"} if include_beta_carbon else set())
        hetatm = (molecule.atom_is_hetatm
                  if len(molecule.atom_is_hetatm) == molecule.n_atoms
                  else [False] * molecule.n_atoms)
        named = np.fromiter(
            (i for i, (name, is_het) in enumerate(zip(names, hetatm))
             if not is_het and name.strip().upper() in allowed),
            dtype=np.int64,
        )
        if len(named):
            return named
    return _topology_backbone(molecule, include_beta_carbon)


__all__ = ["peptide_backbone"]
