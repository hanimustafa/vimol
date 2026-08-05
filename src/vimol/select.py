"""Fast, stateless atom-selection helpers."""
from __future__ import annotations

import numpy as np

from typing import List, Optional, Tuple

from .molecule import Molecule
from .bonds import perceive_bonds


_PEPTIDE_BACKBONE = frozenset(("N", "CA", "C"))
# Long enough to tolerate ordinary carboxylate C--O distances, but shorter
# than a C--OH single bond (about 1.43 A). This matters for hydrogen-free
# threonine/serine: their terminal O must not make Cbeta look carbonyl-like.
_CARBONYL_CO_MAX2 = 1.36 ** 2


def heavy_atoms(molecule: Molecule) -> np.ndarray:
    """Return every non-hydrogen atom in file order."""
    return np.fromiter(
        (i for i, symbol in enumerate(molecule.symbols)
         if symbol.strip().upper() != "H"),
        dtype=np.int64,
    )


def largest_ring_system(molecule: Molecule) -> np.ndarray:
    """Return the largest fused ring system in the heavy-atom bond graph.

    An atom lies on at least one cycle exactly when it survives the graph's
    2-core, so repeatedly stripping terminal atoms leaves the cyclic part and
    nothing else; the largest connected component of that core is the answer.
    Both passes are O(N+E).

    A depth-first cycle basis would be just as cheap but is not a well-defined
    answer: which fundamental cycles it finds depends on the file's atom
    order, so the same molecule written twice can yield different subsets.
    The 2-core does not, and it keeps fused systems (naphthalene, steroids,
    porphyrins) whole -- which is what a rigid-core RMSD subset wants anyway.

    Explicit bonds are preferred; unnamed formats fall back to the same
    spatially hashed bond perception used by the viewer.
    """
    n = molecule.n_atoms
    if n < 3:
        return np.empty(0, dtype=np.int64)

    bonds = molecule.bonds or perceive_bonds(molecule)
    adjacency = [set() for _ in range(n)]
    symbols = molecule.symbols
    for i, j, _order in bonds:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            continue
        if (symbols[i].strip().upper() == "H"
                or symbols[j].strip().upper() == "H"):
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)

    # Peel to the 2-core. Dropping a terminal atom can strand its neighbor,
    # so the walk continues until nothing is left below degree two.
    degree = np.fromiter((len(row) for row in adjacency), dtype=np.int64, count=n)
    alive = degree > 0
    pending = [i for i in range(n) if alive[i] and degree[i] < 2]
    while pending:
        atom = pending.pop()
        if not alive[atom]:
            continue
        alive[atom] = False
        for other in adjacency[atom]:
            if alive[other]:
                degree[other] -= 1
                if degree[other] < 2:
                    pending.append(other)

    # Every remaining component is a ring or a fused ring system; the smallest
    # possible one is a three-membered cycle.
    components = []
    seen = np.zeros(n, dtype=np.bool_)
    for root in range(n):
        if not alive[root] or seen[root]:
            continue
        seen[root] = True
        component = [root]
        stack = [root]
        while stack:
            atom = stack.pop()
            for other in adjacency[atom]:
                if alive[other] and not seen[other]:
                    seen[other] = True
                    component.append(other)
                    stack.append(other)
        components.append(tuple(sorted(component)))

    if not components:
        return np.empty(0, dtype=np.int64)
    # Stable tie-break: the earliest system in atom/file order wins.
    chosen = min(components, key=lambda system: (-len(system), system))
    return np.asarray(chosen, dtype=np.int64)


def peptide_motifs(molecule: Molecule) -> List[List[Tuple[int, int, int, Optional[int]]]]:
    """Runs of ``(N, Cα, C, O)`` motifs, in chain order, from geometry alone.

    This is what lets a format with no atom names -- an xyz file -- still be
    read as a protein: the motif is recognized from elements and connectivity,
    and the runs are linked through the real peptide bonds. ``O`` is the
    carbonyl oxygen, or None where the file has none (a chain's last residue
    written without its terminal oxygens).

    Shared by the selection presets and by ``residues.infer_residues``, which
    hangs side chains off the Cα of each motif. Returned as runs rather than a
    flat list because both callers need to know where one chain stops: a
    ribbon must not leap a chain break, and neither must residue numbering.
    """
    return _peptide_motifs(molecule)[0]


def _topology_backbone(molecule: Molecule, include_beta_carbon: bool) -> np.ndarray:
    """Infer N-Cα-C(-O) motifs from elements and connectivity in O(N+E)."""
    runs, neighbors, symbols = _peptide_motifs(molecule)
    selected = set()
    for run in runs:
        for nitrogen, alpha, carbonyl, _oxygen in run:
            # Backbone here deliberately means the peptide path. Carbonyl O is
            # a branch off that path and must not participate in an RMSD
            # subset.
            selected.update((nitrogen, alpha, carbonyl))
            if include_beta_carbon:
                sidechain = [j for j in neighbors[alpha]
                             if symbols[j] == "C" and j != carbonyl]
                if sidechain:
                    beta = min(
                        sidechain,
                        key=lambda j: float(np.dot(
                            molecule.positions[j] - molecule.positions[alpha],
                            molecule.positions[j] - molecule.positions[alpha])))
                    selected.add(beta)
    return np.asarray(sorted(selected), dtype=np.int64)


def _peptide_motifs(molecule: Molecule):
    """(runs, neighbors, symbols) -- the shared half of both callers' work."""
    n = molecule.n_atoms
    if n < 4:
        return [], [], molecule.symbols
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

    # Walk the peptide bonds into ordered runs: the successor of a motif is the
    # one whose N its carbonyl is bonded to. A motif nobody points at starts a
    # run, which is also how the N-terminus is found.
    successor, predecessor = {}, {}
    for i in kept:
        for neighbor in neighbors[motifs[i][2]]:
            for j in motif_by_nitrogen.get(neighbor, ()):
                if j in kept and j != i:
                    successor[i] = j
                    predecessor[j] = i
    runs, seen = [], set()
    starts = [i for i in sorted(kept) if i not in predecessor]
    for start in starts + [i for i in sorted(kept) if i not in seen]:
        if start in seen:
            continue
        run, at = [], start
        while at is not None and at not in seen:
            seen.add(at)
            nitrogen, alpha, carbonyl = motifs[at]
            oxygens = carbonyl_oxygens(carbonyl)
            run.append((nitrogen, alpha, carbonyl, oxygens[0] if oxygens else None))
            at = successor.get(at)
        if run:
            runs.append(run)
    return runs, neighbors, symbols


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


__all__ = ["heavy_atoms", "largest_ring_system", "peptide_backbone",
           "peptide_motifs"]
