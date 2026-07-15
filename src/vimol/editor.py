"""In-place structure editing: grow fragments and birth new molecules.

These operations mutate a :class:`~vimol.molecule.Molecule` and re-perceive
its bonds. Geometry (which directions to place neighbors in) comes from
:mod:`vimol.templates`; bond *lengths* are the sum of covalent radii, matching
how :mod:`vimol.bonds` perceives connectivity.

The public operations mirror the interactive gestures:

* :func:`birth_molecule` -- click empty space -> a fresh capped atom (methane).
* :func:`grow_at_atom`   -- click an atom     -> edit the structure there.
* :func:`replace_atom`   -- the heavy-atom half of ``grow_at_atom``.

``grow_at_atom`` splits on what was clicked: a hydrogen is *promoted* to the
building element and its freed valences capped ("click an H, it becomes a
carbon with three new hydrogens"); a heavier atom is *replaced* in place by
the building element, snapping its terminal hydrogens and topping up its
valency with new ones.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .molecule import Molecule
from .bonds import perceive_bonds
from . import elements
from . import templates
from .templates import AtomTemplate


def _bond_length(a: str, b: str) -> float:
    return elements.covalent_radius(a) + elements.covalent_radius(b)


def _neighbors(mol: Molecule, idx: int) -> List[int]:
    out = []
    for i, j, _order in mol.bonds:
        if i == idx:
            out.append(j)
        elif j == idx:
            out.append(i)
    return out


def _reperceive(mol: Molecule) -> None:
    """Recompute distance-based bonds after a geometry change.

    Every edit ends here, which would otherwise silently erase manual bonds
    (see :func:`add_manual_bond`) -- so the distance-perceived list is unioned
    with ``mol.manual_bonds``, deduped on the normalized ``(i, j)`` pair, so a
    manual bond that also happens to be within auto range is never listed
    twice.
    """
    bonds = perceive_bonds(mol)
    seen = {(i, j) for i, j, _order in bonds}
    for i, j, order in mol.manual_bonds:
        if (i, j) not in seen:
            bonds.append((i, j, order))
            seen.add((i, j))
    mol.bonds = bonds


def add_manual_bond(mol: Molecule, i: int, j: int, order: int = 1) -> bool:
    """Record an explicit bond between *i* and *j* that survives re-perception.

    Normalizes to ``i < j``. Returns False (no change) if ``i == j`` or the
    pair is already in ``mol.manual_bonds``; otherwise appends and
    re-perceives (which unions it into ``mol.bonds``) and returns True. If
    the pair is already within the auto-perception distance, the manual bond
    is still recorded -- that is what makes the drawn bond robust to later
    geometry changes that pull the atoms apart.
    """
    if i == j:
        return False
    if i > j:
        i, j = j, i
    if any(a == i and b == j for a, b, _order in mol.manual_bonds):
        return False
    mol.manual_bonds.append((i, j, order))
    _reperceive(mol)
    return True


def _cap(mol: Molecule, center: np.ndarray, dirs: np.ndarray,
         center_elem: str, cap_elem: str) -> None:
    """Place a *cap_elem* atom along each unit direction in *dirs*.

    Every atom placed here is editor-created, so it joins ``mol.new_atoms``
    -- this single spot covers the caps for :func:`birth_molecule`,
    :func:`replace_atom`'s valence fill, and :func:`_promote_hydrogen`'s
    freed valences all at once.
    """
    r = _bond_length(center_elem, cap_elem)
    for d in dirs:
        p = center + np.asarray(d, float) * r
        idx = mol.add_atom(cap_elem, p[0], p[1], p[2])
        mol.new_atoms.add(idx)


# Sample density for the fallback direction search: fine enough that the
# worst-case angular error (~7 deg) is invisible at bond scale, still cheap.
_FILL_SPHERE_SAMPLES = 256


def _fibonacci_sphere(n: int) -> np.ndarray:
    """*n* roughly-evenly distributed unit vectors."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def _fill_directions(existing, n: int) -> np.ndarray:
    """Pick *n* unit directions, one at a time, each far from all others.

    ``templates.free_direction`` handles the generic VSEPR-ish case, but it
    degenerates on symmetric arrangements (it can return a direction that is
    already occupied). When its answer lies within 60 degrees of an occupied
    direction we instead take the sphere point maximizing the minimum angle
    to everything placed so far.
    """
    dirs = [templates._normalize(d) for d in existing]
    out = []
    for _ in range(n):
        d = templates.free_direction(dirs)
        if dirs and max(float(np.dot(d, e)) for e in dirs) > 0.5:   # cos 60 deg
            pts = _fibonacci_sphere(_FILL_SPHERE_SAMPLES)
            worst = (pts @ np.array(dirs).T).max(axis=1)  # closest occupied, per point
            d = pts[int(np.argmin(worst))]
        out.append(d)
        dirs.append(d)
    return np.array(out)


def replace_atom(mol: Molecule, idx: int, element: str = "C",
                 template: Optional[AtomTemplate] = None) -> int:
    """Replace atom *idx* with *element*, keeping it anchored in place.

    The atom is relabeled where it stands. Terminal hydrogen neighbors are
    snapped to the new element-H bond length along their existing directions;
    heavy neighbors never move. If coordination is below the template's
    valence, hydrogens are added on free sites until it is met; excess
    coordination is left alone (hypervalency is the user's business).

    Returns *idx*.
    """
    element = elements.normalize_symbol(element)
    tmpl = template or templates.default_template(element)
    pos = mol.positions[idx].copy()
    neigh = _neighbors(mol, idx)
    mol.symbols[idx] = element
    for j in neigh:
        if mol.symbols[j] == "H" and _neighbors(mol, j) == [idx]:
            d = templates._normalize(mol.positions[j] - pos)
            mol.positions[j] = pos + d * _bond_length(element, "H")
    n_add = tmpl.valence - len(neigh)
    if n_add > 0:
        if not neigh:       # lone atom: full template, like birth_molecule
            _cap(mol, pos, tmpl.free_directions(), element, tmpl.cap)
        else:
            existing = [templates._normalize(mol.positions[j] - pos) for j in neigh]
            _cap(mol, pos, _fill_directions(existing, n_add), element, tmpl.cap)
    _reperceive(mol)
    return idx


def delete_atom(mol: Molecule, idx: int) -> None:
    """Remove atom *idx* (and its own terminal hydrogens) from the molecule.

    Terminal-H neighbors of *idx* -- hydrogens bonded to nothing but *idx* --
    are swept away with it. That one rule covers both gestures: deleting a
    heavy atom takes its whole hydrogen shell (methane's carbon removes the
    CH4); deleting a hydrogen has no hydrogens of its own, so only it goes.

    Bonds are re-perceived rather than reindexed, and no auto-capping is done:
    heavy neighbors that lose a bond are left dangling and under-coordinated
    (deletion is a plain erase, not a substitution like :func:`replace_atom`).
    """
    victims = {idx}
    for j in _neighbors(mol, idx):
        if mol.symbols[j] == "H" and _neighbors(mol, j) == [idx]:
            victims.add(j)
    keep = [i for i in range(mol.n_atoms) if i not in victims]
    # Remap manual bonds across the row removal *before* rebuilding symbols/
    # positions below: drop any pair touching a deleted atom, and shift the
    # survivors' indices old -> new via the same mapping `keep` implies.
    # `keep` is ascending, so remapping never reorders a surviving pair.
    remap = {old: new for new, old in enumerate(keep)}
    mol.manual_bonds = [
        (remap[i], remap[j], order)
        for i, j, order in mol.manual_bonds
        if i not in victims and j not in victims
    ]
    mol.new_atoms = {remap[i] for i in mol.new_atoms if i not in victims}
    mol.symbols = [mol.symbols[i] for i in keep]
    # fancy-index the surviving rows (a copy); empty keep -> a clean (0, 3) array
    mol.positions = mol.positions[keep]
    _reperceive(mol)


def birth_molecule(mol: Molecule, position, element: str = "C",
                   template: Optional[AtomTemplate] = None) -> int:
    """Create a fresh, fully-capped atom at *position* (e.g. methane).

    Returns the index of the central atom. The central atom gets one cap in
    every template direction, so carbon becomes CH4, oxygen becomes water, etc.
    """
    tmpl = template or templates.default_template(element)
    p = np.asarray(position, float)
    center = mol.add_atom(element, p[0], p[1], p[2])
    mol.new_atoms.add(center)
    _cap(mol, p, tmpl.free_directions(), element, tmpl.cap)
    _reperceive(mol)
    return center


def _promote_hydrogen(mol: Molecule, h_idx: int, element: str,
                      template: AtomTemplate) -> int:
    """Turn a hydrogen into *element*, repositioned and capped.

    The H is bonded to some parent P. We move it out to the correct
    ``element``-P bond length along the same P->H direction, relabel it, and cap
    its freed valences. With no parent (a lone H) it simply becomes the center
    of a fresh capped atom in place.
    """
    neigh = _neighbors(mol, h_idx)
    if not neigh:
        # lone hydrogen: reuse it as the center of a free molecule
        mol.symbols[h_idx] = elements.normalize_symbol(element)
        mol.new_atoms.add(h_idx)     # moved and changed element -> new segment
        _cap(mol, mol.positions[h_idx].copy(), template.free_directions(),
             element, template.cap)
        _reperceive(mol)
        return h_idx
    parent = neigh[0]
    parent_pos = mol.positions[parent].copy()
    parent_elem = mol.symbols[parent]
    d = templates._normalize(mol.positions[h_idx] - parent_pos)  # parent -> H
    center = parent_pos + d * _bond_length(parent_elem, element)
    mol.symbols[h_idx] = elements.normalize_symbol(element)
    mol.positions[h_idx] = center
    mol.new_atoms.add(h_idx)         # moved and changed element -> new segment
    opens = template.open_directions(-d)
    _cap(mol, center, opens, element, template.cap)
    _reperceive(mol)
    return h_idx


def grow_at_atom(mol: Molecule, idx: int, element: str = "C",
                 template: Optional[AtomTemplate] = None) -> int:
    """Edit the structure at atom *idx* with the selected *element*.

    * A hydrogen is *promoted*: it becomes the new element (moved to the right
      bond length) and its freed valences are capped -- "click an H, it turns
      into a carbon with three new hydrogens".
    * Any heavier atom is *replaced* in place -- see :func:`replace_atom`.

    Returns the index of the resulting central atom.
    """
    tmpl = template or templates.default_template(element)
    if mol.symbols[idx] == "H":
        return _promote_hydrogen(mol, idx, element, tmpl)
    return replace_atom(mol, idx, element, tmpl)


# Perception's own distance tolerance (see bonds.perceive_bonds) -- a manual
# bond only counts as "stretched" once it exceeds even that generous margin.
_PERCEPTION_TOLERANCE = 0.45
# How far a clash pair is pushed past the perception cutoff during cleanup,
# so the false bond is unambiguously gone (not just barely) on re-perception.
_CLASH_CLEARANCE = 0.2
# A new-atom bond within this many angstrom of the ideal length is treated as
# an editor-intended bond, not a proximity accident.
_CLASH_SLOP = 0.05


def cleanup_targets(mol: Molecule) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """(clash_pairs, stretched_manual) that pressing 'c' would fix.

    A pure, cheap function of current state -- no event bookkeeping -- so
    callers (e.g. the status-bar hint) can call it every render and always
    get an honest answer.

    * clash pair: a perceived bond with at least one endpoint in
      ``mol.new_atoms``, not itself a manual bond, whose length deviates from
      the covalent-sum ideal by more than :data:`_CLASH_SLOP`. The editor
      places every bond it *intends* at exactly the ideal length and nothing
      moves atoms afterward, so a new-atom bond at a non-ideal length can
      only be a proximity accident. Old-old bonds are never flagged --
      loaded geometry is not ours to judge.
    * stretched manual bond: a manual bond longer than
      ``ideal + _PERCEPTION_TOLERANCE`` -- i.e. one that only exists because
      it is manual, not because the atoms are actually close.
    """
    manual_set = {(i, j) for i, j, _order in mol.manual_bonds}
    clash_pairs: List[Tuple[int, int]] = []
    for i, j, _order in mol.bonds:
        if i not in mol.new_atoms and j not in mol.new_atoms:
            continue
        if (i, j) in manual_set:
            continue
        ideal = _bond_length(mol.symbols[i], mol.symbols[j])
        length = float(np.linalg.norm(mol.positions[i] - mol.positions[j]))
        if abs(length - ideal) > _CLASH_SLOP:
            clash_pairs.append((i, j))

    stretched: List[Tuple[int, int]] = []
    for i, j, _order in mol.manual_bonds:
        ideal = _bond_length(mol.symbols[i], mol.symbols[j])
        length = float(np.linalg.norm(mol.positions[i] - mol.positions[j]))
        if length > ideal + _PERCEPTION_TOLERANCE:
            stretched.append((i, j))

    return clash_pairs, stretched


def cleanup(mol: Molecule, iterations: int = 100, step: float = 0.2) -> bool:
    """Relax steric clashes and stretched manual bonds ('c').

    A deliberately simple fixed-target spring relaxation -- a cosmetic tidy-
    up, not a force field. One spring per currently bonded pair:

    * clash pairs are pushed to just past the perception cutoff, so the
      false bond they created disappears on re-perception;
    * stretched manual bonds are pulled to the covalent-sum ideal, so they
      read as a real bond;
    * every other current bond holds its *current* length -- not the
      covalent ideal, which would distort a loaded real-world structure --
      simply keeping the rest of the molecule from falling apart while the
      two groups above move.

    Atoms in ``mol.new_atoms`` move ~7x more than the rest (weight 1.0 vs
    0.15): the new segment does almost all the moving, but old atoms are not
    frozen solid (a fully pinned clash could be unresolvable).

    Returns False (no change) if :func:`cleanup_targets` finds nothing to
    fix; otherwise relaxes, re-perceives, "accepts" the result by clearing
    ``mol.new_atoms``, and returns True.
    """
    clash_pairs, stretched = cleanup_targets(mol)
    if not clash_pairs and not stretched:
        return False

    handled = set(clash_pairs) | set(stretched)
    springs: List[Tuple[int, int, float]] = []
    for i, j in clash_pairs:
        ideal = _bond_length(mol.symbols[i], mol.symbols[j])
        springs.append((i, j, ideal + _PERCEPTION_TOLERANCE + _CLASH_CLEARANCE))
    for i, j in stretched:
        springs.append((i, j, _bond_length(mol.symbols[i], mol.symbols[j])))
    for i, j, _order in mol.bonds:
        if (i, j) in handled:
            continue
        springs.append((i, j, float(np.linalg.norm(mol.positions[i] - mol.positions[j]))))

    weight = np.array([1.0 if k in mol.new_atoms else 0.15 for k in range(mol.n_atoms)])
    pos = mol.positions
    for _ in range(iterations):
        forces = np.zeros_like(pos)
        for i, j, target in springs:
            delta = pos[j] - pos[i]
            length = float(np.linalg.norm(delta))
            if length < 1e-6:      # perception already rejects sub-0.4A contacts
                continue
            f = (length - target) * (delta / length)
            forces[i] += f
            forces[j] -= f
        pos = pos + step * weight[:, None] * forces
    mol.positions = pos

    _reperceive(mol)
    mol.new_atoms.clear()
    return True
