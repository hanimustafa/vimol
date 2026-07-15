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

from typing import List, Optional

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
    """Recompute distance-based bonds after a geometry change."""
    mol.bonds = perceive_bonds(mol)


def _cap(mol: Molecule, center: np.ndarray, dirs: np.ndarray,
         center_elem: str, cap_elem: str) -> None:
    """Place a *cap_elem* atom along each unit direction in *dirs*."""
    r = _bond_length(center_elem, cap_elem)
    for d in dirs:
        p = center + np.asarray(d, float) * r
        mol.add_atom(cap_elem, p[0], p[1], p[2])


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
            pts = _fibonacci_sphere(256)
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
    tmpl = template or templates.default_template(element)
    element = elements.normalize_symbol(element)
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


def birth_molecule(mol: Molecule, position, element: str = "C",
                   template: Optional[AtomTemplate] = None) -> int:
    """Create a fresh, fully-capped atom at *position* (e.g. methane).

    Returns the index of the central atom. The central atom gets one cap in
    every template direction, so carbon becomes CH4, oxygen becomes water, etc.
    """
    tmpl = template or templates.default_template(element)
    p = np.asarray(position, float)
    center = mol.add_atom(element, p[0], p[1], p[2])
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
