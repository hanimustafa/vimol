"""In-place structure editing: grow fragments and birth new molecules.

These operations mutate a :class:`~vimol.molecule.Molecule` and re-perceive
its bonds. Geometry (which directions to place neighbors in) comes from
:mod:`vimol.templates`; bond *lengths* are the sum of covalent radii, matching
how :mod:`vimol.bonds` perceives connectivity.

The three public operations mirror the interactive gestures:

* :func:`birth_molecule` -- click empty space -> a fresh capped atom (methane).
* :func:`grow_at_atom`   -- click an atom     -> extend the structure there.

``grow_at_atom`` special-cases hydrogen exactly as the spec describes: the H is
promoted to the building element (carbon) and its freed valences are capped, so
"click an H, it becomes a carbon with three new hydrogens".
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


def _grow_onto(mol: Molecule, parent_idx: int, element: str,
               template: AtomTemplate) -> int:
    """Attach a new *element* atom to an existing parent atom, capped."""
    parent_pos = mol.positions[parent_idx].copy()
    parent_elem = mol.symbols[parent_idx]
    neigh = _neighbors(mol, parent_idx)
    neighbor_dirs = [
        templates._normalize(mol.positions[j] - parent_pos) for j in neigh
    ]
    site = templates.free_direction(neighbor_dirs)
    center = parent_pos + site * _bond_length(parent_elem, element)
    new_idx = mol.add_atom(element, center[0], center[1], center[2])
    # cap the new atom's remaining valences, its attachment slot facing back
    opens = template.open_directions(-site)
    _cap(mol, center, opens, element, template.cap)
    _reperceive(mol)
    return new_idx


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
    """Extend the structure at atom *idx* with a fragment of *element*.

    * A hydrogen is *promoted*: it becomes the new element (moved to the right
      bond length) and its freed valences are capped -- "click an H, it turns
      into a carbon with three new hydrogens".
    * Any heavier atom gets a new capped *element* atom bonded to a free site.

    Returns the index of the resulting central atom.
    """
    tmpl = template or templates.default_template(element)
    if mol.symbols[idx] == "H":
        return _promote_hydrogen(mol, idx, element, tmpl)
    return _grow_onto(mol, idx, element, tmpl)
