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

from dataclasses import dataclass
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


def _record_intended_bond(mol: Molecule, i: int, j: int, order: int = 1) -> bool:
    """Append a deliberate bond to ``mol.manual_bonds`` without re-perceiving.

    Normalizes to ``i < j`` and dedupes; returns False (no change) when
    ``i == j`` or the pair is already recorded. Every bond the editor
    *intends* -- the ones it creates on purpose, as opposed to what distance
    perception happens to find -- goes through here, so cleanup can tell
    intended connectivity from proximity accidents. Callers are responsible
    for the (single) :func:`_reperceive` that unions these into ``mol.bonds``.
    """
    if i == j:
        return False
    if i > j:
        i, j = j, i
    if any(a == i and b == j for a, b, _order in mol.manual_bonds):
        return False
    mol.manual_bonds.append((i, j, order))
    return True


def add_manual_bond(mol: Molecule, i: int, j: int, order: int = 1) -> bool:
    """Record an explicit bond between *i* and *j* that survives re-perception.

    Normalizes to ``i < j``. Returns False (no change) if ``i == j`` or the
    pair is already in ``mol.manual_bonds``; otherwise appends and
    re-perceives (which unions it into ``mol.bonds``) and returns True. If
    the pair is already within the auto-perception distance, the manual bond
    is still recorded -- that is what makes the drawn bond robust to later
    geometry changes that pull the atoms apart.
    """
    if not _record_intended_bond(mol, i, j, order):
        return False
    _reperceive(mol)
    return True


def _cap(mol: Molecule, center_idx: int, center: np.ndarray, dirs: np.ndarray,
         center_elem: str, cap_elem: str) -> None:
    """Place a *cap_elem* atom along each unit direction in *dirs*.

    Every atom placed here is editor-created, so it joins ``mol.new_atoms``,
    and its bond to *center_idx* is deliberate, so it is recorded as a
    manual (intended) bond -- this single spot covers the caps for
    :func:`birth_molecule`, :func:`replace_atom`'s valence fill, and
    :func:`_promote_hydrogen`'s freed valences all at once.
    """
    r = _bond_length(center_elem, cap_elem)
    for d in dirs:
        p = center + np.asarray(d, float) * r
        idx = mol.add_atom(cap_elem, p[0], p[1], p[2])
        mol.new_atoms.add(idx)
        _record_intended_bond(mol, center_idx, idx)


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
            _cap(mol, idx, pos, tmpl.free_directions(), element, tmpl.cap)
        else:
            existing = [templates._normalize(mol.positions[j] - pos) for j in neigh]
            _cap(mol, idx, pos, _fill_directions(existing, n_add), element, tmpl.cap)
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
    # Surviving neighbors of a deleted atom just lost a bond -- their local
    # geometry may no longer be ideal (e.g. a tetrahedral center left with 3
    # neighbors still at 109.47 deg instead of a relaxed trigonal 120 deg),
    # so they need the same cleanup attention as a freshly created atom.
    affected = {j for v in victims for j in _neighbors(mol, v) if j not in victims}
    # Marking a survivor as needing cleanup makes its bonds eligible for
    # clash scrutiny (see cleanup_targets) -- but its OTHER bonds predate
    # this deletion and are untouched by it, so they can never be a
    # proximity accident. Promote them to manual now, before that
    # eligibility exists, or a later cleanup could mistake perfectly good
    # connectivity for a false bond and push it past the perception cutoff.
    for a in affected:
        for b in _neighbors(mol, a):
            if b not in victims:
                _record_intended_bond(mol, a, b)
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
    mol.new_atoms = {remap[i] for i in mol.new_atoms if i not in victims} \
        | {remap[j] for j in affected}
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
    _cap(mol, center, p, tmpl.free_directions(), element, tmpl.cap)
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
        _cap(mol, h_idx, mol.positions[h_idx].copy(), template.free_directions(),
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
    _record_intended_bond(mol, parent, h_idx)   # the bond being grown across
    opens = template.open_directions(-d)
    _cap(mol, h_idx, center, opens, element, template.cap)
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


def cleanup_targets(mol: Molecule) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """(clash_pairs, stretched_manual) that pressing 'c' would fix.

    A pure, cheap function of current state -- no event bookkeeping -- so
    callers (e.g. the status-bar hint) can call it every render and always
    get an honest answer.

    * clash pair: a perceived bond with at least one endpoint in
      ``mol.new_atoms`` whose pair is NOT in ``mol.manual_bonds``. Every
      bond the editor deliberately creates is recorded as a manual bond
      (see :func:`_record_intended_bond`), so intent is recorded rather
      than inferred: any unrecorded new-atom bond is a proximity accident,
      even one at exactly the ideal length. Old-old bonds are never
      flagged -- loaded geometry is not ours to judge.
    * stretched manual bond: a manual bond longer than
      ``ideal + _PERCEPTION_TOLERANCE`` -- i.e. one that only exists because
      it is manual, not because the atoms are actually close.
    """
    clash_pairs: List[Tuple[int, int]] = []
    for i, j, _order in mol.bonds:
        if i not in mol.new_atoms and j not in mol.new_atoms:
            continue
        if any(mi == i and mj == j for mi, mj, _mo in mol.manual_bonds):
            continue
        clash_pairs.append((i, j))

    stretched: List[Tuple[int, int]] = []
    for i, j, _order in mol.manual_bonds:
        ideal = _bond_length(mol.symbols[i], mol.symbols[j])
        length = float(np.linalg.norm(mol.positions[i] - mol.positions[j]))
        if length > ideal + _PERCEPTION_TOLERANCE:
            stretched.append((i, j))

    return clash_pairs, stretched


# Angle (1-3 / Urey-Bradley) springs act at half the bond springs' stiffness:
# they shape, the bond springs place -- and halving keeps the iteration stable
# for 4-coordinate centers, which gain several extra springs per atom.
_ANGLE_STIFFNESS = 0.5
# The unregistered-coordination repulsion fallback (theta=180) targets a
# configuration no arrangement of >2 points can actually reach -- every pair
# simultaneously antipodal is geometrically impossible past 2 neighbors -- so
# it never settles to zero force. Kept soft and repulsive-only (see
# cleanup_advance) so it stops pulling once neighbors are reasonably spread,
# instead of perpetually over-stretching the real bonds to those neighbors.
_REPULSION_STIFFNESS = 0.08


def _build_springs(mol: Molecule, clash_pairs, stretched) -> List[Tuple[int, int, float, float, bool]]:
    """The fixed spring set for one cleanup run: (i, j, target, stiffness, repulsive_only).

    Bond springs (stiffness 1.0, not repulsive-only), one per currently
    bonded pair:

    * clash pairs are pushed to just past the perception cutoff, so the
      false bond they created disappears on re-perception;
    * stretched manual bonds are pulled to the covalent-sum ideal, so they
      read as a real bond;
    * every other current bond holds its *current* length -- not the
      covalent ideal, which would distort a loaded real-world structure --
      simply keeping the rest of the molecule from falling apart while the
      two groups above move.

    Angle springs: classic 1-3 (Urey-Bradley) springs around each *center*
    in ``mol.new_atoms`` plus the endpoints of stretched manual bonds, so
    bond-length springs cannot collapse the local geometry. The center's
    template (``TEMPLATES[(symbol, n_neighbors)]``) gives the ideal angle
    theta, at :data:`_ANGLE_STIFFNESS`; a coordination with no registered
    template falls back to theta=180 (repulsion) at the much gentler
    :data:`_REPULSION_STIFFNESS`, and *repulsive-only* -- it never pulls a
    pair back together once they're already past the (unreachable) target,
    only pushes apart when closer. Each pair of the center's bonded
    neighbors gets a spring at the law-of-cosines distance for theta over
    the two bond springs' targets.
    Old, uninvolved centers get none -- loaded geometry is still not ours
    to judge. A neighbor pair that already carries a bond spring keeps it:
    a clash spring's target is raised to the angle target when the angle
    target is longer (one spring per pair, and the geometry still clears
    the perception cutoff); a real bond between the two neighbors (a ring)
    is never fought.
    """
    clash_set = set(clash_pairs)
    targets = {}                       # (i, j) -> bond-spring target, i < j
    for i, j in clash_pairs:
        ideal = _bond_length(mol.symbols[i], mol.symbols[j])
        targets[(i, j)] = ideal + _PERCEPTION_TOLERANCE + _CLASH_CLEARANCE
    for i, j in stretched:
        targets[(i, j)] = _bond_length(mol.symbols[i], mol.symbols[j])
    for i, j, _order in mol.bonds:
        if (i, j) not in targets:
            targets[(i, j)] = float(np.linalg.norm(mol.positions[i] - mol.positions[j]))

    neighbors = {}                     # adjacency over the current bond list
    for i, j, _order in mol.bonds:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)

    centers = set(mol.new_atoms)
    for i, j in stretched:
        centers.add(i)
        centers.add(j)

    angle_targets = {}                 # (a, b) -> (1-3 target, repulsive_only, stiffness)
    for k in sorted(centers):
        neigh = neighbors.get(k, [])
        if len(neigh) < 2:
            continue
        tmpl = templates.TEMPLATES.get((mol.symbols[k], len(neigh)))
        repulsive_only = tmpl is None
        if repulsive_only:
            # Unregistered coordination (e.g. a 5-coordinate carbon from
            # linking two methanes): fall back to pure repulsion. theta=180
            # makes each pair's law-of-cosines target the antipodal chord
            # La+Lb -- unreachable for >2 neighbors, so this floor is kept
            # repulsive-only and gentle (see cleanup_advance / the module
            # constant above).
            cos_t = -1.0
            stiff = _REPULSION_STIFFNESS
        else:
            cos_t = float(np.dot(tmpl.directions[0], tmpl.directions[1]))
            stiff = _ANGLE_STIFFNESS
        neigh = sorted(neigh)
        for x in range(len(neigh)):
            for y in range(x + 1, len(neigh)):
                a, b = neigh[x], neigh[y]
                la = targets[(k, a) if k < a else (a, k)]
                lb = targets[(k, b) if k < b else (b, k)]
                t = float(np.sqrt(la * la + lb * lb - 2.0 * la * lb * cos_t))
                if (a, b) in clash_set:
                    # the false bond joins two intended neighbors: let the
                    # geometry target win (it clears the cutoff by more)
                    targets[(a, b)] = max(targets[(a, b)], t)
                elif (a, b) in targets:
                    continue           # a real neighbor-neighbor bond (a ring)
                elif (a, b) not in angle_targets:
                    angle_targets[(a, b)] = (t, repulsive_only, stiff)

    springs = [(i, j, t, 1.0, False) for (i, j), t in targets.items()]
    springs += [(a, b, t, stiff, rep) for (a, b), (t, rep, stiff) in angle_targets.items()]
    return springs


@dataclass
class RelaxState:
    """An in-flight cleanup relaxation, fixed at :func:`cleanup_prepare` time.

    ``springs`` is the (i, j, target, stiffness, repulsive_only) list from
    :func:`_build_springs`; ``weights`` is the per-atom mobility vector (1.0
    for new atoms, 0.15 otherwise). Both stay constant across
    :func:`cleanup_advance` calls -- targets are never re-derived from the
    moving geometry.
    """
    springs: List[Tuple[int, int, float, float, bool]]
    weights: np.ndarray


# A spring is considered "already at target" below this displacement -- well
# under chemical significance, just enough to absorb floating-point noise.
_PREPARE_EPS = 1e-3


def cleanup_prepare(mol: Molecule) -> Optional[RelaxState]:
    """Set up a cleanup relaxation, or None when there is nothing to fix.

    Builds the full spring set (see :func:`_build_springs`) and freezes it,
    with per-atom weights, into a :class:`RelaxState` for
    :func:`cleanup_advance` to step through. Atoms in ``mol.new_atoms`` get
    weight 1.0 and the rest 0.15 (~7x less): the new/edited segment does
    almost all the moving, but old atoms are not frozen solid (a fully
    pinned clash could be unresolvable).

    Available any time there is a real spring to relax -- not just the
    urgent clash/stretched-manual-bond cases :func:`cleanup_targets` flags
    for the status-bar hint. A spring on an already-ideal bond or angle has
    its target equal to the current geometry by construction, so it never
    trips this check; one only fires when a bond is a clash/stretched, or
    when a center in ``mol.new_atoms`` (which :func:`delete_atom` also
    populates with a deletion's surviving neighbors) sits away from its
    ideal angle -- e.g. three leftover substituents still at a tetrahedral
    spacing after their fourth was deleted.
    """
    clash_pairs, stretched = cleanup_targets(mol)
    springs = _build_springs(mol, clash_pairs, stretched)
    pos = mol.positions

    def _spring_deviates(i, j, target, rep):
        length = float(np.linalg.norm(pos[j] - pos[i]))
        if rep and length >= target:    # repulsive-only: already spread enough
            return False
        return abs(length - target) > _PREPARE_EPS

    if not any(_spring_deviates(i, j, t, rep) for i, j, t, _stiff, rep in springs):
        return None
    weights = np.array([1.0 if k in mol.new_atoms else 0.15 for k in range(mol.n_atoms)])
    return RelaxState(springs=springs, weights=weights)


def cleanup_advance(mol: Molecule, state: RelaxState,
                    iterations: int = 4, step: float = 0.15) -> float:
    """Run a few spring iterations in place; return the max atom displacement.

    Each iteration accumulates the axial force ``stiffness * (L - target) *
    unit(i->j)`` per spring (attractive when too long, repulsive when too
    short; pairs with ``L < 1e-6`` are skipped) and moves every atom by
    ``step * weight * force``. A repulsive-only spring (the unregistered-
    coordination fallback -- see :func:`_build_springs`) contributes nothing
    once ``L >= target``: its target is a geometrically unreachable ideal
    for >2 neighbors, so without this it would pull forever and
    over-stretch the real bonds to those neighbors as a side effect. The
    return value -- the largest single-atom displacement over the whole
    call -- is the caller's convergence signal. Bonds are deliberately NOT
    re-perceived here: mid-animation the clash bond visibly stretches as
    the fragments separate, and pops off in :func:`cleanup_finish`.
    """
    pos = mol.positions
    start = pos.copy()
    for _ in range(iterations):
        forces = np.zeros_like(pos)
        for i, j, target, stiff, rep in state.springs:
            delta = pos[j] - pos[i]
            length = float(np.linalg.norm(delta))
            if length < 1e-6:      # perception already rejects sub-0.4A contacts
                continue
            deviation = length - target
            if rep and deviation >= 0:
                continue
            f = (stiff * deviation) * (delta / length)
            forces[i] += f
            forces[j] -= f
        pos = pos + step * state.weights[:, None] * forces
    mol.positions = pos
    if mol.n_atoms == 0:
        return 0.0
    return float(np.linalg.norm(pos - start, axis=1).max())


def cleanup_finish(mol: Molecule) -> None:
    """Accept a relaxed geometry: re-perceive bonds, clear ``new_atoms``.

    The false clash bonds -- now stretched past the perception cutoff --
    disappear here, and clearing ``new_atoms`` means repeated 'c' presses
    do not keep kneading the molecule (and the status-bar hint goes away).
    """
    _reperceive(mol)
    mol.new_atoms.clear()


def cleanup(mol: Molecule, iterations: int = 100, step: float = 0.2) -> bool:
    """Relax steric clashes and stretched manual bonds ('c'), in one shot.

    The convenience composition of :func:`cleanup_prepare` ->
    :func:`cleanup_advance` -> :func:`cleanup_finish` -- a deliberately
    simple fixed-target spring relaxation (see :func:`_build_springs`), not
    a force field. Interactive hosts that want the relaxation animated call
    the three stages themselves, a few iterations per frame.

    Returns False (no change) when there is nothing to fix; otherwise
    relaxes, re-perceives, "accepts" the result by clearing
    ``mol.new_atoms``, and returns True.
    """
    state = cleanup_prepare(mol)
    if state is None:
        return False
    cleanup_advance(mol, state, iterations=iterations, step=step)
    cleanup_finish(mol)
    return True
