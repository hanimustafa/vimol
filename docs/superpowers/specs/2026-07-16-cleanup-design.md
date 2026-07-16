# 'c' cleanup: relax steric clashes and stretched manual bonds

**Date:** 2026-07-16
**Status:** Approved — amended by Revision 2 (see end): editor-intended
bonds become manual bonds, angle (geometry) springs, animated stepping.

## Problem

Bond perception is distance-based, so atoms added by editing can appear
"bonded" to parts of the molecule they merely landed near — a steric-clash
false positive the editor knows is not intended connectivity. Conversely,
option-drag manual bonds can be far longer than a real bond. Pressing `c`
relaxes the geometry with a deliberately simple spring formula that moves
the *new* segment much more than the old one: spurious too-close contacts
get pushed apart until their false bond disappears, and stretched manual
bonds get pulled toward a real bond length. The status bar hints
`press c to cleanup` whenever either condition exists.

Keep the implementation simple. This is a cosmetic relaxation, not a force
field.

## Tracking what is "new"

- `Molecule.new_atoms: Set[int]` (new dataclass field, `default_factory=set`)
  — indices of atoms created by editing since load (or since the last
  cleanup "accepted" the geometry).
- `editor.py` marks every atom it creates: the caps placed by `_cap`, the
  centers from `birth_molecule`, the hydrogens added by `replace_atom`'s
  valence fill, and the promoted hydrogen in `_promote_hydrogen` (it moved
  and changed element — it belongs to the new segment). `replace_atom`'s
  relabeled anchor keeps its position and is NOT marked.
- `editor.delete_atom` remaps `new_atoms` across the row removal with the
  same keep-mask mapping it already applies to `manual_bonds`.
- Widget undo snapshots grow a fifth element: `(symbols, positions, bonds,
  manual_bonds, new_atoms)`; `undo()` restores `mol.new_atoms = set(...)`.
- `new_atoms` does NOT join `_signature()` — cleanup changes positions,
  which the signature already sees.

## Detecting cleanup targets

Pure function in `editor.py`, recomputed from current state (no event
bookkeeping — the hint is always honest):

```python
def cleanup_targets(mol) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """(clash_pairs, stretched_manual) that pressing 'c' would fix."""
```

- **Clash pair:** a perceived bond `(i, j, _)` where at least one endpoint
  is in `new_atoms`, the pair is not in `manual_bonds`, and its length
  deviates from the ideal `covalent_radius(i) + covalent_radius(j)` by
  more than 0.05 Å. Rationale: the editor places every bond it *intends*
  at exactly the ideal length, and nothing moves atoms afterward — so a
  new-atom bond at a non-ideal length can only be a proximity accident.
  Old-old bonds are never flagged (loaded geometry is not ours to judge).
- **Stretched manual bond:** `(i, j)` in `manual_bonds` with length
  greater than `ideal + 0.45` (the perception tolerance) — i.e. a bond
  that only exists because it is manual.

Both empty → nothing to clean.

## The relaxation

```python
def cleanup(mol, iterations: int = 100, step: float = 0.2) -> bool:
```

1. Compute `cleanup_targets`; if both lists are empty, return False.
2. Build one spring per current bonded pair, with target lengths fixed
   once, up front:
   - clash pairs → `ideal + 0.45 + 0.2` (pushed just past the perception
     cutoff, so the false bond disappears on re-perception);
   - stretched manual bonds → `ideal` (pulled to a real bond length);
   - every other current bond → its *current* length (holds the existing
     geometry together — NOT the covalent ideal, which would distort
     loaded real-world structures).
3. Iterate `iterations` times: for each spring compute the axial force
   `(L - target) * unit(i->j)` (attractive when too long, repulsive when
   too short; skip pairs with `L < 1e-6`), accumulate `+f` on `i` and
   `-f` on `j`, then move every atom by `step * weight * force`, where
   `weight = 1.0` for atoms in `new_atoms` and `0.15` otherwise — the new
   segment does almost all the moving, but old atoms are not frozen solid
   (a fully pinned clash could be unresolvable).
4. `_reperceive(mol)`, then `mol.new_atoms.clear()` — the relaxed
   geometry is "accepted": repeated `c` presses do not keep kneading the
   molecule, and the hint disappears.
5. Return True.

## Key & UI wiring

- `'c'` joins `_EDIT_DRIVER_KEYS` in `viewer.py` (claimed only when
  editable; it is currently unbound in both keymaps).
- `_driver_key`: `elif key == "c" and self.editable:` → `return
  self.widget.cleanup()`.
- New widget method `cleanup() -> bool`: take a snapshot, call
  `editor.cleanup(mol)`; if it returns True, commit the snapshot to the
  undo stack (the same snapshot-then-commit-on-change pattern
  `_end_bond_gesture` uses), `_refresh_dirty()`, and return True; if
  False, discard the snapshot and return False.
- **Hint:** `_status_bar` appends a trailer piece ` ⚠ c cleanup` (warm
  warning color, e.g. orange) when `self.editable` and
  `editor.cleanup_targets(mol)` reports anything. Appearing/disappearing
  shifts the trailer exactly like `[MODIFIED]` already does; it is a
  function of model state, not hover state, so the button-span stability
  tests are unaffected.
- `_HELP_EDIT` gains `  c .................. cleanup clashes / long bonds`;
  README's editing section gets a matching row and a short paragraph.

## Testing

New cases in `tests/test_editing.py`:

- Marking: `birth_molecule` / `grow_at_atom` / `replace_atom` populate
  `new_atoms` with exactly the created (and promoted) indices;
  `delete_atom` remaps them.
- `cleanup_targets`: a new atom placed within bonding range of an
  unrelated old atom is flagged as a clash; the editor's own intended
  bonds are not; an old-old close pair is not; a manual bond beyond the
  threshold is flagged as stretched; a short manual bond is not.
- `cleanup`: after `c`, a clash pair's distance exceeds the perception
  cutoff and the false bond is gone from `mol.bonds`; old atoms moved
  far less than new atoms (compare max displacements); a stretched
  manual bond ends near ideal length and stays in `mol.bonds`;
  `new_atoms` is empty afterward; returns False (and moves nothing)
  when there is nothing to clean.
- Widget/viewer: `'c'` triggers cleanup when editable, is undoable
  (`u` restores pre-cleanup positions and `new_atoms`), sets `dirty`;
  `'c'` does nothing in a read-only viewer; the status-bar hint appears
  when a clash exists, appears after drawing an over-long manual bond,
  and disappears after pressing `c`.

## Revision 2 (2026-07-16, post-review by user)

Three defects in the shipped v1, fixed together:

### 2a. Editor-intended bonds join `manual_bonds`

Editing *knows* the connectivity it creates; that knowledge was thrown
away. Every bond the editor deliberately makes is now recorded through
the same store as option-drag bonds:

- `_cap` records `(center_idx, cap_idx)` for every cap it places — it
  needs the center's index, so its signature gains one (every caller
  already has it).
- `birth_molecule`: center ↔ each cap. `_promote_hydrogen`: parent ↔
  promoted atom, promoted ↔ its caps. `replace_atom`: anchor ↔ each
  fill hydrogen (the anchor's pre-existing bonds stay perception-owned).
- Recording appends to `mol.manual_bonds` (normalized `i < j`, deduped)
  with the op's single existing `_reperceive` picking up the union — do
  not re-perceive once per bond.

**Consequence — clash detection simplifies and hardens:** a clash pair
is now simply a perceived bond with at least one endpoint in
`new_atoms` whose pair is NOT in `manual_bonds`. The
deviation-from-ideal heuristic (`_CLASH_SLOP`) is deleted — intent is
now recorded, not inferred, so even an accidental contact at exactly
ideal length is correctly flagged.

### 2b. Angle springs enforce local geometry

Bond-length springs alone let angles collapse. Add classic 1-3
(Urey-Bradley) springs, reusing the template registry's geometry:

- For each **center** atom `k` in `new_atoms` ∪ {endpoints of stretched
  manual bonds}: look up `TEMPLATES.get((symbol(k), n_neighbors(k)))`;
  no entry (hypervalent, unknown) → skip that center. The ideal angle θ
  is the angle between the template's first two directions (109.47°
  tetrahedral, 120° trigonal, 180° linear, etc. — uniform across pairs
  for every registered template).
- For each pair of bonded neighbors `(a, b)` of `k`: add a spring
  between `a` and `b` targeting the law-of-cosines distance
  `sqrt(La² + Lb² − 2·La·Lb·cosθ)`, where `La`/`Lb` are the bond
  springs' target lengths for `(k,a)` and `(k,b)`.
- Angle springs act at **half stiffness** (scale their force by 0.5) —
  they shape, the bond springs place; halving also keeps the iteration
  stable for 4-coordinate centers, which gain 3 extra springs per atom.
- Old, uninvolved centers get no angle springs — loaded geometry is
  still not ours to judge.

### 2c. The relaxation animates

`c` no longer teleports atoms; the user watches the molecule settle
(~half a second). The monolithic `editor.cleanup` splits:

- `cleanup_prepare(mol) -> Optional[RelaxState]` — computes targets;
  None when nothing to fix. `RelaxState` (small dataclass in
  `editor.py`) carries the fixed spring list (bond + angle, each with
  target and stiffness) and the per-atom weight vector.
- `cleanup_advance(mol, state, iterations=4, step=0.15) -> float` —
  runs a few spring iterations in place, returns the max per-atom
  displacement of the call (convergence signal).
- `cleanup_finish(mol)` — `_reperceive` + `new_atoms.clear()`. Bonds
  are NOT re-perceived mid-animation: the clash bond visibly stretches
  as the fragments separate and pops off at the end.
- `cleanup(mol)` stays as the convenience that runs
  prepare → advance×N → finish (API compat + non-animated tests).

Widget:

- `start_cleanup() -> bool`: snapshot, `cleanup_prepare`; None → False
  (discard snapshot). Otherwise commit the snapshot to the undo stack
  immediately (one `u` undoes the whole animation), store the state and
  a frame budget (~30 frames), return True.
- `cleanup_active` property; `cleanup_tick() -> bool`: one
  `cleanup_advance` call per frame; finish (via `cleanup_finish` +
  `_refresh_dirty`) when the budget runs out or displacement drops
  below 1e-3, returning False from then on.
- `undo()` and `set_molecule()` cancel an in-flight animation (drop the
  state, no finish) — same lifecycle discipline as the bond gesture.
  A `c` press while an animation is active is ignored.

Viewer: `'c'` calls `start_cleanup()`; the run loop gains an
autospin-style block — while `widget.cleanup_active`, call
`cleanup_tick()` each frame, mark `changed`, and refresh
`_last_interact` so supersampling stays in fast mode during the motion.

### Revision-2 tests

- Intended-bond recording: after `birth_molecule`, center↔cap pairs are
  in `manual_bonds`; after `grow_at_atom` on an H, parent↔promoted and
  promoted↔cap pairs are; after `replace_atom` valence fill, anchor↔H
  pairs are.
- Hardened clash detection: a new atom placed at *exactly* ideal
  bonding distance from an unrelated old atom is still flagged (the old
  slop heuristic would have missed it).
- Angle enforcement: a deliberately distorted new fragment (e.g. CH4
  with one H squeezed to ~70° of another) relaxes to within a few
  degrees of 109.47° after `cleanup`.
- Animation: `start_cleanup()` returns True and `cleanup_active` is
  True; repeated `cleanup_tick()` calls produce at least two distinct
  intermediate position states before returning False; after finish the
  clash is resolved and `new_atoms` is empty; a single `u` restores the
  pre-cleanup positions; `undo()` mid-animation cancels it (no further
  ticks mutate); `start_cleanup()` with nothing to fix returns False and
  pushes no undo entry.
- Viewer: `KeyEvent("c")` dispatch starts the animation
  (`widget.cleanup_active`); the existing hint/read-only tests keep
  passing.

## Revision 3 (2026-07-16, post-review by user)

**Defect:** cleanup does not spread the neighbors of a center whose
coordination number has no registered template (e.g. a 5-coordinate
carbon made by linking two methanes). `_build_springs` looks up
`TEMPLATES[(element, n_neighbors)]` and `continue`s past the center when
there is no entry, so no 1-3 springs are added and the hydrogens stay
clumped where they were built.

**Fix — repulsion fallback via θ = 180°.** The angle springs already
target the law-of-cosines chord
`sqrt(La² + Lb² − 2·La·Lb·cos θ)`. "Spread the neighbors apart by mutual
repulsion" is exactly `cos θ = −1` (θ = 180°), which makes each pair's
target `La + Lb` — the antipodal chord. Since the real chord can never
exceed `La + Lb` (triangle inequality), such a spring only ever pushes
apart, and the equilibrium where every pair pushes equally is the
maximally-spread arrangement: trigonal-bipyramidal for 5, octahedral for
6, and so on (VSEPR as point repulsion, emergent from one setpoint).

So in `_build_springs`, at the center loop:

- `tmpl = TEMPLATES.get((symbol(k), len(neigh)))`.
- Registered (`tmpl is not None`): unchanged — `cos_t =
  dot(tmpl.directions[0], tmpl.directions[1])` (exact ideal angle;
  tetrahedral stays exactly 109.47°).
- Unregistered (`tmpl is None`): `cos_t = -1.0` instead of `continue`.

Everything downstream is identical: same `t = sqrt(la² + lb² − 2·la·lb·
cos_t)`, same half stiffness (`_ANGLE_STIFFNESS`), same
clash-pair/ring/`angle_targets` handling, same per-atom weights, same
"only centers in `new_atoms` ∪ stretched-manual endpoints" scope (loaded
geometry still untouched). No coordination cap — repulsion handles
arbitrary N harmlessly.

### Revision-3 tests

- A 5-coordinate carbon (build two methanes far apart, `add_manual_bond`
  their carbons, `cleanup`) ends with its five neighbors spread out: the
  minimum pairwise neighbor angle at the carbon rises well above the
  clumped starting value, toward the ~90° floor of a trigonal
  bipyramid (assert, say, min angle > 80°).
- Regression: a plain tetrahedral carbon (`birth_molecule`, distort one
  H into a clash, `cleanup`) still relaxes to within a couple degrees of
  109.47° — the registered path is unchanged.

## Revision 4 (2026-07-16, post-review by user)

**Principle: connectivity is frozen input to a cleanup.** At the moment
'c' is pressed, the connectivity is fully known — the implicit
(distance-perceived) bonds of old atoms plus the explicit manual bonds
of edited atoms, as they stand in `mol.bonds`. Cleanup must NEVER change
that connectivity, implicitly or explicitly, beyond the one change it
exists to make: removing the identified clash pairs. In particular, two
hydrogens the relaxation happens to push within bonding range must NOT
gain a bond.

**Defect fixed:** `cleanup_finish` re-perceived bonds from the relaxed
geometry, so relaxation motion could mint phantom bonds (drifted-close
pairs) or silently drop stretched ones.

**Fix:** `cleanup_prepare` freezes `final_bonds` — the press-time bond
list minus the clash pairs — into the `RelaxState`, and
`cleanup_finish(mol, state)` assigns it verbatim instead of
re-perceiving. Consequences the tests pin down: no bond is ever added
for drifted atoms; no press-time bond is ever dropped however far it
stretched; clash bonds are removed even if the frame budget ran out
before their pair fully cleared the perception cutoff.
