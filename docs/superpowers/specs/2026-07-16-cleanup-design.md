# 'c' cleanup: relax steric clashes and stretched manual bonds

**Date:** 2026-07-16
**Status:** Approved

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
