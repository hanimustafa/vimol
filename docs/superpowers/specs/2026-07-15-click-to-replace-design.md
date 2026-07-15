# Click-to-replace on heavy atoms

**Date:** 2026-07-15
**Status:** Approved

## Summary

Change the editor's heavy-atom click gesture from *grow* to *replace*.
Clicking a heavy atom replaces it with the currently selected
element/geometry template. Clicking a hydrogen keeps its existing
*promote* behavior. Growth therefore happens only via hydrogen clicks
and empty-space clicks (the GaussView interaction model).

## Behavior

`replace_atom(mol, idx, element, template)` — new public operation in
`src/vimol/editor.py`:

1. **Relabel in place.** The clicked atom's symbol becomes the new
   element; its position is unchanged (it stays the anchor of the
   surrounding skeleton).
2. **Snap terminal hydrogens.** Each neighbor that is a hydrogen with
   no other bonds is repositioned along its existing bond direction to
   the correct new-element–H bond length (sum of covalent radii).
   Heavy neighbors are never moved.
3. **Fill valency.** If the atom's coordination (number of bonded
   neighbors) is less than `template.valence`, add hydrogens one at a
   time using `templates.free_direction` (VSEPR-ish placement away
   from existing neighbors) at the correct bond length, until
   coordination equals valency. `free_direction` degenerates for
   symmetric arrangements (it can return a direction already occupied
   by a previous cap); when its result lies within 60° of an existing
   bond direction, fall back to the direction maximizing the minimum
   angle to all occupied directions, searched over a unit sphere.
   A lone (unbonded) atom skips the iteration and caps with the
   template's full `free_directions()`, like `birth_molecule`.
   If coordination ≥ valency, add nothing — the relabel alone is the
   whole operation. Hypervalency is allowed and is the user's
   responsibility.
4. **Re-perceive bonds** (`perceive_bonds`, distance-based). The
   0.45 Å tolerance comfortably absorbs the ~0.1 Å covalent-radius
   changes from relabeling, so existing bonds survive.

## Gesture wiring

- `grow_at_atom` keeps its name and signature, so `widget.py` and
  other callers are untouched. Its heavy-atom branch dispatches to
  `replace_atom` instead of `_grow_onto`; the hydrogen branch
  (`_promote_hydrogen`) is unchanged.
- `_grow_onto` is deleted — nothing calls it anymore.
- Docstrings in `editor.py` are updated to describe the new gesture
  vocabulary: empty space births, H-click promotes/grows, heavy-click
  replaces.

## Intended consequences

- Same-element click becomes a "repair" gesture: it fills the atom's
  valency with hydrogens (no-op when already saturated).
- Replacing a saturated atom with a lower-valence element keeps all
  neighbors (e.g. CH4 carbon + O selected → hypervalent OH4). Excess
  hydrogens are *not* trimmed.

## Testing

Rewrite the heavy-atom cases in `tests/test_editing.py`; new cases:

- Replace relabels in place (position unchanged, symbol changed).
- Terminal-H neighbors snap to the new element–H bond length;
  non-terminal and heavy neighbors do not move.
- Valency filling adds exactly `valence − coordination` hydrogens,
  each at the correct bond length.
- Coordination ≥ valency adds nothing (hypervalent case).
- Same-element replace fills valency (repair gesture).
- Lone (unbonded) heavy atom: replace relabels and caps to a full
  free-standing fragment.
- Hydrogen click still promotes (regression).
