# Manual bonds: option-drag between atoms, rubber-band preview

**Date:** 2026-07-16
**Status:** Approved

## Summary

Holding option/alt and pressing the left button on an atom starts a bond
gesture: a vector (rubber band) grows from that atom and follows the mouse.
Releasing over another atom creates an explicit **manual bond** between the
two — drawn like any other bond, and kept even when the pair's distance
exceeds the automatic bond-perception threshold. Releasing over empty
space, or over the anchor atom itself, cancels.

## The persistence problem (why "manual" bonds need a home)

Every editor operation ends with `_reperceive(mol)`, which *replaces*
`mol.bonds` wholesale with distance-perceived bonds — any long manual bond
would be silently erased by the next edit. So manual bonds get their own
authoritative store:

- `Molecule.manual_bonds: List[Tuple[int, int, int]]` (new dataclass field,
  default empty; `(i, j, order)` with `i < j`, like `bonds`).
- `editor._reperceive` becomes: perceive from distance, then union in
  `mol.manual_bonds`, deduplicating on `(i, j)` (a manual bond whose pair
  is also within perception range must not appear twice).
- `editor.delete_atom` must **remap** `manual_bonds` across the row
  removal: drop any manual bond touching a deleted atom, and shift the
  indices of the survivors (old index -> new index, same mapping the
  `keep` mask implies) — *before* its final `_reperceive`.
- Other editor ops (`birth_molecule`, `replace_atom`, `grow_at_atom`,
  `_promote_hydrogen`) only append atoms, so existing indices are stable
  and no remap is needed; their existing `_reperceive` call picks the
  union up automatically.

New editor operation:

```python
def add_manual_bond(mol: Molecule, i: int, j: int, order: int = 1) -> bool:
```

Normalizes to `i < j`; returns False (no change) if `i == j` or the pair
is already in `mol.manual_bonds`; otherwise appends and `_reperceive`s
(which unions it into `mol.bonds`) and returns True. If the pair is
already auto-perceived (within threshold), the manual bond is *still
recorded* — that makes the drawn bond robust: it survives later geometry
changes that pull the atoms apart.

Scope note: the XYZ writer has no bond block, so manual bonds are a
session/UI construct — they are not persisted by `s`ave. Not a defect;
out of scope.

## Widget state & bookkeeping

`src/vimol/widget.py`:

- Undo snapshots grow a fourth element: `(symbols, positions, bonds,
  manual_bonds)`; `undo()` restores `mol.manual_bonds = list(...)` too.
- `_signature()` currently hashes only `(symbols, positions.tobytes())` —
  a manual bond changes the model without moving an atom, so the dirty
  flag would never trip. Append `tuple(mol.manual_bonds)` to the
  signature tuple. (`mark_saved`/`_refresh_dirty` then work unchanged.)

## Gesture wiring (`widget.handle_mouse`)

New state: `self._bond_anchor: Optional[int] = None`.

- **down** (left button, `ev.alt`, editable, atom under cursor): set
  `_bond_anchor` to the picked atom, install the rubber-band preview (see
  below), and do NOT start a normal drag (`_drag_button` stays None so
  the drag branch won't rotate/pan). Alt+down over empty space falls
  through to normal handling.
- **drag / move** while `_bond_anchor is not None`: update the preview
  vector to point from the anchor atom to `unproject(x, y)`; return True
  (redraw).
- **up** while `_bond_anchor is not None`: pick the atom under the
  release point. If it is a different atom, push an undo snapshot, call
  `editor.add_manual_bond(mol, anchor, target)`, and run the usual
  post-mutation bookkeeping (`_refresh_dirty`; colors don't change and
  atom count is stable, so no `_base_colors` refresh needed). If the
  release is over empty space, the anchor itself, or an already-bonded
  pair (add_manual_bond returned False): no undo entry, no dirty change.
  Either way, remove the preview and clear `_bond_anchor`, return True.

The gesture works whenever `editable` — independent of append/delete
mode (the alt modifier is what distinguishes it; append/delete click
semantics are untouched because those fire on plain, non-alt clicks).

## Rubber-band preview

Reuses the existing per-atom vector-field arrow machinery
(`molecule.vector_fields` + `arrows.build_arrow_geometry`), which both
render backends already draw:

- On gesture start the widget appends a dedicated preview
  `VectorField` — an `(n_atoms, 3)` zeros array with only the anchor
  row set, `scale=1.0`, a thin radius (e.g. 0.06) and a distinctive
  color (e.g. warm yellow `(1.0, 0.85, 0.3)`, matching the hover
  highlight); a small arrow head is fine (default head fractions).
- Each drag update rewrites the anchor row to
  `unproject(cursor) - positions[anchor]`.
- On gesture end (bond made or cancelled) the preview field is removed
  from `mol.vector_fields`. The widget keeps a reference to its own
  preview field object so it never touches user-supplied fields.

## Viewer

No new keys, no status-bar changes — the gesture is modifier-driven and
self-revealing (the arrow appears under your hand). `_HELP_EDIT` gains
one line: `     option-drag atom -> atom ... draw a bond (kept beyond auto range)`.
README editing section gets a matching bullet.

## Testing

New cases in `tests/test_editing.py`:

- `editor.add_manual_bond`: creates a long bond that appears in
  `mol.bonds` after re-perception; duplicate/self pairs return False;
  a manual bond within auto range doesn't duplicate in `mol.bonds`.
- Manual bond survives a subsequent edit (e.g. `grow_at_atom` elsewhere
  re-perceives and the long bond is still in `mol.bonds`).
- `editor.delete_atom` remaps: deleting an atom *below* a manual pair
  shifts its indices; deleting one endpoint drops the manual bond.
- Widget gesture (synthetic events, like the existing `_click` helper):
  alt+down on atom A, drag, up on atom B → `mol.bonds` contains the
  pair, `dirty` is set, `undo()` removes it (and `manual_bonds` is
  restored); alt+down on atom, up over empty space → no bond, no undo
  entry, no dirty; during the drag a preview vector field exists and
  after release it is gone; plain (non-alt) clicks in append mode still
  build as before.
- Alt+down over empty space behaves as a normal press (no crash, no
  preview).
