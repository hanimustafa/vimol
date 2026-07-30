# Measurement table: compare a measurement across all loaded structures

**Issue:** VIM-6 · **Date:** 2026-07-30 · **Status:** Approved
**Depends on:** `docs/design/multi-structure.md` §4.1 (structure strip), §8
(`StructureSet.measure`, already speced) · measure-mode symbol conventions
from `docs/superpowers/specs/2026-07-16-measure-mode-design.md`

## Summary

Today `m` shows one live measurement in the status bar, gone the moment the
pick list changes. This adds a way to *pin* a measurement as a column next to
the structure strip, evaluated against every loaded structure at once, so a
user comparing 200 conformers/frames can build up several columns (distance,
angle, dihedral) and read every structure's value down the column.

Only meaningful when the strip is shown (`len(structures) > 1`); with a single
structure there is nothing to compare against.

## Data model

`StructureSet.measure(indices) -> List[Tuple[str, Optional[float]]]`
(`structures.py`, new) — per §8: iterates `self.entries` in order, returns
`(label, value)`. An entry's `value` is `None` unless
`entry.molecule.symbols == active.molecule.symbols` element-by-element (full
array, not just length) — same-topology gate, "degrade gracefully" per VIM-6's
acceptance criteria. Otherwise `value` is the raw float distance (Å) / angle
(deg) / signed dihedral (deg) at `indices`, evaluated on the entry's own
*source* coordinates (transform-invariant, decoupling this from VIM-4).

Extracted from `editor.py`: `measure_value(mol, sel) -> Optional[float]`, the
numeric core of the existing `measurement()` string formatter (same math,
same 2/3/4-atom branching). `measurement()` becomes a thin wrapper so its
tested output strings are unchanged.

Viewer-side, a committed column is `(header: str, indices: Tuple[int, ...])`
in `Viewer._measure_columns: List[...]`. `indices` are active-local, fixed at
commit time (measure_sel is already documented as active-local). `header` is
also computed once at commit time from *that* active structure's symbols, so
it doesn't retroactively change if a later active structure disagrees.

## Committing and removing columns

- **Commit**: `enter` joins `_BASE_DRIVER_KEYS` (works read-only, like `m`).
  While `widget.measure_mode` and `len(widget.measure_sel) >= 2`: append
  `(header, tuple(measure_sel))` to `_measure_columns` (no-op if the same
  index tuple is already a column) and clear `measure_sel`, ready for the
  next pick. Otherwise `enter` is unclaimed (falls through as today — it does
  nothing while the strip lacks focus). This never collides with the strip's
  own `enter` (activate row cursor), which only fires `if self._list_focused`.
- **Remove**: click the `×` at the end of a column header. Header spans are
  recorded per-column during draw (mirroring `_list_row_spans`); a mouse-down
  hit inside one deletes that column, checked before the existing
  `_list_index_at_row` row-click handling.
- Columns are cleared when `widget.set_molecule()` replaces the whole set
  (existing "reload discards everything" precedent). They are **not** cleared
  by `undo()` — indices can go stale if editing changes atom count, which is
  accepted as an out-of-scope edge case: VIM-6's acceptance criteria targets
  read-only comparison, not editing interaction. A stale column is only ever
  wrong for the structure(s) actually edited afterward, and remains
  removable by hand.

## Header & value conventions

Header text, computed at commit time from the active structure's element
symbols at each picked (0-based) index — same raw-index convention as the
existing `#idx` hover/status-bar text, just without the `#`:

| picks | header | example |
|---|---|---|
| 2 (distance) | `<Sym><i>-<Sym><j>` | `O5-N2` |
| 3 (angle) | `∠<Sym><i>-<Sym><j>-<Sym><k>` | `∠C1-C2-C3` |
| 4 (dihedral) | `φ<Sym><i>-<Sym><j>-<Sym><k>-<Sym><l>` | `φC4-C5-C6-C7` |

`∠` and `φ` match the symbols already used in the live status-bar readout
(measure-mode design). Each rendered header ends with a `×` (the click-to-
remove target).

Cell values are bare numbers, no unit suffix (the header's glyph already says
what kind of number it is): 3 decimals for distance, 1 decimal for angle/
dihedral. A structure whose full symbol list doesn't match the active
structure's renders `—`.

## Rendering

Extends `_draw_list` horizontally — no new rows. The strip's existing
`STRUCTURES N` header row grows rightward with one block per column; every
structure row grows the same way, in lockstep with the existing vertical
scroll (same rows, just wider).

- Each column's width is `max(len(header text), longest formatted value or
  "—")`, content padded one space each side; header left-aligned, values
  right-aligned.
- **Interleaved tinting**: alternating background across two theme shades
  (new `theme.measure_col_bg_a` / `_b`), assigned by column position (1st,
  3rd, ... vs 2nd, 4th, ...). This tint is its own layer, painted on every
  row *including* the active/cursor row — it does not compete with
  `list_active_bg`/`list_cursor_bg`, which stay scoped to the existing
  structure-name region exactly as today. A plain (untinted) single-space
  gap separates adjacent columns.
- `_update_geometry` reserves `list_w + measure_w` columns total (`measure_w`
  = 0 with no columns), shifting `_img_origin_px` further right accordingly.
- Row click hit-testing (`_in_list_zone`, `_list_row_spans`) extends across
  the full row width including measurement columns — clicking a structure's
  measurement cells activates it exactly like clicking its label; there is
  no separate interaction on a value cell.
- Not in scope for this pass: horizontal overflow when columns don't fit the
  terminal width. Columns beyond the available width are simply not drawn
  (matches how the strip already degrades on a short/narrow terminal).

## Testing

- `editor.measure_value`: same numeric cases as the existing `measurement()`
  string tests (known distance/angle/dihedral, `None` for <2 picks);
  `measurement()`'s existing exact-string tests must keep passing unchanged.
- `StructureSet.measure`: matching-topology entries get computed values
  (including the active entry itself); an entry with different symbols
  anywhere gets `None`; empty set returns `[]`.
- Viewer: `enter` commits a column and clears the pick list; committing the
  same indices twice doesn't duplicate; `enter` outside measure mode or with
  <2 picks is a no-op; clicking a header's `×` removes that column and
  reflows the rest; a row click under a measurement column still activates
  the structure; the table draws only when `len(structures) > 1`; degrade
  cell renders `—` for a mismatched structure.
