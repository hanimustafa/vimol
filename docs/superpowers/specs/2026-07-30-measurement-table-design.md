# Measurement table: compare a measurement across all loaded structures

**Issue:** VIM-6 · **Date:** 2026-07-30 · **Status:** Revision 2 (no explicit
commit key — see §"Committing and removing columns")
**Depends on:** `docs/design/multi-structure.md` §4.1 (structure strip), §8
(`StructureSet.measure`, already speced) · measure-mode symbol conventions
from `docs/superpowers/specs/2026-07-16-measure-mode-design.md`

## Summary

Today `m` shows one live measurement in the status bar, gone the moment the
pick list changes. This adds a table next to the structure strip that tracks
the CURRENT pick live — as soon as 2 atoms are selected, updating in place as
more are added (distance → angle → dihedral of the same atoms) — and turns it
into a permanent column the moment the user moves on to a different
measurement, so a user comparing 200 conformers/frames can build up several
columns and read every structure's value down each one, with no separate
"save this" gesture to remember.

Only meaningful when the strip is shown (`len(structures) > 1`); with a single
structure there is nothing to compare against.

**Revision 2** replaces the original `enter`-to-commit design (rejected by the
user in review: they wanted the column to appear immediately, not after an
extra keypress) with the freeze-on-reset model described below, and fixes two
bugs the original design's `_measure_w` caching introduced (stale terminal
content and a mouse/hover offset — see "Committing and removing columns" and
"Rendering").

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

Viewer-side, a FROZEN column is `(header: str, indices: Tuple[int, ...])` in
`Viewer._measure_columns: List[...]`. `indices` are active-local (measure_sel
is already documented as active-local). `header` is computed once, at freeze
time, from *that* active structure's symbols, so it doesn't retroactively
change if a later active structure disagrees.

The LIVE pick (`widget.measure_sel` itself) is not stored anywhere separate
— `Viewer._measure_layout` (see "Rendering") reads it directly each draw and
renders it as one more column, header and all, whenever `widget.measure_mode`
and `len(measure_sel) >= 2`. There is nothing to freeze until the pick stops
being live.

## Committing and removing columns

**No explicit commit key.** A pick list becomes a permanent column the
moment it stops being the live one — i.e. whenever something is about to
replace or clear `measure_sel` that ISN'T simply extending it by one atom.
`Viewer._freeze_measure_sel(sel)` does the freezing: appends
`(header, sel)` to `_measure_columns` (no-op if the same index tuple is
already a column, so re-measuring the same pair twice doesn't duplicate),
and is a no-op itself for `len(sel) < 2`. Four call sites, each capturing
`measure_sel` *before* whatever is about to invalidate it:

- **A genuine reset while picking** — `_dispatch`'s widget-event branch
  snapshots `measure_sel` before calling `widget.handle_event`, then compares
  before/after: if the new list does NOT extend the old one (the widget's
  own 5th-click reset, or an empty-space click clearing it), the OLD list
  freezes. A plain extension (2 atoms → 3 → 4) does *not* freeze — that's
  the same live column updating in place.
- **Measure mode switching off** (`m`) — freezes the current pick first;
  `set_measure_mode(False)` is about to clear it, and turning measuring off
  reads as "I'm done with this one", not "throw it away".
- **The active structure changing** (`_activate_structure`, `_cycle_frame` —
  `n`/`p`, `1`–`9`, a strip click) — `widget.refresh_active()` clears
  `measure_sel` because indices are active-local, but switching frames
  mid-measurement is exactly the trajectory-browsing workflow VIM-6 exists
  for; silently discarding the pick there would defeat the point.
- Undo and `widget.set_molecule()` are **not** freeze points (see below) —
  they clear `measure_sel` the same way they always have, with nothing
  special captured first.

**Remove**: click the `×` at the end of a FROZEN column's header (the live
column has none — nothing to remove yet, it resolves on its own). Header
spans are recorded per-column during draw (mirroring `_list_row_spans`); a
mouse-down hit inside one deletes that column, checked before the existing
`_list_index_at_row` row-click handling.

Columns are **not** cleared by `undo()`, nor by anything else: indices can
go stale if editing changes an entry's atom count (deleting atoms shifts or
removes what a pinned index pointed at), which is accepted as an
out-of-scope edge case — VIM-6's acceptance criteria targets read-only
comparison, not editing interaction. `StructureSet.measure` guards the crash
this could otherwise cause (an index past an entry's current atom count
reports `None`, same as any other degrade case) but does not try to detect
or repair staleness; a column left wrong by an edit is only ever wrong for
the structure(s) actually edited, and remains removable by hand.
(`widget.set_molecule()`, which wholesale-replaces the structure set, is not
currently called anywhere in the interactive `Viewer` — there is no reload
action to hook — so this spec makes no claim about it.)

## Header & value conventions

Header text, computed from the active structure's element symbols at each
picked (0-based) index — same raw-index convention as the existing `#idx`
hover/status-bar text, just without the `#` — recomputed live for the
in-progress column, frozen at the moment a completed one freezes:

| picks | header | example |
|---|---|---|
| 2 (distance) | `<Sym><i>-<Sym><j>` | `O5-N2` |
| 3 (angle) | `∠<Sym><i>-<Sym><j>-<Sym><k>` | `∠C1-C2-C3` |
| 4 (dihedral) | `φ<Sym><i>-<Sym><j>-<Sym><k>-<Sym><l>` | `φC4-C5-C6-C7` |

`∠` and `φ` match the symbols already used in the live status-bar readout
(measure-mode design). A FROZEN header ends with a `×` (the click-to-remove
target); the live column's header does not — see "Committing and removing
columns".

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
  = 0 with no columns), shifting `_img_origin_px` (and hence the mouse→image
  coordinate mapping used for hover/pick) further right accordingly.
- Row click hit-testing (`_in_list_zone`, `_list_row_spans`) extends across
  the full row width including measurement columns — clicking a structure's
  measurement cells activates it exactly like clicking its label; there is
  no separate interaction on a value cell.
- **`_measure_w` caching and its two failure modes.** `_in_list_zone` is hit
  on every mouse move, so it reads a cached `_measure_w` rather than
  recomputing the layout (which calls `StructureSet.measure` per column) on
  every pointer event. `_refresh_measure_w()` re-syncs that cache right after
  a freeze/removal, so a hit test later in the same input burst sees it
  immediately rather than one tick stale. That immediate sync, though,
  means `_update_geometry`'s own before/after comparison on `_measure_w`
  finds nothing changed by the time it runs — which would silently skip
  resizing `_img_cols`/`_img_origin_px` and the terminal clear that a real
  width change needs, leaving stale header text on screen and (more
  seriously) a stale mouse→image origin that shows up as hover/pick landing
  on the wrong atom right after a column appears or disappears. Fixed with
  a `_geometry_dirty` flag: `_refresh_measure_w` sets it, `_update_geometry`
  ORs it into `changed` and clears it once handled. Belt-and-suspenders,
  `_draw_list`'s row writer also emits erase-to-end-of-line after every
  row, so a shrinking table can never leave a wider previous frame's
  content sitting past the new, narrower content regardless of this path.
- **Overflow**: columns are dropped (from the end) once they would leave
  fewer than a fixed minimum of viewport columns for the 3D image itself —
  a silent truncation, not an error. Reordering or scrolling the table
  itself is out of scope for this pass; a user who pins more measurements
  than fit removes one to see the next.

## Testing

- `editor.measure_value`: same numeric cases as the existing `measurement()`
  string tests (known distance/angle/dihedral, `None` for <2 picks);
  `measurement()`'s existing exact-string tests must keep passing unchanged.
- `StructureSet.measure`: matching-topology entries get computed values
  (including the active entry itself); an entry with different symbols
  anywhere gets `None`; empty set returns `[]`; an index past an entry's
  current atom count (a pinned column outlived an edit that shrank it)
  degrades to `None` instead of raising.
- Viewer: a live column appears at 2 picks and updates in place (not as a
  new column) as a 3rd/4th atom extends it; a fresh pick after a completed
  4-atom selection freezes the old one and starts a new live column; an
  empty-space click freezes whatever was live; re-freezing the same indices
  doesn't duplicate; switching measure mode off freezes the live pick;
  switching the active structure (`n`/`p`, `1`-`9`, a strip click) freezes
  the live pick rather than silently discarding it; clicking a FROZEN
  header's `×` removes that column and reflows the rest, and the live
  column has no `×` to click; a row click under a measurement column still
  activates the structure; the table draws only when `len(structures) > 1`;
  degrade cell renders `—` for a mismatched structure; enough pinned columns
  to exceed the terminal width drop the excess rather than driving
  `_img_cols` to zero/negative or widening any row past the terminal;
  `_img_origin_px` (and so the mouse→image mapping) is correct immediately
  after a column appears or disappears, not one tick stale.
