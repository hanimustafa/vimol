# Measurement mode: 'm' for distance / angle / dihedral readout

**Date:** 2026-07-16
**Status:** Approved

## Summary

Pressing `m` toggles measurement mode: the mouse pointer becomes the
`cell` shape (OSC 22 push, popped on disarm — the precision plus-cross,
the closest match in the fixed CSS-cursor set to a caliper), a
`∡MEASURE` badge appears in the status bar, and clicking atoms builds an
ordered selection whose measurement (distance → angle → dihedral) is
shown live in the status bar. Pressing `m` again returns to normal mode.

## Availability & mode exclusivity

- Available in BOTH editable and read-only viewers — measuring is
  non-destructive, so it is not gated on `editable`. `m` joins
  `_BASE_DRIVER_KEYS` (it is currently unbound in both keymaps).
- Mutually exclusive with append and delete modes: arming any of the
  three disarms the other two (widget setters clear each other, same
  single-active-tool model as append/delete today).
- Pointer shape uses the same push/pop stack as delete's crosshair:
  arming pushes `cell`, disarming (via `m`, or switching to `a`/`x`)
  pops. Transitions between two pointer-owning modes (delete <->
  measure) must pop the old shape before pushing the new one — the
  viewer keeps a `_pointer_pushed` flag so pushes and pops always pair.

## Selection model (`MoleculeWidget`)

- New state: `measure_mode: bool`, `set_measure_mode(on)` (no editable
  gate), and `measure_sel: List[int]` (ordered, up to 4).
- A left *click* (down+up within the existing ~3 px threshold; drags
  still rotate) in measure mode:
  * on an atom not in the selection: append it. If the selection
    already held 4 atoms, RESET first — the clicked atom becomes the
    first pick of a fresh selection.
  * on an atom already selected: no-op.
  * on empty space: clear the selection.
- The selection is cleared by `set_measure_mode(False)`, `undo()`,
  and `set_molecule()` (stale-index discipline, like the gestures).
- `_apply_highlight` tints every atom in `measure_sel` with the
  existing yellow highlight color; hover still works on top (hover of
  an unselected atom shows the usual hover tint).

## Measurement math

Pure function in `editor.py` (testable without a terminal):

```python
def measurement(mol: Molecule, sel: Sequence[int]) -> str:
```

- len < 2  -> ""            (nothing to report)
- len == 2 -> "d(i–j) = 1.523 Å"          (euclidean distance)
- len == 3 -> "∠(i–j–k) = 109.5°"          (angle at j between j->i and j->k)
- len == 4 -> "φ(i–j–k–l) = 60.0°"         (standard dihedral: signed angle
  between the (i,j,k) and (j,k,l) planes about the j–k axis)

Distances to 3 decimals, angles to 1. Indices rendered as the atom
numbers (matching the hover readout's `#idx` convention is fine).

## Status bar

- `∡MEASURE` badge, styled and sized like `✎APPEND`/`✗DELETE`, shown
  when `measure_mode` (badge visible in read-only viewers too).
- While `measure_mode` and `len(measure_sel) >= 2`, the LEFT segment
  (where hover text goes) shows the measurement string instead of the
  hover/molecule-info text. With 0–1 selections the normal left-segment
  behavior applies.

## Help & docs

- `_HELP_VIEW` and `_HELP_EDIT` both gain
  `  m .................. measure (click 2/3/4 atoms: distance/angle/dihedral)`.
- README: a `Measure` row in the interactive-controls table (the
  classic one, since it works read-only) and a short paragraph.

## Testing

- `editor.measurement`: exact strings for a known distance (e.g. two
  atoms 1.523 Å apart), a known 90°/109.47° angle, a known dihedral
  (e.g. ±60° staggered), and "" for len<2. Dihedral sign convention:
  assert absolute value to stay convention-agnostic, plus a 0°/180°
  planar case.
- Widget: click sequence builds the ordered selection; 5th click
  resets to fresh single-atom selection; empty-space click clears;
  re-clicking a selected atom is a no-op; drag does not select;
  highlight covers all selected atoms; undo/set_molecule clear.
- Viewer: `m` toggles in read-only AND editable viewers; mutual
  exclusion with `a`/`x`; badge shown; measurement string appears in
  the status bar after two clicks; pointer pushes `cell` on arm and
  pops on disarm INCLUDING the delete->measure and measure->append
  transitions (assert balanced push/pop bytes via the monkeypatched
  write_bytes technique).
