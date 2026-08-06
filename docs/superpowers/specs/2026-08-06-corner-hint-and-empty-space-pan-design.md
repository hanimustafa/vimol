# Bottom-left control hint, and opt+click-empty-space pan

## Problem

Two small, related UX gaps in the terminal viewer:

1. The full binding reference only appears when the user presses `?`. There is
   no always-visible reminder of the handful of bindings a first-time user
   reaches for immediately (`z` center, `r` align, `1`-`6` display modes).
2. `opt+click`/`opt+drag` is only meaningful over an atom (manual bond in
   edit mode, or alignment picking otherwise). Pressed over empty space, it
   either silently rotates the camera (edit mode) or arms alignment picking
   with no atom to pick (view mode) — there is no way to pan/translate the
   scene with `opt`+left-button the way `right`/`mid`-drag already does.

## 1. Persistent bottom-left hint

A new, always-on caption in the bottom-left corner of the render viewport,
directly above the status bar row. Unlike the `?` help panel, it is not
toggled — it is drawn every frame, the same way the status bar is.

Content (fixed, three lines):

```
z    center
r    align
1-6  display modes
```

Rendering: plain dim text, no box border (distinct from the bordered `?`
panel) — reuses the theme's help text color at reduced emphasis. Implemented
as a new geometry helper (parallel to `_help_geometry`) and draw method
(parallel to `_draw_help`) in `viewer.py`, called unconditionally from the
same place `_draw_help` is called today.

Suppressed whenever another overlay claims the same screen region: the `?`
help panel, periodic table picker, geometry picker, selection picker, or
file browser.

Out of scope: no mention of `opt+click` (below) in this hint, per explicit
instruction — the hint stays exactly the three lines above.

## 2. opt+click on empty space pans the scene

In `MoleculeWidget.handle_mouse` (`widget.py`), both alt-click branches
(`self.editable` and not) currently decide behavior purely from
`ev.alt`/`ev.button`, without checking whether an atom is under the cursor
at press time in the *miss* case:

- **Editable, miss**: comment says "fall through to a normal press" — the
  ensuing drag rotates the camera like a plain unmodified drag.
- **Not editable**: `set_alignment_mode(True, preserve=True)` fires
  unconditionally on `down`, regardless of hit-test. A miss still arms
  alignment mode; on `up`, `_alignment_click`'s own `_pick_active_only`
  miss-check clears any existing `align_sel` as a "click empty space to
  reset" shortcut.

New behavior: hit-test with `_pick_active_only` (the same method the
existing click-resolution paths already use) at `down` time in both
branches. If it hits an atom, behavior is unchanged (bond gesture / arm
alignment picking). If it misses:

- Do not start a bond gesture and do not arm alignment mode.
- Start a pan drag: reuse the existing pan path (the drag dispatch already
  treats `_drag_button in (1, 2)` or `_drag_shift` as pan) by setting up
  the down-handler so the subsequent `drag` events pan instead of orbit.
- A plain `opt+click` (no real drag) on empty space now does nothing. This
  intentionally retires the "empty click clears the alignment selection"
  shortcut in view mode — confirmed acceptable.

## 3. Documentation

Add one binding line for opt+click-empty-space pan to the existing `?` help
panel (`_HELP_TAIL` in `viewer.py`, near the existing pan/roll bindings) and
to the README's control rundown. The new persistent bottom-left hint (§1)
does **not** mention this binding.

## Testing

- `tests/test_editing.py` / `tests/test_align.py`: add coverage for
  opt+click+drag on empty space panning the camera (assert camera pan state
  changes, no bond/selection side effects) in both editable and non-editable
  widgets, and confirm opt+click on an atom is unaffected.
- Add coverage confirming a plain opt+click (no drag) on empty space is now
  a no-op in view mode (no longer clears `align_sel`).
- New viewer-level test(s) for the bottom-left hint's presence/absence
  (visible by default; hidden while help/pickers/file-browser are open).
