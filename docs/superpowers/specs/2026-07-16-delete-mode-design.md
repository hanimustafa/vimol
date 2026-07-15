# Delete mode: 'x' to remove atoms, crosshair pointer

**Date:** 2026-07-16
**Status:** Approved

## Summary

A third editing mode, alongside append (`a`) and view/rotate: pressing `x`
toggles delete mode, in which clicking an atom removes it (and, if it was a
heavy atom, its own terminal hydrogens with it). While delete mode is active
the terminal's mouse pointer changes to a `crosshair` shape, using the
Kitty-family terminals' OSC 22 pointer-shape escape sequence, so the cursor
itself signals the destructive tool is armed.

## Pointer shape mechanism

Kitty-protocol terminals (kitty; support elsewhere unconfirmed) implement
OSC 22, a simple escape sequence that changes the OS mouse pointer icon
while the terminal has focus, from a fixed set of ~30 CSS-cursor-style
shape names. There is no literal "X" glyph in that set; the closest visual
read for a destructive/targeting tool is `crosshair`.

New helpers in `src/vimol/kitty.py`, alongside the existing `delete_image`/
`clear_all_images` (which also return raw bytes for the caller to write,
rather than writing directly):

```python
def set_pointer_shape(shape: str) -> bytes:
    return f"\x1b]22;{shape}\x1b\\".encode()

def reset_pointer_shape() -> bytes:
    return b"\x1b]22;\x1b\\"
```

`viewer.py` writes `set_pointer_shape("crosshair")` the instant delete mode
turns on, and `reset_pointer_shape()` the instant it turns off. `_exit()`
(session cleanup, already unconditionally resets mouse-reporting and the
text cursor) also unconditionally writes `reset_pointer_shape()`, so a
quit or kill never leaves the terminal's cursor stuck as a crosshair.
Since a non-supporting terminal simply ignores an OSC sequence it doesn't
recognize, no capability probe is needed — this matches how `_CLEAR` and
the mouse-mode enable/disable sequences are already written unconditionally
regardless of terminal.

## `delete_atom` editor operation

New function in `src/vimol/editor.py`:

```python
def delete_atom(mol: Molecule, idx: int) -> None:
```

1. **Find cascade victims.** Terminal-H neighbors of `idx` — hydrogens
   bonded *only* to `idx` — are marked for removal alongside it. This one
   rule naturally covers both gestures with no special-casing: clicking a
   heavy atom sweeps its own hydrogens with it (deleting methane's carbon
   removes the whole CH4); clicking a hydrogen directly has no
   hydrogens-of-its-own, so exactly that one atom goes.
2. **Remove rows.** Drop `idx` and its cascade victims from `symbols` and
   `positions`.
3. **Re-perceive bonds** (`perceive_bonds`, distance-based) rather than
   hand-patching bond indices after the removal — sidesteps the reindexing
   bookkeeping a mid-list removal would otherwise require, and is
   consistent with how every other `editor.py` operation ends.
4. **No auto-capping.** Heavy neighbors that lose a bond to the deleted
   atom are left exactly as they are — dangling, under-coordinated. No new
   hydrogen appears. (Contrast with `replace_atom`, which does top up
   valence — deletion is a plain erase, not a substitution.)

Deleting the last atom in the molecule is not special-cased; it just leaves
`mol.n_atoms == 0`, same as any other structural edit.

## Mode wiring

`src/vimol/widget.py`:

- `self.delete_mode: bool = False`, alongside the existing `append_mode`.
- `set_delete_mode(on: bool)`, mirroring `set_append_mode`: stays off
  unless `self.editable`. Turning delete mode on clears `append_mode`;
  `set_append_mode` turning append mode on clears `delete_mode` — the two
  are mutually exclusive, matching the existing single-active-tool model.
- `handle_mouse`'s left-click-not-drag branch (currently
  `if self.editable and self.append_mode and was_left and not ev.shift`)
  gets a sibling: `elif self.editable and self.delete_mode and was_left and
  not ev.shift` → calls a new `_delete_at(x, y)`. A click on empty space
  (no atom under the cursor) is a no-op — nothing to delete there.
- `_delete_at`: picks the atom under the click first. Unlike `_edit_at`
  (which always mutates — a click in append mode either grows onto an atom
  or births one in empty space), a delete click can be a genuine no-op, so
  the undo push must be conditional: if no atom was hit, return `False`
  immediately with no undo entry and no mutation. If an atom was hit, push
  an undo snapshot (`_push_undo`), call `editor.delete_atom`, then refresh
  `_base_colors`, clear `hovered`/`selected`, call `_refresh_dirty()`, and
  return `True` — the same post-mutation bookkeeping `_edit_at` already
  does.

`src/vimol/viewer.py`:

- `'x'` joins `_EDIT_DRIVER_KEYS` (claimed only when `self.editable`,
  exactly like `s`/`u`/`o`).
- `_driver_key` gets `elif key == "x" and self.editable:` → toggles
  `self.widget.delete_mode` via `set_delete_mode`, and writes the
  pointer-shape escape (`set_pointer_shape("crosshair")` on,
  `reset_pointer_shape()` off) on that transition.
- The existing `'a'` handler, after calling `set_append_mode(True)`, also
  writes `reset_pointer_shape()` if delete mode had been active (mode
  switched via `a`, not `x`) so the crosshair doesn't linger into append
  mode.

## UI feedback

- Status bar (`_status_bar`): a `✗DELETE` badge, styled like the existing
  `✎APPEND` one, shown when `self.editable and self.widget.delete_mode`.
  No element/geometry pills are shown in delete mode — deletion doesn't
  use a build element.
- Hover highlight (`widget._apply_highlight`): while `delete_mode` is on,
  the hovered atom is tinted red instead of the append/view-mode yellow —
  a direct preview of "this is what disappears if you click here," reusing
  the existing tint mechanism with a mode-conditional color.

## Docs

- README: interactive-controls table gets a `Delete` row; the `### Editing`
  section gets a paragraph on delete mode (key, cascade behavior, no
  auto-cap, crosshair pointer).
- `src/vimol/viewer.py`'s `_HELP_EDIT` in-app help list gets a
  `x .................. delete` line.

## Testing

New cases in `tests/test_editing.py`:

- `editor.delete_atom`: deleting a lone heavy atom with terminal H's
  removes the whole group; deleting a hydrogen directly removes only that
  atom; deleting a mid-chain atom leaves its neighbor's bond count reduced
  with no new hydrogen added; deleting the molecule's last atom leaves
  `n_atoms == 0`.
- `widget`: `set_delete_mode`/`set_append_mode` mutual exclusivity;
  `set_delete_mode` stays off when not editable; a click on an atom in
  delete mode removes it, marks `dirty`, and is undoable via `undo()`; a
  click on empty space in delete mode is a no-op — no crash, no change, no
  undo entry pushed, `dirty` stays whatever it was.
- `viewer`: `'x'` toggles `delete_mode` only when editable; the status bar
  shows `✗DELETE` when active; toggling delete mode on/off (and viewer
  `_exit()`) writes the expected pointer-shape escape bytes (captured via
  monkeypatching `kitty.write_bytes`, the same technique
  `test_closing_picker_erases_only_panel_rows` already uses).
