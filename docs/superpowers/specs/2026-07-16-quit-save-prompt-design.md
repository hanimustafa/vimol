# ESC quit-save prompt for modified files

**Date:** 2026-07-16
**Status:** Approved

## Summary

Pressing ESC in the normal viewer state with unsaved modifications no
longer quits silently. A confirm bar asks
`unsaved changes — save before quitting? (y/n/Esc)`; `y` routes through
the existing save-filename prompt and quits after a successful save,
`n` quits without saving, ESC cancels the quit.

## Behavior

- Trigger: `escape` key, `self._mode == "normal"`, `self.editable`, and
  `self.widget.dirty`. Everything else about ESC is unchanged: inside
  pickers/help/save prompts it keeps closing those; with a clean model
  it quits immediately as today.
- `q` remains an immediate, no-questions quit (deliberate force-quit
  path), and Ctrl-C remains an emergency exit.
- New mode `"quit_confirm"`, rendered in the status bar like
  `save_confirm` (warm/red styling):
  * `y`/`Y`/`enter` -> `_open_save_prompt()` with a `_quit_after_save`
    flag set; after `_do_save` succeeds while the flag is set, the
    viewer stops (`self._running = False`). A FAILED save (exception
    path in `_do_save`) clears the flag and stays running, showing the
    existing error message.
  * `n`/`N` -> quit without saving.
  * `escape`/`\x03` -> back to `"normal"`, still running, model still
    dirty.
- Escaping out of the filename prompt (`save_input`) cancels the quit
  entirely (clears `_quit_after_save`, stays running) — ESC there
  already means "cancel the prompt", and a cancelled save must never
  fall through to "quit anyway, discarding changes".

## Testing

- ESC with dirty model enters `quit_confirm`, viewer still running.
- `n` quits (running flag false).
- ESC in the confirm cancels (mode normal, still running, still dirty).
- `y` opens the save prompt pre-filled; completing the save (fresh
  filename, tmp_path) writes the file AND stops the viewer.
- ESC inside the save prompt after `y` cancels the quit (running,
  dirty, mode normal, `_quit_after_save` cleared).
- ESC with a CLEAN model quits immediately (no prompt).
- Read-only viewer: ESC quits immediately regardless (dirty is
  impossible without editing, but the guard is on `editable` anyway).
- `q` with a dirty model still quits immediately.
