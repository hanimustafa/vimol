# Bottom-left control hint, and opt+click-empty-space pan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-on bottom-left control hint to the terminal viewer, and make `opt+click`/`opt+drag` on empty space pan the scene instead of silently rotating or arming alignment picking with nothing to pick.

**Architecture:** Two independent changes in the existing terminal-viewer codebase (no new modules). `widget.py`'s shared mouse-event handler gains a hit-test-before-branch so a miss on `opt`+left-button starts a pan drag; `viewer.py`'s per-frame overlay dispatch gains a new always-drawn corner element, refactored into its own testable method alongside the existing help-panel/picker dispatch.

**Tech Stack:** Python 3.8+, numpy (camera math), raw ANSI/SGR terminal escapes (no UI framework) — matches the rest of `vimol`.

## Global Constraints

- The persistent bottom-left hint shows exactly these three lines, nothing more: `z` → center, `r` → align, `1-6` → display modes. Do **not** add an `opt+click` line to it (explicit instruction).
- `opt+click`/`opt+drag` on an atom is unchanged in both edit and view mode — only the *miss* (empty-space) case changes.
- A plain `opt+click` (no real drag) on empty space now does nothing; it no longer clears an active alignment/RMSD-picking selection in view mode (confirmed acceptable — "translate wins").
- The new pan gesture is documented in the `?` help panel and the README control rundown, but explicitly **not** in the new bottom-left hint.
- Follow existing code style: no new comments beyond explaining non-obvious "why", reuse existing helpers (`_pick_active_only`, `_sgr_bg`/`_sgr_fg`, `_kv`/`_help_row`/`_help_note`) rather than duplicating them.

---

## File Structure

- Modify `src/vimol/widget.py`: `MoleculeWidget.__init__` (new `_alt_pan_miss` flag) and `MoleculeWidget.handle_mouse` (hit-test-before-branch on `down`; guard the alignment-click shortcut on `up`).
- Modify `src/vimol/viewer.py`: new module constants `_CORNER_HINT_KEY_W`/`_CORNER_HINT_LINES`, new `Viewer._corner_hint_geometry`/`Viewer._draw_corner_hint`, new `Viewer._draw_active_overlay` (extracted from the existing inline dispatch in `_draw`), one new `_help_note` line in `_HELP_HEAD`.
- Modify `README.md`: one clause added to the "In the terminal" walkthrough.
- Modify `tests/test_editing.py`: widget-level coverage for the new empty-space pan behavior (editable and view mode), and the retired clear-selection shortcut.
- Modify `tests/test_vimol.py`: coverage for the corner hint's content/geometry and the overlay-dispatch suppression rule.

---

### Task 1: opt+click on empty space pans instead of rotating/arming alignment

**Files:**
- Modify: `src/vimol/widget.py:94-96` (add `_alt_pan_miss` flag)
- Modify: `src/vimol/widget.py:232-293` (`handle_mouse`, the `down` and `up` action blocks)
- Test: `tests/test_editing.py`

**Interfaces:**
- Consumes: `MoleculeWidget._pick_active_only(px, py) -> Optional[int]` (existing, `widget.py:749`), `MoleculeWidget.pan(dx_px, dy_px) -> None` (existing, `widget.py:188`, invoked indirectly via the existing drag dispatch).
- Produces: `MoleculeWidget._alt_pan_miss: bool` — True for the remainder of an alt-gesture that started on empty space; read by the `up` handler to suppress `_alignment_click`. No other task depends on this.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_editing.py`, right after the existing `_alt_drag` helper (`widget.py` uses `_atom_px`/`_alt_drag`, already imported/defined above this point in the file):

```python
def _far_px(widget):
    """A pixel guaranteed to miss every atom: the widget's far corner."""
    return widget.scene.width - 5.0, widget.scene.height - 5.0


def _alt_click(widget, x, y):
    """Simulate a plain option-click (down+up, no drag) at one pixel."""
    widget.handle_event(MouseEvent("down", x, y, button=0, alt=True, pixel=True))
    return widget.handle_event(MouseEvent("up", x, y, button=0, alt=True, pixel=True))


def test_alt_drag_on_empty_space_pans_when_editable():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    fx, fy = _far_px(w)
    w.handle_event(MouseEvent("down", fx, fy, button=0, alt=True, pixel=True))
    w.handle_event(MouseEvent("drag", fx - 20, fy - 20, button=0, alt=True, pixel=True))
    w.handle_event(MouseEvent("up", fx - 20, fy - 20, button=0, alt=True, pixel=True))
    assert not np.array_equal(w.scene.camera.pan, np.zeros(2))
    assert np.array_equal(w.scene.camera.rotation, np.eye(3))   # not rotated
    assert w._bond_anchor is None                                 # no bond gesture started


def test_alt_down_on_an_atom_still_starts_a_bond_gesture_when_editable():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    w.handle_event(MouseEvent("down", ax, ay, button=0, alt=True, pixel=True))
    assert w._bond_anchor == 0
    assert np.array_equal(w.scene.camera.pan, np.zeros(2))        # gesture, not a pan


def test_alt_drag_on_empty_space_pans_when_not_editable():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    w = MoleculeWidget(mol, 200, 200, backend="cpu")   # editable defaults False
    fx, fy = _far_px(w)
    w.handle_event(MouseEvent("down", fx, fy, button=0, alt=True, pixel=True))
    w.handle_event(MouseEvent("drag", fx - 20, fy - 20, button=0, alt=True, pixel=True))
    w.handle_event(MouseEvent("up", fx - 20, fy - 20, button=0, alt=True, pixel=True))
    assert not np.array_equal(w.scene.camera.pan, np.zeros(2))
    assert w.align_mode is False       # a miss no longer arms alignment picking


def test_alt_click_on_an_atom_still_arms_alignment_when_not_editable():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    ax, ay = _atom_px(w, 0)
    _alt_click(w, ax, ay)
    assert w.align_mode is True
    assert w.align_sel == [0]


def test_alt_click_on_empty_space_no_longer_clears_an_active_alignment_selection():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    ax, ay = _atom_px(w, 0)
    _alt_click(w, ax, ay)                 # picks atom 0, arms alignment mode
    assert w.align_sel == [0]
    fx, fy = _far_px(w)
    _alt_click(w, fx, fy)                 # a plain click (no drag) on empty space
    assert w.align_sel == [0]             # retired: opt+click there now pans instead
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_editing.py -k "alt_drag_on_empty_space or alt_down_on_an_atom or alt_click_on_an_atom or alt_click_on_empty_space" -v`
Expected: FAIL — `test_alt_drag_on_empty_space_pans_when_editable` and the `_when_not_editable` variant fail because `camera.pan` is still `[0, 0]` (today's miss falls through to a plain rotate); `test_alt_click_on_empty_space_no_longer_clears...` fails because `align_sel` becomes `[]`.

- [ ] **Step 3: Add the `_alt_pan_miss` flag**

In `src/vimol/widget.py`, in `MoleculeWidget.__init__`, change:

```python
        self._drag_button: Optional[int] = None
        self._drag_shift = False
        self._last = (0.0, 0.0)
```

to:

```python
        self._drag_button: Optional[int] = None
        self._drag_shift = False
        # True for the rest of an alt-gesture that started on empty space
        # (opt+click/opt+drag pans there instead of bonding/picking) -- read
        # by the 'up' handler to suppress the alignment-click shortcut.
        self._alt_pan_miss = False
        self._last = (0.0, 0.0)
```

- [ ] **Step 4: Rewrite the `down` action block**

In `src/vimol/widget.py`, `handle_mouse`, replace:

```python
        if ev.action == "down":
            # A fresh press while a bond gesture is still live means its 'up'
            # was lost (focus change, dropped event): tear the stale gesture
            # down so its preview arrow can't be orphaned in vector_fields.
            if self._bond_anchor is not None:
                self._cancel_bond_gesture()
            if ev.button == 0 and ev.alt and self.editable:
                # The option gesture acts on the active structure, so it must
                # see the active structure: a composite pick would let a
                # tinted overlay atom in front intercept the press and lose
                # both the bond anchor and the subset-pick shortcut.
                idx = self._pick_active_only(x, y)
                if idx is not None:
                    # Start a bond gesture -- and deliberately do NOT set
                    # _drag_button, so the drag branch below won't rotate the
                    # camera while the gesture is live.
                    self._bond_anchor = idx
                    self._press = (x, y)
                    self._bond_drag_distance2 = 0.0
                    self._start_bond_preview(idx)
                    return False
                # alt+down over empty space: fall through to a normal press.
            elif ev.button == 0 and ev.alt:
                # Option-click is the always-available shortcut into subset
                # picking. Preserve a loaded named selection so the click
                # edits a live copy rather than starting from nothing.
                self.set_alignment_mode(True, preserve=True)
            self._drag_button = ev.button
            self._drag_shift = ev.shift
            self._last = (x, y)
            self._press = (x, y)
            if self.picking:
                self.selected = self._active_local_pick(x, y)
            return False
```

with:

```python
        if ev.action == "down":
            # A fresh press while a bond gesture is still live means its 'up'
            # was lost (focus change, dropped event): tear the stale gesture
            # down so its preview arrow can't be orphaned in vector_fields.
            if self._bond_anchor is not None:
                self._cancel_bond_gesture()
            self._alt_pan_miss = False
            if ev.button == 0 and ev.alt:
                # The option gesture acts on the active structure, so it must
                # see the active structure: a composite pick would let a
                # tinted overlay atom in front intercept the press and lose
                # both the bond anchor and the subset-pick shortcut.
                idx = self._pick_active_only(x, y)
                if idx is not None and self.editable:
                    # Start a bond gesture -- and deliberately do NOT set
                    # _drag_button, so the drag branch below won't rotate the
                    # camera while the gesture is live.
                    self._bond_anchor = idx
                    self._press = (x, y)
                    self._bond_drag_distance2 = 0.0
                    self._start_bond_preview(idx)
                    return False
                if idx is not None:
                    # Option-click is the always-available shortcut into
                    # subset picking. Preserve a loaded named selection so
                    # the click edits a live copy rather than starting from
                    # nothing.
                    self.set_alignment_mode(True, preserve=True)
                else:
                    # opt+click/opt+drag over empty space pans the scene
                    # instead of rotating (edit mode) or arming alignment
                    # picking with nothing to pick (view mode).
                    self._alt_pan_miss = True
            self._drag_button = ev.button
            self._drag_shift = self._alt_pan_miss or ev.shift
            self._last = (x, y)
            self._press = (x, y)
            if self.picking:
                self.selected = self._active_local_pick(x, y)
            return False
```

- [ ] **Step 5: Guard the alignment-click shortcut in the `up` action block**

In `src/vimol/widget.py`, `handle_mouse`, change:

```python
            elif self.align_mode and was_left and not ev.shift:
                dx = x - self._press[0]
                dy = y - self._press[1]
                if dx * dx + dy * dy <= 9.0:
                    return self._alignment_click(x, y)
```

to:

```python
            elif self.align_mode and was_left and not ev.shift and not self._alt_pan_miss:
                dx = x - self._press[0]
                dy = y - self._press[1]
                if dx * dx + dy * dy <= 9.0:
                    return self._alignment_click(x, y)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_editing.py -k "alt_drag_on_empty_space or alt_down_on_an_atom or alt_click_on_an_atom or alt_click_on_empty_space" -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `pytest tests/test_editing.py tests/test_align.py -v`
Expected: PASS — in particular, the existing alt-drag-on-an-atom bond tests and the RMSD subset-picking tests in `test_align.py` (which always mock `_pick_active_only` to return a hit) are unaffected.

- [ ] **Step 8: Commit**

```bash
git add src/vimol/widget.py tests/test_editing.py
git commit -m "$(cat <<'EOF'
opt+click on empty space now pans instead of rotating or picking

A miss on the option gesture used to silently fall through to a plain
rotate in edit mode, or unconditionally arm alignment picking (and let
a follow-up empty-space click clear it) in view mode. Both now start a
pan drag instead, matching right/mid-drag -- hitting an atom is
unchanged in either mode.
EOF
)"
```

---

### Task 2: Persistent bottom-left control hint

**Files:**
- Modify: `src/vimol/viewer.py:169-175` (new module constants, next to `_HELP_TITLE`/`_HELP_KEY_W`/`_HELP_COL_W`)
- Modify: `src/vimol/viewer.py:1994-2003` (extract `_draw_active_overlay`, add the corner-hint branch)
- Modify: `src/vimol/viewer.py` (new `_corner_hint_geometry`/`_draw_corner_hint` methods, placed next to `_help_geometry`/`_draw_help`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: `Viewer._sgr_bg`/`Viewer._sgr_fg` (existing, `viewer.py:1219-1224`), `Viewer.theme.help_bg`/`Viewer.theme.list_muted_fg` (existing `Theme` fields, `theme.py`), `Viewer._list_w`/`Viewer._measure_w`/`Viewer._cols`/`Viewer._rows` (existing instance state), `kitty.write_bytes` (existing).
- Produces: `Viewer._corner_hint_geometry() -> Tuple[int, int, int, int]` (0-based `top, left, width, height`), `Viewer._draw_corner_hint() -> bytes`, `Viewer._draw_active_overlay() -> None` (replaces the inline dispatch in `_draw`; no other task depends on these beyond this one).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_vimol.py`, near the existing `_help_rows`/help-panel tests (after `_help_rows`, i.e. after line 2834):

```python
def _corner_hint_rows(v):
    """{0-based screen row: the SGR text drawn there} for one _draw_corner_hint()."""
    parts = re.split(r"\x1b\[(\d+);(\d+)H", v._draw_corner_hint().decode("utf-8", "replace"))
    return {int(parts[i]) - 1: parts[i + 2] for i in range(1, len(parts), 3)}


def test_corner_hint_shows_exactly_the_three_essential_bindings(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v._cols, v._rows = 100, 30
        v._list_w = 0
        rows = _corner_hint_rows(v)
        text = "".join(_visible(t) for t in rows.values())
        assert "z" in text and "center" in text
        assert "r" in text and "align" in text
        assert "1-6" in text and "display modes" in text
        assert "opt" not in text.lower()          # not documented here, by design
    finally:
        os.close(fd)


def test_corner_hint_sits_above_the_status_bar_right_of_the_structure_list(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v._cols, v._rows = 100, 30
        v._list_w = 20
        top, left, width, height = v._corner_hint_geometry()
        assert left == v._list_w + v._measure_w
        assert top + height <= v._rows - 1        # 0-based: status bar sits at row _rows-1
    finally:
        os.close(fd)


def test_draw_active_overlay_shows_corner_hint_by_default(tmp_path, monkeypatch):
    v, fd = _multi_viewer(tmp_path)
    try:
        calls = []
        for name in ("_draw_help", "_draw_periodic_table", "_draw_geometry_picker",
                     "_draw_selection_picker", "_draw_file_browser", "_draw_corner_hint"):
            monkeypatch.setattr(v, name, lambda *_a, n=name: calls.append(n))
        v._draw_active_overlay()
        assert calls == ["_draw_corner_hint"]
    finally:
        os.close(fd)


@pytest.mark.parametrize("mode_attr,mode_value", [
    ("_show_help", True),
    ("_mode", "periodic_table"),
    ("_mode", "geometry_picker"),
    ("_mode", "selection_picker"),
    ("_mode", "file_browser"),
])
def test_draw_active_overlay_suppresses_corner_hint_for_other_overlays(
        tmp_path, monkeypatch, mode_attr, mode_value):
    v, fd = _multi_viewer(tmp_path)
    try:
        calls = []
        for name in ("_draw_help", "_draw_periodic_table", "_draw_geometry_picker",
                     "_draw_selection_picker", "_draw_file_browser", "_draw_corner_hint"):
            monkeypatch.setattr(v, name, lambda *_a, n=name: calls.append(n))
        setattr(v, mode_attr, mode_value)
        v._draw_active_overlay()
        assert calls != ["_draw_corner_hint"]
        assert "_draw_corner_hint" not in calls
    finally:
        os.close(fd)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_vimol.py -k corner_hint -v`
Expected: FAIL with `AttributeError: 'Viewer' object has no attribute '_draw_corner_hint'` (and `_corner_hint_geometry`, `_draw_active_overlay`).

- [ ] **Step 3: Add the module constants**

In `src/vimol/viewer.py`, right after:

```python
_HELP_TITLE = " vimol — terminal molecular viewer "
_HELP_KEY_W = 18        # 'key' plus its dot leader, before the value's space
_HELP_COL_W = 36        # one whole key/value column, left column to right
```

add:

```python

# Persistent bottom-left control hint (unlike the '?' panel above, this one
# is never toggled -- drawn every frame the same way the status bar is).
_CORNER_HINT_KEY_W = 5
_CORNER_HINT_LINES = [
    f"{'z':<{_CORNER_HINT_KEY_W}}center",
    f"{'r':<{_CORNER_HINT_KEY_W}}align",
    f"{'1-6':<{_CORNER_HINT_KEY_W}}display modes",
]
```

- [ ] **Step 4: Add `_corner_hint_geometry` and `_draw_corner_hint`**

In `src/vimol/viewer.py`, immediately after `_draw_help` (which ends with `return bytes(out)` following the `kitty.write_bytes(bytes(out), self.fd_out)` call, `viewer.py:2047-2048`), add:

```python
    def _corner_hint_geometry(self) -> Tuple[int, int, int, int]:
        """(top, left, width, height) of the persistent bottom-left control
        hint, 0-based cell coords -- anchored to the render viewport (right
        of the structure list, never over it), stacked directly above the
        status bar."""
        lines = _CORNER_HINT_LINES
        left = self._list_w + self._measure_w
        avail_w = max(self._cols - left, 1)
        width = min(max(len(l) for l in lines), avail_w)
        height = min(len(lines), max(self._rows - 1, 1))
        top = max(0, (self._rows - 1) - height)
        return (top, left, width, height)

    def _draw_corner_hint(self) -> bytes:
        top, left, width, height = self._corner_hint_geometry()
        bg = self._sgr_bg(self.theme.help_bg)
        fg = self._sgr_fg(self.theme.list_muted_fg)
        out = bytearray()
        for k in range(height):
            row = _CORNER_HINT_LINES[k][:width].ljust(width)
            out.extend(b"\x1b[%d;%dH" % (top + k + 1, left + 1))
            out.extend(f"{bg}{fg}{row}\x1b[0m".encode("utf-8", "replace"))
        kitty.write_bytes(bytes(out), self.fd_out)
        return bytes(out)

    def _draw_active_overlay(self) -> None:
        """Draw whichever single overlay currently owns the screen -- the
        '?' help panel, a modal picker, or (when none of those claim it)
        the persistent bottom-left control hint."""
        if self._show_help:
            self._draw_help()
        elif self._mode == "periodic_table":
            self._draw_periodic_table()
        elif self._mode == "geometry_picker":
            self._draw_geometry_picker()
        elif self._mode == "selection_picker":
            self._draw_selection_picker()
        elif self._mode == "file_browser":
            self._draw_file_browser()
        else:
            self._draw_corner_hint()
```

- [ ] **Step 5: Wire it into the render loop**

In `src/vimol/viewer.py`, `_draw`, replace:

```python
        if self._show_help:
            self._draw_help()
        elif self._mode == "periodic_table":
            self._draw_periodic_table()
        elif self._mode == "geometry_picker":
            self._draw_geometry_picker()
        elif self._mode == "selection_picker":
            self._draw_selection_picker()
        elif self._mode == "file_browser":
            self._draw_file_browser()
```

with:

```python
        self._draw_active_overlay()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_vimol.py -k corner_hint -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Run the full test suite to check for regressions**

Run: `pytest tests/test_vimol.py -v`
Expected: PASS — including the existing `test_viewer_help_panel_*` tests, which call `_draw_help()` directly and are unaffected by the `_draw()`-internal refactor.

- [ ] **Step 8: Commit**

```bash
git add src/vimol/viewer.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
Add a persistent bottom-left control hint

z / r / 1-6 are shown always, not just behind '?' -- a new corner
caption drawn every frame like the status bar, suppressed whenever the
help panel or a modal picker already owns the screen.
EOF
)"
```

---

### Task 3: Document opt+click-empty-space pan in the help panel and README

**Files:**
- Modify: `src/vimol/viewer.py:170-176` (`_HELP_HEAD`)
- Modify: `README.md:23-24`
- Test: none new (existing help-panel tests already assert on substrings, not exact line counts)

**Interfaces:**
- Consumes: `_help_note` (existing helper, `viewer.py:165`).
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Add the binding line to the help panel**

In `src/vimol/viewer.py`, change:

```python
_HELP_HEAD = [
    _help_row(_kv("Mouse drag", "rotate"), _kv("Wheel / + -", "zoom")),
    _help_row(_kv("Right / mid drag", "pan"), _kv("[ / ]", "roll")),
    _help_row(_kv("Hover", "identify atom"), _kv("Arrows / h j k l", "rotate")),
    _help_row(_kv("1 2 3 4", "ball / spacefill / licorice / wire")),
    _help_row(_kv("5 / 6", "ribbon / glyph: lettered residues (proteins)")),
]
```

to:

```python
_HELP_HEAD = [
    _help_row(_kv("Mouse drag", "rotate"), _kv("Wheel / + -", "zoom")),
    _help_row(_kv("Right / mid drag", "pan"), _kv("[ / ]", "roll")),
    _help_note("opt+drag empty space · also pans"),
    _help_row(_kv("Hover", "identify atom"), _kv("Arrows / h j k l", "rotate")),
    _help_row(_kv("1 2 3 4", "ball / spacefill / licorice / wire")),
    _help_row(_kv("5 / 6", "ribbon / glyph: lettered residues (proteins)")),
]
```

- [ ] **Step 2: Add the clause to the README**

In `README.md`, change:

```markdown
`vimol` with no file opens this bundled C60 demo. Drag to rotate, scroll to
zoom, hover an atom to identify it, `m` to measure distances/angles/dihedrals
```

to:

```markdown
`vimol` with no file opens this bundled C60 demo. Drag to rotate, scroll to
zoom, option-drag empty space to pan, hover an atom to identify it, `m` to
measure distances/angles/dihedrals
```

- [ ] **Step 3: Run the full test suite to check for regressions**

Run: `pytest tests/test_vimol.py -k help -v`
Expected: PASS — `test_viewer_help_panel_is_a_closed_box_of_uniform_width` and `test_viewer_help_panel_never_reaches_the_status_bar` only assert on substrings/box shape, not exact content, so the new line doesn't break them.

- [ ] **Step 4: Commit**

```bash
git add src/vimol/viewer.py README.md
git commit -m "$(cat <<'EOF'
Document opt+drag-empty-space pan in the help panel and README

EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (corner hint content, suppression, styling) → Task 2. §2 (hit-test-before-branch, pan-on-miss, retiring the clear-selection shortcut) → Task 1. §3 (docs) → Task 3. Testing section's three bullets → Task 1 Step 1 (behavior), Task 2 Step 1 (hint presence/absence).
- **Placeholder scan:** none found — every step has literal code and exact commands.
- **Type consistency:** `_corner_hint_geometry` returns `Tuple[int, int, int, int]` matching `_help_geometry`'s existing signature/import (`Tuple` already imported in `viewer.py` for that reason); `_alt_pan_miss: bool` is read only within `handle_mouse`, no cross-file signature to keep in sync.
