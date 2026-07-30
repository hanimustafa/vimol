# Theme System, Color Sweep & Copyable Status Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the interactive viewer a detected dark/light theme with a `ctrl-t` override, fix the color bugs and gaps that surface on a light terminal, and make the status bar's left-field text (hover info, molecule name, live measurement, or the xyz comment — often an energy value) copyable via an OSC 52 yank.

**Architecture:** One new pure module (`theme.py`, `Theme` dataclass + `DARK`/`LIGHT` + detection helpers) that `viewer.py` reads from instead of its current module-level color constants. `kitty.py` gains two small protocol additions (OSC 11 background query folded into the existing combined startup probe; an OSC 52 clipboard-write helper). `elements.py` and `widget.py` gain a light-mode color-override path that reuses the existing `style.color_override` mechanism, so `Molecule`/`StructureSet`/`render.py` need no changes at all. `parsers/xyz.py` stops truncating the comment line at parse time.

**Tech Stack:** Python ≥3.8, numpy only. Tests with pytest (`python3 -m pytest`, not `python` — not on PATH in this environment).

**Spec:** `docs/design/theme-and-aesthetics.md` (VIM-12; implements VIM-13, VIM-14, VIM-15)

> **Amended during execution** (both on review feedback; the spec is the
> authority, this plan is kept as the historical record):
> 1. **Task 3 and Task 9 were reverted.** The OSC 52 `osc52_copy` helper and
>    the `y` yank binding are gone — shift+drag already gives native
>    selection in kitty/Ghostty/WezTerm. Task 5's xyz full-comment fix stays.
> 2. **Task 7's strip background fix was inverted.** Ordinary strip rows stay
>    transparent (`bg=None`) rather than painting a `list_panel_bg`; light
>    readability comes from the theme's foreground palette instead. The
>    `list_panel_bg` field does not exist. Painting the panel opaque also made
>    the OSC 11 theme correction visible as a black-then-recolor flash.

## Global Constraints

- Only dependency is `numpy>=1.20`; do not add others (no `pyperclip` or similar for the OSC 52 clipboard write — it's a plain escape sequence written to `fd_out`, no library needed).
- Run tests as `python3 -m pytest` from the repo root.
- Tests live in `tests/test_vimol.py` (this repo keeps its viewer/kitty/elements/xyz tests in one file, not split per module) — reuse its existing imports (`vimol`, `elements`, `kitty`, `Renderer`, `Style`, `Scene`, `loads`) and its `EX` (examples dir) and `_multi_viewer(tmp_path, **kw)` helper.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Every `Theme` field name below is fixed by the spec (`docs/design/theme-and-aesthetics.md` §1) — do not rename or consolidate fields even where two look similar; the mapping from old module constant to new field must stay 1:1 and mechanical.
- `DARK`'s values must be byte-for-byte the values of the constants it replaces — the dark theme is a no-op refactor. Anything that renders differently under `DARK` after this work is a bug.

---

### Task 1: `theme.py` — Theme dataclass, DARK/LIGHT, detection helpers

**Files:**
- Create: `src/vimol/theme.py`
- Test: `tests/test_vimol.py`

**Interfaces:**
- Produces: `Theme` (frozen dataclass, fields listed below), `DARK: Theme`, `LIGHT: Theme`, `luminance(rgb: Tuple[int,int,int]) -> float`, `from_colorfgbg(value: str) -> Optional[Theme]`, `resolve(explicit: Optional[str], osc11_rgb: Optional[Tuple[int,int,int]], colorfgbg: Optional[str]) -> Theme`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_vimol.py` (near the top-level tests, after `test_element_data`):

```python
def test_theme_dark_matches_original_constants():
    from vimol import theme
    assert theme.DARK.panel_bg == (30, 33, 44)
    assert theme.DARK.panel_fg == (230, 232, 240)
    assert theme.DARK.list_active_bg == (37, 45, 64)
    assert theme.DARK.pt_border_fg == (60, 200, 180)
    assert theme.DARK.cleanup_hint_fg == (255, 170, 60)


def test_theme_luminance():
    from vimol import theme
    assert theme.luminance((255, 255, 255)) == pytest.approx(255.0)
    assert theme.luminance((0, 0, 0)) == 0.0
    assert theme.luminance((255, 170, 60)) == pytest.approx(
        0.299 * 255 + 0.587 * 170 + 0.114 * 60)


def test_theme_from_colorfgbg():
    from vimol import theme
    assert theme.from_colorfgbg("15;0") is theme.DARK
    assert theme.from_colorfgbg("0;15") is theme.LIGHT
    assert theme.from_colorfgbg("0;7") is theme.LIGHT
    assert theme.from_colorfgbg("15;1") is theme.DARK
    assert theme.from_colorfgbg("garbage") is None
    assert theme.from_colorfgbg("") is None


def test_theme_resolve_precedence():
    from vimol import theme
    # explicit wins over everything
    assert theme.resolve("light", (10, 10, 10), "15;0") is theme.LIGHT
    assert theme.resolve("dark", (250, 250, 250), "0;15") is theme.DARK
    # osc11 wins over colorfgbg
    assert theme.resolve(None, (245, 245, 245), "15;0") is theme.LIGHT
    assert theme.resolve(None, (10, 10, 10), "0;15") is theme.DARK
    # colorfgbg is the fallback when osc11 is unknown
    assert theme.resolve(None, None, "0;15") is theme.LIGHT
    # default is dark
    assert theme.resolve(None, None, None) is theme.DARK
    assert theme.resolve(None, None, "garbage") is theme.DARK
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k theme -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vimol.theme'`

- [ ] **Step 3: Write `src/vimol/theme.py`**

```python
"""Chrome color palettes for the interactive viewer, dark and light.

Every field here replaces a module-level color constant that used to live in
viewer.py -- see docs/design/theme-and-aesthetics.md sec 1 for the mapping.
DARK's values are exactly those old constants (a no-op refactor); LIGHT is
new content, tuned for a white/light terminal background.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    panel_bg: RGB
    panel_fg: RGB
    edit_prefix_fg: RGB
    warn_bg: RGB
    warn_fg: RGB
    input_bg: RGB
    input_fg: RGB
    help_bg: RGB
    help_fg: RGB
    list_header_fg: RGB
    list_muted_fg: RGB
    list_label_fg: RGB
    list_dim_fg: RGB
    list_active_bg: RGB
    list_cursor_bg: RGB
    list_rule_fg: RGB
    list_cap_bg: RGB
    list_panel_bg: RGB
    cleanup_hint_fg: RGB
    pt_bg: RGB
    pt_border_fg: RGB
    pt_text_fg: RGB
    pt_dim_fg: RGB
    pt_gap_bg: RGB


DARK = Theme(
    name="dark", panel_bg=(30, 33, 44), panel_fg=(230, 232, 240),
    edit_prefix_fg=(150, 155, 170), warn_bg=(60, 30, 30), warn_fg=(250, 230, 230),
    input_bg=(44, 40, 30), input_fg=(240, 236, 220),
    help_bg=(20, 22, 30), help_fg=(220, 220, 230),
    list_header_fg=(139, 146, 165), list_muted_fg=(110, 118, 135),
    list_label_fg=(232, 236, 244), list_dim_fg=(200, 206, 216),
    list_active_bg=(37, 45, 64), list_cursor_bg=(28, 33, 46),
    list_rule_fg=(60, 66, 84), list_cap_bg=(42, 49, 66),
    list_panel_bg=(18, 20, 26), cleanup_hint_fg=(255, 170, 60),
    pt_bg=(18, 20, 26), pt_border_fg=(60, 200, 180),
    pt_text_fg=(220, 220, 230), pt_dim_fg=(110, 114, 126), pt_gap_bg=(40, 42, 50),
)

LIGHT = Theme(
    name="light", panel_bg=(225, 227, 232), panel_fg=(30, 32, 38),
    edit_prefix_fg=(90, 95, 110), warn_bg=(255, 225, 225), warn_fg=(120, 20, 20),
    input_bg=(255, 247, 214), input_fg=(90, 70, 10),
    help_bg=(238, 239, 243), help_fg=(35, 37, 44),
    list_header_fg=(70, 76, 95), list_muted_fg=(120, 126, 142),
    list_label_fg=(20, 22, 28), list_dim_fg=(55, 60, 72),
    list_active_bg=(202, 210, 230), list_cursor_bg=(216, 220, 230),
    list_rule_fg=(190, 195, 206), list_cap_bg=(206, 211, 222),
    list_panel_bg=(233, 235, 240), cleanup_hint_fg=(170, 90, 0),
    pt_bg=(238, 240, 244), pt_border_fg=(0, 140, 125),
    pt_text_fg=(30, 32, 38), pt_dim_fg=(120, 125, 138), pt_gap_bg=(220, 223, 230),
)


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def from_colorfgbg(value: str) -> Optional[Theme]:
    """COLORFGBG is "fg;bg" (basic-16-color indices). 7 or 15 (white/
    bright-white) in the background field means a light terminal; any other
    parseable value means dark. Unparseable or empty -> None (skip to the
    next fallback)."""
    if not value:
        return None
    parts = value.split(";")
    if not parts:
        return None
    try:
        bg_code = int(parts[-1])
    except ValueError:
        return None
    return LIGHT if bg_code in (7, 15) else DARK


def resolve(explicit: Optional[str], osc11_rgb: Optional[Tuple[int, int, int]],
            colorfgbg: Optional[str]) -> Theme:
    """explicit ("dark"/"light") -> OSC 11 background color -> COLORFGBG -> DARK."""
    if explicit == "light":
        return LIGHT
    if explicit == "dark":
        return DARK
    if osc11_rgb is not None:
        return LIGHT if luminance(osc11_rgb) > 140 else DARK
    guess = from_colorfgbg(colorfgbg) if colorfgbg else None
    if guess is not None:
        return guess
    return DARK
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k theme -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/vimol/theme.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
Add theme.py: dark/light chrome palettes + detection precedence (VIM-13)

Pure module, no viewer wiring yet: DARK reproduces every existing
chrome color constant exactly, LIGHT is new. resolve() centralizes
the explicit -> OSC 11 -> COLORFGBG -> dark fallback ladder so it's
testable without a real terminal.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `kitty.py` — OSC 11 background-color detection

**Files:**
- Modify: `src/vimol/kitty.py:245-284` (TerminalProbe dataclass + regexes), `src/vimol/kitty.py:289-312` (`probe_query_bytes`), `src/vimol/kitty.py:315-379` (`_parse_probe_pieces` / `parse_probe_reply`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TerminalProbe.bg_rgb: Optional[Tuple[int, int, int]]`; `probe_query_bytes()` output now includes the OSC 11 query; `parse_probe_reply()`/`_parse_probe_pieces()` extract it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_vimol.py`:

```python
def test_probe_query_bytes_includes_osc11():
    data = kitty.probe_query_bytes()
    assert b"\x1b]11;?\x1b\\" in data


def test_parse_probe_reply_extracts_osc11_background_st_terminated():
    buf = (b"\x1b_Gi=31;OK\x1b\\"
           b"\x1b[?1016;2$y"
           b"\x1b]11;rgb:1e1e/2020/2828\x1b\\"
           b"\x1b[6;18;9t"
           b"\x1b[?62c")
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.bg_rgb == (0x1e, 0x20, 0x28)


def test_parse_probe_reply_extracts_osc11_background_bel_terminated_two_digit():
    buf = (b"\x1b_Gi=31;OK\x1b\\"
           b"\x1b]11;rgb:f0/f2/f5\x07"
           b"\x1b[?62c")
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.bg_rgb == (0xf0, 0xf2, 0xf5)


def test_parse_probe_reply_bg_rgb_none_when_terminal_silent():
    buf = b"\x1b[?62c"   # only the DA1 fence answered
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.bg_rgb is None


def test_parse_probe_reply_bg_rgb_none_before_da1_fence():
    # no DA1 yet -> whole probe is None, not a premature verdict
    buf = b"\x1b]11;rgb:1e1e/2020/2828\x1b\\"
    assert kitty.parse_probe_reply(buf) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k osc11 -v`
Expected: FAIL — `probe_query_bytes()` has no OSC 11 query yet, `TerminalProbe` has no `bg_rgb` attribute (AttributeError).

- [ ] **Step 3: Implement**

In `src/vimol/kitty.py`, add the `bg_rgb` field to `TerminalProbe` (right after `shm`, in the dataclass starting at line 245):

```python
    # True when the terminal accepted a shared-memory (t=s) transfer, i.e. it
    # runs on this machine and supports the protocol's local-client path;
    # False when it refused or ignored it; None when we never asked.
    shm: Optional[bool] = None
    # The terminal's own background color from an OSC 11 query, downsampled
    # to 8 bits/channel; None if it didn't answer (theme.resolve() then
    # falls back to COLORFGBG/default -- see theme.py).
    bg_rgb: Optional[Tuple[int, int, int]] = None
```

Add the OSC 11 query and its reply regex near the other `_RE_*` patterns (after `_RE_DA1`, still before `probe_query_bytes`):

```python
_OSC11_QUERY = b"\x1b]11;?\x1b\\"
_RE_OSC11_REPLY = re.compile(
    rb"\x1b\]11;rgb:([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})(?:\x1b\\|\x07)")


def _osc11_channel(hexstr: bytes) -> int:
    """A 2- or 4-hex-digit OSC 11 channel, downsampled to 8 bits: terminals
    disagree on whether they reply with 8 or 16 bits/channel, and for a
    16-bit channel the high byte is what every other tool in the wild uses
    as the 8-bit approximation."""
    return int(hexstr[:2], 16)
```

Add the query to `probe_query_bytes` (`kitty.py:289`) — append it to the returned bytes, updating the docstring to mention the fifth question:

```python
def probe_query_bytes(shm_name: Optional[str] = None) -> bytes:
    """The combined capability query, sent as ONE write (one SSH round trip).

    Five questions back to back: (1) a Kitty graphics *query* (``a=q`` with a
    1x1 dummy pixel -- validated and answered, never displayed or stored);
    (2) DECRQM for SGR-Pixels mouse (1016); (3) ``CSI 16 t`` for the exact
    cell size; (4) an OSC 11 query for the terminal's own background color
    (theme auto-detection, see theme.py); (5) DA1 (``CSI c``) as a
    universally-answered fence. Terminals ignore the queries they don't
    recognize (the graphics APC included), so this is safe to fire at
    anything that calls itself a terminal. Requires the tty to be in raw
    mode to read the replies.

    Pass *shm_name* (a shared-memory object holding one 24-bit pixel, from
    :func:`shm_write`) to add a sixth question: a ``t=s`` query. Answering it
    OK requires the terminal to actually open that object, so the reply
    proves both "supports the local-client path" and "runs on this machine"
    -- no env-var guessing about SSH. The terminal unlinks the object when it
    reads it, and a refusal leaves it for :func:`shm_cleanup`.
    """
    gfx = (b"\x1b_Gi=%d,s=1,v=1,a=q,t=d,f=24;" % _PROBE_GFX_ID
           + base64.standard_b64encode(b"\x00\x00\x00") + b"\x1b\\")
    if shm_name is not None:
        gfx += (b"\x1b_Gi=%d,s=1,v=1,a=q,t=s,f=24;" % _PROBE_SHM_ID
                + base64.standard_b64encode(shm_name.encode()) + b"\x1b\\")
    return (gfx + b"\x1b[?1016$p" + b"\x1b[16t" + _OSC11_QUERY + b"\x1b[c")
```

Extend `_parse_probe_pieces` (`kitty.py:315`) to extract `bg_rgb` and return it (six-tuple instead of five):

```python
def _parse_probe_pieces(buf: bytes):
    """Extract (graphics, pixel_mouse, cell_px, shm, bg_rgb, spans) from
    reply bytes.

    graphics is None when no graphics reply is present at all -- only the
    caller knows whether that silence is meaningful (it is once the DA1
    fence has arrived). shm, by contrast, is False on silence: an unanswered
    t=s query is a "no" the same way an error reply is, and callers that
    never asked overwrite it with None. bg_rgb is simply None on silence --
    there is no "asked and refused" state for OSC 11, only answered/silent.
    """
    spans = []
    graphics = None
    m = _RE_GFX_REPLY.search(buf)
    if m:
        graphics = m.group(1).startswith(b"OK")
        spans.append(m.span())
    shm = False
    m = _RE_SHM_REPLY.search(buf)
    if m:
        shm = m.group(1).startswith(b"OK")
        spans.append(m.span())
    pixel = False
    m = _RE_DECRQM_1016.search(buf)
    if m:
        # 1=set 2=reset 3=perm-set 4=perm-reset: any of these means the mode
        # is *recognized*; 0 means unknown.
        pixel = int(m.group(1)) in (1, 2, 3, 4)
        spans.append(m.span())
    cell = None
    m = _RE_CELL_SIZE.search(buf)
    if m:
        ch, cw = int(m.group(1)), int(m.group(2))   # reply is height;width
        if cw > 0 and ch > 0:
            cell = (float(cw), float(ch))
        spans.append(m.span())
    bg_rgb = None
    m = _RE_OSC11_REPLY.search(buf)
    if m:
        bg_rgb = (_osc11_channel(m.group(1)), _osc11_channel(m.group(2)),
                  _osc11_channel(m.group(3)))
        spans.append(m.span())
    return graphics, pixel, cell, shm, bg_rgb, spans
```

Update both call sites of `_parse_probe_pieces` to unpack six values and thread `bg_rgb` into the `TerminalProbe` they build — `parse_probe_reply` (`kitty.py:363`):

```python
def parse_probe_reply(buf: bytes) -> Optional[TerminalProbe]:
    """Parse an accumulating reply buffer; None until the DA1 fence arrives.

    Once the fence is in, a missing graphics reply is a definitive "no
    graphics support" (the terminal processed our queries in order and
    answered the later one), so ``graphics`` is always True/False here. The
    same in-order reasoning makes a missing t=s reply a definitive "no
    shared memory". A missing OSC 11 reply, by contrast, stays None --
    plenty of terminals just don't implement it, and that's not a verdict
    the way "no graphics" is.
    """
    m_da1 = _RE_DA1.search(buf)
    if m_da1 is None:
        return None
    graphics, pixel, cell, shm, bg_rgb, spans = _parse_probe_pieces(buf)
    spans.append(m_da1.span())
    return TerminalProbe(graphics=bool(graphics), pixel_mouse=pixel, cell_px=cell,
                         leftover=_probe_leftover(buf, spans), shm=shm, bg_rgb=bg_rgb)
```

And the timeout path inside `probe_terminal` (`kitty.py:437`):

```python
    graphics, pixel, cell, shm, bg_rgb, spans = _parse_probe_pieces(buf)
    return _finish(TerminalProbe(graphics=graphics, pixel_mouse=pixel, cell_px=cell,
                                 leftover=_probe_leftover(buf, spans), shm=shm, bg_rgb=bg_rgb))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k "osc11 or probe" -v`
Expected: all passed

- [ ] **Step 5: Run the full existing kitty/probe-adjacent tests to check nothing regressed**

Run: `python3 -m pytest tests/test_vimol.py -k "kitty or backend or mouse_enable" -v`
Expected: all passed (unchanged behavior for everything except the new field)

- [ ] **Step 6: Commit**

```bash
git add src/vimol/kitty.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
kitty: fold an OSC 11 background query into the startup probe (VIM-13)

One more question in the existing combined write/read -- no extra
round trip. TerminalProbe.bg_rgb is None when the terminal doesn't
answer (most don't implement OSC 11; that's not the same "definitive
no" the DA1-fenced graphics/shm replies get).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `kitty.py` — OSC 52 clipboard write

**Files:**
- Modify: `src/vimol/kitty.py` (add near the other write helpers, after `probe_terminal`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Produces: `osc52_copy(text: str) -> bytes`

- [ ] **Step 1: Write the failing tests**

```python
def test_osc52_copy_encodes_base64_clipboard_sequence():
    seq = kitty.osc52_copy("hello")
    assert seq == b"\x1b]52;c;" + base64.standard_b64encode(b"hello") + b"\x1b\\"


def test_osc52_copy_handles_unicode_and_padding_edge_cases():
    for text in ("d(#1-#2) = 1.523 Å", "a", "ab", "abc", ""):
        seq = kitty.osc52_copy(text)
        assert seq.startswith(b"\x1b]52;c;")
        assert seq.endswith(b"\x1b\\")
        payload = seq[len(b"\x1b]52;c;"):-len(b"\x1b\\")]
        assert base64.standard_b64decode(payload) == text.encode("utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k osc52 -v`
Expected: FAIL with `AttributeError: module 'vimol.kitty' has no attribute 'osc52_copy'`

- [ ] **Step 3: Implement**

Add to `src/vimol/kitty.py`, after `probe_terminal` and before the `# Image encoding` section header:

```python
# --------------------------------------------------------------------------
# Clipboard write (OSC 52)
# --------------------------------------------------------------------------
def osc52_copy(text: str) -> bytes:
    """Bytes that write *text* to the system clipboard via OSC 52.

    Supported by kitty, Ghostty, WezTerm, and iTerm2 directly, and passed
    through by tmux with ``set-clipboard on`` -- the standard way a terminal
    application writes the clipboard over SSH with no local clipboard
    access of its own. ``c`` selects the system clipboard (as opposed to a
    primary/X11 selection).
    """
    payload = base64.standard_b64encode(text.encode("utf-8"))
    return b"\x1b]52;c;" + payload + b"\x1b\\"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k osc52 -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/vimol/kitty.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
kitty: add osc52_copy() for a clipboard-write escape sequence (VIM-15)

Plain OSC 52, no new dependency -- used by the viewer's upcoming 'y'
yank binding to copy the status bar's underlying text.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `elements.py` — light-mode color overrides

**Files:**
- Modify: `src/vimol/elements.py` (add near `_DEFAULT_COLOR`, after the `_COLORS` table)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `themed_base_colors(symbols: Sequence[str], base_colors: np.ndarray, theme: str) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
def test_themed_base_colors_dark_is_passthrough():
    from vimol import elements as els
    base = np.array([[1.0, 1.0, 1.0], [0.35, 0.35, 0.38]])
    out = els.themed_base_colors(["H", "C"], base, "dark")
    assert out is base   # no copy -- byte-identical to today


def test_themed_base_colors_light_overrides_unreadable_entries():
    from vimol import elements as els
    base = np.array([[1.0, 1.0, 1.0], [0.35, 0.35, 0.38]])   # H, C
    out = els.themed_base_colors(["H", "C"], base, "light")
    assert not np.allclose(out[0], [1.0, 1.0, 1.0])   # H moved off pure white
    assert np.allclose(out[1], [0.35, 0.35, 0.38])    # C untouched
    assert base[0].tolist() == [1.0, 1.0, 1.0]         # original array untouched


def test_themed_base_colors_light_covers_all_override_entries():
    from vimol import elements as els
    syms = ["H", "He", "Ag", "Pt", "Hg", "C"]
    base = np.array([els.element_color(s) for s in syms])
    out = els.themed_base_colors(syms, base, "light")
    for i, sym in enumerate(syms[:-1]):   # every override entry changed
        assert not np.allclose(out[i], base[i]), sym
    assert np.allclose(out[-1], base[-1])  # carbon, not in the override table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k themed_base_colors -v`
Expected: FAIL with `AttributeError: module 'vimol.elements' has no attribute 'themed_base_colors'`

- [ ] **Step 3: Implement**

Add to `src/vimol/elements.py`, right after the `_DEFAULT_COLOR = ...` line:

```python
# Light-mode overrides for CPK entries that don't survive a light/white
# terminal background -- only the near-white ones move; everything else in
# _COLORS is saturated enough to read on either background.
_LIGHT_OVERRIDES = {
    "H": (0.55, 0.55, 0.58), "He": (0.45, 0.75, 0.78), "Ag": (0.45, 0.45, 0.45),
    "Pt": (0.45, 0.45, 0.55), "Hg": (0.40, 0.40, 0.55),
}
```

And after `element_color()`:

```python
def themed_base_colors(symbols, base_colors, theme: str):
    """*base_colors* (N,3), with _LIGHT_OVERRIDES entries substituted where
    *symbols* names one and *theme* is "light". Returns *base_colors*
    unchanged (same array, no copy) for "dark" -- callers rely on this to
    stay byte-identical to pre-theme rendering when the theme is dark."""
    if theme != "light":
        return base_colors
    out = base_colors.copy()
    for i, sym in enumerate(symbols):
        override = _LIGHT_OVERRIDES.get(normalize_symbol(sym))
        if override is not None:
            out[i] = override
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k themed_base_colors -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/vimol/elements.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
elements: light-mode overrides for near-white CPK colors (VIM-14)

H/He/Ag/Pt/Hg are the only CPK entries that vanish against a white
terminal background. themed_base_colors() is a pure array transform,
a no-op for the dark theme -- widget.py wires it in next via the
existing style.color_override mechanism, no Molecule/StructureSet
changes needed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `parsers/xyz.py` — keep the full comment line

**Files:**
- Modify: `src/vimol/parsers/xyz.py:46`
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `parse()` no longer truncates `Molecule.name`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_vimol.py`, near `test_xyz_roundtrip_and_bonds`:

```python
def test_xyz_keeps_full_comment_past_60_chars():
    from vimol.parsers import xyz as xyz_parser
    long_comment = "SCF Energy = -76.123456789012 Hartree, converged in 42 cycles, RMS grad 1e-9"
    assert len(long_comment) > 60
    text = f"1\n{long_comment}\nO 0.0 0.0 0.0\n"
    mols = xyz_parser.parse(text)
    assert mols[0].name == long_comment
    # round-trips through dumps()/parse() unchanged
    again = xyz_parser.parse(xyz_parser.dumps(mols[0]))
    assert again[0].name == long_comment
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_vimol.py -k full_comment -v`
Expected: FAIL — `mols[0].name` is truncated to 60 chars.

- [ ] **Step 3: Implement**

In `src/vimol/parsers/xyz.py`, change line 46 from:

```python
            name=comment.strip()[:60],
```

to:

```python
            name=comment.strip(),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_vimol.py -k full_comment -v`
Expected: PASS

- [ ] **Step 5: Run the full xyz-adjacent tests to check nothing relied on the cap**

Run: `python3 -m pytest tests/test_vimol.py -k "xyz or c60 or pdb" -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add src/vimol/parsers/xyz.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
xyz: keep the full comment line on Molecule.name (VIM-15)

The 60-char parse-time cap silently dropped the tail of long
ORCA/Gaussian energy comments and shortened them further on every
save round-trip. Display call sites that need a bounded width
already truncate independently (_truncate_middle in viewer.py) --
nothing else needed the cap.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `widget.py` — wire the theme into `_apply_highlight`

**Files:**
- Modify: `src/vimol/widget.py:20-31` (imports), `src/vimol/widget.py:48-91` (`__init__`), `src/vimol/widget.py:686-720` (`_apply_highlight`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: `elements.themed_base_colors` (Task 4).
- Produces: `MoleculeWidget.theme: str` (default `"dark"`).

- [ ] **Step 1: Write the failing tests**

```python
def test_widget_defaults_to_dark_theme_and_is_passthrough():
    from vimol.widget import MoleculeWidget
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    w = MoleculeWidget(mol)
    assert w.theme == "dark"
    w.render()
    assert w.style.color_override is None   # no hover/theme override active


def test_widget_light_theme_sets_color_override_with_no_hover():
    from vimol.widget import MoleculeWidget
    mol = vimol.load(os.path.join(EX, "methane.xyz"))   # C + 4 H
    w = MoleculeWidget(mol)
    w.theme = "light"
    w.render()
    assert w.style.color_override is not None
    h_idx = mol.symbols.index("H")
    assert not np.allclose(w.style.color_override[h_idx], [1.0, 1.0, 1.0])


def test_widget_light_theme_hover_tint_applies_on_top_of_override():
    from vimol.widget import MoleculeWidget
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    w = MoleculeWidget(mol)
    w.theme = "light"
    h_idx = mol.symbols.index("H")
    w.hovered = h_idx
    w.render()
    cols = w.style.color_override
    assert cols is not None
    # hovered atom gets the yellow hover tint, not the plain light override
    assert cols[h_idx][0] > 0.5 and cols[h_idx][1] > 0.5   # warm/bright, not the muted grey override
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k "widget_defaults_to_dark or widget_light_theme" -v`
Expected: FAIL with `AttributeError: 'MoleculeWidget' object has no attribute 'theme'`

- [ ] **Step 3: Implement**

In `src/vimol/widget.py`, add the import (after `from . import editor`, line 31):

```python
from . import elements
```

In `__init__` (`widget.py:48`), add `self.theme = "dark"` — place it right after `self.editable = editable` (line 61), grouped with the other simple session-state flags:

```python
        self.editable = editable
        self.theme = "dark"                     # "dark" | "light"; Viewer keeps this in sync
```

Replace `_apply_highlight` (`widget.py:686-720`) in full:

```python
    def _apply_highlight(self) -> None:
        # widget._base_colors is the composite's base colors (design §3):
        # CPK for the first-drawn (active) entry, tint for the rest.
        composite = self.scene.structures.composite()
        self._base_colors = composite.base_colors
        themed = elements.themed_base_colors(composite.molecule.symbols,
                                             composite.base_colors, self.theme)
        hi = self.hovered if self.hovered is not None else self.selected
        if hi is None and not self.measure_sel:
            # themed is a no-op passthrough for "dark", so this stays
            # byte-identical to pre-theme rendering in the dark case.
            self.style.color_override = themed if self.theme == "light" else None
            return
        # hovered/selected/measure_sel are ACTIVE-LOCAL indices (design §3);
        # map them through the composite's offset before writing into the
        # composite-sized color array.
        active_index = self.scene.structures.active_index
        if not (composite.sources == active_index).any():
            # The active structure isn't drawn at all (hidden -- design §4.3
            # allows this without advancing active_index), so there is no
            # composite slot to map hovered/selected/measure_sel into.
            self.style.color_override = themed if self.theme == "light" else None
            return
        cols = themed.copy()
        yellow = np.array([1.0, 0.95, 0.3])
        # every picked atom in the live measurement selection gets the same
        # yellow tint as a hover -- hover (below) is applied on top, so it
        # still shows through even for an atom that is also selected.
        if self.measure_sel:
            g_sel = composite.globalize(active_index, np.asarray(self.measure_sel, dtype=np.int64))
            for gidx in g_sel:
                cols[gidx] = np.clip(cols[gidx] * 0.4 + yellow * 0.9, 0, 1)
        if hi is not None:
            # brighten + tint the highlighted atom: red in delete mode (a preview of
            # "this disappears if you click here"), yellow otherwise.
            tint = np.array([1.0, 0.2, 0.2]) if self.delete_mode else yellow
            ghi = int(composite.globalize(active_index, np.array([hi]))[0])
            cols[ghi] = np.clip(cols[ghi] * 0.4 + tint * 0.9, 0, 1)
        self.style.color_override = cols
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k "widget_defaults_to_dark or widget_light_theme" -v`
Expected: 3 passed

- [ ] **Step 5: Run the existing widget/highlight tests to check nothing regressed**

Run: `python3 -m pytest tests/test_vimol.py -k "hover or highlight or flat_atom or color_override" -v`
Expected: all passed (dark-theme default keeps every existing assertion byte-identical)

- [ ] **Step 6: Commit**

```bash
git add src/vimol/widget.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
widget: wire theme into _apply_highlight via themed_base_colors (VIM-14)

MoleculeWidget.theme defaults to "dark" (unaffected embedders, e.g.
examples/embed_demo.py, see no change). Light theme applies the
element-color overrides even with nothing hovered by seeding
color_override from themed_base_colors() instead of None; hover/
measure tints still apply on top exactly as before.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `viewer.py` — replace chrome color constants with `self.theme`, fix the strip background bug

**Files:**
- Modify: `src/vimol/viewer.py` (imports; module constants block; `_status_bar`, `_edit_buttons`, `_draw_help`, `_list_cap`, `_list_legend`, `_draw_list`, `_pt_cell_text`, `_draw_periodic_table`, `_draw_geometry_picker`, `Viewer.__init__`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: `theme.DARK` (Task 1).
- Produces: `Viewer.theme: theme.Theme` (defaults to `theme.DARK` in this task; live detection/`ctrl-t` land in Task 8).

This task is a mechanical color-constant-to-`self.theme.field` refactor plus the strip background fix — no behavior change under the dark theme other than the strip painting a real (if visually near-identical) background on every row instead of none.

- [ ] **Step 1: Write the failing tests**

```python
def test_viewer_defaults_to_dark_theme(monkeypatch):
    from vimol.viewer import Viewer, theme as viewer_theme
    # Task 7 hardcodes theme.DARK, but Task 8 makes this env-dependent --
    # clear both now so this test stays valid (not flaky under whatever the
    # running shell happens to export) once that lands.
    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    assert v.theme is viewer_theme.DARK


def test_viewer_strip_rows_always_carry_an_explicit_background(tmp_path):
    """Every structure-strip row -- not just active/cursor -- must paint a
    background SGR, else its foreground (tuned for one theme) washes out
    against whichever background the terminal itself has (the reported
    bug)."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 24
        data = v._draw_list()
        text = data.decode("utf-8", "replace")
        # every row must set an explicit 24-bit background, not rely on the
        # terminal's own -- count opening bg SGRs against the row count.
        assert text.count("\x1b[48;2;") >= 6   # header + >=3 structures + legend rows
    finally:
        os.close(fd)


def test_viewer_status_bar_uses_theme_panel_colors():
    from vimol.viewer import Viewer
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    v._update_geometry()
    bar = v._status_bar()
    r, g, b = v.theme.panel_bg
    assert f"\x1b[48;2;{r};{g};{b}m" in bar
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k "defaults_to_dark_theme or strip_rows_always or panel_colors" -v`
Expected: FAIL — `Viewer` has no `.theme` attribute yet, and the strip test likely already fails the `>= 6` count today (only active/cursor rows carry a background).

- [ ] **Step 3: Implement — imports and module constants**

In `src/vimol/viewer.py`, add the import (with the other `from . import` lines, after `from . import periodic_table`, line 28):

```python
from . import theme
```

Delete the "Structure-list strip colors" comment block and its eight constants (`viewer.py:65-76`) — from:

```python
# Structure-list strip colors (design §4.1).
# Structure-strip palette (design §4.1). A dark, calm panel: the active row
# is a full-width background rather than a leader glyph, and the cursor row a
# subtler one so the two stay distinguishable when they differ.
_LIST_HEADER_FG = (139, 146, 165)   # panel header and group (file) names
_LIST_MUTED_FG = (110, 118, 135)    # row index, legend descriptions
_LIST_LABEL_FG = (232, 236, 244)    # the active row's label; key-cap text
_LIST_DIM_FG = (200, 206, 216)      # every other row's label
_LIST_ACTIVE_BG = (37, 45, 64)      # active-row highlight, full panel width
_LIST_CURSOR_BG = (28, 33, 46)      # cursor-row highlight (subtler)
_LIST_RULE_FG = (60, 66, 84)        # the separator rule
_LIST_CAP_BG = (42, 49, 66)         # legend "key cap" background
```

to just:

```python
# Structure-list strip layout (design §4.1) -- colors live in theme.py now.
```

Delete `_CLEANUP_HINT_FG` and its comment (`viewer.py:175-176`):

```python
# Warm warning color for the status bar's "press c to cleanup" hint.
_CLEANUP_HINT_FG = (255, 170, 60)
```

→ delete both lines entirely.

Delete the periodic-table color constants and their comment (`viewer.py:178-183`):

```python
# Periodic-table picker panel colors.
_PT_BG = (18, 20, 26)
_PT_BORDER_FG = (60, 200, 180)      # teal accent, matches the geometry pill
_PT_TEXT_FG = (220, 220, 230)
_PT_DIM_FG = (110, 114, 126)
_PT_GAP_BG = (40, 42, 50)
```

→ delete entirely (all five constants + comment).

Everything else in that constants block (`_STATUS_ZONE_ROWS`, `_GEOM_HINT`, `_LEFT_WIDTH`, `_LIST_W_MIN`/`_LIST_W_MAX`, `_LIST_ROWS_ABOVE`/`_LIST_ROWS_BELOW`/`_LIST_WHEEL_STEP`, the timing/pacing constants, `_STARTUP_SCALE`) is layout/timing, not color — leave unchanged.

- [ ] **Step 4: Implement — `Viewer.__init__`**

Add `self.theme = theme.DARK` in `__init__` (`viewer.py:186`) — place it right after `self.editable = editable` (line 195):

```python
        self.source_path = source_path
        self.editable = editable
        self.theme = theme.DARK   # live detection lands in Task 8; ctrl-t overrides it
```

And sync it onto the widget (which now has its own `.theme`, Task 6) right after `self.widget = MoleculeWidget(...)` is constructed (`viewer.py:218-220`):

```python
        self.widget = MoleculeWidget(self.structures, 320, 240, style=self.style,
                                     supersample=1, picking=picking, backend=backend,
                                     editable=editable)
        self.widget.theme = self.theme.name
```

- [ ] **Step 5: Implement — `_list_cap` and `_list_legend`**

Replace `_list_cap` (`viewer.py:692-696`) — it must become an instance method (it now needs `self.theme`), dropping `@classmethod`:

```python
    def _list_cap(self, key: str):
        """A legend "key cap": the key text padded one space either side on a
        lighter background."""
        return (f" {key} ", self._sgr_bg(self.theme.list_cap_bg) + self._sgr_fg(self.theme.list_label_fg))
```

Replace the `muted = ...` line inside `_list_legend` (`viewer.py:705`):

```python
        muted = self._sgr_fg(self.theme.list_muted_fg)
```

(the rest of `_list_legend`, lines 702-711, is unchanged — `cap = self._list_cap` still works, bound methods don't care whether the underlying def used to be a classmethod).

- [ ] **Step 6: Implement — `_draw_list`**

Replace the header/blank-row block (`viewer.py:753-760`):

```python
        muted = self._sgr_fg(self.theme.list_muted_fg)
        head_fg = self._sgr_fg(self.theme.list_header_fg)
        marker = ("↑" if first > 0 else "") + ("↓" if first + cap < len(rows) else "")
        title = f" STRUCTURES {len(sset)}"
        gap = max(0, list_w - len(title) - len(marker))
        put(0, self._list_line([(title, head_fg), (" " * gap, ""),
                                (marker, "\x1b[2m" + head_fg)], list_w, bg=self.theme.list_panel_bg))
        put(1, self._list_line([], list_w, bg=self.theme.list_panel_bg))   # air under the header
```

Replace the group-header branch (`viewer.py:767-772`):

```python
            if kind == "group":
                name = self._truncate_middle(text, max(1, list_w - 1))
                if not put(row0, self._list_line([(" ", ""), (name, head_fg)], list_w,
                                                 bg=self.theme.list_panel_bg)):
                    break
                drawn_rows += 1
                continue
```

Replace the per-entry-row block (`viewer.py:773-802`):

```python
            entry = sset[i]
            tint = tuple(int(max(0.0, min(1.0, c)) * 255) for c in entry.tint)
            active = i == sset.active_index
            # The active row IS its background (no leader glyph); the cursor
            # row gets a subtler one, so the two stay tellable apart when
            # they differ (design §4.3). Every OTHER row now gets the
            # theme's own panel background too (design bug fix, docs/design/
            # theme-and-aesthetics.md §3): leaving it unset meant those rows
            # sat on the terminal's own background with foreground colors
            # tuned only for a dark one. A row that is IN THE OVERLAY wears
            # its own tint on the label -- with no leader glyph and no key
            # binding for it, that tint is the only way to read the overlay
            # set off the screen at all (membership is opt+click only).
            bg = (self.theme.list_active_bg if active
                  else self.theme.list_cursor_bg if i == self._list_cursor
                  else self.theme.list_panel_bg)
            dim = "\x1b[2m" if not entry.visible else ""
            # The tint outranks the active row's near-white label:
            # opt+clicking the active row has to change something on screen,
            # and the background is already saying which row is active.
            label_fg = (self._sgr_fg(tint) if entry.marked
                        else self._sgr_fg(self.theme.list_label_fg if active else self.theme.list_dim_fg))
            segs = [
                (" ", ""),
                ("█" if entry.visible else "░", dim + self._sgr_fg(tint)),
                (" ", ""),
                (f"{i + 1:>{idx_w}}", dim + muted),
                (" ", ""),
                (self._truncate_middle(text, label_w), dim + label_fg),
            ]
            if not put(row0, self._list_line(segs, list_w, bg=bg)):
                break
            drawn_rows += 1
            self._list_row_spans.append((row0, 0, list_w))
            self._list_row_struct.append(i)
```

Replace the rule/legend/status-lines tail (`viewer.py:804-819`):

```python
        row0 = _LIST_ROWS_ABOVE + drawn_rows
        rule = "─" * max(0, list_w - 2)
        put(row0, self._list_line([(" ", ""), (rule, self._sgr_fg(self.theme.list_rule_fg))], list_w,
                                  bg=self.theme.list_panel_bg))
        for k, segs in enumerate(self._list_legend(), start=1):
            put(row0 + k, self._list_line(segs, list_w, bg=self.theme.list_panel_bg))
        # Status lines last: on a short panel they are the first thing to
        # fall off the bottom (put() simply refuses), the legend the last.
        row0 += 1 + len(self._list_legend())
        if sset.overlay:
            drawn = sset.drawn_indices()
            membership = "+".join(str(i + 1) for i in drawn)
            aligned = any(not sset[i].transform.is_identity for i in drawn)
            status = f" overlay {membership}" + (" · aligned" if aligned else "")
            put(row0, self._list_line([(status, muted)], list_w, bg=self.theme.list_panel_bg))
            row0 += 1
        put(row0, self._list_line([(" camera shared", muted)], list_w, bg=self.theme.list_panel_bg))
        return bytes(out)
```

- [ ] **Step 7: Implement — `_draw_help`**

Replace (`viewer.py:1088-1095`):

```python
    def _draw_help(self):
        out = bytearray()
        bg_r, bg_g, bg_b = self.theme.help_bg
        fg_r, fg_g, fg_b = self.theme.help_fg
        sgr = b"\x1b[48;2;%d;%d;%dm\x1b[38;2;%d;%d;%dm" % (bg_r, bg_g, bg_b, fg_r, fg_g, fg_b)
        for k, line in enumerate(_help_lines(self.editable)):
            out += b"\x1b[%d;3H" % (2 + k)
            out += sgr
            out += (" " + line.ljust(58)).encode()
            out += b"\x1b[0m"
        kitty.write_bytes(bytes(out), self.fd_out)
```

- [ ] **Step 8: Implement — periodic-table picker**

Replace `_pt_cell_text` (`viewer.py:1134-1151`) — drop `@staticmethod`, it now needs `self.theme`:

```python
    def _pt_cell_text(self, cell, cursor: bool) -> str:
        """The 4-char escaped label for one periodic-table cell."""
        if cell is None:
            return "    "
        if cell.symbol is None:
            bg, fg = self.theme.pt_gap_bg, self.theme.pt_dim_fg
            label = f"{cell.text:^4}"
        else:
            rgb = elements.element_color(cell.symbol)
            bg = tuple(int(v * 255) for v in rgb)
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            fg = (10, 12, 14) if lum > 140 else (245, 246, 250)
            label = f"{cell.symbol:^4}"
        seg = f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m"
        if cursor:
            return f"{seg}\x1b[1m\x1b[7m{label}\x1b[27m\x1b[22m\x1b[0m"
        return f"{seg}{label}\x1b[0m"
```

Replace the border/bg_only/text_fg setup lines in `_draw_periodic_table` (`viewer.py:1156-1159`):

```python
        border = (f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
                  f"\x1b[38;2;{self.theme.pt_border_fg[0]};{self.theme.pt_border_fg[1]};{self.theme.pt_border_fg[2]}m")
        bg_only = f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
        text_fg = f"\x1b[38;2;{self.theme.pt_text_fg[0]};{self.theme.pt_text_fg[1]};{self.theme.pt_text_fg[2]}m"
```

(everything else in `_draw_periodic_table`, lines 1160-1188, is unchanged — it only reads the local `border`/`bg_only`/`text_fg` variables just redefined).

- [ ] **Step 9: Implement — geometry picker**

Replace the equivalent three lines in `_draw_geometry_picker` (`viewer.py:1257-1260`):

```python
        border = (f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
                  f"\x1b[38;2;{self.theme.pt_border_fg[0]};{self.theme.pt_border_fg[1]};{self.theme.pt_border_fg[2]}m")
        bg_only = f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
        text_fg = f"\x1b[38;2;{self.theme.pt_text_fg[0]};{self.theme.pt_text_fg[1]};{self.theme.pt_text_fg[2]}m"
```

(the rest of `_draw_geometry_picker` is unchanged.)

- [ ] **Step 10: Implement — `_edit_buttons` and `_status_bar`**

In `_edit_buttons` (`viewer.py:1473`), replace:

```python
        text = f"\x1b[38;2;150;155;170m{prefix}\x1b[0m{elem_btn} {geom_btn}"
```

with:

```python
        r, g, b = self.theme.edit_prefix_fg
        text = f"\x1b[38;2;{r};{g};{b}m{prefix}\x1b[0m{elem_btn} {geom_btn}"
```

In `_status_bar` (`viewer.py:1501-1510`), replace the three mode-specific early returns:

```python
        if self._mode == "save_input":
            body = f" Save to: {self._input_buf}█   Enter save · Esc cancel "
            bg_r, bg_g, bg_b = self.theme.input_bg
            fg_r, fg_g, fg_b = self.theme.input_fg
            return f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m\x1b[38;2;{fg_r};{fg_g};{fg_b}m{body}\x1b[0m"
        if self._mode == "save_confirm":
            name = os.path.basename(self._input_buf.strip())
            body = f" {name} exists — replace? (y/n) "
            bg_r, bg_g, bg_b = self.theme.warn_bg
            fg_r, fg_g, fg_b = self.theme.warn_fg
            return f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m\x1b[38;2;{fg_r};{fg_g};{fg_b}m{body}\x1b[0m"
        if self._mode == "quit_confirm":
            body = " unsaved changes — save before quitting? (y/n/Esc) "
            bg_r, bg_g, bg_b = self.theme.warn_bg
            fg_r, fg_g, fg_b = self.theme.warn_fg
            return f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m\x1b[38;2;{fg_r};{fg_g};{fg_b}m{body}\x1b[0m"
```

Replace the `base = ...` line (`viewer.py:1529`):

```python
        pr, pg, pb = self.theme.panel_bg
        fr, fg2, fb = self.theme.panel_fg
        base = f"\x1b[48;2;{pr};{pg};{pb}m\x1b[38;2;{fr};{fg2};{fb}m"
```

(`fg2` avoids shadowing the outer `fg` used a few lines later for `_geom_geometry`-adjacent code — check the surrounding function for any other local named `fg`; there is none in `_status_bar`, so `fg2` is only there to avoid confusion with the tuple-unpacked `g` channel name — rename freely, just keep it out of collision with `rep`/`spin`/`base`/`mod` etc. already in scope.)

Replace the cleanup-hint block (`viewer.py:1554-1559`):

```python
        cleanup_hint = ""
        if self.editable:
            clash, stretched = editor.cleanup_targets(mol)
            if clash or stretched:
                r, g, b = self.theme.cleanup_hint_fg
                cleanup_hint = f"  \x1b[38;2;{r};{g};{b}m\x1b[1m⚠ c cleanup\x1b[22m{base}"
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k "defaults_to_dark_theme or strip_rows_always or panel_colors" -v`
Expected: 3 passed

- [ ] **Step 12: Run the FULL suite to check nothing regressed**

Run: `python3 -m pytest -v`
Expected: all passed, with no test edits needed. Every existing strip
assertion (`tests/test_vimol.py`'s `test_viewer_list_*` tests) checks for a
*specific literal* color tuple — e.g. `_sgr_bg((37, 45, 64)) in active` /
`not in cursor` — and every one of those literals is `theme.DARK`'s value
for the corresponding field, unchanged by this task. None of them assert
the generic absence of *any* background SGR on a non-active row, so adding
`self.theme.list_panel_bg` there doesn't trip anything. (Verified by reading
every `test_viewer_list_*` test in `tests/test_vimol.py` before writing this
plan — if a failure shows up here anyway, stop and re-examine rather than
assuming this note is still accurate.)

- [ ] **Step 13: Commit**

```bash
git add src/vimol/viewer.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
viewer: read chrome colors from theme.py, fix strip background bug (VIM-13, VIM-14)

Every _LIST_*/_PT_*/_CLEANUP_HINT_FG module constant and inline
status-bar/help-panel color is now self.theme.<field> instead of a
hardcoded tuple -- Viewer.theme defaults to theme.DARK (detection and
ctrl-t land next), so this is behavior-neutral except for one real
fix: every structure-strip row now paints an explicit background
(self.theme.list_panel_bg) instead of only the active/cursor rows,
which is what made non-active rows unreadable on a light terminal.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `viewer.py` — theme detection wiring + `ctrl-t`

**Files:**
- Modify: `src/vimol/viewer.py` (`Viewer.__init__`, `_finish_startup`, `_driver_key`, `_BASE_DRIVER_KEYS`, `_help_lines`/`_HELP_TAIL`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: `theme.resolve`, `kitty.TerminalProbe.bg_rgb` (Tasks 1, 2).
- Produces: live theme resolution at startup; `ctrl-t` toggle.

- [ ] **Step 1: Write the failing tests**

```python
def test_viewer_ctrl_t_toggles_theme(monkeypatch):
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    v._update_geometry()
    assert v.theme is vimol_theme.DARK
    assert v.widget.theme == "dark"
    assert v._dispatch([KeyEvent("\x14")]) is True
    assert v.theme is vimol_theme.LIGHT
    assert v.widget.theme == "light"
    assert v._dispatch([KeyEvent("\x14")]) is True
    assert v.theme is vimol_theme.DARK
    assert v.widget.theme == "dark"


def test_viewer_ctrl_t_available_read_only(monkeypatch):
    """Like d/g/t/n/p/m, ctrl-t is a base driver key -- available whether
    or not editing is enabled."""
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY), editable=False)
    assert v.theme is vimol_theme.DARK
    assert v._dispatch([KeyEvent("\x14")]) is True
    assert v.theme is vimol_theme.LIGHT


def test_viewer_frame0_theme_guess_uses_colorfgbg_before_probe(monkeypatch):
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)   # explicit would outrank COLORFGBG
    monkeypatch.setenv("COLORFGBG", "0;15")
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    assert v.theme is vimol_theme.LIGHT   # guessed synchronously, no probe run yet


def test_viewer_finish_startup_upgrades_theme_from_osc11(monkeypatch):
    from vimol import theme as vimol_theme, kitty as vimol_kitty
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    probe = vimol_kitty.TerminalProbe(graphics=True, pixel_mouse=False, cell_px=(9.0, 18.0),
                                      rtt=0.001, bg_rgb=(245, 245, 245))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY), probe=probe)
    assert v.theme is vimol_theme.DARK   # frame-0 guess: no COLORFGBG, default dark
    v._old_termios = object()            # _finish_startup's "probe already available" path
    v._finish_startup()
    assert v.theme is vimol_theme.LIGHT  # OSC 11 corrected it
    assert v.widget.theme == "light"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k "ctrl_t or frame0_theme or finish_startup_upgrades" -v`
Expected: FAIL — `\x14` isn't a driver key yet, and `_finish_startup` never touches `self.theme`.

- [ ] **Step 3: Implement — `_BASE_DRIVER_KEYS`**

In `src/vimol/viewer.py`, extend the set (`viewer.py:121-122`):

```python
_BASE_DRIVER_KEYS = {"q", "escape", "a", "?", "d", "g", "t", "n", "p", "m", "y", "\x03", "\x14",
                      "alt+up", "alt+down"}
```

(`y` is added here too, ahead of Task 9, since both are base-level toggles being added to the same set — Task 9 implements what `y` actually does; leaving it unclaimed for one task would make `_driver_key`'s `else: return False` swallow it silently until Task 9, which is fine either way, but claiming it now avoids a second edit to this line.)

- [ ] **Step 4: Implement — frame-0 synchronous guess**

In `Viewer.__init__`, change the line added in Task 7:

```python
        self.theme = theme.DARK   # live detection lands in Task 8; ctrl-t overrides it
```

to:

```python
        # Frame 0 draws before the startup probe's OSC 11 reply can possibly
        # be in hand (the probe itself runs after the first paint -- see
        # _finish_startup), so this is a synchronous best guess: COLORFGBG or
        # DARK. _finish_startup upgrades it once the probe replies.
        self.theme = theme.resolve(os.environ.get("VIMOL_THEME"), None,
                                   os.environ.get("COLORFGBG"))
```

- [ ] **Step 5: Implement — upgrade in `_finish_startup`**

In `_finish_startup` (`viewer.py:395`), inside the `if probe is not None:` branch, add the theme upgrade right after the existing `if probe.cell_px is not None: ...` block (`viewer.py:416-417`):

```python
        if probe is not None:
            if probe.cell_px is not None:
                self._cell_px = probe.cell_px
            resolved = theme.resolve(os.environ.get("VIMOL_THEME"), probe.bg_rgb,
                                     os.environ.get("COLORFGBG"))
            if resolved is not self.theme:
                self.theme = resolved
                self.widget.theme = resolved.name
            self._enable_mouse(probe.pixel_mouse)
```

- [ ] **Step 6: Implement — `ctrl-t` in `_driver_key`**

In `_driver_key` (`viewer.py:1775`), add a branch — place it next to the existing `t` (transparent) handling (`viewer.py:1829-1832`):

```python
        elif key == "t":
            self.style.transparent = not self.style.transparent
            kitty.write_bytes(_CLEAR, self.fd_out)
            self._last_interact = time.time()
        elif key == "\x14":
            self.theme = theme.LIGHT if self.theme is theme.DARK else theme.DARK
            self.widget.theme = self.theme.name
            kitty.write_bytes(_CLEAR, self.fd_out)
            self._last_interact = time.time()
```

- [ ] **Step 7: Implement — help text**

In `_HELP_TAIL` (`viewer.py:108-113`), add a line next to the existing `t` entry:

```python
_HELP_TAIL = [
    "  n / p / opt+up/dn .. next/prev frame   d .................. depth cue",
    "  t .................. transparent bg    g .................. hi-quality",
    "  ctrl-t ............. light/dark theme  y .................. yank status text",
    "  f / r / z .......... re-fit / reset    ? .................. toggle help",
    "  q / Esc ............ quit",
]
```

(the `y` entry here is a placeholder label for Task 9's binding — claiming the help line now keeps this task and Task 9 from touching the same list twice; if Task 9 changes the wording that's expected.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k "ctrl_t or frame0_theme or finish_startup_upgrades" -v`
Expected: 4 passed

- [ ] **Step 9: Run the full suite**

Run: `python3 -m pytest -v`
Expected: all passed

- [ ] **Step 10: Commit**

```bash
git add src/vimol/viewer.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
viewer: live theme detection + ctrl-t override (VIM-13)

Frame 0 guesses dark/COLORFGBG synchronously (the startup probe runs
after the first paint, so its OSC 11 reply isn't in hand yet);
_finish_startup upgrades the guess once the probe replies, via the
same explicit -> OSC11 -> COLORFGBG -> dark ladder theme.resolve()
centralizes. ctrl-t (\x14) flips DARK/LIGHT for the session, synced
onto widget.theme, following the same clear-and-redraw pattern the
existing 't' transparency toggle already uses.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `viewer.py` — `y` yanks the status bar's left field via OSC 52

**Files:**
- Modify: `src/vimol/viewer.py` (`_driver_key`, `_HELP_TAIL` wording)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: `kitty.osc52_copy` (Task 3).
- Produces: `y` key behavior.

`_status_bar` (`viewer.py:1498`) already computes the exact string to copy as `raw_left` — see `viewer.py:1523-1524`:

```python
        raw_left = measure or self.widget.pick_refusal or hov or (self._msg or
            f"{mol.name or 'molecule'}  {mol.formula()}  {mol.n_atoms} atoms")
```

`_driver_key` doesn't have easy access to that computation without duplicating it (it's built from several `self.widget`/`self.style` reads local to `_status_bar`). Rather than duplicate the fallback chain, factor it out into its own method first so both `_status_bar` and the new `y` handler call the same code.

- [ ] **Step 1: Write the failing tests**

```python
def test_viewer_left_field_text_matches_status_bar_computation():
    from vimol.viewer import Viewer
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    v._update_geometry()
    assert v._left_field_text() == f"{mol.name or 'molecule'}  {mol.formula()}  {mol.n_atoms} atoms"


def test_viewer_y_key_copies_left_field_via_osc52(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent
    import base64

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    mol.name = "SCF Energy = -40.5183 Hartree"
    out = tmp_path / "out.bin"
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._update_geometry()
        assert v._dispatch([KeyEvent("y")]) is True
    finally:
        os.close(fd)
    data = out.read_bytes()
    expected_payload = base64.standard_b64encode(v._left_field_text().encode("utf-8"))
    assert (b"\x1b]52;c;" + expected_payload + b"\x1b\\") in data
    assert v._msg == "copied"


def test_viewer_y_key_copies_full_untruncated_comment_not_display_clipped(tmp_path):
    """The status bar's visible field is width-clipped; yank must copy the
    real underlying string regardless of terminal width."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent
    import base64

    long_comment = "SCF Energy = -76.123456789012 Hartree, converged in 42 cycles, RMS grad 1e-9"
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    mol.name = long_comment
    out = tmp_path / "out.bin"
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._cols, v._rows = 40, 24   # narrow enough that the display field clips
        assert v._dispatch([KeyEvent("y")]) is True
    finally:
        os.close(fd)
    data = out.read_bytes()
    expected_payload = base64.standard_b64encode(f"{long_comment}  {mol.formula()}  {mol.n_atoms} atoms".encode("utf-8"))
    assert (b"\x1b]52;c;" + expected_payload + b"\x1b\\") in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k "left_field_text or y_key" -v`
Expected: FAIL — `_left_field_text` doesn't exist, `y` isn't handled.

- [ ] **Step 3: Implement — factor out `_left_field_text`**

In `_status_bar` (`viewer.py:1511-1524`), replace:

```python
        mol = self.widget.molecule
        hov = self.widget.atom_info(self.widget.hovered)
        # a live measurement readout (2+ picks in measure mode) outranks the
        # hover text; with 0-1 picks measurement() is "" and the normal
        # left-segment behavior applies.
        measure = (editor.measurement(mol, self.widget.measure_sel)
                   if self.widget.measure_mode else "")
        # The molecule "name" is the xyz file's comment line (parsers/xyz.py
        # keeps it at 60 chars, which stays the real ceiling here). It gets
        # no cap of its own: the left field's width below is what bounds it,
        # so a wide terminal actually shows the comment instead of clipping
        # it to a stub while the middle of the bar sits empty.
        raw_left = measure or self.widget.pick_refusal or hov or (self._msg or
            f"{mol.name or 'molecule'}  {mol.formula()}  {mol.n_atoms} atoms")
```

with:

```python
        mol = self.widget.molecule
        raw_left = self._left_field_text()
```

Add the new method just above `_status_bar` (right after `_build_segment`, before `def _status_bar`):

```python
    def _left_field_text(self) -> str:
        """The status bar's left-field text: live measurement readout, a
        pick refusal, hover atom info, a transient message, or the
        molecule's name/formula/atom-count summary -- whichever applies,
        highest-priority first. This is the exact (untruncated) string 'y'
        yanks (Task 9) and _status_bar (below) clips/pads for display."""
        mol = self.widget.molecule
        hov = self.widget.atom_info(self.widget.hovered)
        # a live measurement readout (2+ picks in measure mode) outranks the
        # hover text; with 0-1 picks measurement() is "" and the normal
        # left-segment behavior applies.
        measure = (editor.measurement(mol, self.widget.measure_sel)
                   if self.widget.measure_mode else "")
        # The molecule "name" is the xyz file's comment line, kept in full
        # (parsers/xyz.py no longer truncates it -- see VIM-15). It gets no
        # cap of its own here: the left field's DISPLAY width is what bounds
        # what's drawn, but 'y' yanks this untruncated string regardless.
        return measure or self.widget.pick_refusal or hov or (self._msg or
            f"{mol.name or 'molecule'}  {mol.formula()}  {mol.n_atoms} atoms")
```

- [ ] **Step 4: Implement — `y` in `_driver_key`**

Add a branch in `_driver_key` (`viewer.py:1775`), next to the existing `m` measure-mode handling so related read-only-safe bindings stay grouped:

```python
        elif key == "y":
            kitty.write_bytes(kitty.osc52_copy(self._left_field_text()), self.fd_out)
            self._msg = "copied"
```

- [ ] **Step 5: Implement — help text wording**

`_HELP_TAIL`'s `y` entry was already added as a placeholder in Task 8, Step 7 — no change needed here; confirm it reads "y .................. yank status text" (it does).

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k "left_field_text or y_key" -v`
Expected: 3 passed

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest -v`
Expected: all passed — in particular re-check `test_viewer_status_bar_shows_pick_refusal_message` (Task 7 didn't touch this, but the `_left_field_text` factor-out did) still passes unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/vimol/viewer.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
viewer: 'y' yanks the status bar's left field via OSC 52 (VIM-15)

vimol's own SGR mouse reporting blocks native terminal drag-select
over the status bar, and the visible text is already width-clipped
for display -- so 'y' copies the underlying untruncated string (live
measurement / pick refusal / hover info / molecule name+formula,
whichever _status_bar is currently showing) straight to the system
clipboard. _left_field_text() factors that selection logic out of
_status_bar so both it and the new binding share one source of truth.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `app.py` — `--theme` CLI flag

**Files:**
- Modify: `src/vimol/app.py` (`make_parser`, `main`)
- Test: `tests/test_vimol.py`

**Interfaces:**
- Consumes: `theme.resolve` (Task 1) is already reachable inside `Viewer.__init__` via `VIMOL_THEME`; this task adds the CLI flag as the higher-precedence explicit source by setting `VIMOL_THEME` from it before constructing the `Viewer` — no `Viewer.__init__` signature change needed.
- Produces: `--theme {dark,light,auto}` flag.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_theme_flag_sets_env_before_viewer_construction(monkeypatch):
    from vimol import app
    monkeypatch.delenv("VIMOL_THEME", raising=False)
    p = app.make_parser()
    args = p.parse_args(["--theme", "light", os.path.join(EX, "methane.xyz")])
    assert args.theme == "light"
    app._apply_theme_arg(args)
    assert os.environ["VIMOL_THEME"] == "light"


def test_cli_theme_flag_auto_clears_env():
    from vimol import app
    os.environ["VIMOL_THEME"] = "light"
    try:
        p = app.make_parser()
        args = p.parse_args(["--theme", "auto"])
        app._apply_theme_arg(args)
        assert "VIMOL_THEME" not in os.environ
    finally:
        os.environ.pop("VIMOL_THEME", None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vimol.py -k cli_theme -v`
Expected: FAIL — no `--theme` argument, no `_apply_theme_arg`.

- [ ] **Step 3: Implement**

In `src/vimol/app.py`, add the argument in `make_parser` (`app.py:61-93`), next to `--backend`:

```python
    p.add_argument("--theme", default="auto", choices=["auto", "dark", "light"],
                   help="chrome color theme; auto detects the terminal's own background")
```

Add a small helper right after `make_parser`:

```python
def _apply_theme_arg(args) -> None:
    """--theme is the highest-precedence source in theme.resolve()'s ladder
    (see docs/design/theme-and-aesthetics.md sec 2) -- implemented by setting
    VIMOL_THEME before the Viewer is constructed, rather than adding a
    parallel parameter to Viewer.__init__ for what VIMOL_THEME already
    covers. "auto" means "let detection decide", i.e. clear any override."""
    if args.theme == "auto":
        os.environ.pop("VIMOL_THEME", None)
    else:
        os.environ["VIMOL_THEME"] = args.theme
```

In `main` (`app.py:143`), call it right after `args = make_parser().parse_args(argv)`:

```python
    args = make_parser().parse_args(argv)
    _apply_theme_arg(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_vimol.py -k cli_theme -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add src/vimol/app.py tests/test_vimol.py
git commit -m "$(cat <<'EOF'
app: add --theme {auto,dark,light} CLI flag (VIM-13)

Implemented by setting VIMOL_THEME before constructing the Viewer --
that env var is already the second-highest rung of theme.resolve()'s
ladder, so this needed no new Viewer.__init__ parameter. "auto"
clears any override and lets OSC 11/COLORFGBG detection decide.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `python3 -m pytest -v`
Expected: all tests pass, zero failures/errors/skips introduced by this work.

- [ ] **Step 2: Grep for any leftover references to the deleted module constants**

Run: `grep -rn "_LIST_HEADER_FG\|_LIST_MUTED_FG\|_LIST_LABEL_FG\|_LIST_DIM_FG\|_LIST_ACTIVE_BG\|_LIST_CURSOR_BG\|_LIST_RULE_FG\|_LIST_CAP_BG\|_CLEANUP_HINT_FG\|_PT_BG\|_PT_BORDER_FG\|_PT_TEXT_FG\|_PT_DIM_FG\|_PT_GAP_BG" src/vimol/`
Expected: no output (every reference now goes through `self.theme`).

- [ ] **Step 3: Manual sanity check, if a Kitty-protocol terminal is available**

```bash
python3 -m vimol examples/c60.xyz
```
Press `ctrl-t` a couple of times and confirm the status bar / structure strip (open a second file to see it, e.g. `python3 -m vimol examples/c60.xyz examples/methane.xyz` if multi-file loading is supported by the installed version, otherwise any single file still shows the status bar) visibly swap between the dark and light palettes without leftover mis-colored cells. Press `y` and paste into another application to confirm the clipboard received the status bar's text. This step is exploratory, not a pass/fail gate — note in the final report whether a suitable terminal was available to check it, since headless/CI environments can't.

- [ ] **Step 4: Update Linear**

Move VIM-13, VIM-14, VIM-15 to **Requires Approval** (per this repo's Linear workflow — set right before code review/commit, which by now has already happened per-task; treat this as catching up their state at the end of the batch) and leave VIM-12 in **Requires Approval** as it already is.
