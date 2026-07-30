# Theme system, color sanity sweep, copyable status bar

**Issue:** VIM-12 · **Date:** 2026-07-30 · **Status:** Approved
**Blocks:** VIM-13 (theme module), VIM-14 (color sweep), VIM-15 (copyable status bar)

## Summary

Add one module, `src/vimol/theme.py`, holding every chrome color the
interactive viewer currently hardcodes as module-level constants in
`viewer.py` (`_LIST_*`, `_PT_*`, `_CLEANUP_HINT_FG`, the inline status-bar/
help-panel colors). Two `Theme` instances, `DARK` and `LIGHT`; `Viewer` gets
`self.theme` and every `\x1b[38;2;…`/`\x1b[48;2;…` chrome callsite reads from
it. The active theme is auto-detected via an OSC 11 terminal-background
query folded into the existing combined startup probe, with `ctrl-t` as a
session-only override.

Alongside that: a handful of CPK element colors that are unreadable against
a light terminal get a light-mode override, and a genuine bug — the
structure-list strip's non-active rows currently paint no background at all
— gets fixed as part of the same sweep, since it's the same "text tuned for
one background" problem.

Separately, the status bar's left field (hover info, molecule name, live
measurement, or the xyz comment line — often an energy value) is currently
impossible to copy, because vimol owns SGR mouse reporting and intercepts
terminal drag-select. A `y` key copies it via OSC 52.

## 1. `theme.py`

```python
@dataclass(frozen=True)
class Theme:
    name: str                                    # "dark" | "light"
    panel_bg: Tuple[int, int, int]                # status bar / normal mode
    panel_fg: Tuple[int, int, int]
    edit_prefix_fg: Tuple[int, int, int]           # "adding " label in _edit_buttons
    warn_bg: Tuple[int, int, int]                  # save_confirm / quit_confirm
    warn_fg: Tuple[int, int, int]
    input_bg: Tuple[int, int, int]                 # save_input
    input_fg: Tuple[int, int, int]
    help_bg: Tuple[int, int, int]
    help_fg: Tuple[int, int, int]
    list_header_fg: Tuple[int, int, int]
    list_muted_fg: Tuple[int, int, int]
    list_label_fg: Tuple[int, int, int]
    list_dim_fg: Tuple[int, int, int]
    list_active_bg: Tuple[int, int, int]
    list_cursor_bg: Tuple[int, int, int]
    list_rule_fg: Tuple[int, int, int]
    list_cap_bg: Tuple[int, int, int]
    list_panel_bg: Tuple[int, int, int]            # NEW FIELD, see §3
    cleanup_hint_fg: Tuple[int, int, int]
    pt_bg: Tuple[int, int, int]
    pt_border_fg: Tuple[int, int, int]
    pt_text_fg: Tuple[int, int, int]
    pt_dim_fg: Tuple[int, int, int]
    pt_gap_bg: Tuple[int, int, int]

DARK = Theme(name="dark", panel_bg=(30, 33, 44), panel_fg=(230, 232, 240),
             edit_prefix_fg=(150, 155, 170), warn_bg=(60, 30, 30), warn_fg=(250, 230, 230),
             input_bg=(44, 40, 30), input_fg=(240, 236, 220),
             help_bg=(20, 22, 30), help_fg=(220, 220, 230),
             list_header_fg=(139, 146, 165), list_muted_fg=(110, 118, 135),
             list_label_fg=(232, 236, 244), list_dim_fg=(200, 206, 216),
             list_active_bg=(37, 45, 64), list_cursor_bg=(28, 33, 46),
             list_rule_fg=(60, 66, 84), list_cap_bg=(42, 49, 66),
             list_panel_bg=(18, 20, 26), cleanup_hint_fg=(255, 170, 60),
             pt_bg=(18, 20, 26), pt_border_fg=(60, 200, 180),
             pt_text_fg=(220, 220, 230), pt_dim_fg=(110, 114, 126), pt_gap_bg=(40, 42, 50))

LIGHT = Theme(name="light", panel_bg=(225, 227, 232), panel_fg=(30, 32, 38),
              edit_prefix_fg=(90, 95, 110), warn_bg=(255, 225, 225), warn_fg=(120, 20, 20),
              input_bg=(255, 247, 214), input_fg=(90, 70, 10),
              help_bg=(238, 239, 243), help_fg=(35, 37, 44),
              list_header_fg=(70, 76, 95), list_muted_fg=(120, 126, 142),
              list_label_fg=(20, 22, 28), list_dim_fg=(55, 60, 72),
              list_active_bg=(202, 210, 230), list_cursor_bg=(216, 220, 230),
              list_rule_fg=(190, 195, 206), list_cap_bg=(206, 211, 222),
              list_panel_bg=(233, 235, 240), cleanup_hint_fg=(170, 90, 0),
              pt_bg=(238, 240, 244), pt_border_fg=(0, 140, 125),
              pt_text_fg=(30, 32, 38), pt_dim_fg=(120, 125, 138), pt_gap_bg=(220, 223, 230))
```

Every existing constant becomes a 1:1 field so the `viewer.py` refactor is
mechanical: each callsite that reads `_LIST_HEADER_FG` reads
`self.theme.list_header_fg` instead, no renaming, no consolidation of
similar-looking colors into one shared field. `DARK`'s values are exactly
today's constants — the dark theme is a no-op refactor, only `LIGHT` is new
content. The light-theme numbers above are a considered starting point, not
final: VIM-13 checks them live in an actual light-background terminal
(kitty/Ghostty/WezTerm, light color scheme) and adjusts any that read
poorly, same as any other visual tuning.

**Pill colors stay theme-invariant.** `_pill()` always paints its own
background and already picks readable black/white text from that
background's own luminance (`viewer.py:1444`) — that logic is local
contrast, not global theme, and needs no change. The geometry pill's teal
accent `(0.17, 0.71, 0.63)` likewise stays constant across themes.

**`render.Style.background` is out of scope.** The interactive viewer
defaults to a transparent render (`transparent=True`) precisely so the
*terminal's* background shows through the molecule — `Style.background` is
only used to pre-fill a non-transparent framebuffer, so it's irrelevant to
the default interactive path. If the user explicitly passes `--opaque` or
`--background`, that's a deliberate choice and must not be silently
overridden by auto-detected theme.

## 2. Detection

### OSC 11 query

One more question added to `kitty.probe_query_bytes()`'s existing combined
write (`kitty.py:289`) — no extra round trip:

```
query: ESC ] 11 ; ? ESC \
reply: ESC ] 11 ; rgb:RRRR/GGGG/BBBB (ST or BEL)
```

Terminals vary on channel width (2 or 4 hex digits) and reply terminator
(`ESC \` or `BEL`); the parser accepts both. For a 4-digit channel, take the
high byte (`"1e1e"` → `0x1e`), matching how every other 16-bit-channel OSC
color reply in the wild is downsampled to 8 bits.

`_parse_probe_pieces` gains a `bg_rgb: Optional[Tuple[int,int,int]]`
extraction (mirrors the existing cell-size/graphics/shm pattern);
`TerminalProbe` gains a `bg_rgb` field. Terminals that don't answer OSC 11
leave it `None` — same "silence is meaningful once the DA1 fence lands, else
unknown" reasoning already documented for `graphics`.

### Threshold and fallback ladder

```python
def luminance(rgb) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b

def from_colorfgbg(value: str) -> Optional[Theme]:
    """Last field of COLORFGBG ("fg;bg"): 7 or 15 -> light, else dark."""

def resolve(explicit, osc11_rgb, colorfgbg) -> Theme:
    """explicit -> osc11 -> colorfgbg -> DARK, in that order."""
```

Precedence, highest first:

1. `--theme dark|light` (CLI flag; explicit, no detection attempted)
2. `VIMOL_THEME=dark|light` (env; CLI flag wins if both are set)
3. OSC 11 reply, thresholded by `luminance` (same 0.299/0.587/0.114 weights
   already used for pill-text contrast, `viewer.py:1444`)
4. `COLORFGBG` heuristic
5. `DARK` (today's only behavior)

**Frame 0 doesn't wait for the probe.** The startup probe runs *after* the
first paint (`viewer.py:395`) so the very first frame shows pixels
immediately; that means OSC 11's reply isn't in hand yet for frame 0. Viewer
picks a synchronous guess for frame 0 from `--theme`/`VIMOL_THEME`/
`COLORFGBG` only (no I/O needed), then `_finish_startup` applies the OSC 11
verdict once the probe replies and forces a redraw if it disagrees with the
guess. This is a one-time correction, not continuous re-polling — same
shape as the existing idle-scale seeding in that method.

### `ctrl-t`

Bytes `\x14` (Ctrl-T), parsed today as a literal `KeyEvent("\x14")` via
`_parse_key_byte` — no new decoder work needed, same path `\x03` (Ctrl-C)
already takes. Added to `_BASE_DRIVER_KEYS` (available in both editable and
read-only viewers, like `d`/`g`/`t`/`n`/`p`/`m`). Flips `self.theme` between
`DARK`/`LIGHT` for the rest of the session; does not persist. Also updates
`self.widget.theme` (§3) so the molecule's light-overridden atoms flip with
it. Follows the same `kitty.write_bytes(_CLEAR, self.fd_out)` pattern the
existing `t` (transparent) toggle already uses (`viewer.py:1831`), so no
previous-theme chrome cells linger before the next full redraw repaints
them. Help text gains a line next to the existing `t` entry.

## 3. Color sweep

**The structure-list strip bug.** `_list_line` (`viewer.py:670`) takes a
`bg=None` default, and every non-active, non-cursor row is drawn with that
default — meaning those rows paint no background SGR at all and sit on
whatever the terminal's own background is, while their foreground colors
(`_LIST_HEADER_FG`, `_LIST_MUTED_FG`, `_LIST_DIM_FG`) are tuned only for a
dark one. Only the active and cursor rows get an explicit background
(`_LIST_ACTIVE_BG`/`_LIST_CURSOR_BG`) and so only those two read correctly
today regardless of terminal background. Fix: pass the new
`theme.list_panel_bg` field as every row's background, including ordinary
ones — `_draw_list` calls `_list_line(..., bg=self.theme.list_panel_bg)`
unconditionally instead of leaving it unset.

**Element colors.** `elements._COLORS` (CPK/Jmol palette) keeps its values
and `element_color()`/`Molecule.element_colors()` keep their signatures
unchanged — no threading a theme through `Molecule`, `StructureSet`, or
`render.py`. Instead this reuses the exact mechanism `MoleculeWidget`
already has for hover/measure tinting: `style.color_override`
(`scene.py:253`, consumed instead of `composite.base_colors` whenever it's
set).

A new pure helper in `elements.py`:

```python
_LIGHT_OVERRIDES = {  # only entries unreadable against a light background
    "H": (0.55, 0.55, 0.58), "He": (0.45, 0.75, 0.78), "Ag": (0.45, 0.45, 0.45),
    "Pt": (0.45, 0.45, 0.55), "Hg": (0.40, 0.40, 0.55),
}

def themed_base_colors(symbols, base_colors, theme: str) -> np.ndarray:
    """base_colors, with _LIGHT_OVERRIDES entries substituted when theme
    is "light"; returned unchanged (same array, no copy) when "dark"."""
```

`MoleculeWidget` gains a `self.theme` attribute (`"dark"` by default —
so direct `MoleculeWidget` use by an embedder that never sets it, per
`examples/embed_demo.py`, is unaffected), set by `Viewer` at startup and on
every `ctrl-t` toggle. `_apply_highlight` (`widget.py:686`) computes
`themed = elements.themed_base_colors(composite.molecule.symbols, composite.base_colors, self.theme)`
once at the top, and uses `themed` everywhere it currently uses
`composite.base_colors` — including the early-return branch (`hi is None
and not measure_sel`), which today sets `color_override = None` and must
instead set `color_override = themed if self.theme == "light" else None` so
the light overrides apply even with nothing hovered. Since
`themed_base_colors` is a no-op passthrough for `"dark"`, this is
byte-identical to today's rendering whenever the theme is dark — matching
the "byte-identical to today's behaviour" invariant `_effective_style`'s
docstring already documents for the no-override path.

Everything else in the CPK table is saturated enough to read on either
background and is left untouched.

**Checked, no change needed** (contrast against the render, not text on a
terminal background, so terminal theme doesn't apply):
- `structures.py`'s 8-color overlay-tint palette — these paint shaded 3D
  spheres with their own highlights/outline, not flat text.
- `widget._BOND_PREVIEW_COLOR` and the hover/delete/measure highlight tints
  in `widget._apply_highlight` — same reasoning, they're rendered geometry.
- The geometry pill's teal accent and `_pill()`'s self-computed text color
  (§1).

## 4. Copyable status bar

**Why not just fix mouse handling.** vimol enables SGR mouse reporting for
its own drag/click/scroll handling, which is exactly what prevents the
terminal's native drag-to-select from reaching the status bar text. Rather
than suspend mouse reporting conditionally (fragile, terminal-dependent,
and the visible text is already truncated/padded to fit column width, so a
raw selection wouldn't give you the full string anyway), a `y` key copies
the *underlying* value directly.

**Binding.** `y`, added to `_BASE_DRIVER_KEYS` (available in both modes,
like `m` measure). Copies whichever string is currently backing the status
bar's left field — the same value `_status_bar` computes into `raw_left`
(`viewer.py:1523`): live measurement text, `pick_refusal`, hover atom info,
or `f"{mol.name or 'molecule'}  {mol.formula()}  {mol.n_atoms} atoms"` — as
the **untruncated** string, not the display-clipped/padded version. On
success, `self._msg = "copied"` flashes briefly in the same slot (existing
`_msg` mechanism already used for e.g. "saved foo.xyz").

**Mechanism: OSC 52.**

```
ESC ] 52 ; c ; <base64 of the text> ESC \
```

`c` targets the system clipboard. Supported by kitty, Ghostty, WezTerm,
iTerm2, and passed through by tmux with `set-clipboard on` — the standard
way a terminal app writes the clipboard over SSH without any local
clipboard access. `kitty.py` gains `osc52_copy(text: str) -> bytes`,
written straight to `fd_out` like every other control sequence in that
module.

**Full xyz comment.** `parsers/xyz.py:46` currently does
`name=comment.strip()[:60]` at *parse* time — a long ORCA/Gaussian energy
comment is already gone before a `y` press could copy it, and a save
round-trip silently shortens it further. Fix: keep the full string on
`Molecule.name`; the 60-char cap moves to display-only call sites that
actually need a bounded width (the strip's structure labels, if any
currently rely on the implicit cap — checked during implementation).
`dumps()` (xyz.py:53) is unaffected either way since it already just writes
`mol.name` verbatim.

## Testing

- `theme.resolve()` precedence, pure unit tests: explicit beats env beats
  OSC 11 beats `COLORFGBG` beats default, and `COLORFGBG` variants
  (`"15;0"`, `"0;15"`, garbage) resolve as documented.
- `kitty.parse_probe_reply` gains cases for an OSC 11 reply present/absent/
  malformed, both terminator styles, both channel widths, alongside the
  existing graphics/pixel-mouse/cell-size cases.
- `osc52_copy` produces the exact expected byte sequence for a plain string
  and one containing bytes needing base64 padding edge cases.
- xyz parse/dump round-trip with a >60-char comment stays intact.
- Existing structure-strip tests (`test_viewer_draw_multi_structure_writes_image_and_list_strip`)
  keep passing with the new unconditional `bg=` — should be behavior-neutral
  under `DARK` since `list_panel_bg` there equals the old "no background"
  look closely enough on a typical dark terminal; if any test asserts on the
  exact absence of a background SGR for a non-active row, it gets updated to
  assert the new explicit one instead.
