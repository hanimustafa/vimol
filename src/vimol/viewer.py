"""Full-screen interactive viewer — a thin driver around MoleculeWidget.

All the interaction logic lives in :class:`vimol.widget.MoleculeWidget` and
input decoding in :class:`vimol.input.InputDecoder`. This class only does the
terminal-owning parts: raw mode, the alternate screen, enabling mouse
reporting, the render/input loop, and a status bar. Embedders who want to
capture the mouse in their own region should use the widget + decoder directly
(see examples/embed_demo.py) rather than this driver.
"""
from __future__ import annotations

import os
import select
import time
from typing import List, Optional, Tuple

from .molecule import Molecule
from .render import Style
from .widget import MoleculeWidget, REPRESENTATIONS
from .bonds import ensure_bonds
from . import editor
from . import kitty
from . import input as _input
from . import elements
from . import templates
from . import periodic_table

# ANSI / terminal control -------------------------------------------------
_ALT_SCREEN_ON = b"\x1b[?1049h"
_ALT_SCREEN_OFF = b"\x1b[?1049l"
_HIDE_CURSOR = b"\x1b[?25l"
_SHOW_CURSOR = b"\x1b[?25h"
_CLEAR = b"\x1b[2J"
_HOME = b"\x1b[H"
# The Kitty graphics protocol draws images *above* normal cell text by
# default (z=0) -- any text this driver writes into the image's row range
# (the help panel, the periodic-table picker) would otherwise be obscured by
# the molecule rather than the other way around. A z-index below -2^31/2
# moves the image under any cell that has an explicit (non-default)
# background color, i.e. under exactly the overlays this driver draws, while
# leaving ordinary image-only cells (no background set) unaffected.
_IMAGE_Z_INDEX = -1_200_000_000
# How many rows above the status bar (inclusive of it) are a "dead zone" that
# never forwards clicks to the 3D viewport -- a guard so a click on or just
# above the element button can't be misread as "click empty space" and birth a
# stray atom. Bumped above a bare 1-2 rows because those near-button misfires
# were still slipping through.
_STATUS_ZONE_ROWS = 4
# Footer hint shown inside the geometry picker (also sizes its minimum width).
_GEOM_HINT = " ↑↓ move · Enter/click select · Esc cancel"
# Fixed visible width of the status bar's left-hand (hover/molecule-info)
# field -- see _status_bar for why this must be constant, not just capped.
_LEFT_WIDTH = 24

_HELP_HEAD = [
    "  vimol — terminal molecular viewer",
    "",
    "  Mouse drag ......... rotate            Wheel / + - ........ zoom",
    "  Right / mid drag ... pan               [ / ] .............. roll",
    "  Hover .............. identify atom      Arrows / h j k l ... rotate",
    "  1 2 3 4 ............ ball / space / licorice / wire",
]
# Shown only when editing is disabled (the classic bindings).
_HELP_VIEW = [
    "  s .................. cycle style       a .................. autospin",
    "  m .................. measure (click 2/3/4 atoms: distance/angle/dihedral)",
]
# Shown only when editing is enabled (Viewer(editable=True), the vimol CLI default).
_HELP_EDIT = [
    "  a .................. append (edit)     o .................. autospin",
    "     click H -> grow · heavy atom -> replace · empty space -> new molecule",
    "     click the [ C ] pill -> pick a different element to build",
    "  x .................. delete            s .................. save",
    "  u .................. undo",
    "     option-drag atom -> atom ... draw a bond (kept beyond auto range)",
    "  c .................. cleanup clashes / long bonds",
    "  m .................. measure (click 2/3/4 atoms: distance/angle/dihedral)",
]
_HELP_TAIL = [
    "  n / p .............. next/prev frame   d .................. depth cue",
    "  t .................. transparent bg    g .................. hi-quality",
    "  f / r / z .......... re-fit / reset    ? .................. toggle help",
    "  q / Esc ............ quit",
]


def _help_lines(editable: bool):
    return _HELP_HEAD + (_HELP_EDIT if editable else _HELP_VIEW) + _HELP_TAIL

# Keys the driver always claims. 'a' is here in both modes but means different
# things: autospin when read-only, append when editable (see _driver_key).
_BASE_DRIVER_KEYS = {"q", "escape", "a", "?", "d", "g", "t", "n", "p", "m", "\x03"}
# Extra keys claimed only when editing is enabled.
_EDIT_DRIVER_KEYS = {"s", "u", "o", "x", "c"}

# Warm warning color for the status bar's "press c to cleanup" hint.
_CLEANUP_HINT_FG = (255, 170, 60)

# Periodic-table picker panel colors.
_PT_BG = (18, 20, 26)
_PT_BORDER_FG = (60, 200, 180)      # teal accent, matches the geometry pill
_PT_TEXT_FG = (220, 220, 230)
_PT_DIM_FG = (110, 114, 126)
_PT_GAP_BG = (40, 42, 50)


class Viewer:
    def __init__(self, molecule: Molecule, frames: Optional[List[Molecule]] = None,
                 style: Optional[Style] = None, fd_in: int = 0, fd_out: int = 1,
                 autospin: bool = False, target_fps: float = 60.0, picking: bool = True,
                 transparent: bool = True, backend: str = "auto",
                 source_path: Optional[str] = None, editable: bool = False):
        self.frames = frames or [molecule]
        self.source_path = source_path
        self.editable = editable
        for m in self.frames:
            ensure_bonds(m)
        self.frame_index = 0
        self.style = style or Style()
        if style is None:
            # default to a terminal-matching transparent background
            self.style.transparent = transparent
        self.fd_in = fd_in
        self.fd_out = fd_out
        self.autospin = autospin
        self.target_fps = target_fps

        self.widget = MoleculeWidget(self.frames[0], 320, 240, style=self.style,
                                     supersample=1, picking=picking, backend=backend,
                                     editable=editable)
        # Editing keys ('a' append, 's' save, 'u' undo, 'o' autospin) are only
        # bound when editing is enabled; otherwise 'a' keeps its classic meaning
        # (autospin) and 's' falls through to the widget (cycle representation).
        self._driver_keys = set(_BASE_DRIVER_KEYS)
        if editable:
            self._driver_keys |= _EDIT_DRIVER_KEYS
        self.decoder = _input.InputDecoder(pixel=False)
        self._max_ss = 2
        self._drawn_ss = None   # supersample of the last frame actually drawn
        # single per-process image id, replaced in place each frame (this is
        # flicker-free in kitty and, unlike a double buffer, never lets a
        # transparent frame ghost the previous one through its cutout).
        self._img_id = kitty.unique_id_base() + 1

        self._running = False
        self._show_help = False
        # modal state: "normal" | "save_input" | "save_confirm" |
        # "quit_confirm" | "periodic_table" | "geometry_picker"
        self._mode = "normal"
        self._quit_after_save = False    # ESC-quit routed through the save prompt
        self._input_buf = ""
        self._msg = ""                   # transient status message (e.g. "saved foo.xyz")
        # periodic-table picker: cursor position (row, col) into periodic_table.GRID
        self._pt_row, self._pt_col = periodic_table.position_of("C")
        # (row, col_start, col_end) of the clickable element / geometry pills in
        # the last drawn status bar, 0-based cell coords; None when not shown.
        self._elem_button_span = None
        self._geom_button_span = None
        # geometry/hybridization picker: the options list and cursor index
        self._geom_opts: List = []
        self._geom_idx = 0
        # True from a mouse-down that landed in the status-bar zone (see
        # _in_status_zone) until the matching up -- keeps a drag that started
        # on the status bar from ever reaching the 3D viewport, even if the
        # pointer strays back over the molecule mid-drag.
        self._status_zone_press = False
        self._last_interact = 0.0
        self._cols = self._rows = 0
        self._img_cols = self._img_rows = 1
        self._cell_px = None                 # exact (cw, ch) from the terminal, if it answered
        self._old_termios = None
        self._geometry_established = False   # True once the real (not placeholder) size is known
        # True while WE own a pushed OSC-22 pointer shape (delete's crosshair,
        # measure's cell). Pushes and pops must pair exactly: an unbalanced pop
        # would clobber a shape pushed by something outside vimol (tmux, the
        # hosting app), and an unbalanced push leaks ours onto their stack.
        self._pointer_pushed = False

    # -- pointer shape (OSC 22 push/pop stack) -----------------------------
    def _push_pointer(self, shape: str) -> None:
        """Push a pointer *shape*, first popping any shape we already own."""
        if self._pointer_pushed:
            kitty.write_bytes(kitty.reset_pointer_shape(), self.fd_out)
        kitty.write_bytes(kitty.set_pointer_shape(shape), self.fd_out)
        self._pointer_pushed = True

    def _pop_pointer(self) -> None:
        """Pop our pushed pointer shape; a no-op when we own none."""
        if self._pointer_pushed:
            kitty.write_bytes(kitty.reset_pointer_shape(), self.fd_out)
            self._pointer_pushed = False

    # -- terminal lifecycle ----------------------------------------------
    def _enter(self):
        import termios
        import tty
        self._old_termios = termios.tcgetattr(self.fd_in) if os.isatty(self.fd_in) else None
        if self._old_termios is not None:
            tty.setraw(self.fd_in)
        kitty.write_bytes(_ALT_SCREEN_ON + _HIDE_CURSOR + _CLEAR, self.fd_out)
        kitty.write_bytes(_input.enable_mouse(pixel=True, hover=self.widget.picking), self.fd_out)
        # probe whether the terminal actually reports pixel coordinates, and
        # (once, in raw mode) ask for its exact cell size so pixel->cell
        # hit-testing lines up with where glyphs are really drawn.
        if self._old_termios is not None:
            self.decoder.pixel = _input.supports_pixel_mouse(self.fd_in, self.fd_out)
            self._cell_px = kitty.query_cell_size_px(self.fd_in, self.fd_out)

    def _exit(self):
        import termios
        cleanup = kitty.delete_image(self._img_id)
        # pop our pointer shape iff we pushed one (quit/kill mid-delete or
        # mid-measure must not leave the cursor stuck) -- but never a bare
        # unbalanced pop, which would clobber a shape pushed outside vimol.
        pointer = kitty.reset_pointer_shape() if self._pointer_pushed else b""
        self._pointer_pushed = False
        kitty.write_bytes(_input.disable_mouse(pixel=True) + cleanup
                          + pointer
                          + _SHOW_CURSOR + _ALT_SCREEN_OFF, self.fd_out)
        if self._old_termios is not None:
            termios.tcsetattr(self.fd_in, termios.TCSADRAIN, self._old_termios)

    # -- geometry ---------------------------------------------------------
    def _update_geometry(self) -> bool:
        cols, rows, xpx, ypx = kitty.terminal_size_px(self.fd_out)
        # Prefer the terminal's authoritative cell size (queried once at enter);
        # fall back to the window-px/cell-count estimate if it didn't answer.
        cw, ch = self._cell_px or kitty.cell_size_px(self.fd_out)
        changed = (cols, rows) != (self._cols, self._rows)
        self._cols, self._rows = cols, rows
        self.widget.set_cell_metrics(cw, ch)
        if changed:
            self._img_rows = max(rows - 1, 1)   # reserve one row for status
            self._img_cols = cols
            w = int(self._img_cols * cw)
            h = int(self._img_rows * ch)
            # The widget was built at a 320x240 placeholder (the real terminal
            # size isn't known until the tty is in raw mode); the first time we
            # learn the real size, fit fresh to it rather than preserving the
            # zoom that was fit for the placeholder. Every later resize (the
            # user actually resizing their terminal) preserves the view as usual.
            self.widget.set_pixel_size(max(w, 16), max(h, 16),
                                       refit=not self._geometry_established)
            self._geometry_established = True
        return changed

    # -- rendering --------------------------------------------------------
    def _target_ss(self) -> int:
        """Supersample factor we *want* right now: 1 while interacting (fast),
        higher once settled (crisp). The loop redraws when this changes so the
        crisp frame lands ~0.25s after you stop, without re-drawing meanwhile.
        """
        idle = (time.time() - self._last_interact) > 0.25 and not self.autospin
        return self._max_ss if idle else 1

    def _draw(self):
        want_ss = self._target_ss()
        if self.widget.scene.supersample != want_ss:
            self.widget.scene.set_supersample(want_ss)
        self._drawn_ss = want_ss

        img = self.widget.render()
        data = kitty.encode_image(img, image_id=self._img_id, placement_id=self._img_id,
                                  cols=self._img_cols, rows=self._img_rows, move_cursor=False,
                                  z_index=_IMAGE_Z_INDEX)
        out = bytearray()
        out += _HOME + data
        out += b"\x1b[%d;1H\x1b[2K" % self._rows
        out += self._status_bar().encode("utf-8", "replace")
        kitty.write_bytes(bytes(out), self.fd_out)
        if self._show_help:
            self._draw_help()
        elif self._mode == "periodic_table":
            self._draw_periodic_table()
        elif self._mode == "geometry_picker":
            self._draw_geometry_picker()

    def _draw_help(self):
        out = bytearray()
        for k, line in enumerate(_help_lines(self.editable)):
            out += b"\x1b[%d;3H" % (2 + k)
            out += b"\x1b[48;2;20;22;30m\x1b[38;2;220;220;230m"
            out += (" " + line.ljust(58)).encode()
            out += b"\x1b[0m"
        kitty.write_bytes(bytes(out), self.fd_out)

    # -- periodic-table picker ---------------------------------------------
    def _pt_geometry(self) -> Tuple[int, int, int, int]:
        """(top, left, width, height) of the picker panel, 0-based cell coords.

        Anchored like a dropdown: horizontally centered on the element button
        that opens it, flush against the row just above the status bar --
        not centered on screen, so it opens right where you clicked instead
        of off in the middle of the viewport.
        """
        cell_w = 4
        grid_w = periodic_table.N_COLS * cell_w
        width = grid_w + 4                          # 2 borders + 1 pad each side
        height = len(periodic_table.GRID) + 4        # border+grid+info+hint+border

        if self._elem_button_span is not None:
            _row, col_start, col_end = self._elem_button_span
            anchor_center = (col_start + col_end) // 2
        else:
            anchor_center = self._cols // 2
        left = max(0, min(anchor_center - width // 2, max(self._cols - width, 0)))
        top = max(0, self._rows - 1 - height)
        return top, left, width, height

    def _pt_cell_at_screen(self, screen_row: int, screen_col: int):
        """The Cell at 0-based screen (row, col), or None if outside the grid."""
        top, left, _width, _height = self._pt_geometry()
        r = screen_row - (top + 1)
        if not (0 <= r < len(periodic_table.GRID)):
            return None
        grid_col_start = left + 2
        if screen_col < grid_col_start:
            return None
        c = (screen_col - grid_col_start) // 4
        if not (0 <= c < periodic_table.N_COLS):
            return None
        return periodic_table.GRID[r][c]

    @staticmethod
    def _pt_cell_text(cell, cursor: bool) -> str:
        """The 4-char escaped label for one periodic-table cell."""
        if cell is None:
            return "    "
        if cell.symbol is None:
            bg, fg = _PT_GAP_BG, _PT_DIM_FG
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

    def _draw_periodic_table(self):
        top, left, width, height = self._pt_geometry()
        inner_w = width - 2
        border = (f"\x1b[48;2;{_PT_BG[0]};{_PT_BG[1]};{_PT_BG[2]}m"
                  f"\x1b[38;2;{_PT_BORDER_FG[0]};{_PT_BORDER_FG[1]};{_PT_BORDER_FG[2]}m")
        bg_only = f"\x1b[48;2;{_PT_BG[0]};{_PT_BG[1]};{_PT_BG[2]}m"
        text_fg = f"\x1b[38;2;{_PT_TEXT_FG[0]};{_PT_TEXT_FG[1]};{_PT_TEXT_FG[2]}m"
        out = bytearray()

        def put(row0: int, col0: int, s: str) -> None:
            out.extend(b"\x1b[%d;%dH" % (row0 + 1, col0 + 1))
            out.extend(s.encode("utf-8", "replace"))

        title = " Pick an element ".center(inner_w, "─")
        put(top, left, f"{border}┌{title}┐\x1b[0m")

        for r, grow in enumerate(periodic_table.GRID):
            cells = "".join(self._pt_cell_text(c, r == self._pt_row and ci == self._pt_col)
                            for ci, c in enumerate(grow))
            put(top + 1 + r, left, f"{border}│ \x1b[0m{bg_only}{cells}\x1b[0m{border} │\x1b[0m")

        cur = periodic_table.GRID[self._pt_row][self._pt_col]
        if cur is not None and cur.symbol is not None:
            geom = templates.default_template(cur.symbol).geometry
            info = f" {elements.element_name(cur.symbol)} ({cur.symbol}) → {geom}"
        elif cur is not None:
            info = f" {cur.note or ''}"
        else:
            info = ""
        hint = " ↑↓←→ move · Enter/click select · Esc cancel"
        info_row = top + 1 + len(periodic_table.GRID)
        put(info_row, left, f"{border}│\x1b[0m{bg_only}{text_fg}{info.ljust(inner_w)}\x1b[0m{border}│\x1b[0m")
        put(info_row + 1, left, f"{border}│\x1b[0m{bg_only}{text_fg}{hint.ljust(inner_w)}\x1b[0m{border}│\x1b[0m")
        put(info_row + 2, left, f"{border}└{'─' * inner_w}┘\x1b[0m")

        kitty.write_bytes(bytes(out), self.fd_out)

    def _open_periodic_table(self) -> None:
        self._mode = "periodic_table"
        self._pt_row, self._pt_col = periodic_table.position_of(self.widget.build_element)
        self._msg = ""

    def _close_periodic_table(self, pick: Optional[str]) -> None:
        if pick is not None:
            self.widget.build_element = pick
            # a new element resets geometry to that element's default template
            self.widget.build_template = None
        # The overlay painted opaque text cells over the molecule image. Erase
        # only those rows (to the default background, which the negatively
        # z-indexed image shows through) instead of clearing the whole screen
        # -- a full \x1b[2J blanks and repaints everything, which reads as an
        # abrupt full-terminal flash; this just lifts the panel off.
        top, _left, _width, height = self._pt_geometry()
        self._mode = "normal"
        self._erase_rows(top, height)

    def _erase_rows(self, top: int, count: int) -> None:
        """Erase `count` terminal rows starting at 0-based row `top`.

        Each row is reset to the default background and cleared, so the
        molecule image (drawn beneath default-background cells) reappears
        there without a whole-screen repaint.
        """
        out = bytearray()
        last = min(top + count, max(self._rows - 1, 0))   # never wipe the status row
        for r in range(max(top, 0), last):
            out += b"\x1b[%d;1H\x1b[0m\x1b[2K" % (r + 1)
        if out:
            kitty.write_bytes(bytes(out), self.fd_out)

    # -- geometry / hybridization picker ----------------------------------
    def _geom_label_width(self) -> int:
        return max((len(o.label()) for o in self._geom_opts), default=8)

    def _geom_geometry(self) -> Tuple[int, int, int, int]:
        """(top, left, width, height) of the geometry picker, anchored above
        the geometry pill and flush against the row above the status bar."""
        title_w = len(f" {self.widget.build_element}: geometry ")
        # inner width must fit the widest of: an option row (" ● label "),
        # the title, and the hint -- else that content overruns the border.
        inner = max(self._geom_label_width() + 4, title_w, len(_GEOM_HINT))
        width = inner + 2
        height = len(self._geom_opts) + 3           # top border + rows + hint + bottom
        if self._geom_button_span is not None:
            _row, c0, c1 = self._geom_button_span
            anchor = (c0 + c1) // 2
        else:
            anchor = self._cols // 2
        left = max(0, min(anchor - width // 2, max(self._cols - width, 0)))
        top = max(0, self._rows - 1 - height)
        return top, left, width, height

    def _geom_row_at_screen(self, screen_row: int, screen_col: int) -> Optional[int]:
        top, left, width, _height = self._geom_geometry()
        if not (left <= screen_col < left + width):
            return None
        i = screen_row - (top + 1)
        if 0 <= i < len(self._geom_opts):
            return i
        return None

    def _draw_geometry_picker(self):
        top, left, width, height = self._geom_geometry()
        inner_w = width - 2
        border = (f"\x1b[48;2;{_PT_BG[0]};{_PT_BG[1]};{_PT_BG[2]}m"
                  f"\x1b[38;2;{_PT_BORDER_FG[0]};{_PT_BORDER_FG[1]};{_PT_BORDER_FG[2]}m")
        bg_only = f"\x1b[48;2;{_PT_BG[0]};{_PT_BG[1]};{_PT_BG[2]}m"
        text_fg = f"\x1b[38;2;{_PT_TEXT_FG[0]};{_PT_TEXT_FG[1]};{_PT_TEXT_FG[2]}m"
        out = bytearray()

        def put(row0: int, col0: int, s: str) -> None:
            out.extend(b"\x1b[%d;%dH" % (row0 + 1, col0 + 1))
            out.extend(s.encode("utf-8", "replace"))

        active = self._active_template()
        title = f" {self.widget.build_element}: geometry ".center(inner_w, "─")
        put(top, left, f"{border}┌{title}┐\x1b[0m")
        for i, opt in enumerate(self._geom_opts):
            is_active = (opt.valence == active.valence and opt.geometry == active.geometry)
            marker = "●" if is_active else " "
            label = f" {marker} {opt.label()}".ljust(inner_w)
            if i == self._geom_idx:
                row_s = f"{bg_only}\x1b[1m\x1b[7m{label}\x1b[27m\x1b[22m"
            else:
                row_s = f"{bg_only}{text_fg}{label}"
            put(top + 1 + i, left, f"{border}│\x1b[0m{row_s}\x1b[0m{border}│\x1b[0m")
        hint = _GEOM_HINT
        hint_row = top + 1 + len(self._geom_opts)
        put(hint_row, left, f"{border}│\x1b[0m{bg_only}{text_fg}{hint.ljust(inner_w)}\x1b[0m{border}│\x1b[0m")
        put(hint_row + 1, left, f"{border}└{'─' * inner_w}┘\x1b[0m")
        kitty.write_bytes(bytes(out), self.fd_out)

    def _open_geometry_picker(self) -> None:
        self._mode = "geometry_picker"
        self._geom_opts = templates.options_for(self.widget.build_element)
        active = self._active_template()
        self._geom_idx = 0
        for i, opt in enumerate(self._geom_opts):
            if opt.valence == active.valence and opt.geometry == active.geometry:
                self._geom_idx = i
                break
        self._msg = ""

    def _close_geometry_picker(self, pick) -> None:
        if pick is not None:
            self.widget.build_template = pick
            self.widget.build_element = pick.element
        top, _left, _width, height = self._geom_geometry()
        self._mode = "normal"
        self._erase_rows(top, height)

    def _geom_activate(self) -> None:
        if self._geom_opts:
            self._close_geometry_picker(self._geom_opts[self._geom_idx])

    def _handle_geom_event(self, ev) -> bool:
        """Drive the geometry picker. Returns True if the display changed."""
        n = len(self._geom_opts)
        if isinstance(ev, _input.KeyEvent):
            key = ev.key
            if key in ("escape", "\x03"):
                self._close_geometry_picker(None); return True
            if key in ("up", "k"):
                self._geom_idx = max(0, self._geom_idx - 1); return True
            if key in ("down", "j"):
                self._geom_idx = min(n - 1, self._geom_idx + 1); return True
            if key == "enter":
                self._geom_activate(); return True
            return False
        if isinstance(ev, _input.MouseEvent):
            if ev.action not in ("down", "move"):
                return False
            col, row = self._event_cell(ev)
            i = self._geom_row_at_screen(row, col)
            if i is None:
                if ev.action == "down":
                    # a click outside the list (the pills were already
                    # intercepted in _dispatch) closes the picker without
                    # picking anything -- same as Escape.
                    self._close_geometry_picker(None)
                    return True
                return False
            if ev.action == "move":
                if i != self._geom_idx:
                    self._geom_idx = i
                    return True
                return False
            self._geom_idx = i
            self._geom_activate()
            return True
        return False

    @staticmethod
    def _pt_nearest_col(row_cells, col: int) -> Optional[int]:
        """The nearest landable column to *col* in *row_cells* (itself if valid)."""
        if row_cells[col] is not None:
            return col
        best = None
        for cc in range(periodic_table.N_COLS):
            if row_cells[cc] is not None and (best is None or abs(cc - col) < abs(best - col)):
                best = cc
        return best

    def _pt_move(self, dr: int, dc: int) -> None:
        """Move the picker cursor, skipping blank (non-existent) grid cells."""
        row, col = self._pt_row, self._pt_col
        if dc:
            c = col
            while True:
                c += dc
                if not (0 <= c < periodic_table.N_COLS):
                    return
                if periodic_table.GRID[row][c] is not None:
                    self._pt_col = c
                    return
        else:
            r = row
            while True:
                r += dr
                if not (0 <= r < len(periodic_table.GRID)):
                    return
                best = self._pt_nearest_col(periodic_table.GRID[r], col)
                if best is not None:
                    self._pt_row, self._pt_col = r, best
                    return

    def _pt_activate(self) -> None:
        """Enter/click on the current cursor cell: pick the element, or jump
        to the lanthanide/actinide row if the cursor is on a gap placeholder."""
        cell = periodic_table.GRID[self._pt_row][self._pt_col]
        if cell is None:
            return
        if cell.symbol is not None:
            self._close_periodic_table(pick=cell.symbol)
        elif cell.jump_row is not None:
            target = periodic_table.GRID[cell.jump_row]
            col = self._pt_nearest_col(target, self._pt_col)
            if col is not None:
                self._pt_row, self._pt_col = cell.jump_row, col

    def _event_cell(self, ev: _input.MouseEvent) -> Tuple[int, int]:
        """A mouse event's position in 0-based (col, row) terminal cells."""
        if ev.pixel:
            cw = self.widget.cell_w or 1.0
            ch = self.widget.cell_h or 1.0
            return int(ev.x // cw), int(ev.y // ch)
        return int(ev.x), int(ev.y)

    def _in_status_zone(self, row: int) -> bool:
        """True for the status bar's row plus a few rows of margin above it.

        Every mouse event landing here is kept from ever reaching the 3D
        viewport (see _dispatch) -- not just clicks on the element button's
        exact span -- so a near-miss click can never be misread as "click
        empty space" and birth an atom right under the button. The margin
        (see _STATUS_ZONE_ROWS) also absorbs any off-by-one in the terminal's
        own pixel/cell rounding, so a click that visually looks like it
        landed on the button still registers even if it decodes a row off.
        """
        if self._rows <= 0:      # geometry not established yet -- nothing to protect
            return False
        return row >= self._rows - _STATUS_ZONE_ROWS

    def _handle_pt_event(self, ev) -> bool:
        """Drive the periodic-table picker. Returns True if the display changed."""
        if isinstance(ev, _input.KeyEvent):
            key = ev.key
            if key in ("escape", "\x03"):
                self._close_periodic_table(pick=None)
                return True
            if key in ("up", "k"):
                self._pt_move(-1, 0); return True
            if key in ("down", "j"):
                self._pt_move(1, 0); return True
            if key in ("left", "h"):
                self._pt_move(0, -1); return True
            if key in ("right", "l"):
                self._pt_move(0, 1); return True
            if key == "enter":
                self._pt_activate()
                return True
            return False
        if isinstance(ev, _input.MouseEvent):
            if ev.action not in ("down", "move"):
                return False
            col, row = self._event_cell(ev)
            cell = self._pt_cell_at_screen(row, col)
            if cell is None:
                if ev.action == "down":
                    # a click outside the grid (the pills were already
                    # intercepted in _dispatch) closes the picker without
                    # picking anything -- same as Escape.
                    self._close_periodic_table(pick=None)
                    return True
                return False
            if ev.action == "move":
                if (cell.row, cell.col) != (self._pt_row, self._pt_col):
                    self._pt_row, self._pt_col = cell.row, cell.col
                    return True
                return False
            self._pt_row, self._pt_col = cell.row, cell.col
            self._pt_activate()
            return True
        return False

    @staticmethod
    def _pill(label: str, bg, fg=None) -> str:
        """A padded, bold, reverse-video 'button' -- a clickable-looking pill."""
        r, g, b = (int(c * 255) if c <= 1 else int(c) for c in bg)
        if fg is None:
            lum = 0.299 * r + 0.587 * g + 0.114 * b       # pick readable text
            fg = (10, 12, 14) if lum > 140 else (245, 246, 250)
        fr, fg_, fb = fg
        return (f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fr};{fg_};{fb}m\x1b[1m"
                f" {label} \x1b[22m\x1b[0m")

    def _active_template(self):
        """The template a build would use now: the chosen one, else the
        current element's default."""
        return self.widget.build_template or templates.default_template(self.widget.build_element)

    def _edit_buttons(self) -> Tuple[str, int, Tuple[int, int], Tuple[int, int]]:
        """'adding [ C ] [ tetrahedral ]' with each token as a colored button.

        Clicking the element pill opens the periodic-table picker; clicking the
        geometry pill opens the geometry/hybridization picker for that element
        (see _open_periodic_table / _open_geometry_picker). Returns the escaped
        text, its own total visible width, and the (start, end) visible-column
        spans of the element pill and the geometry pill, each relative to this
        text's own start -- so callers can locate both clickable buttons
        without re-deriving _pill's layout by hand.
        """
        elem = self.widget.build_element
        geom = self._active_template().geometry
        prefix = "adding "
        elem_visible = f" {elem} "
        geom_visible = f" {geom} "
        elem_btn = self._pill(elem, elements.element_color(elem))
        geom_btn = self._pill(geom, (0.17, 0.71, 0.63))       # teal accent
        text = f"\x1b[38;2;150;155;170m{prefix}\x1b[0m{elem_btn} {geom_btn}"
        elem_start = len(prefix)
        elem_end = elem_start + len(elem_visible)
        geom_start = elem_end + 1                # +1 for the space between the pills
        geom_end = geom_start + len(geom_visible)
        return text, geom_end, (elem_start, elem_end), (geom_start, geom_end)

    @staticmethod
    def _build_segment(pieces) -> Tuple[str, int, List[int]]:
        """Join (escaped_text, visible_len) pieces into one string.

        Returns the joined escaped text, its total visible length, and each
        piece's visible-column offset from the segment's own start -- so a
        caller can locate a button embedded in one of the pieces without
        re-deriving the layout of everything drawn before it.
        """
        parts = []
        offsets = []
        total = 0
        for escaped, vis_len in pieces:
            offsets.append(total)
            parts.append(escaped)
            total += vis_len
        return "".join(parts), total, offsets

    def _status_bar(self) -> str:
        self._elem_button_span = None
        self._geom_button_span = None
        if self._mode == "save_input":
            body = f" Save to: {self._input_buf}█   Enter save · Esc cancel "
            return f"\x1b[48;2;44;40;30m\x1b[38;2;240;236;220m{body}\x1b[0m"
        if self._mode == "save_confirm":
            name = os.path.basename(self._input_buf.strip())
            body = f" {name} exists — replace? (y/n) "
            return f"\x1b[48;2;60;30;30m\x1b[38;2;250;230;230m{body}\x1b[0m"
        if self._mode == "quit_confirm":
            body = " unsaved changes — save before quitting? (y/n/Esc) "
            return f"\x1b[48;2;60;30;30m\x1b[38;2;250;230;230m{body}\x1b[0m"
        mol = self.widget.molecule
        hov = self.widget.atom_info(self.widget.hovered)
        # a live measurement readout (2+ picks in measure mode) outranks the
        # hover text; with 0-1 picks measurement() is "" and the normal
        # left-segment behavior applies.
        measure = (editor.measurement(mol, self.widget.measure_sel)
                   if self.widget.measure_mode else "")
        raw_left = measure or hov or (self._msg or
            f"{(mol.name or 'molecule')[:22]}  {mol.formula()}  {mol.n_atoms} atoms")
        # Fixed-width, not just truncated: hover text changes on every mouse
        # move and can be shorter *or* longer than the previous frame's, so
        # only a constant width (never just a cap) keeps the trailer -- and
        # the clickable element button in it -- from drifting sideways as
        # the cursor merely moves around the molecule.
        left = (raw_left[:_LEFT_WIDTH - 1] + "…") if len(raw_left) > _LEFT_WIDTH else raw_left.ljust(_LEFT_WIDTH)
        rep = self.style.representation
        frame = f"  {self.frame_index+1}/{len(self.frames)}" if len(self.frames) > 1 else ""
        spin = " ⟳" if self.autospin else ""
        px = " px" if self.decoder.pixel else ""
        backend = "gpu" if self.widget.scene.backend == "gl" else "cpu"
        base = "\x1b[48;2;30;33;44m\x1b[38;2;230;232;240m"
        mod = " [MODIFIED]" if (self.editable and self.widget.dirty) else ""
        hint = "  s save  q quit" if self.editable else "  q quit"
        show_buttons = self.editable and self.widget.append_mode
        show_delete = self.editable and self.widget.delete_mode
        show_measure = self.widget.measure_mode      # read-only-safe: no editable gate

        # Everything from the representation tag onward is a "trailer" built
        # from (escaped, visible_len) pieces and right-anchored via padding
        # below -- independent of `left`'s length. `left` carries hover text
        # that changes on every mouse move; if the clickable element button
        # merely followed it in one concatenated string, moving the mouse
        # (with no click at all) would shift the button out from under the
        # cursor before the next click landed.
        pieces = [(f"[{rep}]", len(rep) + 2), (frame, len(frame)), (spin, len(spin))]
        if show_buttons:
            pieces.append((f"  {base}\x1b[1m✎APPEND\x1b[22m", 9))          # "  ✎APPEND"
        elif show_delete:
            pieces.append((f"  {base}\x1b[1m✗DELETE\x1b[22m", 9))          # "  ✗DELETE"
        elif show_measure:
            pieces.append((f"  {base}\x1b[1m∡MEASURE\x1b[22m", 10))        # "  ∡MEASURE"
        pieces.append((mod, len(mod)))
        # Cleanup hint: recomputed from model state every render (no hover
        # dependence), so it appears/disappears exactly like [MODIFIED] does
        # and never disturbs the button-span stability tests.
        cleanup_hint = ""
        if self.editable:
            clash, stretched = editor.cleanup_targets(mol)
            if clash or stretched:
                r, g, b = _CLEANUP_HINT_FG
                cleanup_hint = f"  \x1b[38;2;{r};{g};{b}m\x1b[1m⚠ c cleanup\x1b[22m{base}"
        cleanup_hint_len = len("  ⚠ c cleanup") if cleanup_hint else 0
        pieces.append((cleanup_hint, cleanup_hint_len))
        elem_piece_idx = None
        if show_buttons:
            buttons_text, buttons_len, elem_rel, geom_rel = self._edit_buttons()
            elem_piece_idx = len(pieces)
            pieces.append((f"  {buttons_text}{base}", 2 + buttons_len))
        q_text = f" q{self.widget.scene.supersample}x{px}"
        pieces.append((q_text, len(q_text)))
        backend_text = f"  [{backend}]"
        pieces.append((backend_text, len(backend_text)))
        help_text = f"  ? help{hint}"
        pieces.append((help_text, len(help_text)))
        pieces.append((" ", 1))

        trailer, trailer_len, offsets = self._build_segment(pieces)
        left_len = 1 + _LEFT_WIDTH   # leading space; `base` itself is 0-width
        pad = max(self._cols - left_len - trailer_len, 0)
        if show_buttons:
            base_col = left_len + pad + offsets[elem_piece_idx] + 2
            self._elem_button_span = (self._rows - 1, base_col + elem_rel[0], base_col + elem_rel[1])
            self._geom_button_span = (self._rows - 1, base_col + geom_rel[0], base_col + geom_rel[1])

        seg = f"{base} {left}{' ' * pad}{trailer}"
        return seg + "\x1b[0m"

    # -- input ------------------------------------------------------------
    def _read(self, timeout: float) -> bytes:
        r, _, _ = select.select([self.fd_in], [], [], timeout)
        if not r:
            return b""
        try:
            return os.read(self.fd_in, 4096)
        except OSError:
            return b""

    def _input_pending(self) -> bool:
        """True if input is already waiting to be read (non-blocking peek)."""
        try:
            r, _, _ = select.select([self.fd_in], [], [], 0)
            return bool(r)
        except (OSError, ValueError):
            return False

    def _span_hit(self, span, col: int, row: int) -> bool:
        """True if (col, row) lands on *span*, with the same one-row tolerance
        _in_status_zone gives the rest of that zone."""
        if span is None or not self._in_status_zone(row):
            return False
        _btn_row, col_start, col_end = span
        return col_start <= col < col_end

    def _dispatch(self, events) -> bool:
        """Apply input events; return True if anything visible changed and the
        frame should be redrawn."""
        changed = False
        for ev in events:
            if self._mode in ("periodic_table", "geometry_picker"):
                # Clicking the pill that opened the current picker closes it
                # again -- a normal toggle button, not a one-way switch.
                # Clicking the OTHER pill instead switches straight to it:
                # close whichever is open, open the one just clicked.
                if isinstance(ev, _input.MouseEvent) and ev.action == "down":
                    col, row = self._event_cell(ev)
                    if self._span_hit(self._elem_button_span, col, row):
                        was_pt = self._mode == "periodic_table"
                        if was_pt:
                            self._close_periodic_table(pick=None)
                        else:
                            self._close_geometry_picker(None)
                            self._open_periodic_table()
                        changed = True
                        continue
                    if self._span_hit(self._geom_button_span, col, row):
                        was_geom = self._mode == "geometry_picker"
                        if was_geom:
                            self._close_geometry_picker(None)
                        else:
                            self._close_periodic_table(pick=None)
                            self._open_geometry_picker()
                        changed = True
                        continue
                handler = (self._handle_pt_event if self._mode == "periodic_table"
                           else self._handle_geom_event)
                if handler(ev):
                    changed = True
                continue
            if self._mode != "normal":
                # While the save prompt is up, keystrokes drive it and mouse
                # events are swallowed (no accidental rotate mid-save).
                if isinstance(ev, _input.KeyEvent) and self._handle_prompt_key(ev.key):
                    changed = True
                continue
            if isinstance(ev, _input.MouseEvent):
                if ev.action == "down":
                    col, row = self._event_cell(ev)
                    self._status_zone_press = self._in_status_zone(row)
                    if self._status_zone_press:
                        # A click on the element pill opens the periodic table,
                        # on the geometry pill opens the geometry picker; any
                        # other click in this zone (a near-miss, or elsewhere on
                        # the status bar) is swallowed -- never forwarded to the
                        # 3D viewport.
                        if self._span_hit(self._elem_button_span, col, row):
                            self._open_periodic_table()
                            changed = True
                        elif self._span_hit(self._geom_button_span, col, row):
                            self._open_geometry_picker()
                            changed = True
                        continue
                elif ev.action in ("drag", "up"):
                    if self._status_zone_press:
                        if ev.action == "up":
                            self._status_zone_press = False
                        continue
                elif ev.action in ("move", "scroll"):
                    _col, row = self._event_cell(ev)
                    if self._in_status_zone(row):
                        continue
            if isinstance(ev, _input.KeyEvent) and ev.key in self._driver_keys:
                if self._driver_key(ev.key):
                    changed = True
            else:
                if self.widget.handle_event(ev):
                    self._last_interact = time.time()
                    self._msg = ""          # a fresh interaction clears "saved …"
                    changed = True
        return changed

    def _driver_key(self, key: str) -> bool:
        """Handle a driver-level key; return True if it changed the view."""
        if key == "escape" and self.editable and self.widget.dirty:
            # Unsaved changes: ask before quitting. 'q' and Ctrl-C stay
            # immediate quits (deliberate force-quit / emergency paths).
            self._mode = "quit_confirm"
            return True
        elif key in ("q", "escape", "\x03"):
            self._running = False
            return False
        elif key == "a":
            if self.editable:
                self.widget.set_append_mode(not self.widget.append_mode)
                self._msg = ""
                # append owns no pointer shape: switching here from delete or
                # measure must drop theirs so it doesn't linger (no-op if
                # nothing is pushed).
                self._pop_pointer()
            else:
                self.autospin = not self.autospin        # classic binding
        elif key == "m":
            # measuring is read-only-safe, so 'm' works without --edit too
            self.widget.set_measure_mode(not self.widget.measure_mode)
            self._msg = ""
            if self.widget.measure_mode:
                self._push_pointer("cell")               # precision plus-cross
            else:
                self._pop_pointer()
        elif key == "x" and self.editable:
            self.widget.set_delete_mode(not self.widget.delete_mode)
            self._msg = ""
            if self.widget.delete_mode:
                self._push_pointer("crosshair")
            else:
                self._pop_pointer()
        elif key == "o" and self.editable:
            self.autospin = not self.autospin            # relocated while editing
        elif key == "s" and self.editable:
            self._open_save_prompt()
        elif key == "u" and self.editable:
            return self.widget.undo()
        elif key == "c" and self.editable:
            # starts the animation; the run loop ticks it frame by frame
            return self.widget.start_cleanup()
        elif key == "?":
            self._show_help = not self._show_help
            if not self._show_help:
                kitty.write_bytes(_CLEAR, self.fd_out)
        elif key == "d":
            self.style.depth_cue = 0.0 if self.style.depth_cue > 0 else 0.55
            self._last_interact = time.time()
        elif key == "g":
            self._max_ss = 3 if self._max_ss == 2 else 2
            self._last_interact = time.time()
        elif key == "t":
            self.style.transparent = not self.style.transparent
            kitty.write_bytes(_CLEAR, self.fd_out)
            self._last_interact = time.time()
        elif key in ("n", "p") and len(self.frames) > 1:
            self.frame_index = (self.frame_index + (1 if key == "n" else -1)) % len(self.frames)
            self.widget.set_molecule(self.frames[self.frame_index])
            self._last_interact = time.time()
        else:
            return False
        return True

    # -- save prompt ------------------------------------------------------
    def _default_save_path(self) -> str:
        if self.source_path:
            return self.source_path
        name = (self.widget.molecule.name or "molecule").strip() or "molecule"
        return f"{name}.xyz"

    def _open_save_prompt(self) -> None:
        self._mode = "save_input"
        self._input_buf = self._default_save_path()
        self._msg = ""

    def _handle_prompt_key(self, key: str) -> bool:
        """Drive the modal save prompt. Returns True if the display changed."""
        if self._mode == "save_input":
            if key == "enter":
                path = self._input_buf.strip()
                if not path:
                    return False
                if os.path.exists(path):
                    self._mode = "save_confirm"   # ask before clobbering
                else:
                    self._do_save(path)
                return True
            if key in ("escape", "\x03"):
                self._mode = "normal"
                self._msg = ""
                # bailing out of the filename prompt cancels a pending
                # quit-after-save entirely: a cancelled save must never fall
                # through to "quit anyway, discarding changes".
                self._quit_after_save = False
                return True
            if key == "backspace":
                self._input_buf = self._input_buf[:-1]
                return True
            if len(key) == 1 and key.isprintable():
                self._input_buf += key
                return True
            return False
        if self._mode == "save_confirm":
            if key in ("y", "Y", "enter"):
                self._do_save(self._input_buf.strip())
                return True
            if key in ("n", "N", "escape", "\x03"):
                self._mode = "save_input"       # back to editing the name
                return True
            return False
        if self._mode == "quit_confirm":
            if key in ("y", "Y", "enter"):
                self._quit_after_save = True    # a successful save then quits
                self._open_save_prompt()
                return True
            if key in ("n", "N"):
                self._running = False           # quit without saving
                return True
            if key in ("escape", "\x03"):
                self._mode = "normal"           # cancel the quit, keep working
                return True
            return False
        return False

    def _do_save(self, path: str) -> None:
        from .parsers import save
        try:
            save(self.widget.molecule, path)
        except (OSError, ValueError) as e:
            self._msg = f"save failed: {e}"
            self._quit_after_save = False       # a failed save stays running
        else:
            self.source_path = path
            self.widget.mark_saved()
            self._msg = f"saved {os.path.basename(path)}"
            if self._quit_after_save:           # ESC-quit routed through save
                self._quit_after_save = False
                self._running = False
        self._mode = "normal"

    # -- main loop --------------------------------------------------------
    def run(self):
        if not os.isatty(self.fd_out):
            raise RuntimeError("vimol.Viewer requires a terminal on stdout")
        self._enter()
        self._running = True
        try:
            self._update_geometry()
            self._draw()
            frame_dt = 1.0 / self.target_fps
            while self._running:
                data = self._read(frame_dt)
                changed = self._dispatch(self.decoder.feed(data) if data
                                         else self.decoder.flush())
                if self._update_geometry():
                    kitty.write_bytes(_CLEAR, self.fd_out)
                    changed = True
                if self.autospin:
                    self.widget.scene.camera.orbit(1.4, 0)  # ~0.014 rad/frame
                    self._last_interact = time.time()
                    changed = True
                if self.widget.cleanup_active:
                    # animate the 'c' relaxation: one tick per frame, kept in
                    # fast (non-supersampled) mode while the atoms settle.
                    self.widget.cleanup_tick()
                    self._last_interact = time.time()
                    changed = True
                if changed:
                    self._draw()
                elif self._target_ss() != self._drawn_ss and not self._input_pending():
                    # Settle to a crisp, supersampled frame once the view stops
                    # moving -- but ONLY in a genuine lull with nothing queued.
                    # The high-quality downsample is a heavy synchronous step
                    # (~0.2s at full screen); running it while a keypress or
                    # mouse-move is waiting would stall that input behind it.
                    self._draw()
        finally:
            self._exit()
