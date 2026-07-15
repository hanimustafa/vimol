"""Full-screen interactive viewer — a thin driver around MoleculeWidget.

All the interaction logic lives in :class:`mviewer.widget.MoleculeWidget` and
input decoding in :class:`mviewer.input.InputDecoder`. This class only does the
terminal-owning parts: raw mode, the alternate screen, enabling mouse
reporting, the render/input loop, and a status bar. Embedders who want to
capture the mouse in their own region should use the widget + decoder directly
(see examples/embed_demo.py) rather than this driver.
"""
from __future__ import annotations

import os
import select
import time
from typing import List, Optional

from .molecule import Molecule
from .render import Style
from .widget import MoleculeWidget, REPRESENTATIONS
from .bonds import ensure_bonds
from . import kitty
from . import input as _input
from . import elements
from . import templates

# ANSI / terminal control -------------------------------------------------
_ALT_SCREEN_ON = b"\x1b[?1049h"
_ALT_SCREEN_OFF = b"\x1b[?1049l"
_HIDE_CURSOR = b"\x1b[?25l"
_SHOW_CURSOR = b"\x1b[?25h"
_CLEAR = b"\x1b[2J"
_HOME = b"\x1b[H"

_HELP_HEAD = [
    "  mviewer — terminal molecular viewer",
    "",
    "  Mouse drag ......... rotate            Wheel / + - ........ zoom",
    "  Right / mid drag ... pan               [ / ] .............. roll",
    "  Hover .............. identify atom      Arrows / h j k l ... rotate",
    "  1 2 3 4 ............ ball / space / licorice / wire",
]
# Shown only when editing is disabled (the classic bindings).
_HELP_VIEW = [
    "  s .................. cycle style       a .................. autospin",
]
# Shown only when editing is enabled (--edit).
_HELP_EDIT = [
    "  a .................. append (edit)     o .................. autospin",
    "     click atom -> grow C · click empty space -> new methane",
    "  s .................. save              u .................. undo",
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
_BASE_DRIVER_KEYS = {"q", "escape", "a", "?", "d", "g", "t", "n", "p", "\x03"}
# Extra keys claimed only when editing is enabled.
_EDIT_DRIVER_KEYS = {"s", "u", "o"}


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
        # modal save prompt: "normal" | "save_input" | "save_confirm"
        self._mode = "normal"
        self._input_buf = ""
        self._msg = ""                   # transient status message (e.g. "saved foo.xyz")
        self._last_interact = 0.0
        self._cols = self._rows = 0
        self._img_cols = self._img_rows = 1
        self._old_termios = None
        self._geometry_established = False   # True once the real (not placeholder) size is known

    # -- terminal lifecycle ----------------------------------------------
    def _enter(self):
        import termios
        import tty
        self._old_termios = termios.tcgetattr(self.fd_in) if os.isatty(self.fd_in) else None
        if self._old_termios is not None:
            tty.setraw(self.fd_in)
        kitty.write_bytes(_ALT_SCREEN_ON + _HIDE_CURSOR + _CLEAR, self.fd_out)
        kitty.write_bytes(_input.enable_mouse(pixel=True, hover=self.widget.picking), self.fd_out)
        # probe whether the terminal actually reports pixel coordinates
        if self._old_termios is not None:
            self.decoder.pixel = _input.supports_pixel_mouse(self.fd_in, self.fd_out)

    def _exit(self):
        import termios
        cleanup = kitty.delete_image(self._img_id)
        kitty.write_bytes(_input.disable_mouse(pixel=True) + cleanup
                          + _SHOW_CURSOR + _ALT_SCREEN_OFF, self.fd_out)
        if self._old_termios is not None:
            termios.tcsetattr(self.fd_in, termios.TCSADRAIN, self._old_termios)

    # -- geometry ---------------------------------------------------------
    def _update_geometry(self) -> bool:
        cols, rows, xpx, ypx = kitty.terminal_size_px(self.fd_out)
        cw, ch = kitty.cell_size_px(self.fd_out)
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
                                  cols=self._img_cols, rows=self._img_rows, move_cursor=False)
        out = bytearray()
        out += _HOME + data
        out += b"\x1b[%d;1H\x1b[2K" % self._rows
        out += self._status_bar().encode("utf-8", "replace")
        kitty.write_bytes(bytes(out), self.fd_out)
        if self._show_help:
            self._draw_help()

    def _draw_help(self):
        out = bytearray()
        for k, line in enumerate(_help_lines(self.editable)):
            out += b"\x1b[%d;3H" % (2 + k)
            out += b"\x1b[48;2;20;22;30m\x1b[38;2;220;220;230m"
            out += (" " + line.ljust(58)).encode()
            out += b"\x1b[0m"
        kitty.write_bytes(bytes(out), self.fd_out)

    @staticmethod
    def _pill(label: str, bg, fg=None) -> str:
        """A padded, bold, reverse-video 'button' -- a clickable-looking pill.

        Clicking these isn't wired yet: this commit only makes them *look* like
        buttons (colored, pressable). A later commit turns a click on one into a
        change of the active element / geometry.
        """
        r, g, b = (int(c * 255) if c <= 1 else int(c) for c in bg)
        if fg is None:
            lum = 0.299 * r + 0.587 * g + 0.114 * b       # pick readable text
            fg = (10, 12, 14) if lum > 140 else (245, 246, 250)
        fr, fg_, fb = fg
        return (f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fr};{fg_};{fb}m\x1b[1m"
                f" {label} \x1b[22m\x1b[0m")

    def _edit_buttons(self) -> str:
        """'adding [ C ] [ tetrahedral ]' with each token as a colored button."""
        elem = self.widget.build_element
        geom = templates.default_template(elem).geometry
        elem_btn = self._pill(elem, elements.element_color(elem))
        geom_btn = self._pill(geom, (0.17, 0.71, 0.63))       # teal accent
        return f"\x1b[38;2;150;155;170madding\x1b[0m {elem_btn} {geom_btn}"

    def _status_bar(self) -> str:
        if self._mode == "save_input":
            body = f" Save to: {self._input_buf}█   Enter save · Esc cancel "
            return f"\x1b[48;2;44;40;30m\x1b[38;2;240;236;220m{body}\x1b[0m"
        if self._mode == "save_confirm":
            name = os.path.basename(self._input_buf.strip())
            body = f" {name} exists — replace? (y/n) "
            return f"\x1b[48;2;60;30;30m\x1b[38;2;250;230;230m{body}\x1b[0m"
        mol = self.widget.molecule
        hov = self.widget.atom_info(self.widget.hovered)
        left = hov if hov else (self._msg or
            f"{(mol.name or 'molecule')[:22]}  {mol.formula()}  {mol.n_atoms} atoms")
        rep = self.style.representation
        frame = f"  {self.frame_index+1}/{len(self.frames)}" if len(self.frames) > 1 else ""
        spin = " ⟳" if self.autospin else ""
        px = " px" if self.decoder.pixel else ""
        backend = "gpu" if self.widget.scene.backend == "gl" else "cpu"
        base = "\x1b[48;2;30;33;44m\x1b[38;2;230;232;240m"
        # editing-only status: [MODIFIED] flag, APPEND indicator, and the
        # (button-styled) active build element / geometry.
        mod = " [MODIFIED]" if (self.editable and self.widget.dirty) else ""
        hint = "  s save  q quit" if self.editable else "  q quit"
        buttons = ""
        if self.editable and self.widget.append_mode:
            buttons = f"  {self._edit_buttons()}{base}"
        append = f"  {base}\x1b[1m✎APPEND\x1b[22m" if (self.editable and self.widget.append_mode) else ""
        seg = f"{base} {left}  [{rep}]{frame}{spin}{append}{mod}{buttons} q{self.widget.scene.supersample}x{px}  [{backend}]  ? help{hint} "
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

    def _dispatch(self, events) -> bool:
        """Apply input events; return True if anything visible changed and the
        frame should be redrawn."""
        changed = False
        for ev in events:
            if self._mode != "normal":
                # While the save prompt is up, keystrokes drive it and mouse
                # events are swallowed (no accidental rotate mid-save).
                if isinstance(ev, _input.KeyEvent) and self._handle_prompt_key(ev.key):
                    changed = True
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
        if key in ("q", "escape", "\x03"):
            self._running = False
            return False
        elif key == "a":
            if self.editable:
                self.widget.set_append_mode(not self.widget.append_mode)
                self._msg = ""
            else:
                self.autospin = not self.autospin        # classic binding
        elif key == "o" and self.editable:
            self.autospin = not self.autospin            # relocated while editing
        elif key == "s" and self.editable:
            self._open_save_prompt()
        elif key == "u" and self.editable:
            return self.widget.undo()
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
        return False

    def _do_save(self, path: str) -> None:
        from .parsers import save
        try:
            save(self.widget.molecule, path)
        except (OSError, ValueError) as e:
            self._msg = f"save failed: {e}"
        else:
            self.source_path = path
            self.widget.mark_saved()
            self._msg = f"saved {os.path.basename(path)}"
        self._mode = "normal"

    # -- main loop --------------------------------------------------------
    def run(self):
        if not os.isatty(self.fd_out):
            raise RuntimeError("mviewer.Viewer requires a terminal on stdout")
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
