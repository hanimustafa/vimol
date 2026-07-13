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

# ANSI / terminal control -------------------------------------------------
_ALT_SCREEN_ON = b"\x1b[?1049h"
_ALT_SCREEN_OFF = b"\x1b[?1049l"
_HIDE_CURSOR = b"\x1b[?25l"
_SHOW_CURSOR = b"\x1b[?25h"
_CLEAR = b"\x1b[2J"
_HOME = b"\x1b[H"

HELP_LINES = [
    "  mviewer — terminal molecular viewer",
    "",
    "  Mouse drag ......... rotate            Wheel / + - ........ zoom",
    "  Right / mid drag ... pan               [ / ] .............. roll",
    "  Hover .............. identify atom      Arrows / h j k l ... rotate",
    "  1 2 3 4 ............ ball / space / licorice / wire",
    "  s .................. cycle style       a .................. autospin",
    "  n / p .............. next/prev frame   d .................. depth cue",
    "  t .................. transparent bg    g .................. hi-quality",
    "  f / r .............. re-fit / reset    ? .................. toggle help",
    "  q / Esc ............ quit",
]

_DRIVER_KEYS = {"q", "escape", "a", "?", "d", "g", "t", "n", "p", "\x03"}


class Viewer:
    def __init__(self, molecule: Molecule, frames: Optional[List[Molecule]] = None,
                 style: Optional[Style] = None, fd_in: int = 0, fd_out: int = 1,
                 autospin: bool = False, target_fps: float = 60.0, picking: bool = True,
                 transparent: bool = True):
        self.frames = frames or [molecule]
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
                                     supersample=1, picking=picking)
        self.decoder = _input.InputDecoder(pixel=False)
        self._max_ss = 2
        # single per-process image id, replaced in place each frame (this is
        # flicker-free in kitty and, unlike a double buffer, never lets a
        # transparent frame ghost the previous one through its cutout).
        self._img_id = kitty.unique_id_base() + 1

        self._running = False
        self._show_help = False
        self._last_interact = 0.0
        self._cols = self._rows = 0
        self._img_cols = self._img_rows = 1
        self._old_termios = None

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
            self.widget.set_pixel_size(max(w, 16), max(h, 16))
        return changed

    # -- rendering --------------------------------------------------------
    def _draw(self):
        now = time.time()
        idle = (now - self._last_interact) > 0.25 and not self.autospin
        want_ss = self._max_ss if idle else 1
        if self.widget.scene.supersample != want_ss:
            self.widget.scene.set_supersample(want_ss)

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
        for k, line in enumerate(HELP_LINES):
            out += b"\x1b[%d;3H" % (2 + k)
            out += b"\x1b[48;2;20;22;30m\x1b[38;2;220;220;230m"
            out += (" " + line.ljust(58)).encode()
            out += b"\x1b[0m"
        kitty.write_bytes(bytes(out), self.fd_out)

    def _status_bar(self) -> str:
        mol = self.widget.molecule
        hov = self.widget.atom_info(self.widget.hovered)
        left = hov if hov else f"{(mol.name or 'molecule')[:22]}  {mol.formula()}  {mol.n_atoms} atoms"
        rep = self.style.representation
        frame = f"  {self.frame_index+1}/{len(self.frames)}" if len(self.frames) > 1 else ""
        spin = " ⟳" if self.autospin else ""
        px = " px" if self.decoder.pixel else ""
        seg = f"\x1b[48;2;30;33;44m\x1b[38;2;230;232;240m {left}  [{rep}]{frame}{spin} q{self.widget.scene.supersample}x{px}  ? help  q quit "
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

    def _dispatch(self, events):
        for ev in events:
            if isinstance(ev, _input.KeyEvent) and ev.key in _DRIVER_KEYS:
                self._driver_key(ev.key)
            else:
                changed = self.widget.handle_event(ev)
                if changed:
                    self._last_interact = time.time()

    def _driver_key(self, key: str):
        if key in ("q", "escape", "\x03"):
            self._running = False
        elif key == "a":
            self.autospin = not self.autospin
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
                if data:
                    self._dispatch(self.decoder.feed(data))
                else:
                    self._dispatch(self.decoder.flush())
                if self._update_geometry():
                    kitty.write_bytes(_CLEAR, self.fd_out)
                if self.autospin:
                    self.widget.scene.camera.orbit(1.4, 0)  # ~0.014 rad/frame
                    self._last_interact = time.time()
                self._draw()
        finally:
            self._exit()
