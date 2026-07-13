"""Interactive terminal viewer widget.

This is the embeddable interactive surface. It puts the terminal into raw mode,
enables SGR mouse reporting, and runs a render/input loop. Rendering is
double-buffered across two Kitty image ids (draw the new frame, then delete the
previous one) so animation is flicker-free.

Embedding
---------
    from mviewer.viewer import Viewer
    Viewer(mol).run()

Or drive a single frame yourself via :class:`mviewer.Scene` and this module's
low-level terminal helpers.
"""
from __future__ import annotations

import os
import select
import sys
import time
from typing import List, Optional

import numpy as np

from .molecule import Molecule
from .render import Style
from .scene import Scene
from .bonds import ensure_bonds
from . import kitty

# ANSI / terminal control -------------------------------------------------
_ALT_SCREEN_ON = b"\x1b[?1049h"
_ALT_SCREEN_OFF = b"\x1b[?1049l"
_HIDE_CURSOR = b"\x1b[?25l"
_SHOW_CURSOR = b"\x1b[?25h"
_CLEAR = b"\x1b[2J"
_HOME = b"\x1b[H"
_MOUSE_ON = b"\x1b[?1000;1002;1006h"   # button + drag-motion + SGR coords
_MOUSE_OFF = b"\x1b[?1000;1002;1006l"

REPRESENTATIONS = ["ball_and_stick", "spacefill", "licorice", "wireframe"]

HELP_LINES = [
    "  mviewer — terminal molecular viewer",
    "",
    "  Mouse drag ......... rotate            Wheel / + - ........ zoom",
    "  Right drag ......... pan               [ / ] .............. roll",
    "  Arrows / h j k l ... rotate            r .................. reset view",
    "  1 2 3 4 ............ ball / space / licorice / wire",
    "  s .................. cycle style       a .................. autospin",
    "  n / p .............. next/prev frame   d .................. depth cue",
    "  g .................. hi-quality        f .................. re-fit",
    "  ? .................. toggle help       q / Esc ............ quit",
]


class Viewer:
    def __init__(self, molecule: Molecule, frames: Optional[List[Molecule]] = None,
                 style: Optional[Style] = None, fd_in: int = 0, fd_out: int = 1,
                 autospin: bool = False, target_fps: float = 30.0):
        self.frames = frames or [molecule]
        for m in self.frames:
            ensure_bonds(m)
        self.frame_index = 0
        self.style = style or Style()
        self.fd_in = fd_in
        self.fd_out = fd_out
        self.autospin = autospin
        self.target_fps = target_fps

        self.scene = Scene(self.frames[0], 320, 240, style=self.style, supersample=1)
        self._max_ss = 2          # idle-quality supersample cap (toggle with 'g')
        self._img_id = 1          # double-buffer image ids toggle 1<->2
        self._prev_img_id = None
        self._running = False
        self._show_help = False
        self._interacting = False
        self._last_interact = 0.0
        self._dragging = False
        self._drag_button = 0
        self._last_mouse = (0, 0)
        self._cols = self._rows = 0
        self._cell_w = 9.0
        self._cell_h = 18.0
        self._status = ""

    # -- terminal lifecycle ----------------------------------------------
    def _enter(self):
        import termios
        import tty
        self._old_termios = termios.tcgetattr(self.fd_in) if os.isatty(self.fd_in) else None
        if self._old_termios is not None:
            tty.setraw(self.fd_in)
        kitty.write_bytes(_ALT_SCREEN_ON + _HIDE_CURSOR + _MOUSE_ON + _CLEAR, self.fd_out)

    def _exit(self):
        import termios
        kitty.write_bytes(kitty.clear_all_images() + _MOUSE_OFF + _SHOW_CURSOR + _ALT_SCREEN_OFF, self.fd_out)
        if self._old_termios is not None:
            termios.tcsetattr(self.fd_in, termios.TCSADRAIN, self._old_termios)

    # -- geometry ---------------------------------------------------------
    def _update_geometry(self) -> bool:
        cols, rows, xpx, ypx = kitty.terminal_size_px(self.fd_out)
        cw, ch = kitty.cell_size_px(self.fd_out)
        changed = (cols, rows) != (self._cols, self._rows)
        self._cols, self._rows, self._cell_w, self._cell_h = cols, rows, cw, ch
        if changed:
            # reserve 1 row for the status bar
            img_rows = max(rows - 1, 1)
            img_cols = cols
            w = int(img_cols * cw)
            h = int(img_rows * ch)
            self._img_cols, self._img_rows = img_cols, img_rows
            self.scene.set_size(max(w, 16), max(h, 16))
        return changed

    # -- rendering --------------------------------------------------------
    def _draw(self):
        now = time.time()
        # adaptive quality: crude while interacting, crisp when idle
        idle = (now - self._last_interact) > 0.25 and not self.autospin
        want_ss = getattr(self, "_max_ss", 2) if idle else 1
        if self.scene.supersample != want_ss:
            self.scene.set_supersample(want_ss)

        img = self.scene.render()
        new_id = 2 if self._img_id == 1 else 1
        data = kitty.encode_image(
            img, image_id=new_id, placement_id=new_id,
            cols=self._img_cols, rows=self._img_rows, move_cursor=False,
        )
        out = bytearray()
        out += _HOME
        out += data
        # delete the previous frame AFTER the new one is drawn (no flash)
        if self._prev_img_id is not None:
            out += kitty.delete_image(self._prev_img_id)
        # status bar on the last row
        out += b"\x1b[%d;1H" % self._rows
        out += b"\x1b[2K"  # clear line
        out += self._status_bar().encode("utf-8", "replace")
        kitty.write_bytes(bytes(out), self.fd_out)
        self._prev_img_id = new_id
        self._img_id = new_id

        if self._show_help:
            self._draw_help()

    def _draw_help(self):
        out = bytearray()
        top = 2
        for k, line in enumerate(HELP_LINES):
            out += b"\x1b[%d;3H" % (top + k)
            out += b"\x1b[48;2;20;22;30m\x1b[38;2;220;220;230m"
            out += (" " + line.ljust(60)).encode()
            out += b"\x1b[0m"
        kitty.write_bytes(bytes(out), self.fd_out)

    def _status_bar(self) -> str:
        mol = self.scene.molecule
        rep = self.style.representation
        frame = f" frame {self.frame_index+1}/{len(self.frames)}" if len(self.frames) > 1 else ""
        spin = " ⟳" if self.autospin else ""
        q = f" q{self.scene.supersample}x"
        name = (mol.name or "molecule")[:24]
        base = f"\x1b[48;2;30;33;44m\x1b[38;2;230;232;240m {name}  {mol.formula()}  {mol.n_atoms} atoms  [{rep}]{frame}{spin}{q}  ? help  q quit "
        return base + "\x1b[0m"

    # -- input ------------------------------------------------------------
    def _read_input(self, timeout: float) -> bytes:
        r, _, _ = select.select([self.fd_in], [], [], timeout)
        if not r:
            return b""
        try:
            return os.read(self.fd_in, 4096)
        except OSError:
            return b""

    def _mark_interact(self):
        self._last_interact = time.time()

    def _handle(self, data: bytes) -> None:
        i = 0
        n = len(data)
        while i < n:
            b = data[i]
            if b == 0x1b:  # escape sequence
                consumed = self._handle_escape(data, i)
                if consumed == 0:
                    # bare ESC -> quit
                    self._running = False
                    return
                i += consumed
                continue
            ch = chr(b)
            self._handle_key(ch)
            i += 1

    def _handle_escape(self, data: bytes, i: int) -> int:
        n = len(data)
        if i + 1 >= n:
            return 0  # lone ESC
        if data[i + 1] == ord("["):
            # CSI sequence
            j = i + 2
            if j < n and data[j] == ord("<"):
                # SGR mouse: ESC [ < b ; x ; y (M|m)
                k = j + 1
                while k < n and data[k] not in (ord("M"), ord("m")):
                    k += 1
                if k >= n:
                    return n - i
                body = data[j + 1:k].decode("ascii", "ignore")
                final = chr(data[k])
                self._handle_mouse(body, final)
                return (k - i) + 1
            # arrow / other CSI: read until a final byte in @..~
            k = j
            while k < n and not (0x40 <= data[k] <= 0x7E):
                k += 1
            if k >= n:
                return n - i
            final = chr(data[k])
            self._handle_csi(final)
            return (k - i) + 1
        return 1  # unknown, skip ESC

    def _handle_csi(self, final: str):
        step = 8.0
        if final == "A":       # up
            self.scene.camera.orbit(0, -step); self._mark_interact()
        elif final == "B":     # down
            self.scene.camera.orbit(0, step); self._mark_interact()
        elif final == "C":     # right
            self.scene.camera.orbit(step, 0); self._mark_interact()
        elif final == "D":     # left
            self.scene.camera.orbit(-step, 0); self._mark_interact()

    def _handle_mouse(self, body: str, final: str):
        try:
            btn_s, x_s, y_s = body.split(";")
            btn = int(btn_s); x = int(x_s); y = int(y_s)
        except ValueError:
            return
        wheel = btn & 64
        button = btn & 3
        motion = btn & 32
        if wheel:
            if button == 0:
                self.scene.camera.zoom_by(1.12)
            else:
                self.scene.camera.zoom_by(1 / 1.12)
            self._mark_interact()
            return
        # convert cell coords to pixel-ish deltas
        if final == "M" and not motion:
            self._dragging = True
            self._drag_button = button
            self._last_mouse = (x, y)
        elif final == "m":
            self._dragging = False
        elif motion and self._dragging:
            lx, ly = self._last_mouse
            dx = (x - lx) * self._cell_w
            dy = (y - ly) * self._cell_h
            self._last_mouse = (x, y)
            shift = False  # SGR doesn't encode shift here reliably; use right button
            if self._drag_button == 2 or shift:
                self.scene.camera.pan_by(dx, -dy)
            else:
                self.scene.camera.orbit(dx, dy, speed=0.006)
            self._mark_interact()

    def _handle_key(self, ch: str):
        cam = self.scene.camera
        if ch in ("q", "\x03"):  # q or Ctrl-C
            self._running = False
        elif ch in ("h",):
            cam.orbit(-8, 0); self._mark_interact()
        elif ch in ("l",):
            cam.orbit(8, 0); self._mark_interact()
        elif ch in ("k",):
            cam.orbit(0, -8); self._mark_interact()
        elif ch in ("j",):
            cam.orbit(0, 8); self._mark_interact()
        elif ch in ("+", "="):
            cam.zoom_by(1.15); self._mark_interact()
        elif ch in ("-", "_"):
            cam.zoom_by(1 / 1.15); self._mark_interact()
        elif ch == "[":
            cam.roll(-0.15); self._mark_interact()
        elif ch == "]":
            cam.roll(0.15); self._mark_interact()
        elif ch == "r":
            cam.reset(); self.scene.fit(keep_orientation=True); self._mark_interact()
        elif ch == "f":
            self.scene.fit(keep_orientation=True); self._mark_interact()
        elif ch == "a":
            self.autospin = not self.autospin
        elif ch == "o":
            self.style.outline = not self.style.outline; self._mark_interact()
        elif ch == "d":
            self.style.depth_cue = 0.0 if self.style.depth_cue > 0 else 0.55; self._mark_interact()
        elif ch == "?":
            self._show_help = not self._show_help
            if not self._show_help:
                kitty.write_bytes(_CLEAR, self.fd_out)
        elif ch in ("1", "2", "3", "4"):
            self.style.representation = REPRESENTATIONS[int(ch) - 1]
            self.scene.fit(keep_orientation=True); self._mark_interact()
        elif ch == "s":
            cur = REPRESENTATIONS.index(self.style.representation)
            self.style.representation = REPRESENTATIONS[(cur + 1) % len(REPRESENTATIONS)]
            self.scene.fit(keep_orientation=True); self._mark_interact()
        elif ch in ("n", "p") and len(self.frames) > 1:
            self.frame_index = (self.frame_index + (1 if ch == "n" else -1)) % len(self.frames)
            rot = self.scene.camera.rotation.copy()
            self.scene.set_molecule(self.frames[self.frame_index])
            self.scene.camera.rotation = rot
            self._mark_interact()
        elif ch == "g":
            # cycle high-quality supersample cap (auto also raises it when idle)
            self._max_ss = 3 if getattr(self, "_max_ss", 2) == 2 else 2
            self._mark_interact()

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
                data = self._read_input(frame_dt)
                if data:
                    self._handle(data)
                if self._update_geometry():
                    kitty.write_bytes(_CLEAR, self.fd_out)
                if self.autospin:
                    self.scene.camera.orbit(1.1, 0)
                    self._mark_interact()
                self._draw()
        finally:
            self._exit()
