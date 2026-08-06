"""Render what vimol writes to a terminal, without a terminal.

vimol paints its molecule with the Kitty graphics protocol, so the pixels
travel as escape codes rather than as text cells. That is what puts the
ordinary terminal recorders out of reach: asciinema captures the bytes
faithfully but ``agg`` renders only cells, and VHS's built-in emulator has no
graphics protocol at all.

So this module renders the frames itself, out of the bytes the program really
emits. A :class:`Session` drives a real :class:`vimol.viewer.Viewer` with its
output pointed at a file instead of a tty and feeds it synthetic input events;
a :class:`Terminal` replays the resulting stream -- ``pyte`` for the text
grid, the Kitty graphics blocks parsed here -- and :class:`Canvas` rasterizes
the result with Pillow.

Nothing in here is imported by vimol itself. It needs ``pyte`` and ``Pillow``,
neither of which is a runtime dependency:

    pip install pyte Pillow
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyte
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

# -- fonts -----------------------------------------------------------------

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_REGULAR, FONT_BOLD = 0, 1

# Menlo carries all but three of the characters vimol can put on a status bar
# or in a panel: the angle sign, the spinner and the copy control live in
# these instead. Without a fallback each of them draws as a .notdef box, which
# is easy to miss because a box is exactly what a box-drawing character should
# look like.
FALLBACK_PATHS = (
    "/System/Library/Fonts/Supplemental/STIXGeneral.otf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)


class Font:
    """A monospace face at one size, with a cache of cell-sized glyph masks.

    Glyphs are drawn cell by cell rather than run by run on purpose: Menlo's
    advance width is fractional, so a hundred characters written as one string
    would drift most of a cell away from the grid the image is placed on.
    """

    def __init__(self, size: int, path: str = FONT_PATH):
        self.regular = ImageFont.truetype(path, size, index=FONT_REGULAR)
        self.bold = ImageFont.truetype(path, size, index=FONT_BOLD)
        self.fallbacks = []
        for fb in FALLBACK_PATHS:
            try:
                self.fallbacks.append(ImageFont.truetype(fb, size))
            except OSError:
                pass
        self.cell_w = int(round(self.regular.getlength("M")))
        ascent, descent = self.regular.getmetrics()
        self.cell_h = int(round((ascent + descent) * 1.08))
        self.baseline = ascent + (self.cell_h - ascent - descent) // 2
        self._masks: Dict[Tuple[str, bool], Image.Image] = {}
        self._notdef = {id(f): self._raw(f, "") for f in
                        [self.regular, self.bold] + self.fallbacks}

    @staticmethod
    def _raw(face, char: str) -> bytes:
        """The glyph's bitmap, as bytes -- comparing this against the face's
        .notdef is the only reliable way to ask FreeType whether a font
        actually has a character. A missing glyph still returns a mask, and
        a non-empty one."""
        try:
            m = face.getmask(char, mode="L")
        except Exception:                              # noqa: BLE001
            return b""
        return bytes(m) if m.size != (0, 0) else b""

    def _face_for(self, char: str, bold: bool):
        primary = self.bold if bold else self.regular
        raw = self._raw(primary, char)
        if raw and raw != self._notdef[id(primary)]:
            return primary
        for fb in self.fallbacks:
            raw = self._raw(fb, char)
            if raw and raw != self._notdef[id(fb)]:
                return fb
        return primary

    def mask(self, char: str, bold: bool) -> Optional[Image.Image]:
        """An L-mode image of *char*, the size of one cell, or None for a
        character with nothing to draw."""
        key = (char, bold)
        if key in self._masks:
            return self._masks[key]
        tile = None
        if char and not char.isspace():
            face = self._face_for(char, bold)
            tile = Image.new("L", (self.cell_w, self.cell_h), 0)
            d = ImageDraw.Draw(tile)
            x = (self.cell_w - face.getlength(char)) / 2
            d.text((x, self.baseline), char, font=face, fill=255, anchor="ls")
            if not tile.getbbox():          # nothing came out: no such glyph
                tile = None
        self._masks[key] = tile
        return tile


# -- the kitty graphics protocol -------------------------------------------

_APC = re.compile(rb"\x1b_G(.*?)\x1b\\", re.S)


@dataclass
class Placement:
    """One image on screen: where it sits, and how big it is told to be."""
    image: Image.Image
    col: int
    row: int
    cols: int
    rows: int
    z: int


def _controls(blob: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in blob.split(b","):
        if b"=" in part:
            k, v = part.split(b"=", 1)
            out[k.decode()] = v.decode()
    return out


class _ImageAssembler:
    """Reassembles the chunked ``m=1`` payloads of one transmission."""

    def __init__(self):
        self.ctrl: Dict[str, str] = {}
        self.payload = bytearray()

    def add(self, ctrl: Dict[str, str], payload: bytes) -> bool:
        if ctrl.get("a") or not self.ctrl:
            self.ctrl = ctrl
            self.payload = bytearray()
        self.payload += payload
        return ctrl.get("m", "0") == "0"        # True when the last chunk lands

    def decode(self) -> Image.Image:
        raw = base64.standard_b64decode(bytes(self.payload))
        if self.ctrl.get("o") == "z":
            raw = zlib.decompress(raw)
        w, h = int(self.ctrl["s"]), int(self.ctrl["v"])
        depth = 4 if self.ctrl.get("f") == "32" else 3
        arr = np.frombuffer(raw, dtype=np.uint8)[:w * h * depth]
        arr = arr.reshape(h, w, depth)
        return Image.fromarray(arr, "RGBA" if depth == 4 else "RGB")


# -- the terminal ----------------------------------------------------------

class Terminal:
    """A text grid plus a set of images: everything vimol puts on a screen.

    The text grid is ``pyte``'s -- a spike over a real frame found vimol's
    output to use only CUP, EL and SGR 0/1/22/38;2/48;2, all of which pyte
    already implements correctly. Only the graphics blocks are parsed here.
    """

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        self.placements: Dict[Tuple[str, str], Placement] = {}
        self._assembler = _ImageAssembler()

    def reset(self) -> None:
        self.screen.reset()
        self.placements.clear()

    def feed(self, data: bytes) -> None:
        pos = 0
        for m in _APC.finditer(data):
            self._feed_text(data[pos:m.start()])
            self._graphics(m.group(1))
            pos = m.end()
        self._feed_text(data[pos:])

    def _feed_text(self, chunk: bytes) -> None:
        if chunk:
            self.stream.feed(chunk.decode("utf-8", "replace"))

    def _graphics(self, blob: bytes) -> None:
        head, _, payload = blob.partition(b";")
        ctrl = _controls(head)
        if ctrl.get("a") == "d":
            key = (ctrl.get("i", ""), ctrl.get("p", ctrl.get("i", "")))
            self.placements.pop(key, None)
            if ctrl.get("d") in ("I", "A"):
                for k in [k for k in self.placements if k[0] == ctrl.get("i", "")]:
                    del self.placements[k]
            return
        if not self._assembler.add(ctrl, payload):
            return
        c = self._assembler.ctrl
        image = self._assembler.decode()
        key = (c.get("i", "1"), c.get("p", c.get("i", "1")))
        # C=1 leaves the cursor alone, which is what vimol uses; either way the
        # placement's own corner is wherever the cursor was when it arrived.
        self.placements[key] = Placement(
            image=image,
            col=self.screen.cursor.x, row=self.screen.cursor.y,
            cols=int(c.get("c", 0)), rows=int(c.get("r", 0)),
            z=int(c.get("z", 0)),
        )


# -- rasterizing -----------------------------------------------------------

def _rgb(value, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """pyte reports 24-bit colours as six hex digits and names the rest."""
    if value in (None, "default"):
        return default
    named = {"black": (0, 0, 0), "red": (205, 66, 66), "green": (100, 190, 110),
             "brown": (190, 150, 60), "blue": (90, 140, 220),
             "magenta": (190, 110, 200), "cyan": (60, 200, 180),
             "white": (230, 232, 240)}
    if value in named:
        return named[value]
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except (ValueError, IndexError):
        return default


@dataclass
class Theme:
    background: Tuple[int, int, int] = (22, 24, 32)
    foreground: Tuple[int, int, int] = (222, 226, 236)
    prompt: Tuple[int, int, int] = (110, 200, 170)
    path: Tuple[int, int, int] = (120, 160, 230)
    cursor: Tuple[int, int, int] = (210, 214, 226)
    pointer: Tuple[int, int, int] = (255, 255, 255)
    padding: int = 14
    radius: int = 10


class Canvas:
    """Rasterizes one :class:`Terminal` state into an image."""

    def __init__(self, font: Font, theme: Theme = Theme()):
        self.font, self.theme = font, theme

    def size(self, cols: int, rows: int) -> Tuple[int, int]:
        pad = self.theme.padding
        return (cols * self.font.cell_w + 2 * pad,
                rows * self.font.cell_h + 2 * pad)

    def render(self, term: Terminal, cursor: Optional[Tuple[int, int]] = None,
               pointer: Optional[Tuple[float, float]] = None,
               click: float = 0.0, badge: Optional[str] = None) -> Image.Image:
        f, th = self.font, self.theme
        pad = th.padding
        w, h = self.size(term.cols, term.rows)
        img = Image.new("RGB", (w, h), th.background)

        # Rounded corners: draw the body onto a rounded mask so the panel reads
        # as a window rather than as a screenshot of one.
        body = Image.new("RGBA", (w - 2 * pad, h - 2 * pad), th.background + (255,))

        for p in sorted(term.placements.values(), key=lambda p: p.z):
            # c/r ask the terminal to scale the image into that many cells;
            # with neither, it is drawn at its own pixel size.
            target = (p.cols * f.cell_w if p.cols else p.image.width,
                      p.rows * f.cell_h if p.rows else p.image.height)
            layer = p.image.convert("RGBA")
            if layer.size != target:
                layer = layer.resize(target, Image.LANCZOS)
            body.alpha_composite(layer, (p.col * f.cell_w, p.row * f.cell_h))

        draw = ImageDraw.Draw(body)
        for y in range(term.rows):
            line = term.screen.buffer[y]
            for x in range(term.cols):
                cell = line[x]
                bg = _rgb(cell.bg, None) if cell.bg != "default" else None
                if cell.reverse:
                    bg = _rgb(cell.fg, th.foreground)
                if bg is not None:
                    draw.rectangle([x * f.cell_w, y * f.cell_h,
                                    (x + 1) * f.cell_w - 1, (y + 1) * f.cell_h - 1],
                                   fill=bg + (255,))
                mask = f.mask(cell.data, cell.bold)
                if mask is not None:
                    fg = _rgb(cell.fg, th.foreground)
                    if cell.reverse:
                        fg = _rgb(cell.bg, th.background)
                    body.paste(fg, (x * f.cell_w, y * f.cell_h), mask)

        if cursor is not None:
            cx, cy = cursor
            draw.rectangle([cx * f.cell_w, cy * f.cell_h + 2,
                            (cx + 1) * f.cell_w - 1, (cy + 1) * f.cell_h - 2],
                           fill=th.cursor + (220,))
        if pointer is not None:
            self._pointer(body, pointer, click)
        if badge:
            self._badge(body, badge)

        rounded = Image.new("L", body.size, 0)
        ImageDraw.Draw(rounded).rounded_rectangle(
            [0, 0, body.size[0] - 1, body.size[1] - 1], th.radius, fill=255)
        img.paste(body.convert("RGB"), (pad, pad), rounded)
        return img

    def _badge(self, body: Image.Image, label: str) -> None:
        """A keycap near the bottom of the frame, so a viewer can see which
        key produced the change they are watching."""
        f = self.font
        pad_x, pad_y = f.cell_w, f.cell_h // 3
        w = int(len(label) * f.cell_w + 2 * pad_x)
        h = f.cell_h + 2 * pad_y
        x = (body.size[0] - w) // 2
        y = body.size[1] - f.cell_h * 3 - h
        cap = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(cap)
        d.rounded_rectangle([0, 0, w - 1, h - 1], 6, fill=(238, 240, 248, 242),
                            outline=(150, 154, 168, 255), width=1)
        for i, ch in enumerate(label):
            mask = f.mask(ch, True)
            if mask is not None:
                cap.paste((28, 30, 38), (pad_x + i * f.cell_w, pad_y), mask)
        body.alpha_composite(cap, (x, y))

    def _pointer(self, body: Image.Image, at: Tuple[float, float],
                 click: float) -> None:
        """The arrow, and a ring that blooms out of it on a click.

        Drawn at four times the size and scaled down, because a plain polygon
        at this size has visibly jagged edges and the eye goes straight to the
        pointer.
        """
        x, y = at
        s = 4
        size = 26 * s
        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if click > 0:
            r = int((6 + 12 * click) * s)
            alpha = int(200 * (1 - click))
            d.ellipse([size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r],
                      outline=(255, 255, 255, alpha), width=2 * s)
        arrow = [(0, 0), (0, 15), (4, 11), (7, 17), (10, 15), (7, 9), (12, 9)]
        pts = [(size // 2 + px * s, size // 2 + py * s) for px, py in arrow]
        d.polygon(pts, fill=(20, 20, 24, 255))
        inner = [(1.4, 2.2), (1.4, 12.2), (4.3, 9.4), (7.1, 14.8),
                 (8.4, 14.0), (5.6, 8.6), (9.0, 8.6)]
        d.polygon([(size // 2 + px * s, size // 2 + py * s) for px, py in inner],
                  fill=(255, 255, 255, 255))
        layer = layer.resize((size // s, size // s), Image.LANCZOS)
        body.alpha_composite(layer, (int(x) - 13, int(y) - 13))


# -- driving a viewer ------------------------------------------------------

class Session:
    """A :class:`vimol.viewer.Viewer` wired to a file instead of a terminal.

    The viewer's resolution controllers adapt to measured frame times and to
    frame-delivery acknowledgements that a file descriptor will never send.
    Left alone they would record the blur ramp instead of the picture, so
    :meth:`_pin` fixes them at full quality.
    """

    def __init__(self, cols: int, rows: int, font: Font, *, structures=None,
                 molecule=None, source_path: Optional[str] = None,
                 style: Optional[str] = None, editable: bool = True,
                 backend: str = "cpu"):
        os.environ["COLUMNS"], os.environ["LINES"] = str(cols), str(rows)
        os.environ.setdefault("VIMOL_THEME", "dark")
        from vimol.render import Style
        from vimol.viewer import Viewer

        self.cols, self.rows, self.font = cols, rows, font
        self._sink = tempfile.NamedTemporaryFile(suffix=".vimol-stream",
                                                 delete=False)
        st = Style()
        st.transparent = True
        if style:
            st.representation = style
        self.viewer = Viewer(molecule if molecule is not None
                             else structures[0].molecule,
                             structures=structures, style=st,
                             fd_out=self._sink.fileno(), backend=backend,
                             source_path=source_path, editable=editable)
        self.terminal = Terminal(cols, rows)
        self._pin()
        self.viewer._update_geometry()

    def _pin(self) -> None:
        v = self.viewer
        v._cell_px = (float(self.font.cell_w), float(self.font.cell_h))
        v._paced = False                 # no fence acks are coming back
        v._interact_scale = 1.0
        v._idle_scale = 1.0
        v._max_ss = 2
        v._target_ss = lambda: 2         # always the settle-quality frame
        v._geometry_dirty = True

    # -- input ------------------------------------------------------------
    def key(self, name: str) -> None:
        from vimol import input as minput
        self.viewer._dispatch([minput.KeyEvent(name)])

    def screen_of(self, index: int) -> Tuple[float, float]:
        """Widget-local pixel of composite atom *index*, by the same
        orthographic projection :meth:`MoleculeWidget.pick` inverts. Clicks
        are aimed with this rather than with guessed coordinates."""
        w = self.viewer.widget
        mol = w.scene.structures.composite().molecule
        cam = w.scene.camera
        Wr, Hr = w.scene.render_size
        v = cam.view_positions(mol.positions)
        sx = Wr * 0.5 + cam.pan[0] + v[index, 0] * cam.zoom
        sy = Hr * 0.5 - cam.pan[1] - v[index, 1] * cam.zoom
        return sx * w.scene.width / Wr, sy * w.scene.height / Hr

    def to_screen(self, widget_px: Tuple[float, float]) -> Tuple[float, float]:
        """Widget-local pixel -> whole-terminal pixel."""
        ox, oy = self.viewer._img_origin_px
        return widget_px[0] + ox, widget_px[1] + oy

    def cell_px(self, col: float, row: float) -> Tuple[float, float]:
        return ((col + 0.5) * self.font.cell_w, (row + 0.5) * self.font.cell_h)

    def mouse(self, action: str, at: Tuple[float, float], button: int = 0,
              **kw) -> None:
        from vimol import input as minput
        self.viewer._dispatch([minput.MouseEvent(
            action, x=float(at[0]), y=float(at[1]), button=button,
            pixel=True, **kw)])

    def click(self, at: Tuple[float, float], **kw) -> None:
        self.mouse("down", at, **kw)
        self.mouse("up", at, **kw)

    def drag(self, frm: Tuple[float, float], to: Tuple[float, float],
             steps: int = 8):
        """Yields after each motion step so a caller can capture the frames."""
        self.mouse("down", frm)
        yield
        for i in range(1, steps + 1):
            t = i / steps
            self.mouse("drag", (frm[0] + (to[0] - frm[0]) * t,
                                frm[1] + (to[1] - frm[1]) * t))
            yield
        self.mouse("up", to)
        yield

    # -- output -----------------------------------------------------------
    def capture(self) -> None:
        """Draw one frame and feed the bytes it wrote into the terminal."""
        f = self._sink
        f.seek(0)
        f.truncate()
        self.viewer._draw()
        f.flush()
        os.lseek(f.fileno(), 0, os.SEEK_SET)
        data = b""
        while True:
            chunk = os.read(f.fileno(), 1 << 22)
            if not chunk:
                break
            data += chunk
        self.terminal.feed(data)

    def close(self) -> None:
        try:
            self._sink.close()
            os.unlink(self._sink.name)
        except OSError:
            pass
