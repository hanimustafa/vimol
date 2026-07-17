"""Kitty terminal graphics protocol encoder + terminal geometry helpers.

Reference: https://sw.kovidgoyal.net/kitty/graphics-protocol/

We transmit raw RGB/RGBA pixels (optionally zlib-compressed) in <=4096-byte
base64 chunks. The public surface is small on purpose so the module can be
lifted into other terminal apps unchanged.
"""
from __future__ import annotations

import base64
import os
import re
import sys
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_ESC = b"\x1b"
_GRAPHICS_START = b"\x1b_G"
_GRAPHICS_END = b"\x1b\\"
_CHUNK = 4096


# --------------------------------------------------------------------------
# Terminal geometry
# --------------------------------------------------------------------------
def terminal_size_px(fd: int = 1) -> Tuple[int, int, int, int]:
    """Return (cols, rows, width_px, height_px) for the terminal on *fd*.

    Falls back to environment / sane defaults when the ioctl is unavailable
    (e.g. output is not a tty). width/height px may be 0 if the terminal does
    not report pixel dimensions.
    """
    cols, rows, xpx, ypx = 80, 24, 0, 0
    try:
        import fcntl
        import struct
        import termios

        buf = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows, cols, xpx, ypx = struct.unpack("HHHH", buf)
    except Exception:
        cols = int(os.environ.get("COLUMNS", cols))
        rows = int(os.environ.get("LINES", rows))
    return cols, rows, xpx, ypx


def cell_size_px(fd: int = 1) -> Tuple[float, float]:
    """(cell_width_px, cell_height_px). Defaults to 9x18 if unknown.

    Derived by dividing the window's reported pixel extent by its cell
    count. That extent can include the terminal's window padding, which
    this then smears across every row/column as a fractional per-cell
    error -- invisible to the continuous, radius-based atom picker but
    enough to make a rigid per-cell grid (the periodic-table picker) land
    on the wrong cell, worse the further down you go as the error adds up.
    Rounding removes the smaller part of that noise; :func:`query_cell_size_px`
    removes it entirely by asking the terminal for the exact cell size, and
    the viewer prefers that when the terminal answers.
    """
    cols, rows, xpx, ypx = terminal_size_px(fd)
    cw = round(xpx / cols) if xpx and cols else 9.0
    ch = round(ypx / rows) if ypx and rows else 18.0
    return float(cw), float(ch)


def query_cell_size_px(fd_in: int = 0, fd_out: int = 1, timeout: float = 0.2):
    """Ask the terminal for its exact cell size in pixels via ``CSI 16 t``.

    Returns ``(cell_width_px, cell_height_px)`` from the terminal's
    ``CSI 6 ; height ; width t`` reply, or ``None`` if it doesn't answer.
    Unlike :func:`cell_size_px` this is the terminal's own authoritative
    cell metric, with no window-padding contamination -- so cell/pixel
    hit-testing lines up exactly with where glyphs are actually drawn.
    Requires the tty to be in raw mode (mirrors input.query_decset).
    """
    import select
    import time as _time

    try:
        os.write(fd_out, b"\x1b[16t")
    except OSError:
        return None
    buf = b""
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        r, _, _ = select.select([fd_in], [], [], max(0.0, deadline - _time.monotonic()))
        if not r:
            break
        try:
            chunk = os.read(fd_in, 64)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        if b"t" in buf and b"\x1b[6;" in buf:
            break
    # parse: ESC [ 6 ; <height> ; <width> t
    marker = b"\x1b[6;"
    i = buf.find(marker)
    if i < 0:
        return None
    j = buf.find(b"t", i)
    if j < 0:
        return None
    try:
        h_str, w_str = buf[i + len(marker):j].split(b";")
        ch = float(int(h_str))
        cw = float(int(w_str))
    except (ValueError, IndexError):
        return None
    if cw <= 0 or ch <= 0:
        return None
    return cw, ch


def unique_id_base(stride: int = 4) -> int:
    """A per-process base for Kitty graphics ``i=``/``p=`` ids.

    Image ids live in a namespace global to the whole kitty *process* (shared
    across all its panes/tabs), not per-window — see the graphics protocol
    spec's note that "IDs are in a global namespace [so] there can easily be
    collisions." Hardcoding small ids like 1/2 means two independent
    vimol instances sharing a kitty process (e.g. two panes) can delete or
    overwrite each other's frames. Deriving the base from the pid keeps
    concurrent instances apart; the stride ensures that even OS-assigned
    sequential pids (common right after spawning several processes) don't
    produce adjacent, overlapping id ranges.
    """
    pid = os.getpid()
    return ((pid * stride) % 0x7FFFFFFF) or 1


def supports_kitty() -> bool:
    """Best-effort detection of Kitty graphics support via environment.

    Environment variables lie in exactly the situations where support matters
    most: SSH strips ``KITTY_WINDOW_ID``/``TERM_PROGRAM``, and remote hosts
    often force ``TERM=xterm-256color`` for a terminal that renders graphics
    perfectly. Treat a True here as trustworthy and a False as merely
    "unknown" -- callers should fall back to :func:`probe_terminal`, which
    asks the terminal itself and is authoritative either way.
    """
    if os.environ.get("VIMOL_FORCE_KITTY"):
        return True
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    term = os.environ.get("TERM", "")
    if "kitty" in term or "ghostty" in term:
        return True
    prog = os.environ.get("TERM_PROGRAM", "").lower()
    if prog in ("ghostty", "wezterm"):
        return True
    if os.environ.get("WEZTERM_PANE"):
        return True
    return False


# --------------------------------------------------------------------------
# Startup probe: one round trip that answers everything
# --------------------------------------------------------------------------
@dataclass
class TerminalProbe:
    """What one :func:`probe_terminal` round trip learned about the terminal.

    ``graphics`` is True/False when the terminal answered the DA1 fence (so
    its silence on the graphics query is a real "no"), and None when nothing
    came back at all (not a terminal / reply lost) -- unknown, not refusal.
    ``rtt`` is the query->fence round-trip time: ~1 ms on a local terminal,
    tens to hundreds of ms over SSH, so it doubles as a free link-latency
    estimate (see Viewer's idle-resolution seeding). ``leftover`` preserves
    any non-reply bytes that arrived interleaved (keys typed during startup)
    so the caller can feed them to its input decoder instead of losing them.
    """
    graphics: Optional[bool]
    pixel_mouse: bool
    cell_px: Optional[Tuple[float, float]]
    rtt: Optional[float] = None
    leftover: bytes = b""


# The probe's graphics query id. With a=q the terminal only *answers* -- no
# image is stored -- so unlike display ids this can't collide across panes.
_PROBE_GFX_ID = 31

# Replies the probe can receive, in the order the queries are sent. The DA1
# reply (CSI ? ... c) is the fence: every xterm-descendant answers it, and
# in-order reply processing guarantees it arrives *after* whichever of the
# earlier replies the terminal supports. Note the DECRQM reply also starts
# with CSI ? but ends in $y, so the DA1 pattern (digits/; then a final 'c')
# cannot match it.
_RE_GFX_REPLY = re.compile(rb"\x1b_Gi=%d;([^\x1b]*)\x1b\\" % _PROBE_GFX_ID)
_RE_DECRQM_1016 = re.compile(rb"\x1b\[\?1016;(\d+)\$y")
_RE_CELL_SIZE = re.compile(rb"\x1b\[6;(\d+);(\d+)t")
_RE_DA1 = re.compile(rb"\x1b\[\?[0-9;]*c")


def probe_query_bytes() -> bytes:
    """The combined capability query, sent as ONE write (one SSH round trip).

    Four questions back to back: (1) a Kitty graphics *query* (``a=q`` with a
    1x1 dummy pixel -- validated and answered, never displayed or stored);
    (2) DECRQM for SGR-Pixels mouse (1016); (3) ``CSI 16 t`` for the exact
    cell size; (4) DA1 (``CSI c``) as a universally-answered fence. Terminals
    ignore the queries they don't recognize (the graphics APC included), so
    this is safe to fire at anything that calls itself a terminal. Requires
    the tty to be in raw mode to read the replies.
    """
    gfx = (b"\x1b_Gi=%d,s=1,v=1,a=q,t=d,f=24;" % _PROBE_GFX_ID
           + base64.standard_b64encode(b"\x00\x00\x00") + b"\x1b\\")
    return gfx + b"\x1b[?1016$p" + b"\x1b[16t" + b"\x1b[c"


def _parse_probe_pieces(buf: bytes):
    """Extract (graphics, pixel_mouse, cell_px, spans) from reply bytes.

    graphics is None when no graphics reply is present at all -- only the
    caller knows whether that silence is meaningful (it is once the DA1
    fence has arrived).
    """
    spans = []
    graphics = None
    m = _RE_GFX_REPLY.search(buf)
    if m:
        graphics = m.group(1).startswith(b"OK")
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
    return graphics, pixel, cell, spans


def _probe_leftover(buf: bytes, spans) -> bytes:
    """*buf* minus the recognized reply spans: bytes the user typed."""
    out = bytearray()
    prev = 0
    for s, e in sorted(spans):
        out += buf[prev:s]
        prev = e
    out += buf[prev:]
    return bytes(out)


def parse_probe_reply(buf: bytes) -> Optional[TerminalProbe]:
    """Parse an accumulating reply buffer; None until the DA1 fence arrives.

    Once the fence is in, a missing graphics reply is a definitive "no
    graphics support" (the terminal processed our queries in order and
    answered the later one), so ``graphics`` is always True/False here.
    """
    m_da1 = _RE_DA1.search(buf)
    if m_da1 is None:
        return None
    graphics, pixel, cell, spans = _parse_probe_pieces(buf)
    spans.append(m_da1.span())
    return TerminalProbe(graphics=bool(graphics), pixel_mouse=pixel, cell_px=cell,
                         leftover=_probe_leftover(buf, spans))


def probe_terminal(fd_in: int = 0, fd_out: int = 1, timeout: float = 1.0) -> TerminalProbe:
    """Ask the terminal what it can do: graphics, pixel mouse, cell size, RTT.

    One write, then reads until the DA1 fence answers (or *timeout*, which
    only real non-terminals hit -- every xterm descendant answers DA1, so
    capable terminals cost exactly one round trip, not a fixed timeout).
    Requires the tty to be in raw mode. On timeout, whatever partial replies
    did arrive are still used, with ``graphics=None`` (unknown, not "no").
    """
    import select
    import time as _time

    try:
        os.write(fd_out, probe_query_bytes())
    except OSError:
        return TerminalProbe(graphics=None, pixel_mouse=False, cell_px=None)
    t0 = _time.monotonic()
    deadline = t0 + timeout
    buf = b""
    while _time.monotonic() < deadline:
        r, _, _ = select.select([fd_in], [], [], max(0.0, deadline - _time.monotonic()))
        if not r:
            break
        try:
            chunk = os.read(fd_in, 512)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        probe = parse_probe_reply(buf)
        if probe is not None:
            probe.rtt = _time.monotonic() - t0
            return probe
    graphics, pixel, cell, spans = _parse_probe_pieces(buf)
    return TerminalProbe(graphics=graphics, pixel_mouse=pixel, cell_px=cell,
                         leftover=_probe_leftover(buf, spans))


# --------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------
def _controls(d: dict) -> bytes:
    return ",".join(f"{k}={v}" for k, v in d.items()).encode("ascii")


def encode_image(
    pixels: np.ndarray,
    *,
    image_id: int = 1,
    placement_id: Optional[int] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    move_cursor: bool = False,
    compress: bool = True,
    compress_level: int = 6,
    z_index: int = 0,
    quiet: int = 2,
) -> bytes:
    """Encode an (H, W, 3|4) uint8 array as Kitty graphics-protocol bytes.

    cols/rows scale the image into that many terminal cells (defaults to the
    image's native pixel size). When *move_cursor* is False (C=1) the cursor is
    left in place, which is what you want when compositing a UI around the
    image. placement_id defaults to image_id (placements are scoped to their
    image, so this is always collision-safe).

    *compress_level* is the zlib level (0-9). The compress step costs about as
    much per frame as the whole raycast, so interactive callers pass level 1
    (roughly half the CPU of the default 6 for ~25% more bytes -- a fine
    trade for a local terminal); a resting still can afford the default for a
    smaller payload.
    """
    if placement_id is None:
        placement_id = image_id
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    fmt = 32 if arr.ndim == 3 and arr.shape[2] == 4 else 24
    raw = arr.tobytes()
    if compress:
        raw = zlib.compress(raw, compress_level)
    payload = base64.standard_b64encode(raw)

    ctrl = {
        "a": "T",          # transmit and display
        "f": fmt,          # 24=RGB, 32=RGBA
        "s": w,
        "v": h,
        "i": image_id,
        "p": placement_id,
        "q": quiet,        # 0=verbose, 1=no ok, 2=no ok/err
    }
    if compress:
        ctrl["o"] = "z"
    if cols:
        ctrl["c"] = int(cols)
    if rows:
        ctrl["r"] = int(rows)
    if not move_cursor:
        ctrl["C"] = 1
    if z_index:
        ctrl["z"] = int(z_index)

    out = bytearray()
    # chunked transfer: first chunk carries controls, subsequent carry only m=
    if len(payload) <= _CHUNK:
        ctrl["m"] = 0
        out += _GRAPHICS_START + _controls(ctrl) + b";" + payload + _GRAPHICS_END
        return bytes(out)

    first = True
    view = memoryview(payload)
    n = len(payload)
    pos = 0
    while pos < n:
        chunk = view[pos:pos + _CHUNK]
        pos += _CHUNK
        last = pos >= n
        if first:
            c = dict(ctrl)
            c["m"] = 0 if last else 1
            out += _GRAPHICS_START + _controls(c) + b";" + bytes(chunk) + _GRAPHICS_END
            first = False
        else:
            m = 0 if last else 1
            out += _GRAPHICS_START + f"m={m}".encode() + b";" + bytes(chunk) + _GRAPHICS_END
    return bytes(out)


def delete_image(image_id: int = 1) -> bytes:
    """Bytes to delete a transmitted image (and its placements) by id."""
    return _GRAPHICS_START + _controls({"a": "d", "d": "I", "i": image_id, "q": 2}) + b";" + _GRAPHICS_END


def clear_all_images() -> bytes:
    return _GRAPHICS_START + b"a=d,d=A,q=2;" + _GRAPHICS_END


def set_pointer_shape(shape: str) -> bytes:
    """OSC 22 bytes to PUSH the OS mouse-pointer icon to a CSS-cursor *shape*.

    Kitty-family terminals honor this while focused; others ignore an OSC they
    don't recognize, so it's safe to write unconditionally (like the mouse-mode
    sequences). Pushing (rather than a plain set) pairs with
    :func:`reset_pointer_shape`'s pop, so disarming restores whatever shape
    was actually active before -- not just the terminal's generic default,
    which a plain set/reset pair would revert to instead. Returns raw bytes
    for the caller to write.
    """
    return f"\x1b]22;>{shape}\x1b\\".encode()


def reset_pointer_shape() -> bytes:
    """OSC 22 bytes to POP the pointer shape pushed by :func:`set_pointer_shape`."""
    return b"\x1b]22;<\x1b\\"


def write_bytes(data: bytes, fd: int = 1) -> None:
    os.write(fd, data)


def png_bytes(pixels: np.ndarray) -> bytes:
    """Encode an (H, W, 3|4) uint8 array as a PNG (stdlib zlib, no Pillow)."""
    import struct

    arr = np.ascontiguousarray(pixels, np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    color_type = {1: 0, 3: 2, 4: 6}[channels]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    # add filter byte 0 per scanline
    stride = w * channels
    raw = bytearray()
    flat = arr.reshape(h, stride)
    for y in range(h):
        raw.append(0)
        raw += flat[y].tobytes()
    idat = zlib.compress(bytes(raw), 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
