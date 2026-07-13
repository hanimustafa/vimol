"""Kitty terminal graphics protocol encoder + terminal geometry helpers.

Reference: https://sw.kovidgoyal.net/kitty/graphics-protocol/

We transmit raw RGB/RGBA pixels (optionally zlib-compressed) in <=4096-byte
base64 chunks. The public surface is small on purpose so the module can be
lifted into other terminal apps unchanged.
"""
from __future__ import annotations

import base64
import os
import sys
import zlib
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
    """Approximate (cell_width_px, cell_height_px). Defaults to 9x18 if unknown."""
    cols, rows, xpx, ypx = terminal_size_px(fd)
    cw = xpx / cols if xpx and cols else 9.0
    ch = ypx / rows if ypx and rows else 18.0
    return cw, ch


def supports_kitty() -> bool:
    """Best-effort detection of Kitty graphics support via environment."""
    if os.environ.get("MVIEWER_FORCE_KITTY"):
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
# Image encoding
# --------------------------------------------------------------------------
def _controls(d: dict) -> bytes:
    return ",".join(f"{k}={v}" for k, v in d.items()).encode("ascii")


def encode_image(
    pixels: np.ndarray,
    *,
    image_id: int = 1,
    placement_id: int = 1,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    move_cursor: bool = False,
    compress: bool = True,
    z_index: int = 0,
    quiet: int = 2,
) -> bytes:
    """Encode an (H, W, 3|4) uint8 array as Kitty graphics-protocol bytes.

    cols/rows scale the image into that many terminal cells (defaults to the
    image's native pixel size). When *move_cursor* is False (C=1) the cursor is
    left in place, which is what you want when compositing a UI around the
    image.
    """
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    fmt = 32 if arr.ndim == 3 and arr.shape[2] == 4 else 24
    raw = arr.tobytes()
    if compress:
        raw = zlib.compress(raw)
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
