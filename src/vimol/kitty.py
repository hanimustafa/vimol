"""Kitty terminal graphics protocol encoder + terminal geometry helpers.

Reference: https://sw.kovidgoyal.net/kitty/graphics-protocol/

We transmit raw RGB/RGBA pixels (optionally zlib-compressed) in <=4096-byte
base64 chunks, or -- to a terminal the startup probe proved is on this
machine -- through a POSIX shared-memory object, which costs a memcpy instead
of a per-frame compress. The public surface is small on purpose so the module
can be lifted into other terminal apps unchanged.
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
# Shared-memory transfer (the protocol's "local client" path, t=s)
# --------------------------------------------------------------------------
# Transmitting pixels as escape codes (t=d) means zlib + base64 of the whole
# frame: measured ~24 ms for a 1600x1000 RGBA frame, against ~3.5 ms to
# render it on the GPU. Handing the terminal a POSIX shared-memory *name*
# instead costs one memcpy (~0.9 ms for the same frame) and lets the terminal
# skip its decompress too. It only works when the terminal runs on this
# machine, which is exactly what probing it proves (see probe_query_bytes).
#
# The terminal unlinks the object once it has read it, so our own cleanup is
# a best-effort mop-up of objects a terminal never consumed.
_SHM_PREFIX = "vimol"
# Darwin caps shm names at 31 chars including the leading slash, so keep the
# pid+counter form short.
_shm_counter = 0
_shm_pending: list = []
# Objects kept alive this many frames before we assume the terminal will
# never read them and unlink. Under frame pacing at most one frame is ever
# in flight, so this is pure belt-and-braces against a wedged terminal.
_SHM_KEEP = 3


def shm_write(data: bytes) -> str:
    """Copy *data* into a fresh POSIX shared-memory object; return its name.

    The name is returned in the form the terminal passes to ``shm_open`` --
    with the leading slash -- which is what the protocol's ``t=s`` payload
    must carry. The mapping is closed immediately (the object outlives it)
    and deliberately NOT unlinked: the terminal does that after reading.
    """
    from multiprocessing import shared_memory
    global _shm_counter
    _shm_counter += 1
    name = f"{_SHM_PREFIX}-{os.getpid()}-{_shm_counter}"
    shm = shared_memory.SharedMemory(create=True, size=max(len(data), 1), name=name)
    try:
        shm.buf[:len(data)] = data
    finally:
        # Python's resource tracker would "helpfully" unlink this at exit and
        # warn about leaked objects -- but the *terminal* owns the unlink, so
        # unregister before closing. Private API; a failure here is harmless
        # (worst case: a warning line at interpreter shutdown).
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        shm.close()
    _shm_pending.append(name)
    if len(_shm_pending) > _SHM_KEEP:
        shm_release(_shm_pending.pop(0))
    return "/" + name


def shm_release(name: str) -> None:
    """Unlink shared-memory object *name* if it still exists (no-op if the
    terminal already consumed and unlinked it, which is the normal case)."""
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=name.lstrip("/"))
    except FileNotFoundError:
        return
    except Exception:
        return
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass


def shm_cleanup() -> None:
    """Unlink every shared-memory object we still have outstanding (call on
    exit: a frame written just before quitting may never have been read)."""
    while _shm_pending:
        shm_release(_shm_pending.pop())


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
    # True when the terminal accepted a shared-memory (t=s) transfer, i.e. it
    # runs on this machine and supports the protocol's local-client path;
    # False when it refused or ignored it; None when we never asked.
    shm: Optional[bool] = None
    # The terminal's own background color from an OSC 11 query, downsampled
    # to 8 bits/channel; None if it didn't answer (theme.resolve() then
    # falls back to COLORFGBG/default -- see theme.py).
    bg_rgb: Optional[Tuple[int, int, int]] = None


# The probe's graphics query ids. With a=q the terminal only *answers* -- no
# image is stored -- so unlike display ids these can't collide across panes.
_PROBE_GFX_ID = 31
_PROBE_SHM_ID = 32

# Replies the probe can receive, in the order the queries are sent. The DA1
# reply (CSI ? ... c) is the fence: every xterm-descendant answers it, and
# in-order reply processing guarantees it arrives *after* whichever of the
# earlier replies the terminal supports. Note the DECRQM reply also starts
# with CSI ? but ends in $y, so the DA1 pattern (digits/; then a final 'c')
# cannot match it.
_RE_GFX_REPLY = re.compile(rb"\x1b_Gi=%d;([^\x1b]*)\x1b\\" % _PROBE_GFX_ID)
_RE_SHM_REPLY = re.compile(rb"\x1b_Gi=%d;([^\x1b]*)\x1b\\" % _PROBE_SHM_ID)
_RE_DECRQM_1016 = re.compile(rb"\x1b\[\?1016;(\d+)\$y")
_RE_CELL_SIZE = re.compile(rb"\x1b\[6;(\d+);(\d+)t")
_RE_DA1 = re.compile(rb"\x1b\[\?[0-9;]*c")
_OSC11_QUERY = b"\x1b]11;?\x1b\\"
_RE_OSC11_REPLY = re.compile(
    rb"\x1b\]11;rgb:([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})/([0-9a-fA-F]{2,4})(?:\x1b\\|\x07)")


def _osc11_channel(hexstr: bytes) -> int:
    """A 2- or 4-hex-digit OSC 11 channel, downsampled to 8 bits: terminals
    disagree on whether they reply with 8 or 16 bits/channel, and for a
    16-bit channel the high byte is what every other tool in the wild uses
    as the 8-bit approximation."""
    return int(hexstr[:2], 16)


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
    return gfx + b"\x1b[?1016$p" + b"\x1b[16t" + _OSC11_QUERY + b"\x1b[c"


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


def probe_terminal(fd_in: int = 0, fd_out: int = 1, timeout: float = 1.0,
                   ask_shm: bool = True) -> TerminalProbe:
    """Ask the terminal what it can do: graphics, pixel mouse, cell size, RTT,
    and (unless *ask_shm* is False) shared-memory transfer.

    One write, then reads until the DA1 fence answers (or *timeout*, which
    only real non-terminals hit -- every xterm descendant answers DA1, so
    capable terminals cost exactly one round trip, not a fixed timeout).
    Requires the tty to be in raw mode. On timeout, whatever partial replies
    did arrive are still used, with ``graphics=None`` (unknown, not "no").

    The shm question carries a real shared-memory object; a terminal that
    accepts it has already unlinked it, and one that didn't gets the object
    released here, so the probe never leaks either way.
    """
    import select
    import time as _time

    name = None
    if ask_shm:
        try:
            name = shm_write(b"\x00\x00\x00")
        except Exception:
            name = None                  # no POSIX shm here: just skip the question
    try:
        os.write(fd_out, probe_query_bytes(shm_name=name))
    except OSError:
        if name is not None:
            shm_release(name)
        return TerminalProbe(graphics=None, pixel_mouse=False, cell_px=None)

    def _finish(probe: TerminalProbe) -> TerminalProbe:
        if name is None:
            probe.shm = None             # never asked: not the same as "no"
        elif probe.shm is not True:
            shm_release(name)            # refused/ignored -- mop up our object
        return probe

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
            return _finish(probe)
    graphics, pixel, cell, shm, bg_rgb, spans = _parse_probe_pieces(buf)
    return _finish(TerminalProbe(graphics=graphics, pixel_mouse=pixel, cell_px=cell,
                                 leftover=_probe_leftover(buf, spans), shm=shm, bg_rgb=bg_rgb))


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
    transmit: str = "direct",
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

    *transmit* selects how the pixels reach the terminal: ``"direct"`` (t=d)
    inlines them as base64 escape-code chunks, which is the only option for a
    terminal on the far side of a link; ``"shm"`` (t=s) copies them into a
    POSIX shared-memory object and sends only its name, which is far cheaper
    but requires a terminal on THIS machine (see shm_write). ``compress`` is
    ignored in shm mode -- compressing a memcpy would be all cost, no benefit.
    """
    if transmit not in ("direct", "shm"):
        raise ValueError(f"unknown transmit {transmit!r} (expected 'direct' or 'shm')")
    if placement_id is None:
        placement_id = image_id
    arr = np.ascontiguousarray(pixels, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    fmt = 32 if arr.ndim == 3 and arr.shape[2] == 4 else 24
    if transmit == "shm":
        compress = False
        payload = base64.standard_b64encode(shm_write(arr.tobytes()).encode())
    else:
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
    if transmit == "shm":
        ctrl["t"] = "s"
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
