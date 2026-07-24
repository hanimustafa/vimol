"""Terminal input decoding: keys and SGR mouse events (cell *and* pixel).

The decoder is stateful so it tolerates escape sequences that get split across
reads. It is deliberately independent of the viewer: an embedding application
can create an :class:`InputDecoder`, feed it whatever bytes it intercepts, and
forward the resulting :class:`MouseEvent` / :class:`KeyEvent` objects to a
:class:`vimol.widget.MoleculeWidget` — so you can capture the mouse inside
your own UI region and drive the molecule with it.

Mouse coordinates come back in *pixels* when the terminal supports SGR-Pixels
(DECSET 1016, a Kitty extension also in Ghostty/WezTerm/foot) and in character
cells otherwise; the ``pixel`` flag on each event says which.
"""
from __future__ import annotations

import os
import select
from dataclasses import dataclass
from typing import List, Optional, Union


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
@dataclass
class MouseEvent:
    action: str          # 'down' | 'up' | 'drag' | 'move' | 'scroll'
    x: float             # pixels if pixel else 0-based cell column
    y: float             # pixels if pixel else 0-based cell row
    button: Optional[int] = None    # 0 left, 1 middle, 2 right (None for move/scroll)
    scroll: Optional[str] = None     # 'up' | 'down' | 'left' | 'right'
    shift: bool = False
    alt: bool = False
    ctrl: bool = False
    pixel: bool = False   # True -> x/y are pixels, False -> x/y are cells


@dataclass
class KeyEvent:
    key: str             # a single char, or a name: 'up' 'down' 'left' 'right'
                         # 'escape' 'enter' 'tab' 'backspace'
    def __str__(self) -> str:
        return self.key


Event = Union[MouseEvent, KeyEvent]

_CSI_FINAL = {"A": "up", "B": "down", "C": "right", "D": "left",
              "H": "home", "F": "end"}


# --------------------------------------------------------------------------
# Mouse enable / disable control sequences
# --------------------------------------------------------------------------
def enable_mouse(pixel: bool = True, hover: bool = False) -> bytes:
    """Bytes to turn on SGR mouse reporting.

    hover=True uses any-event tracking (1003) so motion is reported even with no
    button held (needed for hover picking); otherwise button-event tracking
    (1002) reports motion only while dragging. pixel=True additionally requests
    SGR-Pixels (1016) for pixel-precise coordinates.
    """
    motion = "1003" if hover else "1002"
    seq = f"\x1b[?1000;{motion};1006h"
    if pixel:
        seq += "\x1b[?1016h"
    return seq.encode()


def disable_mouse(pixel: bool = True) -> bytes:
    seq = "\x1b[?1000;1002;1003;1006l"
    if pixel:
        seq += "\x1b[?1016l"
    return seq.encode()


def query_decset(mode: int, fd_in: int = 0, fd_out: int = 1, timeout: float = 0.2):
    """Probe a DECSET private mode via DECRQM; return its reported value.

    Returns the numeric value from the terminal's ``CSI ? mode ; value $y``
    reply (1=set, 2=reset, 3=perm-set, 4=perm-reset, 0=not recognized) or None
    if the terminal did not answer. Requires the tty to be in raw mode.
    """
    try:
        os.write(fd_out, f"\x1b[?{mode}$p".encode())
    except OSError:
        return None
    buf = b""
    end_marker = b"$y"
    import time as _time
    # We cannot call time.time() indirectly? it's fine here (not a workflow script).
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        r, _, _ = select.select([fd_in], [], [], max(0.0, deadline - _time.monotonic()))
        if not r:
            break
        try:
            chunk = os.read(fd_in, 256)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        if end_marker in buf:
            break
    # parse: ESC [ ? <mode> ; <value> $ y
    marker = f"\x1b[?{mode};".encode()
    i = buf.find(marker)
    if i < 0:
        return None
    j = buf.find(end_marker, i)
    if j < 0:
        return None
    try:
        val = int(buf[i + len(marker):j])
    except ValueError:
        return None
    return val


def supports_pixel_mouse(fd_in: int = 0, fd_out: int = 1, timeout: float = 0.5) -> bool:
    """True if the terminal reports SGR-Pixels (1016) as a recognized mode.

    The default timeout is generous (0.5s) on purpose: the DECRQM round-trip
    is one-time startup cost, but over SSH a tight timeout loses the race and
    reports "no pixel mouse" for a terminal that in fact supports it. A wrong
    answer here no longer breaks the mouse -- the viewer keeps the wire format
    and the decoder consistent either way (see Viewer._enable_mouse) -- but a
    right answer preserves pixel-precise dragging and picking over SSH.
    (The viewer itself now learns this from kitty.probe_terminal's combined
    round trip; this helper remains for embedders driving InputDecoder
    directly, e.g. examples/embed_demo.py.)
    """
    val = query_decset(1016, fd_in, fd_out, timeout=timeout)
    return val in (1, 2, 3, 4)


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------
class InputDecoder:
    """Turn raw terminal bytes into :class:`Event` objects.

    Parameters
    ----------
    pixel:
        Whether SGR mouse coordinates should be interpreted as pixels. Set this
        to match whether you enabled mode 1016 (see :func:`supports_pixel_mouse`).
    """

    def __init__(self, pixel: bool = False):
        self.pixel = pixel
        self._buf = bytearray()

    def feed(self, data: bytes) -> List[Event]:
        self._buf += data
        events: List[Event] = []
        while self._buf:
            ev, consumed = self._parse_one()
            if consumed == 0:
                break  # incomplete sequence, wait for more bytes
            if ev is not None:
                events.append(ev)
            del self._buf[:consumed]
        return events

    def flush(self) -> List[Event]:
        """Emit any pending unterminated sequence (e.g. a lone ESC -> escape)."""
        if len(self._buf) == 1 and self._buf[0] == 0x1B:
            self._buf.clear()
            return [KeyEvent("escape")]
        return []

    # -- internals --------------------------------------------------------
    def _parse_one(self):
        b = self._buf
        c = b[0]
        if c != 0x1B:
            return self._parse_key_byte(c), 1
        # escape sequence
        if len(b) < 2:
            return None, 0  # wait; flush() handles a lingering lone ESC
        if b[1] == ord("["):
            return self._parse_csi()
        if b[1] == ord("O"):  # SS3 (application arrows)
            if len(b) < 3:
                return None, 0
            name = _CSI_FINAL.get(chr(b[2]))
            return (KeyEvent(name) if name else None), 3
        # Alt+key
        return KeyEvent(chr(b[1])), 2

    def _parse_key_byte(self, c: int):
        if c == 0x0D:
            return KeyEvent("enter")
        if c == 0x09:
            return KeyEvent("tab")
        if c in (0x7F, 0x08):
            return KeyEvent("backspace")
        return KeyEvent(chr(c))

    def _parse_csi(self):
        b = self._buf
        n = len(b)
        # SGR mouse: ESC [ < params (M|m)
        if n >= 3 and b[2] == ord("<"):
            k = 3
            while k < n and b[k] not in (ord("M"), ord("m")):
                k += 1
            if k >= n:
                return None, 0  # incomplete
            body = bytes(b[3:k]).decode("ascii", "ignore")
            final = chr(b[k])
            return self._decode_sgr_mouse(body, final), k + 1
        # generic CSI: read until final byte in @..~
        k = 2
        while k < n and not (0x40 <= b[k] <= 0x7E):
            k += 1
        if k >= n:
            return None, 0
        final = chr(b[k])
        name = _CSI_FINAL.get(final)
        if name is None:
            return None, k + 1
        # Modified arrows/home/end arrive as "CSI 1 ; <modifier> <final>"
        # (xterm's key-modifier encoding: modifier-1 is a shift/alt/ctrl/meta
        # bitmask). Only Alt/Option is consumed today -- terminals disagree on
        # whether Option reports as "Alt" (bit 2, modifier 3) or "Meta" (bit
        # 8, modifier 9), so both are treated as alt, matching how the SGR
        # mouse decoder above already conflates its own alt bit.
        params = bytes(b[2:k]).decode("ascii", "ignore")
        if ";" in params:
            try:
                mod = int(params.split(";")[1])
            except (ValueError, IndexError):
                mod = 1
            bits = mod - 1
            if bits & (2 | 8):
                name = f"alt+{name}"
        return KeyEvent(name), k + 1

    def _decode_sgr_mouse(self, body: str, final: str) -> Optional[MouseEvent]:
        try:
            bs, xs, ys = body.split(";")
            code = int(bs); x = int(xs); y = int(ys)
        except ValueError:
            return None
        shift = bool(code & 4)
        alt = bool(code & 8)
        ctrl = bool(code & 16)
        wheel = bool(code & 64)
        motion = bool(code & 32)
        low = code & 3
        # coordinates: SGR is 1-based; make cells 0-based. Pixels stay as-is.
        if self.pixel:
            px, py = float(x), float(y)
        else:
            px, py = float(x - 1), float(y - 1)

        if wheel:
            direction = {0: "up", 1: "down", 2: "left", 3: "right"}.get(low, "up")
            return MouseEvent("scroll", px, py, scroll=direction,
                              shift=shift, alt=alt, ctrl=ctrl, pixel=self.pixel)
        if motion:
            if low == 3:
                return MouseEvent("move", px, py, button=None,
                                  shift=shift, alt=alt, ctrl=ctrl, pixel=self.pixel)
            return MouseEvent("drag", px, py, button=low,
                              shift=shift, alt=alt, ctrl=ctrl, pixel=self.pixel)
        action = "down" if final == "M" else "up"
        return MouseEvent(action, px, py, button=low,
                          shift=shift, alt=alt, ctrl=ctrl, pixel=self.pixel)
