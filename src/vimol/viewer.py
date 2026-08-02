"""Full-screen interactive viewer — a thin driver around MoleculeWidget.

All the interaction logic lives in :class:`vimol.widget.MoleculeWidget` and
input decoding in :class:`vimol.input.InputDecoder`. This class only does the
terminal-owning parts: raw mode, the alternate screen, enabling mouse
reporting, the render/input loop, and a status bar. Embedders who want to
capture the mouse in their own region should use the widget + decoder directly
(see examples/embed_demo.py) rather than this driver.
"""
from __future__ import annotations

import math
import os
import re
import select
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .molecule import Molecule
from .render import Style
from .widget import MoleculeWidget, REPRESENTATIONS
from .bonds import ensure_bonds
from .structures import StructureSet
from . import editor
from . import kitty
from . import input as _input
from . import elements
from . import templates
from . import periodic_table
from . import select as atom_selection
from . import theme
from . import file_browser
from .parsers import load_all
from .parsers import xyz as xyz_parser

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
_SELECTION_OPTIONS = (
    "Manual", "Backbone", "Backbone + Cβ", "Heavy atoms", "Largest ring system",
)
# English phrasing for a ⊂RMSD hover tip (design 2026-08-02, VIM-30) --
# "Manual" has no entry, so a hand-picked/derived selection always falls
# back to its raw atom list instead.
_SELECTION_PHRASES = {
    "Backbone": "the backbone",
    "Backbone + Cβ": "the backbone + Cβ",
    "Heavy atoms": "heavy atoms",
    "Largest ring system": "the largest ring system",
}
_SELECTION_HINT = " ↑↓ move · Enter/click select · Esc cancel"
# What an RMSD column shows on the row it was fitted ONTO. Distinct from the
# em-dash every other blank cell uses (design 2026-08-02): a reference row is
# not a measurement that failed or never ran, it is the frame everything else
# was measured against, and the two must not look alike.
_SELF_CELL = "Self"
_BROWSER_HINT = " ↑↓ move · Enter open · ~ home · Esc cancel"
# Fallback visible width of the status bar's left-hand (hover/molecule-info)
# field, used ONLY before the terminal size is known (_cols == 0, i.e. before
# the first _update_geometry). Once it is known the field is sized
# dynamically to every column the right-anchored trailer does not need -- see
# _status_bar for why that width must never depend on the hover text.
_LEFT_WIDTH = 24

# Structure-list strip width, in columns (design §4.1): a fifth of the
# terminal, clamped so it's never cramped nor a runaway hog on a wide one.
_LIST_W_MIN = 18
_LIST_W_MAX = 28

# Measurement table (design 2026-07-30, VIM-6): the 3D image always keeps at
# least this many columns, however many measurements are pinned -- columns
# beyond that are silently dropped (_measure_layout) rather than shrinking
# the viewport to nothing or negative.
_MEASURE_MIN_VIEWPORT_COLS = 20
@dataclass
class _SubsetRMSDColumn:
    """One persistent subset fit shown beside the structure list.

    ``indices`` are raw positions into the reference molecule, so they only
    mean anything at the revision they were picked at -- ``reference_revision``
    is what lets a later edit be noticed instead of silently re-fitting on
    whatever atoms have shifted into those slots (design §5, "stale").
    """
    select_id: int
    reference_index: int
    reference_revision: int
    indices: Tuple[int, ...]
    labels: Tuple[str, ...]
    values: List[Optional[float]]
    # Which selection preset produced ``indices`` (a name from
    # _SELECTION_OPTIONS), if any and if it hasn't since been hand-edited --
    # None for a manual/option-click pick. Lets the hover tip read like
    # English ("aligning on the backbone") instead of a raw atom dump
    # (design 2026-08-02, VIM-30).
    preset_label: Optional[str] = None

    @property
    def header(self) -> str:
        # select_id/full_id share one counter (Viewer._next_rmsd_id)
        # precisely so a ⊂RMSD and a ∀RMSD column can never collide.
        return f"⊂RMSD#{self.select_id}"


@dataclass
class _FullRMSDColumn:
    """One persistent all-atom fit against a particular main frame."""
    full_id: int
    reference_index: int
    reference_revision: int
    values: List[Optional[float]]

    @property
    def header(self) -> str:
        return f"∀RMSD#{self.full_id}"

# Structure-list strip layout (design §4.1) -- colors live in theme.py now.
# Strip rows spent on chrome rather than entries: the header above, and the
# separator plus footer lines below. _list_capacity() derives what fits from
# these, and _draw_list lays the panel out to match.
_LIST_ROWS_ABOVE = 2                # header + a blank row
_LIST_ROWS_BELOW = 5                # separator + the four legend/hint lines
_LIST_WHEEL_STEP = 3                # display rows per mouse-wheel notch

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
    "     with 2+ structures: a table column tracks it live, click × to remove",
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
    "     with 2+ structures: a table column tracks it live, click × to remove",
]
_HELP_TAIL = [
    "  A .................. add a structure file",
    "  n / p / opt+up/dn .. next/prev frame   d .................. depth cue",
    "  t .................. transparent bg    g .................. hi-quality",
    "  ctrl-t ............. light/dark theme",
    "  f / z .............. re-fit / reset    ? .................. toggle help",
    "  overlay: r .......... align all → ∀RMSD#N  R ... pick atoms → ⊂RMSD#N",
    "     Shift+S / click select ... Backbone, Heavy atoms, or Ring system",
    "     option-click atom ........ additive manual subset selection",
    "     RMSD#N column: hover title for spec · click arm selection · R recalculate",
    "  q / Esc ............ quit",
]


def _help_lines(editable: bool):
    return _HELP_HEAD + (_HELP_EDIT if editable else _HELP_VIEW) + _HELP_TAIL

# Keys the driver always claims. 'a' is here in both modes but means different
# things: autospin when read-only, append when editable (see _driver_key).
_BASE_DRIVER_KEYS = {"q", "escape", "a", "A", "?", "d", "g", "t", "n", "p", "m", "r", "R", "S", "\x03", "\x14",
                      "alt+up", "alt+down"}
# Extra keys claimed only when editing is enabled.
_EDIT_DRIVER_KEYS = {"s", "u", "o", "x", "c"}

# Interactive frame budget (seconds) for the dynamic-resolution controller.
# The budget has to be reachable at full resolution or the controller can
# never sit at 1.0 and the image is permanently resampled -- which is what
# VIM-31 was. The worst full-resolution frame observed in the field log cost
# ~28ms end to end (render + encode + the terminal's own texture upload) at
# 2968x1856; that sample was the first frame after an idle pause, so it is an
# upper bound rather than the steady-state cost, and the budget is set to
# clear it with headroom. The old 120 fps (8.3ms) target could not be met at
# full resolution by any measurement taken, so it bought frame rate nobody
# asked for with sharpness everybody notices -- input arrives at 60 Hz at
# best, so frames past that are discarded anyway. 30 fps is smooth for
# orbiting a molecule.
_INTERACT_BUDGET = 1.0 / 30.0
# Lowest interactive render scale. This is a legibility floor, not a
# performance one: below about half resolution the terminal's upscale turns
# the molecule to mush, and a frame that cheap is almost never render-bound
# anyway (encode+transmit dominate and do not shrink with resolution).
_SCALE_FLOOR = 0.5
# Consecutive over-budget frames required before giving up resolution. One
# slow frame is not evidence of a too-high resolution -- the first frame after
# an idle pause pays for a cold cache/GPU wake-up and can cost several times
# the budget all by itself (VIM-31).
_SLOW_FRAMES_BEFORE_STEP_DOWN = 2
# Budget (seconds) for a *settle* frame: render + encode + transmit of the
# crisp full-quality still that lands after you stop moving. On a slow link
# (SSH) a full-resolution frame is megabytes and can take seconds to arrive;
# the idle-scale controller trades resolution to keep the settle under this.
_IDLE_BUDGET = 0.30
# A probe round trip at or under this is "clearly local" (same machine or
# LAN): start idle frames at full resolution. Anything slower -- or no
# answer -- starts at half resolution and earns its way up via the idle
# controller as fast settles are actually measured.
_LOCAL_RTT = 0.02
# The terminal's DA1 reply (CSI ? ... c), used as a delivery fence after each
# settle frame -- see _arm_settle_fence for why write() timing can't be
# trusted for a one-shot frame. Give up on an unanswered fence after this
# long (only non-answering terminals ever hit it; DA1 is universal).
_DA1_REPLY = re.compile(rb"\x1b\[\?[0-9;]*c")
_FENCE_TIMEOUT = 5.0

# Opt-in per-frame timing log, for diagnosing latency/jitter over slow links
# (SSH). Off unless VIMOL_TIMING is set; then every frame appends one line to
# $VIMOL_TIMING (or ~/.vimol-timing.log), and quitting prints the session's
# worst per-stage latencies. `tail -f` the log in a second pane while
# interacting. The whole facility is a no-op -- one boolean check per frame --
# when the env var is unset, so it is safe to ship enabled-by-flag.
_TIMING_DEFAULT_LOG = os.path.expanduser("~/.vimol-timing.log")


def _timing_log_path() -> Optional[str]:
    """The timing-log path if VIMOL_TIMING is set, else None (feature off).

    ``VIMOL_TIMING=1`` (or any non-path truthy value) logs to the default
    path; ``VIMOL_TIMING=/some/file`` logs there instead.
    """
    val = os.environ.get("VIMOL_TIMING")
    if not val:
        return None
    return val if os.sep in val else _TIMING_DEFAULT_LOG
# Starting interactive render scale. Deliberately conservative: the FIRST
# frame is drawn with it before any timing data exists, and it must clear
# the 50 ms time-to-first-image target even cold (first-call numpy warmup,
# ~25 ms) on a full-screen Retina terminal -- measured ~40 ms at 0.3, vs
# ~180 ms at 0.5 and ~880 ms for the old full-quality first frame. The
# controller then ramps it to whatever this machine actually sustains
# within a few frames, and the settle path replaces it with a crisp still
# ~0.25 s after startup anyway.
_STARTUP_SCALE = 0.3


class Viewer:
    def __init__(self, molecule: Molecule, frames: Optional[List[Molecule]] = None,
                 style: Optional[Style] = None, fd_in: int = 0, fd_out: int = 1,
                 autospin: bool = False, target_fps: float = 120.0, picking: bool = True,
                 transparent: bool = True, backend: str = "auto",
                 source_path: Optional[str] = None, editable: bool = False,
                 probe: Optional[kitty.TerminalProbe] = None,
                 structures: Optional[StructureSet] = None,
                 auto_bonds: bool = True, bond_tolerance: float = 0.45):
        self.source_path = source_path
        self.editable = editable
        self._auto_bonds = auto_bonds
        self._bond_tolerance = bond_tolerance
        # Frame 0 draws before the startup probe's OSC 11 reply can possibly
        # be in hand (the probe itself runs after the first paint -- see
        # _finish_startup), so this is a synchronous best guess: COLORFGBG, or
        # else the last theme a real OSC 11 reply confirmed (most sessions
        # run in the same terminal repeatedly, so this is usually right and
        # is what keeps a light terminal from flickering dark-then-light on
        # every startup), or else DARK. _finish_startup upgrades it once the
        # probe replies.
        self.theme = theme.resolve(os.environ.get("VIMOL_THEME"), None,
                                   os.environ.get("COLORFGBG"), theme.read_cached())
        # Set once the user presses ctrl-t: a manual choice outranks a probe
        # reply that lands afterwards (see _apply_probe_theme).
        self._theme_pinned = False
        if structures is not None:
            self.structures = structures
        else:
            frame_list = frames or [molecule]
            if auto_bonds:
                for m in frame_list:
                    ensure_bonds(m, tolerance=bond_tolerance)
            self.structures = StructureSet()
            basename = os.path.basename(source_path) if source_path else None
            multi = len(frame_list) > 1
            for i, m in enumerate(frame_list):
                stem = basename or (m.name or "structure")
                label = f"{stem}#{i + 1}" if multi else stem
                self.structures.append(m, label=label, path=source_path)
        self.style = style or Style()
        if style is None:
            # default to a terminal-matching transparent background
            self.style.transparent = transparent
        self.fd_in = fd_in
        self.fd_out = fd_out
        self.autospin = autospin
        self.target_fps = target_fps

        self.widget = MoleculeWidget(self.structures, 320, 240, style=self.style,
                                     supersample=1, picking=picking, backend=backend,
                                     editable=editable)
        self.widget.theme = self.theme.name
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
        # "quit_confirm" | "periodic_table" | "geometry_picker" |
        # "selection_picker"
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
        # Top-right copy control, in terminal-cell coords.
        self._copy_xyz_span = None
        # geometry/hybridization picker: the options list and cursor index
        self._geom_opts: List = []
        self._geom_idx = 0
        self._selection_menu_idx = 0
        # file browser: the directory on screen, the cursor in it, and
        # the first visible row. _browser_last_dir survives a cancel so
        # reopening resumes where you were rather than jumping back.
        self._browser_dir = ""
        self._browser_entries: List[file_browser.Entry] = []
        self._browser_idx = 0
        self._browser_scroll = 0
        self._browser_last_dir: Optional[str] = None
        self._select_hint_span = None
        # True from a mouse-down that landed in the status-bar zone (see
        # _in_status_zone) until the matching up -- keeps a drag that started
        # on the status bar from ever reaching the 3D viewport, even if the
        # pointer strays back over the molecule mid-drag.
        self._status_zone_press = False
        self._last_interact = 0.0
        # dynamic-resolution factor while moving; starts conservative so the
        # very first frame (drawn before any timing data exists) stays fast.
        # Exception: a GPU backend on a provably local terminal (the probe's
        # shm handshake proves same-machine) renders a full-res frame in
        # ~5 ms -- far inside the interactive budget -- so it starts at full
        # resolution with no blurry ramp-up; the controller still steps the
        # scale down if measured frames actually blow the budget.
        self._interact_scale = (1.0 if (self.widget.scene.backend == "gl"
                                        and probe is not None and probe.shm)
                                else _STARTUP_SCALE)
        # resolution factor for the crisp settle frame; probe-seeded in
        # _finish_startup, then adapted by _next_idle_scale measurements.
        self._idle_scale = 1.0
        # consecutive over-budget interactive frames; see _next_render_scale.
        self._slow_frames = 0
        # capability probe handed in by the CLI (it may have probed already
        # to decide whether to launch at all); None -> probe in _finish_startup.
        self._probe = probe
        # How frame pixels reach the terminal: "direct" (base64 escape codes)
        # or "shm" (a shared-memory name), chosen in _finish_startup from the
        # probe. See _draw for why this is what governs whether interactive
        # frames can afford full resolution at all.
        self._transmit = "direct"
        # frame delivery fence (see _arm_fence): when it went out, which kind
        # of frame it trails ("interact" | "settle"), that frame's local
        # (render+encode+write) cost, and a tail of raw input scanned for the
        # terminal's reply.
        self._fence_t0 = None
        self._fence_kind = ""
        self._fence_base = 0.0
        self._fence_buf = b""
        # Self-clocked frame pacing: when True (the startup probe confirmed
        # the terminal answers DA1), every frame is fenced and the next one
        # is not rendered until the previous is acknowledged -- at most ONE
        # frame in flight. Without this, a slow link's kernel/SSH buffers
        # queue dozens of stale frames (they absorb megabytes before write()
        # ever blocks) and the terminal plays back a seconds-old backlog
        # while you drag: the classic bufferbloat jitter.
        self._paced = False
        self._link_rtt = 0.0
        # Late-probe mode: the startup probe's quick window missed the reply
        # (a congested link can stall seconds); keep watching the input
        # stream for it instead of concluding "no fence support". While
        # active, raw input is held in _late_buf -- the graphics-query reply
        # (an APC string) would otherwise decode as garbage KeyEvents -- and
        # released to the decoder when the reply lands or the watch times out.
        self._late_t0 = None
        self._late_buf = b""
        # True while a mouse button is held anywhere: a drag IS interaction
        # even when a congested link delivers its motion events in bursts
        # with long silent gaps -- without this, those gaps faked "idle" and
        # injected the heavy supersampled settle still mid-drag.
        self._button_held = False
        # How long after the last interaction the settle still may fire.
        # Stretched with the link RTT once known: on a bursty link, event
        # gaps shorter than a couple of round trips prove nothing.
        self._idle_after = 0.25
        # Opt-in timing instrumentation (VIMOL_TIMING); see _timing_log_path.
        self._timing_path = _timing_log_path()
        self._tlog = None
        self._tmax = {}          # stage name -> worst ms seen
        self._t_last_draw_end = None
        self._t_last_read = None
        self._cols = self._rows = 0
        self._img_cols = self._img_rows = 1
        self._cell_px = None                 # exact (cw, ch) from the terminal, if it answered
        self._old_termios = None
        self._geometry_established = False   # True once the real (not placeholder) size is known
        # structure-list strip (design §4): column width (0 -- no strip --
        # unless len(structures) > 1), the image's placement origin once the
        # strip reserves columns on its left, and the keyboard-focus state.
        self._list_w = 0
        self._img_origin_px = (0, 0)
        self._list_focused = False
        # (screen_row, col_start, col_end) per DRAWN structure row, refreshed
        # by every _draw_list() call and used for click hit-testing, with
        # _list_row_struct[k] the structure index row k belongs to (group
        # headers and off-screen rows appear in neither).
        self._list_row_spans: List[Tuple[int, int, int]] = []
        self._list_row_struct: List[int] = []
        # (screen row, col start/end, first/last structure index) for each
        # visible per-file ALL button.
        self._list_group_all_spans: List[Tuple[int, int, int, int, int]] = []
        # Same shape, for each per-file × (design VIM-32): removes every
        # entry in [first, end) from the pane. The global ALL (row 0, the
        # "STRUCTURES N" header) is drawn separately and never gets one.
        self._list_group_remove_spans: List[Tuple[int, int, int, int, int]] = []
        # (screen row, col start, col end) for the global ALL button on the
        # "STRUCTURES N" header row (design VIM-28) -- kept separate from
        # _list_group_all_spans so existing per-file span indices/counts are
        # untouched by its presence.
        self._global_all_span: Optional[Tuple[int, int, int]] = None
        # Exact visible filename/frame-label hit boxes.  The associated text
        # is always the source path exactly as supplied by the caller (plus a
        # group-local ``#frame N`` suffix for frame rows), never the compact
        # or middle-truncated display name.
        self._list_path_hover_spans: List[Tuple[int, int, int, str]] = []
        self._list_path_hover_tip = ""
        # File groups collapsed to one ``N frames`` summary row. Keys use the
        # same stable identity as grouping, so scrolling or changing the main
        # frame never loses collapse state.
        self._collapsed_groups = set()
        self._list_group_toggle_spans: List[Tuple[int, int, int, int, int]] = []
        self._list_group_summary_spans: List[Tuple[int, int, int, int, int]] = []
        # One past the last row the strip painted last time. Collapsing can
        # shrink the strip by hundreds of rows at once, and nothing else
        # repaints that column, so _draw_list clears back down to here.
        self._list_drawn_bottom = 0
        # index of the first VISIBLE display row (see _list_display_rows):
        # the strip scrolls, so a 100-frame trajectory is reachable.
        self._list_scroll = 0
        # the row cursor: distinct from active_index (design §4.3) so
        # j/k/1-9/space/z/h have an unambiguous target while list-focused.
        self._list_cursor = self.structures.active_index
        # True from a mouse-down that landed on the strip until the matching
        # up -- mirrors _status_zone_press so a drag started on the strip
        # never reaches the viewport.
        self._list_zone_press = False
        # measurement table (design 2026-07-30 rev. 2, VIM-6): FROZEN columns
        # as (header cell text, active-local atom indices) -- the live pick
        # list (widget.measure_sel) itself renders as one more, un-frozen
        # column at the end whenever it holds 2+ atoms, updating in place as
        # it grows (distance -> angle -> dihedral). _measure_w is the extra
        # width the table currently reserves next to the strip, and
        # _measure_header_spans is the click hit-test for each FROZEN
        # column's × (the live column has none -- nothing to remove yet).
        self._measure_columns: List[Tuple[str, Tuple[int, ...]]] = []
        self._full_rmsd_columns: List[_FullRMSDColumn] = []
        self._rmsd_columns: List[_SubsetRMSDColumn] = []
        # One counter shared by both column kinds (design 2026-08-02): a
        # ∀RMSD and a ⊂RMSD column must never be numbered alike.
        self._next_rmsd_id = 1
        # Set by _activate_selection_preset when a real preset (not Manual)
        # is chosen; consumed (and cleared) by _finish_subset_alignment onto
        # the new column's own preset_label. Cleared early if the pick is
        # hand-edited before it's ever committed, so a since-modified
        # selection never gets credited to the preset that started it.
        self._pending_selection_label: Optional[str] = None
        self._active_subset_id: Optional[int] = None
        self._subset_hover_tip = ""
        self._full_rmsd_hover_tip = ""
        self._measure_w = 0
        self._measure_header_spans: List[Tuple[int, int, int, int]] = []
        self._subset_header_spans: List[Tuple[int, int, int, int]] = []
        self._subset_remove_spans: List[Tuple[int, int, int, int]] = []
        self._full_rmsd_header_spans: List[Tuple[int, int, int, int]] = []
        self._full_rmsd_remove_spans: List[Tuple[int, int, int, int]] = []
        self._measure_layout_sources: List[Tuple[str, int]] = []
        # (key, layout) memo for _measure_layout -- it's recomputed every
        # _update_geometry tick (see there) plus once per _draw_list, and
        # its body calls StructureSet.measure() per column, each of which
        # walks every entry's full symbol list. Unchanged inputs must not
        # pay that cost dozens of times a second at idle with a column
        # pinned; the key covers everything the body reads.
        self._measure_layout_cache = None
        # Set whenever _measure_columns changes outside of _update_geometry's
        # own polling (i.e. by _refresh_measure_w) so the NEXT _update_geometry
        # call still recomputes image size/origin and triggers a full clear --
        # otherwise, since _refresh_measure_w already syncs _measure_w for the
        # sake of an immediate hit-test, _update_geometry's own before/after
        # comparison would see no difference and skip that work entirely.
        self._geometry_dirty = False
        # True while WE own a pushed OSC-22 pointer shape (delete's crosshair,
        # measure's cell). Pushes and pops must pair exactly: an unbalanced pop
        # would clobber a shape pushed by something outside vimol (tmux, the
        # hosting app), and an unbalanced push leaks ours onto their stack.
        self._pointer_pushed = False

    # -- files x frames: one axis, backed by self.structures (design §2) ---
    @property
    def frames(self) -> List[Molecule]:      # deprecated; use .structures
        return self.structures.molecules

    @property
    def frame_index(self) -> int:
        return self.structures.active_index

    @frame_index.setter
    def frame_index(self, i: int) -> None:
        self._activate_structure(i)

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

    def _enable_mouse(self, pixel: bool) -> None:
        """Turn on mouse reporting with the wire format and the decoder locked
        to the same coordinate mode.

        The terminal reports mouse coordinates in *pixels* (SGR-Pixels, DECSET
        1016) or *cells*, and the decoder must be told which. One flag drives
        both sides here, so they can never disagree -- the invariant behind
        the SSH dead-mouse fix (a raced probe once told the decoder "cells"
        while the wire was in pixel mode, scaling every click off-screen).
        """
        self.decoder.pixel = pixel
        kitty.write_bytes(_input.enable_mouse(pixel=pixel, hover=self.widget.picking),
                          self.fd_out)

    def _apply_probe_theme(self, probe) -> None:
        """Re-run the theme ladder now that the probe's OSC 11 answer is in.

        Called from BOTH probe landing sites -- the quick window in
        _finish_startup and the late watch in _late_probe_tick. The late one
        matters most: a congested link is exactly where the quick window
        misses, and skipping the upgrade there would leave those sessions
        stuck on the frame-0 COLORFGBG/DARK guess forever, i.e. auto-detection
        silently doing nothing on the links it was written for.

        A manual ctrl-t always wins: once the user has stated a preference,
        a probe reply landing afterwards must not yank it back.
        """
        resolved = theme.resolve(os.environ.get("VIMOL_THEME"), probe.bg_rgb,
                                 os.environ.get("COLORFGBG"), theme.read_cached())
        # Remember what a REAL OSC 11 answer said, for next run's frame-0
        # guess (see __init__) -- but only that, not an explicit override or
        # the COLORFGBG heuristic, which aren't the terminal telling us
        # anything itself.
        if probe.bg_rgb is not None and not os.environ.get("VIMOL_THEME"):
            theme.write_cached(
                (theme.LIGHT if theme.luminance(probe.bg_rgb) > 140 else theme.DARK).name)
        if self._theme_pinned or resolved is self.theme:
            return
        self.theme = resolved
        self.widget.theme = resolved.name
        # Repaint rather than wait for the next settle: the chrome that DOES
        # paint an opaque background (status bar, help, pickers) would
        # otherwise sit in the wrong palette until something else happened to
        # force a frame, which reads as a delayed colour flash.
        kitty.write_bytes(_CLEAR, self.fd_out)
        self._last_interact = time.time()

    def _finish_startup(self) -> None:
        """Probe the terminal and arm the mouse -- AFTER the first paint.

        Runs the combined capability probe (pixel mouse, exact cell size, link
        round-trip time -- one write, one reply, see kitty.probe_terminal)
        deliberately after the first frame is already on screen: the probe
        costs a link round trip, and nothing it answers is needed to *draw* --
        only to point, click, and pick the settle resolution. So startup shows
        pixels immediately and spends the RTT while the user is still looking
        at them. Keystrokes typed during the probe are preserved (the reply
        parser returns them as leftover bytes) and dispatched, not dropped.
        """
        probe = self._probe
        if probe is None and self._old_termios is not None:
            # Quick window only: a local terminal answers in ~1ms, and a
            # link too congested to answer in this window is handled by the
            # LATE watch below -- a longer blocking wait here would just
            # freeze startup on bad links and still sometimes miss.
            probe = kitty.probe_terminal(self.fd_in, self.fd_out, timeout=0.35)
            self._probe = probe
        if probe is not None:
            if probe.cell_px is not None:
                self._cell_px = probe.cell_px
            self._apply_probe_theme(probe)
            self._enable_mouse(probe.pixel_mouse)
            # Seed the settle resolution from the measured link latency: a
            # clearly-local terminal starts crisp at full resolution; a
            # remote/unknown link starts at half (its megabyte-scale full-res
            # stills are what made SSH feel frozen) and the idle controller
            # raises it as fast settles are actually observed.
            self._idle_scale = 1.0 if (probe.rtt is not None
                                       and probe.rtt <= _LOCAL_RTT) else 0.5
            # A terminal that accepted our shared-memory object is provably on
            # this machine: send pixels that way and skip the per-frame zlib
            # (~24 of the ~27 ms a full-resolution 1600x1000 frame used to
            # cost). That is what lets _next_render_scale ride near 1.0 while
            # you drag instead of settling around 0.34 -- the interactive
            # blur was never bandwidth, it was compressing every frame.
            # shm implies a replying terminal, hence _paced below, so exactly
            # one frame is ever in flight. A wedged terminal (fence timeout)
            # is bounded separately by kitty._SHM_KEEP, which only ever
            # recycles stale frames -- see the test for that invariant.
            self._transmit = "shm" if probe.shm else "direct"
            if self._transmit == "shm" and self.widget.scene.backend == "gl":
                # GPU + provably-local terminal (see __init__): interactive
                # frames fit the budget at full resolution, so jump straight
                # there instead of earning it back over ~10 measured frames.
                self._interact_scale = 1.0
            if probe.rtt is not None:
                # the terminal answers DA1: fence every frame and self-clock
                # to the link (one frame in flight, no bufferbloat backlog).
                self._paced = True
                self._link_rtt = probe.rtt
                self._idle_after = min(1.0, 0.25 + 2.0 * probe.rtt)
            elif self._old_termios is not None:
                # The reply is late, not absent (this happens on exactly the
                # congested links that need pacing most -- concluding "no
                # support" here re-enables the bufferbloat firehose). The
                # queries are already on the wire; keep watching for the
                # DA1 in the run loop and switch pacing on when it lands.
                self._late_t0 = time.monotonic()
                self._late_buf = b""
            if probe.leftover:
                # mouse reporting was off during the probe, so leftover bytes
                # can only be keystrokes -- deliver them.
                self._dispatch(self.decoder.feed(probe.leftover))
        else:
            # no way to probe (stdin not a tty): cells on both sides, and
            # keep full-resolution settles (the status quo for local use).
            self._enable_mouse(False)

    def _late_probe_tick(self, data: bytes) -> None:
        """Watch for the startup probe's late reply; upgrade when it lands.

        While active, raw input is buffered rather than decoded: the
        graphics-query reply is an APC string the decoder would shred into
        garbage KeyEvents ('3' and '1' in ``Gi=31;OK`` would switch the
        representation!). Everything the user typed is released to the
        decoder the moment the reply arrives; Ctrl-C still quits instantly.
        On the reply: pacing on, RTT recorded, pixel mouse upgraded if
        supported. After 15s of silence, give up and release the (stripped)
        buffer -- an unpaced session on a mute terminal beats a frozen one.
        """
        if b"\x03" in data:                      # emergency quit, never held
            self._running = False
            return
        self._late_buf += data
        p = kitty.parse_probe_reply(self._late_buf) if data else None
        if p is not None:
            self._late_t0 = None
            self._late_buf = b""
            if p.cell_px is not None:
                self._cell_px = p.cell_px
            self._apply_probe_theme(p)
            if p.pixel_mouse and not self.decoder.pixel:
                self._enable_mouse(True)         # upgrade wire+decoder together
            self._paced = True
            self._idle_scale = 0.5               # reply was slow: assume remote
            self._idle_after = 1.0               # and assume bursty input gaps
            # A late reply's age includes user think-time, so it is useless
            # as an RTT sample; leave _link_rtt 0 and let the per-frame acks
            # min-track the real baseline (every ack wait >= true RTT).
            if p.leftover:
                self._dispatch(self.decoder.feed(p.leftover))
            return
        if time.monotonic() - self._late_t0 > 15.0:
            # never answered: release what the user typed, minus anything
            # that looks like a stray probe reply, and stay unpaced.
            buf = kitty.strip_probe_replies(self._late_buf)
            self._late_t0 = None
            self._late_buf = b""
            if buf:
                self._dispatch(self.decoder.feed(buf))

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
        # A POSIX shm object outlives its creator, and the frame we wrote just
        # before quitting may never have been read: unlink what's left.
        kitty.shm_cleanup()
        # With VIMOL_TIMING on, print the session's worst per-stage latencies
        # to the shell after quitting, and record them in the log too.
        summary = self._tsummary()
        if summary:
            self._tline("worst", **{k: v for k, v in self._tmax.items()})
            try:
                os.write(self.fd_out, (summary + f"\n(log: {self._timing_path})\n").encode())
            except OSError:
                pass

    # -- geometry ---------------------------------------------------------
    def _list_width(self) -> int:
        """Column width of the structure-list strip (design §4.1); 0 (no
        strip) unless more than one structure is loaded."""
        if len(self.structures) <= 1:
            return 0
        return min(_LIST_W_MAX, max(_LIST_W_MIN, self._cols // 5))

    def _in_list_zone(self, col: int) -> bool:
        """True for any column inside the structure-list strip OR the
        measurement table beside it (design §4.2, extended 2026-07-30) --
        the horizontal twin of _in_status_zone.

        Reads the cached ``_measure_w`` rather than recomputing: a full
        recompute runs ``StructureSet.measure`` per column, too costly to
        redo on every mouse move at hundreds of structures. ``_update_geometry``
        refreshes it once a tick, and ``_refresh_measure_w`` refreshes it
        immediately on commit/removal, so it is never more than one
        dispatch-call stale."""
        return self._list_w > 0 and col < self._list_w + self._measure_w

    def _list_index_at_row(self, row: int) -> Optional[int]:
        """The structure index whose row was drawn at screen *row*, else None.

        Group headers and the header/separator/legend rows own no structure,
        so they answer None -- a click on them is inert. The mapping is the
        one _draw_list actually emitted (_list_row_struct), never row
        arithmetic redone here: with group headers and a scroll offset in
        play, recomputing it is exactly how the two silently drift apart."""
        for k, (r0, _c0, _c1) in enumerate(self._list_row_spans):
            if r0 == row:
                return self._list_row_struct[k]
        return None

    def _list_group_key(self, i: int):
        """What decides whether structure *i* shares a file with its
        neighbours: the source path, or -- for in-memory structures with no
        path -- the '<stem>#k' label prefix the multi-model loader assigns."""
        entry = self.structures[i]
        if entry.path is not None:
            return ("path", entry.path)
        if "#" in entry.label:
            return ("label", entry.label.rsplit("#", 1)[0])
        return ("index", i)          # nothing it could be grouped with

    def _list_path_display_names(self) -> dict:
        """Shortest unique trailing path for every path-backed file.

        Start at the basename.  Whenever two displayed names still collide,
        prepend one parent node to just those names and repeat.  Thus
        ``a/run/mol.xyz`` and ``b/run/mol.xyz`` become
        ``a/run/mol.xyz`` / ``b/run/mol.xyz``, while an unrelated
        ``water.xyz`` stays exactly ``water.xyz``.
        """
        paths = list(dict.fromkeys(
            entry.path for entry in self.structures if entry.path is not None))
        normalized = {path: os.path.normpath(path) for path in paths}
        names = {
            path: os.path.basename(normalized[path]) or normalized[path]
            for path in paths
        }
        parents = {path: os.path.dirname(normalized[path]) for path in paths}

        while True:
            counts = Counter(names.values())
            colliding = [path for path in paths if counts[names[path]] > 1]
            if not colliding:
                break
            changed = False
            for path in colliding:
                parent = parents[path]
                node = os.path.basename(parent)
                if node:
                    names[path] = os.path.join(node, names[path])
                    parents[path] = os.path.dirname(parent)
                    changed = True
                elif names[path] != normalized[path]:
                    # Preserve a leading root separator when the complete
                    # absolute path is the only remaining differentiator.
                    names[path] = normalized[path]
                    changed = True
                elif names[path] != path:
                    names[path] = path
                    changed = True
            if not changed:
                # Repeated references to the exact same normalized path are
                # one filename; there is no higher directory that can make
                # them different, so do not loop forever.
                break
        return names

    def _list_group_name(self, i: int, grouped: bool,
                         path_names: Optional[dict] = None) -> str:
        """The display name for structure *i*'s file: its basename, or the
        '<stem>#k' label's stem when there is no path. A structure that is
        alone in its group and has no path keeps its full label -- the label
        is what identifies it, and nothing else on the strip repeats it."""
        entry = self.structures[i]
        if entry.path is not None:
            if path_names is None:
                path_names = self._list_path_display_names()
            return path_names[entry.path]
        if grouped and "#" in entry.label:
            return entry.label.rsplit("#", 1)[0]
        return entry.label

    def _list_display_rows(self) -> List[Tuple[str, Optional[int], str]]:
        """The strip's rows, top to bottom, as ``(kind, structure_index,
        text)`` -- the one place display rows and structure indices are
        related (design §4.1).

        A file contributing several structures becomes a non-selectable
        ``("group", first_index, basename)`` header followed by one
        ``("struct", i, "frame k")`` row per model, so a 100-frame trajectory
        names its file once instead of a hundred times. Path-backed singleton
        files also receive a section whenever several files are open or the
        viewer is in overlay mode. Runs are consecutive: structures load in
        file order, and grouping across a gap would reorder the strip.
        """
        rows: List[Tuple[str, Optional[int], str]] = []
        n = len(self.structures)
        path_names = self._list_path_display_names()
        path_groups = set()
        for k in range(n):
            key = self._list_group_key(k)
            if key[0] == "path":
                path_groups.add(key)
        i = 0
        while i < n:
            key = self._list_group_key(i)
            j = i + 1
            while j < n and self._list_group_key(j) == key:
                j += 1
            sectioned = (j - i > 1
                         or (key[0] == "path"
                             and (self.structures.overlay or len(path_groups) > 1)))
            if sectioned:
                rows.append(("group", i, self._list_group_name(
                    i, True, path_names)))
                if key in self._collapsed_groups:
                    rows.append(("collapsed", i, f"{j - i} frames"))
                else:
                    rows.extend(("struct", m, f"frame {m - i + 1}")
                                for m in range(i, j))
            else:
                rows.append(("struct", i, self._list_group_name(
                    i, False, path_names)))
            i = j
        return rows

    def _list_group_end(self, first: int) -> int:
        """Exclusive end of the consecutive file group starting at *first*."""
        key = self._list_group_key(first)
        end = first + 1
        while end < len(self.structures) and self._list_group_key(end) == key:
            end += 1
        return end

    def _list_group_all_hit(self, col: int, row: int):
        for r0, c0, c1, first, end in self._list_group_all_spans:
            if r0 == row and c0 <= col < c1:
                return first, end
        return None

    def _list_group_remove_hit(self, col: int, row: int):
        for r0, c0, c1, first, end in self._list_group_remove_spans:
            if r0 == row and c0 <= col < c1:
                return first, end
        return None

    def _global_all_hit(self, col: int, row: int) -> bool:
        span = self._global_all_span
        return bool(span is not None and row == span[0] and span[1] <= col < span[2])

    def _list_path_hover_hit(self, col: int, row: int) -> str:
        for r0, c0, c1, tip in self._list_path_hover_spans:
            if r0 == row and c0 <= col < c1:
                return tip
        return ""

    def _list_frame_path_tip(self, i: int) -> str:
        """Original input path plus this entry's group-local frame number."""
        entry = self.structures[i]
        if entry.path is None:
            return ""
        key = self._list_group_key(i)
        first = i
        while first > 0 and self._list_group_key(first - 1) == key:
            first -= 1
        return f"{entry.path}#frame {i - first + 1}"
    @staticmethod
    def _list_group_span_hit(spans, col: int, row: int):
        for r0, c0, c1, first, end in spans:
            if r0 == row and c0 <= col < c1:
                return first, end
        return None

    def _toggle_list_group_collapsed(self, first: int, end: int) -> None:
        key = self._list_group_key(first)
        if key in self._collapsed_groups:
            self._collapsed_groups.remove(key)
            state = "expanded"
        else:
            self._collapsed_groups.add(key)
            state = "collapsed"
        # The display can shrink from hundreds of rows to two. Clamp now,
        # rather than leaving one draw where the file vanishes below a stale
        # scroll offset.
        self._list_scroll = min(self._list_scroll, self._list_max_scroll())
        self._list_focused = True
        self._msg = f"{self._list_group_name(first, True)}: {state}"

    @staticmethod
    def _list_collapsed_measure_cells(layout, first: int, end: int) -> List[str]:
        """Local minimum cell for every column across ``[first, end)``."""
        result = []
        for _header, _width, cells, _removable in layout:
            candidates = []
            for cell in cells[first:end]:
                bare = cell.rstrip("↑↓")
                try:
                    number = float(bare)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    candidates.append((number, bare))
            if not candidates:
                result.append("—")
            else:
                _number, formatted = min(candidates, key=lambda item: item[0])
                result.append(formatted + "↓")
        return result

    def _toggle_list_group_all(self, first: int, end: int,
                               label: Optional[str] = None) -> None:
        """Fill a file's overlay membership, or reduce it to the main frame.

        *label* names the range in status messages; it defaults to the file
        at *first*, but the global ALL (design VIM-28) calls this with
        ``(0, len(structures))`` and its own label -- naming it after
        whichever file happens to start the set would be misleading."""
        sset = self.structures
        label = label if label is not None else self._list_group_name(first, True)
        all_selected = all(i == sset.active_index or sset[i].marked
                           for i in range(first, end))
        if all_selected:
            for i in range(first, end):
                sset[i].marked = False
            # Never leave overlay=True with no marks: StructureSet interprets
            # that as "draw everything". The main frame is always retained.
            sset.active.marked = True
            # The main frame need not live in the file being cleared -- saying
            # it does would credit this file with a row it does not own.
            held = first <= sset.active_index < end
            self._msg = "%s: %s" % (label, "main frame only" if held else "hidden")
        else:
            for i in range(first, end):
                sset[i].marked = True
            self._msg = f"{label}: all selected"
        sset.overlay = True
        sset.invalidate()
        self._list_focused = True

    def _remove_file(self, first: int, end: int) -> None:
        """Delete every entry in one file group -- all its frames disappear
        from the pane in one step (design VIM-32).

        Removing the very last file exits the viewer exactly as 'q'/Escape
        already do, including the unsaved-changes prompt: discarding unsaved
        edits silently would be the one genuinely destructive reading here.

        The entries themselves (plus the active index and solo state that go
        with them) are StructureSet's to re-base -- see
        ``StructureSet.remove_range``. What stays here is the state only the
        Viewer holds: the row cursor, and both RMSD column kinds, which store
        absolute entry indices (``values`` is a per-entry positional list,
        ``reference_index`` an absolute pointer). A column anchored inside
        the removed range is dropped -- its reference no longer means
        anything -- and one anchored after it is re-based with the same
        arithmetic rather than left pointing at the wrong row.
        """
        sset = self.structures
        if end - first >= len(sset):
            if self.editable and self.widget.dirty:
                self._mode = "quit_confirm"
            else:
                self._running = False
            return
        name = self._list_group_name(first, True)
        active_entry = sset.entries[sset.active_index]

        sset.remove_range(first, end)
        remaining = len(sset)
        self._list_cursor = sset.index_after_removal(
            self._list_cursor, first, end, remaining)
        self._list_scroll = min(self._list_scroll, self._list_max_scroll())

        def _reindex(columns):
            kept = []
            for column in columns:
                if first <= column.reference_index < end:
                    continue    # its reference frame is gone
                del column.values[first:end]
                column.reference_index = sset.index_after_removal(
                    column.reference_index, first, end, remaining)
                kept.append(column)
            return kept

        self._full_rmsd_columns = _reindex(self._full_rmsd_columns)
        self._rmsd_columns = _reindex(self._rmsd_columns)
        if self._active_subset_id is not None and not any(
                c.select_id == self._active_subset_id for c in self._rmsd_columns):
            self._active_subset_id = None
            self.widget.align_sel = []
        # A tip left over from a column this removal just dropped would
        # otherwise sit in the status bar until the next mouse move.
        self._subset_hover_tip = ""
        self._full_rmsd_hover_tip = ""

        self._measure_layout_cache = None
        self._refresh_measure_w()
        self._geometry_dirty = True
        # Only refit the camera / reset editor state (undo, dirty, hover) if
        # the active STRUCTURE itself changed identity -- an unrelated file's
        # removal must not discard the active file's edit history.
        if sset.entries[sset.active_index] is not active_entry:
            self.widget.refresh_active()
        self._msg = f"{name}: removed"

    # -- strip scrolling ---------------------------------------------------
    def _list_capacity(self) -> int:
        """How many display rows the strip can show at the current height.

        The single source of truth for the entry block's height: _draw_list
        slices the rows with it, and the scroll clamps are computed from it,
        so the two cannot disagree about what fits."""
        max_row = max(self._rows - 1, 0)     # never draw over the status bar
        return max(1, max_row - _LIST_ROWS_ABOVE - _LIST_ROWS_BELOW)

    def _list_max_scroll(self) -> int:
        return max(0, len(self._list_display_rows()) - self._list_capacity())

    def _list_scroll_to(self, first: int) -> bool:
        """Put display row *first* at the top of the strip, clamped. True if
        the offset actually moved."""
        first = max(0, min(self._list_max_scroll(), int(first)))
        if first == self._list_scroll:
            return False
        self._list_scroll = first
        return True

    def _list_scroll_by(self, delta: int) -> bool:
        return self._list_scroll_to(self._list_scroll + delta)

    def _list_ensure_visible(self, i: int) -> None:
        """Scroll the minimum needed to bring structure *i*'s row into view.

        Works in DISPLAY rows, not structure indices -- group headers push
        the two apart, and doing this arithmetic on the structure index is
        exactly how a cursor ends up 'visible' at the wrong row."""
        rows = self._list_display_rows()
        row = None
        for r, (kind, k, _text) in enumerate(rows):
            if kind == "struct" and k == i:
                row = r
                break
            if (kind == "collapsed" and k is not None
                    and k <= i < self._list_group_end(k)):
                row = r
                break
        if row is None:
            return
        # scrolling up to the first frame of a file brings its header along:
        # a bare 'frame 7' with no filename above it says nothing.
        top = row - 1 if (row and rows[row - 1][0] == "group") else row
        cap = self._list_capacity()
        if top < self._list_scroll:
            self._list_scroll_to(top)
        elif row >= self._list_scroll + cap:
            self._list_scroll_to(row - cap + 1)

    @staticmethod
    def _truncate_middle(s: str, width: int) -> str:
        """Middle-truncate *s* to *width* visible columns, keeping the file
        extension legible (design §4.1) -- 'traj_2024_long.xyz' -> 'tra….xyz'."""
        if width <= 0:
            return ""
        if len(s) <= width:
            return s
        if width == 1:
            return "…"
        left = (width - 1 + 1) // 2
        right = width - 1 - left
        return s[:left] + "…" + (s[-right:] if right else "")

    @staticmethod
    def _sgr_fg(rgb) -> str:
        return "\x1b[38;2;%d;%d;%dm" % tuple(rgb)

    @staticmethod
    def _sgr_bg(rgb) -> str:
        return "\x1b[48;2;%d;%d;%dm" % tuple(rgb)

    @classmethod
    def _list_line(cls, segments, width: int, bg=None) -> str:
        """Lay out one strip row as exactly *width* VISIBLE columns.

        *segments* are ``(text, sgr)`` pairs whose text must be escape-free:
        SGR sequences are zero-width, so measuring a pre-decorated string
        with len() is what corrupts the layout. Each styled segment is
        closed with a full reset and the row background re-applied, so a
        segment carrying its own background (a legend key cap) cannot leak
        into the rest of the row."""
        base = cls._sgr_bg(bg) if bg else ""
        parts = [base]
        used = 0
        for text, sgr in segments:
            if used >= width:
                break
            piece = text[:width - used]
            used += len(piece)
            parts.append(f"{sgr}{piece}\x1b[0m{base}" if sgr else piece)
        parts.append(" " * (width - used))
        parts.append("\x1b[0m")
        return "".join(parts)

    def _list_cap(self, key: str):
        """A legend "key cap": the key text padded one space either side on a
        lighter background."""
        return (f" {key} ", self._sgr_bg(self.theme.list_cap_bg) + self._sgr_fg(self.theme.list_label_fg))

    def _list_legend(self):
        """The legend's rows, as segment lists (design §4.1). Sized to fit the
        narrowest strip (_LIST_W_MIN); wider strips just leave more air."""
        muted = self._sgr_fg(self.theme.list_muted_fg)
        cap = self._list_cap
        select_style = (self._sgr_bg(self.theme.list_cap_bg)
                        + self._sgr_fg(self.theme.list_label_fg) + "\x1b[1m")
        return [
            [(" ", ""), cap("1"), ("-", muted), cap("9"), (" jump to", muted)],
            [(" ", ""), cap("n"), cap("p"), (" next/prev", muted)],
            [(" ", ""), cap("z"), (" solo ", muted), cap("h"), (" hide", muted)],
            [(" Shft+S to ", muted), ("select", select_style)],
        ]

    def _draw_copy_controls(self) -> bytes:
        """Draw the right-anchored copy pill and publish its hit box."""
        self._copy_xyz_span = None
        xyz = " ⧉ XYZ "
        right = max(self._cols - 1, 0)  # one quiet cell at the terminal edge
        left = right - len(xyz)
        # Never paint over the structure/measurement header on a terminal too
        # narrow to accommodate both regions honestly.
        if left < self._list_w + self._measure_w or self._rows <= 1:
            return b""

        bg_b = self._sgr_bg(self.theme.measure_col_bg_b)
        fg = self._sgr_fg(self.theme.list_label_fg)
        text = f"{bg_b}{fg}\x1b[1m{xyz}\x1b[22m\x1b[0m"
        self._copy_xyz_span = (0, left, left + len(xyz))
        return (b"\x1b[1;%dH" % (left + 1)
                + text.encode("utf-8", "replace"))

    @staticmethod
    def _plain_span_hit(span, col: int, row: int) -> bool:
        return bool(span is not None and row == span[0]
                    and span[1] <= col < span[2])

    def _current_xyz_text(self) -> str:
        """Extended XYZ for exactly the frames currently drawn.

        Alignment transforms are baked into the copied coordinates so a
        pasted multi-frame XYZ reproduces the view's geometry.  Each source
        molecule's original XYZ comment remains its frame comment.
        """
        chunks = []
        for i in self.structures.drawn_indices():
            entry = self.structures[i]
            source = entry.molecule
            copied = Molecule(
                symbols=list(source.symbols),
                positions=entry.transform.apply(source.positions),
                name=source.name,
            )
            chunks.append(xyz_parser.dumps(copied))
        return "".join(chunks)

    def _copy_current_xyz(self) -> None:
        kitty.write_bytes(
            kitty.clipboard_set_text(self._current_xyz_text()), self.fd_out)
        self._msg = "current XYZ copied"

    def _draw_list(self) -> bytes:
        """Render the structure-list strip (design §4.1): ANSI text written
        straight into terminal cells, the same technique as the periodic-
        table/geometry pickers -- costs nothing to render and never touches
        the Kitty image.

        Layout, top to bottom: a muted header (with the scroll affordance),
        a blank row, the scrolled window of display rows, an inset rule, the
        key legend, and the overlay/camera status lines if they still fit.
        Every row goes through _list_line, so all of them measure exactly
        list_w visible columns however narrow the strip gets -- widened by
        the measurement table's own columns when any are pinned (design
        2026-07-30).
        """
        out = bytearray()
        list_w = self._list_w
        sset = self.structures
        self._list_row_spans = []
        self._list_row_struct = []
        self._list_group_all_spans = []
        self._list_group_remove_spans = []
        self._global_all_span = None
        self._list_path_hover_spans = []
        self._list_group_toggle_spans = []
        self._list_group_summary_spans = []
        self._measure_header_spans = []
        self._subset_header_spans = []
        self._subset_remove_spans = []
        self._full_rmsd_header_spans = []
        self._full_rmsd_remove_spans = []
        self._select_hint_span = None
        layout = self._measure_layout(list_w)
        total_w = list_w + self._layout_width(layout)
        max_row = max(self._rows - 1, 0)   # never draw over the status bar

        def put(row0: int, s: str) -> bool:
            """Emit one strip row; False when it fell off the bottom.

            Callers register click spans only for rows this returned True
            for, so what the hit test believes was drawn and what actually
            reached the terminal can never disagree."""
            if row0 >= max_row:
                return False
            out.extend(b"\x1b[%d;1H" % (row0 + 1))
            out.extend(s.encode("utf-8", "replace"))
            # Erase-to-end-of-line as well as the reset: belt-and-suspenders
            # against a shrinking table (a column just removed, or the
            # dedup'd live column disappearing) leaving a wider PREVIOUS
            # frame's content sitting past this frame's narrower row (design
            # 2026-07-30 rev. 2 -- the primary fix is _geometry_dirty above,
            # this just makes the failure mode impossible regardless).
            out.extend(b"\x1b[0m\x1b[K")
            return True

        rows = self._list_display_rows()
        cap = self._list_capacity()
        # Re-clamp here as well as on every scroll: a terminal resize changes
        # the capacity under a stored offset that was legal for the old one.
        self._list_scroll = max(0, min(max(0, len(rows) - cap), self._list_scroll))
        first = self._list_scroll
        visible = rows[first:first + cap]

        muted = self._sgr_fg(self.theme.list_muted_fg)
        head_fg = self._sgr_fg(self.theme.list_header_fg)
        dim_fg = self._sgr_fg(self.theme.list_dim_fg)

        def draw_group(row0: int, group_i: int, text: str) -> bool:
            """Draw a file header plus ALL button and register its hit span."""
            end = self._list_group_end(group_i)
            all_selected = all(k == sset.active_index or sset[k].marked
                               for k in range(group_i, end))
            button = " ALL "
            x_glyph = "×"
            tail = button + x_glyph   # " ALL ×" -- the × removes the whole file
            name = self._truncate_middle(
                text, max(1, list_w - len(tail) - 2))
            # Right-aligned (design VIM-27): flush against list_w rather than
            # immediately after the name, so every file's controls line up
            # in the same column regardless of its filename's length.
            button_col = max(1 + len(name) + 1, list_w - len(tail))
            gap = button_col - (1 + len(name))
            x_col = button_col + len(button)
            button_style = (self._sgr_bg(
                self.theme.measure_col_bg_a if all_selected
                else self.theme.list_cap_bg)
                + self._sgr_fg(self.theme.list_label_fg if all_selected
                               else self.theme.list_muted_fg)
                + ("\x1b[1m" if all_selected else ""))
            x_style = self._sgr_fg(self.theme.list_muted_fg)
            segs = [(" ", ""), (name, head_fg), (" " * gap, ""),
                    (button, button_style), (x_glyph, x_style)]
            if not put(row0, self._list_line(segs, total_w)):
                return False
            entry = sset[group_i]
            if entry.path is not None:
                self._list_path_hover_spans.append(
                    (row0, 1, 1 + len(name), entry.path))
            self._list_group_remove_spans.append(
                (row0, x_col, x_col + len(x_glyph), group_i, end))
            self._list_group_toggle_spans.append(
                (row0, 1, 1 + len(name), group_i, end))
            self._list_group_all_spans.append(
                (row0, button_col, button_col + len(button), group_i, end))
            return True

        def measure_segs(row0: int, raw_cells: List[str], align: str) -> list:
            """Tinted measurement-column segments for one row (design
            2026-07-30 rev. 2). *raw_cells[k]* is column k's unpadded content
            for this row -- the header cell (ending in ' ×' when removable)
            when align is 'left', a formatted value (or '—') when 'right'.
            Interleaves the two theme background shades by column position,
            independent of the row's own active/cursor background. A
            removable header cell also registers its '×' click span at
            row0 -- the live (non-removable) column has no span: nothing to
            click yet."""
            segs = []
            col_offset = list_w
            for k, (_header, width, _values, removable) in enumerate(layout):
                bg = (self.theme.measure_col_bg_a if k % 2 == 0
                      else self.theme.measure_col_bg_b)
                raw = raw_cells[k]
                padded = raw.ljust(width) if align == "left" else raw.rjust(width)
                cell_text = f" {padded} "
                # "Self" and the em-dash are both non-values: keep them muted
                # so the eye lands on the actual numbers.
                fg = (head_fg if align == "left"
                      else muted if raw in ("—", _SELF_CELL) else dim_fg)
                style = self._sgr_bg(bg) + fg
                # Extrema are recomputed by _measure_layout on every data
                # revision.  Keep the underline tight around the value and
                # its arrow rather than underlining the cell's padding too.
                if align == "right" and ("↑" in raw or "↓" in raw):
                    left_pad = " " + (" " * (width - len(raw)))
                    segs.append((left_pad, style))
                    segs.append((raw, style + "\x1b[4m"))
                    segs.append((" ", style))
                else:
                    segs.append((cell_text, style))
                if align == "left" and removable:
                    x_col = col_offset + len(raw)   # raw ends in ' ×': × is its last char
                    self._measure_header_spans.append((row0, x_col, x_col + 1, k))
                if (align == "left" and k < len(self._measure_layout_sources)
                        and self._measure_layout_sources[k][0] == "subset"):
                    subset_idx = self._measure_layout_sources[k][1]
                    self._subset_header_spans.append(
                        (row0, col_offset, col_offset + len(cell_text), subset_idx))
                    if raw.endswith(" ×"):
                        x_col = col_offset + len(raw)
                        self._subset_remove_spans.append(
                            (row0, x_col, x_col + 1, subset_idx))
                if (align == "left" and k < len(self._measure_layout_sources)
                        and self._measure_layout_sources[k][0] == "full_rmsd"):
                    full_idx = self._measure_layout_sources[k][1]
                    self._full_rmsd_header_spans.append(
                        (row0, col_offset, col_offset + len(cell_text), full_idx))
                    if raw.endswith(" ×"):
                        x_col = col_offset + len(raw)
                        self._full_rmsd_remove_spans.append(
                            (row0, x_col, x_col + 1, full_idx))
                col_offset += len(cell_text)
                if k != len(layout) - 1:
                    segs.append((" ", ""))          # untinted gap between columns
                    col_offset += 1
            return segs

        marker = ("\u2191" if first > 0 else "") + ("\u2193" if first + cap < len(rows) else "")
        title = f" STRUCTURES {len(sset)}"
        # Global ALL (design VIM-28): fills/clears every loaded structure's
        # overlay membership in one click, reusing the exact predicate and
        # fill/clear logic the per-file button already uses (just over the
        # whole set instead of one file's range). No \u00d7 of its own -- that
        # only ever belongs to a single removable file (design VIM-32).
        #
        # The scroll marker is the more load-bearing affordance (design
        # \u00a74.1) and keeps its reserved space at the right edge no matter
        # what; the button only ever gets whatever room is left, and simply
        # doesn't render on a strip too narrow for both -- it must never
        # crowd the marker out.
        global_button = " ALL "
        available = max(0, list_w - len(title) - len(marker))
        show_button = available >= len(global_button)
        gap = available - len(global_button) if show_button else available
        header_segs = [(title, head_fg), (" " * gap, "")]
        self._global_all_span = None
        if show_button:
            global_selected = bool(sset.entries) and all(
                k == sset.active_index or sset[k].marked for k in range(len(sset)))
            global_button_style = (self._sgr_bg(
                self.theme.measure_col_bg_a if global_selected
                else self.theme.list_cap_bg)
                + self._sgr_fg(self.theme.list_label_fg if global_selected
                               else self.theme.list_muted_fg)
                + ("\x1b[1m" if global_selected else ""))
            button_col = len(title) + gap
            header_segs.append((global_button, global_button_style))
            self._global_all_span = (0, button_col, button_col + len(global_button))
        header_segs.append((marker, "\x1b[2m" + head_fg))
        if layout:
            header_segs += measure_segs(0, [h for h, _w, _v, _r in layout], "left")
        put(0, self._list_line(header_segs, total_w))
        # The second chrome row is normally breathing room. Once a file's
        # header scrolls away, it becomes a sticky copy instead, keeping both
        # the source name and its ALL control visible throughout the file.
        sticky = None
        if (0 < first < len(rows)
                and rows[first][0] in ("struct", "collapsed")):
            struct_i = rows[first][1]
            for candidate in range(first - 1, -1, -1):
                if rows[candidate][0] != "group":
                    continue
                group_i = rows[candidate][1]
                if self._list_group_key(group_i) == self._list_group_key(struct_i):
                    sticky = rows[candidate]
                break
        if sticky is None:
            put(1, self._list_line([], total_w))
        else:
            draw_group(1, sticky[1], sticky[2])

        idx_w = max(2, len(str(len(sset))))
        label_w = max(1, list_w - (4 + idx_w))       # pad+swatch+sp+idx+sp
        drawn_rows = 0
        for n_row, (kind, i, text) in enumerate(visible):
            row0 = _LIST_ROWS_ABOVE + n_row
            if kind == "group":
                if not draw_group(row0, i, text):
                    break
                drawn_rows += 1
                continue
            if kind == "collapsed":
                end = self._list_group_end(i)
                prefix = " ▸ "
                visible_label = self._truncate_middle(
                    text, max(1, list_w - len(prefix)))
                segs = [(prefix, muted),
                        (visible_label.ljust(max(1, list_w - len(prefix))),
                         head_fg)]
                if layout:
                    segs += measure_segs(
                        row0, self._list_collapsed_measure_cells(
                            layout, i, end), "right")
                if not put(row0, self._list_line(segs, total_w)):
                    break
                drawn_rows += 1
                self._list_group_summary_spans.append(
                    (row0, len(prefix), len(prefix) + len(visible_label), i, end))
                continue
            entry = sset[i]
            tint = tuple(int(max(0.0, min(1.0, c)) * 255) for c in entry.tint)
            active = i == sset.active_index
            # The active row IS its background (no leader glyph); the cursor
            # row gets a subtler one, so the two stay tellable apart when
            # they differ (design §4.3). Every OTHER row stays background-
            # less on purpose: the panel is transparent, so the terminal's
            # own background shows through and the strip never paints a
            # slab that fights it. Readability on a light terminal comes
            # from the THEME'S FOREGROUNDS (list_dim_fg and friends flip
            # dark on light), not from painting an opaque panel. A row that
            # is IN THE OVERLAY wears its own tint on the label -- with no
            # leader glyph and no key binding for it, that tint is the only
            # way to read the overlay set off the screen at all (membership
            # is opt+click only).
            bg = (self.theme.list_active_bg if active
                  else self.theme.list_cursor_bg if i == self._list_cursor else None)
            dim = "\x1b[2m" if not entry.visible else ""
            # The tint outranks the active row's near-white label:
            # opt+clicking the active row has to change something on screen,
            # and the background is already saying which row is active.
            label_fg = (self._sgr_fg(tint) if entry.marked
                        else self._sgr_fg(self.theme.list_label_fg if active else self.theme.list_dim_fg))
            visible_label = self._truncate_middle(text, label_w)
            segs = [
                (" ", ""),
                ("\u2588" if entry.visible else "\u2591", dim + self._sgr_fg(tint)),
                (" ", ""),
                (f"{i + 1:>{idx_w}}", dim + muted),
                (" ", ""),
                (visible_label.ljust(label_w), dim + label_fg),
            ]
            if layout:
                segs += measure_segs(row0, [vals[i] for _h, _w, vals, _r in layout], "right")
            if not put(row0, self._list_line(segs, total_w, bg=bg)):
                break
            drawn_rows += 1
            # Row click hit-testing extends across the measurement columns
            # too (design 2026-07-30): clicking a structure's values
            # activates it exactly like clicking its label.
            self._list_row_spans.append((row0, 0, total_w))
            self._list_row_struct.append(i)
            frame_tip = self._list_frame_path_tip(i)
            if frame_tip:
                label_col = 4 + idx_w
                self._list_path_hover_spans.append(
                    (row0, label_col, label_col + len(visible_label), frame_tip))

        row0 = _LIST_ROWS_ABOVE + drawn_rows
        rule = "\u2500" * max(0, list_w - 2)
        put(row0, self._list_line([(" ", ""), (rule, self._sgr_fg(self.theme.list_rule_fg))], total_w))
        legend = self._list_legend()
        for k, segs in enumerate(legend, start=1):
            if put(row0 + k, self._list_line(segs, total_w)) and k == len(legend):
                # Exact span of the visibly button-like word itself.
                word_start = len(" Shft+S to ")
                self._select_hint_span = (row0 + k, word_start,
                                          word_start + len("select"))
        # Status lines last: on a short panel they are the first thing to
        # fall off the bottom (put() simply refuses), the legend the last.
        row0 += 1 + len(self._list_legend())
        if sset.overlay:
            drawn = sset.drawn_indices()
            membership = "+".join(str(i + 1) for i in drawn)
            aligned = any(not sset[i].transform.is_identity for i in drawn)
            status = f" overlay {membership}" + (" \u00b7 aligned" if aligned else "")
            put(row0, self._list_line([(status, muted)], total_w))
            row0 += 1
        put(row0, self._list_line([(" camera shared", muted)], total_w))
        # Collapsing a file can retire hundreds of rows in one draw. The
        # molecule image sits to the right of the strip, so nothing else ever
        # repaints the column the strip just gave up -- clear it here, the
        # same way a closing picker clears itself, or the retired rows keep
        # showing 'frame N' underneath the legend.
        bottom = row0 + 1
        for r in range(bottom, min(self._list_drawn_bottom,
                                   max(self._rows - 1, 0))):
            out += b"\x1b[%d;1H\x1b[0m\x1b[2K" % (r + 1)
        self._list_drawn_bottom = bottom
        return bytes(out)

    def _update_geometry(self) -> bool:
        cols, rows, xpx, ypx = kitty.terminal_size_px(self.fd_out)
        # Prefer the terminal's authoritative cell size (queried once at enter);
        # fall back to the window-px/cell-count estimate if it didn't answer.
        cw, ch = self._cell_px or kitty.cell_size_px(self.fd_out)
        prev_cols, prev_rows = self._cols, self._rows
        # _cols/_rows update BEFORE _list_width(): it reads self._cols, and
        # computing it first would size the strip off the terminal's
        # PREVIOUS width for one extra tick on every real resize.
        self._cols, self._rows = cols, rows
        self.widget.set_cell_metrics(cw, ch)
        list_w = self._list_width()
        # Recomputed every tick (cheap): unlike the strip's own width, the
        # measurement table can change with no terminal resize at all --
        # committing/removing a column mid-session -- so its width can't
        # rely solely on the (cols, rows) size check below (design 2026-07-30).
        measure_w = self._measure_width(list_w)
        # _geometry_dirty (set by _refresh_measure_w) is ITS OWN trigger,
        # independent of the measure_w before/after comparison right after it:
        # _refresh_measure_w already syncs self._measure_w immediately after a
        # freeze/removal (so an in-burst hit test sees it right away), which
        # would otherwise make THIS comparison see no difference and silently
        # skip resizing _img_cols/_img_origin_px and the terminal clear a real
        # width change needs -- leaving stale header text on screen and (more
        # seriously) a stale mouse->image origin (design 2026-07-30 rev. 2).
        changed = ((cols, rows) != (prev_cols, prev_rows)
                   or list_w != self._list_w or measure_w != self._measure_w
                   or self._geometry_dirty)
        if changed:
            first_geometry = not self._geometry_established
            self._list_w = list_w
            self._measure_w = measure_w
            total_w = list_w + measure_w
            self._img_rows = max(rows - 1, 1)   # reserve one row for status
            self._img_cols = cols - total_w
            self._img_origin_px = (total_w * cw, 0)
            w = int(self._img_cols * cw)
            h = int(self._img_rows * ch)
            # The widget was built at a 320x240 placeholder (the real terminal
            # size isn't known until the tty is in raw mode); the first time we
            # learn the real size, fit fresh to it rather than preserving the
            # zoom that was fit for the placeholder. Every later resize (the
            # user actually resizing their terminal) preserves the view as usual.
            self.widget.set_pixel_size(max(w, 16), max(h, 16),
                                       refit=first_geometry)
            self._geometry_established = True
            if first_geometry:
                # frame_index is applied before the terminal's real height is
                # known. Re-run visibility now: the placeholder one-row
                # capacity otherwise scrolls frame 1's file header away at
                # startup, and a nonzero --frame cannot be positioned well.
                self._list_ensure_visible(self.structures.active_index)
            self._geometry_dirty = False
        return changed

    # -- rendering --------------------------------------------------------
    def _target_ss(self) -> int:
        """Supersample factor we *want* right now: 1 while interacting (fast),
        higher once settled (crisp). The loop redraws when this changes so the
        crisp frame lands shortly after you stop, without re-drawing meanwhile.

        A held mouse button is interaction, full stop -- on a congested link
        the drag's motion events arrive in bursts separated by long silent
        gaps, and treating those gaps as idle injected the heavy supersampled
        still mid-drag, stalling the link for seconds. The time threshold
        itself is also stretched by the link RTT (see _idle_after).
        """
        idle = (not self._button_held
                and (time.time() - self._last_interact) > self._idle_after
                and not self.autospin)
        return self._max_ss if idle else 1

    @staticmethod
    def _next_render_scale(current: float, elapsed: float,
                           budget: float = _INTERACT_BUDGET,
                           slow_streak: int = _SLOW_FRAMES_BEFORE_STEP_DOWN
                           ) -> float:
        """Dynamic-resolution controller: the next interactive render scale.

        Aims the whole frame pipeline -- render (either backend), encode, and
        delivery over the terminal link -- at *budget* by trading resolution
        for speed while the camera is moving (pixels scale with the square of
        the factor, so the correction is a square root). *budget* is ~30 fps
        locally, but a paced remote session passes max(that, link RTT): on a
        high-latency link 30 fps is physically impossible no matter how few
        pixels are sent, and chasing it would floor the resolution for
        nothing -- the right target there is "transfer costs no more than the
        RTT that's already unavoidable". Clamped to [_SCALE_FLOOR, 1].

        Three rules keep it from parking below 1.0 and staying there, which
        is what VIM-31 was (2024 of 7342 field frames pinned at the floor):

        * Stepping DOWN needs *slow_streak* consecutive over-budget frames.
          A single cold frame -- the first one after an idle pause, paying
          for a cache/GPU wake-up -- measured ~28ms against an 8.3ms budget
          and used to knock the scale straight from 1.0 to 0.53.
        * Stepping UP happens on ANY under-budget frame. The old rule only
          climbed below budget*0.5, but a working controller's steady state
          is just *under* budget -- so its own target zone was the one zone
          it could never leave, and the descent was one-way.
        * Landing *near* 1.0 is worth nothing, because sharpness is binary:
          below 1.0 the terminal resamples every pixel on display, so 0.99 is
          exactly as blurry as 0.7 and merely wastes the pixels it rendered.
          So the climb predicts what full resolution would cost and jumps
          straight to 1.0 when it fits -- and deliberately stops short of 1.0
          when it doesn't, rather than snapping up and being knocked back
          every few frames.
        """
        if elapsed > budget:
            if slow_streak < _SLOW_FRAMES_BEFORE_STEP_DOWN:
                return current       # one slow frame proves nothing
            factor = (budget / elapsed) ** 0.5
            return max(_SCALE_FLOOR, current * max(factor, 0.5))
        # Under budget: climb, but never past what the measurement says we can
        # afford. Frame time is dominated by pixel count, so a frame that cost
        # *elapsed* at *current* would cost budget at exactly this scale:
        affordable = current * (budget / elapsed) ** 0.5 if elapsed > 0 else 1.0
        if affordable >= 1.0:
            return 1.0               # full resolution genuinely fits
        # It doesn't fit. Ease toward the affordable scale and stop there --
        # deliberately NOT snapping to 1.0, which would blow the budget, get
        # stepped back down, and snap again: a ~4-frame pulse between crisp
        # and resampled is worse to look at than steady softness.
        return min(affordable, current * 1.15)

    @staticmethod
    def _next_idle_scale(current: float, elapsed: float) -> float:
        """Settle-frame resolution controller: the next idle render scale.

        Same square-root correction as :meth:`_next_render_scale`, but aimed
        at the much larger settle budget (see _IDLE_BUDGET) and fed the crisp
        still's TRUE end-to-end time: render + encode + *delivery to the
        terminal*, where delivery is measured by the DA1 fence (see
        _arm_settle_fence) rather than by timing write() -- which over SSH
        returns as soon as the bytes enter kernel/SSH buffers and says
        nothing about when they reach the screen. Locally the fence answers
        in about a millisecond and the scale rides at 1.0; over a slow link
        it answers only once the megabytes have actually arrived, and the
        scale settles wherever a still lands within budget. The 0.35 floor
        keeps even a very slow link legible (~1/8 of the pixels).
        """
        if elapsed > _IDLE_BUDGET:
            return max(0.35, current * max((_IDLE_BUDGET / elapsed) ** 0.5, 0.5))
        if elapsed < _IDLE_BUDGET * 0.5:
            return min(1.0, current * 1.15)
        return current

    # -- opt-in timing instrumentation (VIMOL_TIMING) ----------------------
    def _tline(self, tag: str, **fields) -> None:
        """Append one timing line; remember each *_ms field's worst value.

        A no-op unless VIMOL_TIMING is set (and only in a live run, never in
        tests/embeds); the early return is the whole per-frame cost when off.
        """
        if self._timing_path is None or not self._running:
            return
        if self._tlog is None:
            try:
                self._tlog = open(self._timing_path, "a", buffering=1)
                self._tlog.write(f"\n=== vimol session {time.strftime('%H:%M:%S')} ===\n")
            except OSError:
                self._tlog = False
        if self._tlog is False:
            return
        parts = [f"{time.monotonic():10.3f}", tag]
        for k, v in fields.items():
            if k.endswith("_ms"):
                if v > self._tmax.get(k, 0.0):
                    self._tmax[k] = v
                parts.append(f"{k[:-3]}={v:7.1f}ms")
            else:
                parts.append(f"{k}={v}")
        self._tlog.write("  ".join(str(p) for p in parts) + "\n")

    def _tsummary(self) -> str:
        worst = "  ".join(f"{k[:-3]}={v:.1f}ms"
                          for k, v in sorted(self._tmax.items(), key=lambda kv: -kv[1]))
        return f"vimol worst latencies: {worst}" if worst else ""

    # -- frame delivery fence ----------------------------------------------
    def _arm_fence(self, kind: str, base_elapsed: float) -> None:
        """Follow a frame with a DA1 fence to time its real delivery.

        A returned write() only proves the frame entered the kernel pty and
        SSH/TCP buffers -- those absorb megabytes, so over a slow link the
        call returns in milliseconds while the user still watches the frame
        crawl in for seconds. Timing writes therefore measures buffer
        capacity, not the link. The terminal answers a DA1 only after
        consuming everything written before it, so the reply's arrival marks
        true delivery -- and while it's outstanding the run loop renders NO
        new frame (one frame in flight, input keeps coalescing), which is
        what keeps a slow link showing the freshest camera state instead of
        a buffered backlog of stale ones. *base_elapsed* (the local
        render+encode+write cost the fence wait doesn't cover) is added back
        when a controller is fed in _fence_tick. ~10 bytes per frame.
        """
        if self._fence_t0 is None:
            kitty.write_bytes(b"\x1b[c", self.fd_out)
            self._fence_t0 = time.perf_counter()
            self._fence_kind = kind
            self._fence_base = base_elapsed
            self._fence_buf = b""

    def _fence_tick(self, data: bytes) -> None:
        """Watch raw input for the fence reply; feed the matching controller.

        Runs on every read while a fence is outstanding. The reply may split
        across reads, so a short tail is buffered; the decoder downstream
        silently ignores the DA1 reply (an unknown CSI final), so consuming
        it here needs no coordination. A settle fence feeds the idle-scale
        controller with the still's true end-to-end time. An interactive
        fence feeds the render-scale controller with the *controllable* part
        (total minus the link RTT, which no resolution can remove) against
        an RTT-aware budget. An unanswered fence times out without feeding
        anything -- never mistake silence for speed.
        """
        if self._fence_t0 is None:
            return
        if data:
            self._fence_buf = (self._fence_buf + data)[-256:]
            if _DA1_REPLY.search(self._fence_buf):
                wait = time.perf_counter() - self._fence_t0
                total = self._fence_base + wait
                kind = self._fence_kind
                self._tline("ack", kind=kind, wait_ms=wait * 1000,
                            total_ms=total * 1000)
                self._fence_t0 = None
                self._fence_buf = b""
                # Every ack wait bounds the true RTT from above, so the
                # minimum converges on the link's baseline even when the
                # startup measurement happened during a congestion spike
                # (or, after a late probe, never happened at all).
                if self._link_rtt <= 0.0 or wait < self._link_rtt:
                    self._link_rtt = wait
                    self._idle_after = min(1.0, 0.25 + 2.0 * self._link_rtt)
                if kind == "settle":
                    self._idle_scale = self._next_idle_scale(self._idle_scale, total)
                    # A settle ends the interaction burst, so the run length
                    # starts over: the cold frame that opens the NEXT burst
                    # must not be counted alongside one from the last.
                    self._slow_frames = 0
                else:
                    controllable = max(total - self._link_rtt, 0.0)
                    live_budget = max(_INTERACT_BUDGET, self._link_rtt)
                    # Run length of consecutive over-budget frames, so one
                    # cold frame after an idle pause can't cost resolution
                    # (see _next_render_scale).
                    self._slow_frames = (self._slow_frames + 1
                                         if controllable > live_budget else 0)
                    self._interact_scale = self._next_render_scale(
                        self._interact_scale, controllable, live_budget,
                        self._slow_frames)
                return
        if time.perf_counter() - self._fence_t0 > _FENCE_TIMEOUT:
            self._fence_t0 = None
            self._fence_buf = b""

    def _draw(self):
        want_ss = self._target_ss()
        interacting = want_ss == 1
        scene = self.widget.scene
        if scene.supersample != want_ss:
            scene.set_supersample(want_ss)
        # dynamic resolution, BOTH backends: interactive frames at the ~30fps
        # adaptive scale, settle frames at the link-speed-adaptive idle scale.
        # No CPU-only gate: a slow link costs the same megabytes per frame no
        # matter which backend rendered them, so a GPU on a fast link simply
        # measures fast and rides at 1.0.
        scene.set_render_scale(self._interact_scale if interacting
                               else self._idle_scale)
        self._drawn_ss = want_ss

        # Time the WHOLE frame -- render *and* the encode/transmit that follow.
        # Compressing and base64-ing the image costs about as much as the
        # raycast itself, so a controller that watched render() alone would
        # keep resolution high while encode+write silently blew the budget.
        frame_start = time.perf_counter()
        img = self.widget.render()
        t_render = time.perf_counter()
        data = kitty.encode_image(img, image_id=self._img_id, placement_id=self._img_id,
                                  cols=self._img_cols, rows=self._img_rows, move_cursor=False,
                                  z_index=_IMAGE_Z_INDEX,
                                  # fast, slightly-larger compression while moving;
                                  # the resting still gets the smaller default.
                                  # (Both ignored on the shm path, which never
                                  # compresses -- see kitty.encode_image.)
                                  compress_level=1 if interacting else 6,
                                  transmit=self._transmit)
        t_encode = time.perf_counter()
        out = bytearray()
        # The image is placed past the strip AND any measurement columns
        # (col list_w+measure_w+1, row 1) instead of the home position --
        # with neither (_list_w == 0) this is exactly _HOME, so single-
        # structure placement is unchanged.
        out += b"\x1b[1;%dH" % (self._list_w + self._measure_w + 1)
        out += data
        if self._list_w:
            out += self._draw_list()
        out += self._draw_copy_controls()
        out += b"\x1b[%d;1H\x1b[2K" % self._rows
        out += self._status_bar().encode("utf-8", "replace")
        kitty.write_bytes(bytes(out), self.fd_out)
        elapsed = time.perf_counter() - frame_start
        if self._timing_path is not None:      # VIMOL_TIMING: per-frame stages
            t_now = time.perf_counter()
            gap_ms = ((frame_start - self._t_last_draw_end) * 1000
                      if self._t_last_draw_end is not None else 0.0)
            self._t_last_draw_end = t_now
            # VIM-31 (blur at small pane sizes): the transmitted image's own
            # pixel size vs. the cell grid it's placed into -- if these
            # disagree at rest (ss=2, scale=1.0), the terminal is rescaling
            # the image on display, not vimol. cell_px_exact is the RAW
            # xpx/cols ratio (kitty.cell_size_px rounds it before this point),
            # so a real mismatch there -- as opposed to _cell_px, the
            # terminal's own CSI-16t answer when it replied -- points at that
            # rounding rather than the render_scale/idle_scale controllers.
            cw, ch = self._cell_px or kitty.cell_size_px(self.fd_out)
            cols, rows, xpx, ypx = kitty.terminal_size_px(self.fd_out)
            self._tline("frame",
                        kind=("interact" if interacting else "SETTLE"),
                        ss=want_ss, scale=round(scene.render_scale, 3),
                        kb=len(data) // 1024,
                        render_ms=(t_render - frame_start) * 1000,
                        encode_ms=(t_encode - t_render) * 1000,
                        write_ms=(t_now - t_encode) * 1000,
                        gap_ms=gap_ms,
                        img_px=f"{img.shape[1]}x{img.shape[0]}",
                        target_px=f"{int(self._img_cols * cw)}x{int(self._img_rows * ch)}",
                        cell_px=f"{cw:.3f}x{ch:.3f}",
                        cell_px_exact=(f"{xpx / cols:.3f}x{ypx / rows:.3f}"
                                      if xpx and cols and ypx and rows else "?"),
                        queried_cell_px=("yes" if self._cell_px else "no"),
                        idle_scale=round(self._idle_scale, 3),
                        interact_scale=round(self._interact_scale, 3))
        if self._paced:
            # Every frame gets a fence: its ack both feeds the matching
            # resolution controller with the TRUE delivery time and gates
            # the next frame (one in flight -- see _arm_fence).
            self._arm_fence("interact" if interacting else "settle", elapsed)
        elif interacting:
            # No fence support confirmed: fall back to write() timing, which
            # is at least honest under sustained motion (the buffers fill and
            # writes block at real link speed).
            self._interact_scale = self._next_render_scale(
                self._interact_scale, elapsed)
        if self._show_help:
            self._draw_help()
        elif self._mode == "periodic_table":
            self._draw_periodic_table()
        elif self._mode == "geometry_picker":
            self._draw_geometry_picker()
        elif self._mode == "selection_picker":
            self._draw_selection_picker()
        elif self._mode == "file_browser":
            self._draw_file_browser()

    def _draw_help(self):
        out = bytearray()
        bg_r, bg_g, bg_b = self.theme.help_bg
        fg_r, fg_g, fg_b = self.theme.help_fg
        sgr = b"\x1b[48;2;%d;%d;%dm\x1b[38;2;%d;%d;%dm" % (bg_r, bg_g, bg_b, fg_r, fg_g, fg_b)
        for k, line in enumerate(_help_lines(self.editable)):
            out += b"\x1b[%d;3H" % (2 + k)
            out += sgr
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

    def _pt_cell_text(self, cell, cursor: bool) -> str:
        """The 4-char escaped label for one periodic-table cell."""
        if cell is None:
            return "    "
        if cell.symbol is None:
            bg, fg = self.theme.pt_gap_bg, self.theme.pt_dim_fg
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
        border = (f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
                  f"\x1b[38;2;{self.theme.pt_border_fg[0]};{self.theme.pt_border_fg[1]};{self.theme.pt_border_fg[2]}m")
        bg_only = f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
        text_fg = f"\x1b[38;2;{self.theme.pt_text_fg[0]};{self.theme.pt_text_fg[1]};{self.theme.pt_text_fg[2]}m"
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
        border = (f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
                  f"\x1b[38;2;{self.theme.pt_border_fg[0]};{self.theme.pt_border_fg[1]};{self.theme.pt_border_fg[2]}m")
        bg_only = f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
        text_fg = f"\x1b[38;2;{self.theme.pt_text_fg[0]};{self.theme.pt_text_fg[1]};{self.theme.pt_text_fg[2]}m"
        out = bytearray()

        def put(row0: int, col0: int, s: str) -> None:
            out.extend(b"\x1b[%d;%dH" % (row0 + 1, col0 + 1))
            out.extend(s.encode("utf-8", "replace"))

        active = self._active_template()
        title = f" {self.widget.build_element}: geometry ".center(inner_w, "─")
        put(top, left, f"{border}┌{title}┐\x1b[0m")
        for i, opt in enumerate(self._geom_opts):
            is_active = (opt.valence == active.valence and opt.geometry == active.geometry)
            # ASCII, not "●": that bullet's East Asian Width is Ambiguous, and a
            # terminal rendering it 2 columns wide misaligns the right border by
            # 1 column (same class of bug as the status-bar ellipsis below).
            marker = "*" if is_active else " "
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

    # -- atom-selection preset picker ------------------------------------
    def _selection_menu_geometry(self) -> Tuple[int, int, int, int]:
        inner = max(max(map(len, _SELECTION_OPTIONS)) + 4,
                    len(" Selection preset "), len(_SELECTION_HINT))
        width = inner + 2
        height = len(_SELECTION_OPTIONS) + 3
        if self._select_hint_span is not None:
            hint_row, c0, c1 = self._select_hint_span
            anchor = (c0 + c1) // 2
            top = max(0, hint_row - height)
        else:
            anchor = max(self._list_w // 2, width // 2)
            top = max(0, self._rows - 1 - height)
        left = max(0, min(anchor - width // 2, max(self._cols - width, 0)))
        return top, left, width, height

    def _selection_menu_row_at(self, row: int, col: int) -> Optional[int]:
        top, left, width, _height = self._selection_menu_geometry()
        if not (left <= col < left + width):
            return None
        option = row - (top + 1)
        return option if 0 <= option < len(_SELECTION_OPTIONS) else None

    def _draw_selection_picker(self) -> None:
        top, left, width, _height = self._selection_menu_geometry()
        inner_w = width - 2
        border = (f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
                  f"\x1b[38;2;{self.theme.pt_border_fg[0]};{self.theme.pt_border_fg[1]};{self.theme.pt_border_fg[2]}m")
        bg = f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
        fg = f"\x1b[38;2;{self.theme.pt_text_fg[0]};{self.theme.pt_text_fg[1]};{self.theme.pt_text_fg[2]}m"
        out = bytearray()

        def put(row0: int, text: str) -> None:
            out.extend(b"\x1b[%d;%dH" % (row0 + 1, left + 1))
            out.extend(text.encode("utf-8", "replace"))

        title = " Selection preset ".center(inner_w, "─")
        put(top, f"{border}┌{title}┐\x1b[0m")
        for i, option in enumerate(_SELECTION_OPTIONS):
            label = f"   {option}".ljust(inner_w)
            if i == self._selection_menu_idx:
                content = f"{bg}\x1b[1m\x1b[7m{label}\x1b[27m\x1b[22m"
            else:
                content = f"{bg}{fg}{label}"
            put(top + 1 + i,
                f"{border}│\x1b[0m{content}\x1b[0m{border}│\x1b[0m")
        hint_row = top + 1 + len(_SELECTION_OPTIONS)
        put(hint_row,
            f"{border}│\x1b[0m{bg}{fg}{_SELECTION_HINT.ljust(inner_w)}"
            f"\x1b[0m{border}│\x1b[0m")
        put(hint_row + 1, f"{border}└{'─' * inner_w}┘\x1b[0m")
        kitty.write_bytes(bytes(out), self.fd_out)

    def _open_selection_picker(self) -> None:
        if not len(self.structures):
            return
        self._mode = "selection_picker"
        self._list_zone_press = False
        self._selection_menu_idx = 0
        self._msg = ""

    def _close_selection_picker(self) -> None:
        top, _left, _width, height = self._selection_menu_geometry()
        self._mode = "normal"
        self._erase_rows(top, height)

    def _activate_selection_preset(self) -> None:
        label = _SELECTION_OPTIONS[self._selection_menu_idx]
        self._close_selection_picker()
        # A previous hover/click is a transient highlight, not part of the
        # alignment set. Clearing both prevents a nearby Cβ from looking as
        # though the Backbone preset selected it.
        self.widget.hovered = None
        self.widget.selected = None
        self._push_pointer("cell")
        if label == "Manual":
            parent = self._selected_subset_column()
            self.widget.set_alignment_mode(True, preserve=True)
            self._active_subset_id = None
            self._pending_selection_label = None      # explicitly hand-picked
            inherited = f" from {parent.header}" if parent is not None else ""
            self._msg = (f"Manual additive selection{inherited}: click atoms"
                         " / Option-click · whitespace clears · r saves after RMSD")
            return

        self._active_subset_id = None
        self._pending_selection_label = None
        self.widget.set_alignment_mode(True)
        molecule = self.structures.active.molecule
        if label == "Heavy atoms":
            indices = atom_selection.heavy_atoms(molecule)
            missing = "no heavy atoms found on the main frame"
        elif label == "Largest ring system":
            indices = atom_selection.largest_ring_system(molecule)
            missing = "no ring system found on the main frame"
        else:
            indices = atom_selection.peptide_backbone(
                molecule,
                include_beta_carbon=label == "Backbone + Cβ",
            )
            missing = "no peptide-backbone motif found on the main frame"
        if not len(indices):
            self.widget.set_alignment_mode(False)
            self._pop_pointer()
            self._msg = missing
            return
        self.widget.align_sel = indices.tolist()
        self._pending_selection_label = label
        inferred = (" (inferred)"
                    if label.startswith("Backbone")
                    and len(molecule.atom_names) != molecule.n_atoms else "")
        self._msg = (f"{label}{inferred}: {len(indices)} main-frame atoms selected"
                     " · Enter align")

    def _handle_selection_picker_event(self, ev) -> bool:
        if isinstance(ev, _input.KeyEvent):
            if ev.key in ("escape", "\x03", "S"):
                self._close_selection_picker()
                return True
            if ev.key in ("up", "k"):
                self._selection_menu_idx = max(0, self._selection_menu_idx - 1)
                return True
            if ev.key in ("down", "j"):
                self._selection_menu_idx = min(
                    len(_SELECTION_OPTIONS) - 1, self._selection_menu_idx + 1)
                return True
            if ev.key == "enter":
                self._activate_selection_preset()
                return True
            return False
        if isinstance(ev, _input.MouseEvent) and ev.action in ("down", "move"):
            col, row = self._event_cell(ev)
            option = self._selection_menu_row_at(row, col)
            if option is None:
                if ev.action == "down":
                    self._close_selection_picker()
                    return True
                return False
            if ev.action == "move":
                if option != self._selection_menu_idx:
                    self._selection_menu_idx = option
                    return True
                return False
            self._selection_menu_idx = option
            self._activate_selection_preset()
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
        r, g, b = self.theme.edit_prefix_fg
        text = f"\x1b[38;2;{r};{g};{b}m{prefix}\x1b[0m{elem_btn} {geom_btn}"
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
            bg_r, bg_g, bg_b = self.theme.input_bg
            fg_r, fg_g, fg_b = self.theme.input_fg
            return f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m\x1b[38;2;{fg_r};{fg_g};{fg_b}m{body}\x1b[0m"
        if self._mode == "save_confirm":
            name = os.path.basename(self._input_buf.strip())
            body = f" {name} exists — replace? (y/n) "
            bg_r, bg_g, bg_b = self.theme.warn_bg
            fg_r, fg_g, fg_b = self.theme.warn_fg
            return f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m\x1b[38;2;{fg_r};{fg_g};{fg_b}m{body}\x1b[0m"
        if self._mode == "quit_confirm":
            body = " unsaved changes — save before quitting? (y/n/Esc) "
            bg_r, bg_g, bg_b = self.theme.warn_bg
            fg_r, fg_g, fg_b = self.theme.warn_fg
            return f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m\x1b[38;2;{fg_r};{fg_g};{fg_b}m{body}\x1b[0m"
        mol = self.widget.molecule
        hov = self.widget.atom_info(self.widget.hovered)
        # a live measurement readout (2+ picks in measure mode) outranks the
        # hover text; with 0-1 picks measurement() is "" and the normal
        # left-segment behavior applies.
        measure = (editor.measurement(mol, self.widget.measure_sel)
                   if self.widget.measure_mode else "")
        # The molecule "name" is the xyz file's comment line, kept in full
        # (parsers/xyz.py no longer truncates it). It gets no cap of its own:
        # the left field's width below is what bounds it, so a wide terminal
        # actually shows the comment instead of clipping it to a stub while
        # the middle of the bar sits empty.
        align_prompt = ("subset %d picked · Enter align · Esc cancel"
                        % len(self.widget.align_sel) if self.widget.align_mode else "")
        raw_left = (self._subset_hover_tip or self._full_rmsd_hover_tip
                    or self._list_path_hover_tip
                    or align_prompt or measure
                    or self.widget.pick_refusal or hov or (self._msg or
                    f"{mol.name or 'molecule'}  {mol.formula()}  {mol.n_atoms} atoms"))
        rep = self.style.representation
        spin = " ⟳" if self.autospin else ""
        px = " px" if self.decoder.pixel else ""
        backend = "gpu" if self.widget.scene.backend == "gl" else "cpu"
        pr, pg, pb = self.theme.panel_bg
        pfr, pfg, pfb = self.theme.panel_fg
        base = f"\x1b[48;2;{pr};{pg};{pb}m\x1b[38;2;{pfr};{pfg};{pfb}m"
        mod = " [MODIFIED]" if (self.editable and self.widget.dirty) else ""
        hint = "  s save  q quit" if self.editable else "  q quit"
        show_buttons = self.editable and self.widget.append_mode
        show_delete = self.editable and self.widget.delete_mode
        show_measure = self.widget.measure_mode      # read-only-safe: no editable gate
        show_align = self.widget.align_mode

        # Everything from the representation tag onward is a "trailer" built
        # from (escaped, visible_len) pieces and right-anchored via padding
        # below -- independent of `left`'s length. `left` carries hover text
        # that changes on every mouse move; if the clickable element button
        # merely followed it in one concatenated string, moving the mouse
        # (with no click at all) would shift the button out from under the
        # cursor before the next click landed.
        pieces = [(f"[{rep}]", len(rep) + 2), (spin, len(spin))]
        if show_buttons:
            pieces.append((f"  {base}\x1b[1m✎APPEND\x1b[22m", 9))          # "  ✎APPEND"
        elif show_delete:
            pieces.append((f"  {base}\x1b[1m✗DELETE\x1b[22m", 9))          # "  ✗DELETE"
        elif show_measure:
            pieces.append((f"  {base}\x1b[1m∡MEASURE\x1b[22m", 10))        # "  ∡MEASURE"
        elif show_align:
            pieces.append((f"  {base}\x1b[1m◎ALIGN\x1b[22m", 9))           # "  ◎ALIGN"
        pieces.append((mod, len(mod)))
        # Cleanup hint: recomputed from model state every render (no hover
        # dependence), so it appears/disappears exactly like [MODIFIED] does
        # and never disturbs the button-span stability tests.
        cleanup_hint = ""
        if self.editable:
            clash, stretched = editor.cleanup_targets(mol)
            if clash or stretched:
                r, g, b = self.theme.cleanup_hint_fg
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
        # Last-resort shedding, for a terminal too narrow to hold the trailer
        # at all. The trailer is right-anchored and its RIGHTMOST pieces are
        # the ones worth keeping ("? help  q quit"), so drop from the FRONT
        # -- clipping the right end would cost the quit hint exactly when the
        # user is most likely to want it. Pieces are shed whole, so no SGR
        # sequence is ever cut in half.
        while self._cols > 0 and trailer_len > self._cols - 1 and len(pieces) > 1:
            pieces.pop(0)
            if elem_piece_idx is not None:
                elem_piece_idx = elem_piece_idx - 1 if elem_piece_idx > 0 else None
            trailer, trailer_len, offsets = self._build_segment(pieces)

        # The left field's width is DYNAMIC: every column the trailer does
        # not need. It must be derived ONLY from things that change rarely --
        # the terminal width and the trailer's own length -- and NEVER from
        # `raw_left`, which carries hover text that changes on every mouse
        # move: a width that tracked it would drag the right-anchored
        # clickable element/geometry buttons out from under the cursor
        # between a move and the click that follows. Padding to a fixed width
        # (never merely truncating) is what keeps that stable.
        #
        # It is also what keeps the bar inside the terminal. The bar sits on
        # the LAST row, which has no row below to wrap into, so an overflow
        # scrolls the whole terminal up instead -- and since hover text is
        # stable while the mouse sits still, every subsequent redraw (each
        # edit forces one) re-triggers it. That is what actually produced the
        # "status line duplicates on every edit" bug: not a second draw, but
        # the same line scrolling up and leaving its old position visible.
        # (The strip does not shrink this budget: _draw_list never emits on
        # the last row, so the bar owns the full width, strip or no strip.)
        #
        # For the same reason the truncation marker MUST be a single-column-
        # guaranteed ASCII character, not the Unicode ellipsis "…" (U+2026):
        # its East Asian Width is "Ambiguous", and terminals/fonts that
        # render it 2 columns wide push this line 1 column over the edge.
        left_w = (max(0, self._cols - 1 - trailer_len)   # 1 = the leading space
                  if self._cols > 0 else _LEFT_WIDTH)    # size not known yet
        if left_w == 0:
            left = ""
        elif len(raw_left) > left_w:
            left = raw_left[:left_w - 1] + ">"
        else:
            left = raw_left.ljust(left_w)
        left_len = 1 + left_w        # leading space; `base` itself is 0-width
        pad = max(self._cols - left_len - trailer_len, 0)
        if elem_piece_idx is not None:
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

    def _select_hint_hit(self, col: int, row: int) -> bool:
        if self._select_hint_span is None:
            return False
        hint_row, col_start, col_end = self._select_hint_span
        return row == hint_row and col_start <= col < col_end

    def _dispatch(self, events) -> bool:
        """Apply input events; return True if anything visible changed and the
        frame should be redrawn."""
        changed = False
        for ev in events:
            # Track physical button state across ALL modes: _target_ss uses
            # it to keep a drag "interacting" through bursty input gaps.
            if isinstance(ev, _input.MouseEvent):
                if ev.action == "down":
                    self._button_held = True
                elif ev.action == "up":
                    self._button_held = False
            if self._mode == "file_browser":
                if self._handle_file_browser_event(ev):
                    changed = True
                continue
            if self._mode == "selection_picker":
                if isinstance(ev, _input.MouseEvent) and ev.action == "down":
                    col, row = self._event_cell(ev)
                    if self._select_hint_hit(col, row):
                        self._close_selection_picker()
                        changed = True
                        continue
                if self._handle_selection_picker_event(ev):
                    changed = True
                continue
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
            if self.widget.align_mode and isinstance(ev, _input.KeyEvent):
                if ev.key == "enter":
                    if self.widget.align_sel:
                        if self._finish_subset_alignment(tuple(self.widget.align_sel)):
                            self.widget.set_alignment_mode(False)
                            self._pop_pointer()
                    else:
                        self._msg = "pick one or more untinted reference atoms, then press Enter"
                    changed = True
                    continue
                if ev.key in ("escape", "\x03"):
                    self.widget.set_alignment_mode(False)
                    self._pop_pointer()
                    self._msg = "subset alignment cancelled"
                    changed = True
                    continue
            if isinstance(ev, _input.KeyEvent) and ev.key == "tab" and len(self.structures) > 1:
                self._list_focused = not self._list_focused
                if self._list_focused:
                    self._list_cursor = self.structures.active_index
                changed = True
                continue
            if self._list_focused and isinstance(ev, _input.KeyEvent):
                if self._handle_list_key(ev.key):
                    changed = True
                    continue
                # unclaimed by the list keymap -- fall through to the driver
                # keys / widget below, so n/p, q, Ctrl-C, ? etc. all keep
                # working while the strip has focus (design §4.3).
            if isinstance(ev, _input.MouseEvent):
                if ev.action == "down":
                    col, row = self._event_cell(ev)
                    if self._plain_span_hit(self._copy_xyz_span, col, row):
                        self._copy_current_xyz()
                        changed = True
                        continue
                    # The list zone is checked BEFORE the status zone: the
                    # strip's footer rows can land inside the status zone's
                    # bottom-of-screen margin on a short terminal, and a
                    # strip click must never be swallowed by that guard.
                    if self._in_list_zone(col):
                        self._list_zone_press = True
                        if self._select_hint_hit(col, row):
                            self._open_selection_picker()
                            changed = True
                            continue
                        if self._global_all_hit(col, row):
                            self._toggle_list_group_all(
                                0, len(self.structures), label="every structure")
                            changed = True
                            continue
                        # Checked before the ALL hit: the × sits flush
                        # against it (design VIM-32), and must win any
                        # boundary ambiguity between the two.
                        group = self._list_group_remove_hit(col, row)
                        if group is not None:
                            self._remove_file(*group)
                            changed = True
                            continue
                        group = self._list_group_all_hit(col, row)
                        if group is not None:
                            self._toggle_list_group_all(*group)
                            changed = True
                            continue
                        group = self._list_group_span_hit(
                            self._list_group_toggle_spans, col, row)
                        if group is None:
                            group = self._list_group_span_hit(
                                self._list_group_summary_spans, col, row)
                        if group is not None:
                            self._toggle_list_group_collapsed(*group)
                            changed = True
                            continue
                        # A measurement column's × removes it (design
                        # 2026-07-30) and must win over the row click below --
                        # the header row owns no structure, so it would
                        # otherwise just be swallowed as a no-op.
                        removed = self._measure_header_hit(col, row)
                        if removed is not None:
                            del self._measure_columns[removed]
                            self._refresh_measure_w()
                            changed = True
                            continue
                        removed_subset = self._subset_remove_hit(col, row)
                        if removed_subset is not None:
                            self._remove_subset_column(removed_subset)
                            changed = True
                            continue
                        removed_full = self._full_rmsd_remove_hit(col, row)
                        if removed_full is not None:
                            self._remove_full_rmsd_column(removed_full)
                            changed = True
                            continue
                        subset_col = self._subset_header_hit(col, row)
                        if subset_col is not None:
                            self._activate_subset_column(subset_col)
                            changed = True
                            continue
                        i = self._list_index_at_row(row)
                        if i is not None:
                            self._list_click(i, opt=ev.alt)
                            changed = True
                        continue
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
                    if self._list_zone_press:
                        if ev.action == "up":
                            self._list_zone_press = False
                        continue
                    if self._status_zone_press:
                        if ev.action == "up":
                            self._status_zone_press = False
                        continue
                elif ev.action in ("move", "scroll"):
                    col, row = self._event_cell(ev)
                    if self._in_list_zone(col):
                        if ev.action == "move":
                            subset_col = self._subset_header_hit(col, row)
                            tip = (self._subset_tip(self._rmsd_columns[subset_col])
                                   if subset_col is not None else "")
                            if tip != self._subset_hover_tip:
                                self._subset_hover_tip = tip
                                changed = True
                            full_col = (None if subset_col is not None
                                       else self._full_rmsd_header_hit(col, row))
                            full_tip = (self._full_rmsd_tip(self._full_rmsd_columns[full_col])
                                       if full_col is not None else "")
                            if full_tip != self._full_rmsd_hover_tip:
                                self._full_rmsd_hover_tip = full_tip
                                changed = True
                            path_tip = self._list_path_hover_hit(col, row)
                            if path_tip != self._list_path_hover_tip:
                                self._list_path_hover_tip = path_tip
                                changed = True
                        # the wheel scrolls the strip; it must never fall
                        # through to the widget and zoom the 3D view.
                        if ev.action == "scroll" and ev.scroll in ("up", "down"):
                            step = _LIST_WHEEL_STEP if ev.scroll == "down" else -_LIST_WHEEL_STEP
                            if self._list_scroll_by(step):
                                changed = True
                        continue
                    if ev.action == "move":
                        if self._subset_hover_tip:
                            self._subset_hover_tip = ""
                            changed = True
                        if self._full_rmsd_hover_tip:
                            self._full_rmsd_hover_tip = ""
                            changed = True
                        if self._list_path_hover_tip:
                            self._list_path_hover_tip = ""
                            changed = True
                    if self._in_status_zone(row):
                        continue
            if isinstance(ev, _input.KeyEvent) and ev.key in self._driver_keys:
                if self._driver_key(ev.key):
                    changed = True
            else:
                # A genuine reset while picking (design 2026-07-30 rev. 2):
                # snapshot the live pick before the widget can clear/replace
                # it, then compare after. Extending it (2 atoms -> 3 -> 4) is
                # the SAME live column updating in place, not a freeze point
                # -- only a non-extending change (the widget's own 5th-click
                # reset, or an empty-space click) freezes the old pick.
                prev_sel = (tuple(self.widget.measure_sel)
                            if self.widget.measure_mode else None)
                prev_align_sel = tuple(self.widget.align_sel)
                parent_subset_id = self._active_subset_id
                if self.widget.handle_event(ev, origin=self._img_origin_px):
                    self._last_interact = time.time()
                    self._msg = ""          # a fresh interaction clears "saved …"
                    changed = True
                new_align_sel = tuple(self.widget.align_sel)
                if new_align_sel != prev_align_sel:
                    # The named column remains immutable. The first actual
                    # Option/manual edit turns its loaded atoms into an
                    # unsaved derived selection; r/Enter creates a new column.
                    if parent_subset_id is not None:
                        parent = next((c for c in self._rmsd_columns
                                      if c.select_id == parent_subset_id), None)
                        self._active_subset_id = None
                        self._msg = (f"derived from {parent.header if parent else '?'}: "
                                     f"{len(new_align_sel)} atoms · r saves after RMSD")
                    # Whatever produced this pick (a preset, a saved column),
                    # it no longer describes what's now selected -- a hand
                    # edit must not get credited to it (design 2026-08-02).
                    self._pending_selection_label = None
                    self._push_pointer("cell")
                if prev_sel is not None and len(prev_sel) >= 2:
                    new_sel = tuple(self.widget.measure_sel)
                    extends = (len(new_sel) >= len(prev_sel)
                               and new_sel[:len(prev_sel)] == prev_sel)
                    if not extends:
                        self._freeze_measure_sel(prev_sel)
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
        elif key == "A":
            self._open_file_browser()
        elif key == "m":
            # measuring is read-only-safe, so 'm' works without --edit too.
            # Turning it off is "I'm done with this measurement", not
            # "throw it away" -- freeze whatever's live first (design
            # 2026-07-30 rev. 2; set_measure_mode(False) is about to clear it).
            if self.widget.measure_mode:
                self._freeze_measure_sel(tuple(self.widget.measure_sel))
            self.widget.set_measure_mode(not self.widget.measure_mode)
            self._msg = ""
            if self.widget.measure_mode:
                self._push_pointer("cell")               # precision plus-cross
            else:
                self._pop_pointer()
        elif key == "r":
            if not self.structures.overlay:
                # 'r' meant camera reset before it became the align key, and
                # it no longer falls through to the widget. Say so rather
                # than reading as a dead binding.
                self._msg = "align needs overlay mode — z resets the camera"
                return True
            selected = self._selected_subset_column()
            if selected is not None:
                if self.structures.active_index != selected.reference_index:
                    self._activate_structure(selected.reference_index)
                self._align_overlay(ref_select=selected.indices,
                                    rmsd_column=selected)
                self._refresh_measure_w()
                self.widget.align_sel = list(selected.indices)
                return True
            if self.widget.align_mode and self.widget.align_sel:
                indices = tuple(self.widget.align_sel)
                if self._finish_subset_alignment(indices):
                    self.widget.set_alignment_mode(False)
                    self._pop_pointer()
                return True
            if self.widget.align_mode:
                self._msg = "select one or more main-frame atoms first"
                return True
            return self._finish_full_alignment()
        elif key == "R":
            if not self.structures.overlay:
                self._msg = "subset align needs overlay mode"
                return True
            selected = self._selected_subset_column()
            if selected is not None:
                if self.structures.active_index != selected.reference_index:
                    self._activate_structure(selected.reference_index)
                self._align_overlay(ref_select=selected.indices,
                                    rmsd_column=selected)
                self._refresh_measure_w()
                self.widget.align_sel = list(selected.indices)
                return True
            self._list_focused = False
            self.widget.set_alignment_mode(True)
            self._pending_selection_label = None      # a fresh manual pick
            self._push_pointer("cell")
            self._msg = "pick untinted reference atoms, then press Enter"
        elif key == "S":
            self._open_selection_picker()
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
        elif key == "\x14":
            self.theme = theme.LIGHT if self.theme is theme.DARK else theme.DARK
            self.widget.theme = self.theme.name
            self._theme_pinned = True        # beats a probe reply landing later
            self._msg = f"{self.theme.name} theme"
            kitty.write_bytes(_CLEAR, self.fd_out)
            self._last_interact = time.time()
        elif key in ("n", "p", "alt+down", "alt+up") and len(self.frames) > 1:
            self._cycle_frame(1 if key in ("n", "alt+down") else -1)
        else:
            return False
        return True

    def _open_file_browser(self) -> None:
        """Browse for a structure to add, starting somewhere useful.

        Where you were last beats the active structure's directory, which in
        turn beats the process cwd: reopening after a cancel should resume,
        not send you back to the top of the walk you just did.
        """
        start = self._browser_last_dir
        if start is None:
            active = self.structures.active.path if len(self.structures) else None
            start = os.path.dirname(os.path.abspath(active or self.source_path or "."))
        self._mode = "file_browser"
        self._list_zone_press = False
        self._msg = ""
        self._browser_goto(start)

    def _browser_goto(self, path: str) -> None:
        """Show *path*, resetting the cursor to the top of its listing."""
        self._browser_dir = os.path.abspath(path)
        self._browser_last_dir = self._browser_dir
        self._browser_entries = file_browser.list_directory(self._browser_dir)
        self._browser_idx = 0
        self._browser_scroll = 0

    def _file_browser_geometry(self) -> Tuple[int, int, int, int]:
        """(top, left, width, height) of the centred browser overlay."""
        widest = max((len(e.name) for e in self._browser_entries), default=0)
        inner = max(widest + 4, len(_BROWSER_HINT), 32)
        inner = min(inner, max(16, self._cols - 4))
        width = inner + 2
        # top border + rows + hint + bottom border, never taller than the screen
        rows_for_entries = max(1, min(len(self._browser_entries) or 1,
                                      max(1, self._rows - 5)))
        height = rows_for_entries + 3
        top = max(0, (self._rows - height) // 2)
        left = max(0, (self._cols - width) // 2)
        return top, left, width, height

    def _browser_visible_rows(self) -> int:
        return max(1, self._file_browser_geometry()[3] - 3)

    def _browser_scroll_to_cursor(self) -> None:
        visible = self._browser_visible_rows()
        self._browser_scroll = min(self._browser_scroll, self._browser_idx)
        if self._browser_idx >= self._browser_scroll + visible:
            self._browser_scroll = self._browser_idx - visible + 1
        self._browser_scroll = max(0, min(
            self._browser_scroll, max(0, len(self._browser_entries) - visible)))

    def _browser_move(self, step: int) -> None:
        if not self._browser_entries:
            return
        self._browser_idx = max(0, min(len(self._browser_entries) - 1,
                                       self._browser_idx + step))
        self._browser_scroll_to_cursor()

    def _browser_row_at(self, row: int, col: int) -> Optional[int]:
        top, left, width, _height = self._file_browser_geometry()
        if not (left <= col < left + width):
            return None
        offset = row - (top + 1)
        if not 0 <= offset < self._browser_visible_rows():
            return None
        index = self._browser_scroll + offset
        return index if index < len(self._browser_entries) else None

    def _browser_activate(self) -> None:
        """Descend into the highlighted directory, or add the file and close."""
        if not self._browser_entries:
            return
        entry = self._browser_entries[self._browser_idx]
        if entry.is_dir:
            self._browser_goto(entry.path)
            return
        self._close_file_browser()
        self._add_file(entry.path)

    def _close_file_browser(self) -> None:
        top, _left, _width, height = self._file_browser_geometry()
        self._mode = "normal"
        self._erase_rows(top, height)

    def _draw_file_browser(self) -> None:
        top, left, width, _height = self._file_browser_geometry()
        inner_w = width - 2
        border = (f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
                  f"\x1b[38;2;{self.theme.pt_border_fg[0]};{self.theme.pt_border_fg[1]};{self.theme.pt_border_fg[2]}m")
        bg = f"\x1b[48;2;{self.theme.pt_bg[0]};{self.theme.pt_bg[1]};{self.theme.pt_bg[2]}m"
        fg = f"\x1b[38;2;{self.theme.pt_text_fg[0]};{self.theme.pt_text_fg[1]};{self.theme.pt_text_fg[2]}m"
        out = bytearray()

        def put(row0: int, text: str) -> None:
            out.extend(b"\x1b[%d;%dH" % (row0 + 1, left + 1))
            out.extend(text.encode("utf-8", "replace"))

        title = f" {self._truncate_middle(self._browser_dir, inner_w - 2)} "
        put(top, f"{border}┌{title.center(inner_w, '─')}┐\x1b[0m")
        visible = self._browser_visible_rows()
        for offset in range(visible):
            index = self._browser_scroll + offset
            if index < len(self._browser_entries):
                entry = self._browser_entries[index]
                name = entry.name + ("/" if entry.is_dir and entry.name != ".." else "")
                label = f"  {self._truncate_middle(name, inner_w - 3)}".ljust(inner_w)
            else:
                label = " " * inner_w
            if index == self._browser_idx and index < len(self._browser_entries):
                content = f"{bg}\x1b[1m\x1b[7m{label}\x1b[27m\x1b[22m"
            else:
                content = f"{bg}{fg}{label}"
            put(top + 1 + offset,
                f"{border}│\x1b[0m{content}\x1b[0m{border}│\x1b[0m")
        hint_row = top + 1 + visible
        put(hint_row,
            f"{border}│\x1b[0m{bg}{fg}{_BROWSER_HINT.ljust(inner_w)}"
            f"\x1b[0m{border}│\x1b[0m")
        put(hint_row + 1, f"{border}└{'─' * inner_w}┘\x1b[0m")
        kitty.write_bytes(bytes(out), self.fd_out)

    def _handle_file_browser_event(self, ev) -> bool:
        if isinstance(ev, _input.KeyEvent):
            if ev.key in ("escape", "\x03", "A"):
                self._close_file_browser()
                return True
            if ev.key in ("up", "k"):
                self._browser_move(-1)
                return True
            if ev.key in ("down", "j"):
                self._browser_move(1)
                return True
            if ev.key == "enter":
                self._browser_activate()
                return True
            if ev.key == "~":
                self._browser_goto(os.path.expanduser("~"))
                return True
            return False
        if isinstance(ev, _input.MouseEvent):
            col, row = self._event_cell(ev)
            if ev.action == "scroll" and ev.scroll in ("up", "down"):
                self._browser_move(-3 if ev.scroll == "up" else 3)
                return True
            index = self._browser_row_at(row, col)
            if index is None:
                return False
            if ev.action == "move":
                if index != self._browser_idx:
                    self._browser_idx = index
                    return True
                return False
            if ev.action == "down":
                self._browser_idx = index
                self._browser_activate()
                return True
        return False

    def _unique_added_stem(self, path: str) -> str:
        """Return a label stem unique among the live structure entries."""
        base = os.path.basename(path) or path
        occupied = set()
        for entry in self.structures:
            label = entry.label
            occupied.add(label.rsplit("#", 1)[0] if "#" in label else label)
        if base not in occupied:
            return base
        suffix = 2
        while f"{base}~{suffix}" in occupied:
            suffix += 1
        return f"{base}~{suffix}"

    def _add_file(self, path: str) -> bool:
        """Load *path* into the running viewer without replacing its state."""
        display_name = os.path.basename(path) or path
        try:
            molecules = load_all(path)
            if not molecules:
                raise ValueError("no molecules parsed")
            if self._auto_bonds:
                for molecule in molecules:
                    ensure_bonds(molecule, tolerance=self._bond_tolerance)
        except Exception as exc:  # parser and filesystem errors belong in the status bar
            self._msg = f"could not add {display_name}: {exc}"
            return False

        old_count = len(self.structures)
        stem = self._unique_added_stem(path)
        multi = len(molecules) > 1
        for offset, molecule in enumerate(molecules):
            label = f"{stem}#{offset + 1}" if multi else stem
            entry = self.structures.append(molecule, label=label, path=path)
            entry.marked = offset == 0

        # A one-file session becomes the same two-file overlay the CLI would
        # have created.  In an established overlay, preserve the user's
        # membership and simply add the new file's first model.
        if old_count == 1:
            self.structures[0].marked = True
        self.structures.overlay = True
        self.structures.invalidate()

        added_count = len(molecules)
        for column in self._full_rmsd_columns:
            column.values.extend([None] * added_count)
        for column in self._rmsd_columns:
            column.values.extend([None] * added_count)
        self._measure_layout_cache = None
        self._refresh_measure_w()
        self._list_scroll = min(self._list_scroll, self._list_max_scroll())
        self.widget.scene.fit(keep_orientation=True)
        self._last_interact = time.time()
        self._msg = (f"added {display_name}" if added_count == 1
                     else f"added {display_name} ({added_count} frames)")
        return True

    def _full_rmsd_column_stale(self, column: _FullRMSDColumn) -> bool:
        if column.reference_index >= len(self.structures):
            return True
        return (self.structures[column.reference_index].revision
                != column.reference_revision)

    def _finish_full_alignment(self) -> bool:
        """Run an all-atom alignment and persist/update its ∀RMSD column."""
        reference_i = self.structures.active_index
        column = next(
            (candidate for candidate in self._full_rmsd_columns
             if candidate.reference_index == reference_i
             and not self._full_rmsd_column_stale(candidate)),
            None,
        )
        if column is None:
            column = _FullRMSDColumn(
                full_id=self._next_rmsd_id,
                reference_index=reference_i,
                reference_revision=self.structures[reference_i].revision,
                values=[None] * len(self.structures),
            )
            self._next_rmsd_id += 1
            self._full_rmsd_columns.append(column)
        result = self._align_overlay(rmsd_column=column)
        self._refresh_measure_w()
        return result

    def _subset_column_stale(self, column: _SubsetRMSDColumn) -> bool:
        """True once the reference geometry moved out from under the pick."""
        if column.reference_index >= len(self.structures):
            return True
        return (self.structures[column.reference_index].revision
                != column.reference_revision)

    def _selected_subset_column(self) -> Optional[_SubsetRMSDColumn]:
        if self._active_subset_id is None:
            return None
        column = next((column for column in self._rmsd_columns
                       if column.select_id == self._active_subset_id), None)
        # Selection indices are local to their saved reference. Never let a
        # stale armed column silently pull the main frame back after an
        # embedding (or another code path) changed StructureSet directly.
        if (column is None
                or column.reference_index != self.structures.active_index):
            self._active_subset_id = None
            self.widget.align_sel = []
            return None
        if self._subset_column_stale(column):
            # An edit renumbered the reference: those indices now point at
            # different atoms. Recalculating would quietly fit the wrong set,
            # so disarm and make the user re-pick.
            self._active_subset_id = None
            self.widget.align_sel = []
            self._msg = (f"{column.header} is stale — the main frame "
                         "was edited; pick the atoms again")
            return None
        return column

    def _subset_tip(self, column: _SubsetRMSDColumn) -> str:
        """Status-bar spec for a ⊂RMSD header hover (design VIM-30): the
        reference frame this fit is anchored to, plus what it fit on.

        Read like English when the selection came from a known preset
        untouched since ("the backbone", "heavy atoms", ...) -- otherwise
        (Manual, or a preset since hand-edited) the raw atom list is the
        only thing that's actually true, so that's what shows."""
        ref_label = self.structures[column.reference_index].label
        phrase = _SELECTION_PHRASES.get(column.preset_label)
        what = phrase if phrase is not None else ",".join(column.labels)
        return f"{ref_label} · aligning on {what}"

    def _full_rmsd_tip(self, column: _FullRMSDColumn) -> str:
        """Status-bar spec for a ∀RMSD header hover: the reference frame only
        -- a whole-molecule fit has no atom subset to name."""
        ref_label = self.structures[column.reference_index].label
        return f"{ref_label} · aligning on all atoms"

    def _finish_subset_alignment(self, indices: Tuple[int, ...]) -> bool:
        """Persist a newly picked subset, align, and expose its RMSD column.

        False when there is nothing to fit against: picking deliberately works
        before an overlay exists (add one with opt+click, then press r), so a
        premature Enter must keep the selection alive rather than saving a
        column of dashes -- at one loaded structure _measure_layout never
        draws that column, leaving it impossible to remove.
        """
        indices = tuple(sorted(indices))
        reference_i = self.structures.active_index
        if not [i for i in self.structures.drawn_indices() if i != reference_i]:
            self._msg = ("selection kept — overlay a structure "
                         "(opt+click a row), then press r to align")
            return False
        existing = next(
            (column for column in self._rmsd_columns
             if column.reference_index == reference_i and column.indices == indices
             and not self._subset_column_stale(column)),
            None,
        )
        if existing is not None:
            # Selection identity is its owning main frame plus atom set, not
            # the click/preset action that happened to request the RMSD. This
            # is the normal path after adding another structure to an overlay:
            # refill its RMSD#N for every current row instead of cloning it.
            self._align_overlay(ref_select=indices, rmsd_column=existing)
            self._active_subset_id = None
            self._pending_selection_label = None
            self._refresh_measure_w()
            return True
        symbols = self.structures[reference_i].molecule.symbols
        labels = tuple(f"{symbols[i]}{i}" for i in indices)
        column = _SubsetRMSDColumn(
            select_id=self._next_rmsd_id,
            reference_index=reference_i,
            reference_revision=self.structures[reference_i].revision,
            indices=indices,
            labels=labels,
            values=[None] * len(self.structures),
            preset_label=self._pending_selection_label,
        )
        self._next_rmsd_id += 1
        self._pending_selection_label = None
        self._rmsd_columns.append(column)
        self._align_overlay(ref_select=indices, rmsd_column=column)
        # A completed pick is saved but not armed. This keeps plain R available
        # for creating the next RMSD#N; clicking a saved header arms it for
        # a rerun.
        self._active_subset_id = None
        self._refresh_measure_w()
        return True

    def _activate_subset_column(self, column_index: int) -> None:
        """Toggle a saved subset header as the active R-recalculation set."""
        column = self._rmsd_columns[column_index]
        if self._active_subset_id == column.select_id:
            self._active_subset_id = None
            self.widget.align_sel = []
            self._msg = f"{column.header} disabled"
            return
        if self.structures.active_index != column.reference_index:
            self._activate_structure(column.reference_index)
        self.widget.set_alignment_mode(False)
        self.widget.align_sel = list(column.indices)
        self._active_subset_id = column.select_id
        phrase = _SELECTION_PHRASES.get(column.preset_label)
        what = phrase if phrase is not None else ",".join(column.labels)
        self._msg = (f"{column.header} enabled · {what}"
                     " · R recalculates · Option-click derives")

    def _align_overlay(self, ref_select=None,
                       rmsd_column: Optional[Union[
                           _SubsetRMSDColumn, _FullRMSDColumn]] = None) -> bool:
        """Align every tinted/drawn structure onto the active untinted one."""
        sset = self.structures
        reference_i = sset.active_index
        mobiles = [i for i in sset.drawn_indices() if i != reference_i]
        if rmsd_column is None:
            rmsd_values: List[Optional[float]] = [None] * len(sset)
        else:
            # A column accumulates results across overlay swaps. Preserve rows
            # that are not part of this run, extend for newly added entries,
            # and replace only rows that are actually recalculated below.
            rmsd_values = list(rmsd_column.values[:len(sset)])
            rmsd_values.extend([None] * (len(sset) - len(rmsd_values)))
        rmsd_values[reference_i] = 0.0
        if not mobiles:
            self._msg = "overlay has no tinted structure to align"
            if rmsd_column is not None:
                rmsd_column.values = rmsd_values
                self._measure_layout_cache = None
            return True

        fitted = []
        failures = []
        # Once one frame from a trajectory has established atom
        # correspondence, every sibling frame with the same atom-identity
        # layout can reuse it. The remaining fits are then O(N) Kabsch calls
        # instead of repeating the subset search for every trajectory frame.
        correspondence_cache = {}
        reference = sset[reference_i].molecule
        ref_counts = Counter(reference.symbols)
        for i in mobiles:
            rmsd_values[i] = None
            mobile = sset[i].molecule
            mobile_counts = Counter(mobile.symbols)
            try:
                # Bond perception can legitimately fluctuate as a trajectory
                # moves through a distance cutoff; atom order within one
                # source trajectory does not. PDB identity is stronger when
                # available. This applies to both ∀RMSD and ⊂RMSD runs.
                identity = (tuple(mobile.atom_keys)
                            if len(mobile.atom_keys) == mobile.n_atoms
                            else tuple(mobile.symbols))
                cache_key = (self._list_group_key(i), identity)
                correspondence = correspondence_cache.get(cache_key)
                if correspondence is not None:
                    mobile_select, matched_reference = correspondence
                    result = sset.align(
                        i, onto=reference_i, select=mobile_select,
                        ref_select=matched_reference)
                elif ref_select is not None:
                    result = sset.align_to_reference_subset(
                        i, onto=reference_i, ref_select=ref_select)
                elif (mobile.symbols == reference.symbols
                      and mobile.n_atoms > 300):
                    # Protein-scale permutation is inherently quadratic and
                    # deliberately capped; known correspondence remains a
                    # tiny SVD at any size.
                    result = sset.align(i, onto=reference_i)
                elif mobile_counts == ref_counts:
                    # Full RMSD-finder semantics, including permutations
                    # within repeated element blocks. Exact index-ordered
                    # rigid copies short-circuit to one Kabsch in align.py.
                    result = sset.align(i, onto=reference_i, permute=True)
                elif all(count <= ref_counts[el]
                         for el, count in mobile_counts.items()):
                    result = sset.align(
                        i, onto=reference_i, subset=True,
                        permute_max_atoms=None)
                elif all(count <= mobile_counts[el]
                         for el, count in ref_counts.items()):
                    result = sset.align_to_reference_subset(
                        i, onto=reference_i,
                        ref_select=list(range(reference.n_atoms)),
                        permute_max_atoms=None)
                else:
                    raise ValueError("no element-compatible complete/subset match")
                if correspondence is None:
                    if result.select is not None and result.ref_select is not None:
                        paired_mobile = result.select
                        paired_reference = result.ref_select
                    elif result.mapping is not None:
                        paired_mobile = (result.mapping >= 0).nonzero()[0]
                        paired_reference = result.mapping[paired_mobile]
                    else:
                        paired_mobile = paired_reference = None
                    if paired_mobile is not None and len(paired_mobile):
                        correspondence_cache[cache_key] = (
                            paired_mobile.copy(), paired_reference.copy())
                fitted.append((sset[i].label, result.rmsd, result.n_fitted))
                rmsd_values[i] = result.rmsd
            except (ValueError, IndexError) as exc:
                failures.append("%s: %s" % (sset[i].label, exc))
        if fitted:
            values = ", ".join("%s %.4f Å (%d)" % row for row in fitted)
            self._msg = "aligned " + values
            if failures:
                self._msg += "; skipped " + "; ".join(failures)
        else:
            self._msg = "alignment failed: " + "; ".join(failures)
        if rmsd_column is not None:
            rmsd_column.values = rmsd_values
            self._measure_layout_cache = None
        return True

    def _measure_header_text(self, sel: Tuple[int, ...]) -> str:
        """Column header for a freshly committed measurement (design
        2026-07-30): element+index labels (the same raw-index convention as
        the hover/status-bar ``#idx`` text, just without the ``#``), joined
        with the same ∠/φ glyphs the live status-bar readout uses. Recomputed
        live for the in-progress column; frozen at the moment a completed one
        freezes, so it does not retroactively change if a different structure
        later becomes active."""
        symbols = self.structures.active.molecule.symbols
        joined = "-".join(f"{symbols[i]}{i}" for i in sel)
        if len(sel) == 2:
            return joined
        if len(sel) == 3:
            return f"∠{joined}"
        return f"φ{joined}"

    def _freeze_measure_sel(self, sel: Tuple[int, ...]) -> None:
        """Turn a just-finished measurement into a permanent table column
        (design 2026-07-30 rev. 2). Called wherever the live pick list is
        about to be replaced by something that ISN'T a continuation of it: a
        fresh pick after the widget's own 5th-click/empty-space reset,
        measure mode switching off, or the active structure changing (its
        indices are active-local, so a frame switch always clears it) -- see
        the call sites in ``_dispatch``, ``_driver_key``'s ``m``, and
        ``_activate_structure``/``_cycle_frame``. Does NOT touch
        ``widget.measure_sel`` itself -- by the time this runs, whatever
        cleared/replaced it has already happened. Re-freezing the same
        indices is a no-op rather than a duplicate column."""
        if len(sel) < 2:
            return
        if not any(sel == indices for _header, indices in self._measure_columns):
            self._measure_columns.append((self._measure_header_text(sel), sel))
        self._refresh_measure_w()

    def _refresh_measure_w(self) -> None:
        """Recompute the cached ``_measure_w`` right after ``_measure_columns``
        changes (freeze/removal, design 2026-07-30) and flag geometry dirty
        so the next ``_update_geometry`` still notices. ``_in_list_zone``
        reads the cache rather than recomputing on every mouse move -- a
        full recompute calls ``StructureSet.measure`` per column, too costly
        to run on every pointer event at hundreds of structures -- so a
        freeze/removal must refresh it right away, or a hit test later in
        the SAME input burst would still see last tick's width instead of
        what was just drawn. The LIVE column growing between individual
        picks (distance -> angle -> dihedral) does NOT call this -- its
        width can lag by up to one tick, an acceptable trade for not
        recomputing the whole table on every click too."""
        self._measure_w = self._measure_width(self._list_w)
        self._geometry_dirty = True

    def _measure_header_hit(self, col: int, row: int) -> Optional[int]:
        """Column index whose header ``×`` control was clicked at (col,
        row), else None -- checked before the general list click so a
        header click never falls through as a (no-op) row click."""
        for r0, c0, c1, col_idx in self._measure_header_spans:
            if r0 == row and c0 <= col < c1:
                return col_idx
        return None

    def _subset_header_hit(self, col: int, row: int) -> Optional[int]:
        """Saved subset-column index under a header click/hover, if any."""
        for r0, c0, c1, column_index in self._subset_header_spans:
            if r0 == row and c0 <= col < c1:
                return column_index
        return None

    def _subset_remove_hit(self, col: int, row: int) -> Optional[int]:
        """Saved subset-column index whose ``×`` was clicked, if any."""
        for r0, c0, c1, column_index in self._subset_remove_spans:
            if r0 == row and c0 <= col < c1:
                return column_index
        return None

    def _full_rmsd_remove_hit(self, col: int, row: int) -> Optional[int]:
        """Saved full-RMSD column index whose ``×`` was clicked."""
        for r0, c0, c1, column_index in self._full_rmsd_remove_spans:
            if r0 == row and c0 <= col < c1:
                return column_index
        return None

    def _full_rmsd_header_hit(self, col: int, row: int) -> Optional[int]:
        """Saved full-RMSD column index under a header hover, if any."""
        for r0, c0, c1, column_index in self._full_rmsd_header_spans:
            if r0 == row and c0 <= col < c1:
                return column_index
        return None

    def _remove_full_rmsd_column(self, column_index: int) -> None:
        column = self._full_rmsd_columns.pop(column_index)
        self._full_rmsd_hover_tip = ""
        self._msg = f"{column.header} deleted"
        self._refresh_measure_w()

    def _remove_subset_column(self, column_index: int) -> None:
        """Delete one saved subset and disarm it if it was active."""
        column = self._rmsd_columns.pop(column_index)
        if self._active_subset_id == column.select_id:
            self._active_subset_id = None
            self.widget.align_sel = []
        self._subset_hover_tip = ""
        self._msg = f"{column.header} deleted"
        self._refresh_measure_w()

    def _measure_layout(self, list_w: int) -> List[Tuple[str, int, List[str], bool]]:
        """Per-column ``(header cell, width, formatted value cells,
        removable)`` for the measurement table (design 2026-07-30 rev. 2).
        Value cells are aligned to ``self.structures.entries`` order.

        The live pick list (``widget.measure_sel``) renders as one more
        column at the end whenever it holds 2+ atoms -- ``removable`` is
        False for it (no ``×``: there's nothing to remove, it resolves on
        its own once the pick moves on) and True for every frozen column in
        ``_measure_columns``. Suppressed when its indices already match a
        frozen column: re-picking the same pair on a new active structure
        (a common trajectory-browsing move -- freeze on frame A, re-pick
        the same atoms on frame B) would otherwise show two identical-
        looking headers side by side until the live one freezes and
        dedups away. Empty with nothing to show at all (no frozen columns
        and no live pick), or when the structure strip itself is hidden --
        the table only ever exists beside it (single-structure viewers
        have nothing to compare against).

        Columns are dropped from the end once they would leave fewer than
        ``_MEASURE_MIN_VIEWPORT_COLS`` for the 3D image itself -- a silent
        truncation (design 2026-07-30), not an error: without it, enough
        pinned columns drive ``_img_cols`` to zero or negative and corrupt
        every row's layout, not just the table's.
        """
        columns = [
            ("measure", i, header, indices, True, None)
            for i, (header, indices) in enumerate(self._measure_columns)]
        live_sel = tuple(self.widget.measure_sel)
        already_frozen = any(live_sel == indices for _h, indices in self._measure_columns)
        if self.widget.measure_mode and len(live_sel) >= 2 and not already_frozen:
            columns.append(("measure_live", -1, self._measure_header_text(live_sel),
                            live_sel, False, None))
        for i, column in enumerate(self._full_rmsd_columns):
            values = list(column.values[:len(self.structures)])
            if len(values) < len(self.structures):
                values.extend([None] * (len(self.structures) - len(values)))
            # The reference-to-itself RMSD is not a measurement.  Mask it
            # before finding extrema so its synthetic zero never claims the
            # column's minimum marker, then label the cell "Self" rather than
            # leaving the em-dash that means "no result here".
            if 0 <= column.reference_index < len(values):
                values[column.reference_index] = None
            cells = self._format_measure_extrema(values, 2)
            if 0 <= column.reference_index < len(cells):
                cells[column.reference_index] = _SELF_CELL
            # Just "RMSD#N" (design 2026-08-02) -- which fit produced it and
            # what it's aligned on is hover-only detail (_full_rmsd_tip,
            # VIM-30), not part of the permanent label.
            header = column.header + (
                "*" if self._full_rmsd_column_stale(column) else "")
            columns.append(("full_rmsd", i, f"{header} ×", (),
                            False, cells[:len(self.structures)]))
        for i, column in enumerate(self._rmsd_columns):
            values = list(column.values[:len(self.structures)])
            if len(values) < len(self.structures):
                values.extend([None] * (len(self.structures) - len(values)))
            if 0 <= column.reference_index < len(values):
                values[column.reference_index] = None
            cells = self._format_measure_extrema(values, 2)
            if 0 <= column.reference_index < len(cells):
                cells[column.reference_index] = _SELF_CELL
            # A '*' after the id is the design's stale marker (§4.4): the
            # numbers were true once, so they stay readable, but they must
            # not be mistaken for a fit against the geometry on screen now.
            header = column.header + ("*" if self._subset_column_stale(column) else "")
            columns.append(("subset", i, f"{header} ×", column.indices,
                            False, cells[:len(self.structures)]))
        if not columns or len(self.structures) <= 1:
            self._measure_layout_sources = []
            return []
        key = (tuple(self._measure_columns), live_sel, self.widget.measure_mode,
               tuple((c.full_id, c.reference_index, c.reference_revision,
                      tuple(c.values)) for c in self._full_rmsd_columns),
               tuple((c.select_id, c.reference_index, c.indices,
                      tuple(c.values)) for c in self._rmsd_columns),
               list_w, self._cols, self.structures.active_index,
               tuple((id(e.molecule), e.revision) for e in self.structures.entries))
        if self._measure_layout_cache is not None and self._measure_layout_cache[0] == key:
            self._measure_layout_sources = self._measure_layout_cache[2]
            return self._measure_layout_cache[1]
        available = max(0, self._cols - list_w - _MEASURE_MIN_VIEWPORT_COLS)
        out: List[Tuple[str, int, List[str], bool]] = []
        sources: List[Tuple[str, int]] = []
        used = 0
        for source, source_index, header, indices, removable, saved_cells in columns:
            if saved_cells is None:
                kind = len(indices)
                values = [v for _, v in self.structures.measure(indices)]
                cells = self._format_measure_extrema(
                    values, 3 if kind == 2 else 1)
            else:
                cells = saved_cells
            header_cell = f"{header} ×" if removable else header
            width = max(len(header_cell), max((len(c) for c in cells), default=1))
            cost = width + 2 + (1 if out else 0)   # padding, plus the gap before it
            if used + cost > available:
                break
            used += cost
            out.append((header_cell, width, cells, removable))
            sources.append((source, source_index))
        self._measure_layout_sources = sources
        self._measure_layout_cache = (key, out, sources)
        return out

    @staticmethod
    def _format_measure_extrema(values: List[Optional[float]],
                                decimals: int) -> List[str]:
        """Format a measurement column and decorate its current extrema.

        Every finite value tied for the maximum receives ``↑`` and every
        value tied for the minimum receives ``↓``.  A sole value (or a
        completely tied column) is correctly both and therefore receives
        ``↑↓``.  Missing and non-finite values remain an undecorated dash.

        This deliberately derives the markers from *values* during layout,
        instead of persisting extrema on a column: adding or updating any
        structure row immediately reviews the whole column on the next draw.
        """
        finite = [float(value) for value in values
                  if value is not None and math.isfinite(float(value))]
        if not finite:
            return ["—"] * len(values)
        lowest = min(finite)
        highest = max(finite)
        cells: List[str] = []
        for value in values:
            if value is None or not math.isfinite(float(value)):
                cells.append("—")
                continue
            number = float(value)
            marker = ("↑" if number == highest else "")
            marker += ("↓" if number == lowest else "")
            cells.append(f"{number:.{decimals}f}{marker}")
        return cells

    @staticmethod
    def _layout_width(layout: List[Tuple[str, int, List[str], bool]]) -> int:
        """Total columns *layout* (as returned by ``_measure_layout``) needs:
        each column's padded width plus a single untinted gap between
        columns (design 2026-07-30)."""
        if not layout:
            return 0
        return sum(w + 2 for _h, w, _v, _r in layout) + (len(layout) - 1)

    def _measure_width(self, list_w: int) -> int:
        return self._layout_width(self._measure_layout(list_w))

    def _cycle_frame(self, step: int) -> None:
        """Advance the active structure by *step* (wrapping); redraw it.

        Goes through StructureSet.cycle_active + widget.refresh_active, NOT
        widget.set_molecule -- set_molecule is redefined (design §3) as
        "replace the set with a single entry", which would silently discard
        every other loaded structure on the first press of 'n'.
        """
        if self.widget.measure_mode:
            # Freeze BEFORE switching: the header text is computed from the
            # CURRENT (about-to-be-old) active structure's symbols, and
            # refresh_active() is about to clear measure_sel anyway since
            # indices are active-local (design 2026-07-30 rev. 2) -- the
            # whole point of the table is comparing across frames, so
            # silently discarding a mid-measurement pick here would defeat it.
            self._freeze_measure_sel(tuple(self.widget.measure_sel))
        self._disarm_alignment_for_main_change()
        self.structures.cycle_active(step)
        self._list_cursor = self.structures.active_index
        self._list_ensure_visible(self._list_cursor)
        self.widget.refresh_active()
        self._last_interact = time.time()

    def _activate_structure(self, i: int) -> None:
        """Make structure *i* the active one: updates StructureSet,
        resyncs the list cursor to it (design §4.3: 'reset to it by every
        set_active'), and refreshes the widget (camera fit, hover/undo
        reset) without discarding the rest of the StructureSet."""
        if self.widget.measure_mode:
            self._freeze_measure_sel(tuple(self.widget.measure_sel))
        self._disarm_alignment_for_main_change()
        self.structures.set_active(i)
        self._list_cursor = i
        self._list_ensure_visible(i)
        self.widget.refresh_active()

    def _disarm_alignment_for_main_change(self) -> None:
        """Drop atom-local selection state before changing the main frame.

        Saved RMSD columns remain available, but none stays armed: its atom
        indices and reference ownership belong to the frame being left.
        """
        self._active_subset_id = None
        self._subset_hover_tip = ""
        self._full_rmsd_hover_tip = ""
        if self.widget.align_mode:
            self.widget.set_alignment_mode(False)
            self._pop_pointer()
        else:
            self.widget.align_sel = []

    def _hide_toggle(self, i: int) -> None:
        """Toggle Structure.visible on row *i* (the 'h' list key, design
        §4.3): refuses to hide the last visible structure, and does NOT
        auto-advance the active index even when hiding the active row."""
        entry = self.structures[i]
        if entry.visible and sum(1 for e in self.structures if e.visible) <= 1:
            self._msg = "at least one structure must stay visible"
            return
        self.structures.toggle_visible(i)

    def _list_click(self, i: int, opt: bool) -> None:
        """Mouse click on structure-list row *i* (design §4.3, focus-
        independent): a plain click replaces the active structure and clears
        the overlay; opt+click adds it to the overlay alongside the current
        one. Clicking any row also gives the strip keyboard focus.

        opt+click is the ONLY way to change overlay membership -- the
        strip's keymap binds nothing to it.

        opt+click is a *toggle*: opt+clicking a row that is already in the
        overlay drops it, leaving the rest of the set alone. Dropping the
        last one also turns the overlay back off -- overlay with an empty
        membership set means "draw every visible structure"
        (StructureSet.drawn_indices), which is the opposite of what dropping
        your last selection asks for."""
        self._list_focused = True
        if opt:
            entry = self.structures[i]
            entry.marked = not entry.marked
            self.structures.overlay = bool(self.structures.marked)
            self._list_cursor = i
        else:
            self._activate_structure(i)
            self.structures.clear_marks()
            self.structures.overlay = False

    def _handle_list_key(self, key: str) -> bool:
        """Keys claimed while the structure-list strip has focus (design
        §4.3); returns True if anything changed. None of these touch the
        pre-existing global bindings for the same keys (representation
        select, roll, orbit) -- those still fire when the list isn't
        focused, since this is only ever called when it is."""
        sset = self.structures
        n = len(sset)
        if n == 0:
            return False
        if key == "escape":
            self._list_focused = False
            return True
        # j/k only -- NOT the plain arrows. up/down must always reach
        # widget.handle_key and orbit the camera, focused or not; opt+up /
        # opt+down are the arrow-flavoured way to walk the structures and
        # fall through to the global driver keys below.
        if key == "j":
            self._list_cursor = min(n - 1, self._list_cursor + 1)
            self._list_ensure_visible(self._list_cursor)
            return True
        if key == "k":
            self._list_cursor = max(0, self._list_cursor - 1)
            self._list_ensure_visible(self._list_cursor)
            return True
        if key in ("home", "end"):
            self._list_cursor = 0 if key == "home" else n - 1
            self._list_scroll_to(0 if key == "home" else self._list_max_scroll())
            return True
        if key in "123456789":
            i = int(key) - 1
            if i < n:
                self._list_cursor = i
                self._list_ensure_visible(i)
                return True
            return False
        # NOT ]/[ -- those are the global roll bindings (widget.handle_key),
        # and the strip must not shadow them. next/prev is n/p, which the
        # strip deliberately leaves unclaimed so it falls through to the
        # global driver keys (design §4.3).
        if key == "enter":
            self._activate_structure(self._list_cursor)
            sset.clear_marks()
            self._list_focused = False
            return True
        if key == "z":
            sset.solo(self._list_cursor)
            return True
        if key == "h":
            self._hide_toggle(self._list_cursor)
            return True
        # NOT 'o' -- that is _EDIT_DRIVER_KEYS' autospin binding when editable,
        # and the strip must not shadow it (design §4.3).
        if key == "v":
            sset.overlay = not sset.overlay
            return True
        return False

    # -- save prompt ------------------------------------------------------
    def _default_save_path(self) -> str:
        """Where 's' proposes to write: the ACTIVE structure's own file.

        Per-structure rather than per-session, because a multi-file session
        (VIM-1) has no single source path -- and because ``source_path``
        alone would otherwise carry one structure's save-as target over to
        the next one the user tabs to. For a single-file session every entry
        was appended with ``path=source_path``, so this returns exactly what
        it always did.
        """
        if self.structures.entries and self.structures.active.path:
            return self.structures.active.path
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
            if self.structures.entries:
                # a save-as retargets THIS structure only; the next one the
                # user tabs to must still default to its own file.
                self.structures.active.path = path
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
            # First paint before anything that waits on the terminal: geometry
            # comes from a local ioctl, and stamping _last_interact makes the
            # first frame an *interactive* one (supersample 1, conservative
            # render scale, fast zlib) -- milliseconds, not the full-quality
            # still, which the normal settle path delivers ~0.25s later. The
            # capability probe (a link round trip) runs only after the image
            # is already on screen; it gates the mouse, not the pixels.
            self._update_geometry()
            self._last_interact = time.time()
            self._draw()
            self._finish_startup()
            frame_dt = 1.0 / self.target_fps
            dirty = False
            while self._running:
                data = self._read(frame_dt)
                # VIMOL_TIMING: input burstiness -- the gap between reads that
                # carried bytes, and how many events each burst decodes to.
                in_gap = 0.0
                if data and self._timing_path is not None:
                    now = time.monotonic()
                    in_gap = ((now - self._t_last_read) * 1000
                              if self._t_last_read is not None else 0.0)
                    self._t_last_read = now
                if self._late_t0 is not None:
                    # startup probe reply still outstanding: hold raw input
                    # back from the decoder until it lands (see the method).
                    self._late_probe_tick(data)
                    events = []
                else:
                    self._fence_tick(data)      # no-op unless a fence is out
                    events = (self.decoder.feed(data) if data
                              else self.decoder.flush())
                if data and events:
                    self._tline("input", n=len(events), bytes=len(data),
                                ingap_ms=in_gap)
                if self._dispatch(events):
                    dirty = True
                if self._update_geometry():
                    kitty.write_bytes(_CLEAR, self.fd_out)
                    dirty = True
                if self.autospin:
                    self.widget.scene.camera.orbit(1.4, 0)  # ~0.014 rad/frame
                    self._last_interact = time.time()
                    dirty = True
                if self.widget.cleanup_active:
                    # animate the 'c' relaxation: one tick per frame, kept in
                    # fast (non-supersampled) mode while the atoms settle.
                    self.widget.cleanup_tick()
                    self._last_interact = time.time()
                    dirty = True
                if self._fence_t0 is not None:
                    # The previous frame hasn't reached the terminal yet.
                    # Keep reading input and folding it into the model
                    # (`dirty` remembers there's something to show) but send
                    # nothing: rendering now would only lengthen the buffer
                    # queue and show the user an ever-staler backlog. When
                    # the ack lands, the next frame carries the LATEST state
                    # -- motion on a slow link skips, but never lags.
                    continue
                if dirty:
                    self._draw()
                    dirty = False
                elif self._target_ss() != self._drawn_ss and not self._input_pending():
                    # Settle to a crisp, supersampled frame once the view stops
                    # moving -- but ONLY in a genuine lull with nothing queued.
                    # The high-quality downsample is a heavy synchronous step
                    # (~0.2s at full screen); running it while a keypress or
                    # mouse-move is waiting would stall that input behind it.
                    self._draw()
        finally:
            self._exit()
