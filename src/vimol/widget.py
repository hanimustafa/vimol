"""MoleculeWidget — the embeddable interaction core.

This is the reusable piece: it owns a :class:`~vimol.scene.Scene` and turns
input events into camera motion. It does *not* touch the terminal, own an input
loop, or read stdin — so you can drop it into any terminal UI, intercept mouse
events in your own region, and forward them here:

    from vimol.widget import MoleculeWidget
    from vimol.input import InputDecoder

    w = MoleculeWidget(mol, width_px, height_px)
    dec = InputDecoder(pixel=True)
    for ev in dec.feed(bytes_you_read):
        w.handle_event(ev, origin=(region_x_px, region_y_px))
    os.write(1, w.to_kitty(cols=region_cols, rows=region_rows))

The full-screen :class:`vimol.viewer.Viewer` is just a thin driver around
this class.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .molecule import Molecule, VectorField
from .render import Style, _atom_radii
from .scene import Scene
from .structures import StructureSet
from .input import MouseEvent, KeyEvent, Event
from . import editor
from . import elements

REPRESENTATIONS = ["ball_and_stick", "spacefill", "licorice", "wireframe",
                   "ribbon", "glyph"]

# Rubber-band preview for the option/alt-drag manual-bond gesture: a thin,
# distinctive-colored arrow that grows from the anchor atom toward the cursor
# (matches the yellow hover highlight, see `_apply_highlight`).
_BOND_PREVIEW_COLOR = (1.0, 0.85, 0.3)
_BOND_PREVIEW_RADIUS = 0.06

# Cleanup animation ('c'): frames before the relaxation is forced to finish,
# and the per-tick max displacement (angstrom) below which it counts as
# settled -- together roughly half a second of visible motion at 60 fps.
_CLEANUP_FRAME_BUDGET = 30
_CLEANUP_SETTLED = 1e-3


class MoleculeWidget:
    def __init__(self, molecule: Union[Molecule, StructureSet], width: int = 320, height: int = 240,
                 style: Optional[Style] = None, supersample: int = 1,
                 picking: bool = True, backend: str = "auto", editable: bool = False):
        self.style = style or Style()
        self.scene = Scene(molecule, width, height, style=self.style, supersample=supersample,
                           backend=backend)
        self.rotate_speed = 0.007   # radians per pixel of drag
        self.zoom_step = 1.12
        self.picking = picking
        self.hovered: Optional[int] = None      # atom index under the cursor
        self.selected: Optional[int] = None     # last clicked atom
        # editing state -- inert unless the host opts in with editable=True
        self.editable = editable
        self.theme = "dark"                     # "dark" | "light"; Viewer keeps this in sync
        self.rep_note = ""                      # why the last style switch didn't do what it looks like
        self.append_mode = False                # 'a': click to build atoms
        self.delete_mode = False                # 'x': click to remove atoms
        self.measure_mode = False               # 'm': click to build a measurement pick list
        self.measure_sel: list = []             # ordered atom indices picked in measure mode
        self.align_mode = False                 # 'R': pick reference atoms, Enter to align
        self.align_sel: list = []               # active/reference-local atom indices
        # set by _guarded_pick when an append/delete/measure click resolves to
        # a non-active structure (design §3, §12.3); a host status bar can
        # show it. None whenever the last guarded pick was not refused.
        self.pick_refusal: Optional[str] = None
        self.build_element = "C"                # element placed while appending
        self.build_template = None              # chosen geometry/valence; None -> element default
        self.dirty = False                      # True once the model is edited
        self._undo_stack: list = []             # snapshots for 'u'
        self._undo_limit = 200
        self._saved_sig = self._signature()     # model state considered "on disk"
        # option/alt-drag manual-bond gesture: the anchor atom (None when idle)
        # and the widget's own rubber-band preview VectorField, if installed.
        self._bond_anchor: Optional[int] = None
        self._bond_field: Optional[VectorField] = None
        self._bond_drag_distance2 = 0.0
        # in-flight cleanup animation ('c'): the editor's RelaxState (None
        # when idle) and how many frames it may still run before finishing.
        self._cleanup_state = None
        self._cleanup_budget = 0
        # cell metrics used only to convert cell-based events to pixels
        self.cell_w = 9.0
        self.cell_h = 18.0
        self._drag_button: Optional[int] = None
        self._drag_shift = False
        self._last = (0.0, 0.0)
        self._press = (0.0, 0.0)                 # where the current press started
        self._base_colors = self.scene.structures.composite().base_colors

    # -- configuration ----------------------------------------------------
    @property
    def molecule(self) -> Molecule:
        return self.scene.molecule

    def set_molecule(self, molecule: Molecule) -> None:
        """Replace the WHOLE StructureSet with a single new entry.

        For a multi-structure session, prefer :meth:`refresh_active` (driven
        by ``StructureSet.set_active``/``cycle_active``) -- this method
        discards every other loaded structure, by design (see
        ``Scene.set_molecule``'s docstring).
        """
        rot = self.scene.camera.rotation.copy()
        self.scene.set_molecule(molecule)
        self.scene.camera.rotation = rot
        self._reset_editor_state()

    def refresh_active(self) -> None:
        """Re-fit and redraw after the active structure changed via
        ``StructureSet.set_active``/``cycle_active`` (``Viewer.frame_index``,
        the structure-list strip) -- unlike :meth:`set_molecule`, the rest of
        the StructureSet is preserved.
        """
        rot = self.scene.camera.rotation.copy()
        self.scene.fit()
        self.scene.camera.rotation = rot
        self._reset_editor_state()

    def _reset_editor_state(self) -> None:
        """Shared by set_molecule/refresh_active: drop everything that
        indexes into (or caches) the molecule that was just switched away
        from."""
        self._base_colors = self.scene.structures.composite().base_colors
        self.hovered = self.selected = None
        self.measure_sel = []                   # stale indices into the old molecule
        self.align_sel = []
        self.align_mode = False
        self.pick_refusal = None
        self._undo_stack.clear()
        self._saved_sig = self._signature()
        self.dirty = False
        # Drop any in-flight bond gesture -- its anchor index and preview
        # field both belong to the molecule being replaced -- and any
        # in-flight cleanup animation, whose springs index into it too.
        self._bond_anchor = None
        self._bond_field = None
        self._cleanup_state = None

    def set_pixel_size(self, width: int, height: int, refit: bool = False) -> None:
        self.scene.set_size(width, height, refit=refit)

    def set_cell_metrics(self, cell_w: float, cell_h: float) -> None:
        self.cell_w, self.cell_h = cell_w, cell_h

    def set_representation(self, rep: str) -> None:
        if rep not in REPRESENTATIONS:
            return
        self.style.representation = rep
        self.rep_note = ""
        if rep in ("ribbon", "glyph"):
            # Build it now rather than discovering at draw time that there was
            # nothing to draw: the renderer silently falls back to
            # ball-and-stick, which without a word here reads as a dead key.
            from .glyphs import glyph_scene_for
            composite = self.scene.structures.composite()
            self.style.glyph_theme = self.theme
            if glyph_scene_for(composite.molecule,
                               self.scene._effective_style(composite)) is None:
                self.rep_note = (f"{rep} needs a peptide backbone — "
                                 "showing ball-and-stick")
        self.scene.fit(keep_orientation=True)

    def cycle_representation(self, step: int = 1) -> None:
        i = REPRESENTATIONS.index(self.style.representation)
        self.set_representation(REPRESENTATIONS[(i + step) % len(REPRESENTATIONS)])

    def _touch_active(self) -> None:
        """Bump the active structure's revision after any in-place mutation
        of its Molecule (design §5's "one new obligation") -- this busts the
        StructureSet's composite cache, whose base_colors/offsets would
        otherwise go stale in shape the moment an edit changes atom count."""
        self.scene.structures.active.touch()

    # -- direct manipulation ---------------------------------------------
    def orbit(self, dx_px: float, dy_px: float) -> None:
        self.scene.camera.orbit(dx_px, dy_px, speed=self.rotate_speed)

    def pan(self, dx_px: float, dy_px: float) -> None:
        # widget px -> render-buffer px (supersample * render_scale), so the
        # molecule tracks the cursor 1:1 even mid dynamic-resolution drag.
        f = self.scene.render_size[0] / max(self.scene.width, 1)
        # screen y is down; move the molecule with the cursor
        self.scene.camera.pan_by(dx_px * f, -dy_px * f)

    def zoom(self, factor: float) -> None:
        self.scene.camera.zoom_by(factor)

    def roll(self, angle: float) -> None:
        self.scene.camera.roll(angle)

    def reset(self) -> None:
        self.scene.camera.reset()
        self.scene.fit(keep_orientation=True)

    def fit(self) -> None:
        self.scene.fit(keep_orientation=True)

    # -- event handling ---------------------------------------------------
    def handle_event(self, ev: Event, origin: Tuple[float, float] = (0.0, 0.0)) -> bool:
        """Apply an event. Returns True if it changed the view."""
        if isinstance(ev, MouseEvent):
            return self.handle_mouse(ev, origin)
        elif isinstance(ev, KeyEvent):
            return self.handle_key(ev.key)
        return False

    def _local_px(self, ev: MouseEvent, origin: Tuple[float, float]) -> Tuple[float, float]:
        """Event coords -> pixels local to the widget's top-left origin."""
        if ev.pixel:
            x, y = ev.x, ev.y
        else:
            x = ev.x * self.cell_w + self.cell_w * 0.5
            y = ev.y * self.cell_h + self.cell_h * 0.5
        return x - origin[0], y - origin[1]

    def handle_mouse(self, ev: MouseEvent, origin: Tuple[float, float] = (0.0, 0.0)) -> bool:
        """Apply a mouse event. Returns True if it changed the view."""
        x, y = self._local_px(ev, origin)
        if ev.action == "scroll":
            self.zoom(self.zoom_step if ev.scroll == "up" else 1 / self.zoom_step)
            return True
        if ev.action == "down":
            # A fresh press while a bond gesture is still live means its 'up'
            # was lost (focus change, dropped event): tear the stale gesture
            # down so its preview arrow can't be orphaned in vector_fields.
            if self._bond_anchor is not None:
                self._cancel_bond_gesture()
            if ev.button == 0 and ev.alt and self.editable:
                # The option gesture acts on the active structure, so it must
                # see the active structure: a composite pick would let a
                # tinted overlay atom in front intercept the press and lose
                # both the bond anchor and the subset-pick shortcut.
                idx = self._pick_active_only(x, y)
                if idx is not None:
                    # Start a bond gesture -- and deliberately do NOT set
                    # _drag_button, so the drag branch below won't rotate the
                    # camera while the gesture is live.
                    self._bond_anchor = idx
                    self._press = (x, y)
                    self._bond_drag_distance2 = 0.0
                    self._start_bond_preview(idx)
                    return False
                # alt+down over empty space: fall through to a normal press.
            elif ev.button == 0 and ev.alt:
                # Option-click is the always-available shortcut into subset
                # picking. Preserve a loaded named selection so the click
                # edits a live copy rather than starting from nothing.
                self.set_alignment_mode(True, preserve=True)
            self._drag_button = ev.button
            self._drag_shift = ev.shift
            self._last = (x, y)
            self._press = (x, y)
            if self.picking:
                self.selected = self._active_local_pick(x, y)
            return False
        if ev.action == "up":
            if self._bond_anchor is not None:
                return self._end_bond_gesture(x, y)
            was_left = self._drag_button == 0
            self._drag_button = None
            # A left click (no meaningful drag) in append mode edits the model.
            if self.editable and self.append_mode and was_left and not ev.shift:
                dx = x - self._press[0]
                dy = y - self._press[1]
                if dx * dx + dy * dy <= 9.0:      # within ~3px -> a click, not a drag
                    return self._edit_at(x, y)
            elif self.editable and self.delete_mode and was_left and not ev.shift:
                dx = x - self._press[0]
                dy = y - self._press[1]
                if dx * dx + dy * dy <= 9.0:      # within ~3px -> a click, not a drag
                    return self._delete_at(x, y)
            # measure mode has no editable gate -- it is read-only-safe.
            elif self.measure_mode and was_left and not ev.shift:
                dx = x - self._press[0]
                dy = y - self._press[1]
                if dx * dx + dy * dy <= 9.0:      # within ~3px -> a click, not a drag
                    return self._measure_click(x, y)
            elif self.align_mode and was_left and not ev.shift:
                dx = x - self._press[0]
                dy = y - self._press[1]
                if dx * dx + dy * dy <= 9.0:
                    return self._alignment_click(x, y)
            return False
        if ev.action == "drag":
            if self._bond_anchor is not None:
                dx = x - self._press[0]
                dy = y - self._press[1]
                self._bond_drag_distance2 = max(
                    self._bond_drag_distance2, dx * dx + dy * dy)
                self._update_bond_preview(x, y)
                return True
            if self._drag_button is not None:
                dx = x - self._last[0]
                dy = y - self._last[1]
                self._last = (x, y)
                # left = rotate, right or shift+left = pan, middle = pan
                if self._drag_button == 2 or self._drag_button == 1 or self._drag_shift:
                    self.pan(dx, dy)
                else:
                    self.orbit(dx, dy)
                return True
            return False
        if ev.action == "move":
            if self._bond_anchor is not None:
                self._update_bond_preview(x, y)
                return True
            if self.picking:
                prev = self.hovered
                self.hovered = self._active_local_pick(x, y)
                return self.hovered != prev
        return False

    def handle_key(self, key: str) -> bool:
        """Apply a view-control key. Returns True if it changed the view.

        Quit/lifecycle keys are intentionally NOT handled here — the host app
        decides those.
        """
        cam = self.scene.camera
        if key in ("h", "left"):
            cam.orbit(-8, 0); return True
        if key in ("l", "right"):
            cam.orbit(8, 0); return True
        if key in ("k", "up"):
            cam.orbit(0, -8); return True
        if key in ("j", "down"):
            cam.orbit(0, 8); return True
        if key in ("+", "="):
            self.zoom(1.15); return True
        if key in ("-", "_"):
            self.zoom(1 / 1.15); return True
        if key == "[":
            cam.roll(-0.15); return True
        if key == "]":
            cam.roll(0.15); return True
        # 'r' resets here for standalone widget use; the Viewer driver claims
        # 'r' for RMSD alignment instead, so 'z' is the one that actually
        # reaches this branch there -- keep it unclaimed by every other
        # keymap (list strip included) so it always fits/resets the scene.
        if key in ("r", "z"):
            self.reset(); return True
        if key == "f":
            self.fit(); return True
        if key in ("1", "2", "3", "4", "5", "6"):
            self.set_representation(REPRESENTATIONS[int(key) - 1]); return True
        if key == "s" and not self.editable:
            # Without editing, 's' keeps its original meaning (cycle style).
            # When editable, the host driver claims 's' for Save instead.
            self.cycle_representation(); return True
        return False

    # -- editing ----------------------------------------------------------
    def set_append_mode(self, on: bool) -> None:
        # append mode is meaningless (and stays off) unless editing is enabled
        self.append_mode = bool(on) and self.editable
        if self.append_mode:
            self.delete_mode = False            # one active build tool at a time
            self.set_measure_mode(False)
            self.set_alignment_mode(False)

    def set_delete_mode(self, on: bool) -> None:
        # delete mode is meaningless (and stays off) unless editing is enabled
        self.delete_mode = bool(on) and self.editable
        if self.delete_mode:
            self.append_mode = False            # one active build tool at a time
            self.set_measure_mode(False)
            self.set_alignment_mode(False)

    def set_measure_mode(self, on: bool) -> None:
        # unlike append/delete, measuring is non-destructive -- no editable gate.
        self.measure_mode = bool(on)
        if self.measure_mode:
            self.append_mode = False            # one active tool at a time
            self.delete_mode = False
            self.set_alignment_mode(False)
        else:
            self.measure_sel = []               # disarming always clears the pick list

    def set_alignment_mode(self, on: bool, preserve: bool = False) -> None:
        """Arm/disarm reference-atom picking for the interactive subset fit."""
        self.align_mode = bool(on)
        if self.align_mode:
            self.append_mode = False
            self.delete_mode = False
            # Do not call set_measure_mode(False) here: that calls back into
            # this method when measure mode is on.
            self.measure_mode = False
            self.measure_sel = []
            if not preserve:
                self.align_sel = []
        else:
            self.align_sel = []

    # -- undo / dirty tracking -------------------------------------------
    def _signature(self):
        """A cheap hashable snapshot of model identity (for the dirty flag)."""
        mol = self.scene.molecule
        return (tuple(mol.symbols), mol.positions.tobytes(), tuple(mol.manual_bonds))

    def _refresh_dirty(self) -> None:
        self.dirty = self._signature() != self._saved_sig

    def mark_saved(self) -> None:
        """Record the current model as the on-disk state (clears [MODIFIED])."""
        self._saved_sig = self._signature()
        self.dirty = False

    def _snapshot(self):
        mol = self.scene.molecule
        return (list(mol.symbols), mol.positions.copy(), list(mol.bonds),
                list(mol.manual_bonds), set(mol.new_atoms),
                list(mol.atom_names), list(mol.atom_is_hetatm),
                list(mol.atom_keys), list(mol.atom_resnames))

    def _commit_undo(self, snapshot) -> None:
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._undo_limit:
            self._undo_stack.pop(0)

    def _push_undo(self) -> None:
        self._commit_undo(self._snapshot())

    def undo(self) -> bool:
        """Revert the most recent edit. Returns True if anything changed."""
        if not self._undo_stack:
            return False
        # Undo can shrink the molecule under a live bond gesture; cancel it
        # (removing its preview arrow too) so a stale anchor index can't be
        # dereferenced by the next drag event. An in-flight cleanup animation
        # is cancelled the same way (dropped, not finished): its springs were
        # built against the geometry this undo is about to replace.
        if self._bond_anchor is not None:
            self._cancel_bond_gesture()
        self._cleanup_state = None
        (symbols, positions, bonds, manual_bonds, new_atoms,
         atom_names, atom_is_hetatm, atom_keys, atom_resnames) = self._undo_stack.pop()
        mol = self.scene.molecule
        # restore in place so the Scene keeps referencing the same object
        mol.symbols = list(symbols)
        mol.positions = positions.copy()
        mol.bonds = list(bonds)
        mol.manual_bonds = list(manual_bonds)
        mol.new_atoms = set(new_atoms)
        mol.atom_names = list(atom_names)
        mol.atom_is_hetatm = list(atom_is_hetatm)
        mol.atom_keys = list(atom_keys)
        mol.atom_resnames = list(atom_resnames)
        self._touch_active()
        self._base_colors = self.scene.structures.composite().base_colors
        self.hovered = self.selected = None
        self.measure_sel = []                    # stale indices into the reverted geometry
        self._refresh_dirty()
        return True

    def unproject(self, px: float, py: float) -> np.ndarray:
        """Widget-local pixel -> world point on the camera's center plane.

        The inverse of the renderer's orthographic projection (see :meth:`pick`),
        evaluated at view-space depth 0 so a click in empty space lands on the
        plane through the molecule's center that faces the camera.
        """
        cam = self.scene.camera
        Wr, Hr = self.scene.render_size
        # widget-local px -> render-buffer px via the true buffer ratio:
        # this is supersample * render_scale (dynamic resolution), not
        # supersample alone.
        rx = px * (Wr / max(self.scene.width, 1))
        ry = py * (Hr / max(self.scene.height, 1))
        vx = (rx - Wr * 0.5 - cam.pan[0]) / cam.zoom
        vy = (Hr * 0.5 - cam.pan[1] - ry) / cam.zoom
        view = np.array([vx, vy, 0.0])
        return view @ cam.rotation + cam.center

    def _edit_at(self, px: float, py: float) -> bool:
        """Perform an append edit at a widget-local pixel. Returns True (redraw).

        A pick that resolves to a non-active structure is refused (design
        §3, §12.3): no undo snapshot, no mutation -- just pick_refusal set
        for the status bar. Distinct from "empty space", which still births
        a new fragment as always.
        """
        mol = self.scene.molecule
        idx = self._guarded_pick(px, py)
        if self.pick_refusal is not None:
            return True
        self._push_undo()                       # snapshot for 'u' before mutating
        tmpl = self.build_template          # None -> editor uses the element default
        if idx is not None:
            editor.grow_at_atom(mol, idx, element=self.build_element, template=tmpl)
        else:
            editor.birth_molecule(mol, self.unproject(px, py),
                                  element=self.build_element, template=tmpl)
        # atom count changed: refresh color cache and drop stale hover/selection
        self._touch_active()
        self._base_colors = self.scene.structures.composite().base_colors
        self.hovered = self.selected = None
        self._refresh_dirty()
        return True

    def _delete_at(self, px: float, py: float) -> bool:
        """Delete the atom under a widget-local pixel. Returns True if one went
        (or a refusal message needs to be shown).

        Unlike :meth:`_edit_at` -- which always mutates -- a delete click can
        land on empty space, so we pick *first* and bail (no undo snapshot, no
        mutation) when nothing is under the cursor. A pick on a non-active
        structure is refused the same way (design §3, §12.3).
        """
        idx = self._guarded_pick(px, py)
        if self.pick_refusal is not None:
            return True
        if idx is None:
            return False
        mol = self.scene.molecule
        self._push_undo()                       # snapshot for 'u' before mutating
        editor.delete_atom(mol, idx)
        # atom count changed: refresh color cache and drop stale hover/selection
        self._touch_active()
        self._base_colors = self.scene.structures.composite().base_colors
        self.hovered = self.selected = None
        self._refresh_dirty()
        return True

    def _measure_click(self, px: float, py: float) -> bool:
        """Extend/clear the measurement pick list at a widget-local pixel.

        * empty space -> clear the selection (a no-op, no redraw, if it was
          already empty).
        * an atom already in the selection -> no-op (no redraw).
        * a fresh atom, selection under 4 -> append it.
        * a fresh atom, selection already has 4 -> RESET first: the clicked
          atom becomes the sole, first pick of a new selection, not an empty
          one -- see the measure-mode spec's "5th click" rule.
        * a pick on a non-active structure is refused (design §3, §12.3):
          measure_sel holds ACTIVE-LOCAL indices only, so an unguarded
          cross-structure click would silently store an index that means a
          different atom in the active structure.
        """
        idx = self._guarded_pick(px, py)
        if self.pick_refusal is not None:
            return True
        if idx is None:
            if self.measure_sel:
                self.measure_sel = []
                return True
            return False
        if idx in self.measure_sel:
            return False
        if len(self.measure_sel) >= 4:
            self.measure_sel = [idx]
        else:
            self.measure_sel = self.measure_sel + [idx]
        return True

    def _alignment_click(self, px: float, py: float) -> bool:
        """Toggle one reference atom, looking through every tinted overlay."""
        idx = self._pick_active_only(px, py)
        if idx is None:
            if self.align_sel:
                self.align_sel = []
                self.selected = None
                return True
            return False
        if idx in self.align_sel:
            self.align_sel = [i for i in self.align_sel if i != idx]
        else:
            self.align_sel = self.align_sel + [idx]
        return True

    # -- manual-bond gesture (option/alt-drag) -----------------------------
    def _start_bond_preview(self, anchor: int) -> None:
        """Install the widget's own rubber-band preview field at gesture start."""
        mol = self.scene.molecule
        vectors = np.zeros((mol.n_atoms, 3))
        self._bond_field = mol.add_vector_field(
            vectors, color=_BOND_PREVIEW_COLOR, scale=1.0, radius=_BOND_PREVIEW_RADIUS)

    def _update_bond_preview(self, px: float, py: float) -> None:
        """Rewrite the anchor row to point from the anchor atom to the cursor."""
        mol = self.scene.molecule
        anchor = self._bond_anchor
        target = self.unproject(px, py)
        self._bond_field.vectors[anchor] = target - mol.positions[anchor]

    def _cancel_bond_gesture(self) -> None:
        """Abort an in-flight bond gesture: drop the preview, clear the anchor."""
        self._remove_bond_preview()
        self._bond_anchor = None
        self._bond_drag_distance2 = 0.0

    def _remove_bond_preview(self) -> None:
        """Drop exactly the widget's own preview field; never touch user fields.

        Uses identity (``is``), not ``in``/``==`` -- :class:`VectorField` is a
        dataclass holding a numpy array, so its generated ``__eq__`` raises on
        an ambiguous truth value as soon as it's compared against any *other*
        field with more than one vector.
        """
        if self._bond_field is not None:
            fields = self.scene.molecule.vector_fields
            for i, f in enumerate(fields):
                if f is self._bond_field:
                    del fields[i]
                    break
            self._bond_field = None

    def _end_bond_gesture(self, px: float, py: float) -> bool:
        """Finish an option gesture: click selects; drag bonds two atoms."""
        mol = self.scene.molecule
        anchor = self._bond_anchor
        self._remove_bond_preview()
        self._bond_anchor = None
        target = self._pick_active_only(px, py)
        if target == anchor and self._bond_drag_distance2 <= 9.0:
            self.set_alignment_mode(True, preserve=True)
            self._bond_drag_distance2 = 0.0
            return self._alignment_click(px, py)
        self._bond_drag_distance2 = 0.0
        if target is not None and target != anchor:
            snapshot = self._snapshot()          # taken before mutating
            if editor.add_manual_bond(mol, anchor, target):
                self._commit_undo(snapshot)
                self._touch_active()
                self._refresh_dirty()
            # else: already a manual bond between this pair -- no-op, no undo entry
        return True

    # -- cleanup ('c') ------------------------------------------------------
    @property
    def cleanup_active(self) -> bool:
        """True while a cleanup animation is in flight (started, not finished)."""
        return self._cleanup_state is not None

    def start_cleanup(self) -> bool:
        """Begin an animated cleanup relaxation. Returns True if one started.

        Returns False -- and pushes no undo entry -- when an animation is
        already running (a 'c' press mid-animation is ignored) or there is
        nothing to fix. Otherwise the undo snapshot is committed up front,
        so a single 'u' undoes the whole animation no matter how many
        :meth:`cleanup_tick` frames it ran.
        """
        if self._cleanup_state is not None:
            return False
        snapshot = self._snapshot()          # taken before mutating
        state = editor.cleanup_prepare(self.scene.molecule)
        if state is None:
            return False
        self._commit_undo(snapshot)
        self._cleanup_state = state
        self._cleanup_budget = _CLEANUP_FRAME_BUDGET
        return True

    def cleanup_tick(self) -> bool:
        """Advance the cleanup animation one frame. Returns True if it moved.

        Runs a few spring iterations; when the frame budget runs out or the
        motion settles (max displacement below :data:`_CLEANUP_SETTLED`),
        finishes via :func:`editor.cleanup_finish` -- assigning the frozen
        press-time connectivity (never re-perceiving: relaxation motion must
        not mint or drop bonds) -- and returns False from then on.
        """
        if self._cleanup_state is None:
            return False
        disp = editor.cleanup_advance(self.scene.molecule, self._cleanup_state)
        self._cleanup_budget -= 1
        if self._cleanup_budget <= 0 or disp < _CLEANUP_SETTLED:
            state, self._cleanup_state = self._cleanup_state, None
            editor.cleanup_finish(self.scene.molecule, state)
        self._touch_active()
        self._refresh_dirty()
        return True

    def cleanup(self) -> bool:
        """Relax steric clashes / stretched manual bonds in one shot.

        The synchronous convenience over :meth:`start_cleanup` +
        :meth:`cleanup_tick`: runs the whole animation to completion at
        once. Returns True if anything moved (one undo entry), False when
        there was nothing to fix (no undo entry).
        """
        if not self.start_cleanup():
            return False
        while self.cleanup_active:
            self.cleanup_tick()
        return True

    # -- picking ----------------------------------------------------------
    def pick(self, px: float, py: float) -> Optional[int]:
        """Return the COMPOSITE index of the front-most atom under widget-local
        pixel (px, py), or None. Uses the same orthographic projection as the
        renderer.

        This is a composite index, not active-local (design §3,
        "Highlighting") -- so a click over an overlaid, non-active structure
        can be recognized (and refused) instead of silently missing. Convert
        with ``composite.locate()`` at the call site; :meth:`_active_local_pick`
        does exactly that for the common "act only on the active structure"
        case.
        """
        mol = self.scene.structures.composite().molecule
        if mol.n_atoms == 0:
            return None
        cam = self.scene.camera
        Wr, Hr = self.scene.render_size
        # widget-local px -> render-buffer px via the true buffer ratio:
        # this is supersample * render_scale (dynamic resolution), not
        # supersample alone.
        rx = px * (Wr / max(self.scene.width, 1))
        ry = py * (Hr / max(self.scene.height, 1))
        v = cam.view_positions(mol.positions)
        ox_s = Wr * 0.5 + cam.pan[0]
        oy_s = Hr * 0.5 - cam.pan[1]
        sx = ox_s + v[:, 0] * cam.zoom
        sy = oy_s - v[:, 1] * cam.zoom
        sz = v[:, 2]
        radii = _atom_radii(mol, self.style) * cam.zoom
        radii = np.maximum(radii, 1.0)
        d2 = (rx - sx) ** 2 + (ry - sy) ** 2
        inside = d2 <= radii * radii
        if not inside.any():
            return None
        idx = np.where(inside)[0]
        return int(idx[np.argmax(sz[idx])])

    def _active_local_pick(self, px: float, py: float) -> Optional[int]:
        """pick(), converted to an active-local index -- None both when
        nothing is under the cursor AND when the pick resolves to a
        non-active structure (used by hover and the manual-bond gesture,
        where silently doing nothing is the right refusal: design §12.3
        restricts editing to the active structure without making the
        composite's other atoms a dead zone to look at)."""
        idx = self.pick(px, py)
        if idx is None:
            return None
        entry_idx, local = self.scene.structures.composite().locate(idx)
        return local if entry_idx == self.scene.structures.active_index else None

    def _pick_active_only(self, px: float, py: float) -> Optional[int]:
        """Pick the active/untinted structure as if tinted overlays were absent.

        Subset alignment explicitly asks the user to pick the main frame.  A
        normal composite pick would let a closer tinted atom intercept that
        click, making a well-overlaid pair paradoxically harder to select.
        """
        entry = self.scene.structures.active
        mol = entry.molecule
        # Bypassing the composite also bypasses its visibility filter. A
        # hidden active structure has no row in composite.sources, so any
        # index picked here would crash _apply_highlight's globalize().
        if mol.n_atoms == 0 or not entry.visible:
            return None
        positions = entry.transform.apply(mol.positions)
        cam = self.scene.camera
        Wr, Hr = self.scene.render_size
        rx = px * (Wr / max(self.scene.width, 1))
        ry = py * (Hr / max(self.scene.height, 1))
        v = cam.view_positions(positions)
        sx = Wr * 0.5 + cam.pan[0] + v[:, 0] * cam.zoom
        sy = Hr * 0.5 - cam.pan[1] - v[:, 1] * cam.zoom
        radii = np.maximum(_atom_radii(mol, self.style) * cam.zoom, 1.0)
        d2 = (rx - sx) ** 2 + (ry - sy) ** 2
        idx = np.flatnonzero(d2 <= radii * radii)
        return None if not len(idx) else int(idx[np.argmax(v[idx, 2])])

    def _guarded_pick(self, px: float, py: float) -> Optional[int]:
        """pick(), converted to an active-local index for an EDIT ACTION
        (append/delete/measure click). Unlike :meth:`_active_local_pick`,
        a foreign-structure hit is not silent: it sets :attr:`pick_refusal`
        (design §3, §12.3) so the caller can redraw without mutating
        anything, and the status bar can explain why the click did nothing.
        Returns None both for "nothing under the cursor" (proceed as
        empty-space) and "refused" (caller must check pick_refusal to tell
        the two apart).
        """
        self.pick_refusal = None
        idx = self.pick(px, py)
        if idx is None:
            return None
        entry_idx, local = self.scene.structures.composite().locate(idx)
        if entry_idx != self.scene.structures.active_index:
            label = self.scene.structures[entry_idx].label
            self.pick_refusal = f"atom belongs to {label} — Tab to activate"
            return None
        return local

    def atom_info(self, idx: Optional[int]) -> str:
        if idx is None:
            return ""
        mol = self.scene.molecule
        p = mol.positions[idx]
        return f"#{idx} {mol.symbols[idx]} ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"

    # -- rendering --------------------------------------------------------
    def _apply_highlight(self) -> None:
        # widget._base_colors is the composite's base colors (design §3):
        # CPK for the first-drawn (active) entry, tint for the rest.
        composite = self.scene.structures.composite()
        self._base_colors = composite.base_colors
        # The glyph skin paints its own palette rather than element colours, so
        # it needs the light/dark decision the Viewer keeps here.
        self.style.glyph_theme = self.theme
        themed = elements.themed_base_colors(composite.molecule.symbols,
                                             composite.base_colors, self.theme)
        hi = self.hovered if self.hovered is not None else self.selected
        align_sel = getattr(self, "align_sel", ())
        if hi is None and not self.measure_sel and not align_sel:
            # themed is a no-op passthrough for "dark", so this stays
            # byte-identical to pre-theme rendering in the dark case.
            self.style.color_override = themed if self.theme == "light" else None
            return
        # hovered/selected/measure_sel are ACTIVE-LOCAL indices (design §3);
        # map them through the composite's offset before writing into the
        # composite-sized color array.
        active_index = self.scene.structures.active_index
        if not (composite.sources == active_index).any():
            # The active structure isn't drawn at all (hidden -- design §4.3
            # allows this without advancing active_index), so there is no
            # composite slot to map hovered/selected/measure_sel into.
            self.style.color_override = themed if self.theme == "light" else None
            return
        cols = themed.copy()
        yellow = np.array([1.0, 0.95, 0.3])
        # every picked atom in the live measurement selection gets the same
        # yellow tint as a hover -- hover (below) is applied on top, so it
        # still shows through even for an atom that is also selected.
        if self.measure_sel:
            g_sel = composite.globalize(active_index, np.asarray(self.measure_sel, dtype=np.int64))
            for gidx in g_sel:
                cols[gidx] = np.clip(cols[gidx] * 0.4 + yellow * 0.9, 0, 1)
        if align_sel:
            cyan = np.array([0.15, 1.0, 0.95])
            g_sel = composite.globalize(active_index, np.asarray(align_sel, dtype=np.int64))
            for gidx in g_sel:
                cols[gidx] = np.clip(cols[gidx] * 0.25 + cyan * 0.95, 0, 1)
        if hi is not None:
            # brighten + tint the highlighted atom: red in delete mode (a preview of
            # "this disappears if you click here"), yellow otherwise.
            tint = np.array([1.0, 0.2, 0.2]) if self.delete_mode else yellow
            ghi = int(composite.globalize(active_index, np.array([hi]))[0])
            cols[ghi] = np.clip(cols[ghi] * 0.4 + tint * 0.9, 0, 1)
        self.style.color_override = cols

    def render(self) -> np.ndarray:
        self._apply_highlight()
        return self.scene.render()

    def to_kitty(self, *, cols=None, rows=None, image_id=None, move_cursor=False) -> bytes:
        self._apply_highlight()
        return self.scene.to_kitty(cols=cols, rows=rows, image_id=image_id,
                                   move_cursor=move_cursor)
