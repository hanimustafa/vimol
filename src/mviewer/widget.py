"""MoleculeWidget — the embeddable interaction core.

This is the reusable piece: it owns a :class:`~mviewer.scene.Scene` and turns
input events into camera motion. It does *not* touch the terminal, own an input
loop, or read stdin — so you can drop it into any terminal UI, intercept mouse
events in your own region, and forward them here:

    from mviewer.widget import MoleculeWidget
    from mviewer.input import InputDecoder

    w = MoleculeWidget(mol, width_px, height_px)
    dec = InputDecoder(pixel=True)
    for ev in dec.feed(bytes_you_read):
        w.handle_event(ev, origin=(region_x_px, region_y_px))
    os.write(1, w.to_kitty(cols=region_cols, rows=region_rows))

The full-screen :class:`mviewer.viewer.Viewer` is just a thin driver around
this class.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .molecule import Molecule
from .render import Style, _atom_radii
from .scene import Scene
from .input import MouseEvent, KeyEvent, Event

REPRESENTATIONS = ["ball_and_stick", "spacefill", "licorice", "wireframe"]


class MoleculeWidget:
    def __init__(self, molecule: Molecule, width: int = 320, height: int = 240,
                 style: Optional[Style] = None, supersample: int = 1,
                 picking: bool = True, backend: str = "auto"):
        self.style = style or Style()
        self.scene = Scene(molecule, width, height, style=self.style, supersample=supersample,
                           backend=backend)
        self.rotate_speed = 0.007   # radians per pixel of drag
        self.zoom_step = 1.12
        self.picking = picking
        self.hovered: Optional[int] = None      # atom index under the cursor
        self.selected: Optional[int] = None     # last clicked atom
        # cell metrics used only to convert cell-based events to pixels
        self.cell_w = 9.0
        self.cell_h = 18.0
        self._drag_button: Optional[int] = None
        self._drag_shift = False
        self._last = (0.0, 0.0)
        self._base_colors = molecule.element_colors()

    # -- configuration ----------------------------------------------------
    @property
    def molecule(self) -> Molecule:
        return self.scene.molecule

    def set_molecule(self, molecule: Molecule) -> None:
        rot = self.scene.camera.rotation.copy()
        self.scene.set_molecule(molecule)
        self.scene.camera.rotation = rot
        self._base_colors = molecule.element_colors()
        self.hovered = self.selected = None

    def set_pixel_size(self, width: int, height: int, refit: bool = False) -> None:
        self.scene.set_size(width, height, refit=refit)

    def set_cell_metrics(self, cell_w: float, cell_h: float) -> None:
        self.cell_w, self.cell_h = cell_w, cell_h

    def set_representation(self, rep: str) -> None:
        if rep in REPRESENTATIONS:
            self.style.representation = rep
            self.scene.fit(keep_orientation=True)

    def cycle_representation(self, step: int = 1) -> None:
        i = REPRESENTATIONS.index(self.style.representation)
        self.set_representation(REPRESENTATIONS[(i + step) % len(REPRESENTATIONS)])

    # -- direct manipulation ---------------------------------------------
    def orbit(self, dx_px: float, dy_px: float) -> None:
        self.scene.camera.orbit(dx_px, dy_px, speed=self.rotate_speed)

    def pan(self, dx_px: float, dy_px: float) -> None:
        ss = self.scene.supersample
        # screen y is down; move the molecule with the cursor
        self.scene.camera.pan_by(dx_px * ss, -dy_px * ss)

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
            self._drag_button = ev.button
            self._drag_shift = ev.shift
            self._last = (x, y)
            if self.picking:
                self.selected = self.pick(x, y)
            return False
        if ev.action == "up":
            self._drag_button = None
            return False
        if ev.action == "drag" and self._drag_button is not None:
            dx = x - self._last[0]
            dy = y - self._last[1]
            self._last = (x, y)
            # left = rotate, right or shift+left = pan, middle = pan
            if self._drag_button == 2 or self._drag_button == 1 or self._drag_shift:
                self.pan(dx, dy)
            else:
                self.orbit(dx, dy)
            return True
        if ev.action == "move":
            if self.picking:
                prev = self.hovered
                self.hovered = self.pick(x, y)
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
        if key in ("r", "z"):
            self.reset(); return True
        if key == "f":
            self.fit(); return True
        if key in ("1", "2", "3", "4"):
            self.set_representation(REPRESENTATIONS[int(key) - 1]); return True
        if key == "s":
            self.cycle_representation(); return True
        return False

    # -- picking ----------------------------------------------------------
    def pick(self, px: float, py: float) -> Optional[int]:
        """Return the index of the front-most atom under widget-local pixel
        (px, py), or None. Uses the same orthographic projection as the renderer.
        """
        mol = self.scene.molecule
        if mol.n_atoms == 0:
            return None
        cam = self.scene.camera
        ss = self.scene.supersample
        rx, ry = px * ss, py * ss
        Wr, Hr = self.scene.render_size
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

    def atom_info(self, idx: Optional[int]) -> str:
        if idx is None:
            return ""
        mol = self.scene.molecule
        p = mol.positions[idx]
        return f"#{idx} {mol.symbols[idx]} ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"

    # -- rendering --------------------------------------------------------
    def _apply_highlight(self) -> None:
        hi = self.hovered if self.hovered is not None else self.selected
        if hi is None:
            self.style.color_override = None
            return
        cols = self._base_colors.copy()
        # brighten + tint the highlighted atom toward yellow
        cols[hi] = np.clip(cols[hi] * 0.4 + np.array([1.0, 0.95, 0.3]) * 0.9, 0, 1)
        self.style.color_override = cols

    def render(self) -> np.ndarray:
        self._apply_highlight()
        return self.scene.render()

    def to_kitty(self, *, cols=None, rows=None, image_id=None, move_cursor=False) -> bytes:
        self._apply_highlight()
        return self.scene.to_kitty(cols=cols, rows=rows, image_id=image_id,
                                   move_cursor=move_cursor)
