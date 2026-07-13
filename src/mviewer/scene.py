"""A Scene binds a molecule, a camera, a style and a renderer together.

It is the main object embedding apps interact with: set the pixel size, call
:meth:`render` to get an RGB numpy array, or :meth:`to_kitty` for protocol
bytes ready to write to a terminal.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .camera import Camera
from .molecule import Molecule
from .render import Renderer, Style
from . import kitty


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // factor, w // factor
    img = img[: h2 * factor, : w2 * factor]
    img = img.reshape(h2, factor, w2, factor, img.shape[2]).astype(np.uint16)
    return (img.mean(axis=(1, 3))).astype(np.uint8)


class Scene:
    def __init__(self, molecule: Molecule, width: int = 640, height: int = 480,
                 style: Optional[Style] = None, supersample: int = 1):
        self.molecule = molecule
        self.style = style or Style()
        self.width = width
        self.height = height
        self.supersample = max(1, int(supersample))
        self.camera = Camera(center=molecule.centroid(), extent=molecule.radius_of_gyration_extent())
        self._renderer = Renderer(width * self.supersample, height * self.supersample)
        self.fit()

    # -- sizing / framing -------------------------------------------------
    def set_size(self, width: int, height: int) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self._renderer.resize(self.width * self.supersample, self.height * self.supersample)
        self.fit()

    def set_supersample(self, factor: int) -> None:
        self.supersample = max(1, int(factor))
        self._renderer.resize(self.width * self.supersample, self.height * self.supersample)
        # preserve framing/orientation, just rescale zoom relative to SS
        self.fit(keep_orientation=True)

    def fit(self, keep_orientation: bool = False) -> None:
        rot = self.camera.rotation.copy()
        pan = self.camera.pan.copy()
        self.camera.center = self.molecule.centroid()
        ext = self.molecule.radius_of_gyration_extent() + self._max_atom_radius()
        self.camera.fit(self._renderer.width, self._renderer.height, ext)
        if keep_orientation:
            self.camera.rotation = rot
            self.camera.pan = pan

    def _max_atom_radius(self) -> float:
        if self.molecule.n_atoms == 0:
            return 0.0
        if self.style.representation == "spacefill":
            return float(self.molecule.vdw_radii().max())
        return float(self.molecule.covalent_radii().max()) * self.style.atom_scale

    def set_molecule(self, molecule: Molecule) -> None:
        self.molecule = molecule
        self.camera.center = molecule.centroid()
        self.fit()

    # -- rendering --------------------------------------------------------
    def render(self) -> np.ndarray:
        img = self._renderer.render(self.molecule, self.camera, self.style)
        return _downsample(img, self.supersample)

    def to_kitty(self, *, image_id: int = 1, cols: Optional[int] = None,
                 rows: Optional[int] = None, move_cursor: bool = False) -> bytes:
        img = self.render()
        return kitty.encode_image(img, image_id=image_id, cols=cols, rows=rows,
                                  move_cursor=move_cursor)

    def to_png(self, path: str) -> None:
        img = self.render()
        with open(path, "wb") as fh:
            fh.write(kitty.png_bytes(img))
