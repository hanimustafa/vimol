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


def _block_sums(img: np.ndarray, factor: int) -> np.ndarray:
    """Per-block uint16 channel sums over factor x factor tiles.

    Strided-slice accumulation instead of ``reshape(...).mean(axis=(1, 3))``:
    the reshape-mean runs in float64 over two non-contiguous axes and was
    the single most expensive step of a supersampled frame (~170 ms at
    3200x2000); f*f strided adds in uint16 do the same reduction ~6x faster.
    uint16 is safe up to factor 16 (16*16*255 < 65536).
    """
    h, w = img.shape[:2]
    ch = img.shape[2]
    h2, w2 = h // factor, w // factor
    img = img[: h2 * factor, : w2 * factor]
    acc = np.zeros((h2, w2, ch), np.uint16)
    for dy in range(factor):
        for dx in range(factor):
            acc += img[dy::factor, dx::factor]
    return acc


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    ch = img.shape[2]
    sums = _block_sums(img, factor)
    if ch == 4:
        # premultiplied-alpha average so anti-aliased edges don't fringe toward
        # black over a transparent background. RGB is already premultiplied
        # (undrawn pixels are 0,0,0,0), so straight = sum(rgb)/coverage.
        rgb_sum = sums[..., :3].astype(np.float32)
        a_sum = sums[..., 3].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            straight = np.where(a_sum[..., None] > 0, 255.0 * rgb_sum / a_sum[..., None], 0.0)
        alpha = a_sum / (factor * factor)
        out = np.dstack([np.clip(straight, 0, 255), alpha])
        return out.astype(np.uint8)
    # integer floor-average == float mean + truncating uint8 cast, so this is
    # pixel-identical to the old reshape/mean path
    return (sums // (factor * factor)).astype(np.uint8)


class Scene:
    def __init__(self, molecule: Molecule, width: int = 640, height: int = 480,
                 style: Optional[Style] = None, supersample: int = 1,
                 backend: str = "auto"):
        self.molecule = molecule
        self.style = style or Style()
        self.width = width
        self.height = height
        self.supersample = max(1, int(supersample))
        self.render_scale = 1.0        # dynamic-resolution factor, see set_render_scale
        self.camera = Camera(center=molecule.centroid(), extent=molecule.radius_of_gyration_extent())
        self._backend_name, self._renderer = self._make_renderer(
            backend, width * self.supersample, height * self.supersample)
        self.fit()

    @staticmethod
    def _make_renderer(backend: str, w: int, h: int):
        if backend not in ("auto", "cpu", "gl"):
            raise ValueError(f"unknown backend {backend!r} (expected 'auto', 'cpu', or 'gl')")
        if backend == "cpu":
            return "cpu", Renderer(w, h)
        if backend == "gl":
            from .gl_render import GLRenderer  # raises if unavailable -- explicit request, no silent fallback
            return "gl", GLRenderer(w, h)
        # "auto": prefer GL, silently fall back to the CPU raycaster if no
        # GL package/context is available (missing moderngl, no driver, no
        # display -- see gl_render.py's docstring).
        try:
            from .gl_render import GLRenderer
            return "gl", GLRenderer(w, h)
        except Exception:
            return "cpu", Renderer(w, h)

    @property
    def backend(self) -> str:
        """Which renderer actually got picked: ``"cpu"`` or ``"gl"``."""
        return self._backend_name

    # -- sizing / framing -------------------------------------------------
    def set_size(self, width: int, height: int, refit: bool = False) -> None:
        """Resize the viewport. By default this preserves the current
        *apparent framing* -- the fraction of the viewport the molecule fills,
        plus any manual scroll-zoom -- rather than the raw pixels-per-angstrom.
        It does that by rescaling zoom/pan by the min-dimension ratio, exactly
        as :meth:`set_supersample` does for a quality switch.

        This matters for two reasons. (1) For a fit-derived zoom the rescale
        reproduces *precisely* what a fresh :meth:`fit` would compute for the
        new size (both reduce to ``min(W,H)/2 / (extent*margin)``), so if the
        host learns the real terminal size in two steps -- an early, slightly
        wrong report followed by the settled one -- the second, non-refit
        resize self-heals to the correct framing instead of freezing the
        molecule at the first size's zoom. (2) For a user's manual zoom it
        keeps the molecule the same on-screen fraction across a terminal
        resize, which is the intuitive behaviour.

        Pass ``refit=True`` to snap to a fresh fit regardless (used the very
        first time a host establishes real geometry from a placeholder size).
        """
        old_min = max(min(self._renderer.width, self._renderer.height), 1)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self._resize_renderer()
        if refit:
            self.fit()
            return
        new_min = max(min(self._renderer.width, self._renderer.height), 1)
        scale = new_min / old_min
        self.camera.zoom *= scale
        self.camera.pan = self.camera.pan * scale

    def set_supersample(self, factor: int) -> None:
        new_ss = max(1, int(factor))
        if new_ss == self.supersample:
            return
        # Rescale zoom/pan (both in supersampled-buffer pixel units) by the
        # exact buffer-size ratio so the apparent on-screen picture --
        # including any manual scroll-to-zoom -- is preserved across a
        # supersample change. This must NOT go through fit()/Camera.fit():
        # that recomputes zoom purely from the molecule's extent, with no
        # memory of the user's current zoom, so it silently discarded any
        # scroll-zoom every time the interactive quality switch (fast while
        # dragging/scrolling -> crisp ~0.25s after stopping) fired.
        old_min = max(min(self._renderer.width, self._renderer.height), 1)
        self.supersample = new_ss
        self._resize_renderer()
        new_min = max(min(self._renderer.width, self._renderer.height), 1)
        scale = new_min / old_min
        self.camera.zoom *= scale
        self.camera.pan = self.camera.pan * scale

    def set_render_scale(self, factor: float) -> None:
        """Render internally at *factor* x the logical resolution (0.1..1).

        Below 1.0 the scene draws fewer pixels and the terminal (or host)
        stretches the image back up -- slightly soft/grainy, but the frame
        cost drops with the square of the factor. Interactive hosts use this
        as dynamic resolution while the camera is moving, snapping back to
        1.0 (and full supersampling) once idle. Zoom/pan rescale by the exact
        buffer-size ratio, like :meth:`set_supersample`, so the apparent
        framing never changes. The 0.1 hard floor sits below the viewer's own
        operating floor, so it only guards against absurd inputs.
        """
        f = min(1.0, max(0.1, float(factor)))
        if f == self.render_scale:
            return
        old_min = max(min(self._renderer.width, self._renderer.height), 1)
        self.render_scale = f
        self._resize_renderer()
        new_min = max(min(self._renderer.width, self._renderer.height), 1)
        scale = new_min / old_min
        self.camera.zoom *= scale
        self.camera.pan = self.camera.pan * scale

    def _resize_renderer(self) -> None:
        self._renderer.resize(
            max(1, int(round(self.width * self.supersample * self.render_scale))),
            max(1, int(round(self.height * self.supersample * self.render_scale))))

    def fit(self, keep_orientation: bool = False, keep_zoom: bool = False) -> None:
        rot = self.camera.rotation.copy()
        pan = self.camera.pan.copy()
        zoom = self.camera.zoom
        self.camera.center = self.molecule.centroid()
        ext = max(
            self.molecule.radius_of_gyration_extent() + self._max_atom_radius(),
            self.molecule.vector_extent(),
        )
        self.camera.fit(self._renderer.width, self._renderer.height, ext)
        if keep_orientation:
            self.camera.rotation = rot
            self.camera.pan = pan
        if keep_zoom:
            self.camera.zoom = zoom

    def _max_atom_radius(self) -> float:
        if self.molecule.n_atoms == 0:
            return 0.0
        if self.style.representation == "spacefill":
            return float(self.molecule.vdw_radii().max())
        return float(self.molecule.vdw_radii().max()) * self.style.atom_scale

    def set_molecule(self, molecule: Molecule) -> None:
        self.molecule = molecule
        self.camera.center = molecule.centroid()
        self.fit()

    @property
    def render_size(self):
        """(width, height) of the internal render buffer, in supersampled px."""
        return self._renderer.width, self._renderer.height

    # -- rendering --------------------------------------------------------
    def render(self) -> np.ndarray:
        if self._backend_name == "gl":
            from .gl_adapter import molecule_to_gl_inputs
            w, h = self._renderer.width, self._renderer.height
            spheres, cylinders, cones, proj, shading = molecule_to_gl_inputs(
                self.molecule, self.camera, self.style, w, h)
            # the GL renderer supersamples/downsamples on the GPU, so it returns
            # a display-size image directly -- no CPU _downsample pass.
            return self._renderer.render(spheres, cylinders, proj, shading,
                                         downsample=self.supersample, cones=cones)
        img = self._renderer.render(self.molecule, self.camera, self.style)
        return _downsample(img, self.supersample)

    def to_kitty(self, *, image_id: Optional[int] = None, cols: Optional[int] = None,
                 rows: Optional[int] = None, move_cursor: bool = False) -> bytes:
        img = self.render()
        if image_id is None:
            # per-process id so concurrent Scenes (e.g. two panes of one
            # kitty process) don't clobber each other's image storage.
            image_id = kitty.unique_id_base() + 1
        return kitty.encode_image(img, image_id=image_id, cols=cols, rows=rows,
                                  move_cursor=move_cursor)

    def to_png(self, path: str) -> None:
        img = self.render()
        with open(path, "wb") as fh:
            fh.write(kitty.png_bytes(img))
