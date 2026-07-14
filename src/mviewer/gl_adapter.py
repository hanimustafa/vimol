"""Bridges mviewer's molecule-aware types to the generic GL renderer.

This is the only module that knows about both ``Molecule``/``Camera``/
``Style`` (mviewer) and ``SphereBatch``/``CylinderBatch``/``ConeBatch``/
``ShadingParams`` (``gl_render.py``, which otherwise has zero mviewer imports).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .arrows import build_arrow_geometry
from .camera import Camera
from .molecule import Molecule
from .render import Style, _atom_radii
from .gl_render import SphereBatch, CylinderBatch, ConeBatch, ShadingParams


def _build_projection(zoom: float, pan: np.ndarray, width: int, height: int, extent: float) -> np.ndarray:
    """Orthographic projection matching ``Camera``'s pixel-space screen
    mapping (``sx = W/2 + pan.x + vx*zoom``, ``sy = H/2 - pan.y - vy*zoom``,
    Y-down pixel space), reworked into a clip-space (Y-up) 4x4 matrix.

    Returned in the usual mathematical convention (row-major indexing,
    ``[nx,ny,nz,1] = M @ [vx,vy,vz,1]``) — ``GLRenderer.render`` transposes
    internally for GLSL's column-major upload, and also reads ``M[2,2]``/
    ``M[2,3]`` directly to reproject each fragment's own analytic depth.

    The Z mapping ``nz = -vz / (4*extent)`` needs no separate near/far
    values: view-space ``+z`` is toward the viewer (``Camera``'s
    convention), so larger ``vz`` (nearer) maps to smaller/negative ``nz``
    (GL's near plane, since the default ``GL_LESS`` test + clear depth
    ``1.0`` means smaller wins). ``4*extent`` is a generous symmetric bound
    around the camera's look-at center — no clipping in practice.
    """
    ext = max(float(extent), 1e-6)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 * zoom / width
    m[0, 3] = 2.0 * pan[0] / width
    m[1, 1] = 2.0 * zoom / height
    m[1, 3] = 2.0 * pan[1] / height
    m[2, 2] = -1.0 / (4.0 * ext)
    m[3, 3] = 1.0
    return m


def molecule_to_gl_inputs(molecule: Molecule, camera: Camera, style: Style,
                          width: int, height: int
                          ) -> Tuple[SphereBatch, CylinderBatch, ConeBatch, np.ndarray, ShadingParams]:
    """Convert a molecule + camera + style into generic GL renderer inputs."""
    if molecule.n_atoms == 0:
        vpos = np.zeros((0, 3), np.float32)
        colors = np.zeros((0, 3), np.float32)
    else:
        vpos = camera.view_positions(molecule.positions).astype(np.float32)
        colors = np.asarray(
            style.color_override if style.color_override is not None else molecule.element_colors(),
            dtype=np.float32,
        )

    if molecule.n_atoms == 0:
        spheres = SphereBatch.empty()
    else:
        radii = _atom_radii(molecule, style).astype(np.float32)
        spheres = SphereBatch(centers=vpos, radii=radii, colors=colors)

    draw_bonds = style.representation in ("ball_and_stick", "wireframe", "licorice") and molecule.bonds
    if draw_bonds:
        rb = style.bond_radius * (1.6 if style.representation == "licorice" else 1.0)
        idx_i = np.array([i for i, j, _order in molecule.bonds], dtype=np.int64)
        idx_j = np.array([j for i, j, _order in molecule.bonds], dtype=np.int64)
        bond_a = vpos[idx_i]
        bond_b = vpos[idx_j]
        bond_r = np.full(len(molecule.bonds), rb, dtype=np.float32)
        bond_ca = colors[idx_i]
        bond_cb = colors[idx_j]
    else:
        bond_a = bond_b = bond_ca = bond_cb = np.zeros((0, 3), np.float32)
        bond_r = np.zeros((0,), np.float32)

    # Arrow shafts are literally cylinders -- fold them into the same batch
    # as bonds. Arrow heads (cones) need the new GL primitive. Reuse the
    # atom view positions already computed above rather than transforming
    # them again.
    geom = build_arrow_geometry(molecule, camera, view_pos=vpos)
    cylinders = CylinderBatch(
        a=np.concatenate([bond_a, geom.shaft_a.astype(np.float32)]),
        b=np.concatenate([bond_b, geom.shaft_b.astype(np.float32)]),
        radii=np.concatenate([bond_r, geom.shaft_radius.astype(np.float32)]),
        colors_a=np.concatenate([bond_ca, geom.shaft_color.astype(np.float32)]),
        colors_b=np.concatenate([bond_cb, geom.shaft_color.astype(np.float32)]),
    )
    cones = ConeBatch(
        base=geom.head_base.astype(np.float32),
        apex=geom.head_apex.astype(np.float32),
        radius=geom.head_radius.astype(np.float32),
        color=geom.head_color.astype(np.float32),
    )

    proj = _build_projection(camera.zoom, camera.pan, width, height, camera.extent)

    shading = ShadingParams(
        ambient=style.ambient,
        specular_strength=style.specular_strength,
        shininess=style.shininess,
        light_dir=tuple(style.light_dir),
        fill_light=style.fill_light,
        depth_cue=style.depth_cue,
        background=tuple(style.background),
        transparent=style.transparent,
    )

    return spheres, cylinders, cones, proj, shading
