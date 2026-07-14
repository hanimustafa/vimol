"""Per-atom vector field -> arrow geometry, shared by both render backends.

Each ``Molecule.vector_fields`` entry is a raw ``(N, 3)`` array of
view-independent directions/magnitudes; this module turns it into view-space
arrow segments (a cylindrical shaft + a conical head per arrow) that
``render.Renderer`` (CPU) and ``gl_adapter`` (GPU) both consume, so the arrow
geometry itself is defined once. Like ``render.py``, this module only depends
on ``camera.py``/``molecule.py`` -- no GL imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .molecule import Molecule


@dataclass
class ArrowGeometry:
    """View-space arrow segments, split into shaft cylinders and head cones."""
    shaft_a: np.ndarray        # (M, 3) view space -- base (at the atom)
    shaft_b: np.ndarray        # (M, 3) view space -- where the head starts
    shaft_radius: np.ndarray   # (M,)
    shaft_color: np.ndarray    # (M, 3)
    head_base: np.ndarray      # (M, 3) view space -- wide end of the cone
    head_apex: np.ndarray      # (M, 3) view space -- arrow tip
    head_radius: np.ndarray    # (M,) -- cone base radius
    head_color: np.ndarray     # (M, 3)

    @staticmethod
    def empty() -> "ArrowGeometry":
        z3 = np.zeros((0, 3), np.float64)
        z1 = np.zeros((0,), np.float64)
        return ArrowGeometry(
            shaft_a=z3.copy(), shaft_b=z3.copy(), shaft_radius=z1.copy(), shaft_color=z3.copy(),
            head_base=z3.copy(), head_apex=z3.copy(), head_radius=z1.copy(), head_color=z3.copy(),
        )


def build_arrow_geometry(mol: Molecule, camera: Camera, view_pos=None) -> ArrowGeometry:
    """Convert ``mol.vector_fields`` into view-space arrow geometry under *camera*.

    *view_pos* optionally supplies the atoms' already-computed view-space
    positions (``camera.view_positions(mol.positions)``); the CPU renderer and
    GL adapter both have it on hand, so passing it avoids re-running that
    per-frame transform.
    """
    if not mol.vector_fields or mol.n_atoms == 0:
        return ArrowGeometry.empty()

    view_pos = camera.view_positions(mol.positions) if view_pos is None else np.asarray(view_pos, float)
    shaft_a, shaft_b, shaft_r, shaft_c = [], [], [], []
    head_base, head_apex, head_r, head_c = [], [], [], []

    for vf in mol.vector_fields:
        vectors = np.asarray(vf.vectors, np.float64)
        if vectors.shape != (mol.n_atoms, 3):
            continue
        mag = np.linalg.norm(vectors, axis=1)
        keep = mag > 1e-9
        if not keep.any():
            continue

        view_vecs = camera.view_directions(vectors[keep])
        base = view_pos[keep]
        tip = base + view_vecs * vf.scale
        frac = float(np.clip(vf.head_length_frac, 0.0, 1.0))
        head_start = base + (tip - base) * (1.0 - frac)

        n = base.shape[0]
        color = np.asarray(vf.color, np.float64)
        shaft_a.append(base)
        shaft_b.append(head_start)
        shaft_r.append(np.full(n, vf.radius))
        shaft_c.append(np.tile(color, (n, 1)))
        head_base.append(head_start)
        head_apex.append(tip)
        head_r.append(np.full(n, vf.radius * vf.head_scale))
        head_c.append(np.tile(color, (n, 1)))

    if not shaft_a:
        return ArrowGeometry.empty()

    return ArrowGeometry(
        shaft_a=np.concatenate(shaft_a), shaft_b=np.concatenate(shaft_b),
        shaft_radius=np.concatenate(shaft_r), shaft_color=np.concatenate(shaft_c),
        head_base=np.concatenate(head_base), head_apex=np.concatenate(head_apex),
        head_radius=np.concatenate(head_r), head_color=np.concatenate(head_c),
    )
