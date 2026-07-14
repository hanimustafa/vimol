"""Orthographic trackball camera.

The camera keeps a 3x3 rotation matrix (world -> view), a look-at center, and a
zoom expressed as pixels-per-angstrom. View space is right-handed with +x
right, +y up, +z toward the viewer, so the renderer keeps the fragment with the
largest z (nearest the camera).
"""
from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, float))
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


class Camera:
    def __init__(self, center=None, extent: float = 5.0):
        self.center = np.zeros(3) if center is None else np.array(center, float)
        self.rotation = np.eye(3)
        # zoom is pixels per angstrom; set by fit()
        self.zoom = 40.0
        self.extent = float(extent)
        # pan offset in screen pixels
        self.pan = np.zeros(2)

    # -- framing ----------------------------------------------------------
    def fit(self, width: int, height: int, extent: float, margin: float = 1.15) -> None:
        """Choose a zoom so a sphere of *extent* angstrom fills the viewport."""
        self.extent = float(extent)
        half = max(min(width, height) * 0.5, 1.0)
        self.zoom = half / (extent * margin)
        self.pan = np.zeros(2)

    # -- interaction ------------------------------------------------------
    def orbit(self, dx: float, dy: float, speed: float = 0.01) -> None:
        """Rotate about screen axes. dx/dy are in pixels."""
        if dx:
            self.rotation = rotation_from_axis_angle([0, 1, 0], dx * speed) @ self.rotation
        if dy:
            self.rotation = rotation_from_axis_angle([1, 0, 0], dy * speed) @ self.rotation

    def roll(self, angle: float) -> None:
        self.rotation = rotation_from_axis_angle([0, 0, 1], angle) @ self.rotation

    def zoom_by(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 1.0, 1e5))

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan = self.pan + np.array([dx, dy], float)

    def reset(self) -> None:
        self.rotation = np.eye(3)
        self.pan = np.zeros(2)

    # -- transforms -------------------------------------------------------
    def view_positions(self, positions: np.ndarray) -> np.ndarray:
        """World coordinates -> view-space coordinates (angstrom)."""
        return (positions - self.center) @ self.rotation.T

    def view_directions(self, directions: np.ndarray) -> np.ndarray:
        """World -> view space for free vectors (rotation only, no translation)."""
        return np.asarray(directions, float) @ self.rotation.T
