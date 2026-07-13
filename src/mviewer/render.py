"""Software raycaster: renders a molecule to an RGB pixel buffer with numpy.

The technique is *impostor rendering* under an orthographic camera. Each atom
is a sphere and each bond a cylinder; instead of tessellating them into
triangles we solve the ray/surface intersection analytically per pixel over the
primitive's screen-space bounding box. A shared z-buffer resolves occlusion, so
spheres and cylinders interpenetrate correctly. Everything is vectorized over
the pixels of each primitive, which keeps interactive frame-rates for
small/medium molecules in pure Python + numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .camera import Camera
from .molecule import Molecule


@dataclass
class Style:
    """Rendering options."""
    representation: str = "ball_and_stick"   # ball_and_stick | spacefill | wireframe | licorice
    atom_scale: float = 0.24                 # multiplier on covalent radius (ball_and_stick)
    bond_radius: float = 0.13                # angstrom (cylinder radius)
    background: Tuple[float, float, float] = (0.05, 0.06, 0.09)
    ambient: float = 0.28
    specular_strength: float = 0.55
    shininess: float = 32.0
    light_dir: Tuple[float, float, float] = (0.35, 0.55, 0.75)
    fill_light: float = 0.25                 # secondary light from opposite side
    depth_cue: float = 0.55                  # 0 = off, 1 = strong fog toward the back
    outline: bool = True                     # subtle silhouette darkening
    color_override: object = None            # optional (N,3) array to replace element colors


def _atom_radii(mol: Molecule, style: Style) -> np.ndarray:
    if style.representation == "spacefill":
        return mol.vdw_radii()
    if style.representation in ("wireframe", "licorice"):
        return np.full(mol.n_atoms, style.bond_radius * (1.6 if style.representation == "licorice" else 1.0))
    return mol.covalent_radii() * style.atom_scale


class Renderer:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)

    def resize(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)

    # ------------------------------------------------------------------
    def render(self, mol: Molecule, camera: Camera, style: Style | None = None) -> np.ndarray:
        style = style or Style()
        W, H = self.width, self.height
        bg = np.array(style.background, np.float32)
        color = np.tile(bg, (H, W, 1)).astype(np.float32)
        zbuf = np.full((H, W), -np.inf, np.float32)
        if mol.n_atoms == 0:
            return (np.clip(color, 0, 1) * 255).astype(np.uint8)

        vpos = camera.view_positions(mol.positions).astype(np.float64)  # (N,3) view space
        zoom = camera.zoom
        ox_s = W * 0.5 + camera.pan[0]
        oy_s = H * 0.5 - camera.pan[1]
        # screen coords (y axis flips: view +y is up, screen +y is down)
        sx = ox_s + vpos[:, 0] * zoom
        sy = oy_s - vpos[:, 1] * zoom
        sz = vpos[:, 2]  # depth, larger == nearer viewer

        base_colors = style.color_override if style.color_override is not None else mol.element_colors()
        base_colors = np.asarray(base_colors, np.float32)

        light = np.array(style.light_dir, np.float64)
        light = light / (np.linalg.norm(light) + 1e-12)
        view_dir = np.array([0.0, 0.0, 1.0])
        halfv = light + view_dir
        halfv = halfv / (np.linalg.norm(halfv) + 1e-12)
        fill = np.array([-light[0], -light[1], 0.6])
        fill = fill / (np.linalg.norm(fill) + 1e-12)

        # depth range for fog
        zmin = float(sz.min()) if len(sz) else 0.0
        zmax = float(sz.max()) if len(sz) else 1.0
        zspan = max(zmax - zmin, 1e-6)

        def shade(normals, albedo):
            # normals: (...,3) view-space unit; albedo: (...,3)
            nl = np.clip(normals @ light, 0, 1)
            nf = np.clip(normals @ fill, 0, 1) * style.fill_light
            nh = np.clip(normals @ halfv, 0, 1)
            spec = np.power(nh, style.shininess) * style.specular_strength
            diff = (style.ambient + nl + nf)[..., None]
            out = albedo * diff + spec[..., None]
            return out

        radii = _atom_radii(mol, style)

        draw_bonds = style.representation in ("ball_and_stick", "wireframe", "licorice")
        if draw_bonds and mol.bonds:
            self._draw_bonds(
                mol, vpos, sx, sy, sz, zoom, ox_s, oy_s, style, base_colors,
                color, zbuf, shade,
            )

        # atoms drawn after bonds; z-buffer keeps whichever is nearer
        order = np.argsort(sz)  # far to near so specular highlights aren't clobbered oddly
        for a in order:
            r = radii[a]
            if r <= 0:
                continue
            sr = r * zoom
            if sr < 0.5:
                sr = 0.5
            cx, cy, cz = sx[a], sy[a], sz[a]
            x0 = max(int(np.floor(cx - sr)), 0)
            x1 = min(int(np.ceil(cx + sr)) + 1, W)
            y0 = max(int(np.floor(cy - sr)), 0)
            y1 = min(int(np.ceil(cy + sr)) + 1, H)
            if x0 >= x1 or y0 >= y1:
                continue
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            gx, gy = np.meshgrid(xs, ys)
            dx = gx - cx
            dy = gy - cy
            d2 = dx * dx + dy * dy
            mask = d2 <= sr * sr
            if not mask.any():
                continue
            # view-plane offsets in angstrom
            ox = dx / zoom
            oy = -dy / zoom
            h2 = r * r - ox * ox - oy * oy
            np.clip(h2, 0.0, None, out=h2)
            h = np.sqrt(h2)
            depth = cz + h
            normals = np.stack([ox / r, oy / r, h / r], axis=-1)
            albedo = np.broadcast_to(base_colors[a], normals.shape).astype(np.float32)
            shaded = shade(normals, albedo)
            self._composite(color, zbuf, x0, x1, y0, y1, mask, depth, shaded, style, cz, zmin, zspan)

        img = np.clip(color, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def _draw_bonds(self, mol, vpos, sx, sy, sz, zoom, ox_s, oy_s, style, base_colors,
                    color, zbuf, shade):
        W, H = self.width, self.height
        rb = style.bond_radius if style.representation != "licorice" else style.bond_radius * 1.6
        srb = rb * zoom
        for (i, j, order) in mol.bonds:
            ax, ay, az = vpos[i]
            bx, by, bz = vpos[j]
            axis = np.array([bx - ax, by - ay, bz - az])
            L = np.linalg.norm(axis)
            if L < 1e-6:
                continue
            u = axis / L
            # screen-space bounding box
            sax, say = sx[i], sy[i]
            sbx, sby = sx[j], sy[j]
            x0 = max(int(np.floor(min(sax, sbx) - srb)), 0)
            x1 = min(int(np.ceil(max(sax, sbx) + srb)) + 1, W)
            y0 = max(int(np.floor(min(say, sby) - srb)), 0)
            y1 = min(int(np.ceil(max(say, sby) + srb)) + 1, H)
            if x0 >= x1 or y0 >= y1:
                continue
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            gx, gy = np.meshgrid(xs, ys)
            # pixel -> view-plane coordinates (angstrom)
            vx = (gx - ox_s) / zoom
            vy = (oy_s - gy) / zoom
            ex = vx - ax
            ey = vy - ay
            uz = u[2]
            A0 = ex * u[0] + ey * u[1]
            a2 = 1.0 - uz * uz
            b2 = -2.0 * A0 * uz
            c2 = ex * ex + ey * ey - A0 * A0 - rb * rb
            if abs(a2) < 1e-9:
                # cylinder axis ~ parallel to view direction: treat as disc
                disc = -c2  # = rb^2 - (ex^2+ey^2-A0^2)
                mask = disc >= 0
                s = np.zeros_like(ex)
            else:
                disc = b2 * b2 - 4 * a2 * c2
                mask = disc >= 0
                sq = np.sqrt(np.clip(disc, 0, None))
                s = (-b2 + sq) / (2 * a2)  # front (larger z) root
            if not mask.any():
                continue
            zview = az + s
            # axial coordinate to clamp to the segment and pick the color half
            t = A0 + uz * s  # (P-a).u
            within = (t >= 0) & (t <= L)
            mask = mask & within
            if not mask.any():
                continue
            # surface normal = (w - (w.u)u)/rb
            wx = ex
            wy = ey
            wz = s  # (z-az)
            proj = t  # w.u
            nx = (wx - proj * u[0]) / rb
            ny = (wy - proj * u[1]) / rb
            nz = (wz - proj * u[2]) / rb
            normals = np.stack([nx, ny, nz], axis=-1)
            # split color at midpoint
            frac = t / L
            ca = base_colors[i]
            cb = base_colors[j]
            albedo = np.where((frac < 0.5)[..., None], ca, cb).astype(np.float32)
            shaded = shade(normals, albedo)
            zmin = float(sz.min())
            zspan = max(float(sz.max()) - zmin, 1e-6)
            self._composite(color, zbuf, x0, x1, y0, y1, mask, zview.astype(np.float32),
                            shaded, style, None, zmin, zspan)

    # ------------------------------------------------------------------
    @staticmethod
    def _composite(color, zbuf, x0, x1, y0, y1, mask, depth, shaded, style, _cz, zmin, zspan):
        sub_z = zbuf[y0:y1, x0:x1]
        win = mask & (depth > sub_z)
        if not win.any():
            return
        rgb = shaded
        if style.depth_cue > 0:
            # fog toward the back: fragments with smaller depth get darker/bluer
            f = (depth - zmin) / zspan  # 0 back .. 1 front
            fog = 1.0 - style.depth_cue * (1.0 - np.clip(f, 0, 1))
            rgb = rgb * fog[..., None]
        sub_c = color[y0:y1, x0:x1]
        sub_c[win] = rgb[win]
        sub_z[win] = depth[win]
