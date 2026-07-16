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

from .arrows import ArrowGeometry, build_arrow_geometry
from .camera import Camera
from .molecule import Molecule


@dataclass
class Style:
    """Rendering options."""
    representation: str = "ball_and_stick"   # ball_and_stick | spacefill | wireframe | licorice
    atom_scale: float = 0.25                 # multiplier on van der Waals radius (ball_and_stick)
    bond_radius: float = 0.10                # angstrom (cylinder radius)
    background: Tuple[float, float, float] = (0.05, 0.06, 0.09)
    transparent: bool = False                # emit RGBA with an alpha cutout (terminal shows through)
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
    # ball-and-stick: scale by van der Waals radius so hydrogens stay visible
    # (their covalent radius is tiny, which would hide them inside the bonds).
    return mol.vdw_radii() * style.atom_scale


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
        transparent = style.transparent
        if transparent:
            # start black so undrawn pixels are premultiplied-zero; alpha comes
            # from the z-buffer at the end (drawn <-> not drawn).
            color = np.zeros((H, W, 3), np.float32)
        else:
            bg = np.array(style.background, np.float32)
            color = np.empty((H, W, 3), np.float32)
            color[...] = bg  # broadcast-fill (cheaper than np.tile's repeat)
        zbuf = np.full((H, W), -np.inf, np.float32)
        if mol.n_atoms == 0:
            return self._finish(color, zbuf, transparent)

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

        # Build arrow geometry up front (reusing vpos instead of recomputing
        # the view transform) so the fog depth range below can include the
        # arrow endpoints -- the GL backend derives its single depth range
        # from atoms + bonds + arrow endpoints, and the fog must match.
        geom = build_arrow_geometry(mol, camera, view_pos=vpos) if mol.vector_fields else None
        has_arrows = geom is not None and geom.shaft_a.shape[0] > 0

        # depth range for fog
        z_parts = [sz]
        if has_arrows:
            z_parts += [geom.shaft_a[:, 2], geom.shaft_b[:, 2],
                        geom.head_base[:, 2], geom.head_apex[:, 2]]
        z_all = np.concatenate([np.asarray(z, np.float64).reshape(-1) for z in z_parts])
        zmin = float(z_all.min()) if z_all.size else 0.0
        zmax = float(z_all.max()) if z_all.size else 1.0
        zspan = max(zmax - zmin, 1e-6)

        # columns are the three lighting directions, so a single (M,3)@(3,3)
        # matmul yields all three dot products at once (one BLAS call instead
        # of three), which is cheaper per fragment than dotting separately.
        shade_mat = np.stack([light, fill, halfv], axis=1)

        def shade(normals, albedo):
            # normals: (M,3) view-space unit; albedo: (M,3) or (3,)
            # dots of unit vectors never exceed 1, so a lower clamp at 0 is all
            # that's needed -- np.maximum avoids np.clip's per-call overhead.
            d = normals @ shade_mat                       # (M,3): n·light, n·fill, n·halfv
            np.maximum(d, 0.0, out=d)
            spec = np.power(d[:, 2], style.shininess) * style.specular_strength
            diff = (style.ambient + d[:, 0] + d[:, 1] * style.fill_light)[:, None]
            return albedo * diff + spec[:, None]

        radii = _atom_radii(mol, style)

        draw_bonds = style.representation in ("ball_and_stick", "wireframe", "licorice")
        if draw_bonds and mol.bonds:
            self._draw_bonds(mol, vpos, zoom, ox_s, oy_s, style, base_colors,
                             color, zbuf, shade, zmin, zspan)

        # atoms drawn after bonds; z-buffer keeps whichever is nearer
        inv_zoom = 1.0 / zoom
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
            # squared screen-space distance to the atom centre, via broadcasting
            # a row and a column (cheaper than materializing a full meshgrid).
            dxr = np.arange(x0, x1) - cx          # (w,)
            dyr = np.arange(y0, y1) - cy          # (h,)
            d2 = (dxr * dxr)[None, :] + (dyr * dyr)[:, None]   # (h,w)
            mask = d2 <= sr * sr
            if not mask.any():
                continue
            # sphere depth: front-surface height above the view plane
            h2 = r * r - d2 * (inv_zoom * inv_zoom)
            np.maximum(h2, 0.0, out=h2)
            depth = cz + np.sqrt(h2)
            sub_z = zbuf[y0:y1, x0:x1]
            win = mask & (depth > sub_z)
            if not win.any():
                continue
            # build normals + shade only the fragments that survive the z-test
            wy, wx = np.nonzero(win)
            dw = depth[wy, wx]
            inv_r = 1.0 / r
            normals = np.empty((wy.size, 3), np.float64)
            normals[:, 0] = dxr[wx] * (inv_zoom * inv_r)
            normals[:, 1] = -dyr[wy] * (inv_zoom * inv_r)
            normals[:, 2] = (dw - cz) * inv_r
            rgb = shade(normals, base_colors[a])
            self._apply_fog(rgb, dw, style, zmin, zspan)
            sub_c = color[y0:y1, x0:x1]
            sub_c[wy, wx] = rgb
            sub_z[wy, wx] = dw

        if has_arrows:
            self._draw_arrow_shafts(geom, zoom, ox_s, oy_s, style, color, zbuf, shade, zmin, zspan)
            self._draw_arrow_heads(geom, zoom, ox_s, oy_s, style, color, zbuf, shade, zmin, zspan)

        return self._finish(color, zbuf, transparent)

    @staticmethod
    def _finish(color: np.ndarray, zbuf: np.ndarray, transparent: bool) -> np.ndarray:
        img = np.clip(color, 0.0, 1.0)
        rgb = (img * 255).astype(np.uint8)
        if not transparent:
            return rgb
        alpha = np.where(zbuf > -np.inf, 255, 0).astype(np.uint8)
        return np.dstack([rgb, alpha])

    # ------------------------------------------------------------------
    def _draw_bonds(self, mol, vpos, zoom, ox_s, oy_s, style, base_colors,
                    color, zbuf, shade, zmin, zspan):
        rb = style.bond_radius if style.representation != "licorice" else style.bond_radius * 1.6
        for (i, j, order) in mol.bonds:
            self._draw_cylinder_segment(vpos[i], vpos[j], rb, base_colors[i], base_colors[j],
                                        zoom, ox_s, oy_s, style, color, zbuf, shade, zmin, zspan)

    def _draw_arrow_shafts(self, geom: ArrowGeometry, zoom, ox_s, oy_s, style,
                           color, zbuf, shade, zmin, zspan):
        for k in range(geom.shaft_a.shape[0]):
            c = geom.shaft_color[k].astype(np.float32)
            self._draw_cylinder_segment(geom.shaft_a[k], geom.shaft_b[k], float(geom.shaft_radius[k]),
                                        c, c, zoom, ox_s, oy_s, style, color, zbuf, shade, zmin, zspan)

    def _draw_arrow_heads(self, geom: ArrowGeometry, zoom, ox_s, oy_s, style,
                          color, zbuf, shade, zmin, zspan):
        for k in range(geom.head_base.shape[0]):
            self._draw_cone_segment(geom.head_base[k], geom.head_apex[k], float(geom.head_radius[k]),
                                    geom.head_color[k], zoom, ox_s, oy_s, style, color, zbuf, shade,
                                    zmin, zspan)

    # ------------------------------------------------------------------
    def _draw_cylinder_segment(self, a, b, radius, color_a, color_b, zoom, ox_s, oy_s,
                               style, color, zbuf, shade, zmin, zspan):
        """Rasterize one capped-cylinder impostor from view-space *a* to *b*.

        Shared by real bonds (``_draw_bonds``, two atom colors split at the
        midpoint) and arrow shafts (``_draw_arrow_shafts``, one uniform color).
        """
        W, H = self.width, self.height
        ax, ay, az = a
        bx, by, bz = b
        axis = np.array([bx - ax, by - ay, bz - az])
        L = np.linalg.norm(axis)
        if L < 1e-6 or radius <= 0:
            return
        u = axis / L
        srb = radius * zoom
        sax, say = ox_s + ax * zoom, oy_s - ay * zoom
        sbx, sby = ox_s + bx * zoom, oy_s - by * zoom
        x0 = max(int(np.floor(min(sax, sbx) - srb)), 0)
        x1 = min(int(np.ceil(max(sax, sbx) + srb)) + 1, W)
        y0 = max(int(np.floor(min(say, sby) - srb)), 0)
        y1 = min(int(np.ceil(max(say, sby) + srb)) + 1, H)
        if x0 >= x1 or y0 >= y1:
            return
        # pixel -> view-plane coordinates (angstrom), broadcasting a row/column
        gx = np.arange(x0, x1)[None, :]
        gy = np.arange(y0, y1)[:, None]
        vx = (gx - ox_s) / zoom
        vy = (oy_s - gy) / zoom
        ex = vx - ax
        ey = vy - ay
        uz = u[2]
        A0 = ex * u[0] + ey * u[1]
        a2 = 1.0 - uz * uz
        b2 = -2.0 * A0 * uz
        c2 = ex * ex + ey * ey - A0 * A0 - radius * radius
        if abs(a2) < 1e-9:
            # cylinder axis ~ parallel to view direction: treat as disc
            disc = -c2  # = radius^2 - (ex^2+ey^2-A0^2)
            mask = disc >= 0
            s = np.zeros_like(ex)
        else:
            disc = b2 * b2 - 4 * a2 * c2
            mask = disc >= 0
            sq = np.sqrt(np.clip(disc, 0, None))
            s = (-b2 + sq) / (2 * a2)  # front (larger z) root
        if not mask.any():
            return
        zview = az + s
        # axial coordinate to clamp to the segment and pick the color half
        t = A0 + uz * s  # (P-a).u
        within = (t >= 0) & (t <= L)
        mask = mask & within
        if not mask.any():
            return
        # surface normal = (w - (w.u)u)/radius
        wx = ex
        wy = ey
        wz = s  # (z-az)
        proj = t  # w.u
        nx = (wx - proj * u[0]) / radius
        ny = (wy - proj * u[1]) / radius
        nz = (wz - proj * u[2]) / radius
        normals = np.stack([nx, ny, nz], axis=-1)
        # split color at midpoint
        frac = t / L
        albedo = np.where((frac < 0.5)[..., None], color_a, color_b).astype(np.float32)
        self._composite(color, zbuf, x0, x1, y0, y1, mask, zview.astype(np.float32),
                        normals, albedo, shade, style, zmin, zspan)

    # ------------------------------------------------------------------
    def _draw_cone_segment(self, base, apex, radius, color_rgb, zoom, ox_s, oy_s,
                           style, color, zbuf, shade, zmin, zspan):
        """Rasterize one cone impostor: *radius* at *base*, tapering linearly
        to a point at *apex* (view space). Used for arrow heads.

        Same analytic setup as ``_draw_cylinder_segment`` (``A0``, ``a2``,
        ``b2`` from the axis/offset), but the target surface radius varies
        linearly along the axis instead of being constant, which folds an
        extra ``(C + D*s)^2`` term into the quadratic. Unlike the cylinder,
        the resulting leading coefficient isn't sign-guaranteed, so both
        roots are evaluated and the nearer-camera (larger ``s``) one that
        satisfies ``0 <= t <= L`` *and* non-negative radius (rejecting the
        mirror nappe beyond the apex that squaring introduces) wins.
        """
        W, H = self.width, self.height
        ax, ay, az = base
        bx, by, bz = apex
        axis = np.array([bx - ax, by - ay, bz - az])
        L = np.linalg.norm(axis)
        if L < 1e-6 or radius <= 0:
            return
        u = axis / L
        R = radius
        srb = R * zoom
        sax, say = ox_s + ax * zoom, oy_s - ay * zoom
        sbx, sby = ox_s + bx * zoom, oy_s - by * zoom
        x0 = max(int(np.floor(min(sax, sbx) - srb)), 0)
        x1 = min(int(np.ceil(max(sax, sbx) + srb)) + 1, W)
        y0 = max(int(np.floor(min(say, sby) - srb)), 0)
        y1 = min(int(np.ceil(max(say, sby) + srb)) + 1, H)
        if x0 >= x1 or y0 >= y1:
            return
        gx = np.arange(x0, x1)[None, :]
        gy = np.arange(y0, y1)[:, None]
        vx = (gx - ox_s) / zoom
        vy = (oy_s - gy) / zoom
        ex = vx - ax
        ey = vy - ay
        uz = u[2]
        A0 = ex * u[0] + ey * u[1]
        a2 = 1.0 - uz * uz
        b2 = -2.0 * A0 * uz
        c2 = ex * ex + ey * ey - A0 * A0  # no -R^2 here (radius varies with t)

        k = R / L
        C = R - k * A0
        D = -k * uz
        a2p = a2 - D * D
        b2p = b2 - 2.0 * C * D
        c2p = c2 - C * C

        near_parallel = np.abs(a2p) < 1e-9
        safe_a2p = np.where(near_parallel, 1.0, a2p)
        disc = b2p * b2p - 4.0 * a2p * c2p
        has_root = disc >= 0
        sq = np.sqrt(np.clip(disc, 0, None))
        s_plus = (-b2p + sq) / (2.0 * safe_a2p)
        s_minus = (-b2p - sq) / (2.0 * safe_a2p)

        safe_b2p = np.where(np.abs(b2p) < 1e-12, 1.0, b2p)
        s_lin = -c2p / safe_b2p
        lin_valid = near_parallel & (np.abs(b2p) >= 1e-12)

        def _valid(s):
            t = A0 + uz * s
            radius_at = C + D * s
            return (t >= 0) & (t <= L) & (radius_at >= -1e-6)

        v_plus = has_root & ~near_parallel & _valid(s_plus)
        v_minus = has_root & ~near_parallel & _valid(s_minus)
        v_lin = lin_valid & _valid(s_lin)

        neg_inf = -1e30
        s = np.maximum(np.maximum(
            np.where(v_plus, s_plus, neg_inf),
            np.where(v_minus, s_minus, neg_inf)),
            np.where(v_lin, s_lin, neg_inf))
        mask = s > neg_inf / 2
        if not mask.any():
            return

        t = A0 + uz * s
        zview = az + s
        wx = ex - t * u[0]
        wy = ey - t * u[1]
        wz = s - t * u[2]
        r_t = np.clip(C + D * s, 1e-6, None)
        nx = (wx / r_t) * L + u[0] * R
        ny = (wy / r_t) * L + u[1] * R
        nz = (wz / r_t) * L + u[2] * R
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
        nlen = np.where(nlen < 1e-9, 1.0, nlen)
        normals = np.stack([nx / nlen, ny / nlen, nz / nlen], axis=-1)
        albedo = np.asarray(color_rgb, np.float32)
        self._composite(color, zbuf, x0, x1, y0, y1, mask, zview.astype(np.float32),
                        normals, albedo, shade, style, zmin, zspan)

    # ------------------------------------------------------------------
    @staticmethod
    def _composite(color, zbuf, x0, x1, y0, y1, mask, depth, normals, albedo,
                   shade, style, zmin, zspan):
        """Depth-test the primitive's bounding box, then shade only the pixels
        that survive.

        Shading (dot products, specular power) and fog are the expensive part,
        so they run *after* the visibility test on the surviving 1-D fragment
        list rather than over the whole rectangle -- pixels outside the mask or
        hidden behind a nearer surface are never shaded at all.
        """
        sub_z = zbuf[y0:y1, x0:x1]
        win = mask & (depth > sub_z)
        if not win.any():
            return
        nrm = normals[win]                                   # (M,3)
        alb = albedo if albedo.ndim == 1 else albedo[win]    # (3,) or (M,3)
        rgb = shade(nrm, alb)                                # (M,3)
        dw = depth[win]
        Renderer._apply_fog(rgb, dw, style, zmin, zspan)
        sub_c = color[y0:y1, x0:x1]
        sub_c[win] = rgb
        sub_z[win] = dw

    @staticmethod
    def _apply_fog(rgb, depth_win, style, zmin, zspan):
        """Darken fragments toward the back of the scene, in place.

        ``rgb`` is the (M,3) list of surviving fragment colors; ``depth_win``
        their depths. Uses ``maximum``/``minimum`` ufuncs rather than
        ``np.clip`` to avoid the latter's per-call scalar checks, which are a
        measurable cost when invoked once per primitive per frame.
        """
        if style.depth_cue <= 0:
            return
        f = (depth_win - zmin) / zspan  # 0 back .. 1 front
        np.maximum(f, 0.0, out=f)
        np.minimum(f, 1.0, out=f)
        rgb *= (1.0 - style.depth_cue * (1.0 - f))[:, None]
