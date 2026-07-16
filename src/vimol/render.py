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

import os
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


# render worker threads: bands of the frame raycast concurrently (numpy
# releases the GIL inside array ops). Capped: beyond ~8 the bands get too
# short and per-primitive python overhead starts to dominate.
_POOL_WORKERS = min(8, os.cpu_count() or 1)


def _atom_radii(mol: Molecule, style: Style) -> np.ndarray:
    if style.representation == "spacefill":
        return mol.vdw_radii()
    if style.representation in ("wireframe", "licorice"):
        return np.full(mol.n_atoms, style.bond_radius * (1.6 if style.representation == "licorice" else 1.0))
    # ball-and-stick: scale by van der Waals radius so hydrogens stay visible
    # (their covalent radius is tiny, which would hide them inside the bonds).
    return mol.vdw_radii() * style.atom_scale


def _fast_pow(x: np.ndarray, exponent: float) -> np.ndarray:
    """x ** exponent, using repeated squaring for power-of-two exponents.

    The specular term raises every fragment's n.h to ``style.shininess``
    (default 32). ``np.power`` with a float exponent goes through exp/log per
    element; for the (very common) integral power-of-two case, log2(e) in-place
    multiplies do the same thing several times faster. Falls back to
    ``np.power`` for any other exponent.
    """
    e = int(exponent)
    if e == exponent and e > 0 and (e & (e - 1)) == 0:
        out = x * x                       # e >= 2 from here (e == 1 is the loop's no-op)
        for _ in range(e.bit_length() - 2):
            np.multiply(out, out, out=out)
        return out if e > 1 else x.copy()
    return np.power(x, exponent)


class Renderer:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        # persistent framebuffers, re-filled per frame -- allocating ~25 MB of
        # fresh pages every frame (np.full/np.empty) costs real time at
        # supersampled full-screen sizes; reuse keeps the pages warm.
        self._color = None
        self._zbuf = None
        self._bg_template = None     # pre-filled background frame (see below)
        self._bg_key = None
        self._pool = None            # lazy render thread pool (see _threads)

    def resize(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self._color = None
        self._zbuf = None
        self._bg_template = None
        self._bg_key = None

    def _framebuffers(self, style: Style):
        H, W = self.height, self.width
        if self._color is None or self._color.shape[:2] != (H, W):
            self._color = np.empty((H, W, 3), np.float32)
            self._zbuf = np.empty((H, W), np.float32)
        if style.transparent:
            self._color.fill(0.0)   # premultiplied-zero for undrawn pixels
        else:
            # Broadcasting a (3,) background into (H, W, 3) is a strided store
            # that measures >10x slower than a flat memcpy at full-screen sizes
            # (~100 ms vs ~8 ms at 3200x2000) -- so keep one pre-filled frame
            # per background color and copy it in whole.
            key = tuple(style.background)
            if self._bg_key != key or self._bg_template is None \
                    or self._bg_template.shape[:2] != (H, W):
                self._bg_template = np.empty((H, W, 3), np.float32)
                self._bg_template[...] = np.asarray(style.background, np.float32)
                self._bg_key = key
            np.copyto(self._color, self._bg_template)
        self._zbuf.fill(-np.inf)
        return self._color, self._zbuf

    # ------------------------------------------------------------------
    def render(self, mol: Molecule, camera: Camera, style: Style | None = None) -> np.ndarray:
        style = style or Style()
        W, H = self.width, self.height
        transparent = style.transparent
        color, zbuf = self._framebuffers(style)
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
        # float32 throughout: fragment lists are large, and halving their
        # width halves the memory traffic of every shading op.
        shade_mat = np.stack([light, fill, halfv], axis=1).astype(np.float32)

        def shade(normals, albedo):
            # normals: (M,3) view-space unit float32; albedo: (M,3) or (3,)
            # dots of unit vectors never exceed 1, so a lower clamp at 0 is all
            # that's needed -- np.maximum avoids np.clip's per-call overhead.
            d = normals @ shade_mat                       # (M,3): n·light, n·fill, n·halfv
            np.maximum(d, 0.0, out=d)
            spec = _fast_pow(d[:, 2], style.shininess)
            spec *= style.specular_strength
            diff = (style.ambient + d[:, 0] + d[:, 1] * style.fill_light)[:, None]
            return albedo * diff + spec[:, None]

        # scalar copies of the light vectors for the full-bbox shading path
        l0, l1, l2 = (float(v) for v in light)
        f0, f1, f2 = (float(v) for v in fill)
        hv0, hv1, hv2 = (float(v) for v in halfv)
        ambient = float(style.ambient)
        fill_w = float(style.fill_light)
        spec_w = float(style.specular_strength)
        shininess = style.shininess
        depth_cue = float(style.depth_cue)

        def shade_write(sub_c, sub_z, win, depth, nx, ny, nz, albedo):
            """Shade a full bounding box and masked-write the winners.

            ``nx``/``ny``/``nz`` may be broadcastable pieces ((1,w), (h,1) or
            (h,w)); ``albedo`` a single (3,) color or an (h,w,3) map. Shading
            the whole box contiguously and writing through
            ``np.copyto(..., where=win)`` measures ~3x faster than gathering
            the winners with ``np.nonzero`` and fancy-index scattering them
            back -- the scatter was the single hottest operation of a frame,
            far outweighing the extra arithmetic on masked-out pixels. Fog is
            folded into the diffuse/specular terms ((a*d+s)*m == a*(d*m)+s*m),
            saving a separate full-box multiply of the (h,w,3) color.
            """
            diff = nx * l0 + ny * l1 + nz * l2
            np.maximum(diff, 0.0, out=diff)
            dfill = nx * f0 + ny * f1 + nz * f2
            np.maximum(dfill, 0.0, out=dfill)
            dh = nx * hv0 + ny * hv1 + nz * hv2
            np.maximum(dh, 0.0, out=dh)
            spec = _fast_pow(dh, shininess)
            spec *= spec_w
            diff += ambient
            dfill *= fill_w
            diff += dfill
            if depth_cue > 0:
                fog = (depth - np.float32(zmin)) * np.float32(1.0 / zspan)
                np.maximum(fog, 0.0, out=fog)
                np.minimum(fog, 1.0, out=fog)
                fog *= depth_cue
                fog += 1.0 - depth_cue
                diff *= fog
                spec *= fog
            rgb = albedo * diff[..., None]
            rgb += spec[..., None]
            np.copyto(sub_c, rgb, where=win[..., None])
            np.copyto(sub_z, depth, where=win)

        radii = _atom_radii(mol, style)
        draw_bonds = style.representation in ("ball_and_stick", "wireframe", "licorice")
        inv_zoom = 1.0 / zoom
        order = np.argsort(sz)  # far to near so specular highlights aren't clobbered oddly

        # Per-primitive screen y-intervals, precomputed once so each band only
        # touches the primitives that actually reach its rows -- the python
        # per-primitive setup is GIL-serialized across bands, so handing every
        # band the whole primitive list would tax exactly the part threads
        # can't parallelize.
        sr_all = np.maximum(radii * zoom, 0.5)
        atom_ylo = sy - sr_all
        atom_yhi = sy + sr_all
        bond_list = list(mol.bonds) if (draw_bonds and mol.bonds) else []
        if bond_list:
            bi = np.array([b[0] for b in bond_list])
            bj = np.array([b[1] for b in bond_list])
            rb = style.bond_radius if style.representation != "licorice" else style.bond_radius * 1.6
            srb = rb * zoom
            bond_ylo = np.minimum(sy[bi], sy[bj]) - srb
            bond_yhi = np.maximum(sy[bi], sy[bj]) + srb

        def draw_band(y_lo: int, y_hi: int) -> None:
            """Raycast the band's primitives into rows [y_lo, y_hi).

            Bands own disjoint row ranges of the shared color/z buffers, so
            they can run on a thread pool with no locking; each band draws
            its primitives in the same global order, so per-pixel results are
            identical to a single full-height pass (the z-test decides every
            pixel independently). numpy releases the GIL inside its array
            ops, which is where nearly all of the time goes.
            """
            if bond_list:
                hit = np.nonzero((bond_yhi >= y_lo) & (bond_ylo < y_hi))[0]
                for k in hit:
                    i, j, _o = bond_list[k]
                    self._draw_cylinder_segment(
                        vpos[i], vpos[j], rb, base_colors[i], base_colors[j],
                        zoom, ox_s, oy_s, style, color, zbuf, shade_write,
                        zmin, zspan, y_lo, y_hi)

            # atoms drawn after bonds; z-buffer keeps whichever is nearer
            in_band = (atom_yhi[order] >= y_lo) & (atom_ylo[order] < y_hi)
            for a in order[in_band]:
                r = radii[a]
                if r <= 0:
                    continue
                sr = r * zoom
                if sr < 0.5:
                    sr = 0.5
                cx, cy, cz = sx[a], sy[a], sz[a]
                x0 = max(int(np.floor(cx - sr)), 0)
                x1 = min(int(np.ceil(cx + sr)) + 1, W)
                y0 = max(int(np.floor(cy - sr)), y_lo)
                y1 = min(int(np.ceil(cy + sr)) + 1, y_hi)
                if x0 >= x1 or y0 >= y1:
                    continue
                # squared screen-space distance to the atom centre, via
                # broadcasting a row and a column (cheaper than materializing
                # a full meshgrid). float32 per-pixel math: python-float
                # scalars don't upcast it, and it halves the bandwidth of
                # every full-bbox pass.
                dxr = np.arange(x0, x1, dtype=np.float32)
                dxr -= np.float32(cx)                 # (w,)
                dyr = np.arange(y0, y1, dtype=np.float32)
                dyr -= np.float32(cy)                 # (h,)
                d2 = (dxr * dxr)[None, :] + (dyr * dyr)[:, None]   # (h,w)
                mask = d2 <= sr * sr
                if not mask.any():
                    continue
                # sphere depth: front-surface height above the view plane
                h2 = float(r * r) - d2 * float(inv_zoom * inv_zoom)
                np.maximum(h2, 0.0, out=h2)
                hgt = np.sqrt(h2)             # height above view plane == r * nz
                depth = hgt + np.float32(cz)
                sub_z = zbuf[y0:y1, x0:x1]
                win = mask & (depth > sub_z)
                if not win.any():
                    continue
                # normals as broadcastable row/column/full pieces --
                # shade_write combines them without ever materializing an
                # (h, w, 3) stack or gathering/scattering fragment lists.
                inv_r = 1.0 / r
                nx = (dxr * np.float32(inv_zoom * inv_r))[None, :]
                ny = (dyr * np.float32(-inv_zoom * inv_r))[:, None]
                nz = hgt * np.float32(inv_r)
                shade_write(color[y0:y1, x0:x1], sub_z, win, depth,
                            nx, ny, nz, base_colors[a])

            if has_arrows:
                self._draw_arrow_shafts(geom, zoom, ox_s, oy_s, style, color,
                                        zbuf, shade_write, zmin, zspan, y_lo, y_hi)
                self._draw_arrow_heads(geom, zoom, ox_s, oy_s, style, color,
                                       zbuf, shade, zmin, zspan, y_lo, y_hi)

        bands = self._band_ranges(H, W, len(mol.bonds) + mol.n_atoms)
        if len(bands) == 1:
            draw_band(0, H)
        else:
            futures = [self._threads().submit(draw_band, lo, hi) for lo, hi in bands]
            for f in futures:
                f.result()

        return self._finish(color, zbuf, transparent)

    def _band_ranges(self, H: int, W: int, n_primitives: int):
        """Split the frame into horizontal bands for the render thread pool.

        Threads only pay off when there is real per-band work: small frames
        or near-empty scenes run single-banded (the per-primitive python
        overhead is serialized by the GIL and would dominate).
        """
        if H * W < 250_000 or n_primitives < 8 or _POOL_WORKERS < 2:
            return [(0, H)]
        n = min(_POOL_WORKERS, max(2, H * W // 200_000))
        edges = np.linspace(0, H, n + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(n)
                if edges[i] < edges[i + 1]]

    def _threads(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=_POOL_WORKERS)
        return self._pool

    @staticmethod
    def _finish(color: np.ndarray, zbuf: np.ndarray, transparent: bool) -> np.ndarray:
        # NOTE: the copying np.clip is deliberate -- np.clip(..., out=...) goes
        # through numpy's slow deprecated-casting path (~4x the copying form),
        # and the copy also keeps the renderer's persistent buffer pristine.
        img = np.clip(color, 0.0, 1.0)
        img *= 255.0
        rgb = img.astype(np.uint8)
        if not transparent:
            return rgb
        alpha = (zbuf > -np.inf).astype(np.uint8)
        alpha *= 255
        return np.dstack([rgb, alpha])

    # ------------------------------------------------------------------
    def _draw_arrow_shafts(self, geom: ArrowGeometry, zoom, ox_s, oy_s, style,
                           color, zbuf, shade_write, zmin, zspan, y_lo, y_hi):
        for k in range(geom.shaft_a.shape[0]):
            c = geom.shaft_color[k].astype(np.float32)
            self._draw_cylinder_segment(geom.shaft_a[k], geom.shaft_b[k], float(geom.shaft_radius[k]),
                                        c, c, zoom, ox_s, oy_s, style, color, zbuf, shade_write, zmin, zspan,
                                        y_lo, y_hi)

    def _draw_arrow_heads(self, geom: ArrowGeometry, zoom, ox_s, oy_s, style,
                          color, zbuf, shade, zmin, zspan, y_lo, y_hi):
        for k in range(geom.head_base.shape[0]):
            self._draw_cone_segment(geom.head_base[k], geom.head_apex[k], float(geom.head_radius[k]),
                                    geom.head_color[k], zoom, ox_s, oy_s, style, color, zbuf, shade,
                                    zmin, zspan, y_lo, y_hi)

    # ------------------------------------------------------------------
    def _draw_cylinder_segment(self, a, b, radius, color_a, color_b, zoom, ox_s, oy_s,
                               style, color, zbuf, shade_write, zmin, zspan,
                               y_lo=0, y_hi=None):
        """Rasterize one capped-cylinder impostor from view-space *a* to *b*.

        Shared by real bonds (``_draw_bonds``, two atom colors split at the
        midpoint) and arrow shafts (``_draw_arrow_shafts``, one uniform color).
        """
        W, H = self.width, self.height
        ax, ay, az = (float(v) for v in a)
        bx, by, bz = (float(v) for v in b)
        ux_, uy_, uz = bx - ax, by - ay, bz - az
        L = (ux_ * ux_ + uy_ * uy_ + uz * uz) ** 0.5
        if L < 1e-6 or radius <= 0:
            return
        ux_, uy_, uz = ux_ / L, uy_ / L, uz / L
        radius = float(radius)
        srb = radius * zoom
        sax, say = ox_s + ax * zoom, oy_s - ay * zoom
        sbx, sby = ox_s + bx * zoom, oy_s - by * zoom
        x0 = max(int(np.floor(min(sax, sbx) - srb)), 0)
        x1 = min(int(np.ceil(max(sax, sbx) + srb)) + 1, W)
        y0 = max(int(np.floor(min(say, sby) - srb)), y_lo)
        y1 = min(int(np.ceil(max(say, sby) + srb)) + 1, H if y_hi is None else y_hi)
        if x0 >= x1 or y0 >= y1:
            return
        # pixel -> view-plane coordinates (angstrom), broadcasting a row/column.
        # All full-bbox math in float32 (python-float scalars don't upcast it);
        # per-fragment quantities (normals, albedo) are materialized only for
        # the winners of the z-test further down, never over the whole box.
        ex = np.arange(x0, x1, dtype=np.float32)
        ex -= np.float32(ox_s)
        ex *= np.float32(1.0 / zoom)
        ex -= np.float32(ax)                 # (w,) view-plane x offset from a
        ey = np.arange(y0, y1, dtype=np.float32)
        ey *= np.float32(-1.0 / zoom)
        ey += np.float32(oy_s / zoom - ay)   # (h,) view-plane y offset from a
        exr = ex[None, :]
        eyc = ey[:, None]
        A0 = exr * np.float32(ux_) + eyc * np.float32(uy_)
        a2 = 1.0 - uz * uz
        b2 = A0 * np.float32(-2.0 * uz)
        c2 = exr * exr + eyc * eyc - A0 * A0 - np.float32(radius * radius)
        if abs(a2) < 1e-9:
            # cylinder axis ~ parallel to view direction: treat as disc
            mask = c2 <= 0                   # radius^2 >= ex^2+ey^2-A0^2
            s = np.zeros_like(A0)
        else:
            disc = b2 * b2 - a2 * 4.0 * c2
            mask = disc >= 0
            np.maximum(disc, 0.0, out=disc)
            sq = np.sqrt(disc)
            s = (sq - b2) * np.float32(0.5 / a2)  # front (larger z) root
        if not mask.any():
            return
        zview = s + np.float32(az)
        # axial coordinate to clamp to the segment and pick the color half
        t = A0 + s * np.float32(uz)          # (P-a).u
        mask &= (t >= 0) & (t <= L)
        if not mask.any():
            return
        sub_z = zbuf[y0:y1, x0:x1]
        win = mask & (zview > sub_z)
        if not win.any():
            return
        # full-bbox normals ((w-(w.u)u)/radius, w = P - a) for shade_write's
        # contiguous shade-and-masked-write path (see its docstring).
        inv_r = np.float32(1.0 / radius)
        nx = (exr - t * np.float32(ux_)) * inv_r
        ny = (eyc - t * np.float32(uy_)) * inv_r
        nz = (s - t * np.float32(uz)) * inv_r
        # split color at midpoint
        albedo = np.where((t < np.float32(0.5 * L))[..., None], color_a, color_b)
        shade_write(color[y0:y1, x0:x1], sub_z, win, zview, nx, ny, nz,
                    albedo.astype(np.float32, copy=False))

    # ------------------------------------------------------------------
    def _draw_cone_segment(self, base, apex, radius, color_rgb, zoom, ox_s, oy_s,
                           style, color, zbuf, shade, zmin, zspan,
                           y_lo=0, y_hi=None):
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
        y0 = max(int(np.floor(min(say, sby) - srb)), y_lo)
        y1 = min(int(np.ceil(max(say, sby) + srb)) + 1, H if y_hi is None else y_hi)
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
