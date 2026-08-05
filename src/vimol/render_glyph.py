"""Software rasterizers for the glyph skin's two new primitives.

Both follow the same contract as ``render.Renderer._draw_cylinder_segment``:
they take view-space geometry, the shared color/z buffers, the band's row
range, and the ``shade_write`` closure that owns lighting, fog and
quantization -- so a glyph shades identically to a sphere or a bond and needs
no lighting code of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .glyph_font import ASPECT, STROKE, segments

_EPS = 1e-9

# How much of the usual specular highlight a flat face keeps. A whole planar
# facet crosses the highlight at once and flares to white together, so a
# ribbon reads as a row of mirrors rather than one matte band.
MATTE = 0.18


def draw_polyhedron(center_view, normals_view, offsets, half_px, albedo,
                    zoom, ox_s, oy_s, xs, ys, width, height,
                    color, alpha, zbuf, shade_write, y_lo, y_hi,
                    flat=False, smooth=None) -> None:
    """Rasterize one convex solid given as half-spaces ``n·(p − c) ≤ d``.

    The camera is orthographic, so every ray is axis-aligned in view space and
    the intersection collapses to per-pixel arithmetic over the solid's
    bounding box. At a pixel whose view-space position is ``(X, Y)``, each
    plane constrains the depth ``Z``: a plane facing the viewer (``n_z > 0``)
    puts a ceiling on it, one facing away puts a floor under it, and one
    edge-on is a flat 2-D mask. The visible surface is the lowest ceiling --
    the nearest point of the solid, since larger ``Z`` is nearer here -- and it
    is real wherever it clears every floor and every mask. The plane that
    supplied that ceiling is the surface normal, so shading comes out of the
    same pass with no extra work.

    ``smooth`` is ``(face_index, tangent_view, half_along, n_start, n_end)``:
    on the two faces beginning at *face_index* the constant plane normal is
    replaced by a normal interpolated along the tangent between the two joint
    normals. That is vertex-normal averaging -- the geometry stays faceted, but
    shading runs continuously across the joins, which is what turns a chain of
    ribbon segments into one curved band instead of a row of flats each
    catching the light in a step.
    """
    cx, cy, cz = (float(v) for v in center_view)
    scx = ox_s + cx * zoom
    scy = oy_s - cy * zoom
    hx, hy = float(half_px[0]), float(half_px[1])
    x0 = max(int(np.floor(scx - hx)), 0)
    x1 = min(int(np.ceil(scx + hx)) + 1, width)
    y0 = max(int(np.floor(scy - hy)), y_lo)
    y1 = min(int(np.ceil(scy + hy)) + 1, y_hi)
    if x0 >= x1 or y0 >= y1:
        return

    inv_zoom = np.float32(1.0 / zoom)
    dx = (xs[x0:x1] - np.float32(scx)) * inv_zoom          # (w,) view X - center X
    dy = (np.float32(scy) - ys[y0:y1]) * inv_zoom          # (h,) view Y - center Y
    h, w = y1 - y0, x1 - x0

    ceiling = np.full((h, w), np.inf, np.float32)
    which = np.zeros((h, w), np.int32)
    floor = np.full((h, w), -np.inf, np.float32)
    inside = None

    normals = np.asarray(normals_view, np.float32)
    offsets = np.asarray(offsets, np.float32)
    for k in range(len(normals)):
        nx, ny, nz = (float(v) for v in normals[k])
        # d - n_x*(X - cx) - n_y*(Y - cy), broadcast from the row/column pieces
        rhs = np.float32(offsets[k]) - (dx * np.float32(nx))[None, :] \
            - (dy * np.float32(ny))[:, None]
        if nz > _EPS:
            t = rhs * np.float32(1.0 / nz)
            closer = t < ceiling
            np.copyto(which, k, where=closer)
            np.copyto(ceiling, t, where=closer)
        elif nz < -_EPS:
            np.maximum(floor, rhs * np.float32(1.0 / nz), out=floor)
        else:
            mask = rhs >= 0.0
            inside = mask if inside is None else (inside & mask)

    win = np.isfinite(ceiling) & (ceiling >= floor)
    if inside is not None:
        win &= inside
    if not win.any():
        return

    depth = ceiling + np.float32(cz)
    sub_z = zbuf[y0:y1, x0:x1]
    win &= depth > sub_z
    if not win.any():
        return

    face = normals[which]                                   # (h, w, 3)
    if smooth is not None:
        first, tangent, up, half_along, n_start, n_end = smooth
        # Everything but the two thin side faces: the ribbon's flat top and
        # bottom, and its end caps. The caps matter as much as the faces --
        # they are internal, one segment's cap sits against the next one's, and
        # left with their own edge-on normals they draw a hairline dark crease
        # across the ribbon at every joint.
        on_face = which >= first
        if on_face.any():
            # Where along the segment each hit landed, as 0..1. The ceiling is
            # +inf outside the solid, and inf * a zero tangent component is a
            # nan the shading would carry into a warning, so flatten it first.
            dz = np.where(np.isfinite(ceiling), ceiling, np.float32(0.0))

            def project(axis):
                return ((dx * np.float32(axis[0]))[None, :]
                        + (dy * np.float32(axis[1]))[:, None]
                        + dz * np.float32(axis[2]))

            s = project(tangent)
            s *= np.float32(0.5 / max(float(half_along), 1e-9))
            s += np.float32(0.5)
            np.clip(s, 0.0, 1.0, out=s)
            # Adjacent joint normals are a few degrees apart, so the lerp is
            # within a fraction of a percent of unit length -- not worth a
            # square root per pixel to renormalize.
            blend = (np.asarray(n_start, np.float32)
                     + s[..., None] * np.asarray(n_end - n_start, np.float32))
            # Which face of the ribbon the hit is on, from which side of the
            # slab it landed. Stable where a normal-vs-normal test is not:
            # a cap is perpendicular to the surface it has to blend into.
            blend = np.where((project(up) >= 0.0)[..., None], blend, -blend)
            face = np.where(on_face[..., None], blend, face)

    shade_write(color[y0:y1, x0:x1],
                alpha[y0:y1, x0:x1] if alpha is not None else None,
                sub_z, win, depth,
                face[..., 0], face[..., 1], face[..., 2],
                np.asarray(albedo, np.float32), flat=flat, specular=MATTE)


# Below this cap height in pixels a stroked capital collapses into a blob, so
# the glyph is dropped rather than smeared across three pixels.
MIN_GLYPH_PX = 8.0


def draw_label(center_view, char: str, size, bias, albedo,
               zoom, ox_s, oy_s, xs, ys, width, height,
               color, alpha, zbuf, shade_write, y_lo, y_hi) -> None:
    """Stamp one screen-aligned bitmap letter at a view-space anchor.

    The letter is a billboard, not a decal painted onto the solid it names: it
    stays square to the viewer at every camera angle, which is what makes a
    labelled diagram readable while it spins. ``bias`` lifts it toward the
    camera by the radius of its own solid, so a letter is never swallowed by
    the shape it belongs to but is still occluded by anything genuinely in
    front of both.
    """
    cx, cy, cz = (float(v) for v in center_view)
    gh = float(size) * zoom
    if gh < MIN_GLYPH_PX:
        return
    gw = gh * ASPECT
    pad = STROKE * gh                       # the stroke straddles the outline
    left = ox_s + cx * zoom - gw * 0.5
    top = oy_s - cy * zoom - gh * 0.5

    x0 = max(int(np.floor(left - pad)), 0)
    x1 = min(int(np.ceil(left + gw + pad)) + 1, width)
    y0 = max(int(np.floor(top - pad)), y_lo)
    y1 = min(int(np.ceil(top + gh + pad)) + 1, y_hi)
    if x0 >= x1 or y0 >= y1:
        return

    # Both axes in cap heights, so one stroke half-width covers every
    # direction; the glyph's own x coordinates are already scaled to match.
    inv = np.float32(1.0 / gh)
    u = ((xs[x0:x1] - np.float32(left)) * inv)[None, :]
    v = ((ys[y0:y1] - np.float32(top)) * inv)[:, None]

    win = np.zeros((y1 - y0, x1 - x0), bool)
    limit = np.float32(STROKE * STROKE)
    for (ax, ay), (bx, by) in segments(char):
        dx, dy = float(bx - ax), float(by - ay)
        span = dx * dx + dy * dy
        du, dv = u - np.float32(ax), v - np.float32(ay)
        if span > 1e-12:
            # Nearest point on the segment, clamped to its ends: round caps,
            # which is what keeps a polyline's corners from splintering.
            t = (du * np.float32(dx / span)) + (dv * np.float32(dy / span))
            np.clip(t, 0.0, 1.0, out=t)
            du = du - t * np.float32(dx)
            dv = dv - t * np.float32(dy)
        win |= (du * du + dv * dv) <= limit
    if not win.any():
        return

    sub_z = zbuf[y0:y1, x0:x1]
    depth = np.full(win.shape, np.float32(cz + float(bias)), np.float32)
    win &= depth > sub_z
    if not win.any():
        return
    # flat: ink, not a lit surface -- it still fogs with distance so a letter
    # at the back of the structure recedes with everything around it.
    shade_write(color[y0:y1, x0:x1],
                alpha[y0:y1, x0:x1] if alpha is not None else None,
                sub_z, win, depth, None, None, None,
                np.asarray(albedo, np.float32), flat=True)


@dataclass
class Prepared:
    """A glyph scene transformed into view space once for the whole frame.

    Bands re-enter the draw path several times per frame, so the camera
    transform and the plane rotation happen here rather than per band -- and
    each primitive's screen y-interval comes along, so a band only touches
    what actually reaches its rows (the same trick the atom and bond loops
    use).
    """
    poly_center: np.ndarray
    poly_normal: np.ndarray
    poly_offset: np.ndarray
    poly_slice: np.ndarray
    poly_half_px: np.ndarray       # (P, 2) screen half-width/half-height
    poly_color: np.ndarray
    poly_flat: np.ndarray
    poly_tangent: np.ndarray       # (P, 3) view-space long axis, for smooth shading
    poly_up: np.ndarray            # (P, 3) view-space face normal axis
    poly_along: np.ndarray         # (P,) half-extent along that axis
    poly_smooth: np.ndarray        # (P, 2, 3) view-space joint normals
    poly_smooth_face: np.ndarray   # (P,) first smooth-shaded plane, or -1
    poly_rows: np.ndarray          # (P, 2) screen y-interval
    cyl_a: np.ndarray
    cyl_b: np.ndarray
    cyl_radius: np.ndarray
    cyl_color: np.ndarray
    cyl_color_b: np.ndarray
    cyl_flat: np.ndarray
    cyl_rows: np.ndarray
    sphere_center: np.ndarray
    sphere_radius: np.ndarray
    sphere_color: np.ndarray
    sphere_flat: np.ndarray
    sphere_order: np.ndarray
    sphere_rows: np.ndarray
    label_center: np.ndarray
    label_size: np.ndarray
    label_bias: np.ndarray
    label_color: np.ndarray
    label_flat: np.ndarray
    label_char: List[str]
    label_rows: np.ndarray


def _rows(centers_y, half_px) -> np.ndarray:
    """Screen y-interval of each primitive, from view-space y already turned
    into screen y by the caller."""
    if not len(centers_y):
        return np.zeros((0, 2))
    return np.column_stack([centers_y - half_px, centers_y + half_px])


def prepare(scene, camera, zoom, oy_s) -> Prepared:
    view = camera.view_positions
    poly_c = view(scene.poly_center) if len(scene.poly_center) else np.zeros((0, 3))
    # One matmul rotates every plane in the scene: offsets are measured from
    # each solid's own center, so a rotation leaves them alone.
    poly_n = (scene.plane_normal @ camera.rotation.T if len(scene.plane_normal)
              else np.zeros((0, 3)))
    cyl_a = view(scene.cyl_a) if len(scene.cyl_a) else np.zeros((0, 3))
    cyl_b = view(scene.cyl_b) if len(scene.cyl_b) else np.zeros((0, 3))
    sph_c = view(scene.sphere_center) if len(scene.sphere_center) else np.zeros((0, 3))
    lab_c = view(scene.label_center) if len(scene.label_center) else np.zeros((0, 3))

    # Screen rectangle of each solid's enclosing box: an axis of the box
    # contributes its half-extent times how much of that axis the screen axis
    # sees. Exact for a box, and for a ribbon segment -- a thin slab -- several
    # times smaller than the square a bounding sphere would ask to be shaded.
    poly_half_px = np.zeros((0, 2))
    poly_tangent = np.zeros((0, 3))
    poly_up = np.zeros((0, 3))
    poly_smooth = np.zeros((0, 2, 3))
    if len(poly_c):
        axes_view = scene.poly_axes @ camera.rotation.T          # (P, 3, 3)
        poly_half_px = np.einsum("pij,pi->pj", np.abs(axes_view[:, :, :2]),
                                 scene.poly_half) * zoom
        poly_tangent = axes_view[:, 0, :]
        poly_up = axes_view[:, 2, :]
        poly_smooth = scene.poly_smooth @ camera.rotation.T

    cyl_rows = np.zeros((0, 2))
    if len(cyl_a):
        ay, by = oy_s - cyl_a[:, 1] * zoom, oy_s - cyl_b[:, 1] * zoom
        pad = scene.cyl_radius * zoom
        cyl_rows = np.column_stack([np.minimum(ay, by) - pad,
                                    np.maximum(ay, by) + pad])

    return Prepared(
        poly_center=poly_c, poly_normal=poly_n, poly_offset=scene.plane_offset,
        poly_slice=scene.poly_slice, poly_half_px=poly_half_px,
        poly_color=scene.poly_color, poly_flat=scene.poly_flat,
        poly_tangent=poly_tangent, poly_up=poly_up,
        poly_along=scene.poly_smooth_span,
        poly_smooth=poly_smooth, poly_smooth_face=scene.poly_smooth_face,
        poly_rows=_rows(oy_s - poly_c[:, 1] * zoom,
                        poly_half_px[:, 1] if len(poly_half_px) else np.zeros(0)),
        cyl_a=cyl_a, cyl_b=cyl_b, cyl_radius=scene.cyl_radius,
        cyl_color=scene.cyl_color, cyl_color_b=scene.cyl_color_b,
        cyl_flat=scene.cyl_flat, cyl_rows=cyl_rows,
        sphere_center=sph_c, sphere_radius=scene.sphere_radius,
        sphere_color=scene.sphere_color, sphere_flat=scene.sphere_flat,
        # far to near, like the atom loop, so specular highlights layer sanely
        sphere_order=np.argsort(sph_c[:, 2]) if len(sph_c) else np.zeros(0, np.int64),
        sphere_rows=_rows(oy_s - sph_c[:, 1] * zoom, scene.sphere_radius * zoom),
        label_center=lab_c, label_size=scene.label_size, label_bias=scene.label_bias,
        label_color=scene.label_color, label_flat=scene.label_flat,
        label_char=scene.label_char,
        label_rows=_rows(oy_s - lab_c[:, 1] * zoom, scene.label_size * zoom * 0.5),
    )


def _in_band(rows: np.ndarray, y_lo: int, y_hi: int) -> np.ndarray:
    if not len(rows):
        return np.zeros(0, np.int64)
    return np.nonzero((rows[:, 1] >= y_lo) & (rows[:, 0] < y_hi))[0]


def draw_band(prep: Prepared, zoom, ox_s, oy_s, xs, ys, width, height,
              color, alpha, zbuf, shade_write, draw_sphere, draw_cylinder,
              y_lo, y_hi) -> None:
    """Draw a prepared glyph scene into rows ``[y_lo, y_hi)``.

    Order matters only for cost, not correctness -- the z-buffer decides every
    pixel -- but solids first, then sticks and nodes, then letters keeps the
    letters' write masks small.
    """
    bounds = prep.poly_slice
    for p in _in_band(prep.poly_rows, y_lo, y_hi):
        lo, hi = int(bounds[p]), int(bounds[p + 1])
        face = int(prep.poly_smooth_face[p])
        smooth = None if face < 0 else (
            face, prep.poly_tangent[p], prep.poly_up[p], float(prep.poly_along[p]),
            prep.poly_smooth[p, 0], prep.poly_smooth[p, 1])
        draw_polyhedron(prep.poly_center[p], prep.poly_normal[lo:hi],
                        prep.poly_offset[lo:hi], prep.poly_half_px[p],
                        prep.poly_color[p], zoom, ox_s, oy_s, xs, ys,
                        width, height, color, alpha, zbuf, shade_write, y_lo, y_hi,
                        flat=bool(prep.poly_flat[p]), smooth=smooth)

    # The cylinder rasterizer shades its whole bounding box and writes only the
    # pixels that pass coverage and depth. Outside the cylinder its "normal"
    # grows as the segment length over the radius, and the sticks and the
    # hydrogen-bond hairlines are the thinnest cylinders vimol draws -- so
    # raising that ratio to the shininess exponent overflows float32, in
    # numbers discarded a line later but noisily enough to spam a terminal.
    with np.errstate(over="ignore"):
        for k in _in_band(prep.cyl_rows, y_lo, y_hi):
            draw_cylinder(prep.cyl_a[k], prep.cyl_b[k], float(prep.cyl_radius[k]),
                          prep.cyl_color[k].astype(np.float32),
                          prep.cyl_color_b[k].astype(np.float32),
                          y_lo, y_hi, bool(prep.cyl_flat[k]))

    if len(prep.sphere_center):
        hit = np.zeros(len(prep.sphere_center), bool)
        hit[_in_band(prep.sphere_rows, y_lo, y_hi)] = True
        for k in prep.sphere_order[hit[prep.sphere_order]]:
            draw_sphere(prep.sphere_center[k], float(prep.sphere_radius[k]),
                        prep.sphere_color[k].astype(np.float32), y_lo, y_hi,
                        bool(prep.sphere_flat[k]))

    for k in _in_band(prep.label_rows, y_lo, y_hi):
        draw_label(prep.label_center[k], prep.label_char[k],
                   float(prep.label_size[k]), float(prep.label_bias[k]),
                   prep.label_color[k], zoom, ox_s, oy_s, xs, ys, width, height,
                   color, alpha, zbuf, shade_write, y_lo, y_hi)
