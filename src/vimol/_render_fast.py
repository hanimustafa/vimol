"""Numba-compiled inner loop for the CPU raycaster.

The numpy renderer (``render.py``) vectorizes per primitive over its
screen-space bounding box: every sphere/cylinder costs ~40 full-box array
passes, most of them over pixels that fail the z-test, plus thousands of
per-call overheads per frame. That is a structural ceiling -- measured
~130 ms/frame for C60 at 2560x1440.

This module compiles the same impostor math to machine code (per-pixel loops
with early z-rejection *before* shading, the way PyMOL's C++ ray tracer and
every other compiled rasterizer do it), which measures 10-25x faster. numba
is a hard dependency (see pyproject.toml), but this module still degrades
gracefully if it's ever missing or fails to compile on some platform:
``ready()`` stays False and ``render.py``'s numpy path runs unchanged.

BIT-PARITY CONTRACT: the kernels below mirror render.py's numpy operations
one-for-one -- same dtypes (float32 per-pixel math, float64 screen setup),
same operation order and association, same scalar casts (numpy converts a
python-float scalar to float32 when it meets a float32 array; typed numpy
float64 scalars promote instead -- both cases are reproduced explicitly).
``tests/test_render_fast.py`` verifies the output is bit-identical to the
numpy path -- run it after touching either side of the contract.
"""
from __future__ import annotations

import os
import threading

import numpy as np

# numba's disk cache (cache=True below) defaults to __pycache__ next to this
# source file -- churn a sync client (Dropbox) then uploads on every
# recompile. Redirect it to the user cache dir before numba is first
# imported (the env var is read at numba import; if some other package
# imported numba already, this is a harmless no-op and the default location
# applies). setdefault, so an explicit NUMBA_CACHE_DIR always wins.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "vimol", "numba"))

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:                     # hard dependency, but degrade gracefully
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):        # stub so module import never fails
        def wrap(fn):
            return fn
        return wrap if not (args and callable(args[0])) else args[0]

    def prange(*args):
        return range(*args)


# 0 = cold, 1 = compiling in background, 2 = ready, 3 = unavailable/failed
_state = 0
_lock = threading.Lock()


def available() -> bool:
    """numba importable on this machine (kernel may still be compiling)."""
    return _HAVE_NUMBA


def ready() -> bool:
    """Kernel compiled and safe to call (first call triggers compilation)."""
    return _state == 2


def warm_async() -> None:
    """Compile the kernel in a daemon thread so the first frames don't stall.

    JIT compilation takes ~1-3 s once per process (cached on disk afterwards
    where the package dir is writable). Frames rendered while this runs use
    the numpy path; the renderer switches over the moment ready() flips.
    """
    global _state
    if not _HAVE_NUMBA:
        return
    with _lock:
        if _state != 0:
            return
        _state = 1
    threading.Thread(target=_warm, daemon=True).start()


def _warm() -> None:
    global _state
    try:
        # (8, 8, 4) rgba: (H, W, 3) and (H, W, 4) C-contiguous buffers share
        # one numba type (uint8 3d C), so this single call compiles the only
        # signature both opaque and transparent frames use.
        color = np.zeros((8, 8, 4), np.uint8)
        zbuf = np.full((8, 8), -np.inf, np.float32)
        bands = np.array([[0, 8]], np.int64)
        sx = np.array([4.0], np.float64)
        sy = np.array([4.0], np.float64)
        sz = np.array([0.0], np.float64)
        radii = np.array([0.5], np.float64)
        acol = np.array([[0.5, 0.5, 0.5]], np.float32)
        aflat = np.zeros(1, np.uint8)
        order = np.array([0], np.int64)
        barr = np.zeros((0, 14), np.float64)
        bcol = np.zeros((0, 6), np.float32)
        bflat = np.zeros(0, np.uint8)
        render_frame(color, zbuf, True, bands,
                     sx, sy, sz, radii, acol, aflat, order,
                     barr, bcol, bflat,
                     10.0, 4.0, 4.0,
                     0.3, 0.5, 0.7, -0.3, -0.5, 0.6, 0.4, 0.5, 0.7,
                     0.28, 0.25, 0.55, 0.55, -1.0, 2.0, 5)
        _state = 2
    except Exception:
        _state = 3


# ---------------------------------------------------------------------------
# compiled kernels
# ---------------------------------------------------------------------------
@njit(inline="always")
def _shade_pixel(color, y, x, alpha_on, depth, nx, ny, nz,
                 ar, ag, ab, flat,
                 l0, l1, l2, f0, f1, f2, hv0, hv1, hv2,
                 ambient, fill_w, spec_w, depth_cue, zmin, zspan, n_squares):
    """Shade one winning fragment and write color/alpha/depth.

    ``color`` is (H, W, 3) opaque or (H, W, 4) rgba; ``alpha_on`` means the
    4th channel exists and coverage is written there. Mirrors
    render.shade_write's per-element op sequence (see module docstring).
    nx/ny/nz/ar..ab/depth are float32; the light scalars are float64
    python-side values that numpy would have cast to float32 at each use --
    the casts here make that explicit.
    """
    if flat:
        if depth_cue > 0.0:
            fog = (depth - np.float32(zmin)) * np.float32(1.0 / zspan)
            if fog < np.float32(0.0):
                fog = np.float32(0.0)
            if fog > np.float32(1.0):
                fog = np.float32(1.0)
            fog *= np.float32(depth_cue)
            fog += np.float32(1.0 - depth_cue)
            rr = ar * fog
            rr *= np.float32(255.0)
            gg = ag * fog
            gg *= np.float32(255.0)
            bb = ab * fog
            bb *= np.float32(255.0)
        else:
            rr = min(ar, np.float32(1.0)) * np.float32(255.0)
            gg = min(ag, np.float32(1.0)) * np.float32(255.0)
            bb = min(ab, np.float32(1.0)) * np.float32(255.0)
    else:
        diff = nx * np.float32(l0) + ny * np.float32(l1) + nz * np.float32(l2)
        if diff < np.float32(0.0):
            diff = np.float32(0.0)
        dfill = nx * np.float32(f0) + ny * np.float32(f1) + nz * np.float32(f2)
        if dfill < np.float32(0.0):
            dfill = np.float32(0.0)
        dh = nx * np.float32(hv0) + ny * np.float32(hv1) + nz * np.float32(hv2)
        if dh < np.float32(0.0):
            dh = np.float32(0.0)
        # _fast_pow for a power-of-two exponent: x*x then repeated squaring,
        # identical rounding sequence to the numpy version.
        spec = dh
        for _ in range(n_squares):
            spec *= spec
        spec *= np.float32(spec_w)
        diff += np.float32(ambient)
        dfill *= np.float32(fill_w)
        diff += dfill
        if depth_cue > 0.0:
            fog = (depth - np.float32(zmin)) * np.float32(1.0 / zspan)
            if fog < np.float32(0.0):
                fog = np.float32(0.0)
            if fog > np.float32(1.0):
                fog = np.float32(1.0)
            fog *= np.float32(depth_cue)
            fog += np.float32(1.0 - depth_cue)
            diff *= fog
            spec *= fog
        rr = ar * diff
        rr += spec
        gg = ag * diff
        gg += spec
        bb = ab * diff
        bb += spec
        if rr > np.float32(1.0):
            rr = np.float32(1.0)
        if gg > np.float32(1.0):
            gg = np.float32(1.0)
        if bb > np.float32(1.0):
            bb = np.float32(1.0)
        rr *= np.float32(255.0)
        gg *= np.float32(255.0)
        bb *= np.float32(255.0)
    # np.copyto(..., casting="unsafe") / setitem float->uint8 truncates.
    color[y, x, 0] = np.uint8(rr)
    color[y, x, 1] = np.uint8(gg)
    color[y, x, 2] = np.uint8(bb)
    if alpha_on:
        color[y, x, 3] = 255
    return


@njit(parallel=True, cache=True)
def _render_kernel(color, zbuf, alpha_on, bands,
                   sx, sy, sz, radii, acol, aflat, order,
                   barr, bcol, bflat,
                   zoom, ox_s, oy_s,
                   l0, l1, l2, f0, f1, f2, hv0, hv1, hv2,
                   ambient, fill_w, spec_w, depth_cue, zmin, zspan, n_squares):
    """Raycast all bonds (list order) then all atoms (far-to-near order)
    into their bands. Bands own disjoint rows, so prange is race-free; the
    z-test makes per-pixel results order-independent up to the same ties the
    numpy path resolves identically (same primitive order per band)."""
    H, W = zbuf.shape
    inv_zoom = 1.0 / zoom
    iz32 = np.float32(1.0 / zoom)
    niz32 = np.float32(-1.0 / zoom)
    izz32 = np.float32(inv_zoom * inv_zoom)
    n_bonds = barr.shape[0]

    for band in prange(bands.shape[0]):
        y_lo = bands[band, 0]
        y_hi = bands[band, 1]

        # -- bonds: capped-cylinder impostors (render._draw_cylinder_segment)
        for k in range(n_bonds):
            ax = barr[k, 0]
            ay = barr[k, 1]
            az = barr[k, 2]
            ux = barr[k, 3]
            uy = barr[k, 4]
            uz = barr[k, 5]
            L = barr[k, 6]
            a2 = barr[k, 7]
            radius = barr[k, 8]
            sax = barr[k, 9]
            say = barr[k, 10]
            sbx = barr[k, 11]
            sby = barr[k, 12]
            srb = barr[k, 13]
            if L < 1e-6 or radius <= 0.0:
                continue
            x0 = max(int(np.floor(min(sax, sbx) - srb)), 0)
            x1 = min(int(np.ceil(max(sax, sbx) + srb)) + 1, W)
            y0 = max(int(np.floor(min(say, sby) - srb)), y_lo)
            y1 = min(int(np.ceil(max(say, sby) + srb)) + 1, y_hi)
            if x0 >= x1 or y0 >= y1:
                continue
            ux32 = np.float32(ux)
            uy32 = np.float32(uy)
            uz32 = np.float32(uz)
            b2c32 = np.float32(-2.0 * uz)
            a2x4_32 = np.float32(a2 * 4.0)
            half_a2_32 = np.float32(0.5 / a2) if abs(a2) >= 1e-9 else np.float32(0.0)
            parallel = abs(a2) < 1e-9
            inv_r32 = np.float32(1.0 / radius)
            r2_32 = np.float32(radius * radius)
            half_L32 = np.float32(0.5 * L)
            L32 = np.float32(L)
            az32 = np.float32(az)
            ey_off32 = np.float32(oy_s / zoom - ay)
            ox32 = np.float32(ox_s)
            ax32 = np.float32(ax)
            flat = bflat[k]
            car, cag, cab = bcol[k, 0], bcol[k, 1], bcol[k, 2]
            cbr, cbg, cbb = bcol[k, 3], bcol[k, 4], bcol[k, 5]
            for y in range(y0, y1):
                ey = np.float32(y) * niz32 + ey_off32
                eyey = ey * ey
                eyuy = ey * uy32
                for x in range(x0, x1):
                    ex = (np.float32(x) - ox32) * iz32 - ax32
                    A0 = ex * ux32 + eyuy
                    c2 = ex * ex + eyey - A0 * A0 - r2_32
                    if parallel:
                        if c2 > np.float32(0.0):
                            continue
                        s = np.float32(0.0)
                    else:
                        b2 = A0 * b2c32
                        disc = b2 * b2 - a2x4_32 * c2
                        if disc < np.float32(0.0):
                            continue
                        sq = np.sqrt(disc)
                        s = (sq - b2) * half_a2_32
                    zview = s + az32
                    t = A0 + s * uz32
                    if t < np.float32(0.0) or t > L32:
                        continue
                    if zview <= zbuf[y, x]:
                        continue
                    nx = (ex - t * ux32) * inv_r32
                    ny = (ey - t * uy32) * inv_r32
                    nz = (s - t * uz32) * inv_r32
                    if t < half_L32:
                        ar, ag, ab = car, cag, cab
                    else:
                        ar, ag, ab = cbr, cbg, cbb
                    zbuf[y, x] = zview
                    _shade_pixel(color, y, x, alpha_on, zview, nx, ny, nz,
                                 ar, ag, ab, flat,
                                 l0, l1, l2, f0, f1, f2, hv0, hv1, hv2,
                                 ambient, fill_w, spec_w, depth_cue, zmin, zspan,
                                 n_squares)

        # -- atoms: sphere impostors (render.draw_band's atom loop)
        for oi in range(order.shape[0]):
            a = order[oi]
            r = radii[a]
            if r <= 0.0:
                continue
            sr = r * zoom
            if sr < 0.5:
                sr = 0.5
            cx = sx[a]
            cy = sy[a]
            cz = sz[a]
            x0 = max(int(np.floor(cx - sr)), 0)
            x1 = min(int(np.ceil(cx + sr)) + 1, W)
            y0 = max(int(np.floor(cy - sr)), y_lo)
            y1 = min(int(np.ceil(cy + sr)) + 1, y_hi)
            if x0 >= x1 or y0 >= y1:
                continue
            cx32 = np.float32(cx)
            cy32 = np.float32(cy)
            cz32 = np.float32(cz)
            sr2 = sr * sr                       # float64: d2 <= sr2 promotes
            rr32 = np.float32(r * r)
            inv_r = 1.0 / r
            nxy32 = np.float32(inv_zoom * inv_r)
            nz32 = np.float32(inv_r)
            flat = aflat[a]
            ar = acol[a, 0]
            ag = acol[a, 1]
            ab = acol[a, 2]
            for y in range(y0, y1):
                dyr = np.float32(y) - cy32
                dyr2 = dyr * dyr
                ny = dyr * np.float32(-inv_zoom * inv_r)
                for x in range(x0, x1):
                    dxr = np.float32(x) - cx32
                    d2 = dxr * dxr + dyr2
                    # numpy compares float32 d2 against the float64 scalar
                    # sr*sr (typed numpy scalar promotes); mirror exactly.
                    if not (d2 <= sr2):
                        continue
                    h2 = rr32 - d2 * izz32
                    if h2 < np.float32(0.0):
                        h2 = np.float32(0.0)
                    hgt = np.sqrt(h2)
                    depth = hgt + cz32
                    if depth <= zbuf[y, x]:
                        continue
                    nx = dxr * nxy32
                    nzv = hgt * nz32
                    zbuf[y, x] = depth
                    _shade_pixel(color, y, x, alpha_on, depth,
                                 nx, ny, nzv, ar, ag, ab, flat,
                                 l0, l1, l2, f0, f1, f2, hv0, hv1, hv2,
                                 ambient, fill_w, spec_w, depth_cue, zmin, zspan,
                                 n_squares)


def render_frame(color, zbuf, alpha_on, bands,
                 sx, sy, sz, radii, acol, aflat, order,
                 barr, bcol, bflat,
                 zoom, ox_s, oy_s,
                 l0, l1, l2, f0, f1, f2, hv0, hv1, hv2,
                 ambient, fill_w, spec_w, depth_cue, zmin, zspan, n_squares):
    """Thin wrapper so _warm and render.py share one call signature."""
    _render_kernel(color, zbuf, alpha_on, bands,
                   sx, sy, sz, radii, acol, aflat, order,
                   barr, bcol, bflat,
                   zoom, ox_s, oy_s,
                   l0, l1, l2, f0, f1, f2, hv0, hv1, hv2,
                   ambient, fill_w, spec_w, depth_cue, zmin, zspan, n_squares)
