"""Generic GPU sphere/cylinder impostor renderer (OpenGL, via moderngl).

This module knows nothing about molecules, atoms, bonds, or vimol's
``Camera``/``Style`` types. It renders batches of analytic spheres and
cylinders — the same "impostor" technique as the CPU raycaster in
``render.py`` (each primitive is a screen-aligned billboard whose fragment
shader solves the ray/surface intersection analytically), just executed on
the GPU with a real depth buffer instead of a manually-maintained z-buffer
array. Callers supply a raw 4x4 projection matrix and plain shading
parameters; nothing here is vimol-specific, so it's reusable outside this
project. See ``gl_adapter.py`` for the vimol-specific glue that turns a
``Molecule`` + ``Camera`` + ``Style`` into the inputs this module expects.

Requires ``moderngl`` (installed with vimol; a GL context must be
creatable at runtime, else Scene's auto backend falls back to the CPU).

Readback orientation: ``render()`` returns the framebuffer in raw
``glReadPixels`` order — row 0 is the NDC -y side of the image — as a
zero-copy read-only view of the readback bytes. Projections built by
``gl_adapter._build_projection`` have their Y row negated precisely so that
this order is already top-down in vimol's screen convention; a textbook
Y-up projection will come out vertically flipped. (The previous tail —
flipud plus the RGBA->RGB slicing copy, the slice being the dominant
cost — measured ~23 ms/frame at 2560x1440, more than the draw and the
readback combined.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

import moderngl


@dataclass
class SphereBatch:
    """A batch of analytic spheres, structure-of-arrays style.

    ``centers``/``radii``/``colors`` must all be in the same coordinate
    space as the ``proj`` matrix passed to :meth:`GLRenderer.render`
    (typically view space, with the convention that larger Z is nearer the
    viewer — see ``gl_adapter.py``).
    """
    centers: np.ndarray  # (N, 3) float
    radii: np.ndarray    # (N,) float
    colors: np.ndarray   # (N, 3) float, 0..1
    flat: np.ndarray = None   # (N,) float, 0/1 -- 1 skips shading (design §4.4/§4.5)

    def __post_init__(self) -> None:
        if self.flat is None:
            self.flat = np.zeros(len(self.radii), np.float32)

    @staticmethod
    def empty() -> "SphereBatch":
        return SphereBatch(
            centers=np.zeros((0, 3), np.float32),
            radii=np.zeros((0,), np.float32),
            colors=np.zeros((0, 3), np.float32),
            flat=np.zeros((0,), np.float32),
        )


@dataclass
class CylinderBatch:
    """A batch of analytic cylinders (capped, but caps aren't drawn — bonds
    are always occluded/occluding via the shared depth buffer instead)."""
    a: np.ndarray         # (M, 3) float — one endpoint
    b: np.ndarray         # (M, 3) float — other endpoint
    radii: np.ndarray     # (M,) float
    colors_a: np.ndarray  # (M, 3) float — color at the `a` end
    colors_b: np.ndarray  # (M, 3) float — color at the `b` end
    flat: np.ndarray = None    # (M,) float, 0/1 -- 1 skips shading (design §4.4/§4.5)

    def __post_init__(self) -> None:
        if self.flat is None:
            self.flat = np.zeros(len(self.radii), np.float32)

    @staticmethod
    def empty() -> "CylinderBatch":
        return CylinderBatch(
            a=np.zeros((0, 3), np.float32),
            b=np.zeros((0, 3), np.float32),
            radii=np.zeros((0,), np.float32),
            colors_a=np.zeros((0, 3), np.float32),
            colors_b=np.zeros((0, 3), np.float32),
            flat=np.zeros((0,), np.float32),
        )


@dataclass
class ConeBatch:
    """A batch of analytic cones (arrow heads): radius *radius* at ``base``,
    tapering linearly to a point at ``apex``. Used for the arrowhead half of
    a vector-field arrow; the shaft half is a plain ``CylinderBatch`` entry."""
    base: np.ndarray    # (K, 3) float
    apex: np.ndarray    # (K, 3) float
    radius: np.ndarray  # (K,) float
    color: np.ndarray   # (K, 3) float

    @staticmethod
    def empty() -> "ConeBatch":
        return ConeBatch(
            base=np.zeros((0, 3), np.float32),
            apex=np.zeros((0, 3), np.float32),
            radius=np.zeros((0,), np.float32),
            color=np.zeros((0, 3), np.float32),
        )


@dataclass
class MeshBatch:
    """An indexed triangle mesh in view space.

    The impostor batches cover everything that is a sphere, a cylinder or a
    cone analytically, which is cheaper and more exact than tessellating them.
    This is for the shapes that are none of those -- the glyph skin's ribbon and
    its lettered tablets -- where real geometry with interpolated normals is
    what makes a curved surface read as curved instead of as a run of facets.

    ``uv`` addresses the glyph atlas; a vertex with a negative ``u`` samples
    nothing and shades as plain surface. Letters ride on the same triangles as
    everything else so they take the same lighting as the face they sit on.
    """
    vertices: np.ndarray      # (V, 3) float32, view space
    normals: np.ndarray       # (V, 3) float32, view space
    colors: np.ndarray        # (V, 3) float32
    flat: np.ndarray          # (V,) float32, 1 = unshaded overlay tint
    uv: np.ndarray            # (V, 2) float32, negative u = no glyph
    indices: np.ndarray       # (I,) uint32

    @staticmethod
    def empty() -> "MeshBatch":
        return MeshBatch(
            vertices=np.zeros((0, 3), np.float32), normals=np.zeros((0, 3), np.float32),
            colors=np.zeros((0, 3), np.float32), flat=np.zeros((0,), np.float32),
            uv=np.zeros((0, 2), np.float32), indices=np.zeros((0,), np.uint32),
        )


_MESH_VERT = """
#version 410 core
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;
layout(location = 3) in float in_flat;
layout(location = 4) in vec2 in_uv;

uniform mat4 u_proj;

out vec3 v_normal;
out vec3 v_color;
out float v_flat;
out vec2 v_uv;
out float v_depth;

void main() {
    v_normal = in_normal;
    v_color = in_color;
    v_flat = in_flat;
    v_uv = in_uv;
    v_depth = in_pos.z;
    gl_Position = u_proj * vec4(in_pos, 1.0);
}
"""

_MESH_FRAG = """
#version 410 core
in vec3 v_normal;
in vec3 v_color;
in float v_flat;
in vec2 v_uv;
in float v_depth;

uniform sampler2D u_glyph;
uniform vec3 u_ink;
uniform vec3 u_light_dir;
uniform vec3 u_half_vec;
uniform vec3 u_fill_dir;
uniform float u_ambient;
uniform float u_fill_light;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_depth_cue;
uniform float u_zmin;
uniform float u_zspan;

out vec4 frag_color;

void main() {
    // Interpolation shortens the normal; renormalizing is what turns the
    // faceted strip into a smooth surface. Faces pointing away belong to the
    // far side of a closed solid seen through its own opening -- light them
    // from the front rather than leaving them black.
    vec3 normal = normalize(v_normal);
    if (normal.z < 0.0) normal = -normal;

    vec3 base = v_color;
    if (v_uv.x >= 2.0) {
        // A letter with nothing behind it to print on: keep the strokes and
        // throw the rest of the quad away. Supersampling is what smooths the
        // edge, the same as it does for every impostor silhouette.
        if (texture(u_glyph, v_uv - vec2(2.0, 0.0)).r < 0.5) discard;
        base = u_ink;
    } else if (v_uv.x >= 0.0) {
        // Coverage doubles as the blend, so the letter takes the surface's own
        // lighting and reads as printed into it rather than stuck on top.
        base = mix(base, u_ink, texture(u_glyph, v_uv).r);
    }

    float nl = clamp(dot(normal, u_light_dir), 0.0, 1.0);
    float nf = clamp(dot(normal, u_fill_dir), 0.0, 1.0) * u_fill_light;
    float nh = clamp(dot(normal, u_half_vec), 0.0, 1.0);
    float spec = pow(nh, u_shininess) * u_specular_strength;
    float diff = u_ambient + nl + nf;
    vec3 shaded = mix(base * diff + vec3(spec), base, v_flat);

    if (u_depth_cue > 0.0) {
        float f = clamp((v_depth - u_zmin) / u_zspan, 0.0, 1.0);
        shaded *= 1.0 - u_depth_cue * (1.0 - f);
    }
    frag_color = vec4(shaded, 1.0);
}
"""


@dataclass
class ShadingParams:
    """Plain Phong/fog shading parameters — no representation/chemistry
    concepts, unlike ``render.Style``."""
    ambient: float = 0.28
    specular_strength: float = 0.55
    shininess: float = 32.0
    light_dir: Tuple[float, float, float] = (0.35, 0.55, 0.75)
    fill_light: float = 0.25
    depth_cue: float = 0.55
    background: Tuple[float, float, float] = (0.05, 0.06, 0.09)
    transparent: bool = False


_ATOM_VERT = """
#version 410 core
layout(location = 0) in vec2 in_corner;
layout(location = 1) in vec3 in_center;
layout(location = 2) in float in_radius;
layout(location = 3) in vec3 in_color;
layout(location = 4) in float in_flat;

uniform mat4 u_proj;

out vec2 v_offset;
out vec3 v_center;
out float v_radius;
out vec3 v_color;
out float v_flat;

void main() {
    v_offset = in_corner * in_radius;
    v_center = in_center;
    v_radius = in_radius;
    v_color = in_color;
    v_flat = in_flat;
    gl_Position = u_proj * vec4(in_center.xy + v_offset, in_center.z, 1.0);
}
"""

_ATOM_FRAG = """
#version 410 core
in vec2 v_offset;
in vec3 v_center;
in float v_radius;
in vec3 v_color;
in float v_flat;

uniform float u_proj_z_scale;
uniform float u_proj_z_bias;
uniform vec3 u_light_dir;
uniform vec3 u_half_vec;
uniform vec3 u_fill_dir;
uniform float u_ambient;
uniform float u_fill_light;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_depth_cue;
uniform float u_zmin;
uniform float u_zspan;

out vec4 frag_color;

void main() {
    float ox = v_offset.x;
    float oy = v_offset.y;
    float r = v_radius;
    float h2 = r * r - ox * ox - oy * oy;
    if (h2 < 0.0) discard;
    float h = sqrt(h2);
    float depth_view = v_center.z + h;
    vec3 normal = vec3(ox / r, oy / r, h / r);

    float nl = clamp(dot(normal, u_light_dir), 0.0, 1.0);
    float nf = clamp(dot(normal, u_fill_dir), 0.0, 1.0) * u_fill_light;
    float nh = clamp(dot(normal, u_half_vec), 0.0, 1.0);
    float spec = pow(nh, u_shininess) * u_specular_strength;
    float diff = u_ambient + nl + nf;
    vec3 shaded = mix(v_color * diff + vec3(spec), v_color, v_flat);

    if (u_depth_cue > 0.0) {
        float f = clamp((depth_view - u_zmin) / u_zspan, 0.0, 1.0);
        float fog = 1.0 - u_depth_cue * (1.0 - f);
        shaded *= fog;
    }

    frag_color = vec4(shaded, 1.0);
    float nz = u_proj_z_scale * depth_view + u_proj_z_bias;
    gl_FragDepth = nz * 0.5 + 0.5;
}
"""

_BOND_VERT = """
#version 410 core
layout(location = 0) in vec2 in_corner;
layout(location = 1) in vec3 in_a;
layout(location = 2) in vec3 in_b;
layout(location = 3) in float in_radius;
layout(location = 4) in vec3 in_color_a;
layout(location = 5) in vec3 in_color_b;
layout(location = 6) in float in_flat;

uniform mat4 u_proj;

out vec2 v_pixel;
out vec3 v_a;
out vec3 v_b;
out float v_radius;
out vec3 v_color_a;
out vec3 v_color_b;
out float v_flat;

void main() {
    vec2 lo = min(in_a.xy, in_b.xy) - vec2(in_radius);
    vec2 hi = max(in_a.xy, in_b.xy) + vec2(in_radius);
    vec2 center = 0.5 * (lo + hi);
    vec2 half_extent = max(0.5 * (hi - lo), vec2(1e-4));
    vec2 pos = center + in_corner * half_extent;

    v_pixel = pos;
    v_a = in_a;
    v_b = in_b;
    v_radius = in_radius;
    v_color_a = in_color_a;
    v_color_b = in_color_b;
    v_flat = in_flat;

    float z_mid = 0.5 * (in_a.z + in_b.z);
    gl_Position = u_proj * vec4(pos, z_mid, 1.0);
}
"""

_BOND_FRAG = """
#version 410 core
in vec2 v_pixel;
in vec3 v_a;
in vec3 v_b;
in float v_radius;
in vec3 v_color_a;
in vec3 v_color_b;
in float v_flat;

uniform float u_proj_z_scale;
uniform float u_proj_z_bias;
uniform vec3 u_light_dir;
uniform vec3 u_half_vec;
uniform vec3 u_fill_dir;
uniform float u_ambient;
uniform float u_fill_light;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_depth_cue;
uniform float u_zmin;
uniform float u_zspan;

out vec4 frag_color;

void main() {
    vec3 axis = v_b - v_a;
    float L = length(axis);
    if (L < 1e-6) discard;
    vec3 u = axis / L;

    float ex = v_pixel.x - v_a.x;
    float ey = v_pixel.y - v_a.y;
    float uz = u.z;
    float A0 = ex * u.x + ey * u.y;
    float a2 = 1.0 - uz * uz;
    float c2 = ex * ex + ey * ey - A0 * A0 - v_radius * v_radius;

    float s;
    bool hit;
    if (abs(a2) < 1e-9) {
        hit = (-c2) >= 0.0;
        s = 0.0;
    } else {
        float b2 = -2.0 * A0 * uz;
        float disc = b2 * b2 - 4.0 * a2 * c2;
        hit = disc >= 0.0;
        float sq = sqrt(max(disc, 0.0));
        s = (-b2 + sq) / (2.0 * a2);
    }
    if (!hit) discard;

    float t = A0 + uz * s;
    if (t < 0.0 || t > L) discard;

    float depth_view = v_a.z + s;

    float proj_t = t;
    vec3 normal = vec3((ex - proj_t * u.x) / v_radius,
                        (ey - proj_t * u.y) / v_radius,
                        (s - proj_t * u.z) / v_radius);

    float frac = t / L;
    vec3 albedo = (frac < 0.5) ? v_color_a : v_color_b;

    float nl = clamp(dot(normal, u_light_dir), 0.0, 1.0);
    float nf = clamp(dot(normal, u_fill_dir), 0.0, 1.0) * u_fill_light;
    float nh = clamp(dot(normal, u_half_vec), 0.0, 1.0);
    float spec = pow(nh, u_shininess) * u_specular_strength;
    float diff = u_ambient + nl + nf;
    vec3 shaded = mix(albedo * diff + vec3(spec), albedo, v_flat);

    if (u_depth_cue > 0.0) {
        float f = clamp((depth_view - u_zmin) / u_zspan, 0.0, 1.0);
        float fog = 1.0 - u_depth_cue * (1.0 - f);
        shaded *= fog;
    }

    frag_color = vec4(shaded, 1.0);
    float nz = u_proj_z_scale * depth_view + u_proj_z_bias;
    gl_FragDepth = nz * 0.5 + 0.5;
}
"""

_CONE_VERT = """
#version 410 core
layout(location = 0) in vec2 in_corner;
layout(location = 1) in vec3 in_base;
layout(location = 2) in vec3 in_apex;
layout(location = 3) in float in_radius;
layout(location = 4) in vec3 in_color;

uniform mat4 u_proj;

out vec2 v_pixel;
out vec3 v_base;
out vec3 v_apex;
out float v_radius;
out vec3 v_color;

void main() {
    // Only the base has nonzero radius (the apex tapers to a point), but
    // padding the bbox by in_radius at both corners is still a safe
    // (slightly generous near the apex) bound -- same approach _BOND_VERT
    // uses for a cylinder's uniform radius.
    vec2 lo = min(in_base.xy, in_apex.xy) - vec2(in_radius);
    vec2 hi = max(in_base.xy, in_apex.xy) + vec2(in_radius);
    vec2 center = 0.5 * (lo + hi);
    vec2 half_extent = max(0.5 * (hi - lo), vec2(1e-4));
    vec2 pos = center + in_corner * half_extent;

    v_pixel = pos;
    v_base = in_base;
    v_apex = in_apex;
    v_radius = in_radius;
    v_color = in_color;

    float z_mid = 0.5 * (in_base.z + in_apex.z);
    gl_Position = u_proj * vec4(pos, z_mid, 1.0);
}
"""

_CONE_FRAG = """
#version 410 core
in vec2 v_pixel;
in vec3 v_base;
in vec3 v_apex;
in float v_radius;
in vec3 v_color;

uniform float u_proj_z_scale;
uniform float u_proj_z_bias;
uniform vec3 u_light_dir;
uniform vec3 u_half_vec;
uniform vec3 u_fill_dir;
uniform float u_ambient;
uniform float u_fill_light;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_depth_cue;
uniform float u_zmin;
uniform float u_zspan;

out vec4 frag_color;

// GLSL port of render.Renderer._draw_cone_segment's math -- see that
// docstring for the derivation. Target radius at axial offset s is
// C + D*s (linear from R at the base to 0 at the apex); this folds an
// extra (C+D*s)^2 term into the cylinder's usual quadratic, so unlike the
// cylinder shader the leading coefficient a2p isn't sign-guaranteed and
// both roots must be checked, picking the nearer-camera (larger s) one
// that lies on the finite cone (0<=t<=L) with non-negative radius
// (rejecting the mirror nappe beyond the apex that squaring introduces).
void main() {
    vec3 axis = v_apex - v_base;
    float L = length(axis);
    if (L < 1e-6 || v_radius <= 0.0) discard;
    vec3 u = axis / L;
    float R = v_radius;

    float ex = v_pixel.x - v_base.x;
    float ey = v_pixel.y - v_base.y;
    float uz = u.z;
    float A0 = ex * u.x + ey * u.y;
    float a2 = 1.0 - uz * uz;
    float b2 = -2.0 * A0 * uz;
    float c2 = ex * ex + ey * ey - A0 * A0;

    float k = R / L;
    float C = R - k * A0;
    float D = -k * uz;
    float a2p = a2 - D * D;
    float b2p = b2 - 2.0 * C * D;
    float c2p = c2 - C * C;

    bool near_parallel = abs(a2p) < 1e-9;
    float safe_a2p = near_parallel ? 1.0 : a2p;
    float disc = b2p * b2p - 4.0 * a2p * c2p;
    bool has_root = disc >= 0.0;
    float sq = sqrt(max(disc, 0.0));
    float s_plus = (-b2p + sq) / (2.0 * safe_a2p);
    float s_minus = (-b2p - sq) / (2.0 * safe_a2p);

    bool b2p_nonzero = abs(b2p) >= 1e-12;
    float safe_b2p = b2p_nonzero ? b2p : 1.0;
    float s_lin = -c2p / safe_b2p;
    bool lin_valid = near_parallel && b2p_nonzero;

    float t_plus = A0 + uz * s_plus;
    float r_plus = C + D * s_plus;
    bool v_plus = has_root && !near_parallel && (t_plus >= 0.0) && (t_plus <= L) && (r_plus >= -1e-6);

    float t_minus = A0 + uz * s_minus;
    float r_minus = C + D * s_minus;
    bool v_minus = has_root && !near_parallel && (t_minus >= 0.0) && (t_minus <= L) && (r_minus >= -1e-6);

    float t_lin = A0 + uz * s_lin;
    float r_lin = C + D * s_lin;
    bool v_lin = lin_valid && (t_lin >= 0.0) && (t_lin <= L) && (r_lin >= -1e-6);

    const float NEG_INF = -1e30;
    float s = NEG_INF;
    if (v_plus) s = max(s, s_plus);
    if (v_minus) s = max(s, s_minus);
    if (v_lin) s = max(s, s_lin);
    if (s <= NEG_INF * 0.5) discard;

    float t = A0 + uz * s;
    float depth_view = v_base.z + s;

    float wx = ex - t * u.x;
    float wy = ey - t * u.y;
    float wz = s - t * u.z;
    float r_t = max(C + D * s, 1e-6);
    vec3 normal = normalize(vec3(wx / r_t, wy / r_t, wz / r_t) * L + u * R);

    float nl = clamp(dot(normal, u_light_dir), 0.0, 1.0);
    float nf = clamp(dot(normal, u_fill_dir), 0.0, 1.0) * u_fill_light;
    float nh = clamp(dot(normal, u_half_vec), 0.0, 1.0);
    float spec = pow(nh, u_shininess) * u_specular_strength;
    float diff = u_ambient + nl + nf;
    vec3 shaded = v_color * diff + vec3(spec);

    if (u_depth_cue > 0.0) {
        float f = clamp((depth_view - u_zmin) / u_zspan, 0.0, 1.0);
        float fog = 1.0 - u_depth_cue * (1.0 - f);
        shaded *= fog;
    }

    frag_color = vec4(shaded, 1.0);
    float nz = u_proj_z_scale * depth_view + u_proj_z_bias;
    gl_FragDepth = nz * 0.5 + 0.5;
}
"""

_QUAD_CORNERS = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)

# Fullscreen-quad box-average downsample: for each output pixel, average the
# ds*ds block of source texels. Exact box filter (matches the CPU _downsample),
# and it runs on the GPU in ~1ms instead of ~200ms in numpy.
_DOWNSAMPLE_VERT = """
#version 410 core
layout(location = 0) in vec2 in_corner;
void main() { gl_Position = vec4(in_corner, 0.0, 1.0); }
"""

_DOWNSAMPLE_FRAG = """
#version 410 core
uniform sampler2D u_src;
uniform int u_factor;
out vec4 frag_color;
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_factor;
    vec4 acc = vec4(0.0);
    for (int dy = 0; dy < u_factor; ++dy)
        for (int dx = 0; dx < u_factor; ++dx)
            acc += texelFetch(u_src, base + ivec2(dx, dy), 0);
    acc /= float(u_factor * u_factor);
    // The source averaged in premultiplied space (undrawn texels are 0,0,0,0),
    // so divide back out to straight alpha -- what the Kitty/PNG encoders want.
    // Opaque scenes have alpha==1 everywhere, making this a no-op there.
    if (acc.a > 0.0) acc.rgb /= acc.a;
    frag_color = acc;
}
"""


def _create_standalone_context() -> moderngl.Context:
    """Create a standalone context, preferring EGL over moderngl's default.

    On Linux, ``moderngl.create_context(standalone=True)`` always picks the
    GLX/X11 backend (see ``glcontext.default_backend``) regardless of
    whether ``DISPLAY`` actually has working GL behind it -- an SSH session
    with X11 forwarding sets ``DISPLAY`` to a forwarded/virtual X server
    that has no real (or even software) GLX support, so GLX context
    creation fails with a real GPU sitting right there. EGL's device
    platform talks to the GPU directly via ``/dev/dri/renderD*``, bypassing
    X entirely, so try it first and fall back to moderngl's default (GLX on
    Linux, CGL on macOS) if EGL isn't available.
    """
    try:
        return moderngl.create_context(standalone=True, backend="egl")
    except Exception:
        return moderngl.create_context(standalone=True)


class GLRenderer:
    """GPU sphere/cylinder impostor renderer.

    Mirrors ``render.Renderer``'s ``__init__(width, height)``/
    ``resize(width, height)`` shape so callers can construct/resize either
    renderer identically, but ``render()`` takes generic primitive batches
    plus a raw projection matrix rather than a molecule-specific object.
    """

    def __init__(self, width: int, height: int):
        self.ctx = _create_standalone_context()
        self.width = int(width)
        self.height = int(height)
        self._quad_vbo = self.ctx.buffer(_QUAD_CORNERS.tobytes())
        self._atom_program = self.ctx.program(vertex_shader=_ATOM_VERT, fragment_shader=_ATOM_FRAG)
        self._bond_program = self.ctx.program(vertex_shader=_BOND_VERT, fragment_shader=_BOND_FRAG)
        self._cone_program = self.ctx.program(vertex_shader=_CONE_VERT, fragment_shader=_CONE_FRAG)
        self._downsample_program = self.ctx.program(
            vertex_shader=_DOWNSAMPLE_VERT, fragment_shader=_DOWNSAMPLE_FRAG)
        self._downsample_vao = self.ctx.vertex_array(
            self._downsample_program, [(self._quad_vbo, "2f", "in_corner")])
        # Persistent instance VBOs and their VAOs: per frame the new batch is
        # streamed into the SAME GL objects (orphan + write) instead of
        # creating and destroying buffer/vertex-array objects per draw --
        # keep GL object identity stable and let the driver recycle storage,
        # the same reason PyMOL keeps geometry in long-lived VBOs.
        self._atom_vbo = self.ctx.buffer(reserve=4)
        self._atom_vao = self.ctx.vertex_array(
            self._atom_program,
            [
                (self._quad_vbo, "2f", "in_corner"),
                (self._atom_vbo, "3f 1f 3f 1f/i", "in_center", "in_radius", "in_color", "in_flat"),
            ],
        )
        self._bond_vbo = self.ctx.buffer(reserve=4)
        self._bond_vao = self.ctx.vertex_array(
            self._bond_program,
            [
                (self._quad_vbo, "2f", "in_corner"),
                (self._bond_vbo, "3f 3f 1f 3f 3f 1f/i", "in_a", "in_b", "in_radius", "in_color_a", "in_color_b", "in_flat"),
            ],
        )
        self._mesh_program = self.ctx.program(vertex_shader=_MESH_VERT,
                                              fragment_shader=_MESH_FRAG)
        self._mesh_vbo = self.ctx.buffer(reserve=4)
        self._mesh_ibo = self.ctx.buffer(reserve=4)
        self._mesh_vao = self.ctx.vertex_array(
            self._mesh_program,
            [(self._mesh_vbo, "3f 3f 3f 1f 2f",
              "in_pos", "in_normal", "in_color", "in_flat", "in_uv")],
            index_buffer=self._mesh_ibo, index_element_size=4)
        self._glyph_tex = None         # lazily uploaded; see _glyph_texture

        self._cone_vbo = self.ctx.buffer(reserve=4)
        self._cone_vao = self.ctx.vertex_array(
            self._cone_program,
            [
                (self._quad_vbo, "2f", "in_corner"),
                (self._cone_vbo, "3f 3f 1f 3f/i", "in_base", "in_apex", "in_radius", "in_color"),
            ],
        )
        self._fbo = None
        self._color_tex = None         # color attachment is a texture so the
        self._depth_rb = None          # downsample pass can sample it
        self._resolve_fbo = None       # smaller FBO the downsample pass writes to
        self._resolve_size = None
        self._build_fbo(self.width, self.height)

    def _build_fbo(self, w: int, h: int) -> None:
        if self._fbo is not None:
            self._fbo.release()
            self._color_tex.release()
            self._depth_rb.release()
        self._color_tex = self.ctx.texture((w, h), components=4, dtype="f1")
        self._depth_rb = self.ctx.depth_renderbuffer((w, h))
        self._fbo = self.ctx.framebuffer(color_attachments=[self._color_tex], depth_attachment=self._depth_rb)

    def _resolve_target(self, w: int, h: int):
        """A cached (rebuilt on size change) color-only FBO at (w, h) that the
        box-average downsample pass renders into."""
        if self._resolve_size != (w, h):
            if self._resolve_fbo is not None:
                self._resolve_fbo.release()
                self._resolve_rb.release()
            self._resolve_rb = self.ctx.renderbuffer((w, h), components=4, dtype="f1")
            self._resolve_fbo = self.ctx.framebuffer(color_attachments=[self._resolve_rb])
            self._resolve_size = (w, h)
        return self._resolve_fbo

    def resize(self, width: int, height: int) -> None:
        width, height = int(width), int(height)
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        self._build_fbo(width, height)

    # ------------------------------------------------------------------
    def render(self, spheres: SphereBatch, cylinders: CylinderBatch,
               proj: np.ndarray, shading: ShadingParams | None = None,
               downsample: int = 1, cones: ConeBatch | None = None,
               mesh: MeshBatch | None = None,
               ink: Tuple[float, float, float] = (0.09, 0.09, 0.11)) -> np.ndarray:
        """Render *spheres*, *cylinders* and *cones* under projection *proj*.

        *cones* defaults to empty so existing call sites that only know about
        spheres/cylinders keep working unchanged.

        *proj* is a 4x4 matrix in the usual mathematical convention
        (``[nx,ny,nz,1] = proj @ [vx,vy,vz,1]``, row-major indexing) mapping
        the primitives' coordinate space to clip-space NDC. Its ``[2,2]``/
        ``[2,3]`` entries double as the scale/bias used to reproject each
        fragment's own analytic depth (see ``gl_adapter._build_projection``
        for how vimol builds one from its orthographic ``Camera``).

        *downsample* (>=1) supersamples: the scene is drawn at the full
        ``(width, height)`` and box-averaged down by this factor **on the GPU**
        (a fullscreen downsample-shader pass, ~1ms) rather than in numpy on the
        CPU (~200ms at full screen). The returned image is therefore
        ``(height//downsample, width//downsample)``. The caller sizes this
        renderer to display-size*factor and passes the same factor here.

        Returns a zero-copy, read-only ``uint8`` view of the readback bytes
        (the underlying ``bytes`` object is kept alive via the array's
        ``.base``), in raw ``glReadPixels`` row order -- see the module
        docstring for the orientation contract. Opaque frames are (H, W, 3),
        transparent ones (H, W, 4) with straight alpha (the downsample
        shader un-premultiplies; the ds==1 path never premultiplies).
        """
        shading = shading or ShadingParams()
        cones = cones or ConeBatch.empty()
        mesh = mesh or MeshBatch.empty()
        W, H = self.width, self.height
        downsample = max(1, int(downsample))

        proj = np.asarray(proj, dtype=np.float64)
        z_scale = float(proj[2, 2])
        z_bias = float(proj[2, 3])
        proj_gl = np.ascontiguousarray(proj.T, dtype=np.float32)  # column-major for GLSL

        centers = np.asarray(spheres.centers, dtype=np.float32).reshape(-1, 3)
        radii_s = np.asarray(spheres.radii, dtype=np.float32).reshape(-1)
        colors_s = np.asarray(spheres.colors, dtype=np.float32).reshape(-1, 3)
        flat_s = np.asarray(spheres.flat, dtype=np.float32).reshape(-1)
        keep_s = radii_s > 0
        centers, radii_s, colors_s, flat_s = (
            centers[keep_s], radii_s[keep_s], colors_s[keep_s], flat_s[keep_s],
        )

        ca = np.asarray(cylinders.a, dtype=np.float32).reshape(-1, 3)
        cb = np.asarray(cylinders.b, dtype=np.float32).reshape(-1, 3)
        radii_c = np.asarray(cylinders.radii, dtype=np.float32).reshape(-1)
        colors_a = np.asarray(cylinders.colors_a, dtype=np.float32).reshape(-1, 3)
        colors_b = np.asarray(cylinders.colors_b, dtype=np.float32).reshape(-1, 3)
        flat_c = np.asarray(cylinders.flat, dtype=np.float32).reshape(-1)
        keep_c = radii_c > 0
        ca, cb, radii_c, colors_a, colors_b, flat_c = (
            ca[keep_c], cb[keep_c], radii_c[keep_c], colors_a[keep_c], colors_b[keep_c], flat_c[keep_c],
        )

        cone_base = np.asarray(cones.base, dtype=np.float32).reshape(-1, 3)
        cone_apex = np.asarray(cones.apex, dtype=np.float32).reshape(-1, 3)
        radii_h = np.asarray(cones.radius, dtype=np.float32).reshape(-1)
        colors_h = np.asarray(cones.color, dtype=np.float32).reshape(-1, 3)
        keep_h = radii_h > 0
        cone_base, cone_apex, radii_h, colors_h = (
            cone_base[keep_h], cone_apex[keep_h], radii_h[keep_h], colors_h[keep_h],
        )

        mesh_v = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        if len(centers) or len(ca) or len(cone_base) or len(mesh_v):
            depths = np.concatenate([centers[:, 2], ca[:, 2], cb[:, 2], cone_base[:, 2],
                                     cone_apex[:, 2], mesh_v[:, 2]])
            zmin = float(depths.min())
            zspan = max(float(depths.max()) - zmin, 1e-6)
        else:
            zmin, zspan = 0.0, 1.0

        light = _normalize(np.asarray(shading.light_dir, dtype=np.float64))
        halfv = _normalize(light + np.array([0.0, 0.0, 1.0]))
        fill = _normalize(np.array([-light[0], -light[1], 0.6]))

        self._fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        if shading.transparent:
            self._fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        else:
            bg = shading.background
            self._fbo.clear(bg[0], bg[1], bg[2], 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_func = "<"

        common = dict(
            u_proj_z_scale=z_scale, u_proj_z_bias=z_bias,
            u_light_dir=tuple(light), u_half_vec=tuple(halfv), u_fill_dir=tuple(fill),
            u_ambient=shading.ambient, u_fill_light=shading.fill_light,
            u_specular_strength=shading.specular_strength, u_shininess=shading.shininess,
            u_depth_cue=shading.depth_cue, u_zmin=zmin, u_zspan=zspan,
        )

        if len(ca):
            self._draw_bonds(ca, cb, radii_c, colors_a, colors_b, flat_c, proj_gl, common)
        if len(centers):
            self._draw_atoms(centers, radii_s, colors_s, flat_s, proj_gl, common)
        if len(cone_base):
            self._draw_cones(cone_base, cone_apex, radii_h, colors_h, proj_gl, common)
        if len(mesh_v) and len(mesh.indices):
            self._draw_mesh(mesh, ink, proj_gl, common)

        ch = 4 if shading.transparent else 3
        if downsample > 1:
            tw, th = W // downsample, H // downsample
            resolve = self._resolve_target(tw, th)
            resolve.use()
            self.ctx.viewport = (0, 0, tw, th)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._color_tex.use(location=0)
            self._downsample_program["u_src"] = 0
            self._downsample_program["u_factor"] = downsample
            self._downsample_vao.render(moderngl.TRIANGLE_STRIP)
            raw = resolve.read(components=ch, dtype="f1")
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(th, tw, ch)
        else:
            raw = self._fbo.read(components=ch, dtype="f1")
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, ch)
        return arr

    # ------------------------------------------------------------------
    def _set_uniforms(self, program, proj_gl: np.ndarray, common: dict) -> None:
        program["u_proj"].write(proj_gl.tobytes())
        for name, value in common.items():
            program[name].value = value

    def _stream(self, vbo, data: np.ndarray) -> None:
        """Replace *vbo*'s contents with *data* (orphan + write).

        ``orphan`` detaches the old storage (the driver recycles it once the
        GPU is done with any in-flight draw that referenced it), so a fresh
        upload never stalls on the previous frame's draw the way a plain
        ``write`` into a mapped/in-use buffer can.
        """
        payload = data.tobytes()
        vbo.orphan(len(payload))
        vbo.write(payload)

    def _glyph_texture(self):
        """The letter atlas, uploaded once and kept for the context's life."""
        if self._glyph_tex is None:
            from .glyph_font import atlas
            image, _boxes = atlas()
            tex = self.ctx.texture((image.shape[1], image.shape[0]), 1,
                                   np.ascontiguousarray(image).tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            # Clamp: a letter's quad maps exactly onto its tile, and wrapping
            # would fetch its neighbour along the seam.
            tex.repeat_x = tex.repeat_y = False
            self._glyph_tex = tex
        return self._glyph_tex

    def _draw_mesh(self, mesh: MeshBatch, ink, proj_gl, common) -> None:
        n = mesh.vertices.shape[0]
        data = np.empty((n, 12), dtype=np.float32)
        data[:, 0:3] = mesh.vertices
        data[:, 3:6] = mesh.normals
        data[:, 6:9] = mesh.colors
        data[:, 9] = mesh.flat
        data[:, 10:12] = mesh.uv
        self._stream(self._mesh_vbo, data)
        indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
        self._stream(self._mesh_ibo, indices)
        self._glyph_texture().use(location=0)
        self._mesh_program["u_glyph"] = 0
        self._mesh_program["u_ink"].value = tuple(float(c) for c in ink)
        self._mesh_program["u_proj"].write(proj_gl.tobytes())
        for name, value in common.items():
            if name in self._mesh_program:
                self._mesh_program[name].value = value
        self._mesh_vao.render(moderngl.TRIANGLES, vertices=len(indices))

    def _draw_atoms(self, centers, radii, colors, flat, proj_gl, common) -> None:
        n = centers.shape[0]
        data = np.empty((n, 8), dtype=np.float32)
        data[:, 0:3] = centers
        data[:, 3] = radii
        data[:, 4:7] = colors
        data[:, 7] = flat
        self._stream(self._atom_vbo, data)
        self._set_uniforms(self._atom_program, proj_gl, common)
        self._atom_vao.render(moderngl.TRIANGLE_STRIP, instances=n)

    def _draw_bonds(self, a, b, radii, colors_a, colors_b, flat, proj_gl, common) -> None:
        n = a.shape[0]
        data = np.empty((n, 14), dtype=np.float32)
        data[:, 0:3] = a
        data[:, 3:6] = b
        data[:, 6] = radii
        data[:, 7:10] = colors_a
        data[:, 10:13] = colors_b
        data[:, 13] = flat
        self._stream(self._bond_vbo, data)
        self._set_uniforms(self._bond_program, proj_gl, common)
        self._bond_vao.render(moderngl.TRIANGLE_STRIP, instances=n)

    def _draw_cones(self, base, apex, radii, colors, proj_gl, common) -> None:
        n = base.shape[0]
        data = np.empty((n, 10), dtype=np.float32)
        data[:, 0:3] = base
        data[:, 3:6] = apex
        data[:, 6] = radii
        data[:, 7:10] = colors
        self._stream(self._cone_vbo, data)
        self._set_uniforms(self._cone_program, proj_gl, common)
        self._cone_vao.render(moderngl.TRIANGLE_STRIP, instances=n)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v
