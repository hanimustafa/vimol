"""Tests for the optional GL backend.

Every test here needs a real, working headless GL context, which needs both
the ``moderngl`` package (`importorskip`, skips the whole module at
collection time if absent) and an actual usable driver/display (checked at
runtime via the `gl_available` fixture, since that can fail even when the
package is installed -- e.g. a CI container with no EGL/GLX).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

moderngl = pytest.importorskip("moderngl")

import mviewer
from mviewer.bonds import ensure_bonds
from mviewer.render import Style
from mviewer.scene import Scene
from mviewer.gl_render import GLRenderer, SphereBatch, CylinderBatch, ConeBatch, ShadingParams
from mviewer.gl_adapter import _build_projection

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.fixture(scope="module")
def gl_available():
    try:
        ctx = moderngl.create_context(standalone=True)
        ctx.release()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no GL context available: {e}")


def _proj(zoom=20.0, pan=(0.0, 0.0), w=200, h=200, extent=5.0):
    return _build_projection(zoom, np.array(pan), w, h, extent)


# -- renderer-only: no Molecule/Camera/Style involved at all -----------------

def test_gl_renderer_is_generic_sphere_only(gl_available):
    """Proves the renderer works from hand-built primitives, no mviewer
    molecule types involved -- the actual test of "generic"."""
    r = GLRenderer(64, 64)
    spheres = SphereBatch(
        centers=np.array([[0.0, 0.0, 0.0]], np.float32),
        radii=np.array([2.0], np.float32),
        colors=np.array([[1.0, 0.0, 0.0]], np.float32),
    )
    img = r.render(spheres, CylinderBatch.empty(), _proj(w=64, h=64), ShadingParams())
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
    assert img[32, 32, 0] > img[32, 32, 2]  # reddish center


def test_gl_renderer_cylinder_only_scene(gl_available):
    """A cylinder-only scene (no spheres) must not crash computing fog range."""
    r = GLRenderer(64, 64)
    cyl = CylinderBatch(
        a=np.array([[-2.0, 0.0, 0.0]], np.float32),
        b=np.array([[2.0, 0.0, 0.0]], np.float32),
        radii=np.array([0.5], np.float32),
        colors_a=np.array([[1.0, 1.0, 0.0]], np.float32),
        colors_b=np.array([[0.0, 1.0, 1.0]], np.float32),
    )
    img = r.render(SphereBatch.empty(), cyl, _proj(w=64, h=64), ShadingParams())
    assert img.shape == (64, 64, 3)
    bg = np.array(ShadingParams().background) * 255
    assert np.abs(img[32, 32].astype(float) - bg).sum() > 30  # bond drawn at center


def test_gl_renderer_cone_only_scene(gl_available):
    """A cone-only scene (the arrow-head primitive, no spheres/cylinders)
    must render and not crash computing fog range -- mirrors
    test_gl_renderer_cylinder_only_scene for the new primitive."""
    cone = ConeBatch(
        base=np.array([[0.0, 0.0, 0.0]], np.float32),
        apex=np.array([[0.0, 0.0, 3.0]], np.float32),
        radius=np.array([1.5], np.float32),
        color=np.array([[1.0, 0.5, 0.0]], np.float32),
    )
    r = GLRenderer(64, 64)
    img = r.render(SphereBatch.empty(), CylinderBatch.empty(), _proj(w=64, h=64),
                   ShadingParams(), cones=cone)
    assert img.shape == (64, 64, 3)
    bg = np.array(ShadingParams().background) * 255
    assert np.abs(img[32, 32].astype(float) - bg).sum() > 30  # cone base fills the center


def test_gl_renderer_empty_scene_is_background_only(gl_available):
    r = GLRenderer(32, 32)
    img = r.render(SphereBatch.empty(), CylinderBatch.empty(), _proj(w=32, h=32),
                   ShadingParams())
    bg = np.array(ShadingParams().background) * 255
    # +-1 tolerance: GL rounds the float clear color to 8-bit, we don't rely
    # on matching its exact rounding convention (truncate vs round-to-nearest)
    assert np.abs(img[0, 0].astype(float) - bg).max() <= 1
    assert np.abs(img[16, 16].astype(float) - bg).max() <= 1


def test_gl_renderer_orientation_matches_screen_convention(gl_available):
    """A sphere offset toward +x/+y (view space) must land toward larger x /
    smaller row index (row 0 = top) -- catches a missing or extra vertical
    flip, which a coverage-count-only check would miss entirely."""
    r = GLRenderer(200, 200)
    spheres = SphereBatch(
        centers=np.array([[3.0, 2.0, 0.0]], np.float32),
        radii=np.array([1.0], np.float32),
        colors=np.array([[0.2, 0.8, 0.2]], np.float32),
    )
    img = r.render(spheres, CylinderBatch.empty(), _proj(), ShadingParams())
    bg = np.array(ShadingParams().background) * 255
    ys, xs = np.where(np.abs(img.astype(float) - bg).sum(axis=-1) > 30)
    assert xs.mean() > 100  # +x (view) -> larger column
    assert ys.mean() < 100  # +y (view, "up") -> smaller row (row 0 is top)


def test_gl_renderer_depth_test_nearer_wins(gl_available):
    """Two overlapping spheres of different colors and depths: the nearer
    one (larger view-space z) must win the overlap -- pins depth-test
    correctness, not just "something was drawn"."""
    r = GLRenderer(120, 120)
    spheres = SphereBatch(
        centers=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], np.float32),
        radii=np.array([2.0, 2.0], np.float32),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32),  # far=red, near=blue
    )
    img = r.render(spheres, CylinderBatch.empty(), _proj(w=120, h=120),
                   ShadingParams(depth_cue=0.0))
    center = img[60, 60]
    assert int(center[2]) > int(center[0])  # blue (nearer) beats red (farther)


def test_gl_renderer_lighting_direction(gl_available):
    """A point offset toward the light direction must be brighter than one
    offset away from it -- pins the light direction/half-vector, which a
    silhouette-only check can't distinguish from a backwards light."""
    r = GLRenderer(200, 200)
    spheres = SphereBatch(
        centers=np.array([[0.0, 0.0, 0.0]], np.float32),
        radii=np.array([2.0], np.float32),
        colors=np.array([[0.4, 0.4, 0.4]], np.float32),  # mid-gray: avoid 255 clipping
    )
    img = r.render(spheres, CylinderBatch.empty(), _proj(), ShadingParams(depth_cue=0.0))
    cx, cy = 100, 100
    off = 25  # pixels, within the sphere's silhouette
    lit = img[cy - off, cx + off].astype(int).sum()      # toward +x, +y (light has +x,+y)
    shadow = img[cy + off, cx - off].astype(int).sum()   # toward -x, -y
    assert lit > shadow


# -- transparent mode ---------------------------------------------------------

def test_gl_renderer_transparent_cutout(gl_available):
    r = GLRenderer(64, 64)
    spheres = SphereBatch(
        centers=np.array([[0.0, 0.0, 0.0]], np.float32),
        radii=np.array([2.0], np.float32),
        colors=np.array([[1.0, 1.0, 1.0]], np.float32),
    )
    img = r.render(spheres, CylinderBatch.empty(), _proj(w=64, h=64),
                   ShadingParams(transparent=True))
    assert img.shape == (64, 64, 4)
    assert img[0, 0, 3] == 0
    assert img[32, 32, 3] == 255


# -- through the mviewer Scene/adapter pipeline, vs the CPU renderer ---------

def test_gl_scene_shape_dtype_parity(gl_available):
    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 120, 120, style=Style(), backend="gl")
    assert scene.backend == "gl"
    img = scene.render()
    assert img.shape == (120, 120, 3)
    assert img.dtype == np.uint8


def test_gl_scene_transparent_rgba_cutout(gl_available):
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 120, 120, style=Style(transparent=True), backend="gl")
    img = scene.render()
    assert img.shape == (120, 120, 4)
    assert img[0, 0, 3] == 0
    assert img[60, 60, 3] == 255


def test_gl_vs_cpu_similar_coverage(gl_available):
    """Not bit-identical (different rasterizers/AA) -- assert comparable
    drawn-pixel footprint and non-background-only output, like the existing
    threshold-based CPU tests. This is deliberately coarse -- the sharper
    orientation/depth/lighting checks above are what actually pin
    correctness; this one just confirms the two backends broadly agree."""
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    cpu = Scene(mol, 160, 160, backend="cpu")
    gl = Scene(mol, 160, 160, backend="gl")
    cpu.camera.orbit(20, -15)
    gl.camera.orbit(20, -15)
    img_cpu = cpu.render()
    img_gl = gl.render()
    bg = np.array(Style().background) * 255
    drawn_cpu = (np.abs(img_cpu.astype(np.float32) - bg).sum(axis=-1) > 30).sum()
    drawn_gl = (np.abs(img_gl.astype(np.float32) - bg).sum(axis=-1) > 30).sum()
    assert drawn_cpu > 0 and drawn_gl > 0
    assert 0.5 < drawn_gl / drawn_cpu < 2.0


def test_gl_vs_cpu_vector_field_parity(gl_available):
    """Same spirit as test_gl_vs_cpu_similar_coverage, but exercising the new
    arrow (shaft + cone head) primitive: both backends should broadly agree
    on drawn footprint, and both must actually draw the arrow in its
    assigned (magenta) color, not just background/element colors."""
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    vectors = np.zeros((mol.n_atoms, 3))
    vectors[0] = [2.5, 0.0, 0.0]
    mol.add_vector_field(vectors, color=(1.0, 0.0, 1.0), radius=0.08, head_scale=3.0)
    cpu = Scene(mol, 160, 160, backend="cpu")
    gl = Scene(mol, 160, 160, backend="gl")
    img_cpu = cpu.render()
    img_gl = gl.render()
    bg = np.array(Style().background) * 255
    drawn_cpu = (np.abs(img_cpu.astype(np.float32) - bg).sum(axis=-1) > 30).sum()
    drawn_gl = (np.abs(img_gl.astype(np.float32) - bg).sum(axis=-1) > 30).sum()
    assert drawn_cpu > 0 and drawn_gl > 0
    assert 0.5 < drawn_gl / drawn_cpu < 2.0

    def magenta_count(img):
        return int(((img[..., 0].astype(int) > 150) & (img[..., 2].astype(int) > 100) &
                    (img[..., 1].astype(int) < 120)).sum())

    assert magenta_count(img_cpu) > 0
    assert magenta_count(img_gl) > 0


def test_gl_scene_picking_unaffected_by_backend(gl_available):
    """Picking is pure analytic CPU math independent of the renderer -- it
    must behave identically regardless of backend."""
    from mviewer.widget import MoleculeWidget

    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 200, 200, supersample=1, backend="gl")
    assert w.scene.backend == "gl"
    cam = w.scene.camera
    Wr, Hr = w.scene.render_size
    v = cam.view_positions(mol.positions)
    sz = v[:, 2]
    front = int(np.argmax(sz))
    sx = Wr * 0.5 + cam.pan[0] + v[front, 0] * cam.zoom
    sy = Hr * 0.5 - cam.pan[1] - v[front, 1] * cam.zoom
    assert w.pick(sx / w.scene.supersample, sy / w.scene.supersample) == front


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
