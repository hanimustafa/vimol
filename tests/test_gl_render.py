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

import vimol
from vimol.bonds import ensure_bonds
from vimol.render import Style
from vimol.scene import Scene
from vimol.gl_render import GLRenderer, SphereBatch, CylinderBatch, ConeBatch, ShadingParams
from vimol.gl_adapter import _build_projection

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
    """Proves the renderer works from hand-built primitives, no vimol
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


# -- through the vimol Scene/adapter pipeline, vs the CPU renderer ---------

def test_gl_scene_shape_dtype_parity(gl_available):
    mol = vimol.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 120, 120, style=Style(), backend="gl")
    assert scene.backend == "gl"
    img = scene.render()
    assert img.shape == (120, 120, 3)
    assert img.dtype == np.uint8


def test_gl_scene_transparent_rgba_cutout(gl_available):
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
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
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
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
    from vimol.widget import MoleculeWidget

    mol = vimol.load(os.path.join(EX, "c60.xyz"))
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



# -- flat shading (design §4.4/§4.5): GL backend ----------------------------

def test_sphere_batch_flat_defaults_to_zeros_matching_length(gl_available):
    """SphereBatch built without `flat` (every existing caller) must keep
    working -- the field defaults to zeros shaped like the other arrays."""
    spheres = SphereBatch(
        centers=np.zeros((3, 3), np.float32),
        radii=np.ones(3, np.float32),
        colors=np.zeros((3, 3), np.float32),
    )
    assert spheres.flat.shape == (3,)
    assert not spheres.flat.any()
    assert SphereBatch.empty().flat.shape == (0,)


def test_cylinder_batch_flat_defaults_to_zeros_matching_length(gl_available):
    cyl = CylinderBatch(
        a=np.zeros((2, 3), np.float32), b=np.zeros((2, 3), np.float32),
        radii=np.ones(2, np.float32),
        colors_a=np.zeros((2, 3), np.float32), colors_b=np.zeros((2, 3), np.float32),
    )
    assert cyl.flat.shape == (2,)
    assert not cyl.flat.any()
    assert CylinderBatch.empty().flat.shape == (0,)


def test_gl_flat_sphere_renders_uniform_color_no_shading(gl_available):
    """Mirrors the CPU flat-shading test: a flat sphere must render with a
    perfectly uniform color across its silhouette (mix(..., v_flat) with
    v_flat=1 collapses to the plain color, no diffuse/specular gradient)."""
    r = GLRenderer(200, 200)
    spheres = SphereBatch(
        centers=np.array([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], np.float32),
        radii=np.array([2.0, 2.0], np.float32),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32),
        flat=np.array([0.0, 1.0], np.float32),
    )
    img = r.render(spheres, CylinderBatch.empty(), _proj(w=200, h=200, extent=6.0),
                   ShadingParams(depth_cue=0.0))
    blue_mask = (img[:, :, 2] > 150) & (img[:, :, 0] < 50)
    assert blue_mask.sum() > 100
    blue_pixels = img[blue_mask]
    assert (blue_pixels == blue_pixels[0]).all()
    red_mask = (img[:, :, 0] > 50) & (img[:, :, 2] < 50)
    assert red_mask.sum() > 100
    red_pixels = img[red_mask]
    assert not (red_pixels == red_pixels[0]).all()


def test_gl_flat_cylinder_renders_uniform_color_no_shading(gl_available):
    r = GLRenderer(160, 80)
    cyl = CylinderBatch(
        a=np.array([[-2.0, 0.0, 0.0]], np.float32),
        b=np.array([[2.0, 0.0, 0.0]], np.float32),
        radii=np.array([1.0], np.float32),
        colors_a=np.array([[0.0, 1.0, 0.0]], np.float32),
        colors_b=np.array([[0.0, 1.0, 0.0]], np.float32),
        flat=np.array([1.0], np.float32),
    )
    img = r.render(SphereBatch.empty(), cyl, _proj(w=160, h=80, extent=3.0),
                   ShadingParams(depth_cue=0.0))
    row = img[40, :, :]
    green_cols = np.where((row[:, 1] > 100) & (row[:, 0] < 60))[0]
    assert len(green_cols) > 5
    pixels = row[green_cols]
    assert (pixels == pixels[0]).all()


def test_gl_adapter_builds_flat_arrays_from_style_flat_mask(gl_available):
    """gl_adapter must translate style.flat_mask into SphereBatch.flat /
    CylinderBatch.flat, with arrow shafts always flat=0 (design: arrows are
    never flat)."""
    from vimol.gl_adapter import molecule_to_gl_inputs

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    vectors = np.zeros((mol.n_atoms, 3))
    vectors[0] = [2.0, 0.0, 0.0]
    mol.add_vector_field(vectors)
    style = Style(flat_mask=np.array([True] + [False] * (mol.n_atoms - 1)))
    from vimol.camera import Camera
    cam = Camera(center=mol.centroid(), extent=mol.radius_of_gyration_extent())
    cam.fit(160, 160, cam.extent)
    spheres, cylinders, cones, proj, shading = molecule_to_gl_inputs(mol, cam, style, 160, 160)
    assert spheres.flat[0] == 1.0
    assert not spheres.flat[1:].any()
    # bonds touching atom 0 should be flat too (endpoint's flag)
    assert cylinders.flat[: len(mol.bonds)].sum() > 0
    # the arrow shaft (appended after real bonds) must stay unflattened
    assert not cylinders.flat[len(mol.bonds):].any()


def test_scene_render_overlay_first_entry_cpk_rest_flat_tinted_gl(gl_available):
    """GL twin of the CPU end-to-end overlay-colouring test: Scene.render()
    through the real gl_adapter/GLRenderer chain, not hand-built batches."""
    from vimol.structures import StructureSet
    from vimol.molecule import Molecule

    sset = StructureSet()
    a = Molecule(symbols=["C", "C"], positions=np.array([[-5.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]))
    b = Molecule(symbols=["C", "C"], positions=np.array([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    sset.append(a, label="a")
    entry_b = sset.append(b, label="b")
    sset.overlay = True
    style = Style(representation="spacefill", depth_cue=0.0)
    scene = Scene(sset, 240, 240, style=style, backend="gl")
    scene.camera.center = np.zeros(3)
    scene.camera.rotation = np.eye(3)
    scene.camera.fit(240, 240, 10.0)
    img = scene.render()

    tint_rgb = tuple(int(c * 255) for c in entry_b.tint)
    tint_mask = (np.abs(img[:, :, 0].astype(int) - tint_rgb[0]) < 12) & \
                (np.abs(img[:, :, 1].astype(int) - tint_rgb[1]) < 12) & \
                (np.abs(img[:, :, 2].astype(int) - tint_rgb[2]) < 12)
    assert tint_mask.sum() > 50
    tinted_pixels = img[tint_mask]
    assert (tinted_pixels == tinted_pixels[0]).all()

    carbon_rgb = tuple(int(c * 255) for c in a.element_colors()[0])
    active_mask = (np.abs(img[:, :, 0].astype(int) - carbon_rgb[0]) < 40) & \
                  (np.abs(img[:, :, 1].astype(int) - carbon_rgb[1]) < 40) & \
                  (np.abs(img[:, :, 2].astype(int) - carbon_rgb[2]) < 40)
    assert active_mask.sum() > 50
    active_pixels = img[active_mask]
    assert not (active_pixels == active_pixels[0]).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
