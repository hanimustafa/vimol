import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mviewer
from mviewer import elements, kitty
from mviewer.bonds import ensure_bonds, perceive_bonds
from mviewer.render import Renderer, Style
from mviewer.scene import Scene
from mviewer.parsers import loads

EX = os.path.join(os.path.dirname(__file__), "..", "examples")

PDB_ETHANOL = """\
HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG A   1       1.520   0.000   0.000  1.00  0.00           C
HETATM    3  O1  LIG A   1       2.030   1.320   0.000  1.00  0.00           O
CONECT    1    2
CONECT    2    1    3
END
"""


def test_element_data():
    assert elements.symbol_to_z("C") == 6
    assert elements.normalize_symbol("fe") == "Fe"
    assert 0.6 < elements.covalent_radius("C") < 0.9
    assert len(elements.element_color("O")) == 3


def test_xyz_roundtrip_and_bonds():
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    assert mol.n_atoms == 12
    ensure_bonds(mol)
    # benzene: 6 ring bonds + 6 C-H = 12 bonds
    assert len(mol.bonds) == 12
    assert mol.formula() == "C6H6"


def test_c60_topology():
    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    assert mol.n_atoms == 60
    assert len(mol.bonds) == 90  # V - E + F = 2  =>  60 - 90 + 32 = 2


def test_pdb_conect():
    mol = loads(PDB_ETHANOL, "pdb")
    assert mol.symbols == ["C", "C", "O"]
    assert (0, 1, 1) in mol.bonds
    assert (1, 2, 1) in mol.bonds


def test_render_produces_image():
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 120, 120, supersample=1)
    img = scene.render()
    assert img.shape == (120, 120, 3)
    assert img.dtype == np.uint8
    # something other than the background must have been drawn
    bg = np.array(scene.style.background) * 255
    drawn = np.abs(img.astype(int) - bg.astype(int)).sum(axis=2) > 30
    assert drawn.sum() > 200


def test_transparent_render_is_rgba_with_cutout():
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 120, 120, style=Style(transparent=True), supersample=1)
    img = scene.render()
    assert img.shape == (120, 120, 4)
    # corners must be fully transparent, the molecule center opaque
    assert img[0, 0, 3] == 0
    assert img[60, 60, 3] == 255


def test_transparent_supersample_no_black_fringe():
    """Premultiplied downsampling: edge pixels must not fringe toward black."""
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 100, 100, style=Style(transparent=True), supersample=3)
    img = scene.render()
    assert img.shape == (100, 100, 4)
    # partially covered edge pixels exist and their (straight) color is not
    # dragged to black by the transparent background
    edge = (img[..., 3] > 20) & (img[..., 3] < 235)
    assert edge.sum() > 0
    assert img[..., :3][edge].max() > 60


def test_hydrogen_ball_bigger_than_bond():
    """Ball-and-stick must scale atoms by vdW radius so H stays visible."""
    from mviewer.render import _atom_radii
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    st = Style(representation="ball_and_stick")
    radii = _atom_radii(mol, st)
    h_idx = [i for i, s in enumerate(mol.symbols) if s == "H"]
    assert min(radii[i] for i in h_idx) > st.bond_radius  # H ball wider than the stick


def test_all_representations_render():
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    for rep in ("ball_and_stick", "spacefill", "licorice", "wireframe"):
        scene = Scene(mol, 80, 80, style=Style(representation=rep))
        img = scene.render()
        assert img.shape == (80, 80, 3)


def test_kitty_encoding_chunks():
    img = np.zeros((64, 64, 3), np.uint8)
    img[10:50, 10:50] = 200
    data = kitty.encode_image(img, image_id=7)
    assert data.startswith(b"\x1b_G")
    assert data.endswith(b"\x1b\\")
    assert b"i=7" in data
    assert b"a=T" in data
    # payload should be chunked with the graphics terminators
    assert data.count(b"\x1b_G") == data.count(b"\x1b\\")


def test_png_roundtrip_header():
    img = np.zeros((16, 16, 3), np.uint8)
    png = kitty.png_bytes(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in png[:32]
    assert png.rstrip().endswith(b"IEND".rjust(4)) or b"IEND" in png


def test_backend_auto_never_raises():
    """`backend="auto"` must silently fall back to CPU with zero GL deps
    installed -- the "never breaks the zero-dependency default" guarantee."""
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 80, 80, backend="auto")
    assert scene.backend in ("cpu", "gl")


def test_backend_cpu_explicit():
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 80, 80, backend="cpu")
    assert scene.backend == "cpu"
    img = scene.render()
    assert img.shape == (80, 80, 3)


def test_backend_invalid_name_raises():
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    with pytest.raises(ValueError):
        Scene(mol, 80, 80, backend="not-a-backend")


def test_backend_gl_explicit_raises_if_unavailable(monkeypatch):
    """An explicit `backend="gl"` request must not silently downgrade to
    CPU -- force the GL import to fail regardless of whether moderngl is
    actually installed in this environment, and assert it raises."""
    monkeypatch.setitem(sys.modules, "moderngl", None)
    monkeypatch.setitem(sys.modules, "mviewer.gl_render", None)
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    with pytest.raises(Exception):
        Scene(mol, 80, 80, backend="gl")


def test_resize_preserves_framing_and_rotation():
    """A plain window/terminal resize (Scene.set_size, refit=False) preserves
    the *apparent framing* -- the fraction of the viewport the molecule fills,
    plus any manual pan -- and the rotation, rather than the raw
    pixels-per-angstrom. Zoom and pan therefore scale by the min-dimension
    ratio (exactly as set_supersample does), keeping the molecule the same
    on-screen size across a resize. (It used to preserve raw zoom, which both
    changed the on-screen fraction on resize and froze the startup framing at
    an early, slightly-wrong terminal size -- see
    test_resize_self_heals_two_step_geometry.)"""
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 400, 300)
    scene.camera.orbit(30, 20)
    scene.camera.zoom_by(2.5)
    scene.camera.pan_by(15, -10)
    rot0 = scene.camera.rotation.copy()
    zoom0 = scene.camera.zoom
    pan0 = scene.camera.pan.copy()

    scene.set_size(500, 350)
    ratio = min(500, 350) / min(400, 300)

    assert np.array_equal(scene.camera.rotation, rot0)
    assert scene.camera.zoom == pytest.approx(zoom0 * ratio)
    assert np.allclose(scene.camera.pan, pan0 * ratio)

    # an explicit fit() (the 'f' key) must still re-fit to the extent
    scene.fit()
    assert scene.camera.zoom != pytest.approx(zoom0 * ratio)
    assert np.array_equal(scene.camera.pan, np.zeros(2))


def test_supersample_change_preserves_manual_zoom_and_pan():
    """set_supersample used to call fit(), which recomputes zoom purely from
    the molecule's extent -- silently discarding any scroll-to-zoom every
    time the interactive quality switch fired (fast while scrolling/dragging,
    crisp ~0.25s after stopping). Zoom/pan must instead rescale by the exact
    supersample ratio, preserving whatever the user had zoomed/panned to."""
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 400, 300)
    scene.set_supersample(2)
    base_zoom = scene.camera.zoom

    scene.set_supersample(1)              # interaction starts: fast quality
    scene.camera.zoom_by(1.12 ** 6)       # user scrolls to zoom in
    scene.camera.pan_by(20, -5)
    zoomed = scene.camera.zoom
    panned = scene.camera.pan.copy()

    scene.set_supersample(2)              # settle back to crisp quality
    assert scene.camera.zoom == pytest.approx(zoomed * 2)
    assert np.allclose(scene.camera.pan, panned * 2)

    scene.set_supersample(1)              # and back down again
    assert scene.camera.zoom == pytest.approx(zoomed)
    assert np.allclose(scene.camera.pan, panned)


def test_z_key_resets_view_like_r():
    """'z' is an alias for 'r': full reset of rotation, pan, and zoom."""
    from mviewer.widget import MoleculeWidget

    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 200, 200)
    fitted_zoom = w.scene.camera.zoom

    w.scene.camera.orbit(45, 30)
    w.scene.camera.zoom_by(3.0)
    w.scene.camera.pan_by(20, 10)

    assert w.handle_key("z") is True
    assert np.array_equal(w.scene.camera.rotation, np.eye(3))
    assert np.array_equal(w.scene.camera.pan, np.zeros(2))
    assert w.scene.camera.zoom == pytest.approx(fitted_zoom)


def test_set_size_refit_vs_preserve():
    """set_size(..., refit=True) is what Viewer uses the first time it learns
    the real terminal size (the widget starts at a 320x240 placeholder before
    that) -- it must fit fresh to the new size, not preserve the zoom that was
    fit for the placeholder. Later resizes (refit=False, the default) preserve
    the user's manual zoom by keeping its *apparent framing*: zoom scales by
    the min-dimension ratio so the molecule stays the same on-screen fraction."""
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 320, 240)  # placeholder-sized, as Viewer.__init__ does
    placeholder_zoom = scene.camera.zoom

    scene.set_size(1200, 800, refit=True)  # first real geometry
    fresh = Scene(mol, 1200, 800)
    assert scene.camera.zoom == pytest.approx(fresh.camera.zoom)
    assert scene.camera.zoom != pytest.approx(placeholder_zoom)

    scene.camera.zoom_by(2.0)               # user scrolls to zoom in
    zoomed = scene.camera.zoom
    scene.set_size(1300, 850)                # a later, genuine resize (refit=False default)
    ratio = min(1300, 850) / min(1200, 800)
    assert scene.camera.zoom == pytest.approx(zoomed * ratio)


def test_resize_self_heals_two_step_geometry():
    """Regression for the startup zoom bug: the viewer opened slightly zoomed
    and only 'z' (a fresh fit) corrected it. Root cause -- the host learns the
    real terminal size in two steps (an early, slightly-wrong report, then the
    settled size), and the second, non-refit resize used to freeze the molecule
    at the first size's zoom. Proportional rescaling makes a fit-derived zoom
    land exactly where a fresh fit for the settled size would, so it self-heals
    without any keypress."""
    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 320, 240)
    scene.set_size(900, 700, refit=True)     # early, slightly-wrong size
    scene.set_size(1200, 800)                # settled real size (refit=False default)
    fresh = Scene(mol, 1200, 800)
    assert scene.camera.zoom == pytest.approx(fresh.camera.zoom)


def test_camera_orbit_changes_view():
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 60, 60)
    before = scene.render().copy()
    scene.camera.orbit(90, 45)
    after = scene.render()
    assert not np.array_equal(before, after)


def test_add_vector_field_validates_length():
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    with pytest.raises(ValueError):
        mol.add_vector_field(np.zeros((mol.n_atoms - 1, 3)))


def test_vector_extent_accounts_for_arrow_tips():
    mol = mviewer.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    assert mol.vector_extent() == 0.0
    mol.add_vector_field(np.array([[5.0, 0.0, 0.0]]))
    assert mol.vector_extent() == pytest.approx(5.0)


def test_view_directions_rotates_but_does_not_translate():
    from mviewer.camera import Camera

    cam = Camera(center=np.array([3.0, -2.0, 1.0]))
    cam.orbit(40, 20)
    v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotated = cam.view_directions(v)
    assert np.allclose(rotated, v @ cam.rotation.T)
    # view_positions on the same numbers (treated as a position, not a free
    # vector) would incorrectly subtract the camera center first
    assert not np.allclose(cam.view_positions(v), rotated)


def test_fit_zooms_out_to_fit_long_vector():
    mol = mviewer.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    small_zoom = Scene(mol, 200, 200).camera.zoom
    mol.add_vector_field(np.array([[10.0, 0.0, 0.0]]))
    big_vector_zoom = Scene(mol, 200, 200).camera.zoom
    assert big_vector_zoom < small_zoom


def test_render_draws_arrow_in_its_assigned_color():
    """Color is the semantic key -- the arrow must render in the vector
    field's own color, not the parent atom's element color."""
    mol = mviewer.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    mol.add_vector_field(np.array([[2.0, 0.0, 0.0]]), color=(1.0, 0.0, 1.0),
                         radius=0.08, head_scale=3.0)
    scene = Scene(mol, 200, 200, backend="cpu", supersample=1)
    img = scene.render()
    cam = scene.camera
    ox_s = scene.render_size[0] * 0.5 + cam.pan[0]
    oy_s = scene.render_size[1] * 0.5 - cam.pan[1]
    tip_x = int(round(ox_s + 2.0 * cam.zoom))
    tip_y = int(round(oy_s))

    def is_magenta(px):
        return px[0] > 150 and px[2] > 100 and px[1] < 120

    region = img[max(tip_y - 4, 0):tip_y + 5, max(tip_x - 4, 0):tip_x + 5]
    magenta = (region[..., 0].astype(int) > 150) & (region[..., 2].astype(int) > 100) & \
              (region[..., 1].astype(int) < 120)
    assert magenta.any()
    center = img[int(round(oy_s)), int(round(ox_s))]
    assert not is_magenta(center)  # the atom itself keeps its element color


def test_input_decoder_keys_and_arrows():
    from mviewer.input import InputDecoder, KeyEvent, MouseEvent

    dec = InputDecoder(pixel=False)
    evs = dec.feed(b"a\x1b[C")  # 'a' then right-arrow
    assert isinstance(evs[0], KeyEvent) and evs[0].key == "a"
    assert isinstance(evs[1], KeyEvent) and evs[1].key == "right"
    # a lone ESC only resolves on flush (ambiguous until then)
    assert dec.feed(b"\x1b") == []
    assert dec.flush() == [KeyEvent("escape")]


def test_input_decoder_split_sequence():
    """An escape sequence split across two feeds must still decode once."""
    from mviewer.input import InputDecoder, MouseEvent

    dec = InputDecoder(pixel=True)
    assert dec.feed(b"\x1b[<0;100;2") == []  # incomplete: buffered
    evs = dec.feed(b"00M")
    assert len(evs) == 1
    ev = evs[0]
    assert isinstance(ev, MouseEvent) and ev.action == "down"
    assert ev.pixel and ev.x == 100 and ev.y == 200  # pixel coords, not 1-based cells


def test_input_decoder_mouse_actions():
    from mviewer.input import InputDecoder, MouseEvent

    dec = InputDecoder(pixel=False)
    (down,) = dec.feed(b"\x1b[<0;5;5M")
    assert down.action == "down" and down.button == 0
    (drag,) = dec.feed(b"\x1b[<32;9;9M")   # motion bit + button 0
    assert drag.action == "drag" and drag.button == 0
    (move,) = dec.feed(b"\x1b[<35;9;9M")   # motion bit + no button (low bits 3)
    assert move.action == "move" and move.button is None
    (up,) = dec.feed(b"\x1b[<0;9;9m")
    assert up.action == "up"
    (scroll,) = dec.feed(b"\x1b[<64;5;5M")
    assert scroll.action == "scroll" and scroll.scroll == "up"


def test_widget_mouse_rotate_pan_zoom():
    from mviewer.widget import MoleculeWidget
    from mviewer.input import MouseEvent

    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 200, 200, supersample=1)

    r0 = w.scene.camera.rotation.copy()
    w.handle_mouse(MouseEvent("down", 100, 100, button=0, pixel=True))
    assert w.handle_mouse(MouseEvent("drag", 140, 110, button=0, pixel=True))
    assert not np.array_equal(r0, w.scene.camera.rotation)  # rotated

    p0 = w.scene.camera.pan.copy()
    w.handle_mouse(MouseEvent("down", 100, 100, button=2, pixel=True))  # right = pan
    w.handle_mouse(MouseEvent("drag", 130, 120, button=2, pixel=True))
    assert not np.array_equal(p0, w.scene.camera.pan)

    z0 = w.scene.camera.zoom
    w.handle_mouse(MouseEvent("scroll", 100, 100, scroll="up", pixel=True))
    assert w.scene.camera.zoom > z0


def test_widget_pick_center_atom():
    """Hovering the projected center of an atom should pick that atom."""
    from mviewer.widget import MoleculeWidget

    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 200, 200, supersample=1)
    cam = w.scene.camera
    Wr, Hr = w.scene.render_size
    v = cam.view_positions(mol.positions)
    sz = v[:, 2]
    front = int(np.argmax(sz))  # front-most atom is unambiguous to pick
    sx = Wr * 0.5 + cam.pan[0] + v[front, 0] * cam.zoom
    sy = Hr * 0.5 - cam.pan[1] - v[front, 1] * cam.zoom
    assert w.pick(sx / w.scene.supersample, sy / w.scene.supersample) == front
    # clicking empty corner picks nothing
    assert w.pick(1, 1) is None


def test_widget_hover_highlight_changes_render():
    from mviewer.widget import MoleculeWidget

    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 120, 120)
    plain = w.render().copy()
    w.hovered = 0
    assert not np.array_equal(plain, w.render())


def test_handle_event_reports_change():
    """handle_event must return whether the view changed -- the interactive
    loop gates redraws on this, so a wrong return means either no redraw on
    input or a redraw every idle frame (terminal flood)."""
    from mviewer.widget import MoleculeWidget
    from mviewer.input import KeyEvent

    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 120, 120)
    assert w.handle_event(KeyEvent("left")) is True     # rotate
    assert w.handle_event(KeyEvent("right")) is True
    assert w.handle_event(KeyEvent("q")) is False        # unbound in the widget (driver-level quit key)


def test_viewer_only_redraws_on_change(tmp_path):
    """An idle viewer must not redraw every loop iteration (that floods the
    terminal with full-frame images); input and the post-settle quality bump
    must still trigger exactly one redraw each."""
    from mviewer.viewer import Viewer
    from mviewer.input import KeyEvent
    import time

    mol = mviewer.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._update_geometry()
        count = {"n": 0}
        orig = v._draw
        v._draw = lambda: (count.__setitem__("n", count["n"] + 1), orig())[1]

        def loop_iter(events):
            changed = v._dispatch(events)
            if v._target_ss() != v._drawn_ss:
                changed = True
            if changed:
                v._draw()

        v._draw()  # initial frame

        count["n"] = 0
        loop_iter([KeyEvent("left")])
        assert count["n"] == 1                      # input -> redraw

        count["n"] = 0
        time.sleep(0.3)                             # cross the 0.25s settle
        for _ in range(20):
            loop_iter([])
        assert count["n"] == 1                      # exactly one crisp bump

        count["n"] = 0
        for _ in range(20):
            loop_iter([])
        assert count["n"] == 0                      # then fully idle -> no draws
    finally:
        os.close(fd)


def test_mouse_enable_sequences():
    from mviewer.input import enable_mouse
    seq = enable_mouse(pixel=True, hover=True)
    assert b"1003" in seq and b"1006" in seq and b"1016" in seq
    seq2 = enable_mouse(pixel=False, hover=False)
    assert b"1002" in seq2 and b"1016" not in seq2


def test_viewer_draw_writes_bytes(tmp_path):
    """_draw should emit Kitty bytes to the output fd."""
    from mviewer.viewer import Viewer

    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    out = tmp_path / "out.bin"
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._update_geometry()
        v._draw()
    finally:
        os.close(fd)
    data = out.read_bytes()
    assert b"\x1b_G" in data  # a graphics command was written


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
