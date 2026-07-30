import base64
import os
import re
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vimol
from vimol import elements, kitty
from vimol.bonds import ensure_bonds, perceive_bonds
from vimol.render import Renderer, Style
from vimol.scene import Scene
from vimol.parsers import loads

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


def test_themed_base_colors_dark_is_passthrough():
    from vimol import elements as els
    base = np.array([[1.0, 1.0, 1.0], [0.35, 0.35, 0.38]])
    out = els.themed_base_colors(["H", "C"], base, "dark")
    assert out is base   # no copy -- byte-identical to today


def test_themed_base_colors_light_overrides_unreadable_entries():
    from vimol import elements as els
    base = np.array([[1.0, 1.0, 1.0], [0.35, 0.35, 0.38]])   # H, C
    out = els.themed_base_colors(["H", "C"], base, "light")
    assert not np.allclose(out[0], [1.0, 1.0, 1.0])   # H moved off pure white
    assert np.allclose(out[1], [0.35, 0.35, 0.38])    # C untouched
    assert base[0].tolist() == [1.0, 1.0, 1.0]         # original array untouched


def test_themed_base_colors_light_covers_all_override_entries():
    from vimol import elements as els
    syms = ["H", "He", "Ag", "Pt", "Hg", "C"]
    base = np.array([els.element_color(s) for s in syms])
    out = els.themed_base_colors(syms, base, "light")
    for i, sym in enumerate(syms[:-1]):   # every override entry changed
        assert not np.allclose(out[i], base[i]), sym
    assert np.allclose(out[-1], base[-1])  # carbon, not in the override table


def test_theme_dark_matches_original_constants():
    from vimol import theme
    assert theme.DARK.panel_bg == (30, 33, 44)
    assert theme.DARK.panel_fg == (230, 232, 240)
    assert theme.DARK.list_active_bg == (37, 45, 64)
    assert theme.DARK.pt_border_fg == (60, 200, 180)
    assert theme.DARK.cleanup_hint_fg == (255, 170, 60)


def test_theme_luminance():
    from vimol import theme
    assert theme.luminance((255, 255, 255)) == pytest.approx(255.0)
    assert theme.luminance((0, 0, 0)) == 0.0
    assert theme.luminance((255, 170, 60)) == pytest.approx(
        0.299 * 255 + 0.587 * 170 + 0.114 * 60)


def test_theme_from_colorfgbg():
    from vimol import theme
    assert theme.from_colorfgbg("15;0") is theme.DARK
    assert theme.from_colorfgbg("0;15") is theme.LIGHT
    assert theme.from_colorfgbg("0;7") is theme.LIGHT
    assert theme.from_colorfgbg("15;1") is theme.DARK
    assert theme.from_colorfgbg("garbage") is None
    assert theme.from_colorfgbg("") is None


def test_theme_resolve_precedence():
    from vimol import theme
    # explicit wins over everything
    assert theme.resolve("light", (10, 10, 10), "15;0") is theme.LIGHT
    assert theme.resolve("dark", (250, 250, 250), "0;15") is theme.DARK
    # osc11 wins over colorfgbg
    assert theme.resolve(None, (245, 245, 245), "15;0") is theme.LIGHT
    assert theme.resolve(None, (10, 10, 10), "0;15") is theme.DARK
    # colorfgbg is the fallback when osc11 is unknown
    assert theme.resolve(None, None, "0;15") is theme.LIGHT
    # default is dark
    assert theme.resolve(None, None, None) is theme.DARK
    assert theme.resolve(None, None, "garbage") is theme.DARK


def test_viewer_defaults_to_dark_theme(monkeypatch):
    from vimol.viewer import Viewer, theme as viewer_theme
    # Task 7 hardcodes theme.DARK, but Task 8 makes this env-dependent --
    # clear both now so this test stays valid (not flaky under whatever the
    # running shell happens to export) once that lands.
    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    assert v.theme is viewer_theme.DARK


def test_viewer_strip_rows_always_carry_an_explicit_background(tmp_path):
    """Every structure-strip row -- not just active/cursor -- must paint a
    background SGR, else its foreground (tuned for one theme) washes out
    against whichever background the terminal itself has (the reported
    bug). Structure #2 (index 1) is neither active (0) nor cursor-highlighted
    here, so its row is exactly the "ordinary" case that used to fall
    through to bg=None."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 24
        v._list_cursor = 0   # keep the cursor off row 1 too
        data = v._draw_list().decode("utf-8", "replace")
        parts = re.split(r"\x1b\[(\d+);1H", data)
        rows = {int(parts[i]) - 1: parts[i + 1] for i in range(1, len(parts), 2)}
        ordinary_row = rows[v._list_row_spans[1][0]]
        r, g, b = v.theme.list_panel_bg
        assert f"\x1b[48;2;{r};{g};{b}m" in ordinary_row
    finally:
        os.close(fd)


def test_viewer_status_bar_uses_theme_panel_colors():
    from vimol.viewer import Viewer
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    v._update_geometry()
    bar = v._status_bar()
    r, g, b = v.theme.panel_bg
    assert f"\x1b[48;2;{r};{g};{b}m" in bar


def test_xyz_roundtrip_and_bonds():
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
    assert mol.n_atoms == 12
    ensure_bonds(mol)
    # benzene: 6 ring bonds + 6 C-H = 12 bonds
    assert len(mol.bonds) == 12
    assert mol.formula() == "C6H6"


def test_xyz_keeps_full_comment_past_60_chars():
    from vimol.parsers import xyz as xyz_parser
    long_comment = "SCF Energy = -76.123456789012 Hartree, converged in 42 cycles, RMS grad 1e-9"
    assert len(long_comment) > 60
    text = f"1\n{long_comment}\nO 0.0 0.0 0.0\n"
    mols = xyz_parser.parse(text)
    assert mols[0].name == long_comment
    # round-trips through dumps()/parse() unchanged
    again = xyz_parser.parse(xyz_parser.dumps(mols[0]))
    assert again[0].name == long_comment


def test_c60_topology():
    mol = vimol.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    assert mol.n_atoms == 60
    assert len(mol.bonds) == 90  # V - E + F = 2  =>  60 - 90 + 32 = 2


def test_pdb_conect():
    mol = loads(PDB_ETHANOL, "pdb")
    assert mol.symbols == ["C", "C", "O"]
    assert (0, 1, 1) in mol.bonds
    assert (1, 2, 1) in mol.bonds


def test_render_produces_image():
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
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
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 120, 120, style=Style(transparent=True), supersample=1)
    img = scene.render()
    assert img.shape == (120, 120, 4)
    # corners must be fully transparent, the molecule center opaque
    assert img[0, 0, 3] == 0
    assert img[60, 60, 3] == 255


def test_transparent_supersample_no_black_fringe():
    """Premultiplied downsampling: edge pixels must not fringe toward black."""
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
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
    from vimol.render import _atom_radii
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    st = Style(representation="ball_and_stick")
    radii = _atom_radii(mol, st)
    h_idx = [i for i, s in enumerate(mol.symbols) if s == "H"]
    assert min(radii[i] for i in h_idx) > st.bond_radius  # H ball wider than the stick


def test_all_representations_render():
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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


def test_kitty_shm_transmission_sends_a_name_not_the_pixels():
    """`transmit="shm"` hands the terminal a shared-memory *name* (protocol
    t=s, the "local client" path) instead of zlib+base64 pixel data -- the
    whole point being that a full-resolution frame costs one memcpy."""
    from multiprocessing import shared_memory
    img = np.zeros((32, 48, 4), np.uint8)
    img[4:20, 4:20] = 200
    data = kitty.encode_image(img, image_id=7, transmit="shm")
    assert b"t=s" in data
    assert b"o=z" not in data                     # nothing compressed
    assert b"f=32" in data and b"s=48" in data and b"v=32" in data
    assert len(data) < 200                        # a name, not 6 KB of pixels
    name = base64.standard_b64decode(data.split(b";", 1)[1].split(b"\x1b\\")[0])
    assert name.startswith(b"/")                  # the shm_open name kitty opens
    shm = shared_memory.SharedMemory(name=name.decode()[1:])
    try:
        assert bytes(shm.buf[:img.nbytes]) == img.tobytes()
    finally:
        shm.close()
        shm.unlink()


def test_probe_query_bytes_includes_osc11():
    data = kitty.probe_query_bytes()
    assert b"\x1b]11;?\x1b\\" in data


def test_parse_probe_reply_extracts_osc11_background_st_terminated():
    buf = (b"\x1b_Gi=31;OK\x1b\\"
           b"\x1b[?1016;2$y"
           b"\x1b]11;rgb:1e1e/2020/2828\x1b\\"
           b"\x1b[6;18;9t"
           b"\x1b[?62c")
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.bg_rgb == (0x1e, 0x20, 0x28)


def test_parse_probe_reply_extracts_osc11_background_bel_terminated_two_digit():
    buf = (b"\x1b_Gi=31;OK\x1b\\"
           b"\x1b]11;rgb:f0/f2/f5\x07"
           b"\x1b[?62c")
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.bg_rgb == (0xf0, 0xf2, 0xf5)


def test_parse_probe_reply_bg_rgb_none_when_terminal_silent():
    buf = b"\x1b[?62c"   # only the DA1 fence answered
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.bg_rgb is None


def test_parse_probe_reply_bg_rgb_none_before_da1_fence():
    # no DA1 yet -> whole probe is None, not a premature verdict
    buf = b"\x1b]11;rgb:1e1e/2020/2828\x1b\\"
    assert kitty.parse_probe_reply(buf) is None


def test_osc52_copy_encodes_base64_clipboard_sequence():
    seq = kitty.osc52_copy("hello")
    assert seq == b"\x1b]52;c;" + base64.standard_b64encode(b"hello") + b"\x1b\\"


def test_osc52_copy_handles_unicode_and_padding_edge_cases():
    for text in ("d(#1-#2) = 1.523 Å", "a", "ab", "abc", ""):
        seq = kitty.osc52_copy(text)
        assert seq.startswith(b"\x1b]52;c;")
        assert seq.endswith(b"\x1b\\")
        payload = seq[len(b"\x1b]52;c;"):-len(b"\x1b\\")]
        assert base64.standard_b64decode(payload) == text.encode("utf-8")


def test_png_roundtrip_header():
    img = np.zeros((16, 16, 3), np.uint8)
    png = kitty.png_bytes(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in png[:32]
    assert png.rstrip().endswith(b"IEND".rjust(4)) or b"IEND" in png


def test_backend_auto_never_raises():
    """`backend="auto"` must silently fall back to CPU with zero GL deps
    installed -- the "never breaks the zero-dependency default" guarantee."""
    mol = vimol.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 80, 80, backend="auto")
    assert scene.backend in ("cpu", "gl")


def test_backend_cpu_explicit():
    mol = vimol.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 80, 80, backend="cpu")
    assert scene.backend == "cpu"
    img = scene.render()
    assert img.shape == (80, 80, 3)


def test_backend_invalid_name_raises():
    mol = vimol.load(os.path.join(EX, "water.xyz"))
    with pytest.raises(ValueError):
        Scene(mol, 80, 80, backend="not-a-backend")


def test_backend_gl_explicit_raises_if_unavailable(monkeypatch):
    """An explicit `backend="gl"` request must not silently downgrade to
    CPU -- force the GL import to fail regardless of whether moderngl is
    actually installed in this environment, and assert it raises."""
    monkeypatch.setitem(sys.modules, "moderngl", None)
    monkeypatch.setitem(sys.modules, "vimol.gl_render", None)
    mol = vimol.load(os.path.join(EX, "water.xyz"))
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
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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
    from vimol.widget import MoleculeWidget

    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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
    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 320, 240)
    scene.set_size(900, 700, refit=True)     # early, slightly-wrong size
    scene.set_size(1200, 800)                # settled real size (refit=False default)
    fresh = Scene(mol, 1200, 800)
    assert scene.camera.zoom == pytest.approx(fresh.camera.zoom)


def test_camera_orbit_changes_view():
    mol = vimol.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 60, 60)
    before = scene.render().copy()
    scene.camera.orbit(90, 45)
    after = scene.render()
    assert not np.array_equal(before, after)


def test_add_vector_field_validates_length():
    mol = vimol.load(os.path.join(EX, "water.xyz"))
    with pytest.raises(ValueError):
        mol.add_vector_field(np.zeros((mol.n_atoms - 1, 3)))


def test_vector_extent_accounts_for_arrow_tips():
    mol = vimol.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    assert mol.vector_extent() == 0.0
    mol.add_vector_field(np.array([[5.0, 0.0, 0.0]]), radius=0.05, head_scale=2.5)
    # tip at 5.0 + a head-radius pad (0.05 * 2.5) so a fat arrowhead at the
    # scene edge isn't clipped by auto-fit
    assert mol.vector_extent() == pytest.approx(5.0 + 0.05 * 2.5)


def test_vector_extent_survives_stale_field_after_add_atom():
    """A vector field attached before more atoms are added goes stale
    (its (N,3) no longer matches n_atoms). vector_extent() and the render
    path must skip it, not raise a broadcast error mid-render."""
    mol = vimol.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    mol.add_vector_field(np.array([[2.0, 0.0, 0.0]]))
    mol.add_atom("O", 1.2, 0.0, 0.0)   # now 2 atoms, field still has 1 row
    assert mol.vector_extent() == 0.0   # stale field skipped, no other fields
    # a full render must not raise
    img = Scene(mol, 80, 80, backend="cpu", supersample=1).render()
    assert img.shape == (80, 80, 3)


def test_view_directions_rotates_but_does_not_translate():
    from vimol.camera import Camera

    cam = Camera(center=np.array([3.0, -2.0, 1.0]))
    cam.orbit(40, 20)
    v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotated = cam.view_directions(v)
    assert np.allclose(rotated, v @ cam.rotation.T)
    # view_positions on the same numbers (treated as a position, not a free
    # vector) would incorrectly subtract the camera center first
    assert not np.allclose(cam.view_positions(v), rotated)


def test_fit_zooms_out_to_fit_long_vector():
    mol = vimol.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    small_zoom = Scene(mol, 200, 200).camera.zoom
    mol.add_vector_field(np.array([[10.0, 0.0, 0.0]]))
    big_vector_zoom = Scene(mol, 200, 200).camera.zoom
    assert big_vector_zoom < small_zoom


def test_render_draws_arrow_in_its_assigned_color():
    """Color is the semantic key -- the arrow must render in the vector
    field's own color, not the parent atom's element color."""
    mol = vimol.Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
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
    from vimol.input import InputDecoder, KeyEvent, MouseEvent

    dec = InputDecoder(pixel=False)
    evs = dec.feed(b"a\x1b[C")  # 'a' then right-arrow
    assert isinstance(evs[0], KeyEvent) and evs[0].key == "a"
    assert isinstance(evs[1], KeyEvent) and evs[1].key == "right"
    # a lone ESC only resolves on flush (ambiguous until then)
    assert dec.feed(b"\x1b") == []
    assert dec.flush() == [KeyEvent("escape")]


def test_input_decoder_alt_arrows():
    """Alt/Option+Up/Down carry a modifier param; plain arrows must not."""
    from vimol.input import InputDecoder, KeyEvent

    dec = InputDecoder(pixel=False)
    assert dec.feed(b"\x1b[A") == [KeyEvent("up")]          # unmodified: unaffected
    assert dec.feed(b"\x1b[1;3A") == [KeyEvent("alt+up")]   # xterm "Alt" modifier (3)
    assert dec.feed(b"\x1b[1;9B") == [KeyEvent("alt+down")]  # "Meta" modifier (9), treated as alt
    assert dec.feed(b"\x1b[1;2A") == [KeyEvent("up")]        # Shift alone: not alt


def test_input_decoder_split_sequence():
    """An escape sequence split across two feeds must still decode once."""
    from vimol.input import InputDecoder, MouseEvent

    dec = InputDecoder(pixel=True)
    assert dec.feed(b"\x1b[<0;100;2") == []  # incomplete: buffered
    evs = dec.feed(b"00M")
    assert len(evs) == 1
    ev = evs[0]
    assert isinstance(ev, MouseEvent) and ev.action == "down"
    assert ev.pixel and ev.x == 100 and ev.y == 200  # pixel coords, not 1-based cells


def test_input_decoder_mouse_actions():
    from vimol.input import InputDecoder, MouseEvent

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
    from vimol.widget import MoleculeWidget
    from vimol.input import MouseEvent

    mol = vimol.load(os.path.join(EX, "c60.xyz"))
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
    from vimol.widget import MoleculeWidget

    mol = vimol.load(os.path.join(EX, "c60.xyz"))
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
    from vimol.widget import MoleculeWidget

    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 120, 120)
    plain = w.render().copy()
    w.hovered = 0
    assert not np.array_equal(plain, w.render())


def test_widget_defaults_to_dark_theme_and_is_passthrough():
    from vimol.widget import MoleculeWidget
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    w = MoleculeWidget(mol)
    assert w.theme == "dark"
    w.render()
    assert w.style.color_override is None   # no hover/theme override active


def test_widget_light_theme_sets_color_override_with_no_hover():
    from vimol.widget import MoleculeWidget
    mol = vimol.load(os.path.join(EX, "methane.xyz"))   # C + 4 H
    w = MoleculeWidget(mol)
    w.theme = "light"
    w.render()
    assert w.style.color_override is not None
    h_idx = mol.symbols.index("H")
    assert not np.allclose(w.style.color_override[h_idx], [1.0, 1.0, 1.0])


def test_widget_light_theme_hover_tint_applies_on_top_of_override():
    from vimol.widget import MoleculeWidget
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    w = MoleculeWidget(mol)
    w.theme = "light"
    h_idx = mol.symbols.index("H")
    w.hovered = h_idx
    w.render()
    cols = w.style.color_override
    assert cols is not None
    # hovered atom gets the yellow hover tint, not the plain light override
    assert cols[h_idx][0] > 0.5 and cols[h_idx][1] > 0.5   # warm/bright, not the muted grey override


def test_handle_event_reports_change():
    """handle_event must return whether the view changed -- the interactive
    loop gates redraws on this, so a wrong return means either no redraw on
    input or a redraw every idle frame (terminal flood)."""
    from vimol.widget import MoleculeWidget
    from vimol.input import KeyEvent

    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
    ensure_bonds(mol)
    w = MoleculeWidget(mol, 120, 120)
    assert w.handle_event(KeyEvent("left")) is True     # rotate
    assert w.handle_event(KeyEvent("right")) is True
    assert w.handle_event(KeyEvent("q")) is False        # unbound in the widget (driver-level quit key)


def test_viewer_only_redraws_on_change(tmp_path):
    """An idle viewer must not redraw every loop iteration (that floods the
    terminal with full-frame images); input and the post-settle quality bump
    must still trigger exactly one redraw each."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent
    import time

    mol = vimol.load(os.path.join(EX, "benzene.xyz"))
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
    from vimol.input import enable_mouse
    seq = enable_mouse(pixel=True, hover=True)
    assert b"1003" in seq and b"1006" in seq and b"1016" in seq
    seq2 = enable_mouse(pixel=False, hover=False)
    assert b"1002" in seq2 and b"1016" not in seq2


def test_viewer_draw_writes_bytes(tmp_path):
    """_draw should emit Kitty bytes to the output fd."""
    from vimol.viewer import Viewer

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
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


def test_viewer_draw_multi_structure_writes_image_and_list_strip(tmp_path):
    """_draw() on a multi-structure viewer -- the real per-frame path, not
    just _draw_list()/_status_bar() called directly -- must emit both the
    Kitty image and the structure-list strip in one frame."""
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    out = tmp_path / "out.bin"
    fd = os.open(str(out), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v._update_geometry()
        v._draw()
    finally:
        os.close(fd)
    data = out.read_bytes()
    assert b"\x1b_G" in data                 # a graphics command was written
    assert b"STRUCTURES 3" in data           # the list strip's header


def test_viewer_multi_frame_cycling_via_keys(tmp_path):
    """Multi-frame files no longer show the old "struc N/total" status-bar
    pill (design §2/§4.1: it becomes the structure-list strip's header);
    n/p and opt+up/down still step the active structure by plain keys."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v._update_geometry()

        bar = v._status_bar()
        assert "struc" not in bar

        assert v._dispatch([KeyEvent("alt+down")]) is True
        assert v.frame_index == 1
        assert v._dispatch([KeyEvent("alt+up")]) is True
        assert v.frame_index == 0
        assert v._dispatch([KeyEvent("n")]) is True
        assert v.frame_index == 1
    finally:
        os.close(fd)


def test_viewer_multi_frame_click_row_switches_active(tmp_path):
    """Clicking a structure-list row (design §4.2/§4.3) replaces the active
    structure, mirroring what the retired status-bar pill used to do."""
    from vimol.viewer import Viewer
    from vimol.input import MouseEvent

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v._update_geometry()
        v._draw_list()   # populate _list_row_spans for the current geometry
        row0, col_start, _col_end = v._list_row_spans[2]   # third row -> benzene
        click = MouseEvent("down", float(col_start), float(row0), button=0)
        assert v._dispatch([click]) is True
        assert v.frame_index == 2
    finally:
        os.close(fd)


def test_viewer_single_frame_has_no_strip_or_list_zone(tmp_path):
    """A single-structure file shows no list strip at all -- no columns
    reserved, no "struc" pill, no list zone to swallow clicks."""
    from vimol.viewer import Viewer

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._update_geometry()
        bar = v._status_bar()
        assert "struc" not in bar
        assert v._list_w == 0
        assert not v._in_list_zone(0)
    finally:
        os.close(fd)


def test_viewer_multi_model_file_gets_hash_k_labels(tmp_path):
    """A multi-model file collapses to one StructureSet entry per model,
    labelled '<basename>#k', 1-based (design §1, §2)."""
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, source_path="traj.xyz", fd_out=fd)
        assert v.structures.labels == ["traj.xyz#1", "traj.xyz#2", "traj.xyz#3"]
        assert len(v.structures) == 3
    finally:
        os.close(fd)


def test_viewer_frame_index_setter_refreshes_widget(tmp_path):
    """`viewer.frame_index = idx` (app.py's --frame path) must actually swap
    what the widget renders -- not just update a number nobody reads."""
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v.frame_index = 2
        assert v.frame_index == 2
        assert v.widget.molecule is v.structures[2].molecule
        assert v.widget.scene.molecule is v.structures[2].molecule
    finally:
        os.close(fd)


def test_viewer_cycle_frame_preserves_the_whole_structure_set(tmp_path):
    """Cycling frames must NOT collapse the StructureSet down to one entry --
    that was Scene.set_molecule's old job, which the design redefines as
    'replace with a single entry' specifically so cycling must avoid it."""
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v._cycle_frame(1)
        assert len(v.structures) == 3
        assert v.frame_index == 1
        assert v.widget.molecule is v.structures[1].molecule
    finally:
        os.close(fd)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# -- structure list strip (design §4) --------------------------------------

def test_viewer_reserves_list_columns_when_multi_structure(tmp_path):
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v._update_geometry()
        assert v._cols == 80  # environment default, no COLUMNS/LINES set
        list_w = min(28, max(18, v._cols // 5))
        assert v._img_cols == v._cols - list_w
        assert v._img_origin_px == (list_w * 9.0, 0)
    finally:
        os.close(fd)


def test_viewer_single_structure_reserves_no_list_columns(tmp_path):
    from vimol.viewer import Viewer

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._update_geometry()
        assert v._img_cols == v._cols
        assert v._img_origin_px == (0, 0)
    finally:
        os.close(fd)


def _multi_viewer(tmp_path, fd_name="out.bin", **kw):
    from vimol.viewer import Viewer
    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / fd_name), os.O_WRONLY | os.O_CREAT, 0o644)
    v = Viewer(frames[0], frames=frames, fd_out=fd, **kw)
    v._update_geometry()
    return v, fd


def test_viewer_list_shows_header_and_one_row_per_structure(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 40   # wide enough that labels aren't middle-truncated
        text = v._draw_list().decode("utf-8", "replace")
        assert "STRUCTURES 3" in text
        for entry in v.structures:
            assert entry.label in text
        assert len(v._list_row_spans) == 3
    finally:
        os.close(fd)


def test_viewer_list_row_body_carries_no_atom_count(tmp_path):
    """The atom-count column is gone from the row body (its width goes to
    the label instead) -- a 60-atom structure's row must not show '60'."""
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "c60.xyz")),
              vimol.load(os.path.join(EX, "water.xyz"))]
    fd = os.open(str(tmp_path / "counts.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, fd_out=fd)
        v._update_geometry()
        # the label itself would otherwise contribute the digits we look for
        v.structures[1].label = "buckyball"
        v._list_w = 40
        # printable text only: RGB colour parameters are full of digits
        text = _visible(v._draw_list().decode("utf-8", "replace"))
        assert "buckyball" in text
        assert str(frames[1].n_atoms) == "60"
        assert "60" not in text
    finally:
        os.close(fd)


def _traj_viewer(tmp_path, n=3, path="traj.xyz", fd_name="traj.bin"):
    """A viewer whose structures all come from one multi-model file."""
    from vimol.viewer import Viewer
    frames = [vimol.load(os.path.join(EX, "methane.xyz")) for _ in range(n)]
    fd = os.open(str(tmp_path / fd_name), os.O_WRONLY | os.O_CREAT, 0o644)
    v = Viewer(frames[0], frames=frames, source_path=path, fd_out=fd)
    v._update_geometry()
    return v, fd


def test_viewer_list_groups_a_multi_model_file_under_one_header(tmp_path):
    """A file contributing more than one structure shows its basename ONCE as
    a group header, with its structures listed beneath as 'frame N' -- not the
    filename repeated once per model (design §4.1)."""
    v, fd = _traj_viewer(tmp_path, n=4)
    try:
        v._list_w = 40
        text = v._draw_list().decode("utf-8", "replace")
        assert text.count("traj.xyz") == 1        # the header, and only there
        for k in range(1, 5):
            assert f"frame {k}" in text
        assert "traj.xyz#1" not in text
        # only the structure rows are clickable; the header is not
        assert len(v._list_row_spans) == 4
    finally:
        os.close(fd)


def test_viewer_list_single_structure_files_get_no_group_header(tmp_path):
    """A file contributing exactly one structure gets no header row -- just
    its own row, labelled with the basename."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol_a = vimol.load(os.path.join(EX, "methane.xyz"))
    mol_b = vimol.load(os.path.join(EX, "water.xyz"))
    sset.append(mol_a, label="methane.xyz", path="/data/methane.xyz")
    sset.append(mol_b, label="water.xyz", path="/data/water.xyz")
    fd = os.open(str(tmp_path / "singles.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol_a, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 40
        text = v._draw_list().decode("utf-8", "replace")
        assert "methane.xyz" in text and "water.xyz" in text
        assert "frame " not in text
        assert "/data/" not in text               # basenames only
        assert len(v._list_row_spans) == 2
        # two rows, two structures, and no group header anywhere
        rows = v._list_display_rows()
        assert [kind for kind, _i, _t in rows] == ["struct", "struct"]
    finally:
        os.close(fd)


def test_viewer_list_mixed_tree_keeps_rows_and_structures_aligned(tmp_path):
    """A grouped file followed by a lone one: the offset between display rows
    and structure indices CHANGES partway down the list, which is exactly
    where row arithmetic drifts. Clicks and 1-9 must still address
    structures."""
    from vimol.input import KeyEvent, MouseEvent
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    other = vimol.load(os.path.join(EX, "water.xyz"))
    sset = StructureSet()
    for k in range(3):
        sset.append(mol, label=f"traj.xyz#{k + 1}", path="/d/traj.xyz")
    sset.append(other, label="apo.pdb", path="/d/apo.pdb")
    fd = os.open(str(tmp_path / "mixed.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        rows = v._list_display_rows()
        assert [kind for kind, _i, _t in rows] == [
            "group", "struct", "struct", "struct", "struct"]
        assert rows[-1][1:] == (3, "apo.pdb")     # lone file: no 'frame 4'
        assert rows[1][2] == "frame 1"

        v._draw_list()
        assert v._list_row_struct == [0, 1, 2, 3]
        row0, col_start, _c1 = v._list_row_spans[3]
        assert v._list_index_at_row(row0) == 3
        assert v._dispatch([MouseEvent("down", float(col_start), float(row0),
                                       button=0)]) is True
        assert v.frame_index == 3

        # 1-9 addresses structures, not display rows
        v._list_focused = True
        assert v._dispatch([KeyEvent("3")]) is True
        assert v._list_cursor == 2
        v._draw_list()
        assert 2 in v._list_row_struct
    finally:
        os.close(fd)


def test_viewer_list_group_header_row_is_not_selectable(tmp_path):
    """Clicking a group header must not activate a structure or crash, and
    the row->structure mapping must survive the extra display row."""
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=3)
    try:
        v._draw_list()
        rows = v._list_display_rows()
        assert rows[0][0] == "group"
        header_row = v._list_row_spans[0][0] - 1   # sits above the first frame
        assert v._list_index_at_row(header_row) is None
        click = MouseEvent("down", 0.0, float(header_row), button=0)
        v._dispatch([click])               # must not raise
        assert v.frame_index == 0

        # a click on the LAST structure row still lands on structure 2
        row0, col_start, _c1 = v._list_row_spans[2]
        assert v._list_index_at_row(row0) == 2
        assert v._dispatch([MouseEvent("down", float(col_start), float(row0),
                                       button=0)]) is True
        assert v.frame_index == 2
    finally:
        os.close(fd)


def test_viewer_list_truncates_long_labels_keeping_extension(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v.structures[0].label = "a_very_long_trajectory_name_here.xyz#1"
        v._list_w = 18
        text = v._draw_list().decode("utf-8", "replace")
        assert "…" in text
        assert "z#1" in text   # the tail (extension/frame suffix) stays legible (design §4.1)
    finally:
        os.close(fd)


def test_viewer_list_active_and_marked_row_markers(tmp_path):
    """The active and marked rows are still told apart at a glance -- by
    colour, not by the old ▸/✓ leaders (design §4.1): the active row is a
    full-width background, a marked row wears its own tint."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v.structures.toggle_mark(2)
        rows = _strip_rows(v)
        active = rows[v._list_row_spans[0][0]]
        marked = rows[v._list_row_spans[2][0]]
        assert _sgr_bg((37, 45, 64)) in active
        assert _sgr_bg((37, 45, 64)) not in marked
        tint = tuple(int(c * 255) for c in v.structures[2].tint)
        assert marked.count(_sgr_fg(tint)) == 2      # swatch and label
    finally:
        os.close(fd)


def test_viewer_list_footer_overlay_and_camera_shared(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v.structures.overlay = True
        v.structures.toggle_mark(2)
        text = v._draw_list().decode("utf-8", "replace")
        assert "overlay 1+3" in text
        assert "camera shared" in text
    finally:
        os.close(fd)


def test_viewer_list_footer_blank_when_overlay_off(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        text = v._draw_list().decode("utf-8", "replace")
        assert "overlay" not in text
        assert "camera shared" in text
    finally:
        os.close(fd)


# -- strip appearance (design §4.1) ----------------------------------------

def _strip_rows(v):
    """{0-based screen row: the SGR text drawn there} for one _draw_list()."""
    parts = re.split(r"\x1b\[(\d+);1H", v._draw_list().decode("utf-8", "replace"))
    return {int(parts[i]) - 1: parts[i + 1] for i in range(1, len(parts), 2)}


def _visible(s):
    """The printable text of an SGR-decorated string (escapes are 0-width)."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)


def _sgr_bg(rgb):
    return "\x1b[48;2;%d;%d;%dm" % rgb


def _sgr_fg(rgb):
    return "\x1b[38;2;%d;%d;%dm" % rgb


def test_viewer_list_every_row_is_exactly_list_w_visible_columns(tmp_path):
    """SGR escapes are zero-width: every drawn row must measure list_w
    PRINTABLE columns, or the panel corrupts the image beside it."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v.structures.overlay = True
        v.structures.toggle_mark(2)
        v.structures[1].visible = False
        v._list_cursor = 1
        for width in (18, 24, 28, 40):
            v._list_w = width
            for row, text in _strip_rows(v).items():
                assert len(_visible(text)) == width, (width, row, repr(text))
    finally:
        os.close(fd)


def test_viewer_list_header_is_muted_not_bold_with_a_blank_row_under_it(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        rows = _strip_rows(v)
        assert "STRUCTURES 3" in _visible(rows[0])
        assert _sgr_fg((139, 146, 165)) in rows[0]
        assert "\x1b[1m" not in rows[0]              # muted, not bold-white
        assert _visible(rows[1]).strip() == ""       # breathing room
    finally:
        os.close(fd)


def test_viewer_list_rows_have_no_leader_glyphs_and_a_block_swatch(tmp_path):
    """The active row is its background, not a '▸'; marks are not a '✓'."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v.structures.toggle_mark(2)
        v.structures[1].visible = False
        rows = _strip_rows(v)
        whole = "".join(rows.values())
        assert "▸" not in whole and "✓" not in whole
        first = v._list_row_spans[0][0]
        assert _visible(rows[first]).startswith(" █")   # pad, then a block swatch
        hidden = _visible(rows[first + 1])
        assert not hidden.startswith(" █")              # hollow while hidden
        assert "\x1b[2m" in rows[first + 1]             # ... and dimmed
    finally:
        os.close(fd)


def test_viewer_list_active_row_background_spans_the_whole_strip(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 24
        v.structures.set_active(1)
        v._list_cursor = 1
        rows = _strip_rows(v)
        active = rows[v._list_row_spans[1][0]]
        assert _sgr_bg((37, 45, 64)) in active
        # the highlight covers the padding too: the row opens with the
        # background and closes only at its very end
        assert active.index(_sgr_bg((37, 45, 64))) == 0
        assert len(_visible(active)) == 24
        # the near-white active label, dimmer on the others
        assert _sgr_fg((232, 236, 244)) in active
        assert _sgr_fg((200, 206, 216)) in rows[v._list_row_spans[0][0]]
        assert _sgr_fg((110, 118, 135)) in active       # muted index
    finally:
        os.close(fd)


def test_viewer_list_cursor_row_stays_distinguishable_from_the_active_row(tmp_path):
    """They differ whenever the cursor has moved without activating."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v.structures.set_active(0)
        v._list_cursor = 2
        rows = _strip_rows(v)
        active = rows[v._list_row_spans[0][0]]
        cursor = rows[v._list_row_spans[2][0]]
        assert _sgr_bg((37, 45, 64)) in active
        assert _sgr_bg((37, 45, 64)) not in cursor
        assert _sgr_bg((28, 33, 46)) in cursor
    finally:
        os.close(fd)


def test_viewer_list_marked_row_is_shown_in_its_tint(tmp_path):
    """▸/✓ leaders are gone, but a mark still has to be visible: the marked
    row's label takes the structure's own tint (design §4.4 -- colour
    identifies a structure everywhere it appears)."""
    v, fd = _multi_viewer(tmp_path)
    try:
        rows = _strip_rows(v)
        row_i = v._list_row_spans[2][0]
        tint = tuple(int(c * 255) for c in v.structures[2].tint)
        assert rows[row_i].count(_sgr_fg(tint)) == 1        # the swatch only
        v.structures.toggle_mark(2)
        rows = _strip_rows(v)
        assert rows[row_i].count(_sgr_fg(tint)) == 2        # swatch AND label

        # The tint beats the active row's near-white label: 'space' on the
        # active row must produce a VISIBLE change, and the full-width
        # background still says which row is active.
        v.structures.set_active(2)
        rows = _strip_rows(v)
        active = rows[v._list_row_spans[2][0]]
        assert _sgr_bg((37, 45, 64)) in active
        assert active.count(_sgr_fg(tint)) == 2
        assert _sgr_fg((232, 236, 244)) not in active
    finally:
        os.close(fd)


def test_viewer_list_separator_is_inset_by_one_column_each_side(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 24
        rows = _strip_rows(v)
        sep = next(t for t in rows.values() if "─" in _visible(t))
        assert _visible(sep) == " " + "─" * 22 + " "
        assert _sgr_fg((60, 66, 84)) in sep
    finally:
        os.close(fd)


def test_viewer_list_legend_renders_key_caps(tmp_path):
    v, fd = _multi_viewer(tmp_path)
    try:
        rows = _strip_rows(v)
        below = "".join(t for r, t in sorted(rows.items())
                        if r > v._list_row_spans[-1][0])
        for word in ("jump to", "next/prev", "solo", "hide"):
            assert word in _visible(below)
        # key caps: padded key text on a lighter background
        assert _sgr_bg((42, 49, 66)) in below
        for cap in (" 1 ", " 9 ", " n ", " p ", " z ", " h "):
            assert cap in _visible(below)
        # ]/[ are the global roll keys; the legend must not advertise them
        # as the strip's next/prev any more (VIM-9). 'space'/'mark' are gone
        # with the mark concept: overlay membership is opt+click only.
        for gone in (" ] ", " [ ", " space ", "mark"):
            assert gone not in _visible(below)
    finally:
        os.close(fd)


def test_viewer_list_panel_degrades_on_a_short_terminal(tmp_path):
    """A short terminal must never draw over the status bar, whatever falls
    off the bottom."""
    v, fd = _multi_viewer(tmp_path)
    try:
        for rows_h in (4, 6, 8, 12, 24):
            v._rows = rows_h
            drawn = _strip_rows(v)
            assert drawn, rows_h
            assert max(drawn) < rows_h - 1, rows_h    # status bar untouched
            for i in v._list_row_struct:
                assert 0 <= i < len(v.structures)
    finally:
        os.close(fd)


# -- structure-list scrolling (design §4.1) --------------------------------

def test_viewer_list_scrolls_to_reach_entries_past_the_viewport(tmp_path):
    """A list longer than the strip is scrollable: entries past the bottom
    are unreachable otherwise (60 frames, ~20 visible rows)."""
    from vimol.input import KeyEvent

    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        v._draw_list()
        assert v._list_scroll == 0
        assert len(v._list_row_struct) < 60          # can't all fit
        assert 59 not in v._list_row_struct          # the last frame is off-screen

        v._list_focused = True
        assert v._dispatch([KeyEvent("end")]) is True
        v._draw_list()
        assert 59 in v._list_row_struct              # ... now reachable
        assert v._list_cursor == 59

        assert v._dispatch([KeyEvent("home")]) is True
        v._draw_list()
        assert v._list_scroll == 0
        assert v._list_cursor == 0
        assert 0 in v._list_row_struct
    finally:
        os.close(fd)


def test_viewer_list_scroll_offset_clamps_at_both_ends(tmp_path):
    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        v._draw_list()
        cap = v._list_capacity()
        assert cap > 1
        for _ in range(200):
            v._list_scroll_by(1)
        assert v._list_scroll == v._list_max_scroll()
        assert v._list_max_scroll() == len(v._list_display_rows()) - cap
        for _ in range(200):
            v._list_scroll_by(-1)
        assert v._list_scroll == 0
    finally:
        os.close(fd)


def test_viewer_wheel_over_the_strip_scrolls_the_list_without_zooming(tmp_path):
    """Mouse-wheel events over the strip scroll it and must NOT reach the
    widget's zoom -- they were swallowed outright before."""
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        v._draw_list()
        zoom0 = v.widget.scene.camera.zoom
        down = MouseEvent("scroll", 1.0, 4.0, scroll="down")
        assert v._dispatch([down]) is True
        assert v._list_scroll > 0
        assert v.widget.scene.camera.zoom == zoom0

        scrolled = v._list_scroll
        up = MouseEvent("scroll", 1.0, 4.0, scroll="up")
        assert v._dispatch([up]) is True
        assert v._list_scroll < scrolled
        assert v.widget.scene.camera.zoom == zoom0

        # over the viewport the wheel still zooms
        assert v._dispatch([MouseEvent("scroll", float(v._list_w + 5), 4.0,
                                       scroll="up")]) is True
        assert v.widget.scene.camera.zoom != zoom0
    finally:
        os.close(fd)


def test_viewer_list_click_hits_the_right_structure_while_scrolled(tmp_path):
    """Row hit-testing must follow the scroll offset -- the easiest thing to
    get wrong, and it silently activates the wrong structure."""
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        v._list_scroll = 12
        v._draw_list()
        row0, col_start, _c1 = v._list_row_spans[3]
        want = v._list_row_struct[3]
        assert want > 3                       # genuinely a scrolled-to row
        assert v._dispatch([MouseEvent("down", float(col_start), float(row0),
                                       button=0)]) is True
        assert v.frame_index == want
    finally:
        os.close(fd)


def test_viewer_list_jk_autoscroll_keeps_the_cursor_visible(tmp_path):
    from vimol.input import KeyEvent

    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        v._list_focused = True
        v._draw_list()
        cap = v._list_capacity()
        for _ in range(cap + 5):
            v._dispatch([KeyEvent("j")])
        v._draw_list()
        assert v._list_cursor == cap + 5
        assert v._list_cursor in v._list_row_struct     # scrolled into view
        assert v._list_scroll > 0
        for _ in range(cap + 5):
            v._dispatch([KeyEvent("k")])
        v._draw_list()
        assert v._list_cursor == 0
        assert v._list_scroll == 0
        assert 0 in v._list_row_struct
    finally:
        os.close(fd)


def test_viewer_activating_an_offscreen_structure_scrolls_it_into_view(tmp_path):
    """n / opt+down / a digit jump can walk the active structure past the
    bottom of the strip; the strip follows it."""
    from vimol.input import KeyEvent

    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        v._draw_list()
        for _ in range(v._list_capacity() + 4):
            v._dispatch([KeyEvent("n")])
        v._draw_list()
        assert v.frame_index in v._list_row_struct
        assert v._list_scroll > 0
    finally:
        os.close(fd)


def test_viewer_list_header_marks_that_there_is_more_to_scroll(tmp_path):
    """An overflowing list says so -- otherwise nothing tells the user the
    strip has more below."""
    v, fd = _traj_viewer(tmp_path, n=60)
    try:
        head = v._draw_list().decode("utf-8", "replace").split("\x1b[2;1H")[0]
        assert "↓" in head and "↑" not in head
        v._list_scroll = v._list_max_scroll()
        head = v._draw_list().decode("utf-8", "replace").split("\x1b[2;1H")[0]
        assert "↑" in head and "↓" not in head
    finally:
        os.close(fd)

    v2, fd2 = _traj_viewer(tmp_path, n=3, fd_name="short.bin")
    try:
        head = v2._draw_list().decode("utf-8", "replace").split("\x1b[2;1H")[0]
        assert "↑" not in head and "↓" not in head     # nothing to scroll
    finally:
        os.close(fd2)


# -- structure-list keymap: Tab focus, list-focused keys (design §4.3) -----

def test_viewer_tab_toggles_list_focus(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        assert v._list_focused is False
        assert v._dispatch([KeyEvent("tab")]) is True
        assert v._list_focused is True
        assert v._dispatch([KeyEvent("tab")]) is True
        assert v._list_focused is False
    finally:
        os.close(fd)


def test_viewer_list_focused_digit_jumps_to_index_without_activating(tmp_path):
    """1-9 while list-focused move the CURSOR, not the active structure
    (design §4.3: 'j/k move a cursor without changing what is rendered')."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        assert v._dispatch([KeyEvent("3")]) is True
        assert v._list_cursor == 2
        assert v.frame_index == 0   # active structure unchanged
    finally:
        os.close(fd)


def test_viewer_list_focused_n_p_cycle_active(tmp_path):
    """next/prev is n/p, not ]/[ (design §4.3). Both are global driver keys,
    so they must still reach the driver through a focused strip -- the strip
    claims nothing for them and unclaimed keys fall through."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        assert v._dispatch([KeyEvent("n")]) is True
        assert v.frame_index == 1
        assert v._dispatch([KeyEvent("n")]) is True
        assert v.frame_index == 2
        assert v._dispatch([KeyEvent("p")]) is True
        assert v.frame_index == 1
        assert v._dispatch([KeyEvent("p")]) is True
        assert v.frame_index == 0
    finally:
        os.close(fd)


def test_viewer_list_focused_bracket_keys_roll_the_camera_not_the_strip(tmp_path):
    """]/[ are the GLOBAL roll bindings (widget.py). The strip used to shadow
    them with next/prev; it must not -- focused or not, they reach the widget
    and roll the camera, leaving the active structure alone."""
    import numpy as np
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        cam = v.widget.scene.camera
        before = cam.rotation.copy()
        assert v._dispatch([KeyEvent("]")]) is True
        rolled = cam.rotation.copy()
        assert not np.allclose(rolled, before)      # ] rolled the camera
        assert v.frame_index == 0                   # ... and did NOT cycle
        assert v._dispatch([KeyEvent("[")]) is True
        assert np.allclose(cam.rotation, before)    # [ rolled it back
        assert v.frame_index == 0
    finally:
        os.close(fd)


def test_viewer_list_focused_space_no_longer_changes_overlay_membership(tmp_path):
    """The user-facing 'mark' concept is gone (VIM-9): overlay membership is
    reachable ONLY by opt+click. 'space' must claim nothing in the strip's
    keymap and must not change what the overlay draws."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v._list_cursor = 1
        before = v.structures.drawn_indices()
        assert v._handle_list_key(" ") is False
        v._dispatch([KeyEvent(" ")])
        assert v.structures.drawn_indices() == before
        assert v.structures.overlay is False
    finally:
        os.close(fd)


def test_viewer_list_legend_is_three_rows_and_matches_the_reserved_footer(tmp_path):
    """The legend lost its 'space mark' row, so the footer height reserved by
    _LIST_ROWS_BELOW (separator + legend) has to shrink with it -- otherwise
    _list_capacity() lies about what fits and the scroll clamps drift."""
    from vimol.viewer import _LIST_ROWS_BELOW

    v, fd = _multi_viewer(tmp_path)
    try:
        assert len(v._list_legend()) == 3
        assert _LIST_ROWS_BELOW == 1 + len(v._list_legend())
    finally:
        os.close(fd)


def test_viewer_list_focused_z_solos_and_restores(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v._list_cursor = 1
        assert v._dispatch([KeyEvent("z")]) is True
        assert [e.visible for e in v.structures] == [False, True, False]
        assert v._dispatch([KeyEvent("z")]) is True
        assert [e.visible for e in v.structures] == [True, True, True]
    finally:
        os.close(fd)


def test_viewer_list_focused_h_hides_but_refuses_to_hide_the_last_visible(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v._list_cursor = 0
        assert v._dispatch([KeyEvent("h")]) is True
        assert v.structures[0].visible is False
        assert v.frame_index == 0   # hiding the active structure does not advance it
        v.structures.toggle_visible(1)
        v.structures.toggle_visible(2)
        # now [1, False, False] visible=[True? no]: bring 0 back so it's the
        # ONLY visible one, then a second 'h' on it must be refused.
        v.structures.toggle_visible(0)
        assert [e.visible for e in v.structures] == [True, False, False]
        v._list_cursor = 0
        v._dispatch([KeyEvent("h")])
        assert v.structures[0].visible is True
        assert "at least one" in v._msg
    finally:
        os.close(fd)


def test_viewer_list_focused_v_toggles_overlay(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        assert v.structures.overlay is False
        assert v._dispatch([KeyEvent("v")]) is True
        assert v.structures.overlay is True
    finally:
        os.close(fd)


def test_viewer_list_focused_o_falls_through_to_autospin_when_editable(tmp_path):
    # The strip no longer claims 'o' (it collided with the editable-mode
    # autospin binding, _EDIT_DRIVER_KEYS); unclaimed keys fall through to
    # the global driver keys (design §4.3).
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path, editable=True)
    try:
        v._list_focused = True
        assert v.autospin is False
        assert v.structures.overlay is False
        assert v._dispatch([KeyEvent("o")]) is True
        assert v.autospin is True
        assert v.structures.overlay is False
    finally:
        os.close(fd)


def test_viewer_list_focused_enter_activates_cursor_clears_marks_returns_focus(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v.structures.toggle_mark(1)
        v._list_cursor = 2
        assert v._dispatch([KeyEvent("enter")]) is True
        assert v.frame_index == 2
        assert v.structures.marked == []
        assert v._list_focused is False
    finally:
        os.close(fd)


def test_viewer_list_focused_escape_returns_focus_without_activating(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v._list_cursor = 2
        assert v._dispatch([KeyEvent("escape")]) is True
        assert v._list_focused is False
        assert v.frame_index == 0
    finally:
        os.close(fd)


def test_viewer_opt_click_row_marks_and_enables_overlay(tmp_path):
    from vimol.viewer import Viewer
    from vimol.input import MouseEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._draw_list()
        row0, col_start, _c1 = v._list_row_spans[1]
        click = MouseEvent("down", float(col_start), float(row0), button=0, alt=True)
        assert v._dispatch([click]) is True
        assert v.structures[1].marked is True
        assert v.structures.overlay is True
        assert v.frame_index == 0   # active structure unchanged by opt+click
    finally:
        os.close(fd)


def test_viewer_opt_click_toggles_a_marked_row_back_off(tmp_path):
    """opt+click is a TOGGLE: opt+clicking an already-marked row unmarks it
    and leaves every other mark intact. Unmarking the LAST mark also turns
    the overlay off -- an empty mark set would otherwise fall back to
    'draw everything visible' (drawn_indices), the opposite of what
    unmarking your last selection means."""
    from vimol.input import MouseEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._draw_list()

        def opt_click(k):
            row0, col_start, _c1 = v._list_row_spans[k]
            return v._dispatch([MouseEvent("down", float(col_start), float(row0),
                                           button=0, alt=True)])

        assert opt_click(1) is True
        assert opt_click(2) is True
        assert [e.marked for e in v.structures] == [False, True, True]
        assert v.structures.overlay is True

        # toggling row 1 back off leaves row 2 marked and the overlay on
        assert opt_click(1) is True
        assert [e.marked for e in v.structures] == [False, False, True]
        assert v.structures.overlay is True

        # ... and dropping the last mark turns the overlay off again
        assert opt_click(2) is True
        assert [e.marked for e in v.structures] == [False, False, False]
        assert v.structures.overlay is False
        assert v.frame_index == 0   # the active structure never moved
    finally:
        os.close(fd)


def test_viewer_global_bindings_unaffected_when_list_not_focused(tmp_path):
    """The pre-existing global 1-4 (representation), [/] (roll), h (orbit)
    bindings must keep working exactly as before when the list isn't
    focused (design §4.3: 'not one existing binding changes')."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        assert v._list_focused is False
        assert v._dispatch([KeyEvent("2")]) is True
        assert v.style.representation == "spacefill"
        rot0 = v.widget.scene.camera.rotation.copy()
        assert v._dispatch([KeyEvent("h")]) is True
        assert not np.array_equal(v.widget.scene.camera.rotation, rot0)
        # these must NOT have touched list state
        assert v._list_cursor == 0
        assert v.structures[0].visible is True
    finally:
        os.close(fd)


def test_viewer_list_zone_drag_latch_never_reaches_viewport(tmp_path):
    """A drag that starts on the strip must never reach the viewport, even
    if it strays over the molecule mid-drag (design §4.2, mirrors
    _status_zone_press)."""
    from vimol.viewer import Viewer
    from vimol.input import MouseEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._draw_list()
        row0, col_start, _c1 = v._list_row_spans[0]
        rot0 = v.widget.scene.camera.rotation.copy()
        down = MouseEvent("down", float(col_start), float(row0), button=0)
        v._dispatch([down])
        # drag strays far to the right, well into the viewport's columns
        drag = MouseEvent("drag", float(v._cols - 1), float(row0), button=0)
        v._dispatch([drag])
        assert np.array_equal(v.widget.scene.camera.rotation, rot0)
    finally:
        os.close(fd)


def test_viewer_viewport_click_still_reaches_widget_past_the_strip(tmp_path):
    """A click in the viewport's own columns (past the reserved strip) must
    still reach the widget, offset by _img_origin_px (design §4.2)."""
    from vimol.viewer import Viewer
    from vimol.input import MouseEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        rot0 = v.widget.scene.camera.rotation.copy()
        col = v._list_w + 5
        down = MouseEvent("down", float(col), 5.0, button=0)
        v._dispatch([down])
        drag = MouseEvent("drag", float(col + 20), 5.0, button=0)
        assert v._dispatch([drag]) is True
        assert not np.array_equal(v.widget.scene.camera.rotation, rot0)
    finally:
        os.close(fd)


def test_viewer_status_bar_shows_pick_refusal_message(tmp_path):
    """A refused cross-structure edit click's message surfaces in the
    left-hand status-bar field (design §3, §12.3)."""
    from vimol.viewer import Viewer

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd, editable=True)
        v._update_geometry()
        v.widget.pick_refusal = "atom belongs to b — Tab to activate"
        bar = v._status_bar()
        assert "atom belongs to b" in bar
    finally:
        os.close(fd)


def test_viewer_list_focused_unclaimed_keys_fall_through_to_global_bindings(tmp_path):
    """n/p (design: 'keep doing plain next/prev regardless of list focus')
    and quit keys must still work while the strip has focus -- only keys the
    list keymap actually claims (§4.3) may be swallowed."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        assert v._dispatch([KeyEvent("n")]) is True
        assert v.frame_index == 1
        assert v._dispatch([KeyEvent("q")]) is False
        assert v._running is False
    finally:
        os.close(fd)


def test_viewer_hiding_active_structure_while_hovered_does_not_crash(tmp_path):
    """Hiding the active row (design §4.3: allowed, does not auto-advance)
    while an atom is hovered must not crash _apply_highlight -- the active
    structure is no longer in the composite's drawn sources at all."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v.widget.hovered = 0
        v._list_focused = True
        v._list_cursor = v.frame_index
        assert v._dispatch([KeyEvent("h")]) is True
        v._draw()   # must not raise
    finally:
        os.close(fd)


def test_viewer_plain_arrows_always_orbit_even_while_list_focused(tmp_path):
    """Plain up/down orbit the camera whether or not the strip has focus; the
    structure-list cursor is driven by j/k, and opt+up/down keep cycling the
    active structure globally."""
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v._list_cursor = 0
        rot0 = v.widget.scene.camera.rotation.copy()
        assert v._dispatch([KeyEvent("down")]) is True
        assert not np.array_equal(v.widget.scene.camera.rotation, rot0)
        assert v._list_cursor == 0          # the arrow did NOT move the cursor
        rot1 = v.widget.scene.camera.rotation.copy()
        assert v._dispatch([KeyEvent("up")]) is True
        assert not np.array_equal(v.widget.scene.camera.rotation, rot1)
        assert v._list_cursor == 0

        # opt+arrows still reach the driver keys while the strip has focus
        assert v._dispatch([KeyEvent("alt+down")]) is True
        assert v.frame_index == 1
        assert v._dispatch([KeyEvent("alt+up")]) is True
        assert v.frame_index == 0

        # j/k remain the focused cursor keys
        assert v._dispatch([KeyEvent("j")]) is True
        assert v._list_cursor == 1
        assert v._dispatch([KeyEvent("k")]) is True
        assert v._list_cursor == 0
    finally:
        os.close(fd)
