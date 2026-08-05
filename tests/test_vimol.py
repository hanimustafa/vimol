import base64
import os
import re
import sys
import tempfile
import time

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
    # a cached last-detected theme is the fallback below COLORFGBG
    assert theme.resolve(None, None, None, cached="light") is theme.LIGHT
    assert theme.resolve(None, None, None, cached="dark") is theme.DARK
    assert theme.resolve(None, None, None, cached="garbage") is theme.DARK
    assert theme.resolve(None, None, "0;15", cached="dark") is theme.LIGHT  # colorfgbg still wins


def test_theme_cache_roundtrip(tmp_path, monkeypatch):
    from vimol import theme
    monkeypatch.setattr(theme, "_CACHE_PATH", str(tmp_path / "cache"))
    assert theme.read_cached() is None       # nothing written yet
    theme.write_cached("light")
    assert theme.read_cached() == "light"
    theme.write_cached("dark")
    assert theme.read_cached() == "dark"


def test_theme_cache_ignores_garbage_file_contents(tmp_path, monkeypatch):
    from vimol import theme
    path = tmp_path / "cache"
    path.write_text("not-a-theme")
    monkeypatch.setattr(theme, "_CACHE_PATH", str(path))
    assert theme.read_cached() is None


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


def test_viewer_strip_ordinary_rows_stay_transparent(tmp_path):
    """The strip panel is transparent: only the active and cursor rows paint
    a background, so the terminal's own shows through everywhere else.
    Structure #2 (index 1) is neither here, so its row is the "ordinary"
    case and must carry no background SGR at all."""
    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 24
        v._list_cursor = 0   # keep the cursor off row 1 too
        data = v._draw_list().decode("utf-8", "replace")
        parts = re.split(r"\x1b\[(\d+);1H", data)
        rows = {int(parts[i]) - 1: parts[i + 1] for i in range(1, len(parts), 2)}
        ordinary_row = rows[v._list_row_spans[1][0]]
        assert "\x1b[48;2;" not in ordinary_row
    finally:
        os.close(fd)


def test_viewer_strip_foregrounds_flip_with_the_theme(tmp_path):
    """Light-terminal readability comes from the FOREGROUND palette, not
    from painting an opaque panel: the dark theme's near-white row labels
    would wash out on a light terminal, so the light theme's must be dark
    enough to read against it (the originally reported bug)."""
    from vimol import theme as vimol_theme

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_w = 24
        dark_rows = v._draw_list().decode("utf-8", "replace")
        assert _sgr_fg(vimol_theme.DARK.list_dim_fg) in dark_rows

        v.theme = vimol_theme.LIGHT
        light_rows = v._draw_list().decode("utf-8", "replace")
        assert _sgr_fg(vimol_theme.LIGHT.list_dim_fg) in light_rows
        # and the light palette is genuinely dark ink, not the dark theme's
        # near-white text reused
        assert vimol_theme.luminance(vimol_theme.LIGHT.list_dim_fg) < 140
        assert vimol_theme.luminance(vimol_theme.DARK.list_dim_fg) > 140
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


def test_viewer_ctrl_t_toggles_theme(monkeypatch):
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    v._update_geometry()
    assert v.theme is vimol_theme.DARK
    assert v.widget.theme == "dark"
    assert v._dispatch([KeyEvent("\x14")]) is True
    assert v.theme is vimol_theme.LIGHT
    assert v.widget.theme == "light"
    assert v._dispatch([KeyEvent("\x14")]) is True
    assert v.theme is vimol_theme.DARK
    assert v.widget.theme == "dark"


def test_viewer_ctrl_t_available_read_only(monkeypatch):
    """Like d/g/t/n/p/m, ctrl-t is a base driver key -- available whether
    or not editing is enabled."""
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY), editable=False)
    assert v.theme is vimol_theme.DARK
    assert v._dispatch([KeyEvent("\x14")]) is True
    assert v.theme is vimol_theme.LIGHT


def test_viewer_frame0_theme_guess_uses_colorfgbg_before_probe(monkeypatch):
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)   # explicit would outrank COLORFGBG
    monkeypatch.setenv("COLORFGBG", "0;15")
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    assert v.theme is vimol_theme.LIGHT   # guessed synchronously, no probe run yet


def test_viewer_finish_startup_upgrades_theme_from_osc11(monkeypatch):
    from vimol import theme as vimol_theme, kitty as vimol_kitty
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    probe = vimol_kitty.TerminalProbe(graphics=True, pixel_mouse=False, cell_px=(9.0, 18.0),
                                      rtt=0.001, bg_rgb=(245, 245, 245))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY), probe=probe)
    assert v.theme is vimol_theme.DARK   # frame-0 guess: no COLORFGBG, default dark
    v._old_termios = object()            # _finish_startup's "probe already available" path
    v._finish_startup()
    assert v.theme is vimol_theme.LIGHT  # OSC 11 corrected it
    assert v.widget.theme == "light"


def test_viewer_frame0_uses_cached_theme_when_no_env_or_colorfgbg(monkeypatch):
    """The flicker fix: on a light terminal with no COLORFGBG set, frame 0
    used to always guess DARK and only correct to LIGHT once the OSC 11
    reply landed in _finish_startup -- a visible dark-then-light flash on
    every single startup. A cached last-detected theme lets frame 0 guess
    right the first time for the common case of running in the same
    terminal repeatedly."""
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    vimol_theme.write_cached("light")
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    assert v.theme is vimol_theme.LIGHT


def test_viewer_probe_writes_cache_only_from_a_real_osc11_reply(monkeypatch, tmp_path):
    """The cache must reflect what the TERMINAL said, not a COLORFGBG guess
    or an explicit override -- either would poison next run's frame-0 guess
    with something that was never actually detected."""
    from vimol import theme as vimol_theme, kitty as vimol_kitty
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))

    # no bg_rgb (terminal never answered OSC 11) -> nothing cached
    v._apply_probe_theme(vimol_kitty.TerminalProbe(graphics=True, pixel_mouse=False,
                                                   cell_px=None, bg_rgb=None))
    assert vimol_theme.read_cached() is None

    # a real answer IS cached
    v._apply_probe_theme(vimol_kitty.TerminalProbe(graphics=True, pixel_mouse=False,
                                                   cell_px=None, bg_rgb=(245, 245, 245)))
    assert vimol_theme.read_cached() == "light"


def test_viewer_probe_does_not_cache_an_explicit_override(monkeypatch):
    from vimol import theme as vimol_theme, kitty as vimol_kitty
    from vimol.viewer import Viewer

    monkeypatch.setenv("VIMOL_THEME", "dark")
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    # terminal genuinely answers "light", but the user forced dark -- that's
    # not the terminal telling us anything, so it must not be cached
    v._apply_probe_theme(vimol_kitty.TerminalProbe(graphics=True, pixel_mouse=False,
                                                   cell_px=None, bg_rgb=(245, 245, 245)))
    assert vimol_theme.read_cached() is None
    assert v.theme is vimol_theme.DARK   # explicit still wins


def test_viewer_late_probe_also_applies_the_osc11_theme(monkeypatch, tmp_path):
    """The late-reply path is the SSH/congested-link case -- exactly where
    auto-detection matters most -- so it must apply bg_rgb too, not just
    cell size and pacing."""
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        assert v.theme is vimol_theme.DARK
        v._late_t0 = time.monotonic()
        v._late_buf = b""
        # a full probe reply arriving late, reporting a light background
        v._late_probe_tick(b"\x1b_Gi=31;OK\x1b\\\x1b]11;rgb:f5f5/f5f5/f5f5\x1b\\\x1b[?62c")
        assert v.theme is vimol_theme.LIGHT
        assert v.widget.theme == "light"
    finally:
        os.close(fd)


def test_viewer_manual_ctrl_t_outranks_a_later_probe_reply(monkeypatch, tmp_path):
    """Once the user has pressed ctrl-t, a probe answer landing afterwards
    must not yank the theme back out from under them."""
    from vimol import theme as vimol_theme
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    monkeypatch.delenv("VIMOL_THEME", raising=False)
    monkeypatch.delenv("COLORFGBG", raising=False)
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, fd_out=fd)
        v._dispatch([KeyEvent("\x14")])          # user pins LIGHT
        assert v.theme is vimol_theme.LIGHT
        v._late_t0 = time.monotonic()
        v._late_buf = b""
        # probe says the terminal is dark -- ignored, the user has spoken
        v._late_probe_tick(b"\x1b_Gi=31;OK\x1b\\\x1b]11;rgb:1010/1010/1010\x1b\\\x1b[?62c")
        assert v.theme is vimol_theme.LIGHT
    finally:
        os.close(fd)


def test_viewer_status_bar_shows_full_untruncated_comment_when_room_allows():
    """The xyz comment is no longer capped at 60 chars on parse, so a wide
    terminal shows the whole energy line instead of a clipped stub."""
    from vimol.viewer import Viewer

    long_comment = "SCF Energy = -76.123456789012 Hartree, converged in 42 cycles"
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    mol.name = long_comment
    v = Viewer(mol, fd_out=os.open(os.devnull, os.O_WRONLY))
    v._cols, v._rows = 200, 24   # wide enough that the left field isn't clipped
    assert long_comment in v._status_bar()


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


@pytest.mark.parametrize("reply", [
    b"\x1b]11;#f0f2f5\x1b\\",          # XParseColor '#rrggbb' form
    b"\x1b]11;rgb:1/2/2\x07",           # 1-digit channels
    b"\x1b]11;rgbi:1.0/1.0/1.0\x1b\\",  # intensity form
])
def test_unparsed_osc_reply_never_leaks_into_leftover(reply):
    """A colour form we can't parse must still be STRIPPED, not handed to the
    input decoder -- '\\x1b]11;#f0f2f5\\x1b\\\\' would otherwise arrive as the
    keystrokes 1,1,f,f,2,f,5 and fire jump-to-structure/re-fit/quality at
    startup."""
    buf = b"\x1b_Gi=31;OK\x1b\\" + reply + b"\x1b[?62c"
    probe = kitty.parse_probe_reply(buf)
    assert probe is not None
    assert probe.leftover == b""


def test_probe_leftover_still_preserves_real_keystrokes():
    """Stripping OSC replies must not eat what the user actually typed."""
    buf = (b"\x1b_Gi=31;OK\x1b\\"
           b"ab"
           b"\x1b]11;rgb:1e1e/2020/2828\x1b\\"
           b"cd"
           b"\x1b[?62c")
    probe = kitty.parse_probe_reply(buf)
    assert probe.bg_rgb == (0x1e, 0x20, 0x28)
    assert probe.leftover == b"abcd"


def test_strip_probe_replies_covers_graphics_and_osc():
    buf = b"x\x1b_Gi=31;OK\x1b\\y\x1b]11;rgb:1e1e/2020/2828\x1b\\z"
    assert kitty.strip_probe_replies(buf) == b"xyz"


def test_png_roundtrip_header():
    img = np.zeros((16, 16, 3), np.uint8)
    png = kitty.png_bytes(img)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in png[:32]
    assert png.rstrip().endswith(b"IEND".rjust(4)) or b"IEND" in png


def test_clipboard_set_text_uses_osc52_in_one_base64_payload():
    """OSC 52 is what terminals actually implement -- Ghostty parses Kitty's
    OSC 5522 without acting on it, so a 5522 write silently does nothing."""
    from vimol import kitty

    text = "2\ncomment\nH 0.0 0.0 0.0\nH 1.0 0.0 0.0\n"
    seq = kitty.clipboard_set_text(text)
    assert seq.startswith(b"\x1b]52;c;")
    assert seq.endswith(b"\x07")
    payload = seq[len(b"\x1b]52;c;"):-1]
    assert base64.standard_b64decode(payload).decode("utf-8") == text


def test_clipboard_set_text_survives_non_ascii_and_empty_input():
    from vimol import kitty

    seq = kitty.clipboard_set_text("\u00c5\u2014\u00e5ngstr\u00f6m\n")
    payload = seq[len(b"\x1b]52;c;"):-1]
    assert (base64.standard_b64decode(payload).decode("utf-8")
            == "\u00c5\u2014\u00e5ngstr\u00f6m\n")
    # An empty selection clears the clipboard rather than emitting junk.
    assert kitty.clipboard_set_text("") == b"\x1b]52;c;\x07"


def test_input_decoder_swallows_split_clipboard_status_reply():
    from vimol.input import InputDecoder, KeyEvent

    decoder = InputDecoder()
    assert decoder.feed(b"\x1b]5522;type=write:status=") == []
    events = decoder.feed(b"DONE\x1b\\x")
    assert events == [KeyEvent("x")]


def test_input_decoder_recovers_from_an_unterminated_osc_introducer():
    """Alt+] sends a bare ESC ] in most terminals, and a terminal reply can
    be truncated. Holding those bytes forever swallows every later keystroke
    -- including q -- leaving no way out of the viewer but killing it."""
    from vimol.input import InputDecoder, KeyEvent

    decoder = InputDecoder()
    assert decoder.feed(b"\x1b]") == []          # nothing to emit yet: could be a real OSC
    decoder.feed(b"qqq")
    # An idle read is the terminal saying no terminator is coming.
    decoder.flush()
    assert decoder.feed(b"q") == [KeyEvent("q")]

    # A partial OSC that never terminates must not eat the keys behind it.
    decoder = InputDecoder()
    decoder.feed(b"\x1b]5522;type=wri")
    decoder.flush()
    assert decoder.feed(b"z") == [KeyEvent("z")]


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


def test_cli_theme_flag_sets_env_before_viewer_construction(monkeypatch):
    from vimol import app
    # setenv (not delenv) so monkeypatch records a restore point: delenv on an
    # ALREADY-ABSENT var records nothing, and _apply_theme_arg's write would
    # then leak "light" into every test that builds a Viewer afterwards.
    monkeypatch.setenv("VIMOL_THEME", "dark")
    p = app.make_parser()
    args = p.parse_args(["--theme", "light", os.path.join(EX, "methane.xyz")])
    assert args.theme == "light"
    app._apply_theme_arg(args)
    assert os.environ["VIMOL_THEME"] == "light"


def test_cli_theme_auto_leaves_env_override_intact(monkeypatch):
    """"auto" is the default, i.e. "no --theme given" -- it must not delete
    the user's VIMOL_THEME, or the env rung of the precedence ladder would
    never work through the CLI at all."""
    from vimol import app
    monkeypatch.setenv("VIMOL_THEME", "light")
    args = app.make_parser().parse_args(["--theme", "auto"])
    app._apply_theme_arg(args)
    assert os.environ["VIMOL_THEME"] == "light"


def test_cli_theme_flag_outranks_env(monkeypatch):
    from vimol import app, theme
    monkeypatch.setenv("VIMOL_THEME", "light")
    args = app.make_parser().parse_args(["--theme", "dark"])
    app._apply_theme_arg(args)
    assert theme.resolve(os.environ.get("VIMOL_THEME"), None, None) is theme.DARK


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
    from vimol import theme as _theme
    frames = [vimol.load(os.path.join(EX, "methane.xyz")),
              vimol.load(os.path.join(EX, "water.xyz")),
              vimol.load(os.path.join(EX, "benzene.xyz"))]
    fd = os.open(str(tmp_path / fd_name), os.O_WRONLY | os.O_CREAT, 0o644)
    v = Viewer(frames[0], frames=frames, fd_out=fd, **kw)
    # Pin the theme: the strip tests below assert specific DARK colour
    # literals, and Viewer's constructor otherwise picks a theme from the
    # AMBIENT environment (VIMOL_THEME/COLORFGBG) -- so without this the
    # suite would fail for anyone whose shell exports COLORFGBG for a light
    # terminal. Tests that want the light palette set v.theme themselves.
    v.theme = _theme.DARK
    v.widget.theme = _theme.DARK.name
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


def test_copy_xyz_control_is_right_anchored_and_writes_osc52(tmp_path, monkeypatch):
    from vimol import kitty
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=2, fd_name="copy-controls.bin")
    try:
        v._cols, v._rows = 100, 30
        rendered = _visible(v._draw_copy_controls().decode("utf-8", "replace"))
        assert "\u29c9 XYZ" in rendered
        assert "PNG" not in rendered
        assert v._copy_xyz_span[0] == 0
        assert v._copy_xyz_span[2] == v._cols - 1

        written = []
        monkeypatch.setattr(kitty, "write_bytes",
                            lambda data, fd=1: written.append((data, fd)))

        row, start, _end = v._copy_xyz_span
        assert v._dispatch([MouseEvent(
            "down", float(start), float(row), button=0)]) is True
        payload, target = written[-1]
        assert target == fd
        assert payload.startswith(b"\x1b]52;c;") and payload.endswith(b"\x07")
        text = base64.standard_b64decode(
            payload[len(b"\x1b]52;c;"):-1]).decode("utf-8")
        assert text.startswith("5\n")
        assert v._msg == "current XYZ copied"
    finally:
        os.close(fd)


def test_copy_control_is_suppressed_when_the_strip_would_be_overpainted(tmp_path):
    v, fd = _traj_viewer(tmp_path, n=2, fd_name="copy-narrow.bin")
    try:
        v._cols, v._rows = 100, 30
        assert v._draw_copy_controls() != b""
        v._list_w = v._cols            # strip claims the whole width
        assert v._draw_copy_controls() == b""
        assert v._copy_xyz_span is None
    finally:
        os.close(fd)


def test_current_xyz_copies_drawn_frames_transformed_with_original_comments(tmp_path):
    from vimol.parsers import xyz as xyz_parser
    from vimol.structures import Transform

    v, fd = _traj_viewer(tmp_path, n=3, fd_name="copy-xyz-frames.bin")
    try:
        for i, entry in enumerate(v.structures):
            entry.molecule.name = f"original xyz comment {i + 1}"
            entry.marked = i != 2
        v.structures.overlay = True
        v.structures[1].transform = Transform(
            translation=np.array([3.0, -2.0, 1.0]))

        text = v._current_xyz_text()
        frames = xyz_parser.parse(text)
        assert [frame.name for frame in frames] == [
            "original xyz comment 1", "original xyz comment 2"]
        assert np.allclose(frames[0].positions,
                           v.structures[0].molecule.positions)
        assert np.allclose(frames[1].positions,
                           v.structures[1].molecule.positions + [3.0, -2.0, 1.0])
        assert "original xyz comment 3" not in text
    finally:
        os.close(fd)


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


def test_clicking_filename_collapses_and_summary_click_expands(tmp_path):
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=4, fd_name="collapse-toggle.bin")
    try:
        v._list_w = 28
        v._draw_list()
        row, c0, c1, first, end = v._list_group_toggle_spans[0]
        assert (first, end) == (0, 4)
        assert v._dispatch([MouseEvent(
            "down", float((c0 + c1) // 2), float(row), button=0)]) is True

        assert v._list_display_rows() == [
            ("group", 0, "traj.xyz"), ("collapsed", 0, "4 frames")]
        collapsed = _visible(v._draw_list().decode("utf-8", "replace"))
        assert "4 frames" in collapsed
        assert "frame 1" not in collapsed and "frame 4" not in collapsed
        assert "ALL" in collapsed

        row, c0, c1, first, end = v._list_group_summary_spans[0]
        assert (first, end) == (0, 4)
        assert v._dispatch([MouseEvent(
            "down", float((c0 + c1) // 2), float(row), button=0)]) is True
        assert [kind for kind, _i, _text in v._list_display_rows()] == [
            "group", "struct", "struct", "struct", "struct"]

        # The filename itself is a true toggle in both directions too.
        v._draw_list()
        row, c0, c1, _first, _end = v._list_group_toggle_spans[0]
        click = MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)
        assert v._dispatch([click]) is True
        assert v._list_display_rows()[-1][0] == "collapsed"
        v._draw_list()
        row, c0, c1, _first, _end = v._list_group_toggle_spans[0]
        assert v._dispatch([MouseEvent(
            "down", float((c0 + c1) // 2), float(row), button=0)]) is True
        assert v._list_display_rows()[-1][0] == "struct"
    finally:
        os.close(fd)


def test_collapsed_file_all_button_selects_and_clears_every_frame(tmp_path):
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=4, fd_name="collapse-all.bin")
    try:
        v.structures.overlay = True
        v._toggle_list_group_collapsed(0, 4)
        v._draw_list()
        assert len(v._list_group_all_spans) == 1
        row, c0, c1, first, end = v._list_group_all_spans[0]
        click = MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)
        assert v._dispatch([click]) is True
        assert all(entry.marked for entry in v.structures)
        assert v._list_display_rows()[-1] == ("collapsed", 0, "4 frames")

        v._draw_list()
        row, c0, c1, _first, _end = v._list_group_all_spans[0]
        assert v._dispatch([MouseEvent(
            "down", float((c0 + c1) // 2), float(row), button=0)]) is True
        assert [entry.marked for entry in v.structures] == [
            True, False, False, False]
        assert v.structures.drawn_indices() == [0]
    finally:
        os.close(fd)


def test_collapsed_group_keeps_active_frame_reachable_and_clamps_scroll(tmp_path):
    v, fd = _traj_viewer(tmp_path, n=20, fd_name="collapse-scroll.bin")
    try:
        v._rows = 14
        v._list_scroll_to(8)
        assert v._list_scroll == 8
        v._activate_structure(12)
        v._toggle_list_group_collapsed(0, 20)
        assert v._list_scroll == 0
        assert v.structures.active_index == 12
        assert v._list_display_rows() == [
            ("group", 0, "traj.xyz"), ("collapsed", 0, "20 frames")]

        v._cycle_frame(1)
        assert v.structures.active_index == 13
        assert v._list_scroll == 0
        assert v._list_display_rows()[-1] == ("collapsed", 0, "20 frames")
    finally:
        os.close(fd)


def test_collapsing_erases_the_frame_rows_it_stops_drawing(tmp_path):
    """Collapsing turns hundreds of rows into two, so the strip stops painting
    most of the column it used to own. Nothing else repaints those cells --
    the molecule image lives to the right -- so a shrinking strip has to clear
    what it vacated, or the old frame rows stay on screen under the legend."""
    import re

    v, fd = _traj_viewer(tmp_path, n=60, fd_name="collapse-erase.bin")
    try:
        v._rows = 30
        v._list_w = 28
        # A cleared row is addressed and then blanked; anything else is content.
        erase = re.compile(rb"\x1b\[(\d+);1H\x1b\[0m\x1b\[2K")
        cup = re.compile(rb"\x1b\[(\d+);(\d+)H")

        def split(payload):
            cleared = {int(m.group(1)) - 1 for m in erase.finditer(payload)}
            touched = {int(m.group(1)) - 1 for m in cup.finditer(payload)}
            return touched - cleared, cleared

        painted, _ = split(v._draw_list())
        assert max(painted) > 20             # the strip really did fill the panel

        v._toggle_list_group_collapsed(0, 60)
        collapsed_painted, cleared = split(v._draw_list())
        assert max(collapsed_painted) < 15   # and really did shrink

        # Every row it gave up is blanked, not left holding a stale 'frame N'.
        assert not (painted - collapsed_painted - cleared)

        # Expanding again refills them rather than clearing what now belongs.
        v._toggle_list_group_collapsed(0, 60)
        repainted, still_cleared = split(v._draw_list())
        assert painted <= repainted
        assert not (repainted & still_cleared)
    finally:
        os.close(fd)


def test_collapsed_summary_gets_sticky_filename_and_live_frame_count(tmp_path):
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset = StructureSet()
    for i in range(3):
        sset.append(mol, label=f"a.xyz#{i + 1}", path="/d/a.xyz")
    for i in range(12):
        sset.append(mol, label=f"b.xyz#{i + 1}", path="/d/b.xyz")
    fd = os.open(str(tmp_path / "collapse-sticky.bin"),
                 os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._rows, v._list_w = 14, 28
        v._toggle_list_group_collapsed(0, 3)
        # Adding another frame to the same source updates the summary rather
        # than freezing the count at collapse time.
        sset.entries.insert(3, sset.entries[2].__class__(
            molecule=mol, label="a.xyz#4", path="/d/a.xyz"))
        assert v._list_display_rows()[1] == ("collapsed", 0, "4 frames")

        v._list_scroll_to(1)  # summary visible, its ordinary header scrolled off
        v._draw_list()
        sticky = [span for span in v._list_group_all_spans if span[0] == 1]
        assert len(sticky) == 1
        assert sticky[0][3:5] == (0, 4)
    finally:
        os.close(fd)


def test_viewer_list_single_structure_files_get_overlay_sections(tmp_path):
    """Each file gets its own section when several files are open, even when
    that file contributes only one structure."""
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
        assert text.count("methane.xyz") == 1 and text.count("water.xyz") == 1
        assert text.count("frame 1") == 2
        # One per file, plus the global ALL on the "STRUCTURES N" header
        # (design VIM-28).
        assert text.count("ALL") == 3
        assert "/data/" not in text               # basenames only
        assert len(v._list_row_spans) == 2
        rows = v._list_display_rows()
        assert [kind for kind, _i, _t in rows] == [
            "group", "struct", "group", "struct"]
    finally:
        os.close(fd)


def test_viewer_list_climbs_parent_tree_until_file_headers_are_unique(tmp_path):
    """Duplicate basenames gain only as much parent context as is needed.

    These two paths still collide at ``shared/mol.xyz``, so disambiguation
    must climb a second time.  An already-unique filename stays compact.
    """
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="mol.xyz", path="/project/run-a/shared/mol.xyz")
    sset.append(mol, label="mol.xyz~2", path="/project/run-b/shared/mol.xyz")
    sset.append(mol, label="water.xyz", path="/project/unique/water.xyz")
    fd = os.open(str(tmp_path / "unique-file-headers.bin"),
                 os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        rows = v._list_display_rows()
        headers = [text for kind, _i, text in rows if kind == "group"]
        assert headers == [
            "run-a/shared/mol.xyz",
            "run-b/shared/mol.xyz",
            "water.xyz",
        ]
    finally:
        os.close(fd)


def test_list_filename_hover_shows_original_relative_path_and_frame(tmp_path):
    from vimol.input import MouseEvent

    original = "relative/input/run/traj.xyz"
    v, fd = _traj_viewer(tmp_path, n=3, path=original,
                         fd_name="relative-path-hover.bin")
    try:
        v._list_w = 24  # force the visible header to truncate
        v._draw_list()

        group = next(span for span in v._list_path_hover_spans
                     if span[3] == original)
        assert v._dispatch([MouseEvent(
            "move", float(group[1]), float(group[0]), button=0)]) is True
        assert original in _visible(v._status_bar())

        frame_tip = f"{original}#frame 2"
        frame = next(span for span in v._list_path_hover_spans
                     if span[3] == frame_tip)
        assert v._dispatch([MouseEvent(
            "move", float(frame[1]), float(frame[0]), button=0)]) is True
        assert frame_tip in _visible(v._status_bar())
    finally:
        os.close(fd)


def test_list_path_hover_preserves_absolute_path_and_clears_off_label(tmp_path):
    from vimol.input import MouseEvent

    original = "/full/project/calculations/traj.xyz"
    v, fd = _traj_viewer(tmp_path, n=2, path=original,
                         fd_name="absolute-path-hover.bin")
    try:
        v._list_w = 24
        v._draw_list()
        group = next(span for span in v._list_path_hover_spans
                     if span[3] == original)
        v._dispatch([MouseEvent(
            "move", float(group[1]), float(group[0]), button=0)])
        assert v._list_path_hover_tip == original

        # The ALL control belongs to the file but is not its filename.
        all_span = v._list_group_all_spans[0]
        assert v._dispatch([MouseEvent(
            "move", float(all_span[1]), float(all_span[0]), button=0)]) is True
        assert v._list_path_hover_tip == ""
    finally:
        os.close(fd)


def test_sticky_file_header_hover_keeps_original_path(tmp_path):
    from vimol.input import MouseEvent

    original = "runs/long/traj.xyz"
    v, fd = _traj_viewer(tmp_path, n=30, path=original,
                         fd_name="sticky-path-hover.bin")
    try:
        v._rows = 14
        v._list_w = 24
        v._list_scroll = 8
        v._draw_list()
        sticky = next(span for span in v._list_path_hover_spans
                      if span[0] == 1 and span[3] == original)
        assert v._dispatch([MouseEvent(
            "move", float(sticky[1]), float(sticky[0]), button=0)]) is True
        assert v._list_path_hover_tip == original
    finally:
        os.close(fd)


def test_viewer_list_mixed_tree_keeps_rows_and_structures_aligned(tmp_path):
    """A grouped file followed by a lone one: the offset between display rows
    and structure indices CHANGES partway down the list, which is exactly
    where row arithmetic drifts. Clicks and the cursor must still address
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
            "group", "struct", "struct", "struct", "group", "struct"]
        assert rows[-1][1:] == (3, "frame 1")
        assert rows[1][2] == "frame 1"

        v._draw_list()
        assert v._list_row_struct == [0, 1, 2, 3]
        row0, col_start, _c1 = v._list_row_spans[3]
        assert v._list_index_at_row(row0) == 3
        assert v._dispatch([MouseEvent("down", float(col_start), float(row0),
                                       button=0)]) is True
        assert v.frame_index == 3

        # the cursor addresses structures, not display rows: the click above
        # left it on 3, and one 'k' must step to structure 2 -- not to the
        # group header that sits between them as a display row.
        v._list_focused = True
        assert v._list_cursor == 3
        assert v._dispatch([KeyEvent("k")]) is True
        assert v._list_cursor == 2
        v._draw_list()
        assert 2 in v._list_row_struct
    finally:
        os.close(fd)


def test_viewer_list_group_all_button_fills_then_clears_to_main(tmp_path):
    """ALL fills a partial file selection; its selected colour and next click
    then clear the file while retaining the untinted main frame."""
    from vimol.input import MouseEvent

    v, fd = _traj_viewer(tmp_path, n=3)
    try:
        v.structures.overlay = True
        v._list_w = 28
        partial = v._draw_list().decode("utf-8", "replace")
        assert len(v._list_group_all_spans) == 1
        row, c0, c1, first, end = v._list_group_all_spans[0]
        assert (first, end) == (0, 3)
        partial_style = _sgr_bg(v.theme.list_cap_bg)
        selected_style = _sgr_bg(v.theme.measure_col_bg_a)
        assert partial_style in partial

        click = MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)
        assert v._dispatch([click]) is True
        assert [entry.marked for entry in v.structures] == [True, True, True]
        selected = v._draw_list().decode("utf-8", "replace")
        assert selected_style in selected

        row, c0, c1, _first, _end = v._list_group_all_spans[0]
        click = MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)
        assert v._dispatch([click]) is True
        assert [entry.marked for entry in v.structures] == [True, False, False]
        assert v.structures.drawn_indices() == [0]
        assert v.structures.overlay is True
        assert "main frame only" in v._msg
    finally:
        os.close(fd)


def test_group_all_button_is_right_aligned_regardless_of_filename_length(tmp_path):
    """The ALL badge sits flush against the strip's right edge (design
    VIM-27): a short and a long filename must land their buttons in exactly
    the same column instead of the button trailing right after the name."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="a.xyz", path="/data/a.xyz")
    sset.append(mol, label="a-much-longer-filename-here.xyz",
                path="/data/a-much-longer-filename-here.xyz")
    fd = os.open(str(tmp_path / "all-align.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 28
        v._draw_list()
        assert len(v._list_group_all_spans) == 2
        cols = {c0 for _row, c0, _c1, _first, _end in v._list_group_all_spans}
        assert len(cols) == 1                     # same start column both times
        # The per-file × (design VIM-32) sits flush against the strip's
        # right edge; the ALL button ends exactly where the × begins.
        assert len(v._list_group_remove_spans) == 2
        x_cols = {c0 for _row, c0, _c1, _first, _end in v._list_group_remove_spans}
        assert len(x_cols) == 1
        _row, _c0, x_c1, _first, _end = v._list_group_remove_spans[0]
        assert x_c1 == v._list_w
        _row, _c0, all_c1, _first, _end = v._list_group_all_spans[0]
        assert all_c1 == next(iter(x_cols))
    finally:
        os.close(fd)


def test_global_all_button_fills_then_clears_every_file(tmp_path):
    """The global ALL (design VIM-28) reuses the per-file fill/clear logic
    over the whole set: it must actually flip StructureSet.overlay/marked
    (not just repaint), and clearing always retains the main frame."""
    from vimol.input import MouseEvent
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="a.xyz", path="/d/a.xyz")
    sset.append(mol, label="b.xyz", path="/d/b.xyz")
    sset.append(mol, label="c.xyz", path="/d/c.xyz")
    fd = os.open(str(tmp_path / "global-all.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 28
        v._draw_list()
        assert v._global_all_span is not None
        row, c0, c1 = v._global_all_span
        assert row == 0

        click = MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)
        assert v._dispatch([click]) is True
        assert [e.marked for e in v.structures] == [True, True, True]
        assert v.structures.overlay is True
        assert "every structure" in v._msg

        v._draw_list()
        row, c0, c1 = v._global_all_span
        click = MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)
        assert v._dispatch([click]) is True
        assert [e.marked for e in v.structures] == [True, False, False]
        assert "main frame only" in v._msg
    finally:
        os.close(fd)


def test_global_all_button_has_no_remove_control(tmp_path):
    """The × only ever belongs to a single removable file (design VIM-32) --
    the global ALL on row 0 must never register one."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="a.xyz", path="/d/a.xyz")
    sset.append(mol, label="b.xyz", path="/d/b.xyz")
    fd = os.open(str(tmp_path / "global-all-no-x.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 28
        v._draw_list()
        assert all(row != 0 for row, _c0, _c1, _first, _end
                  in v._list_group_remove_spans)
    finally:
        os.close(fd)


def test_global_all_button_yields_to_the_scroll_marker_when_narrow(tmp_path):
    """The scroll marker is the more load-bearing affordance (design §4.1):
    on a strip too narrow for both, the global ALL button simply doesn't
    render rather than crowding the marker out."""
    v, fd = _traj_viewer(tmp_path, n=60, fd_name="global-all-narrow.bin")
    try:
        v._list_w = 18   # _LIST_W_MIN -- the narrowest the strip ever gets
        head = v._draw_list().decode("utf-8", "replace").split("\x1b[2;1H")[0]
        assert "↓" in head
        assert v._global_all_span is None
    finally:
        os.close(fd)


def _click_group_remove(v, first: int):
    """Click the × of the file group starting at structure index *first*."""
    row, c0, c1, _first, _end = next(
        span for span in v._list_group_remove_spans if span[3] == first)
    from vimol.input import MouseEvent
    return v._dispatch([MouseEvent(
        "down", float((c0 + c1) // 2), float(row), button=0)])


def test_group_remove_x_deletes_all_frames_of_that_file(tmp_path):
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    for i in range(3):
        sset.append(mol, label=f"a.xyz#{i + 1}", path="/d/a.xyz")
    sset.append(mol, label="b.xyz", path="/d/b.xyz")
    fd = os.open(str(tmp_path / "remove-file.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 28
        v._draw_list()

        assert _click_group_remove(v, 0) is True
        assert len(v.structures) == 1
        assert v.structures[0].label == "b.xyz"
        assert v.structures.active_index == 0
        assert "a.xyz" in v._msg and "removed" in v._msg
    finally:
        os.close(fd)


def test_group_remove_reindexes_rmsd_columns_across_the_gap(tmp_path):
    """Both RMSD column kinds store absolute entry indices. Removing a file
    that sits before a column's reference must shift it down, not leave it
    pointing at whatever slid into the old slot."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer, _FullRMSDColumn, _SubsetRMSDColumn

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="doomed1", path="/d/doomed.xyz")
    sset.append(mol, label="doomed2", path="/d/doomed.xyz")
    sset.append(mol, label="ref", path="/d/ref.xyz")
    sset.append(mol, label="other", path="/d/other.xyz")
    fd = os.open(str(tmp_path / "reindex.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 28
        v._full_rmsd_columns.append(_FullRMSDColumn(
            full_id=1, reference_index=2, reference_revision=0,
            values=[None, None, 0.0, 0.4]))
        v._rmsd_columns.append(_SubsetRMSDColumn(
            select_id=1, reference_index=2, reference_revision=0,
            indices=(0,), labels=("C0",), values=[None, None, 0.0, 0.5]))
        v._draw_list()

        assert _click_group_remove(v, 0) is True     # deletes doomed1+doomed2
        assert [e.label for e in v.structures] == ["ref", "other"]

        full = v._full_rmsd_columns[0]
        assert full.reference_index == 0
        assert full.values == [0.0, 0.4]
        subset = v._rmsd_columns[0]
        assert subset.reference_index == 0
        assert subset.values == [0.0, 0.5]
    finally:
        os.close(fd)


def test_group_remove_drops_rmsd_column_whose_reference_was_removed(tmp_path):
    """A column anchored on the file being removed no longer means anything
    -- it must be dropped, not linger with a reference index into thin air."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer, _FullRMSDColumn

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="ref", path="/d/ref.xyz")
    sset.append(mol, label="other", path="/d/other.xyz")
    fd = os.open(str(tmp_path / "drop-column.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._list_w = 28
        v._full_rmsd_columns.append(_FullRMSDColumn(
            full_id=1, reference_index=0, reference_revision=0,
            values=[0.0, 0.4]))
        v._active_subset_id = None
        v._draw_list()

        assert _click_group_remove(v, 0) is True     # removes "ref" itself
        assert v._full_rmsd_columns == []
    finally:
        os.close(fd)


def test_group_remove_unrelated_file_keeps_active_editor_state(tmp_path):
    """Removing a file that is not the active one must not discard the
    active file's dirty flag or undo history -- refitting the camera and
    resetting editor state is only for when the active structure itself
    changes identity."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    sset.append(mol, label="active.xyz", path="/d/active.xyz")
    sset.append(mol, label="other.xyz", path="/d/other.xyz")
    fd = os.open(str(tmp_path / "unrelated-remove.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd, editable=True)
        v._update_geometry()
        v._list_w = 28
        v.widget.dirty = True
        v.widget._undo_stack.append(("marker",))
        v._draw_list()

        assert _click_group_remove(v, 1) is True     # removes "other.xyz"
        assert len(v.structures) == 1
        assert v.widget.dirty is True
        assert v.widget._undo_stack == [("marker",)]
    finally:
        os.close(fd)


def test_group_remove_last_file_exits_without_prompt_when_clean(tmp_path):
    """A single-structure viewer never shows the strip at all (len <= 1), so
    the only way to reach "the last file" through the × is a lone file that
    still has 2+ frames -- removing its whole group empties the pane."""
    v, fd = _traj_viewer(tmp_path, n=3, fd_name="last-file-clean.bin")
    try:
        v._running = True
        v._list_w = 28
        v._draw_list()

        _click_group_remove(v, 0)
        assert v._running is False
        assert v._mode != "quit_confirm"
    finally:
        os.close(fd)


def test_group_remove_last_file_prompts_when_dirty(tmp_path):
    """Removing the last file with unsaved edits pending must ask first --
    the same gate 'Escape' already uses -- rather than silently discarding
    them by exiting outright."""
    v, fd = _traj_viewer(tmp_path, n=3, fd_name="last-file-dirty.bin")
    try:
        # _traj_viewer builds a read-only Viewer; editing must be on for
        # dirty/quit_confirm to mean anything.
        v.editable = True
        v._running = True
        v._list_w = 28
        v.widget.dirty = True
        v._draw_list()

        _click_group_remove(v, 0)
        assert v._mode == "quit_confirm"
        assert v._running is True             # still alive, waiting for y/n/Esc
    finally:
        os.close(fd)


def test_viewer_list_keeps_current_file_header_sticky_while_scrolled(tmp_path):
    v, fd = _traj_viewer(tmp_path, n=20)
    try:
        v._list_w = 28
        v._list_scroll_to(8)
        text = v._draw_list().decode("utf-8", "replace")
        assert "traj.xyz" in text
        assert "ALL" in text
        assert len(v._list_group_all_spans) == 1
        row, _c0, _c1, first, end = v._list_group_all_spans[0]
        assert row == 1
        assert (first, end) == (0, 20)
        assert all(span[0] >= 2 for span in v._list_row_spans)
    finally:
        os.close(fd)


def test_viewer_list_sticky_header_follows_scrolling_into_the_next_file(tmp_path):
    """The sticky row names whichever file the top visible frame belongs to,
    not whichever header happened to be first. It must also stand down while
    that file's own header is still on screen, or the name appears twice."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    for k in range(20):
        sset.append(mol, label=f"a.xyz#{k}", path="/d/a.xyz")
    for k in range(20):
        sset.append(mol, label=f"b.xyz#{k}", path="/d/b.xyz")
    # Pathless and unnumbered: gets no header of its own, so the nearest
    # header above it belongs to somebody else.
    sset.append(mol, label="scratch")
    for k in range(20):
        sset.append(mol, label=f"c.xyz#{k}", path="/d/c.xyz")
    fd = os.open(str(tmp_path / "sticky-two-file.bin"),
                 os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(mol, structures=sset, fd_out=fd)
        v._update_geometry()
        v._rows = 14
        v._list_w = 28
        rows = v._list_display_rows()

        def sticky_name(display_row):
            v._list_scroll_to(display_row)
            assert v._list_scroll == display_row      # not clamped short
            v._draw_list()
            spans = [s for s in v._list_group_all_spans if s[0] == 1]
            return v._list_group_name(spans[0][3], True) if spans else None

        def row_of(structure_index):
            return next(r for r, (kind, i, _t) in enumerate(rows)
                        if kind == "struct" and i == structure_index)

        assert sticky_name(row_of(4)) == "a.xyz"      # inside the first file
        assert sticky_name(row_of(24)) == "b.xyz"     # inside the second

        # The sticky copy carries a live ALL button, not a picture of one --
        # and it must fill the file it names rather than the first file.
        from vimol.input import MouseEvent
        row, c0, c1, first, end = [
            s for s in v._list_group_all_spans if s[0] == 1][0]
        assert (first, end) == (20, 40)
        assert v._dispatch([
            MouseEvent("down", float((c0 + c1) // 2), float(row), button=0)]) is True
        assert all(v.structures[i].marked for i in range(20, 40))
        assert not any(v.structures[i].marked for i in range(1, 20))
        # b.xyz's own header is the top row, so nothing may be duplicated.
        assert rows[row_of(20) - 1][0] == "group"
        assert sticky_name(row_of(20) - 1) is None

        # Naming b.xyz over a row b.xyz does not own is worse than naming
        # nothing, so the orphan row must clear the sticky header entirely.
        orphan = row_of(40)
        assert rows[orphan - 1][0] == "struct"        # no header introduced it
        assert sticky_name(orphan) is None
    finally:
        os.close(fd)


def test_viewer_list_path_names_terminate_on_indistinguishable_paths(tmp_path):
    """Disambiguation climbs until names differ, so two spellings of the same
    file have no parent left to climb to. That must end, not spin."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    def headers(paths):
        sset = StructureSet()
        mol = vimol.load(os.path.join(EX, "methane.xyz"))
        for k, path in enumerate(paths):
            sset.append(mol, label=f"m{k}", path=path)
        fd = os.open(str(tmp_path / "paths.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            v = Viewer(mol, structures=sset, fd_out=fd)
            names = v._list_path_display_names()
            return [names[p] for p in paths]
        finally:
            os.close(fd)

    # Same file, two spellings: no parent node distinguishes them.
    assert headers(["/a/b/x.xyz", "/a//b/x.xyz"]) == ["/a/b/x.xyz", "/a//b/x.xyz"]
    # A root-level file has no parent directory to prepend either.
    assert headers(["/mol.xyz", "mol.xyz"]) == ["/mol.xyz", "mol.xyz"]
    assert headers(["/x.xyz", "/y/x.xyz"]) == ["/x.xyz", "y/x.xyz"]


def test_viewer_first_real_geometry_restores_startup_file_header(tmp_path):
    """frame_index is assigned while terminal height is still zero. The real
    first geometry pass must undo that placeholder-capacity scroll."""
    from vimol.viewer import Viewer

    frames = [vimol.load(os.path.join(EX, "methane.xyz")) for _ in range(8)]
    fd = os.open(str(tmp_path / "startup-header.bin"),
                 os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(frames[0], frames=frames, source_path="traj.xyz", fd_out=fd)
        v.frame_index = 0
        assert v._list_scroll == 1       # one-row placeholder hid the header
        v._update_geometry()
        assert v._list_scroll == 0
        text = v._draw_list().decode("utf-8", "replace")
        assert "traj.xyz" in text and "ALL" in text
        assert v._list_group_all_spans[0][0] == 2  # normal header, not sticky fallback
    finally:
        os.close(fd)


def test_viewer_list_group_all_does_not_claim_a_main_frame_it_does_not_own(tmp_path):
    """Clearing a file that does not contain the main frame hides it outright.
    Reporting 'main frame only' there would credit the file with a row that
    lives in a different file, and the strip would show it contributing none."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    for label, path in (("a.xyz#1", "/data/a.xyz"), ("a.xyz#2", "/data/a.xyz"),
                        ("b.xyz#1", "/data/b.xyz"), ("b.xyz#2", "/data/b.xyz")):
        sset.append(vimol.load(os.path.join(EX, "methane.xyz")),
                    label=label, path=path).marked = True
    sset.active_index = 0                        # main frame lives in a.xyz
    sset.overlay = True
    fd = os.open(str(tmp_path / "twofile.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(sset[0].molecule, structures=sset, fd_out=fd)
        v._update_geometry()

        v._toggle_list_group_all(2, 4)           # clear b.xyz
        assert [entry.marked for entry in sset] == [True, True, False, False]
        assert sset.drawn_indices() == [0, 1]
        assert v._msg == "b.xyz: hidden"

        v._toggle_list_group_all(0, 2)           # clear a.xyz, which owns it
        assert sset.drawn_indices() == [0]
        assert v._msg == "a.xyz: main frame only"
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


# -- measurement table (design 2026-07-30 rev. 2, VIM-6) --------------------

def _measure_mol(dx=0.0):
    from vimol.molecule import Molecule
    # 5 distinct elements: the 5th (Cl) is only used to click "one atom past
    # a completed dihedral" and trigger the widget's own 5th-click reset.
    return Molecule(symbols=["C", "N", "O", "F", "Cl"], positions=np.array([
        [0.0, 0.0, 0.0],
        [1.0 + dx, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]))


def _measure_viewer(tmp_path, fd_name="measure.bin"):
    """Two same-topology structures (a.xyz, b.xyz -- so a frozen column
    resolves for both) plus a third with different elements (c.xyz, the
    degrade case)."""
    from vimol.molecule import Molecule
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    sset.append(_measure_mol(0.0), label="a.xyz")
    sset.append(_measure_mol(0.5), label="b.xyz")
    sset.append(Molecule(symbols=["O", "H"], positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])),
                label="c.xyz")
    fd = os.open(str(tmp_path / fd_name), os.O_WRONLY | os.O_CREAT, 0o644)
    v = Viewer(sset[0].molecule, structures=sset, fd_out=fd)
    v._update_geometry()
    v._list_w = 40
    return v, fd


def test_collapsed_summary_uses_local_minimum_in_every_measurement_column(tmp_path):
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer, _FullRMSDColumn, _SubsetRMSDColumn

    sset = StructureSet()
    for i, dx in enumerate((0.0, 0.5, 1.0, 1.5)):
        path = "/data/a.xyz" if i < 2 else "/data/b.xyz"
        sset.append(_measure_mol(dx), label=f"frame {i + 1}", path=path)
    fd = os.open(str(tmp_path / "collapsed-measures.bin"),
                 os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(sset[0].molecule, structures=sset, fd_out=fd)
        v._cols, v._rows, v._list_w = 200, 30, 40
        v._freeze_measure_sel((0, 1))
        v._full_rmsd_columns.append(_FullRMSDColumn(
            full_id=1, reference_index=0, reference_revision=0,
            values=[0.0, 0.4, 0.3, 0.2]))
        v._rmsd_columns.append(_SubsetRMSDColumn(
            select_id=1, reference_index=0, reference_revision=0,
            indices=(0, 1), labels=("C0", "N1"),
            values=[0.0, 0.5, None, 0.1]))
        v._toggle_list_group_collapsed(2, 4)

        layout = v._measure_layout(v._list_w)
        assert v._list_collapsed_measure_cells(layout, 2, 4) == [
            "2.000↓", "0.20↓", "0.10↓"]

        v._draw_list()
        row = v._list_group_summary_spans[0][0]
        summary = _visible(_strip_rows(v)[row])
        assert "2 frames" in summary
        assert "2.000↓" in summary
        assert "0.20↓" in summary
        assert "0.10↓" in summary
    finally:
        os.close(fd)


def test_collapsed_summary_measurement_is_dash_when_group_has_no_values(tmp_path):
    v, fd = _measure_viewer(tmp_path, fd_name="collapsed-dash.bin")
    try:
        layout = [("test", 4, ["1.0↓", "—", "—"], False)]
        assert v._list_collapsed_measure_cells(layout, 1, 3) == ["—"]
    finally:
        os.close(fd)


def _atom_px_in_viewer(v, idx):
    """Screen-pixel location of ACTIVE-LOCAL atom *idx*, using the same
    analytic projection as pick() -- offset by the viewer's own image
    origin so it can be dispatched through v._dispatch (which subtracts
    that origin back out before it ever reaches the widget)."""
    w = v.widget
    mol = w.molecule
    cam = w.scene.camera
    ss = w.scene.supersample
    Wr, Hr = w.scene.render_size
    vp = cam.view_positions(mol.positions[idx:idx + 1])[0]
    ox_s = Wr * 0.5 + cam.pan[0]
    oy_s = Hr * 0.5 - cam.pan[1]
    sx = ox_s + vp[0] * cam.zoom
    sy = oy_s - vp[1] * cam.zoom
    return sx / ss + v._img_origin_px[0], sy / ss + v._img_origin_px[1]


def _click_atom(v, idx):
    from vimol.input import MouseEvent
    x, y = _atom_px_in_viewer(v, idx)
    v._dispatch([MouseEvent("down", x, y, button=0, pixel=True)])
    return v._dispatch([MouseEvent("up", x, y, button=0, pixel=True)])


def _click_empty(v, local_x=500.0, local_y=30.0):
    """A widget-local pixel with no atom under it -- near the TOP of the
    image (not the bottom): a low local_y keeps the terminal row well clear
    of _in_status_zone's margin, which would otherwise swallow the click
    before it ever reaches the widget."""
    from vimol.input import MouseEvent
    x, y = local_x + v._img_origin_px[0], local_y + v._img_origin_px[1]
    v._dispatch([MouseEvent("down", x, y, button=0, pixel=True)])
    return v._dispatch([MouseEvent("up", x, y, button=0, pixel=True)])


def test_measure_live_column_appears_at_two_picks_and_updates_in_place(tmp_path):
    """No commit key: the live pick renders as soon as it holds 2 atoms, and
    extending it to 3/4 updates the SAME column (still exactly one), not a
    second one -- distance -> angle -> dihedral of the SAME atoms."""
    from vimol.input import KeyEvent

    v, fd = _measure_viewer(tmp_path)
    try:
        v._dispatch([KeyEvent("m")])
        assert v._measure_layout(v._list_w) == []      # 0-1 picks: nothing yet

        v.widget.measure_sel = [0, 1]                  # as if two atoms were clicked
        layout = v._measure_layout(v._list_w)
        assert len(layout) == 1
        header_cell, _w, _cells, removable = layout[0]
        assert header_cell == "C0-N1"                  # no × -- nothing to remove yet
        assert removable is False

        v.widget.measure_sel = [0, 1, 2]                # extends -> same column, now an angle
        layout = v._measure_layout(v._list_w)
        assert len(layout) == 1
        assert layout[0][0] == "∠C0-N1-O2"

        v.widget.measure_sel = [0, 1, 2, 3]             # extends again -> a dihedral
        layout = v._measure_layout(v._list_w)
        assert len(layout) == 1
        assert layout[0][0] == "φC0-N1-O2-F3"
    finally:
        os.close(fd)


def test_measure_new_atom_click_after_completed_dihedral_freezes_old_and_starts_fresh(tmp_path):
    """A fresh atom click once the pick already holds 4 is the widget's own
    reset (not a continuation): the finished dihedral freezes, and the
    clicked atom starts a brand new (as yet sub-2, so invisible) pick."""
    v, fd = _measure_viewer(tmp_path)
    try:
        v.widget.set_measure_mode(True)
        for idx in (0, 1, 2, 3):
            assert _click_atom(v, idx)
        assert v.widget.measure_sel == [0, 1, 2, 3]
        assert v._measure_columns == []                 # not frozen yet -- still live

        assert _click_atom(v, 4)                         # 5th distinct atom: widget resets
        assert v.widget.measure_sel == [4]
        assert len(v._measure_columns) == 1
        header, indices = v._measure_columns[0]
        assert indices == (0, 1, 2, 3)
        assert header == "φC0-N1-O2-F3"
        # the frozen column still renders; the new 1-atom pick has nothing
        # live to show ON TOP OF it yet (needs 2+ to appear)
        layout = v._measure_layout(v._list_w)
        assert len(layout) == 1
        assert layout[0][0] == "φC0-N1-O2-F3 ×"
        assert layout[0][3] is True                       # removable: it's frozen
    finally:
        os.close(fd)


def test_measure_empty_space_click_freezes_pending_selection(tmp_path):
    v, fd = _measure_viewer(tmp_path)
    try:
        v.widget.set_measure_mode(True)
        assert _click_atom(v, 0)
        assert _click_atom(v, 1)
        assert v.widget.measure_sel == [0, 1]

        assert _click_empty(v)                           # empty corner: clears
        assert v.widget.measure_sel == []
        assert len(v._measure_columns) == 1
        assert v._measure_columns[0][1] == (0, 1)
    finally:
        os.close(fd)


def test_measure_recommitting_same_indices_is_not_duplicated(tmp_path):
    v, fd = _measure_viewer(tmp_path)
    try:
        v._freeze_measure_sel((0, 1))
        v._freeze_measure_sel((0, 1))
        assert len(v._measure_columns) == 1
    finally:
        os.close(fd)


def test_measure_mode_off_freezes_the_live_pick(tmp_path):
    from vimol.input import KeyEvent

    v, fd = _measure_viewer(tmp_path)
    try:
        v._dispatch([KeyEvent("m")])
        v.widget.measure_sel = [0, 1]                   # as if two atoms were clicked
        assert v._dispatch([KeyEvent("m")]) is True      # toggling off freezes it first
        assert v.widget.measure_mode is False
        assert len(v._measure_columns) == 1
        assert v._measure_columns[0][1] == (0, 1)
    finally:
        os.close(fd)


def test_measure_switching_active_structure_freezes_the_live_pick(tmp_path):
    """The whole point of the table is comparing across frames -- switching
    frames mid-measurement must freeze the pick, not silently drop it."""
    from vimol.input import KeyEvent

    v, fd = _measure_viewer(tmp_path)
    try:
        v._dispatch([KeyEvent("m")])
        v.widget.measure_sel = [0, 1]
        assert v._dispatch([KeyEvent("n")]) is True
        assert v.frame_index == 1
        assert v.widget.measure_sel == []                # cleared for the new active structure
        assert len(v._measure_columns) == 1
        assert v._measure_columns[0][1] == (0, 1)
    finally:
        os.close(fd)


def test_measure_header_glyphs_for_distance_angle_and_dihedral(tmp_path):
    v, fd = _measure_viewer(tmp_path)
    try:
        assert v._measure_header_text((0, 1)) == "C0-N1"
        assert v._measure_header_text((0, 1, 2)) == "∠C0-N1-O2"
        assert v._measure_header_text((0, 1, 2, 3)) == "φC0-N1-O2-F3"
    finally:
        os.close(fd)


def test_measure_table_renders_values_and_degrades_for_mismatched_topology(tmp_path):
    v, fd = _measure_viewer(tmp_path)
    try:
        v._freeze_measure_sel((0, 1))
        layout = v._measure_layout(v._list_w)
        assert layout[0][2] == ["1.000↓", "1.500↑", "—"]
        text = v._draw_list().decode("utf-8", "replace")
        assert "C0-N1 ×" in text
        # active (a.xyz) and b.xyz share topology -> real numbers;
        # c.xyz (O/H) does not -> degrades to the em-dash
        assert "1.000" in text     # a.xyz: |C-N| == 1.0
        assert "1.500" in text     # b.xyz: dx=0.5 -> |C-N| == 1.5
        assert "—" in text
        # The ANSI underline is applied only to each marked value, once for
        # the minimum and once for the maximum.
        assert text.count("\x1b[4m") == 2
    finally:
        os.close(fd)


def test_measure_extrema_are_reviewed_when_a_structure_row_is_added(tmp_path):
    """A pinned column must not retain the old maximum after it gains a row."""
    v, fd = _measure_viewer(tmp_path)
    try:
        v._freeze_measure_sel((0, 1))
        assert v._measure_layout(v._list_w)[0][2] == [
            "1.000↓", "1.500↑", "—"]

        v.structures.append(_measure_mol(1.0), label="d.xyz")
        cells = v._measure_layout(v._list_w)[0][2]
        assert cells == ["1.000↓", "1.500", "—", "2.000↑"]
    finally:
        os.close(fd)


def test_measure_extrema_marks_all_ties_and_ignores_missing_values():
    from vimol.viewer import Viewer

    assert Viewer._format_measure_extrema([2.0, None, 2.0], 3) == [
        "2.000↑↓", "—", "2.000↑↓"]


def test_measure_extrema_treats_non_finite_values_as_missing():
    """A NaN or an infinity must not become the column's marked maximum, and
    must not drag every real value into looking like a joint minimum."""
    from vimol.viewer import Viewer

    cells = Viewer._format_measure_extrema(
        [1.0, float("nan"), float("inf"), float("-inf"), 3.0], 3)
    assert cells == ["1.000↓", "—", "—", "—", "3.000↑"]
    # A column of nothing but non-finite values has no extrema at all.
    assert Viewer._format_measure_extrema(
        [float("nan"), float("inf")], 3) == ["—", "—"]


def test_measure_click_header_x_removes_frozen_column(tmp_path):
    from vimol.input import MouseEvent

    v, fd = _measure_viewer(tmp_path)
    try:
        v._freeze_measure_sel((0, 1))
        v._draw_list()
        assert len(v._measure_header_spans) == 1
        row0, col_start, _c1, col_idx = v._measure_header_spans[0]
        assert col_idx == 0
        assert v._dispatch([MouseEvent("down", float(col_start), float(row0),
                                       button=0)]) is True
        assert v._measure_columns == []
    finally:
        os.close(fd)


def test_measure_live_column_has_no_removal_span(tmp_path):
    """The live column has no × -- there is nothing to remove yet, it
    resolves on its own once the pick moves on."""
    v, fd = _measure_viewer(tmp_path)
    try:
        v.widget.set_measure_mode(True)
        v.widget.measure_sel = [0, 1]
        v._draw_list()
        assert v._measure_header_spans == []
    finally:
        os.close(fd)


def test_measure_row_click_under_value_column_still_activates_structure(tmp_path):
    from vimol.input import MouseEvent

    v, fd = _measure_viewer(tmp_path)
    try:
        v._freeze_measure_sel((0, 1))
        v._draw_list()
        row0, _c0, c1 = v._list_row_spans[1]     # b.xyz's row
        click_col = c1 - 1                       # inside the measurement portion
        assert click_col >= v._list_w            # sanity: really past the strip
        assert v._dispatch([MouseEvent("down", float(click_col), float(row0),
                                       button=0)]) is True
        assert v.frame_index == 1
    finally:
        os.close(fd)


def test_measure_columns_interleave_two_background_tints(tmp_path):
    from vimol import theme as _theme

    v, fd = _measure_viewer(tmp_path)
    try:
        v.theme = _theme.DARK
        v._cols = 200   # ample room: two columns must not overflow-truncate here
        v._freeze_measure_sel((0, 1))
        v._freeze_measure_sel((0, 1, 2))
        text = v._draw_list().decode("utf-8", "replace")
        assert _sgr_bg(_theme.DARK.measure_col_bg_a) in text
        assert _sgr_bg(_theme.DARK.measure_col_bg_b) in text
    finally:
        os.close(fd)


def test_measure_columns_committed_but_not_rendered_with_a_single_structure():
    from vimol.molecule import Molecule
    from vimol.viewer import Viewer

    mol = Molecule(symbols=["C", "N"], positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu")
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    v._freeze_measure_sel((0, 1))
    assert len(v._measure_columns) == 1              # freezing itself isn't gated
    assert v._measure_layout(v._list_w) == []         # nothing to compare against, so no table
    assert v._measure_width(v._list_w) == 0


def test_measure_columns_overflow_are_dropped_not_corrupting_the_viewport(tmp_path, monkeypatch):
    """Enough pinned columns to exceed the terminal width must not drive
    _img_cols to zero/negative or widen rows past the terminal (design
    2026-07-30) -- excess columns are silently dropped instead.

    _update_geometry re-queries the fd's actual (ioctl-fallback) size on
    every call, so a plain ``v._cols = N`` doesn't stick across a second
    call -- the terminal width has to be faked at the source."""
    from vimol import kitty as _kitty
    from vimol.viewer import _MEASURE_MIN_VIEWPORT_COLS

    v, fd = _measure_viewer(tmp_path)
    try:
        monkeypatch.setattr(_kitty, "terminal_size_px", lambda fd: (60, 24, 0, 0))
        for sel in ((0, 1), (0, 1, 2), (0, 1, 2, 3)):
            v._freeze_measure_sel(sel)
        assert len(v._measure_columns) == 3          # all three frozen...
        v._update_geometry()
        assert v._cols == 60
        assert v._img_cols > 0
        assert v._list_w + v._measure_w + _MEASURE_MIN_VIEWPORT_COLS <= v._cols
        layout = v._measure_layout(v._list_w)
        assert len(layout) < 3                       # ...but not all three fit
        for row_text in _strip_rows(v).values():
            assert len(_visible(row_text)) <= v._cols
    finally:
        os.close(fd)


def test_measure_geometry_updates_immediately_after_a_column_appears(tmp_path):
    """Regression test: _refresh_measure_w syncing _measure_w immediately
    (for an in-burst hit test) must not stop _update_geometry from noticing
    the width actually changed -- otherwise _img_cols/_img_origin_px go
    stale and the mouse-to-image mapping drifts out from under a redraw
    that already shows the new column (design 2026-07-30)."""
    v, fd = _measure_viewer(tmp_path)
    try:
        img_cols0, origin0 = v._img_cols, v._img_origin_px
        v._freeze_measure_sel((0, 1))
        assert v._geometry_dirty is True
        assert v._update_geometry() is True          # must still report a change
        assert v._img_cols < img_cols0                # viewport shrank to make room
        assert v._img_origin_px[0] > origin0[0]        # image moved right with it
        assert v._geometry_dirty is False
    finally:
        os.close(fd)


def test_measure_columns_align_across_rows_with_differing_label_lengths(tmp_path):
    """Regression test: the label cell must be padded to label_w, not just
    truncated -- otherwise rows whose label text is shorter than label_w
    (nearly all of them) shift their measurement columns left of the
    header, and rows disagree with each other whenever their label
    lengths differ (design 2026-07-30)."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer

    sset = StructureSet()
    for i in range(11):
        # labels of differing length ("frame 1" vs "frame 10") are the
        # whole point: a plain truncate (no pad) drifts between them.
        sset.append(_measure_mol(0.01 * i), label=f"frame {i + 1}")
    fd = os.open(str(tmp_path / "align.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    v = Viewer(sset[0].molecule, structures=sset, fd_out=fd)
    v._update_geometry()
    v._list_w = 40
    try:
        v._freeze_measure_sel((0, 1))
        layout = v._measure_layout(v._list_w)
        _header_cell, width, cells, _removable = layout[0]
        rows = _strip_rows(v)
        col0 = v._list_w + 1                      # +1: measure_segs' leading pad space
        for (row0, _c0, _c1), entry_i in zip(v._list_row_spans, v._list_row_struct):
            expected = cells[entry_i].rjust(width)
            actual = _visible(rows[row0])[col0:col0 + width]
            assert actual == expected, (row0, entry_i, actual, expected)
    finally:
        os.close(fd)


def test_measure_live_column_suppressed_when_it_matches_a_frozen_one(tmp_path):
    """Re-picking the same atom pair after it's already frozen (e.g. on a
    new active structure while trajectory-browsing) must not show a
    duplicate header alongside the frozen one until the live pick moves
    on (design 2026-07-30)."""
    v, fd = _measure_viewer(tmp_path)
    try:
        v._freeze_measure_sel((0, 1))
        v.widget.set_measure_mode(True)
        v.widget.measure_sel = [0, 1]
        layout = v._measure_layout(v._list_w)
        assert len(layout) == 1
        assert layout[0][0] == "C0-N1 ×"
        assert layout[0][3] is True
    finally:
        os.close(fd)


def _help_rows(v):
    """{0-based screen row: the SGR text drawn there} for one _draw_help()."""
    parts = re.split(r"\x1b\[(\d+);(\d+)H", v._draw_help().decode("utf-8", "replace"))
    return {int(parts[i]) - 1: parts[i + 2] for i in range(1, len(parts), 3)}


@pytest.mark.parametrize("editable", [False, True])
@pytest.mark.parametrize("cols,rows_h", [(120, 40), (80, 30), (60, 12), (30, 6), (8, 5)])
def test_viewer_help_panel_is_a_closed_box_of_uniform_width(tmp_path, editable,
                                                            cols, rows_h):
    """The old panel ljust()ed to a hardcoded 58 and never truncated, so the
    long lines spilled out of the tint and it read as having no edges. Every
    row must now measure the SAME visible width and start/end on a border --
    including at the sizes where the geometry has to clamp and clip."""
    v, fd = _multi_viewer(tmp_path, editable=editable)
    try:
        v._cols, v._rows = cols, rows_h
        rows = _help_rows(v)
        widths = {len(_visible(t)) for t in rows.values()}
        assert len(widths) == 1, widths
        top, bottom = min(rows), max(rows)
        assert _visible(rows[top]).startswith("┌")
        assert _visible(rows[top]).endswith("┐")
        assert _visible(rows[bottom]).startswith("└")
        assert _visible(rows[bottom]).endswith("┘")
        for r in range(top + 1, bottom):
            assert _visible(rows[r]).startswith("│")
            assert _visible(rows[r]).endswith("│")
        body = "".join(_visible(rows[r]) for r in range(top + 1, bottom))
        if (cols, rows_h) == (120, 40):                # only there is it whole
            assert "toggle this help" in body
            assert "focus the structure list" in body  # the strip keymap
    finally:
        os.close(fd)


@pytest.mark.parametrize("cols,rows_h", [(120, 40), (80, 30), (60, 12), (30, 6), (8, 5)])
def test_viewer_help_panel_never_reaches_the_status_bar(tmp_path, cols, rows_h):
    """Whatever the terminal size, the box stays on screen and off the status
    row -- and says '─ more ─' in its foot when it had to clip."""
    v, fd = _multi_viewer(tmp_path, editable=True)
    try:
        v._cols, v._rows = cols, rows_h
        drawn = _help_rows(v)
        assert drawn
        assert min(drawn) >= 0
        assert max(drawn) < rows_h - 1                  # status bar untouched
        assert all(len(_visible(t)) <= cols for t in drawn.values())
        from vimol.viewer import _help_lines
        clipped = len(_help_lines(True)) > len(drawn) - 2
        assert ("more" in _visible(drawn[max(drawn)])) is clipped
    finally:
        os.close(fd)


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
        for word in ("next/prev", "all keys"):
            assert word in _visible(below)
        # key caps: padded key text on a lighter background
        assert _sgr_bg((42, 49, 66)) in below
        for cap in (" n ", " p ", " ? "):
            assert cap in _visible(below)
        # ]/[ are the global roll keys; the legend must not advertise them
        # as the strip's next/prev any more (VIM-9). 'space'/'mark' are gone
        # with the mark concept: overlay membership is opt+click only. 'z'
        # and 1-9 are global keys (camera reset, representation) that the
        # strip must not shadow, so it must not advertise them either.
        for gone in (" ] ", " [ ", " space ", "mark", "solo", "hide",
                     " z ", " h ", " 1 ", " 9 ", "jump to"):
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


def test_viewer_list_focused_digits_still_set_the_representation(tmp_path):
    """1-9 used to jump the list cursor, which shadowed the global
    representation keys 1-4 whenever the strip had focus. The strip claims
    nothing there now; j/k and n/p already covered the jump."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        assert v._handle_list_key("3") is False    # unclaimed by the strip
        assert v._dispatch([KeyEvent("2")]) is True
        assert v.style.representation == "spacefill"
        assert v._list_cursor == 0                 # cursor untouched
        assert v.frame_index == 0                  # active structure unchanged
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


def test_viewer_list_legend_rows_match_the_reserved_footer(tmp_path):
    """Footer reservation equals separator + all legend/hint rows."""
    from vimol.viewer import _LIST_ROWS_BELOW

    v, fd = _multi_viewer(tmp_path)
    try:
        assert len(v._list_legend()) == 3
        assert _LIST_ROWS_BELOW == 1 + len(v._list_legend())
    finally:
        os.close(fd)


def test_viewer_list_focused_z_still_resets_the_camera(tmp_path):
    """'z' must never be shadowed by the list strip -- solo used to claim it
    there, silently breaking the global camera fit/reset binding whenever the
    strip had focus."""
    from vimol.viewer import Viewer
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path)
    try:
        v._list_focused = True
        v._list_cursor = 1
        assert v._handle_list_key("z") is False   # unclaimed by the strip
        v.widget.scene.camera.orbit(45, 30)
        assert v._dispatch([KeyEvent("z")]) is True
        assert np.array_equal(v.widget.scene.camera.rotation, np.eye(3))
        assert [e.visible for e in v.structures] == [True, True, True]
        # the strip's legend now advertises '?', so it had better reach the
        # driver through a focused strip too
        assert v._handle_list_key("?") is False
        assert v._dispatch([KeyEvent("?")]) is True
        assert v._show_help is True
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


# -- subset-alignment fixes (review 2026-07-31) --------------------------

def _overlay_viewer(tmp_path, **kw):
    """A three-structure overlay with editing on, ready for align tests."""
    v, fd = _multi_viewer(tmp_path, editable=True, **kw)
    v.structures.overlay = True
    return v, fd


def test_subset_column_disarms_when_the_reference_geometry_changes(tmp_path):
    """A saved column's atom indices belong to the revision they were picked
    at. Editing the reference shifts them, so an armed column must not stay
    armed and silently re-fit on whatever atoms now occupy those indices."""
    from vimol import editor

    v, fd = _overlay_viewer(tmp_path)
    try:
        v._finish_subset_alignment((0, 1, 2))
        column = v._rmsd_columns[0]
        v._activate_subset_column(0)
        assert v._selected_subset_column() is column

        entry = v.structures.active
        editor.delete_atom(entry.molecule, 4)
        entry.touch()

        assert v._selected_subset_column() is None
        assert v.widget.align_sel == []
        assert "RMSD#1" in v._msg
    finally:
        os.close(fd)


def test_subset_column_header_marks_itself_stale_after_an_edit(tmp_path):
    """The numbers stay on screen (they were true once) but must not read as
    current -- otherwise a stale RMSD is indistinguishable from a fresh one.
    The stale marker rides right after the id, before the ×."""
    from vimol import editor

    v, fd = _overlay_viewer(tmp_path)
    try:
        v._finish_subset_alignment((0, 1, 2))
        v._list_w = 40
        v._draw_list()
        header_cell = v._measure_layout(v._list_w)[0][0]
        assert header_cell == "⊂RMSD#1 ×"

        entry = v.structures.active
        editor.delete_atom(entry.molecule, 4)
        entry.touch()

        v._draw_list()
        header_cell = v._measure_layout(v._list_w)[0][0]
        assert header_cell == "⊂RMSD#1* ×"
    finally:
        os.close(fd)


def test_rmsd_columns_share_one_id_sequence_and_round_to_2_decimals(tmp_path):
    """Both column kinds render as "signRMSD#N" (design 2026-08-02): the ⊂/∀
    sign glyph stays (it's informative at a glance), but there's no more
    select/all word in the label -- that detail is hover-only (VIM-30).
    select_id/full_id share one counter so a ∀RMSD and a ⊂RMSD column can
    never be numbered alike. Values round to 2 decimals, keeping the column
    as narrow as the sign+id/× header allows."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer, _FullRMSDColumn, _SubsetRMSDColumn

    sset = StructureSet()
    sset.append(_measure_mol(0.0), label="a.xyz")
    sset.append(_measure_mol(0.123456), label="b.xyz")
    fd = os.open(str(tmp_path / "rmsd-naming.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(sset[0].molecule, structures=sset, fd_out=fd)
        v._cols, v._rows, v._list_w = 200, 30, 40
        v._full_rmsd_columns.append(_FullRMSDColumn(
            full_id=1, reference_index=0, reference_revision=0,
            values=[0.0, 0.123456]))
        v._rmsd_columns.append(_SubsetRMSDColumn(
            select_id=2, reference_index=0, reference_revision=0,
            indices=(0, 1), labels=("C0", "N1"), values=[0.0, 0.987654]))

        layout = v._measure_layout(v._list_w)
        assert [h for h, _w, _v, _r in layout] == ["∀RMSD#1 ×", "⊂RMSD#2 ×"]
        # The reference row reads "Self", not the em-dash that means "no
        # result here"; it is excluded from the extrema either way, so the
        # sole remaining value is trivially both.
        assert [cells for _h, _w, cells, _r in layout] == [
            ["Self", "0.12↑↓"], ["Self", "0.99↑↓"]]
        for header_cell, width, cells, _removable in layout:
            assert width == max(len(header_cell), max(len(c) for c in cells))

        text = v._draw_list().decode("utf-8", "replace")
        assert "∀RMSD#1 ×" in text and "⊂RMSD#2 ×" in text
    finally:
        os.close(fd)


def test_rmsd_reference_row_reads_self_and_a_never_run_row_stays_a_dash(tmp_path):
    """The whole point of "Self" (design 2026-08-02): a row that WAS the
    reference and a row that simply has no result must not look alike. Both
    used to render the same em-dash."""
    from vimol.structures import StructureSet
    from vimol.viewer import Viewer, _FullRMSDColumn

    sset = StructureSet()
    for i, dx in enumerate((0.0, 0.4, 0.8)):
        sset.append(_measure_mol(dx), label=f"f{i}.xyz", path=f"/d/f{i}.xyz")
    fd = os.open(str(tmp_path / "self-cell.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        v = Viewer(sset[0].molecule, structures=sset, fd_out=fd)
        v._cols, v._rows, v._list_w = 200, 30, 40
        # Row 1 was fitted; row 2 never was (it was not in the overlay).
        v._full_rmsd_columns.append(_FullRMSDColumn(
            full_id=1, reference_index=0, reference_revision=0,
            values=[0.0, 0.4, None]))

        cells = v._measure_layout(v._list_w)[0][2]
        assert cells == ["Self", "0.40↑↓", "—"]

        text = v._draw_list().decode("utf-8", "replace")
        assert "Self" in text
    finally:
        os.close(fd)


def test_option_click_arms_alignment_through_an_overlaid_atom(tmp_path):
    """_pick_active_only exists so a tinted overlay atom cannot intercept a
    subset pick. The option-click that ARMS picking has to use it too."""
    from vimol.input import MouseEvent

    v, fd = _overlay_viewer(tmp_path)
    try:
        active = v.structures[0].molecule
        other = v.structures[1].molecule
        # One active atom far from the camera, one overlay atom in front of
        # it at the same screen position; everything else off to the side.
        active.positions = np.array([[0.0, 0.0, -5.0]] + [[50.0, 0.0, 0.0]] * (active.n_atoms - 1))
        other.positions = np.array([[0.0, 0.0, 5.0]] + [[50.0, 0.0, 0.0]] * (other.n_atoms - 1))
        v.structures[0].marked = v.structures[1].marked = True
        for e in v.structures.entries:
            e.touch()
        v.structures.invalidate()
        v.widget.refresh_active()

        w = v.widget
        x, y = _atom_px_in_viewer(v, 0)
        lx, ly = w._local_px(MouseEvent("down", x, y, pixel=True), v._img_origin_px)
        assert w._active_local_pick(lx, ly) is None    # overlay wins the composite pick
        assert w._pick_active_only(lx, ly) == 0        # ... but the active atom is there

        v._dispatch([MouseEvent("down", x, y, button=0, alt=True, pixel=True)])
        v._dispatch([MouseEvent("up", x, y, button=0, alt=True, pixel=True)])
        assert w.align_mode is True
        assert w.align_sel == [0]
    finally:
        os.close(fd)


def test_enter_without_an_overlay_keeps_the_pick_instead_of_saving_a_column(tmp_path):
    """Picking before an overlay exists is deliberate (opt+click a row, then
    r). Enter must not turn that into a column of dashes -- at one loaded
    structure _measure_layout never draws it, so it could not be removed."""
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path, editable=True)
    try:
        v.structures.overlay = False
        v.widget.set_alignment_mode(True)
        v.widget.align_sel = [0, 1, 2]
        assert v._dispatch([KeyEvent("enter")]) is True

        assert v._rmsd_columns == []            # nothing saved ...
        assert v.widget.align_sel == [0, 1, 2]  # ... and the pick survives
        assert v.widget.align_mode is True
        assert "overlay" in v._msg
    finally:
        os.close(fd)


def test_r_outside_overlay_says_why_nothing_happened(tmp_path):
    """'r' used to reset the camera; it is now an overlay-only align key and
    reaches the widget no more. Silence reads as a broken keybinding."""
    from vimol.input import KeyEvent

    v, fd = _multi_viewer(tmp_path, editable=True)
    try:
        v.structures.overlay = False
        v._msg = ""
        v._dispatch([KeyEvent("r")])
        assert "overlay" in v._msg
    finally:
        os.close(fd)


def test_alignment_pick_ignores_a_hidden_active_structure(tmp_path):
    """_pick_active_only bypasses the composite on purpose, so it also
    bypasses visibility. Picking an atom that is not drawn arms a selection
    _apply_highlight cannot globalize -- composite.sources has no row for a
    hidden active structure, and colouring it raises IndexError."""
    from vimol.input import MouseEvent

    v, fd = _overlay_viewer(tmp_path)
    try:
        for e in v.structures.entries:
            e.marked = True
        v.structures.active.visible = False
        v.structures.invalidate()
        assert v.structures.active_index not in v.structures.drawn_indices()

        w = v.widget
        cx, cy = w.scene.width * 0.5, w.scene.height * 0.5
        assert w._pick_active_only(cx, cy) is None

        w.set_alignment_mode(True)
        v._dispatch([MouseEvent("down", cx, cy, button=0, alt=True, pixel=True)])
        v._dispatch([MouseEvent("up", cx, cy, button=0, alt=True, pixel=True)])
        assert w.align_sel == []
        w._apply_highlight()          # must not raise
    finally:
        os.close(fd)


# -- structure strip: scrolling must not rebuild the whole row layout --------
def _big_strip_viewer(tmp_path, n_frames=1200):
    """A viewer holding one big single-file trajectory, as the strip sees it."""
    from vimol.viewer import Viewer
    from vimol.structures import StructureSet
    from vimol import theme as _theme
    mol = vimol.load(os.path.join(EX, "water.xyz"))
    sset = StructureSet()
    for i in range(n_frames):
        sset.append(vimol.load(os.path.join(EX, "water.xyz")),
                    label=f"traj.xyz#{i + 1}", path=str(tmp_path / "traj.xyz"))
    fd = os.open(str(tmp_path / "out.bin"), os.O_WRONLY | os.O_CREAT, 0o644)
    v = Viewer(mol, structures=sset, fd_out=fd)
    v.theme = _theme.DARK
    v._cols, v._rows = 200, 50
    v._update_geometry()
    return v, fd


def test_scrolling_the_strip_does_not_rebuild_the_row_layout(tmp_path):
    """Scrolling changes which rows are shown, never what the rows ARE.

    The layout is O(n) with ~3 _list_group_key calls per entry, and the
    scroll path walks it twice per wheel notch (_list_max_scroll clamps with
    it, then _draw_list slices it). Rebuilding it per notch measured 13.9 ms
    at 5000 frames -- the reported "scrolling is very slow".

    Asserted as "the cost does not grow with the frame count" rather than as
    a wall-clock bound (which would flake) or as zero work (a few O(1)
    lookups per draw are legitimate -- the sticky header needs them).
    """
    def scroll_cost(n_frames):
        v, fd = _big_strip_viewer(tmp_path, n_frames=n_frames)
        try:
            v._draw_list()                  # prime the cache
            calls = []
            real = type(v)._list_group_key
            type(v)._list_group_key = lambda self, i: (calls.append(i), real(self, i))[1]
            try:
                for _ in range(10):
                    v._list_scroll_by(3)
                    v._draw_list()
            finally:
                type(v)._list_group_key = real
            return len(calls)
        finally:
            os.close(fd)

    small, large = scroll_cost(300), scroll_cost(1200)
    assert small == large, (
        f"scrolling cost grew with the frame count ({small} -> {large} "
        "group-key lookups for 4x the frames) -- the layout is being rebuilt")


def test_group_bounds_do_not_walk_the_group(tmp_path):
    """_list_group_end and _list_frame_path_tip used to rediscover a group's
    boundaries by walking it -- once per drawn row, O(n) each. They now read
    the bounds the cached layout pass already computed."""
    v, fd = _big_strip_viewer(tmp_path)
    try:
        v._draw_list()
        n = len(v.structures)
        assert v._list_group_end(0) == n          # one file: one group
        assert v._list_group_end(n // 2) == n
        assert v._list_frame_path_tip(0).endswith("#frame 1")
        assert v._list_frame_path_tip(41).endswith("#frame 42")

        calls = []
        real = type(v)._list_group_key
        type(v)._list_group_key = lambda self, i: (calls.append(i), real(self, i))[1]
        try:
            v._list_group_end(0)
            v._list_frame_path_tip(41)
        finally:
            type(v)._list_group_key = real
        assert not calls, f"{len(calls)} group-key calls to answer two lookups"
    finally:
        os.close(fd)


def test_row_layout_cache_follows_a_reassigned_path(tmp_path):
    """`label`/`path` decide how rows group, and saving under a new name
    reassigns `path` on the active entry. The cache key must notice."""
    v, fd = _big_strip_viewer(tmp_path, n_frames=6)
    try:
        before = v._list_display_rows()
        assert before[0][0] == "group"        # one file, so one header
        v.structures[0].path = str(tmp_path / "renamed.xyz")
        after = v._list_display_rows()
        assert after != before, "row layout served stale after a path change"
    finally:
        os.close(fd)


def test_row_layout_cache_follows_a_reassigned_label(tmp_path):
    v, fd = _big_strip_viewer(tmp_path, n_frames=6)
    try:
        for e in v.structures.entries:
            e.path = None                     # fall back to label-based grouping
        before = v._list_display_rows()
        v.structures[2].label = "unrelated"
        assert v._list_display_rows() != before, (
            "row layout served stale after a label change")
    finally:
        os.close(fd)


def test_row_layout_cache_follows_collapse_and_overlay(tmp_path):
    v, fd = _big_strip_viewer(tmp_path, n_frames=6)
    try:
        rows = v._list_display_rows()
        v._toggle_list_group_collapsed(0, 6)
        collapsed = v._list_display_rows()
        assert collapsed != rows
        assert any(r[0] == "collapsed" for r in collapsed)
        v._toggle_list_group_collapsed(0, 6)
        assert v._list_display_rows() == rows
    finally:
        os.close(fd)
