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


def test_camera_orbit_changes_view():
    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    scene = Scene(mol, 60, 60)
    before = scene.render().copy()
    scene.camera.orbit(90, 45)
    after = scene.render()
    assert not np.array_equal(before, after)


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
