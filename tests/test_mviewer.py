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


def test_viewer_input_handling_headless():
    """Drive the Viewer's input parser without a real TTY."""
    from mviewer.viewer import Viewer

    mol = mviewer.load(os.path.join(EX, "water.xyz"))
    v = Viewer(mol)
    v._cell_w, v._cell_h = 9.0, 18.0
    r0 = v.scene.camera.rotation.copy()
    # arrow key right
    v._handle(b"\x1b[C")
    assert not np.array_equal(r0, v.scene.camera.rotation)
    # representation switch
    v._handle(b"2")
    assert v.style.representation == "spacefill"
    # SGR mouse: press, drag, release
    v._handle(b"\x1b[<0;10;10M")
    assert v._dragging
    r1 = v.scene.camera.rotation.copy()
    v._handle(b"\x1b[<32;20;18M")  # motion with button held
    assert not np.array_equal(r1, v.scene.camera.rotation)
    v._handle(b"\x1b[<0;20;18m")
    assert not v._dragging
    # wheel zoom
    z = v.scene.camera.zoom
    v._handle(b"\x1b[<64;5;5M")
    assert v.scene.camera.zoom != z
    # quit
    v._running = True
    v._handle(b"q")
    assert v._running is False


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
