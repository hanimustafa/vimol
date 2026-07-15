"""Tests for interactive molecule editing: templates, editor, save, widget."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mviewer
from mviewer import elements, templates, editor
from mviewer.molecule import Molecule
from mviewer.bonds import ensure_bonds
from mviewer.parsers import xyz as xyz_parser
from mviewer.parsers import save, loads
from mviewer.widget import MoleculeWidget
from mviewer.input import MouseEvent, KeyEvent

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def _angle(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))


# -- templates -------------------------------------------------------------
def test_tetrahedral_directions_are_unit_and_109():
    t = templates.default_template("C")
    d = t.directions
    assert d.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9)
    # every pair of tetrahedral directions is ~109.47 deg apart
    for i in range(4):
        for j in range(i + 1, 4):
            assert abs(_angle(d[i], d[j]) - 109.47) < 0.1


def test_open_directions_orient_toward_parent():
    t = templates.default_template("C")
    attach = np.array([0.0, 0.0, 1.0])      # parent lies along +z
    opens = t.open_directions(attach)
    assert opens.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(opens, axis=1), 1.0, atol=1e-9)
    # each open site sits ~109.47 from the attachment direction and from the others
    for o in opens:
        assert abs(_angle(o, attach) - 109.47) < 0.1
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(_angle(opens[i], opens[j]) - 109.47) < 0.1


def test_registry_has_entry_per_valence_combo():
    assert templates.TEMPLATES[("N", 3)].geometry == "pyramidal"
    assert templates.TEMPLATES[("O", 2)].geometry == "bent"
    assert templates.default_template("O").valence == 2


def test_free_direction_avoids_existing_neighbors():
    # a single neighbor along +x -> free site should point roughly -x
    d = templates.free_direction([[1.0, 0.0, 0.0]])
    assert d[0] < 0
    # two opposed neighbors -> perpendicular, not a zero vector
    d2 = templates.free_direction([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.linalg.norm(d2) == pytest.approx(1.0, abs=1e-6)


# -- editor: birth ---------------------------------------------------------
def test_birth_methane():
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    assert mol.n_atoms == 5
    assert mol.symbols[c] == "C"
    assert mol.symbols.count("H") == 4
    assert mol.formula() == "CH4"
    # four C-H bonds at the covalent-sum distance
    assert len(mol.bonds) == 4
    ch = elements.covalent_radius("C") + elements.covalent_radius("H")
    for h in range(1, 5):
        assert np.linalg.norm(mol.positions[h] - mol.positions[0]) == pytest.approx(ch, abs=1e-6)


def test_birth_water_geometry():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0], element="O")
    assert mol.formula() == "H2O"


# -- editor: grow ----------------------------------------------------------
def test_grow_promotes_hydrogen_to_methyl():
    # ethane-ish start: one C with a single H sticking out along +x
    mol = Molecule(symbols=["C", "H"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]))
    ensure_bonds(mol)
    editor.grow_at_atom(mol, 1)          # click the H
    # H (index 1) became a carbon; three fresh H were added
    assert mol.symbols[1] == "C"
    assert mol.symbols.count("C") == 2
    assert mol.symbols.count("H") == 3
    # the new carbon sits at the proper C-C distance from the parent carbon
    cc = 2 * elements.covalent_radius("C")
    assert np.linalg.norm(mol.positions[1] - mol.positions[0]) == pytest.approx(cc, abs=1e-6)
    # and it is bonded to the parent carbon
    assert (0, 1, 1) in mol.bonds


def test_grow_on_heavy_atom_attaches_methyl():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.grow_at_atom(mol, 0)          # click the lone carbon
    assert mol.symbols.count("C") == 2
    assert mol.symbols.count("H") == 3   # new carbon capped with 3 H


def test_lone_hydrogen_becomes_methane():
    mol = Molecule(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.grow_at_atom(mol, 0)
    assert mol.formula() == "CH4"


# -- xyz writer ------------------------------------------------------------
def test_xyz_dumps_roundtrip():
    mol = Molecule()
    editor.birth_molecule(mol, [1.0, 2.0, 3.0])
    mol.name = "test methane"
    text = xyz_parser.dumps(mol)
    back = loads(text, "xyz")
    assert back.n_atoms == mol.n_atoms
    assert back.symbols == mol.symbols
    np.testing.assert_allclose(back.positions, mol.positions, atol=1e-6)
    assert back.name == "test methane"


def test_save_dispatch_and_bad_format(tmp_path):
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    p = tmp_path / "out.xyz"
    save(mol, str(p))
    assert p.exists()
    assert mviewer.load(str(p)).formula() == "CH4"
    with pytest.raises(ValueError):
        save(mol, str(tmp_path / "out.pdb"))    # writing PDB is unsupported


# -- widget append ---------------------------------------------------------
def _click(widget, x, y):
    """Simulate a left down+up at the same pixel (a click, not a drag)."""
    widget.handle_event(MouseEvent("down", x, y, button=0, pixel=True))
    return widget.handle_event(MouseEvent("up", x, y, button=0, pixel=True))


def test_widget_click_empty_space_births_molecule():
    mol = Molecule()                      # empty scene
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    assert not w.dirty
    changed = _click(w, 100, 100)         # click center
    assert changed
    assert w.dirty
    assert w.molecule.formula() == "CH4"


def test_widget_not_editable_ignores_append():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu")   # editable defaults False
    w.set_append_mode(True)
    assert not w.append_mode               # cannot enter append mode
    _click(w, 100, 100)
    assert w.molecule.n_atoms == 0         # click builds nothing
    assert not w.dirty


def test_widget_undo_reverts_edits():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    _click(w, 100, 100)                     # birth CH4
    n1 = w.molecule.n_atoms
    _click(w, 100, 100)                     # second click grows onto the methane
    n2 = w.molecule.n_atoms
    assert n2 > n1 and w.dirty
    assert w.undo()                         # back to one methane
    assert w.molecule.n_atoms == n1
    assert w.undo()                         # back to empty
    assert w.molecule.n_atoms == 0
    assert not w.dirty                      # undone all the way -> clean again
    assert not w.undo()                     # nothing left to undo


def test_widget_undo_dirty_after_save_baseline():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    _click(w, 100, 100)
    w.mark_saved()                          # current state is "on disk"
    assert not w.dirty
    w.undo()                                # diverge from the saved state
    assert w.dirty


def test_widget_unproject_matches_pick_center():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    # the atom is at the world origin == camera center; unprojecting the screen
    # point that picks it must land back near the origin.
    Wr, Hr = w.scene.render_size
    ss = w.scene.supersample
    cx = (Wr * 0.5) / ss
    cy = (Hr * 0.5) / ss
    assert w.pick(cx, cy) == 0
    world = w.unproject(cx, cy)
    np.testing.assert_allclose(world, [0.0, 0.0, 0.0], atol=1e-6)


# -- viewer save prompt ----------------------------------------------------
def _new_viewer(tmp_path, source=None):
    from mviewer.viewer import Viewer
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    return Viewer(mol, backend="cpu", source_path=source, editable=True)


def _type(v, text):
    for ch in text:
        v._handle_prompt_key(ch)


def test_save_prompt_new_file_saves_directly(tmp_path):
    v = _new_viewer(tmp_path)
    v.widget.dirty = True
    v._open_save_prompt()
    assert v._mode == "save_input"
    target = tmp_path / "fresh.xyz"
    v._input_buf = ""
    _type(v, str(target))
    v._handle_prompt_key("enter")
    assert v._mode == "normal"            # brand-new file: saved, no confirm
    assert target.exists()
    assert not v.widget.dirty
    assert "saved" in v._msg


def test_save_prompt_existing_file_asks_to_replace(tmp_path):
    target = tmp_path / "exists.xyz"
    target.write_text("0\n\n")
    v = _new_viewer(tmp_path)
    v._open_save_prompt()
    v._input_buf = str(target)
    v._handle_prompt_key("enter")
    assert v._mode == "save_confirm"      # existing file: must confirm
    v._handle_prompt_key("n")             # decline -> back to editing the name
    assert v._mode == "save_input"
    v._handle_prompt_key("enter")
    v._handle_prompt_key("y")             # confirm replace
    assert v._mode == "normal"
    assert mviewer.load(str(target)).formula() == "CH4"


def test_save_prompt_default_path_uses_source(tmp_path):
    v = _new_viewer(tmp_path, source="/some/where/mol.xyz")
    v._open_save_prompt()
    assert v._input_buf == "/some/where/mol.xyz"


def test_save_prompt_escape_cancels(tmp_path):
    v = _new_viewer(tmp_path)
    v.widget.dirty = True
    v._open_save_prompt()
    v._handle_prompt_key("escape")
    assert v._mode == "normal"
    assert v.widget.dirty                 # cancel leaves the model dirty


def test_viewer_dispatch_routes_edit_keys(tmp_path):
    from mviewer.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    # 'a' toggles append mode (not autospin)
    v._dispatch([KeyEvent("a")])
    assert v.widget.append_mode and not v.autospin
    # 'o' toggles autospin (relocated from 'a')
    v._dispatch([KeyEvent("o")])
    assert v.autospin
    # a click in empty space through the full dispatch path edits the model
    v._dispatch([MouseEvent("down", 20, 20, button=0, pixel=True),
                 MouseEvent("up", 20, 20, button=0, pixel=True)])
    assert v.widget.dirty
    # 'u' undoes the edit back to the lone carbon
    v._dispatch([KeyEvent("u")])
    assert v.widget.molecule.n_atoms == 1 and not v.widget.dirty


def test_viewer_readonly_keeps_classic_bindings():
    from mviewer.viewer import Viewer
    mol = mviewer.load(os.path.join(EX, "methane.xyz"))
    v = Viewer(mol, backend="cpu")               # editable defaults False
    # 'a' is autospin (classic), NOT append
    v._dispatch([KeyEvent("a")])
    assert v.autospin and not v.widget.append_mode
    # 's' is NOT claimed as save -> it reaches the widget and cycles the style
    rep0 = v.style.representation
    v._dispatch([KeyEvent("s")])
    assert v._mode == "normal"                   # no save prompt opened
    assert v.style.representation != rep0        # representation cycled instead
    # 'u' does nothing in read-only mode
    assert not v._dispatch([KeyEvent("u")])


def test_viewer_edit_buttons_render_element_and_geometry():
    from mviewer.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_append_mode(True)
    bar = v._status_bar()
    assert "adding" in bar
    assert "C" in bar and "tetrahedral" in bar   # element + geometry buttons
    assert "\x1b[48;2;" in bar                    # colored (button) backgrounds
    # read-only viewer shows no build buttons
    v2 = Viewer(mol, backend="cpu")
    assert "tetrahedral" not in v2._status_bar()


def test_widget_drag_does_not_edit():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    w.handle_event(MouseEvent("down", 100, 100, button=0, pixel=True))
    w.handle_event(MouseEvent("drag", 140, 100, button=0, pixel=True))
    w.handle_event(MouseEvent("up", 140, 100, button=0, pixel=True))
    assert not w.dirty                    # a drag rotates, it must not build
    assert w.molecule.n_atoms == 0
