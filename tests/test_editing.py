"""Tests for interactive molecule editing: templates, editor, save, widget."""
import os
import sys
import re

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vimol
from vimol import elements, templates, editor, periodic_table as pt
from vimol.molecule import Molecule
from vimol.bonds import ensure_bonds
from vimol.parsers import xyz as xyz_parser
from vimol.parsers import save, loads
from vimol.widget import MoleculeWidget
from vimol.input import MouseEvent, KeyEvent

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
    assert vimol.load(str(p)).formula() == "CH4"
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
    from vimol.viewer import Viewer
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
    assert vimol.load(str(target)).formula() == "CH4"


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
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
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
    from vimol.viewer import Viewer
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
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
    from vimol.viewer import Viewer
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


# -- periodic-table layout --------------------------------------------------
def test_periodic_table_covers_all_118_elements_once():
    seen = set()
    n_gaps = 0
    for row in pt.GRID:
        for cell in row:
            if cell is None:
                continue
            if cell.symbol is None:
                n_gaps += 1
                continue
            assert cell.symbol not in seen, f"duplicate {cell.symbol}"
            seen.add(cell.symbol)
    assert n_gaps == 2   # La-Lu and Ac-Lr placeholders
    assert seen == set(elements.SYMBOLS[1:])


def test_periodic_table_known_positions():
    assert pt.position_of("H") == (0, 0)
    assert pt.position_of("He") == (0, 17)
    assert pt.position_of("C") == (1, 13)
    assert pt.position_of("Og") == (6, 17)
    assert pt.position_of("La") == (7, 3)
    assert pt.position_of("Lu") == (7, 17)
    assert pt.position_of("Ac") == (8, 3)
    assert pt.position_of("Lr") == (8, 17)
    assert pt.position_of("not-a-real-element") == pt.position_of("C")  # fallback


def test_periodic_table_gap_cells_jump_to_f_block_rows():
    gap_ln = pt.cell_at(5, 2)
    gap_an = pt.cell_at(6, 2)
    assert gap_ln.symbol is None and gap_ln.jump_row == 7
    assert gap_an.symbol is None and gap_an.jump_row == 8


def test_element_name():
    assert elements.element_name("C") == "Carbon"
    assert elements.element_name("Og") == "Oganesson"
    assert elements.element_name("Fe") == "Iron"


# -- periodic-table picker (Viewer) -----------------------------------------
def _viewer_in_append_mode(cols=100, rows=30):
    from vimol.viewer import Viewer
    mol = Molecule()
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(240, 200)
    v._cols, v._rows = cols, rows
    v._dispatch([KeyEvent("a")])
    v._status_bar()   # populate v._elem_button_span
    return v


def test_status_bar_element_button_span_only_in_append_mode():
    v = _viewer_in_append_mode()
    assert v._elem_button_span is not None
    row, c0, c1 = v._elem_button_span
    assert row == v._rows - 1
    assert c1 - c0 == len(f" {v.widget.build_element} ")

    from vimol.viewer import Viewer
    v2 = Viewer(Molecule(), backend="cpu", editable=True)
    v2.widget.set_pixel_size(240, 200)
    v2._cols, v2._rows = 100, 30
    v2._status_bar()   # append mode never toggled on
    assert v2._elem_button_span is None


def test_click_element_button_opens_picker_at_current_element():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    changed = v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert changed and v._mode == "periodic_table"
    assert (v._pt_row, v._pt_col) == pt.position_of(v.widget.build_element)


def test_click_geometry_pill_opens_geometry_picker_not_periodic_table():
    v = _viewer_in_append_mode()
    row, gc0, gc1 = v._geom_button_span
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "geometry_picker"


def test_click_between_and_past_the_pills_opens_nothing():
    v = _viewer_in_append_mode()
    row, _ec0, ec1 = v._elem_button_span
    _grow, gc0, gc1 = v._geom_button_span
    # the single space between the two pills, and well past the geometry pill
    for col in (ec1, gc1 + 6):
        if ec1 < gc0:                    # ensure there really is a gap column
            v._dispatch([MouseEvent("down", col, row, button=0, pixel=False)])
            assert v._mode == "normal"


def test_picker_keyboard_navigation_and_select():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "periodic_table"

    v._dispatch([KeyEvent("right")])
    v._dispatch([KeyEvent("down")])
    target = pt.GRID[v._pt_row][v._pt_col]
    assert target is not None and target.symbol is not None

    v._dispatch([KeyEvent("enter")])
    assert v._mode == "normal"
    assert v.widget.build_element == target.symbol


def test_picker_escape_cancels_without_changing_element():
    v = _viewer_in_append_mode()
    before = v.widget.build_element
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    v._dispatch([KeyEvent("right")])   # move around first
    v._dispatch([KeyEvent("escape")])
    assert v._mode == "normal"
    assert v.widget.build_element == before


def test_picker_gap_cell_jumps_without_selecting():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    v._pt_row, v._pt_col = 5, 2   # the La-Lu gap
    v._dispatch([KeyEvent("enter")])
    assert v._mode == "periodic_table"   # jump, not select -- picker stays open
    landed = pt.GRID[v._pt_row][v._pt_col]
    assert landed.symbol == "La"


def test_picker_mouse_click_selects_cell_under_cursor():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    top, left, _w, _h = v._pt_geometry()
    screen_row = top + 1   # grid row 0
    screen_col = left + 2 + 17 * 4 + 1   # grid col 17 (He)
    v._dispatch([MouseEvent("down", screen_col, screen_row, button=0, pixel=False)])
    assert v._mode == "normal"
    assert v.widget.build_element == "He"


def test_picker_click_outside_panel_is_a_no_op():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    changed = v._dispatch([MouseEvent("down", 0, 0, button=0, pixel=False)])
    assert not changed
    assert v._mode == "periodic_table"


def test_picker_anchors_above_the_button_not_screen_center():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    top, left, width, height = v._pt_geometry()
    # flush against the row just above the status bar, not vertically centered
    assert top + height - 1 == v._rows - 2
    # horizontally centered on the button, not on the whole screen width
    button_center = (c0 + c1) // 2
    assert abs((left + width // 2) - button_center) <= 1
    screen_center = v._cols // 2
    assert left + width // 2 != screen_center


def test_status_bar_button_column_stable_across_hover_and_name_changes():
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C", "O"], positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(240, 200)
    v._cols, v._rows = 100, 30
    v._dispatch([KeyEvent("a")])
    v._status_bar()
    baseline = v._elem_button_span
    assert baseline is not None

    v.widget.hovered = 0
    v._status_bar()
    assert v._elem_button_span == baseline, "hovering a short-symbol atom must not move the button"

    v.widget.hovered = 1
    v._status_bar()
    assert v._elem_button_span == baseline

    v.widget.hovered = None
    mol.name = "a-much-longer-molecule-name-than-before"
    v._status_bar()
    assert v._elem_button_span == baseline, "a longer molecule name must not move the button"

    v._msg = "saved something.xyz"
    v._status_bar()
    assert v._elem_button_span == baseline, "a transient status message must not move the button"


# -- status-bar dead zone: protect it from accidental 3D-viewport clicks ---
def test_status_zone_click_opens_picker_even_one_row_off():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row - 1, button=0, pixel=False)])
    assert v._mode == "periodic_table"


def test_status_zone_near_miss_click_never_births_an_atom():
    v = _viewer_in_append_mode()
    status_row = v._rows - 1
    for click_row in (status_row, status_row - 1):
        # column 2 is the status bar's left (hover/molecule-info) area -- in the
        # dead zone but on neither pill, so it must be swallowed silently.
        v._dispatch([MouseEvent("down", 2, click_row, button=0, pixel=False)])
        v._dispatch([MouseEvent("up", 2, click_row, button=0, pixel=False)])
        assert v._mode == "normal"
        assert v.widget.molecule.n_atoms == 0


def test_status_zone_does_not_swallow_normal_viewport_clicks():
    v = _viewer_in_append_mode()
    v._dispatch([MouseEvent("down", 20, 5, button=0, pixel=False)])
    v._dispatch([MouseEvent("up", 20, 5, button=0, pixel=False)])
    assert v.widget.molecule.n_atoms == 5   # a fresh methane, built normally


def test_status_zone_drag_stays_suppressed_even_if_it_enters_the_viewport():
    v = _viewer_in_append_mode()
    rot_before = v.widget.scene.camera.rotation.copy()
    v._dispatch([MouseEvent("down", 5, v._rows - 1, button=0, pixel=False)])
    v._dispatch([MouseEvent("drag", 5, 10, button=0, pixel=False)])
    v._dispatch([MouseEvent("up", 5, 10, button=0, pixel=False)])
    assert (v.widget.scene.camera.rotation == rot_before).all()


def test_viewport_drag_starting_outside_the_zone_still_rotates():
    v = _viewer_in_append_mode()
    rot_before = v.widget.scene.camera.rotation.copy()
    v._dispatch([MouseEvent("down", 5, 10, button=0, pixel=False)])
    v._dispatch([MouseEvent("drag", 40, 10, button=0, pixel=False)])
    v._dispatch([MouseEvent("up", 40, 10, button=0, pixel=False)])
    assert not (v.widget.scene.camera.rotation == rot_before).all()


# -- clicking the pill again toggles the picker closed ----------------------
def test_clicking_the_same_pill_closes_the_picker():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "periodic_table"
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "normal"
    assert v.widget.build_element == "C"   # unchanged, just closed


def test_pill_toggle_full_cycle_and_row_tolerance():
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "periodic_table"
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "normal"
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "periodic_table"
    # closing also tolerates the same one-row margin as opening does
    v._dispatch([MouseEvent("down", c0 + 0.5, row - 1, button=0, pixel=False)])
    assert v._mode == "normal"


def test_cell_size_px_rounds_to_whole_pixels(monkeypatch):
    import vimol.kitty as kitty
    # a window size that would otherwise produce a fractional cell height
    monkeypatch.setattr(kitty, "terminal_size_px", lambda fd=1: (80, 24, 726, 434))
    cw, ch = kitty.cell_size_px()
    assert cw == float(round(726 / 80))
    assert ch == float(round(434 / 24))
    assert cw == int(cw) and ch == int(ch)


def test_query_cell_size_px_parses_csi_16t_reply(monkeypatch):
    import vimol.kitty as kitty
    import os as _os
    import select as _select
    reply = {"data": b"\x1b[6;38;19t"}
    monkeypatch.setattr(_os, "write", lambda fd, data: len(data))
    monkeypatch.setattr(_select, "select", lambda r, w, x, t: ([r[0]], [], []))

    def fake_read(fd, n):
        d = reply["data"]
        reply["data"] = b""
        return d
    monkeypatch.setattr(_os, "read", fake_read)
    cw, ch = kitty.query_cell_size_px(0, 1)
    assert (cw, ch) == (19.0, 38.0)   # reply is height;width -> (w, h)


def test_query_cell_size_px_returns_none_without_reply(monkeypatch):
    import vimol.kitty as kitty
    import os as _os
    import select as _select
    monkeypatch.setattr(_os, "write", lambda fd, data: len(data))
    monkeypatch.setattr(_select, "select", lambda r, w, x, t: ([], [], []))  # never ready
    assert kitty.query_cell_size_px(0, 1, timeout=0.01) is None


# -- taller status-bar dead zone (guard pushed higher) ----------------------
def test_status_zone_guard_blocks_several_rows_above_status():
    from vimol.viewer import _STATUS_ZONE_ROWS
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    for dr in range(_STATUS_ZONE_ROWS):
        click_row = v._rows - 1 - dr
        v._dispatch([MouseEvent("down", c1 + 6, click_row, button=0, pixel=False)])
        v._dispatch([MouseEvent("up", c1 + 6, click_row, button=0, pixel=False)])
        assert v.widget.molecule.n_atoms == 0, f"stray atom at row offset {dr}"


def test_click_just_above_the_zone_still_builds():
    from vimol.viewer import _STATUS_ZONE_ROWS
    v = _viewer_in_append_mode()
    click_row = v._rows - 1 - _STATUS_ZONE_ROWS   # first row outside the guard
    v._dispatch([MouseEvent("down", 20, click_row, button=0, pixel=False)])
    v._dispatch([MouseEvent("up", 20, click_row, button=0, pixel=False)])
    assert v.widget.molecule.n_atoms == 5


# -- closing the picker lifts only its rows (no full-screen repaint) --------
def test_update_geometry_prefers_queried_cell_size():
    from vimol.viewer import Viewer
    v = Viewer(Molecule(), backend="cpu", editable=True)
    v._cell_px = (11.0, 23.0)          # as if the terminal answered CSI 16 t
    v._update_geometry()
    assert v.widget.cell_w == 11.0 and v.widget.cell_h == 23.0


def test_closing_picker_erases_only_panel_rows(monkeypatch):
    import vimol.kitty as kitty
    v = _viewer_in_append_mode()
    row, c0, c1 = v._elem_button_span
    v._dispatch([MouseEvent("down", c0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "periodic_table"
    top, _left, _w, height = v._pt_geometry()

    captured = bytearray()
    monkeypatch.setattr(kitty, "write_bytes", lambda data, fd: captured.extend(data))
    v._dispatch([KeyEvent("escape")])
    text = bytes(captured).decode("utf-8", "replace")

    assert "\x1b[2J" not in text, "close must not full-clear the screen"
    erased = sorted(int(m) for m in re.findall(r"\x1b\[(\d+);1H\x1b\[0m\x1b\[2K", text))
    assert erased, "panel rows should be erased"
    assert erased[0] == top + 1                     # first panel row (1-based)
    assert max(erased) <= v._rows - 1               # never wipes the status row


# -- template registry: per-element geometry options -----------------------
def test_options_for_lists_hybridizations_most_bonds_first():
    labels = [o.geometry for o in templates.options_for("C")]
    assert labels == ["tetrahedral", "trigonal", "linear"]     # sp3, sp2, sp
    valences = [o.valence for o in templates.options_for("N")]
    assert valences == sorted(valences, reverse=True)          # descending
    assert templates.options_for("O")[0].geometry == "bent"


def test_options_for_unknown_element_falls_back_to_one_default():
    opts = templates.options_for("Fe")
    assert len(opts) == 1
    assert opts[0].element == "Fe"


def test_template_label_reads_naturally():
    t = templates.TEMPLATES[("C", 4)]
    assert t.label() == "tetrahedral · sp3 · 4 bonds"
    assert templates.TEMPLATES[("H", 1)].label() == "terminal · 1 bond"   # no hybridization


# -- geometry picker (Viewer) ----------------------------------------------
def test_click_geometry_pill_opens_picker_at_active_option():
    v = _viewer_in_append_mode()
    row, gc0, gc1 = v._geom_button_span
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "geometry_picker"
    assert v._geom_opts[v._geom_idx].geometry == "tetrahedral"   # C's default


def test_geometry_picker_select_sets_build_template_and_affects_building():
    v = _viewer_in_append_mode()
    row, gc0, gc1 = v._geom_button_span
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    v._dispatch([KeyEvent("down")])                 # tetrahedral -> trigonal (sp2)
    chosen = v._geom_opts[v._geom_idx]
    v._dispatch([KeyEvent("enter")])
    assert v._mode == "normal"
    assert v.widget.build_template is chosen
    # building now uses sp2 carbon: a fresh atom is capped with 3 H, not 4
    v._dispatch([MouseEvent("down", 20, 5, button=0, pixel=False)])
    v._dispatch([MouseEvent("up", 20, 5, button=0, pixel=False)])
    assert v.widget.molecule.formula() == "CH3"


def test_geometry_picker_escape_leaves_template_unchanged():
    v = _viewer_in_append_mode()
    before = v.widget.build_template
    row, gc0, gc1 = v._geom_button_span
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    v._dispatch([KeyEvent("down")])
    v._dispatch([KeyEvent("escape")])
    assert v._mode == "normal"
    assert v.widget.build_template is before


def test_clicking_geometry_pill_again_toggles_it_closed():
    v = _viewer_in_append_mode()
    row, gc0, gc1 = v._geom_button_span
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "geometry_picker"
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    assert v._mode == "normal"


def test_picking_new_element_resets_geometry_to_its_default():
    v = _viewer_in_append_mode()
    # choose a non-default geometry for carbon first
    row, gc0, gc1 = v._geom_button_span
    v._dispatch([MouseEvent("down", gc0 + 0.5, row, button=0, pixel=False)])
    v._dispatch([KeyEvent("down")])
    v._dispatch([KeyEvent("enter")])
    assert v.widget.build_template is not None
    # now pick oxygen from the periodic table -> template resets to O's default
    erow, ec0, ec1 = v._elem_button_span
    v._dispatch([MouseEvent("down", ec0 + 0.5, erow, button=0, pixel=False)])
    v._pt_row, v._pt_col = pt.position_of("O")
    v._dispatch([KeyEvent("enter")])
    assert v.widget.build_element == "O"
    assert v.widget.build_template is None            # reset to default
    assert v._active_template().geometry == "bent"    # O's default
