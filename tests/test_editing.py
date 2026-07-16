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
from vimol import app as vimol_app

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


def test_click_heavy_atom_replaces_it():
    # lone carbon, oxygen selected -> the C itself becomes a capped O (water)
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.grow_at_atom(mol, 0, element="O")
    assert mol.formula() == "H2O"
    assert mol.symbols[0] == "O"          # replaced in place, not grown onto


def test_lone_hydrogen_becomes_methane():
    mol = Molecule(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.grow_at_atom(mol, 0)
    assert mol.formula() == "CH4"


# -- editor: replace --------------------------------------------------------
def test_replace_relabels_in_place_hypervalent():
    # CH4 carbon -> N (valence 3): coordination 4 >= 3, so relabel only
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    assert editor.replace_atom(mol, c, "N", templates.TEMPLATES[("N", 3)]) == c
    assert mol.symbols[c] == "N"
    assert mol.n_atoms == 5                    # nothing added, nothing removed
    np.testing.assert_allclose(mol.positions[c], [0.0, 0.0, 0.0], atol=1e-9)


def test_replace_snaps_terminal_hydrogens():
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4 at C-H length
    editor.replace_atom(mol, c, "N", templates.TEMPLATES[("N", 3)])
    nh = elements.covalent_radius("N") + elements.covalent_radius("H")
    for h in range(1, 5):
        assert np.linalg.norm(mol.positions[h] - mol.positions[c]) == pytest.approx(nh, abs=1e-6)


def test_replace_keeps_heavy_neighbors_and_fills_valency():
    # C-C stub; replace atom 0 with O (default: bent, 2 bonds) -> one H added
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    editor.replace_atom(mol, 0, "O")
    assert mol.symbols[0] == "O"
    np.testing.assert_allclose(mol.positions[1], [1.52, 0.0, 0.0], atol=1e-9)
    assert mol.symbols.count("H") == 1
    oh = elements.covalent_radius("O") + elements.covalent_radius("H")
    assert np.linalg.norm(mol.positions[2] - mol.positions[0]) == pytest.approx(oh, abs=1e-6)


def test_replace_fills_multiple_hydrogens_without_overlap():
    # water O -> C (valence 4): the 2 existing H snap, 2 more are added
    mol = Molecule()
    o = editor.birth_molecule(mol, [0.0, 0.0, 0.0], element="O")   # H2O
    editor.replace_atom(mol, o, "C")
    assert mol.formula() == "CH4"
    ch = elements.covalent_radius("C") + elements.covalent_radius("H")
    hs = [i for i, s in enumerate(mol.symbols) if s == "H"]
    for h in hs:
        assert np.linalg.norm(mol.positions[h] - mol.positions[o]) == pytest.approx(ch, abs=1e-6)
    # the naive free_direction loop would stack the last two H on top of
    # each other; no H pair may come close to colliding
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            assert np.linalg.norm(mol.positions[hs[i]] - mol.positions[hs[j]]) > 1.0


def test_replace_same_element_repairs_valency():
    # sp2 CH3 clicked with default (sp3) carbon selected -> gains one H
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0],
                              template=templates.TEMPLATES[("C", 3)])   # CH3
    editor.replace_atom(mol, c, "C")
    assert mol.formula() == "CH4"


def test_replace_lone_atom_caps_fully():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.replace_atom(mol, 0, "O")
    assert mol.formula() == "H2O"


def test_replace_mid_chain_atom_snaps_h_and_keeps_skeleton():
    # ethane-like: replace one CH3 carbon with N -> heavy C stays, its 3 H snap
    mol = Molecule()
    c0 = editor.birth_molecule(mol, [0.0, 0.0, 0.0])         # CH4
    h = next(i for i, s in enumerate(mol.symbols) if s == "H")
    c1 = editor.grow_at_atom(mol, h)                          # promote H -> ethane
    heavy_pos = mol.positions[c0].copy()
    editor.replace_atom(mol, c1, "N", templates.TEMPLATES[("N", 3)])
    assert mol.symbols[c1] == "N"
    np.testing.assert_allclose(mol.positions[c0], heavy_pos, atol=1e-9)  # skeleton pinned
    nh = elements.covalent_radius("N") + elements.covalent_radius("H")
    n_pos = mol.positions[c1]
    snapped = [i for i in range(mol.n_atoms)
               if mol.symbols[i] == "H"
               and np.linalg.norm(mol.positions[i] - n_pos) < 1.2]
    assert len(snapped) == 3                                  # N keeps its 3 H...
    for i in snapped:
        assert np.linalg.norm(mol.positions[i] - n_pos) == pytest.approx(nh, abs=1e-6)


# -- editor: delete --------------------------------------------------------
def test_delete_heavy_atom_sweeps_its_hydrogens():
    # deleting methane's carbon takes its four terminal H with it
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    editor.delete_atom(mol, c)
    assert mol.n_atoms == 0                                   # whole group gone
    assert mol.bonds == []


def test_delete_hydrogen_removes_only_that_atom():
    # clicking a hydrogen has no hydrogens-of-its-own, so exactly it goes
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])              # CH4
    h = next(i for i, s in enumerate(mol.symbols) if s == "H")
    editor.delete_atom(mol, h)
    assert mol.formula() == "CH3"                             # one H fewer
    assert mol.n_atoms == 4
    assert mol.symbols.count("H") == 3


def test_delete_mid_chain_atom_leaves_neighbor_dangling_no_cap():
    # ethane: delete one carbon -> its heavy neighbor loses a bond, no new H
    mol = Molecule()
    c0 = editor.birth_molecule(mol, [0.0, 0.0, 0.0])         # CH4
    h = next(i for i, s in enumerate(mol.symbols) if s == "H")
    editor.grow_at_atom(mol, h)                              # promote H -> ethane C2H6
    assert mol.formula() == "C2H6"
    c1 = next(i for i, s in enumerate(mol.symbols) if s == "C" and i != c0)
    editor.delete_atom(mol, c1)                              # remove c1 + its 3 terminal H
    # a bare methyl remains -- CH4 (not CH3) would mean a hydrogen was auto-capped
    assert mol.formula() == "CH3"
    assert mol.n_atoms == 4
    # the surviving carbon is bonded only to its 3 hydrogens (the C-C bond is gone)
    c = mol.symbols.index("C")
    neigh = editor._neighbors(mol, c)
    assert len(neigh) == 3
    assert all(mol.symbols[j] == "H" for j in neigh)


def test_delete_last_atom_leaves_empty_molecule():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    ensure_bonds(mol)
    editor.delete_atom(mol, 0)
    assert mol.n_atoms == 0
    assert mol.symbols == []
    assert mol.positions.shape == (0, 3)


# -- editor: manual bonds ----------------------------------------------------
def test_add_manual_bond_creates_long_bond_that_survives_reperception():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    assert editor.add_manual_bond(mol, 0, 1) is True
    assert (0, 1, 1) in mol.bonds
    assert (0, 1, 1) in mol.manual_bonds
    # any later re-perceive (what every editor op ends with) must not drop it
    editor._reperceive(mol)
    assert (0, 1, 1) in mol.bonds


def test_add_manual_bond_rejects_self_and_duplicate_pairs():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    assert editor.add_manual_bond(mol, 0, 0) is False
    assert editor.add_manual_bond(mol, 0, 1) is True
    assert editor.add_manual_bond(mol, 1, 0) is False   # order-independent duplicate
    assert len(mol.manual_bonds) == 1


def test_add_manual_bond_within_auto_range_does_not_duplicate_in_bonds():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    assert len(mol.bonds) == 1                          # already auto-perceived
    assert editor.add_manual_bond(mol, 0, 1) is True
    assert mol.bonds.count((0, 1, 1)) == 1               # not duplicated
    assert mol.manual_bonds == [(0, 1, 1)]


def test_manual_bond_survives_an_unrelated_subsequent_edit():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    editor.add_manual_bond(mol, 0, 1)
    editor.birth_molecule(mol, [20.0, 0.0, 0.0])   # unrelated edit elsewhere, re-perceives
    assert (0, 1, 1) in mol.bonds


def test_delete_atom_remaps_manual_bond_indices():
    # atom 0 is isolated (far away, no bonds); deleting it must shift the
    # manual pair (1, 2) down to (0, 1).
    mol = Molecule(symbols=["He", "C", "C"],
                   positions=np.array([[-50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    editor.add_manual_bond(mol, 1, 2)
    assert mol.manual_bonds == [(1, 2, 1)]
    editor.delete_atom(mol, 0)
    assert mol.n_atoms == 2
    assert mol.manual_bonds == [(0, 1, 1)]
    assert (0, 1, 1) in mol.bonds


def test_delete_atom_drops_manual_bond_touching_deleted_atom():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    editor.add_manual_bond(mol, 0, 1)
    editor.delete_atom(mol, 1)
    assert mol.n_atoms == 1
    assert mol.manual_bonds == []


# -- editor: new_atoms tracking ---------------------------------------------
def test_birth_molecule_marks_all_atoms_new():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    assert mol.new_atoms == set(range(mol.n_atoms))


def test_grow_promotes_hydrogen_marks_promoted_atom_and_caps():
    # ethane-ish start: one C with a single H sticking out along +x
    mol = Molecule(symbols=["C", "H"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]))
    ensure_bonds(mol)
    mol.new_atoms.clear()          # pretend the starting C-H is loaded, not editor-made
    editor.grow_at_atom(mol, 1)    # click the H -> promotes to C, caps with 3 H
    # the promoted atom (1) and its 3 new caps are marked; the parent C (0) is not
    assert mol.new_atoms == {1, 2, 3, 4}


def test_replace_atom_marks_added_hydrogens_not_relabeled_anchor():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    mol.new_atoms.clear()
    editor.replace_atom(mol, 0, "O")     # relabels atom 0 in place, adds 1 fill H
    assert 0 not in mol.new_atoms        # relabeled anchor keeps its position -> not marked
    assert mol.new_atoms == {2}          # the added fill hydrogen is marked


def test_birth_molecule_center_and_caps_all_marked_via_lone_atom_replace():
    # replace_atom on a lone atom takes the birth_molecule-style full-cap path;
    # every added cap must be marked (the anchor itself is never "added").
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    mol.new_atoms.clear()
    editor.replace_atom(mol, 0, "O")     # lone atom -> full water cap
    assert 0 not in mol.new_atoms
    assert mol.new_atoms == {1, 2}


def test_delete_atom_remaps_new_atoms_indices():
    # atom 0 is isolated; deleting it must shift the new-atom index (2) down to 1.
    mol = Molecule(symbols=["He", "C", "C"],
                   positions=np.array([[-50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    ensure_bonds(mol)
    mol.new_atoms = {2}
    editor.delete_atom(mol, 0)
    assert mol.n_atoms == 2
    assert mol.new_atoms == {1}


def test_delete_atom_drops_new_atoms_touching_deleted_atom():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    ensure_bonds(mol)
    mol.new_atoms = {1}
    editor.delete_atom(mol, 1)
    assert mol.new_atoms == set()


# -- editor: intended bonds are recorded as manual bonds (rev 2a) ------------
def test_birth_molecule_records_intended_bonds_as_manual():
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    pairs = {(i, j) for i, j, _o in mol.manual_bonds}
    assert pairs == {(c, h) for h in range(1, 5)}            # center <-> each cap


def test_promote_hydrogen_records_parent_and_cap_bonds():
    mol = Molecule(symbols=["C", "H"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]))
    ensure_bonds(mol)
    editor.grow_at_atom(mol, 1)          # promote the H -> ethane-like
    pairs = {(i, j) for i, j, _o in mol.manual_bonds}
    # parent <-> promoted, and promoted <-> each of its 3 caps
    assert pairs == {(0, 1), (1, 2), (1, 3), (1, 4)}


def test_replace_atom_records_anchor_fill_bonds_not_preexisting():
    # C-C stub; replace atom 0 with O -> one fill H added and recorded; the
    # pre-existing heavy C-C bond stays perception-owned (NOT manual).
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    editor.replace_atom(mol, 0, "O")
    pairs = {(i, j) for i, j, _o in mol.manual_bonds}
    assert pairs == {(0, 2)}             # anchor <-> fill hydrogen only


def test_cleanup_targets_flags_clash_at_exactly_ideal_distance():
    # A new atom that lands at *exactly* the ideal bonding distance from an
    # unrelated old atom is still an accident -- intent is recorded, not
    # inferred, so it must be flagged (the old slop heuristic missed this).
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    ensure_bonds(mol)
    ideal = elements.covalent_radius("C") * 2
    idx = mol.add_atom("C", ideal, 0.0, 0.0)
    mol.new_atoms.add(idx)
    editor._reperceive(mol)
    assert (0, idx, 1) in mol.bonds      # perceived, at exactly ideal length
    clash, stretched = editor.cleanup_targets(mol)
    assert clash == [(0, idx)]
    assert stretched == []


# -- editor: cleanup_targets --------------------------------------------------
def test_cleanup_targets_ignores_old_old_bonds_flags_new_atom_clash():
    # two old atoms, already bonded (loaded geometry) -- never flagged regardless
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    assert len(mol.bonds) == 1
    clash, stretched = editor.cleanup_targets(mol)
    assert clash == [] and stretched == []

    # a new atom lands near atom 0 at a non-ideal (proximity-accident) distance
    idx = mol.add_atom("C", -1.0, 1.0, 0.0)
    mol.new_atoms.add(idx)
    editor._reperceive(mol)
    pairs = [(i, j) for i, j, _o in mol.bonds]
    assert (0, idx) in pairs
    assert (idx, 1) not in pairs and (1, idx) not in pairs   # no accidental 2nd bond
    clash, stretched = editor.cleanup_targets(mol)
    assert clash == [(0, idx)]
    assert stretched == []


def test_cleanup_targets_does_not_flag_editors_own_ideal_length_bonds():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])         # CH4, every bond at ideal length
    clash, stretched = editor.cleanup_targets(mol)
    assert clash == [] and stretched == []


def test_cleanup_targets_flags_stretched_manual_bond_not_short_one():
    far = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]))
    editor.add_manual_bond(far, 0, 1)
    clash, stretched = editor.cleanup_targets(far)
    assert stretched == [(0, 1)]

    near = Molecule(symbols=["C", "C"],
                    positions=np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]))
    editor.add_manual_bond(near, 0, 1)
    clash, stretched = editor.cleanup_targets(near)
    assert stretched == []


# -- editor: cleanup ----------------------------------------------------------
def test_cleanup_pushes_clash_apart_and_barely_moves_old_atoms():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    idx = mol.add_atom("C", -1.0, 1.0, 0.0)
    mol.new_atoms.add(idx)
    editor._reperceive(mol)
    pairs = [(i, j) for i, j, _o in mol.bonds]
    assert (0, idx) in pairs
    old_pos_before = mol.positions[[0, 1]].copy()
    new_pos_before = mol.positions[idx].copy()

    assert editor.cleanup(mol) is True

    old_disp = np.linalg.norm(mol.positions[[0, 1]] - old_pos_before, axis=1).max()
    new_disp = np.linalg.norm(mol.positions[idx] - new_pos_before)
    assert new_disp > old_disp        # the new segment did almost all the moving
    pairs_after = [(i, j) for i, j, _o in mol.bonds]
    assert (0, idx) not in pairs_after                # false bond is gone
    assert mol.new_atoms == set()


def test_cleanup_pulls_stretched_manual_bond_toward_ideal_length():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]))
    editor.add_manual_bond(mol, 0, 1)
    assert editor.cleanup(mol) is True
    ideal = elements.covalent_radius("C") * 2
    length = np.linalg.norm(mol.positions[1] - mol.positions[0])
    assert length == pytest.approx(ideal, abs=0.01)
    assert (0, 1, 1) in mol.bonds                      # stays a bond
    assert mol.new_atoms == set()


def test_cleanup_relaxes_distorted_angles_toward_tetrahedral():
    # Bond-length springs alone would resolve the clash but leave the angle
    # collapsed; the 1-3 (Urey-Bradley) springs restore local geometry.
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4, all new
    ch = elements.covalent_radius("C") + elements.covalent_radius("H")
    # squeeze H4 to ~55 deg of H1 (kept at the ideal C-H length): close enough
    # that perception sees a false H1-H4 bond (chord 0.99 A < the 1.07 A cutoff)
    d1 = mol.positions[1] / np.linalg.norm(mol.positions[1])
    d4 = mol.positions[4] / np.linalg.norm(mol.positions[4])
    perp = d4 - np.dot(d4, d1) * d1
    perp /= np.linalg.norm(perp)
    theta = np.radians(55.0)
    mol.positions[4] = ch * (np.cos(theta) * d1 + np.sin(theta) * perp)
    editor._reperceive(mol)
    pairs = [(i, j) for i, j, _o in mol.bonds]
    assert (1, 4) in pairs                                    # the false H-H bond
    assert editor.cleanup_targets(mol)[0] == [(1, 4)]

    assert editor.cleanup(mol) is True
    v1 = mol.positions[1] - mol.positions[c]
    v4 = mol.positions[4] - mol.positions[c]
    assert abs(_angle(v1, v4) - 109.47) < 5.0                 # tetrahedral again
    pairs_after = [(i, j) for i, j, _o in mol.bonds]
    assert (1, 4) not in pairs_after                          # false bond gone


def test_cleanup_registered_coordination_still_hits_ideal_angle():
    # Regression: the repulsion fallback must not disturb the registered
    # path -- a distorted tetrahedral carbon still lands on ~109.47 deg.
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    ch = elements.covalent_radius("C") + elements.covalent_radius("H")
    d1 = mol.positions[1] / np.linalg.norm(mol.positions[1])
    d4 = mol.positions[4] / np.linalg.norm(mol.positions[4])
    perp = d4 - np.dot(d4, d1) * d1
    perp /= np.linalg.norm(perp)
    mol.positions[4] = ch * (np.cos(np.radians(55.0)) * d1 + np.sin(np.radians(55.0)) * perp)
    editor._reperceive(mol)
    assert editor.cleanup(mol) is True
    v1 = mol.positions[1] - mol.positions[c]
    v4 = mol.positions[4] - mol.positions[c]
    assert abs(_angle(v1, v4) - 109.47) < 2.0                 # registered path unchanged


def test_cleanup_spreads_unregistered_coordination_by_repulsion():
    # Link two methanes at their carbons -> a 5-coordinate carbon, which has
    # no (C, 5) template. Without a repulsion fallback its neighbors stay
    # clumped; with theta=180 the law-of-cosines target becomes La+Lb (pure
    # repulsion), spreading them toward a trigonal bipyramid.
    mol = Molecule()
    c0 = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    c1 = editor.birth_molecule(mol, [10.0, 0.0, 0.0])         # a second CH4, far off
    editor.add_manual_bond(mol, c0, c1)                       # C0 now 5-coordinate
    neigh0 = editor._neighbors(mol, c0)
    assert len(neigh0) == 5

    def min_neighbor_angle(center, neigh):
        c = mol.positions[center]
        vs = [mol.positions[n] - c for n in neigh]
        return min(_angle(vs[a], vs[b])
                   for a in range(len(vs)) for b in range(a + 1, len(vs)))

    editor.cleanup(mol)
    neigh0 = editor._neighbors(mol, c0)
    assert len(neigh0) == 5                                   # still 5-coordinate
    assert min_neighbor_angle(c0, neigh0) > 80.0             # spread, not clumped


def test_cleanup_unregistered_repulsion_does_not_overstretch_real_bonds():
    # The theta=180 (full antipodal) target used for unregistered coordination
    # is geometrically unreachable for any pair once there are >2 neighbors
    # (no arrangement of 5 points can have every pair 180 deg apart), so a
    # naive symmetric spring toward it never reaches zero force and keeps
    # dragging the real C-H bonds outward. It must be repulsive-only (and
    # gentle) so it stops pulling once the neighbors are already reasonably
    # spread, instead of perpetually over-stretching real bonds.
    mol = Molecule()
    c0 = editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    c1 = editor.birth_molecule(mol, [8.0, 0.0, 0.0])
    editor.add_manual_bond(mol, c0, c1)
    editor.cleanup(mol)   # both carbons now 5-coordinate (unregistered)

    ch = elements.covalent_radius("C") + elements.covalent_radius("H")
    for i, j, _o in mol.bonds:
        if {mol.symbols[i], mol.symbols[j]} == {"C", "H"}:
            d = np.linalg.norm(mol.positions[i] - mol.positions[j])
            assert d < ch + 0.12                         # was drifting to ch+0.18..0.19


def test_cleanup_returns_false_and_moves_nothing_when_nothing_to_clean():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])         # only ideal-length bonds
    before = mol.positions.copy()
    assert editor.cleanup(mol) is False
    np.testing.assert_allclose(mol.positions, before)


def test_delete_atom_promotes_survivors_existing_bonds_to_manual():
    # A pair of methanes NOT built via the editor (plain construction +
    # distance perception, like a loaded file) -- their C-H bonds are
    # ordinary perceived bonds, not recorded as manual/intended. Deleting
    # one H marks the survivor carbon as needing cleanup attention; its
    # OTHER, untouched bonds predate the deletion and must be promoted to
    # manual so a later cleanup can never mistake them for proximity
    # accidents just because their carbon is now "new".
    tmpl = templates.default_template("C")
    r = elements.covalent_radius("C") + elements.covalent_radius("H")
    syms = ["C"]; pos = [[0.0, 0.0, 0.0]]
    for d in tmpl.free_directions():
        syms.append("H"); pos.append(np.array([0.0, 0.0, 0.0]) + d * r)
    mol = Molecule(symbols=syms, positions=np.array(pos))
    ensure_bonds(mol)
    assert mol.manual_bonds == []                     # none recorded, as if loaded

    editor.delete_atom(mol, 1)                          # C + 3 H left
    remaining_h = [i for i, s in enumerate(mol.symbols) if s == "H"]
    assert len(remaining_h) == 3
    manual_pairs = {(i, j) for i, j, _o in mol.manual_bonds}
    for h in remaining_h:
        assert (0, h) in manual_pairs                   # promoted, not left exposed


def test_two_linked_methanes_survive_delete_then_cleanup_without_losing_bonds():
    # The reported bug: link two methanes, cleanup (5-coordinate carbons),
    # delete one H per carbon (back to 4-coordinate), cleanup again -- no
    # C-H bond should ever vanish.
    mol = Molecule()
    c0 = editor.birth_molecule(mol, [0.0, 0.0, 0.0])
    c1 = editor.birth_molecule(mol, [8.0, 0.0, 0.0])
    editor.add_manual_bond(mol, c0, c1)
    editor.cleanup(mol)

    def neigh(mol, k):
        return [j for i, j, _o in mol.bonds if i == k] + [i for i, j, _o in mol.bonds if j == k]

    h0 = next(x for x in neigh(mol, c0) if mol.symbols[x] == "H")
    h1 = next(x for x in neigh(mol, c1) if mol.symbols[x] == "H")
    editor.delete_atom(mol, h0)
    if h0 < h1:
        h1 -= 1
    editor.delete_atom(mol, h1)
    editor.cleanup(mol)

    assert len(mol.bonds) == 7                          # nothing lost
    assert mol.formula() == "C2H6"


def test_cleanup_available_after_delete_even_with_no_new_atoms():
    # a settled (not "new") tetrahedral methane, then delete one H -- no atom
    # was added, so the old clash/stretched-only gate would find nothing to
    # fix and 'c' would silently do nothing. Deletion should mark the
    # survivors as needing attention so cleanup is available "anytime".
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])     # perfect CH4
    mol.new_atoms.clear()                               # simulate an accepted/loaded molecule
    editor.delete_atom(mol, 1)                          # C + 3 H left, still at tetrahedral angles
    assert editor.cleanup_targets(mol) == ([], [])       # no clash, no stretched manual bond
    assert editor.cleanup_prepare(mol) is not None        # yet cleanup is available
    assert editor.cleanup(mol) is True

    remaining_h = [i for i, s in enumerate(mol.symbols) if s == "H"]
    assert len(remaining_h) == 3
    angles = []
    for x in range(3):
        for y in range(x + 1, 3):
            a = mol.positions[remaining_h[x]] - mol.positions[c]
            b = mol.positions[remaining_h[y]] - mol.positions[c]
            angles.append(_angle(a, b))
    # relaxed toward trigonal (120 deg), not left at the old tetrahedral 109.47
    assert min(angles) > 115.0


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


def _atom_px(widget, idx):
    """Screen-pixel location of atom *idx*, using the same math as pick()."""
    mol = widget.molecule
    cam = widget.scene.camera
    ss = widget.scene.supersample
    Wr, Hr = widget.scene.render_size
    v = cam.view_positions(mol.positions[idx:idx + 1])[0]
    ox_s = Wr * 0.5 + cam.pan[0]
    oy_s = Hr * 0.5 - cam.pan[1]
    sx = ox_s + v[0] * cam.zoom
    sy = oy_s - v[1] * cam.zoom
    return sx / ss, sy / ss


def _alt_drag(widget, x0, y0, x1, y1):
    """Simulate an option/alt-drag gesture: alt+down -> alt+drag -> alt+up."""
    widget.handle_event(MouseEvent("down", x0, y0, button=0, alt=True, pixel=True))
    widget.handle_event(MouseEvent("drag", x1, y1, button=0, alt=True, pixel=True))
    return widget.handle_event(MouseEvent("up", x1, y1, button=0, alt=True, pixel=True))


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
    _click(w, 180, 20)                      # second click: empty corner, new methane
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


# -- widget manual bonds (option/alt-drag) ----------------------------------
def _far_pair():
    return Molecule(symbols=["C", "C"],
                    positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))


def test_widget_alt_drag_creates_manual_bond_undoable():
    mol = _far_pair()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    bx, by = _atom_px(w, 1)
    assert not w.dirty
    changed = _alt_drag(w, ax, ay, bx, by)
    assert changed
    assert (0, 1, 1) in w.molecule.bonds
    assert w.dirty
    assert w.undo()
    assert (0, 1, 1) not in w.molecule.bonds
    assert w.molecule.manual_bonds == []
    assert not w.dirty


def test_widget_alt_drag_release_empty_space_cancels():
    mol = _far_pair()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    assert w.pick(5, 5) is None                 # precondition: corner is empty
    changed = _alt_drag(w, ax, ay, 5, 5)
    assert changed                               # still redraws (preview cleared)
    assert w.molecule.manual_bonds == []
    assert not w.dirty
    assert not w.undo()                          # no undo entry was pushed


def test_widget_alt_drag_release_on_anchor_cancels():
    mol = _far_pair()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    changed = _alt_drag(w, ax, ay, ax, ay)       # release back on the anchor
    assert changed
    assert w.molecule.manual_bonds == []
    assert not w.dirty
    assert not w.undo()


def test_widget_alt_drag_preview_field_appears_and_disappears():
    mol = _far_pair()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    bx, by = _atom_px(w, 1)
    assert len(mol.vector_fields) == 0
    w.handle_event(MouseEvent("down", ax, ay, button=0, alt=True, pixel=True))
    assert len(mol.vector_fields) == 1           # preview installed on gesture start
    w.handle_event(MouseEvent("drag", bx, by, button=0, alt=True, pixel=True))
    assert len(mol.vector_fields) == 1           # still just the one preview field
    np.testing.assert_allclose(mol.vector_fields[0].vectors[0],
                               w.unproject(bx, by) - mol.positions[0], atol=1e-6)
    w.handle_event(MouseEvent("up", bx, by, button=0, alt=True, pixel=True))
    assert len(mol.vector_fields) == 0           # preview removed after release


def test_widget_alt_drag_preview_removal_ignores_other_vector_fields():
    # a pre-existing user vector field (e.g. dipole arrows) sits in the list
    # ahead of the gesture's own preview; removal must use identity, not
    # dataclass equality (which would raise on numpy array comparison).
    mol = _far_pair()
    mol.add_vector_field(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    bx, by = _atom_px(w, 1)
    changed = _alt_drag(w, ax, ay, bx, by)
    assert changed
    assert len(mol.vector_fields) == 1               # only the user field remains
    assert mol.vector_fields[0].vectors[0, 0] == pytest.approx(1.0)


def test_widget_undo_mid_gesture_cancels_it_and_removes_preview():
    # undo() can shrink the molecule under a live gesture; a stale anchor
    # index must not survive into the next drag event (IndexError), and the
    # preview arrow must not leak into the restored state's vector_fields.
    mol = _far_pair()                                # 2 atoms
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    _click(w, 5, 5)                                  # empty corner -> +CH4, 7 atoms
    assert w.molecule.n_atoms == 7
    ax, ay = _atom_px(w, 5)
    w.handle_event(MouseEvent("down", ax, ay, button=0, alt=True, pixel=True))
    assert w._bond_anchor == 5
    assert w.undo()                                  # back to 2 atoms mid-gesture
    assert w.molecule.n_atoms == 2
    assert w._bond_anchor is None                    # gesture cancelled
    assert w.molecule.vector_fields == []            # preview did not leak
    # the drag that used to crash with IndexError is now a plain no-op drag
    w.handle_event(MouseEvent("drag", 50, 50, button=0, alt=True, pixel=True))
    w.handle_event(MouseEvent("up", 50, 50, button=0, alt=True, pixel=True))
    assert w.molecule.manual_bonds == []


def test_widget_second_alt_down_mid_gesture_does_not_leak_preview():
    # a missed 'up' (focus loss, dropped event) leaves a gesture live; the
    # next alt+down must tear the old preview down, not orphan it.
    mol = _far_pair()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    ax, ay = _atom_px(w, 0)
    bx, by = _atom_px(w, 1)
    w.handle_event(MouseEvent("down", ax, ay, button=0, alt=True, pixel=True))
    w.handle_event(MouseEvent("down", bx, by, button=0, alt=True, pixel=True))
    assert len(mol.vector_fields) == 1               # one preview, not two
    assert w._bond_anchor == 1                       # the newer gesture won
    assert w.pick(5, 5) is None
    w.handle_event(MouseEvent("up", 5, 5, button=0, alt=True, pixel=True))
    assert mol.vector_fields == []                   # nothing orphaned
    assert w._bond_anchor is None


def test_widget_alt_down_over_empty_space_behaves_as_normal_press():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    assert w.pick(5, 5) is None
    changed = w.handle_event(MouseEvent("down", 5, 5, button=0, alt=True, pixel=True))
    assert not changed
    assert w._bond_anchor is None
    assert len(mol.vector_fields) == 0
    assert w._drag_button == 0                   # fell through to a normal press


def test_widget_plain_click_in_append_mode_still_builds_without_alt():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    changed = _click(w, 100, 100)
    assert changed and w.dirty
    assert w.molecule.formula() == "CH4"         # alt-drag wiring did not interfere


# -- widget delete mode ----------------------------------------------------
def test_delete_and_append_modes_are_mutually_exclusive():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    assert w.append_mode and not w.delete_mode
    w.set_delete_mode(True)                    # turning delete on clears append
    assert w.delete_mode and not w.append_mode
    w.set_append_mode(True)                    # ...and back the other way
    assert w.append_mode and not w.delete_mode


def test_delete_mode_stays_off_when_not_editable():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu")   # editable defaults False
    w.set_delete_mode(True)
    assert not w.delete_mode


def test_widget_delete_mode_click_atom_removes_it_and_is_undoable():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_delete_mode(True)
    Wr, Hr = w.scene.render_size
    ss = w.scene.supersample
    cx, cy = (Wr * 0.5) / ss, (Hr * 0.5) / ss
    assert w.pick(cx, cy) == 0                  # the lone carbon sits dead center
    changed = _click(w, cx, cy)
    assert changed and w.dirty
    assert w.molecule.n_atoms == 0              # last atom gone (0-atom bookkeeping ok)
    assert w.undo() and w.molecule.n_atoms == 1 # delete is undoable


def test_widget_delete_mode_click_empty_space_is_noop():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])         # CH4, centered
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_delete_mode(True)
    assert w.pick(180, 20) is None              # empty corner (precondition)
    n0 = w.molecule.n_atoms
    changed = _click(w, 180, 20)
    assert not changed                          # nothing under the cursor -> no-op
    assert not w.dirty                          # dirty untouched
    assert w.molecule.n_atoms == n0
    assert not w.undo()                         # and no undo snapshot was pushed


def test_delete_mode_hover_tints_red_not_yellow():
    mol = Molecule(symbols=["C", "O"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.hovered = 0
    w.set_append_mode(True)
    w._apply_highlight()
    append_col = w.style.color_override[0].copy()
    w.set_delete_mode(True)
    w._apply_highlight()
    delete_col = w.style.color_override[0]
    assert delete_col[0] >= delete_col[1]                # red preview, not yellow
    assert not np.allclose(delete_col, append_col)


# -- viewer delete mode ----------------------------------------------------
def test_viewer_x_toggles_delete_mode_only_when_editable():
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    v._dispatch([KeyEvent("x")])
    assert v.widget.delete_mode
    v._dispatch([KeyEvent("x")])
    assert not v.widget.delete_mode
    # read-only viewer: 'x' is not claimed, delete mode never turns on
    v2 = Viewer(mol, backend="cpu")
    v2._dispatch([KeyEvent("x")])
    assert not v2.widget.delete_mode


def test_viewer_x_and_a_are_mutually_exclusive_modes():
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    v._dispatch([KeyEvent("x")])
    assert v.widget.delete_mode and not v.widget.append_mode
    v._dispatch([KeyEvent("a")])                 # switch to append via 'a'
    assert v.widget.append_mode and not v.widget.delete_mode


def test_status_bar_shows_delete_badge_when_active():
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    assert "✗DELETE" not in v._status_bar()
    v._dispatch([KeyEvent("x")])
    bar = v._status_bar()
    assert "✗DELETE" in bar
    assert "adding" not in bar                   # no build pills in delete mode


def test_delete_mode_writes_pointer_shape_escapes(monkeypatch):
    import vimol.kitty as kitty
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    captured = bytearray()
    monkeypatch.setattr(kitty, "write_bytes", lambda data, fd=1: captured.extend(data))
    v._dispatch([KeyEvent("x")])                 # delete on -> crosshair
    assert kitty.set_pointer_shape("crosshair") in bytes(captured)
    captured.clear()
    v._dispatch([KeyEvent("x")])                 # delete off -> reset
    assert kitty.reset_pointer_shape() in bytes(captured)


def test_switching_to_append_via_a_resets_pointer(monkeypatch):
    import vimol.kitty as kitty
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    v._dispatch([KeyEvent("x")])                 # delete on
    captured = bytearray()
    monkeypatch.setattr(kitty, "write_bytes", lambda data, fd=1: captured.extend(data))
    v._dispatch([KeyEvent("a")])                 # crosshair must not linger into append
    assert kitty.reset_pointer_shape() in bytes(captured)


def test_exit_resets_pointer_shape(monkeypatch):
    # _exit pops the pointer shape iff this viewer actually pushed one --
    # see _pointer_pushed / _push_pointer / _pop_pointer. Deliberately arm
    # delete mode first so there is a push to unwind; the balance is what
    # keeps a stray pop from clobbering a shape pushed by something outside
    # vimol (an outer tmux/terminal push), see the no-op counterpart below.
    import vimol.kitty as kitty
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    v._dispatch([KeyEvent("x")])                 # delete on -> crosshair pushed
    captured = bytearray()
    monkeypatch.setattr(kitty, "write_bytes", lambda data, fd=1: captured.extend(data))
    v._exit()                                    # quit mid-delete un-sets the crosshair
    assert kitty.reset_pointer_shape() in bytes(captured)


def test_exit_does_not_pop_pointer_when_nothing_was_pushed(monkeypatch):
    import vimol.kitty as kitty
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    captured = bytearray()
    monkeypatch.setattr(kitty, "write_bytes", lambda data, fd=1: captured.extend(data))
    v._exit()                                    # never armed delete/measure -> no pop
    assert kitty.reset_pointer_shape() not in bytes(captured)


def test_m_toggles_measure_mode_in_readonly_and_editable_viewers():
    from vimol.viewer import Viewer
    for editable in (False, True):
        mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
        v = Viewer(mol, backend="cpu", editable=editable)
        v.widget.set_pixel_size(200, 200)
        v._cols, v._rows = 100, 30
        v._dispatch([KeyEvent("m")])
        assert v.widget.measure_mode, f"editable={editable}"
        v._dispatch([KeyEvent("m")])
        assert not v.widget.measure_mode, f"editable={editable}"


def test_measure_mode_mutually_exclusive_with_append_and_delete():
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    v._dispatch([KeyEvent("a")])
    v._dispatch([KeyEvent("m")])
    assert v.widget.measure_mode and not v.widget.append_mode
    v._dispatch([KeyEvent("x")])
    assert v.widget.delete_mode and not v.widget.measure_mode
    v._dispatch([KeyEvent("m")])
    assert v.widget.measure_mode and not v.widget.delete_mode


def test_status_bar_shows_measure_badge_and_readout():
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=False)   # read-only: badge still shows
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    assert "∡MEASURE" not in v._status_bar()
    v._dispatch([KeyEvent("m")])
    assert "∡MEASURE" in v._status_bar()
    v.widget.measure_sel = [0, 1]                    # as if two atoms were clicked
    assert "d(#0–#1) = 1.500 Å" in v._status_bar()


def test_measure_pointer_pushes_cell_and_pops_balanced_across_transitions(monkeypatch):
    import vimol.kitty as kitty
    from vimol.viewer import Viewer
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    pushes, pops = [], []

    def capture(data, fd=1):
        push_cell = kitty.set_pointer_shape("cell")
        push_cross = kitty.set_pointer_shape("crosshair")
        pop = kitty.reset_pointer_shape()
        buf = bytes(data)
        for token, sink in ((push_cell, pushes), (push_cross, pushes), (pop, pops)):
            start = 0
            while (at := buf.find(token, start)) != -1:
                sink.append(token)
                start = at + len(token)
    monkeypatch.setattr(kitty, "write_bytes", capture)

    v._dispatch([KeyEvent("m")])                 # arm measure -> push cell
    assert pushes == [kitty.set_pointer_shape("cell")] and not pops
    v._dispatch([KeyEvent("x")])                 # measure -> delete: pop, then push crosshair
    assert pops == [kitty.reset_pointer_shape()]
    assert pushes[-1] == kitty.set_pointer_shape("crosshair")
    v._dispatch([KeyEvent("m")])                 # delete -> measure: pop, push cell
    v._dispatch([KeyEvent("a")])                 # measure -> append: pop, no push
    v._dispatch([KeyEvent("a")])                 # append off: nothing pushed -> no pop
    assert len(pushes) == len(pops), "every push must have exactly one matching pop"


def test_pointer_shape_helpers_build_osc22_push_pop_sequences():
    # Plain set/reset only restores the terminal's generic default shape,
    # not whatever was actually active before -- push/pop is OSC 22's
    # purpose-built mechanism for "temporarily override, then perfectly
    # restore", so arming uses a push and disarming a matching pop.
    from vimol import kitty
    assert kitty.set_pointer_shape("crosshair") == b"\x1b]22;>crosshair\x1b\\"
    assert kitty.reset_pointer_shape() == b"\x1b]22;<\x1b\\"


# -- measurement math (editor.measurement) ----------------------------------
def test_measurement_below_two_atoms_is_empty():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    assert editor.measurement(mol, []) == ""
    assert editor.measurement(mol, [0]) == ""


def test_measurement_distance():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.523, 0.0, 0.0]]))
    s = editor.measurement(mol, [0, 1])
    assert s.startswith("d(")
    assert "1.523" in s
    assert "Å" in s


def test_measurement_angle_90():
    mol = Molecule(symbols=["C", "C", "C"],
                   positions=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    s = editor.measurement(mol, [0, 1, 2])
    assert s.startswith("∠(")
    assert "90.0" in s
    assert "°" in s


def test_measurement_angle_10947():
    d = templates.default_template("C").directions
    mol = Molecule(symbols=["C", "C", "C"],
                   positions=np.array([d[0], [0.0, 0.0, 0.0], d[1]]))
    s = editor.measurement(mol, [0, 1, 2])
    assert "109.5" in s


def test_measurement_dihedral_60_magnitude():
    import math
    theta = math.radians(60.0)
    i = np.array([1.0, 0.0, 0.0])
    j = np.array([0.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    l = k + np.array([math.cos(theta), math.sin(theta), 0.0])
    mol = Molecule(symbols=["C", "C", "C", "C"], positions=np.array([i, j, k, l]))
    s = editor.measurement(mol, [0, 1, 2, 3])
    assert s.startswith("φ(")
    m = re.search(r"=\s*(-?[\d.]+)°", s)
    assert m is not None
    assert abs(abs(float(m.group(1))) - 60.0) < 0.15


def test_measurement_dihedral_planar_0_and_180():
    i = np.array([1.0, 0.0, 0.0])
    j = np.array([0.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    l_syn = k + np.array([1.0, 0.0, 0.0])
    l_anti = k + np.array([-1.0, 0.0, 0.0])
    mol_syn = Molecule(symbols=["C", "C", "C", "C"], positions=np.array([i, j, k, l_syn]))
    mol_anti = Molecule(symbols=["C", "C", "C", "C"], positions=np.array([i, j, k, l_anti]))
    s_syn = editor.measurement(mol_syn, [0, 1, 2, 3])
    s_anti = editor.measurement(mol_anti, [0, 1, 2, 3])
    m_syn = re.search(r"=\s*(-?[\d.]+)°", s_syn)
    m_anti = re.search(r"=\s*(-?[\d.]+)°", s_anti)
    assert abs(float(m_syn.group(1))) < 0.15
    assert abs(abs(float(m_anti.group(1))) - 180.0) < 0.15


# -- widget measure mode ------------------------------------------------------
def test_measure_mode_available_without_editable():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu")   # editable defaults False
    w.set_measure_mode(True)
    assert w.measure_mode                  # no editable gate, unlike append/delete


def test_measure_mode_mutually_exclusive_with_append_and_delete():
    mol = Molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    w.set_measure_mode(True)
    assert w.measure_mode and not w.append_mode
    w.set_delete_mode(True)
    assert w.delete_mode and not w.measure_mode
    w.set_measure_mode(True)
    assert w.measure_mode and not w.delete_mode
    w.set_append_mode(True)
    assert w.append_mode and not w.measure_mode


def test_measure_click_sequence_builds_ordered_selection():
    mol = Molecule(symbols=["C", "C", "C", "C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0],
                                        [-5.0, 0.0, 0.0], [0.0, -5.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    w.set_measure_mode(True)
    order = [0, 1, 2, 3]
    for idx in order:
        px, py = _atom_px(w, idx)
        changed = _click(w, px, py)
        assert changed
    assert w.measure_sel == order
    # re-clicking an already-selected atom is a no-op
    px, py = _atom_px(w, 1)
    changed = _click(w, px, py)
    assert not changed
    assert w.measure_sel == order
    # a 5th distinct atom resets to a fresh single-atom selection
    px, py = _atom_px(w, 4)
    changed = _click(w, px, py)
    assert changed
    assert w.measure_sel == [4]


def test_measure_click_empty_space_clears_selection():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    w.set_measure_mode(True)
    px, py = _atom_px(w, 0)
    _click(w, px, py)
    assert w.measure_sel == [0]
    changed = _click(w, 190, 190)          # empty corner
    assert changed
    assert w.measure_sel == []


def test_measure_drag_does_not_select():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    w.set_measure_mode(True)
    px, py = _atom_px(w, 0)
    w.handle_event(MouseEvent("down", px, py, button=0, pixel=True))
    w.handle_event(MouseEvent("drag", px + 40, py, button=0, pixel=True))
    w.handle_event(MouseEvent("up", px + 40, py, button=0, pixel=True))
    assert w.measure_sel == []


def test_measure_highlight_covers_all_selected_atoms():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    w.set_measure_mode(True)
    for idx in (0, 1):
        px, py = _atom_px(w, idx)
        _click(w, px, py)
    w._apply_highlight()
    cols = w.style.color_override
    assert cols is not None
    base = w._base_colors
    assert not np.allclose(cols[0], base[0])
    assert not np.allclose(cols[1], base[1])


def test_measure_selection_cleared_by_undo_and_set_molecule():
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), )
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    _click(w, 100, 100)                     # a real edit, so undo has something to revert
    w.set_measure_mode(True)
    px, py = _atom_px(w, 0)
    _click(w, px, py)
    assert w.measure_sel == [0]
    w.undo()
    assert w.measure_sel == []
    w.set_measure_mode(True)
    _click(w, px, py)
    assert w.measure_sel == [0]
    w.set_molecule(Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]])))
    assert w.measure_sel == []


def test_measure_mode_disarm_clears_selection():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu")
    w.set_measure_mode(True)
    px, py = _atom_px(w, 0)
    _click(w, px, py)
    assert w.measure_sel == [0]
    w.set_measure_mode(False)
    assert w.measure_sel == []


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


# -- widget/viewer cleanup ('c') --------------------------------------------
def _clashy_molecule():
    """An old-old bonded pair plus one editor-new atom clashing with atom 0."""
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    idx = mol.add_atom("C", -1.0, 1.0, 0.0)
    mol.new_atoms.add(idx)
    editor._reperceive(mol)
    return mol


def test_widget_cleanup_relaxes_and_is_undoable():
    mol = _clashy_molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    pos_before = mol.positions.copy()
    new_atoms_before = set(mol.new_atoms)
    assert not w.dirty
    assert w.cleanup() is True
    assert w.dirty
    assert mol.new_atoms == set()
    assert w.undo()
    np.testing.assert_allclose(mol.positions, pos_before)
    assert mol.new_atoms == new_atoms_before
    assert not w.dirty


def test_widget_cleanup_returns_false_when_nothing_to_clean():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])       # ideal-length bonds only
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    assert w.cleanup() is False
    assert not w.dirty
    assert not w.undo()                               # no snapshot was pushed


def test_viewer_c_key_starts_cleanup_only_when_editable():
    from vimol.viewer import Viewer
    mol = _clashy_molecule()
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    assert v._dispatch([KeyEvent("c")])
    assert v.widget.cleanup_active               # 'c' starts the animation...
    while v.widget.cleanup_active:               # ...the run loop ticks it out
        v.widget.cleanup_tick()
    assert v.widget.dirty
    assert v.widget.molecule.new_atoms == set()

    v2 = Viewer(_clashy_molecule(), backend="cpu")    # editable defaults False
    v2.widget.set_pixel_size(200, 200)
    v2._cols, v2._rows = 100, 30
    assert not v2._dispatch([KeyEvent("c")])
    assert not v2.widget.cleanup_active
    assert not v2.widget.dirty


def test_status_bar_shows_cleanup_hint_when_clash_exists_and_hides_after_c():
    from vimol.viewer import Viewer
    mol = _clashy_molecule()
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    assert "cleanup" in v._status_bar()
    v._dispatch([KeyEvent("c")])
    while v.widget.cleanup_active:
        v.widget.cleanup_tick()
    assert "cleanup" not in v._status_bar()


# -- widget cleanup animation (rev 2c) ---------------------------------------
def test_widget_start_cleanup_animates_ticks_then_finishes():
    mol = _clashy_molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    pos0 = mol.positions.copy()
    assert w.start_cleanup() is True
    assert w.cleanup_active
    frames = []
    for _ in range(100):                          # safety bound; budget is ~30
        if not w.cleanup_tick():
            break
        frames.append(mol.positions.copy())
    assert not w.cleanup_active
    # at least two distinct intermediate states -- it animates, not teleports
    distinct = [f for i, f in enumerate(frames)
                if i == 0 or not np.allclose(f, frames[i - 1])]
    assert len(distinct) >= 2
    pairs = [(i, j) for i, j, _o in mol.bonds]
    assert (0, 2) not in pairs                    # clash resolved at finish
    assert mol.new_atoms == set()
    assert w.dirty
    assert w.undo()                               # a single 'u' undoes it all
    np.testing.assert_allclose(mol.positions, pos0)
    assert not w.dirty


def test_widget_start_cleanup_nothing_to_fix_returns_false():
    mol = Molecule()
    editor.birth_molecule(mol, [0.0, 0.0, 0.0])   # nothing to clean
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    assert w.start_cleanup() is False
    assert not w.cleanup_active
    assert not w.cleanup_tick()                   # inert when nothing started
    assert not w.undo()                           # no undo entry was pushed


def test_widget_undo_mid_animation_cancels_cleanup():
    mol = _clashy_molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    pos0 = mol.positions.copy()
    assert w.start_cleanup()
    assert w.cleanup_tick()                       # partway through
    assert w.undo()                               # cancel + restore, no finish
    assert not w.cleanup_active
    np.testing.assert_allclose(mol.positions, pos0)
    before = mol.positions.copy()
    assert not w.cleanup_tick()                   # no further tick mutates
    np.testing.assert_allclose(mol.positions, before)


def test_widget_c_during_animation_is_ignored():
    mol = _clashy_molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    pos0 = mol.positions.copy()
    assert w.start_cleanup() is True
    w.cleanup_tick()
    assert w.start_cleanup() is False             # mid-animation press ignored
    while w.cleanup_active:
        w.cleanup_tick()
    assert w.undo()                               # exactly one undo entry...
    np.testing.assert_allclose(mol.positions, pos0)
    assert not w.undo()                           # ...not two


def test_widget_set_molecule_cancels_animation():
    mol = _clashy_molecule()
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    assert w.start_cleanup()
    w.set_molecule(Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]])))
    assert not w.cleanup_active
    assert not w.cleanup_tick()                   # inert against the new molecule


def test_status_bar_shows_cleanup_hint_after_over_long_manual_bond():
    from vimol.viewer import Viewer
    mol = _far_pair()
    v = Viewer(mol, backend="cpu", editable=True)
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    assert "cleanup" not in v._status_bar()
    ax, ay = _atom_px(v.widget, 0)
    bx, by = _atom_px(v.widget, 1)
    _alt_drag(v.widget, ax, ay, bx, by)
    assert "cleanup" in v._status_bar()


def test_status_bar_no_cleanup_hint_when_not_editable():
    from vimol.viewer import Viewer
    mol = _clashy_molecule()
    v = Viewer(mol, backend="cpu")                    # editable defaults False
    v.widget.set_pixel_size(200, 200)
    v._cols, v._rows = 100, 30
    assert "cleanup" not in v._status_bar()


# -- CLI: no-arg default opens the bundled demo -----------------------------
def test_default_demo_path_resolves_to_bundled_c60():
    path = vimol_app._default_demo_path()
    assert path is not None and os.path.exists(path)
    assert os.path.basename(path) == "c60.xyz"


def test_cli_with_no_file_uses_demo_default_not_help(capsys):
    # non-tty stdout (pytest) means it can't actually open the interactive
    # viewer -- it should get as far as trying to (exit 4), not fall back to
    # --help (exit 1), proving args.file was filled in with the demo path.
    rc = vimol_app.main([])
    assert rc == 4
    assert "usage:" not in capsys.readouterr().out
