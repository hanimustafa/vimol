import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vimol
from vimol.molecule import Molecule
from vimol.structures import Transform, AlignmentResult, Structure


def test_transform_identity_apply_is_noop():
    t = Transform()
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = t.apply(pos)
    assert np.allclose(out, pos)


def test_transform_identity_is_identity_true():
    assert Transform().is_identity is True


def test_transform_rotation_translation_apply():
    # 90 degree rotation about z: (x,y,z) -> (-y,x,z)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = Transform(rotation=rot, translation=np.array([10.0, 0.0, 0.0]))
    pos = np.array([[1.0, 0.0, 0.0]])
    out = t.apply(pos)
    assert np.allclose(out, [[10.0, 1.0, 0.0]])
    assert t.is_identity is False


def test_transform_apply_directions_ignores_translation():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = Transform(rotation=rot, translation=np.array([100.0, 100.0, 100.0]))
    vec = np.array([[1.0, 0.0, 0.0]])
    out = t.apply_directions(vec)
    assert np.allclose(out, [[0.0, 1.0, 0.0]])


def test_transform_inverse_round_trips():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = Transform(rotation=rot, translation=np.array([5.0, -2.0, 1.0]))
    pos = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    out = t.inverse().apply(t.apply(pos))
    assert np.allclose(out, pos)


def test_transform_compose_applies_other_first():
    t1 = Transform(translation=np.array([1.0, 0.0, 0.0]))
    t2 = Transform(translation=np.array([0.0, 1.0, 0.0]))
    composed = t1.compose(t2)
    pos = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(composed.apply(pos), t1.apply(t2.apply(pos)))


def test_transform_key_differs_for_different_transforms():
    a = Transform()
    b = Transform(translation=np.array([1.0, 0.0, 0.0]))
    assert a.key() != b.key()
    assert a.key() == Transform().key()


def _mol(n=2):
    m = Molecule()
    for i in range(n):
        m.add_atom("C", float(i), 0.0, 0.0)
    return m


def test_alignment_result_holds_fit_report():
    t = Transform()
    res = AlignmentResult(rmsd=0.42, n_fitted=10, transform=t, ref_label="a.xyz", method="index")
    assert res.rmsd == 0.42
    assert res.select is None
    assert res.stale is False


def test_structure_defaults():
    mol = _mol()
    s = Structure(molecule=mol, label="a.xyz")
    assert s.molecule is mol
    assert s.visible is True
    assert s.marked is False
    assert s.transform.is_identity
    assert s.revision == 0
    assert s.alignment is None
    assert s.undo_stack == []
    assert s.saved_sig is None


def test_structure_touch_bumps_revision():
    s = Structure(molecule=_mol(), label="a.xyz")
    s.touch()
    s.touch()
    assert s.revision == 2


from vimol.structures import StructureSet, TINTS


def test_tints_palette_has_eight_distinct_colors():
    assert len(TINTS) == 8
    assert len(set(TINTS)) == 8


def test_structure_set_append_assigns_tint_from_palette_by_index():
    sset = StructureSet()
    s0 = sset.append(_mol(), label="a")
    s1 = sset.append(_mol(), label="b")
    assert s0.tint == TINTS[0]
    assert s1.tint == TINTS[1]


def test_structure_set_len_iter_getitem():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    assert len(sset) == 2
    labels = [s.label for s in sset]
    assert labels == ["a", "b"]
    assert sset[0].label == "a"
    assert sset["b"].label == "b"
    with pytest.raises(KeyError):
        sset["nope"]


def test_structure_set_molecules_and_labels():
    sset = StructureSet()
    m0 = _mol()
    m1 = _mol()
    sset.append(m0, label="a")
    sset.append(m1, label="b")
    assert sset.molecules == [m0, m1]
    assert sset.labels == ["a", "b"]


def test_structure_set_active_and_set_active_by_index_and_label():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    assert sset.active_index == 0
    assert sset.active.label == "a"
    sset.set_active(1)
    assert sset.active_index == 1
    sset.set_active("a")
    assert sset.active_index == 0


def test_structure_set_cycle_active_wraps():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.append(_mol(), label="c")
    sset.cycle_active(1)
    assert sset.active_index == 1
    sset.cycle_active(1)
    sset.cycle_active(1)
    assert sset.active_index == 0
    sset.cycle_active(-1)
    assert sset.active_index == 2


def test_structure_set_remove_by_label():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.remove("a")
    assert sset.labels == ["b"]


def test_structure_set_marks():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.append(_mol(), label="c")
    sset.toggle_mark(1)
    assert sset.marked == [sset[1]]
    sset.toggle_mark(1)
    assert sset.marked == []
    sset.toggle_mark(0)
    sset.toggle_mark(2)
    sset.clear_marks()
    assert sset.marked == []


def test_structure_set_drawn_indices_no_overlay_is_active_only():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.set_active(1)
    assert sset.drawn_indices() == [1]


def test_structure_set_drawn_indices_active_hidden_yields_empty():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.toggle_visible(0)
    assert sset.drawn_indices() == []


def test_structure_set_drawn_indices_overlay_no_marks_is_all_visible():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.append(_mol(), label="c")
    sset.set_active(1)
    sset.overlay = True
    assert sset.drawn_indices() == [1, 0, 2]


def test_structure_set_drawn_indices_overlay_with_marks_is_active_plus_marked():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.append(_mol(), label="c")
    sset.set_active(0)
    sset.overlay = True
    sset.toggle_mark(2)
    assert sset.drawn_indices() == [0, 2]


def test_structure_set_solo_toggle_restores_prior_visibility():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    sset.append(_mol(), label="b")
    sset.append(_mol(), label="c")
    sset.toggle_visible(1)   # b starts hidden
    sset.solo(0)
    assert [s.visible for s in sset] == [True, False, False]
    sset.solo(0)             # second press restores
    assert [s.visible for s in sset] == [True, False, True]
