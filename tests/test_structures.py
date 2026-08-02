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


def test_index_after_removal_rebases_below_above_and_inside_the_range():
    shift = StructureSet.index_after_removal
    # remove [2, 5) from 8 entries -> 5 remain
    assert shift(1, 2, 5, 5) == 1        # below: untouched
    assert shift(5, 2, 5, 5) == 2        # above: slides down by the width
    assert shift(7, 2, 5, 5) == 4
    assert shift(3, 2, 5, 5) == 2        # inside: collapses onto `first`
    # Removing the tail leaves nothing at `first`, so it clamps to the last
    # survivor rather than pointing one past the end.
    assert shift(6, 4, 8, 4) == 3


def test_remove_range_rebases_active_index_onto_the_entry_it_named():
    """The active index names an ENTRY, not a slot: removing a file above it
    must leave it on the same structure, not on whatever slid into its old
    position."""
    sset = StructureSet()
    for label in "abcde":
        sset.append(_mol(), label=label)
    sset.active_index = 4                      # "e"
    active = sset.active

    sset.remove_range(1, 3)                    # drop "b", "c"
    assert sset.labels == ["a", "d", "e"]
    assert sset.active is active               # still "e" ...
    assert sset.active_index == 2              # ... at its new position


def test_remove_range_taking_the_active_entry_falls_onto_its_successor():
    sset = StructureSet()
    for label in "abcd":
        sset.append(_mol(), label=label)
    sset.active_index = 1                      # "b", inside the removal
    sset.remove_range(1, 3)                    # drop "b", "c"
    assert sset.labels == ["a", "d"]
    assert sset.active.label == "d"            # what slid into slot 1


def test_remove_range_keeps_the_solo_snapshot_aligned_with_what_survives():
    """solo() zips _solo_restore against entries, so a snapshot left longer
    than the rows it describes restores visibility onto the wrong ones."""
    sset = StructureSet()
    for label in "abcd":
        sset.append(_mol(), label=label)
    sset["b"].visible = False                  # the state solo must restore
    sset.solo(3)                               # snapshot taken here
    assert [e.visible for e in sset] == [False, False, False, True]

    # Removed from the FRONT on purpose: zip() would silently truncate a
    # stale tail, so only a leading removal exposes a snapshot that was
    # never re-based.
    sset.remove_range(0, 2)                    # drop "a", "b" while soloed
    sset.unsolo()
    assert [e.visible for e in sset] == [True, True]   # "c" and "d" both were


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


def _mol_with_bond():
    m = Molecule()
    m.add_atom("C", 0.0, 0.0, 0.0)
    m.add_atom("H", 1.0, 0.0, 0.0)
    m.add_bond(0, 1, 1)
    return m


def test_composite_single_visible_identity_is_zero_copy():
    sset = StructureSet()
    mol = _mol_with_bond()
    sset.append(mol, label="a")
    comp = sset.composite()
    assert comp.molecule is mol
    assert np.allclose(comp.base_colors, mol.element_colors())
    assert not comp.flat.any()
    assert list(comp.offsets) == [0, 2]
    assert list(comp.sources) == [0]


def test_composite_single_visible_non_identity_transform_moves_positions():
    from vimol.structures import Transform
    sset = StructureSet()
    mol = _mol_with_bond()
    entry = sset.append(mol, label="a")
    entry.transform = Transform(translation=np.array([5.0, 0.0, 0.0]))
    comp = sset.composite()
    assert comp.molecule is not mol
    assert np.allclose(comp.molecule.positions, mol.positions + [5.0, 0.0, 0.0])
    # still alone -> CPK colors, not flat
    assert np.allclose(comp.base_colors, mol.element_colors())
    assert not comp.flat.any()


def test_composite_overlay_first_entry_cpk_rest_flat_tinted():
    sset = StructureSet()
    a = sset.append(_mol_with_bond(), label="a")
    b = sset.append(_mol_with_bond(), label="b")
    sset.overlay = True
    comp = sset.composite()
    assert comp.sources.tolist() == [0, 1]
    assert list(comp.offsets) == [0, 2, 4]
    assert np.allclose(comp.base_colors[0:2], a.molecule.element_colors())
    assert not comp.flat[0:2].any()
    assert np.allclose(comp.base_colors[2:4], b.tint)
    assert comp.flat[2:4].all()


def test_composite_bonds_offset_by_entry():
    sset = StructureSet()
    sset.append(_mol_with_bond(), label="a")
    sset.append(_mol_with_bond(), label="b")
    sset.overlay = True
    comp = sset.composite()
    assert comp.molecule.bonds == [(0, 1, 1), (2, 3, 1)]


def test_composite_locate_and_globalize_round_trip():
    sset = StructureSet()
    sset.append(_mol_with_bond(), label="a")
    sset.append(_mol_with_bond(), label="b")
    sset.overlay = True
    comp = sset.composite()
    assert comp.locate(0) == (0, 0)
    assert comp.locate(1) == (0, 1)
    assert comp.locate(2) == (1, 0)
    assert comp.locate(3) == (1, 1)
    g = comp.globalize(1, np.array([0, 1]))
    assert list(g) == [2, 3]


def test_composite_cache_hit_returns_same_object_when_nothing_changed():
    sset = StructureSet()
    sset.append(_mol_with_bond(), label="a")
    sset.append(_mol_with_bond(), label="b")
    sset.overlay = True
    c1 = sset.composite()
    c2 = sset.composite()
    assert c1 is c2


def test_composite_cache_invalidated_by_touch():
    sset = StructureSet()
    a = sset.append(_mol_with_bond(), label="a")
    sset.append(_mol_with_bond(), label="b")
    sset.overlay = True
    c1 = sset.composite()
    a.molecule.positions = a.molecule.positions + 1.0
    a.touch()
    c2 = sset.composite()
    assert c1 is not c2
    assert np.allclose(c2.molecule.positions[0:2], a.molecule.positions)


def test_composite_cache_invalidated_by_mark_change():
    sset = StructureSet()
    sset.append(_mol_with_bond(), label="a")
    sset.append(_mol_with_bond(), label="b")
    sset.append(_mol_with_bond(), label="c")
    sset.set_active(0)
    sset.overlay = True
    c1 = sset.composite()
    sset.toggle_mark(2)
    c2 = sset.composite()
    assert c1 is not c2
    assert comp_sources_matches(c2, [0, 2])


def comp_sources_matches(comp, expected):
    return comp.sources.tolist() == expected


def test_composite_explicit_invalidate_forces_rebuild():
    sset = StructureSet()
    sset.append(_mol_with_bond(), label="a")
    c1 = sset.composite()
    sset.invalidate()
    c2 = sset.composite()
    assert c1 is not c2
    assert c1.molecule is c2.molecule  # same underlying data, just re-served


def test_composite_concatenates_and_rotates_vector_fields():
    from vimol.structures import Transform
    sset = StructureSet()
    a = sset.append(_mol_with_bond(), label="a")
    b = sset.append(_mol_with_bond(), label="b")
    a.molecule.add_vector_field(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    b.transform = Transform(rotation=rot)
    sset.overlay = True
    comp = sset.composite()
    assert len(comp.molecule.vector_fields) == 1
    vf = comp.molecule.vector_fields[0]
    assert vf.vectors.shape == (4, 3)
    assert np.allclose(vf.vectors[0:2], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(vf.vectors[2:4], 0.0)


# -- cross-structure measurement (VIM-6) ------------------------------------

def _mol_dist(d):
    return Molecule(symbols=["C", "C"], positions=np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]))


def test_measure_matching_topology_returns_computed_distance_per_entry():
    sset = StructureSet()
    sset.append(_mol_dist(1.0), label="a")
    sset.append(_mol_dist(2.5), label="b")
    result = sset.measure([0, 1])
    assert result == [("a", pytest.approx(1.0)), ("b", pytest.approx(2.5))]


def test_measure_guards_index_past_current_atom_count():
    """A pinned index that has outlived an entry's atom count (e.g. the UI
    committed it, then editing shrank the active structure) must degrade to
    None, not raise -- the symbols check alone can't catch this since an
    entry always matches itself trivially by identity."""
    sset = StructureSet()
    sset.append(_mol_dist(1.0), label="a")   # 2 atoms: index 5 is out of range
    assert sset.measure([0, 5]) == [("a", None)]


def test_measure_mismatched_symbols_yields_none():
    sset = StructureSet()
    sset.append(_mol_dist(1.0), label="a")
    sset.append(Molecule(symbols=["C", "N"], positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])),
                label="b")
    result = sset.measure([0, 1])
    assert result[0] == ("a", pytest.approx(1.0))
    assert result[1] == ("b", None)


def test_measure_active_entry_is_always_included():
    sset = StructureSet()
    sset.append(_mol_dist(1.0), label="a")
    sset.append(_mol_dist(2.5), label="b")
    sset.set_active(1)
    result = sset.measure([0, 1])
    assert result[1] == ("b", pytest.approx(2.5))


def test_measure_empty_set_returns_empty_list():
    sset = StructureSet()
    assert sset.measure([0, 1]) == []
