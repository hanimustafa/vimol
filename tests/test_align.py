import os
import time

import numpy as np
import pytest

import vimol
from vimol.align import (
    _linear_sum_assignment_numpy,
    kabsch,
    subset_search,
    superpose,
    superpose_to_reference_subset,
)
from vimol.input import KeyEvent, MouseEvent
from vimol.molecule import Molecule
from vimol.structures import StructureSet
from vimol.viewer import Viewer


def _mol(symbols, positions, name=""):
    return Molecule(symbols=list(symbols), positions=np.asarray(positions, dtype=float),
                    name=name)


def _rigid(points):
    rotation = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])
    translation = np.array([4.5, -2.25, 7.0])
    return points @ rotation.T + translation


def test_kabsch_recovers_exact_proper_transform():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    q = _rigid(p)
    rotation, translation, rmsd = kabsch(p, q)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rmsd < 1e-12
    assert np.allclose(p @ rotation.T + translation, q)


def test_kabsch_does_not_fit_a_reflection_as_a_rotation():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    q = p.copy()
    q[:, 0] *= -1
    rotation, _translation, rmsd = kabsch(p, q)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rmsd > 0.1


def test_numpy_rectangular_assignment_matches_scipy_cost():
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(42)
    for shape in ((1, 5), (4, 7), (12, 12), (15, 30)):
        cost = rng.random(shape)
        rows, cols = _linear_sum_assignment_numpy(cost)
        erows, ecols = scipy.linear_sum_assignment(cost)
        assert cost[rows, cols].sum() == pytest.approx(cost[erows, ecols].sum())


def test_permutation_search_recovers_shuffled_atoms():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    symbols = ["C", "N", "O", "H"]
    q = _rigid(p)
    shuffle = np.array([2, 0, 3, 1])
    mobile = _mol(symbols, p)
    reference = _mol([symbols[i] for i in shuffle], q[shuffle])
    result = superpose(mobile, reference, permute=True, trials=256, candidates=24)
    assert result.method == "permute"
    assert result.rmsd < 1e-10
    assert np.allclose(result.transform.apply(p), q)


def test_permutation_search_recovers_shuffle_with_repeated_elements():
    rng = np.random.default_rng(91)
    p = rng.normal(size=(12, 3))
    symbols = ["C"] * 6 + ["H"] * 6
    q = _rigid(p)
    shuffle = rng.permutation(len(p))
    result = superpose(
        _mol(symbols, p),
        _mol([symbols[i] for i in shuffle], q[shuffle]),
        permute=True, trials=512, candidates=32)
    assert result.rmsd < 1e-10
    assert np.allclose(result.transform.apply(p), q)


def test_subset_search_finds_query_inside_larger_shuffled_target():
    query = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    query_symbols = ["C", "N", "O", "H"]
    embedded = _rigid(query)
    target = np.vstack(([[20, 20, 20], [-12, 8, 2]], embedded))
    target_symbols = ["C", "H"] + query_symbols
    shuffle = np.array([4, 0, 2, 5, 1, 3])
    target = target[shuffle]
    target_symbols = [target_symbols[i] for i in shuffle]

    rotation, translation, mapping, rmsd = subset_search(
        query, target, query_symbols, target_symbols, candidates=24)
    assert rmsd < 1e-10
    assert len(set(mapping)) == len(query)
    assert [target_symbols[i] for i in mapping] == query_symbols
    assert np.allclose(query @ rotation.T + translation, target[mapping])


def test_reference_subset_errors_name_the_structure_that_is_actually_short():
    """superpose_to_reference_subset searches for the reference selection
    INSIDE the mobile, so subset_search sees the two structures swapped. Its
    errors reach the user verbatim next to a structure label, so naming them
    the wrong way round sends the user to inspect the wrong file."""
    reference = _mol(["C", "C", "O"], [[0, 0, 0], [1.5, 0, 0], [0, 1.4, 0]])
    mobile = _mol(["C"] * 5, [[0, 0, 0], [1.5, 0, 0], [3, 0, 0],
                              [4.5, 0, 0], [6, 0, 0]])
    with pytest.raises(ValueError) as excinfo:
        superpose_to_reference_subset(mobile, reference, [0, 1, 2])
    # The MOBILE is the one with no oxygen.
    assert "mobile has 0 O atom(s)" in str(excinfo.value)
    assert "reference selection requires 1" in str(excinfo.value)

    wide_reference = _mol(["C"] * 4, [[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
    narrow_mobile = _mol(["C", "C"], [[0, 0, 0], [1.5, 0, 0]])
    with pytest.raises(ValueError) as excinfo:
        superpose_to_reference_subset(narrow_mobile, wide_reference, [0, 1, 2, 3])
    # The SELECTION is the one that is too large.
    assert "reference selection to be no larger than mobile" in str(excinfo.value)


def test_direct_subset_errors_keep_the_plain_mobile_onto_reference_wording():
    """The guard for the fix above: every other caller passes the structures
    the natural way round, so renaming the two sides inside the shared helper
    -- rather than at the one inverted call site -- would break them here."""
    reference = _mol(["C", "C", "O"], [[0, 0, 0], [1.5, 0, 0], [0, 1.4, 0]])
    mobile = _mol(["C", "C", "N"], [[0, 0, 0], [1.5, 0, 0], [0, 1.4, 0]])
    with pytest.raises(ValueError) as excinfo:
        superpose(mobile, reference, subset=True)
    assert "reference has 0 N atom(s)" in str(excinfo.value)
    assert "mobile requires 1" in str(excinfo.value)

    wide_mobile = _mol(["C"] * 4, [[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
    with pytest.raises(ValueError) as excinfo:
        superpose(wide_mobile, reference, subset=True)
    assert "requires mobile to be no larger than reference" in str(excinfo.value)


def test_reference_subset_alignment_moves_whole_larger_mobile():
    reference_points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [8, 8, 8]], dtype=float)
    mobile_core = _rigid(reference_points[:4])
    mobile = _mol(["C", "N", "O", "H", "C", "H"],
                  np.vstack((mobile_core, [[30, 30, 30], [32, 31, 30]])))
    reference = _mol(["C", "N", "O", "H", "S"], reference_points)
    result = superpose_to_reference_subset(mobile, reference, [0, 1, 2, 3])
    assert result.method == "subset"
    assert result.n_fitted == 4
    assert result.rmsd < 1e-10
    assert np.allclose(result.transform.apply(mobile.positions[:4]), reference_points[:4])
    assert np.array_equal(result.mapping[:4], [0, 1, 2, 3])


def test_subset_alignment_prefers_bonded_match_over_disconnected_lower_rmsd():
    reference = _mol(
        ["C", "C", "C"],
        [[0, 0, 0], [1.5, 0, 0], [3.0, .2, 0]])
    reference.bonds = [(0, 1, 1), (1, 2, 1)]
    connected = _rigid(reference.positions + [[0, 0, 0], [0, .08, 0], [0, 0, 0]])
    disconnected = _rigid(reference.positions)
    mobile = _mol(["C"] * 6, np.vstack((connected, disconnected)))
    mobile.bonds = [(0, 1, 1), (1, 2, 1)]
    structures = StructureSet()
    structures.append(reference, label="reference")
    structures.append(mobile, label="mobile")

    result = structures.align_to_reference_subset(1, onto=0, ref_select=[0, 1, 2])
    assert np.array_equal(result.select, [0, 1, 2])
    assert result.rmsd > 0.0  # proves the disconnected exact copy was rejected
    assert result.rmsd < 0.1


def test_alignment_composes_with_visible_main_frame_transform():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    first = _mol(["C", "N", "O", "S"], points)
    second = _mol(["C", "N", "O", "S"], _rigid(points))
    third = _mol(["C", "N", "O", "S"], points + [20, -7, 4])
    structures = StructureSet()
    structures.append(first, label="first")
    structures.append(second, label="second")
    structures.append(third, label="third")

    structures.align(1, onto=0)
    assert not structures[1].transform.is_identity
    structures.set_active(1)
    result = structures.align_to_reference_subset(
        2, onto=1, ref_select=[0, 1, 2, 3])
    displayed_main = structures[1].transform.apply(second.positions)
    displayed_mobile = result.transform.apply(third.positions)
    assert np.allclose(displayed_mobile, displayed_main)


def test_long_backbone_uses_largest_viable_segment_for_shorter_mobile():
    def peptide(residues):
        symbols = []
        positions = []
        bonds = []
        selected = []
        for r in range(residues):
            i = len(symbols)
            x = 3.8 * r
            symbols.extend(("N", "C", "C", "O"))
            positions.extend(((x, 0, 0), (x + 1.4, .2, 0),
                              (x + 2.7, 0, 0), (x + 3.2, -1.1, 0)))
            bonds.extend(((i, i + 1, 1), (i + 1, i + 2, 1),
                          (i + 2, i + 3, 2)))
            if r:
                bonds.append((i - 2, i, 1))
            selected.extend((i, i + 1, i + 2))
        molecule = _mol(symbols, positions)
        molecule.bonds = bonds
        return molecule, selected

    reference, ref_select = peptide(5)
    mobile, _mobile_select = peptide(2)
    mobile.positions = _rigid(mobile.positions)
    structures = StructureSet()
    structures.append(reference, label="long")
    structures.append(mobile, label="short")

    result = structures.align_to_reference_subset(
        1, onto=0, ref_select=ref_select)
    assert result.n_fitted == 6
    assert len(result.select) == len(result.ref_select) == 6
    assert result.rmsd < 1e-10


def test_heavy_atom_subset_drops_terminal_reference_atom_missing_from_mobile():
    reference = _mol(
        ["C", "N", "O", "O"],
        [[0, 0, 0], [1.2, 0, 0], [0, 1.4, 0], [8, 8, 8]],
    )
    mobile_core = _rigid(reference.positions[:3])
    mobile = _mol(
        ["C", "N", "O", "H"],
        np.vstack((mobile_core, [[30, 30, 30]])),
    )
    structures = StructureSet()
    structures.append(reference, label="reference")
    structures.append(mobile, label="mobile")

    result = structures.align_to_reference_subset(
        1, onto=0, ref_select=[0, 1, 2, 3])
    assert result.n_fitted == 3
    assert result.rmsd < 1e-10
    assert np.array_equal(result.select, [0, 1, 2])
    assert np.array_equal(result.ref_select, [0, 1, 2])


def test_subset_alignment_reuses_correspondence_across_trajectory_frames(
        tmp_path, monkeypatch):
    reference = _mol(
        ["C", "N", "O", "O"],
        [[0, 0, 0], [1.2, 0, 0], [0, 1.4, 0], [8, 8, 8]],
    )
    core = _rigid(reference.positions[:3])
    first = _mol(["C", "N", "O", "H"],
                 np.vstack((core, [[30, 30, 30]])))
    second = _mol(["C", "N", "O", "H"],
                  np.vstack((core + [0.02, 0, 0], [[31, 30, 30]])))
    structures = StructureSet()
    structures.append(first, label="traj.xyz#1", path="/data/traj.xyz").marked = True
    structures.append(second, label="traj.xyz#2", path="/data/traj.xyz").marked = True
    structures.append(reference, label="reference.xyz",
                      path="/data/reference.xyz").marked = True
    structures.active_index = 2
    structures.overlay = True
    fd = os.open(str(tmp_path / "reuse.out"), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        viewer = Viewer(reference, structures=structures, fd_out=fd, backend="cpu")
        original = structures.align_to_reference_subset
        searches = []

        def counted_search(*args, **kwargs):
            searches.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(structures, "align_to_reference_subset", counted_search)
        viewer._align_overlay(ref_select=(0, 1, 2, 3))
        assert searches == [0]
        assert structures[0].alignment.n_fitted == 3
        assert structures[1].alignment.n_fitted == 3
    finally:
        os.close(fd)


def _overlay_viewer(tmp_path):
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    reference = _mol(["C", "N", "O", "H"], p, "reference")
    mobile = _mol(["C", "N", "O", "H", "C"],
                  np.vstack((_rigid(p), [[40, 40, 40]])), "mobile")
    sset = StructureSet()
    sset.append(reference, label="reference")
    sset.append(mobile, label="mobile")
    sset[1].marked = True
    fd = os.open(str(tmp_path / "viewer.out"), os.O_WRONLY | os.O_CREAT, 0o644)
    viewer = Viewer(reference, structures=sset, fd_out=fd, backend="cpu")
    return viewer, fd


def test_r_and_R_align_nothing_outside_overlay_but_say_why(tmp_path):
    """Neither key may touch the camera or a transform here. 'r' used to mean
    camera reset and no longer reaches the widget at all, so staying silent
    reads as a broken binding -- both keys explain themselves instead."""
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.widget.scene.camera.orbit(20, 10)
        rotation = viewer.widget.scene.camera.rotation.copy()
        assert viewer._dispatch([KeyEvent("r")]) is True
        assert "overlay" in viewer._msg
        assert viewer._dispatch([KeyEvent("R")]) is True
        assert "overlay" in viewer._msg
        assert np.array_equal(viewer.widget.scene.camera.rotation, rotation)
        assert viewer.widget.align_mode is False
        assert viewer.structures[1].transform.is_identity
    finally:
        os.close(fd)


def test_R_picks_reference_subset_and_enter_aligns_overlay(tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        assert viewer._dispatch([KeyEvent("R")]) is True
        assert viewer.widget.align_mode is True

        picks = iter([0, 1, 2, 3, 2])
        monkeypatch.setattr(viewer.widget, "_pick_active_only", lambda _x, _y: next(picks))
        for _ in range(4):
            assert viewer.widget._alignment_click(0, 0) is True
        assert viewer.widget.align_sel == [0, 1, 2, 3]
        # Picking an already selected atom toggles it off, then back on.
        assert viewer.widget._alignment_click(0, 0) is True
        assert viewer.widget.align_sel == [0, 1, 3]
        viewer.widget.align_sel.append(2)

        assert viewer._dispatch([KeyEvent("enter")]) is True
        assert viewer.widget.align_mode is False
        assert len(viewer._rmsd_columns) == 1
        column = viewer._rmsd_columns[0]
        assert column.header == "⊂RMSD #select1"
        assert column.labels == ("C0", "N1", "O2", "H3")
        assert column.values[0] == 0.0
        assert column.values[1] < 1e-10
        result = viewer.structures[1].alignment
        assert result is not None
        assert result.n_fitted == 4
        assert result.rmsd < 1e-10
        moved = viewer.structures[1].transform.apply(
            viewer.structures[1].molecule.positions[:4])
        assert np.allclose(moved, viewer.structures[0].molecule.positions)
    finally:
        os.close(fd)


def test_subset_rmsd_header_hover_click_and_R_recalculate(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer._cols = 200
        viewer._rows = 24
        viewer._list_w = 20
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2, 3))
        original = viewer._rmsd_columns[0].values[1]

        text = viewer._draw_list().decode("utf-8", "replace")
        assert "⊂RMSD #select1" in text
        assert len(viewer._subset_header_spans) == 1
        row, start, end, column_index = viewer._subset_header_spans[0]
        assert column_index == 0
        header_col = (start + end) // 2

        assert viewer._dispatch([
            MouseEvent("move", float(header_col), float(row), button=0)]) is True
        assert viewer._subset_hover_tip == "aligning on C0,N1,O2,H3"
        assert "aligning on C0,N1,O2,H3" in viewer._status_bar()

        assert viewer._dispatch([
            MouseEvent("down", float(header_col), float(row), button=0)]) is True
        assert viewer._active_subset_id == 1
        assert viewer.widget.align_sel == [0, 1, 2, 3]

        # Change a fitted source atom. R must update #select1 in place, not
        # create #select2 or enter a fresh picking session.
        viewer.structures[1].molecule.positions[1] += [0.25, 0.0, 0.0]
        viewer.structures[1].touch()
        assert viewer._dispatch([KeyEvent("R")]) is True
        assert len(viewer._rmsd_columns) == 1
        assert viewer._rmsd_columns[0].values[1] > original + 1e-4
        assert viewer.widget.align_mode is False
        assert viewer.widget.align_sel == [0, 1, 2, 3]
    finally:
        os.close(fd)


def test_subset_rmsd_column_x_deletes_saved_selection(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer._cols = 200
        viewer._rows = 24
        viewer._list_w = 20
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        viewer._activate_subset_column(0)
        viewer._draw_list()
        assert len(viewer._subset_remove_spans) == 1
        row, start, _end, column_index = viewer._subset_remove_spans[0]
        assert column_index == 0

        assert viewer._dispatch([
            MouseEvent("down", float(start), float(row), button=0)]) is True
        assert viewer._rmsd_columns == []
        assert viewer._active_subset_id is None
        assert viewer.widget.align_sel == []
        assert viewer._msg == "#select1 deleted"
    finally:
        os.close(fd)


def test_changing_main_frame_disarms_named_subset_before_global_r(
        tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        viewer._activate_subset_column(0)
        assert viewer._active_subset_id == 1
        assert viewer.widget.align_sel == [0, 1, 2]

        viewer._activate_structure(1)
        assert viewer.structures.active_index == 1
        assert viewer._active_subset_id is None
        assert viewer.widget.align_mode is False
        assert viewer.widget.align_sel == []
        # The column still exists; it is simply no longer armed.
        assert len(viewer._rmsd_columns) == 1

        calls = []
        monkeypatch.setattr(
            viewer, "_align_overlay",
            lambda *args, **kwargs: calls.append((args, kwargs)) or True)
        assert viewer._dispatch([KeyEvent("r")]) is True
        assert calls == [((), {})]
        assert viewer.structures.active_index == 1
    finally:
        os.close(fd)


def test_cycle_main_frame_clears_live_manual_selection(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.widget.set_alignment_mode(True)
        viewer.widget.align_sel = [0, 1, 2]
        viewer._cycle_frame(1)
        assert viewer.structures.active_index == 1
        assert viewer.widget.align_mode is False
        assert viewer.widget.align_sel == []
        assert viewer._active_subset_id is None
    finally:
        os.close(fd)


def test_global_r_rejects_stale_subset_after_external_active_change(
        tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        viewer._activate_subset_column(0)
        # Simulate an embedding changing StructureSet without the Viewer
        # helper. The r guard must still honor the actual active/main frame.
        viewer.structures.set_active(1)
        calls = []
        monkeypatch.setattr(
            viewer, "_align_overlay",
            lambda *args, **kwargs: calls.append((args, kwargs)) or True)

        assert viewer._dispatch([KeyEvent("r")]) is True
        assert viewer.structures.active_index == 1
        assert viewer._active_subset_id is None
        assert viewer.widget.align_sel == []
        assert calls == [((), {})]
    finally:
        os.close(fd)


def test_clicking_active_subset_header_disables_it_for_a_new_R_pick(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        viewer._activate_subset_column(0)
        assert viewer._active_subset_id == 1
        viewer._activate_subset_column(0)
        assert viewer._active_subset_id is None
        assert viewer.widget.align_sel == []
        assert viewer._dispatch([KeyEvent("R")]) is True
        assert viewer.widget.align_mode is True
        viewer.widget.align_sel = [0, 1, 3]
        assert viewer._dispatch([KeyEvent("enter")]) is True
        assert [column.header for column in viewer._rmsd_columns] == [
            "⊂RMSD #select1", "⊂RMSD #select2"]
    finally:
        os.close(fd)


def test_repeating_same_subset_after_adding_overlay_updates_column_in_place(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        column = viewer._rmsd_columns[0]
        assert viewer._next_subset_id == 2
        assert len(column.values) == 2

        source = viewer.structures[1].molecule
        added = _mol(source.symbols, source.positions + [7.0, -3.0, 2.0], "added")
        viewer.structures.append(added, label="added")
        viewer.structures[2].marked = True
        viewer.structures.overlay = True

        viewer._finish_subset_alignment((2, 1, 0))  # same set, different pick order
        assert viewer._rmsd_columns == [column]
        assert viewer._next_subset_id == 2
        assert column.indices == (0, 1, 2)
        assert len(column.values) == 3
        assert column.values[2] is not None
        assert viewer.structures[2].alignment is not None
    finally:
        os.close(fd)


def test_swapping_overlay_preserves_previous_rmsd_row(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        source = viewer.structures[1].molecule
        third = _mol(source.symbols, source.positions + [5.0, -4.0, 3.0], "third")
        viewer.structures.append(third, label="third")
        viewer.structures.overlay = True
        # First run: only structure 2 is tinted.
        viewer.structures[1].marked = True
        viewer.structures[2].marked = False
        viewer._finish_subset_alignment((0, 1, 2))
        column = viewer._rmsd_columns[0]
        previous = column.values[1]
        assert previous is not None
        assert column.values[2] is None

        # Swap structure 2 out and structure 3 in, then repeat the same set.
        viewer.structures[1].marked = False
        viewer.structures[2].marked = True
        viewer._finish_subset_alignment((0, 1, 2))
        assert len(viewer._rmsd_columns) == 1
        assert column.values[1] == previous
        assert column.values[2] is not None
    finally:
        os.close(fd)


def test_lowercase_r_aligns_complete_matching_overlay(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        # Use equal topologies so lowercase r takes the microsecond Kabsch path.
        viewer.structures[1].molecule.symbols.pop()
        viewer.structures[1].molecule.positions = viewer.structures[1].molecule.positions[:4]
        viewer.structures.overlay = True
        assert viewer._dispatch([KeyEvent("r")]) is True
        result = viewer.structures[1].alignment
        assert result is not None and result.method == "permute"
        assert result.rmsd < 1e-10
    finally:
        os.close(fd)


def test_kabsch_fast_path_handles_thousands_of_atoms_quickly():
    rng = np.random.default_rng(7)
    p = rng.normal(size=(5000, 3))
    q = _rigid(p)
    start = time.perf_counter()
    _rotation, _translation, rmsd = kabsch(p, q)
    elapsed = time.perf_counter() - start
    assert rmsd < 1e-10
    # Generous enough for shared CI, strict enough to catch an accidental O(n²) path.
    assert elapsed < 0.25


def test_alignment_api_is_public():
    assert vimol.kabsch is kabsch
    assert callable(vimol.superpose)


def test_pdb_preserves_atom_identity_and_backbone_presets():
    pdb = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.000   2.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.000   0.000   0.000  1.00  0.00           C
HETATM    6  CA  CA  A   2       9.000   9.000   9.000  1.00  0.00          CA
END
"""
    molecule = vimol.loads(pdb, "pdb")
    assert molecule.atom_names == ["N", "CA", "C", "O", "CB", "CA"]
    assert len(molecule.atom_keys) == molecule.n_atoms
    assert np.array_equal(vimol.select.peptide_backbone(molecule), [0, 1, 2])
    assert np.array_equal(
        vimol.select.peptide_backbone(molecule, include_beta_carbon=True),
        [0, 1, 2, 4])


def test_shift_S_and_clickable_hint_open_backbone_menu(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer._cols = 100
        viewer._rows = 24
        viewer._list_w = 20
        # Shift+S is a selection tool, so it is available before any overlay
        # is created as well as while comparing structures.
        viewer.structures.overlay = False
        viewer._draw_list()
        assert viewer._select_hint_span is not None
        row, start, end = viewer._select_hint_span
        text = viewer._draw_list().decode("utf-8", "replace")
        assert "Shft+S to" in text and "select" in text

        assert viewer._dispatch([
            MouseEvent("down", float((start + end) // 2), float(row), button=0)]) is True
        assert viewer._mode == "selection_picker"
        # Clicking the visible hint again is a toggle.
        assert viewer._dispatch([
            MouseEvent("down", float((start + end) // 2), float(row), button=0)]) is True
        assert viewer._mode == "normal"
        assert viewer._dispatch([KeyEvent("S")]) is True
        assert viewer._mode == "selection_picker"
    finally:
        os.close(fd)


def test_backbone_plus_cb_menu_selects_main_frame_only(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        reference = viewer.structures[0].molecule
        reference.atom_names = ["N", "CA", "C", "CB"]
        reference.atom_is_hetatm = [False] * 4
        reference.atom_keys = [f"ATOM|A|1||{name}|" for name in reference.atom_names]
        mobile = viewer.structures[1].molecule
        mobile.atom_names = ["CB", "N", "CA", "C", "O"]
        mobile.atom_is_hetatm = [False] * 5
        mobile.atom_keys = [f"ATOM|B|1||{name}|" for name in mobile.atom_names]
        viewer.structures.overlay = True

        assert viewer._dispatch([KeyEvent("S")]) is True
        assert viewer._dispatch([KeyEvent("down")]) is True
        assert viewer._dispatch([KeyEvent("down")]) is True
        assert viewer._selection_menu_idx == 2
        assert viewer._dispatch([KeyEvent("enter")]) is True
        assert viewer._mode == "normal"
        assert viewer.widget.align_mode is True
        assert viewer.widget.align_sel == [0, 1, 2, 3]
        assert "4 main-frame atoms selected" in viewer._msg
        # No selection state is ever written to the tinted molecule.
        assert mobile.atom_names == ["CB", "N", "CA", "C", "O"]
    finally:
        os.close(fd)


def test_backbone_preset_reports_missing_names(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer._open_selection_picker()
        viewer._selection_menu_idx = 1
        viewer._activate_selection_preset()
        assert viewer.widget.align_mode is False
        assert "no peptide-backbone motif" in viewer._msg
    finally:
        os.close(fd)


def test_heavy_atoms_preset_reports_an_all_hydrogen_frame(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer.structures[0].molecule.symbols = ["H"] * 4
        viewer._open_selection_picker()
        viewer._selection_menu_idx = 3
        viewer._activate_selection_preset()
        assert viewer.widget.align_mode is False
        assert viewer.widget.align_sel == []
        assert "no heavy atoms" in viewer._msg
    finally:
        os.close(fd)


def test_ring_system_preset_reports_an_acyclic_frame(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        molecule = viewer.structures[0].molecule
        molecule.bonds = [(0, 1, 1), (1, 2, 1), (0, 3, 1)]
        viewer._open_selection_picker()
        viewer._selection_menu_idx = 4
        viewer._activate_selection_preset()
        assert viewer.widget.align_mode is False
        assert viewer.widget.align_sel == []
        assert "no ring system" in viewer._msg

        # Control: one more bond closes C-N-O, so the empty result above is
        # the acyclic graph talking and not an unreachable preset.
        molecule.bonds = molecule.bonds + [(0, 2, 1)]
        viewer._open_selection_picker()
        viewer._selection_menu_idx = 4
        viewer._activate_selection_preset()
        assert viewer.widget.align_mode is True
        assert viewer.widget.align_sel == [0, 1, 2]
    finally:
        os.close(fd)


def test_heavy_atoms_excludes_hydrogen_only():
    molecule = _mol(
        ["H", "C", "N", "O", "S", "H"],
        np.zeros((6, 3)),
    )
    assert np.array_equal(vimol.select.heavy_atoms(molecule), [1, 2, 3, 4])


def test_largest_ring_system_selects_largest_disjoint_ring():
    molecule = _mol(["C"] * 10, np.zeros((10, 3)))
    # A triangle, a six-membered ring, and an acyclic tail.
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (0, 2, 1),
        (3, 4, 1), (4, 5, 1), (5, 6, 1),
        (6, 7, 1), (7, 8, 1), (3, 8, 1), (8, 9, 1),
    ]
    assert np.array_equal(vimol.select.largest_ring_system(molecule),
                          [3, 4, 5, 6, 7, 8])


def test_largest_ring_system_keeps_fused_rings_whole_whatever_the_atom_order():
    # Naphthalene: two six-rings sharing the 4--9 edge. A depth-first cycle
    # basis answers 6 or 10 here depending on the file's atom order; the whole
    # fused system is the only order-independent answer.
    base = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 9), (9, 0),
            (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    for shift in range(10):
        relabel = [(i + shift) % 10 for i in range(10)]
        molecule = _mol(["C"] * 10, np.zeros((10, 3)))
        molecule.bonds = [(relabel[i], relabel[j], 1) for i, j in base]
        assert np.array_equal(vimol.select.largest_ring_system(molecule),
                              np.arange(10))


def test_largest_ring_system_ignores_hydrogens_and_returns_empty_for_chain():
    molecule = _mol(["C", "C", "C", "H"], np.zeros((4, 3)))
    molecule.bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
    assert vimol.select.largest_ring_system(molecule).size == 0


def test_largest_ring_system_strips_substituents_down_to_the_cycle():
    # A benzene ring carrying a three-atom tail: peeling must not stop after
    # one pass, or the atom that the terminal atom was hanging off survives.
    molecule = _mol(["C"] * 9, np.zeros((9, 3)))
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 0, 1),
        (0, 6, 1), (6, 7, 1), (7, 8, 1),
    ]
    assert np.array_equal(vimol.select.largest_ring_system(molecule),
                          [0, 1, 2, 3, 4, 5])


def test_largest_ring_system_perceives_bonds_for_unnamed_xyz():
    angles = np.arange(6) * (np.pi / 3.0)
    positions = np.column_stack((1.4 * np.cos(angles),
                                 1.4 * np.sin(angles),
                                 np.zeros(6)))
    molecule = _mol(["C"] * 6, positions)
    assert molecule.bonds == []
    assert np.array_equal(vimol.select.largest_ring_system(molecule),
                          np.arange(6))


def test_new_selection_presets_select_main_frame_only(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        reference = viewer.structures[0].molecule
        reference.symbols = ["C", "C", "C", "C", "C", "C", "H"]
        reference.positions = np.zeros((7, 3))
        reference.bonds = [
            (0, 1, 1), (1, 2, 1), (2, 3, 1),
            (3, 4, 1), (4, 5, 1), (0, 5, 1), (0, 6, 1),
        ]
        mobile = viewer.structures[1].molecule
        mobile_symbols = list(mobile.symbols)

        viewer._open_selection_picker()
        viewer._selection_menu_idx = 3
        viewer._activate_selection_preset()
        assert viewer.widget.align_sel == [0, 1, 2, 3, 4, 5]
        assert "Heavy atoms" in viewer._msg

        viewer._open_selection_picker()
        viewer._selection_menu_idx = 4
        viewer._activate_selection_preset()
        assert viewer.widget.align_sel == [0, 1, 2, 3, 4, 5]
        assert "Largest ring system" in viewer._msg
        assert mobile.symbols == mobile_symbols
    finally:
        os.close(fd)


def test_unnamed_backbone_is_inferred_from_one_bond_graph_pass():
    # N-CA-C(=O), with CB plus an amide side chain hanging off CB. The latter
    # must not be mistaken for peptide backbone.
    molecule = _mol(
        ["N", "C", "C", "O", "C", "C", "O", "N"],
        [[0, 0, 0], [1.45, 0, 0], [2.8, 0, 0], [3.7, 0.7, 0],
         [1.45, 1.5, 0], [1.45, 2.9, 0], [2.3, 3.7, 0], [0.5, 3.5, 0]])
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (2, 3, 2), (1, 4, 1),
        (4, 5, 1), (5, 6, 2), (5, 7, 1),
    ]
    assert np.array_equal(vimol.select.peptide_backbone(molecule), [0, 1, 2])
    assert np.array_equal(
        vimol.select.peptide_backbone(molecule, include_beta_carbon=True),
        [0, 1, 2, 4])


def test_inferred_backbone_rejects_spurious_carbonyl_to_beta_bond():
    molecule = _mol(
        ["N", "C", "C", "O", "C"],
        [[0, 0, 0], [1.45, 0, 0], [2.8, 0, 0], [3.7, 0.7, 0],
         [1.45, 1.5, 0]])
    # Some non-PDB bond perception can include this longer 2--4 contact.
    # It must not make Cβ look like a second Cα.
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (2, 3, 2), (1, 4, 1), (2, 4, 1),
    ]
    assert np.array_equal(vimol.select.peptide_backbone(molecule), [0, 1, 2])
    assert np.array_equal(
        vimol.select.peptide_backbone(molecule, include_beta_carbon=True),
        [0, 1, 2, 4])


def test_hydrogen_free_threonine_hydroxyl_is_not_a_carbonyl():
    # N-CA-C(=O), with threonine's CA-CB(-OH)-CG2 side chain. Without the
    # hydroxyl hydrogen, O-gamma is terminal in the heavy-atom graph; bond
    # order / length must distinguish it from the true carbonyl oxygen.
    molecule = _mol(
        ["N", "C", "C", "O", "C", "O", "C"],
        [[0, 0, 0], [1.45, 0, 0], [2.8, 0, 0], [3.75, 0.75, 0],
         [1.45, 1.53, 0], [0.28, 2.31, 0], [2.72, 2.28, 0]])
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (2, 3, 2),
        (1, 4, 1), (4, 5, 1), (4, 6, 1),
    ]
    assert np.array_equal(vimol.select.peptide_backbone(molecule), [0, 1, 2])
    assert np.array_equal(
        vimol.select.peptide_backbone(molecule, include_beta_carbon=True),
        [0, 1, 2, 4])


def test_hydrogen_free_threonine_is_safe_with_perceived_single_bonds():
    # XYZ-style input has no bond orders, so the same distinction falls back
    # to the short carbonyl C-O distance and remains a linear graph pass.
    molecule = _mol(
        ["N", "C", "C", "O", "C", "O", "C"],
        [[0, 0, 0], [1.45, 0, 0], [2.8, 0, 0], [3.75, 0.75, 0],
         [1.45, 1.53, 0], [0.28, 2.31, 0], [2.72, 2.28, 0]])
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (2, 3, 1),
        (1, 4, 1), (4, 5, 1), (4, 6, 1),
    ]
    assert np.array_equal(vimol.select.peptide_backbone(molecule), [0, 1, 2])
    assert np.array_equal(
        vimol.select.peptide_backbone(molecule, include_beta_carbon=True),
        [0, 1, 2, 4])


def test_inferred_backbone_keeps_only_continuous_peptide_path():
    # Two linked residues plus an isolated N-CA-C=O lookalike at the far
    # right. The isolated C-C pair must not join the peptide subset.
    molecule = _mol(
        ["N", "C", "C", "O", "N", "C", "C", "O",
         "C", "O", "C", "N"],
        [[0, 0, 0], [1.45, 0, 0], [2.8, 0, 0], [3.7, .7, 0],
         [3.9, -1.0, 0], [5.3, -1.0, 0], [6.65, -1.0, 0], [7.55, -.3, 0],
         [12, 0, 0], [12.9, .7, 0], [10.65, 0, 0], [9.3, 0, 0]])
    molecule.bonds = [
        (0, 1, 1), (1, 2, 1), (2, 3, 2), (2, 4, 1),
        (4, 5, 1), (5, 6, 1), (6, 7, 2),
        (8, 9, 2), (8, 10, 1), (10, 11, 1),
    ]
    assert np.array_equal(
        vimol.select.peptide_backbone(molecule), [0, 1, 2, 4, 5, 6])


def test_mouse_clicking_popup_option_selects_inferred_backbone(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        reference = viewer.structures[0].molecule
        reference.symbols = ["N", "C", "C", "O"]
        reference.positions = np.array(
            [[0, 0, 0], [1.45, 0, 0], [2.8, 0, 0], [3.7, 0.7, 0]], dtype=float)
        reference.bonds = [(0, 1, 1), (1, 2, 1), (2, 3, 2)]
        viewer._cols = 100
        viewer._rows = 24
        viewer._list_w = 18
        viewer.structures.overlay = True
        viewer._open_selection_picker()
        top, left, width, _height = viewer._selection_menu_geometry()

        assert viewer._dispatch([
            MouseEvent("down", float(left + width // 2), float(top + 2), button=0)
        ]) is True
        assert viewer._mode == "normal"
        assert viewer.widget.align_mode is True
        assert viewer.widget.align_sel == [0, 1, 2]
        assert "(inferred)" in viewer._msg
    finally:
        os.close(fd)


def test_manual_selection_works_without_overlay_and_whitespace_clears(
        tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = False
        assert viewer._dispatch([KeyEvent("S")]) is True
        assert viewer._dispatch([KeyEvent("enter")]) is True  # Manual
        assert viewer.widget.align_mode is True

        picks = iter([0, 2, None])
        monkeypatch.setattr(
            viewer.widget, "_pick_active_only", lambda _x, _y: next(picks))
        assert viewer.widget._alignment_click(0, 0) is True
        assert viewer.widget._alignment_click(0, 0) is True
        assert viewer.widget.align_sel == [0, 2]
        assert viewer.widget._alignment_click(0, 0) is True
        assert viewer.widget.align_sel == []
    finally:
        os.close(fd)


def test_manual_selection_survives_adding_overlay_and_lowercase_r_aligns(tmp_path):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = False
        viewer.structures[1].marked = False
        viewer._open_selection_picker()
        viewer._activate_selection_preset()  # Manual
        viewer.widget.align_sel = [0, 1, 2, 3]

        viewer._list_click(1, opt=True)
        assert viewer.structures.overlay is True
        assert viewer.widget.align_sel == [0, 1, 2, 3]
        assert viewer._dispatch([KeyEvent("r")]) is True
        assert viewer.widget.align_mode is False
        assert len(viewer._rmsd_columns) == 1
        assert viewer._rmsd_columns[0].indices == (0, 1, 2, 3)
        assert viewer.structures[1].alignment is not None
        assert viewer.structures[1].alignment.rmsd < 1e-10
    finally:
        os.close(fd)


def test_manual_from_named_selection_is_additive_and_saves_a_child(
        tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        viewer._activate_subset_column(0)
        parent = viewer._rmsd_columns[0]
        assert viewer.widget.align_sel == [0, 1, 2]

        assert viewer._dispatch([KeyEvent("S")]) is True
        assert viewer._dispatch([KeyEvent("enter")]) is True  # Manual
        assert viewer.widget.align_mode is True
        assert viewer.widget.align_sel == [0, 1, 2]
        assert viewer._active_subset_id is None

        monkeypatch.setattr(viewer.widget, "_pick_active_only", lambda _x, _y: 3)
        assert viewer.widget._alignment_click(0, 0) is True
        assert viewer.widget.align_sel == [0, 1, 2, 3]
        # The derived selection has no persistent identity before an RMSD.
        assert len(viewer._rmsd_columns) == 1
        assert viewer._dispatch([KeyEvent("r")]) is True
        assert len(viewer._rmsd_columns) == 2
        assert parent.indices == (0, 1, 2)
        assert viewer._rmsd_columns[1].indices == (0, 1, 2, 3)
        assert viewer._rmsd_columns[1].select_id != parent.select_id
    finally:
        os.close(fd)


def test_option_click_derives_from_named_selection_without_reset(
        tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer._cols = 100
        viewer._rows = 30
        viewer._list_w = 20
        viewer.structures.overlay = True
        viewer._finish_subset_alignment((0, 1, 2))
        viewer._activate_subset_column(0)
        parent = viewer._rmsd_columns[0]
        monkeypatch.setattr(viewer.widget, "_active_local_pick", lambda _x, _y: 3)
        monkeypatch.setattr(viewer.widget, "_pick_active_only", lambda _x, _y: 3)

        # The down arms additive picking; the up performs the atom toggle.
        viewer._dispatch([MouseEvent(
            "down", 40.0, 5.0, button=0, alt=True, pixel=False)])
        assert viewer.widget.align_sel == [0, 1, 2]
        assert viewer._dispatch([MouseEvent(
            "up", 40.0, 5.0, button=0, alt=True, pixel=False)]) is True
        assert viewer.widget.align_sel == [0, 1, 2, 3]
        assert viewer._active_subset_id is None
        assert len(viewer._rmsd_columns) == 1

        assert viewer._dispatch([KeyEvent("r")]) is True
        assert len(viewer._rmsd_columns) == 2
        assert parent.indices == (0, 1, 2)
        assert viewer._rmsd_columns[1].indices == (0, 1, 2, 3)
    finally:
        os.close(fd)


def test_option_click_starts_manual_selection_without_overlay(
        tmp_path, monkeypatch):
    viewer, fd = _overlay_viewer(tmp_path)
    try:
        viewer._cols = 100
        viewer._rows = 30
        viewer._list_w = 20
        viewer.structures.overlay = False
        monkeypatch.setattr(viewer.widget, "_active_local_pick", lambda _x, _y: 2)
        monkeypatch.setattr(viewer.widget, "_pick_active_only", lambda _x, _y: 2)

        viewer._dispatch([MouseEvent(
            "down", 40.0, 5.0, button=0, alt=True, pixel=False)])
        assert viewer._dispatch([MouseEvent(
            "up", 40.0, 5.0, button=0, alt=True, pixel=False)]) is True
        assert viewer.widget.align_mode is True
        assert viewer.widget.align_sel == [2]
        assert viewer._rmsd_columns == []
    finally:
        os.close(fd)


def test_pdb_keyed_backbone_alignment_bypasses_geometric_search_at_protein_scale():
    rng = np.random.default_rng(23)
    n = 800
    reference_positions = rng.normal(size=(n, 3))
    mobile_positions = _rigid(reference_positions)
    names = ["CA"] * n
    keys = [f"ATOM|A|{i + 1}||CA|" for i in range(n)]
    reference = _mol(["C"] * n, reference_positions)
    mobile = _mol(["C"] * n, mobile_positions)
    reference.atom_names = names.copy()
    mobile.atom_names = names.copy()
    reference.atom_is_hetatm = [False] * n
    mobile.atom_is_hetatm = [False] * n
    reference.atom_keys = keys.copy()
    mobile.atom_keys = keys.copy()
    structures = StructureSet()
    structures.append(reference, label="reference")
    structures.append(mobile, label="mobile")

    start = time.perf_counter()
    result = structures.align_to_reference_subset(
        1, onto=0, ref_select=np.arange(n, dtype=np.int64))
    elapsed = time.perf_counter() - start
    assert result.n_fitted == n
    assert result.rmsd < 1e-10
    assert elapsed < 0.25
