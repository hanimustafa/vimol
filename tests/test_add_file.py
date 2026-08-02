import os

import pytest

import vimol
from vimol import file_dialog
from vimol.input import KeyEvent
from vimol.viewer import Viewer, _FullRMSDColumn, _SubsetRMSDColumn

EX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _write_xyz(path, comments):
    path.write_text("".join(
        f"1\n{comment}\nHe 0.0 0.0 {i}.0\n"
        for i, comment in enumerate(comments)
    ))


def test_macos_file_picker_returns_selected_path(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = "/data/chosen molecule.xyz\n"
        stderr = ""

    monkeypatch.setattr(file_dialog.sys, "platform", "darwin")
    monkeypatch.setattr(file_dialog.subprocess, "run",
                        lambda *args, **kwargs: (calls.append((args, kwargs)), Result())[1])

    assert file_dialog.choose_structure_file() == "/data/chosen molecule.xyz"
    command = calls[0][0][0]
    assert command[0] == "/usr/bin/osascript"
    assert "choose file" in command[-1]
    assert calls[0][1]["shell"] is False


def test_macos_file_picker_cancel_returns_none(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "execution error: User canceled. (-128)"

    monkeypatch.setattr(file_dialog.sys, "platform", "darwin")
    monkeypatch.setattr(file_dialog.subprocess, "run", lambda *a, **k: Result())
    assert file_dialog.choose_structure_file() is None


def test_macos_file_picker_surfaces_unexpected_failure(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "execution error: automation denied (-1743)"

    monkeypatch.setattr(file_dialog.sys, "platform", "darwin")
    monkeypatch.setattr(file_dialog.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(file_dialog.FileDialogError, match="automation denied"):
        file_dialog.choose_structure_file()


def test_file_picker_reports_unsupported_platform(monkeypatch):
    monkeypatch.setattr(file_dialog.sys, "platform", "linux")
    with pytest.raises(file_dialog.FileDialogError, match="macOS"):
        file_dialog.choose_structure_file()


def test_uppercase_a_adds_single_file_as_overlay(tmp_path, monkeypatch):
    initial = tmp_path / "initial.xyz"
    added = tmp_path / "added.xyz"
    _write_xyz(initial, ["initial"])
    _write_xyz(added, ["added"])
    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd)
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(added))

        assert v._dispatch([KeyEvent("A")])
        assert v.structures.labels == ["initial.xyz", "added.xyz"]
        assert [entry.path for entry in v.structures] == [str(initial), str(added)]
        assert [entry.marked for entry in v.structures] == [True, True]
        assert v.structures.overlay is True
        assert v.structures.active_index == 0
        assert v._msg == "added added.xyz"
    finally:
        os.close(fd)


def test_uppercase_a_adds_every_frame_but_marks_only_first(tmp_path, monkeypatch):
    initial = tmp_path / "initial.xyz"
    trajectory = tmp_path / "trajectory.xyz"
    _write_xyz(initial, ["initial"])
    _write_xyz(trajectory, ["one", "two", "three"])
    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd)
        monkeypatch.setattr(file_dialog, "choose_structure_file",
                            lambda: str(trajectory))

        v._dispatch([KeyEvent("A")])
        assert v.structures.labels == [
            "initial.xyz", "trajectory.xyz#1", "trajectory.xyz#2",
            "trajectory.xyz#3",
        ]
        assert [entry.marked for entry in v.structures] == [True, True, False, False]
        assert [entry.molecule.name for entry in v.structures[1:]] == ["one", "two", "three"]
    finally:
        os.close(fd)


def test_add_file_extends_existing_rmsd_value_arrays(tmp_path, monkeypatch):
    initial = tmp_path / "initial.xyz"
    added = tmp_path / "added.xyz"
    _write_xyz(initial, ["initial"])
    _write_xyz(added, ["one", "two"])
    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd)
        v._full_rmsd_columns.append(_FullRMSDColumn(1, 0, 0, [None]))
        v._rmsd_columns.append(_SubsetRMSDColumn(
            1, 0, 0, (0,), ("He1",), [None]))
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(added))

        v._dispatch([KeyEvent("A")])
        assert v._full_rmsd_columns[0].values == [None, None, None]
        assert v._rmsd_columns[0].values == [None, None, None]
    finally:
        os.close(fd)


def test_add_file_cancel_and_parse_failure_do_not_mutate_viewer(tmp_path, monkeypatch):
    initial = tmp_path / "initial.xyz"
    broken = tmp_path / "broken.xyz"
    _write_xyz(initial, ["initial"])
    broken.write_text("not molecular data\n")
    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd)
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: None)
        assert v._dispatch([KeyEvent("A")])
        assert len(v.structures) == 1
        assert v._msg == "add file cancelled"

        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(broken))
        assert v._dispatch([KeyEvent("A")])
        assert len(v.structures) == 1
        assert v._msg.startswith("could not add broken.xyz:")
    finally:
        os.close(fd)


def test_add_file_disambiguates_repeated_basenames(tmp_path, monkeypatch):
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    third_dir = tmp_path / "three"
    first_dir.mkdir()
    second_dir.mkdir()
    third_dir.mkdir()
    paths = [directory / "same.xyz" for directory in (first_dir, second_dir, third_dir)]
    for i, path in enumerate(paths):
        _write_xyz(path, [str(i)])
    mol = vimol.load(str(paths[0]))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(paths[0]), fd_out=fd)
        choices = iter(map(str, paths[1:]))
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: next(choices))
        v._dispatch([KeyEvent("A")])
        v._dispatch([KeyEvent("A")])
        assert v.structures.labels == ["same.xyz", "same.xyz~2", "same.xyz~3"]
    finally:
        os.close(fd)


def test_add_file_respects_disabled_automatic_bond_perception(tmp_path, monkeypatch):
    initial = tmp_path / "initial.xyz"
    added = tmp_path / "added.xyz"
    initial.write_text("1\ninitial\nC 0 0 0\n")
    added.write_text("2\nclose carbons\nC 0 0 0\nC 1.2 0 0\n")
    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd, auto_bonds=False)
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(added))
        v._dispatch([KeyEvent("A")])
        assert v.structures[1].molecule.bonds == []
    finally:
        os.close(fd)


def test_add_file_preserves_established_overlay_membership(tmp_path, monkeypatch):
    paths = [tmp_path / f"{name}.xyz" for name in ("a", "b", "c")]
    for path in paths:
        _write_xyz(path, [path.stem])
    mol = vimol.load(str(paths[0]))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(paths[0]), fd_out=fd)
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(paths[1]))
        v._dispatch([KeyEvent("A")])
        v.structures[0].marked = False
        v.structures[1].marked = True

        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(paths[2]))
        v._dispatch([KeyEvent("A")])
        assert [entry.marked for entry in v.structures] == [False, True, True]
    finally:
        os.close(fd)


def test_viewer_honours_no_bonds_on_the_single_file_startup_path(tmp_path):
    """VIM-22: the flag survived parsing, then Viewer.__init__ re-perceived
    bonds unconditionally and undid it. Threading it through is what makes
    the picker able to load added files under the same rule."""
    path = tmp_path / "h2.xyz"
    path.write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n")
    molecules = vimol.parsers.load_all(str(path))
    assert molecules[0].bonds == []                  # the parser respected it
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        Viewer(molecules[0], frames=molecules, backend="cpu",
               source_path=str(path), fd_out=fd, auto_bonds=False)
        assert molecules[0].bonds == []              # ...and so does the viewer

        fresh = vimol.parsers.load_all(str(path))
        Viewer(fresh[0], frames=fresh, backend="cpu",
               source_path=str(path), fd_out=fd)
        assert len(fresh[0].bonds) == 1              # default still perceives
    finally:
        os.close(fd)


def test_add_file_leaves_a_frozen_measurement_column_intact(tmp_path, monkeypatch):
    """Adding a file must not disturb a column the user already pinned: the
    old rows keep their values and the new row simply joins as unmeasurable."""
    initial = tmp_path / "initial.xyz"
    initial.write_text("2\ninitial\nC 0 0 0\nN 1.0 0 0\n")
    same = tmp_path / "same.xyz"
    same.write_text("2\nsame\nC 0 0 0\nN 1.5 0 0\n")
    other = tmp_path / "other.xyz"
    other.write_text("1\nlone\nHe 0 0 0\n")

    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd, backend="cpu")
        v._update_geometry()
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(same))
        v._dispatch([KeyEvent("A")])
        v._freeze_measure_sel((0, 1))
        before = list(v._measure_layout(v._list_w)[0][2])
        assert before == ["1.000↓", "1.500↑"]

        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(other))
        v._dispatch([KeyEvent("A")])
        after = list(v._measure_layout(v._list_w)[0][2])
        # The two original rows are untouched; the one-atom file cannot
        # resolve a two-atom distance and says so rather than shifting them.
        assert after == ["1.000↓", "1.500↑", "—"]
    finally:
        os.close(fd)


def test_adding_the_same_path_twice_stays_one_file_section(tmp_path, monkeypatch):
    """Grouping keys on the source path, so re-adding a file extends its
    existing section rather than opening a second one with the same name."""
    path = tmp_path / "repeat.xyz"
    _write_xyz(path, ["only"])
    mol = vimol.load(os.path.join(EX_DIR, "methane.xyz"))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path="methane.xyz", fd_out=fd, backend="cpu")
        v._update_geometry()
        monkeypatch.setattr(file_dialog, "choose_structure_file", lambda: str(path))
        v._dispatch([KeyEvent("A")])
        v._dispatch([KeyEvent("A")])

        assert len(v.structures) == 3
        # Labels stay distinct so alignment failures name the right entry...
        assert [e.label for e in v.structures][1:] == ["repeat.xyz", "repeat.xyz~2"]
        # ...but the strip shows one section, because it is one file.
        headers = [t for kind, _i, t in v._list_display_rows() if kind == "group"]
        assert headers.count("repeat.xyz") == 1
        frames = [t for kind, _i, t in v._list_display_rows() if kind == "struct"]
        assert frames[-2:] == ["frame 1", "frame 2"]
    finally:
        os.close(fd)
