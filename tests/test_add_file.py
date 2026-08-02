import os

import pytest

import vimol
from vimol import file_dialog
from vimol.input import KeyEvent
from vimol.viewer import Viewer, _FullRMSDColumn, _SubsetRMSDColumn


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
