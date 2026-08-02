import os

import pytest

import vimol
from vimol.input import KeyEvent
from vimol.viewer import Viewer, _FullRMSDColumn, _SubsetRMSDColumn

EX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def _write_xyz(path, comments):
    path.write_text("".join(
        f"1\n{comment}\nHe 0.0 0.0 {i}.0\n"
        for i, comment in enumerate(comments)
    ))


def test_uppercase_a_adds_single_file_as_overlay(tmp_path, monkeypatch):
    initial = tmp_path / "initial.xyz"
    added = tmp_path / "added.xyz"
    _write_xyz(initial, ["initial"])
    _write_xyz(added, ["added"])
    mol = vimol.load(str(initial))
    fd = os.open(os.devnull, os.O_WRONLY)
    try:
        v = Viewer(mol, source_path=str(initial), fd_out=fd)

        assert v._add_file(str(added))
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

        v._add_file(str(trajectory))
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

        v._add_file(str(added))
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
        # Cancelling is now "open the browser, press Esc" rather than a
        # dialog returning None, but it must still leave the set untouched.
        v._dispatch([KeyEvent("A")])
        v._dispatch([KeyEvent("escape")])
        assert v._mode == "normal"
        assert len(v.structures) == 1

        assert v._add_file(str(broken)) is False
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
        for path in paths[1:]:
            v._add_file(str(path))
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
        v._add_file(str(added))
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
        v._add_file(str(paths[1]))
        v.structures[0].marked = False
        v.structures[1].marked = True

        v._add_file(str(paths[2]))
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
        v._add_file(str(same))
        v._freeze_measure_sel((0, 1))
        before = list(v._measure_layout(v._list_w)[0][2])
        assert before == ["1.000↓", "1.500↑"]

        v._add_file(str(other))
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
        v._add_file(str(path))
        v._add_file(str(path))

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


def _browser_viewer(tmp_path, start):
    """A viewer whose active structure lives in *start*, so A opens there."""
    mol = vimol.load(os.path.join(EX_DIR, "methane.xyz"))
    fd = os.open(os.devnull, os.O_WRONLY)
    v = Viewer(mol, source_path=os.path.join(start, "methane.xyz"),
               fd_out=fd, backend="cpu")
    v._cols, v._rows = 100, 24
    v._update_geometry()
    return v, fd


def test_uppercase_a_opens_the_browser_at_the_active_structures_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "here.xyz").write_text("1\nhere\nHe 0 0 0\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        assert v._dispatch([KeyEvent("A")]) is True
        assert v._mode == "file_browser"
        assert v._browser_dir == str(tmp_path)
        assert [e.name for e in v._browser_entries] == ["..", "sub", "here.xyz"]
    finally:
        os.close(fd)


def test_browser_enter_descends_into_a_directory_and_parent_returns(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.xyz").write_text("1\ndeep\nHe 0 0 0\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        v._browser_idx = 1                       # 'sub'
        v._dispatch([KeyEvent("enter")])
        assert v._browser_dir == str(nested)
        assert [e.name for e in v._browser_entries] == ["..", "deep.xyz"]
        assert v._browser_idx == 0               # cursor resets, not left dangling

        v._dispatch([KeyEvent("enter")])         # '..'
        assert v._browser_dir == str(tmp_path)
    finally:
        os.close(fd)


def test_browser_enter_on_a_file_adds_it_and_closes(tmp_path):
    (tmp_path / "added.xyz").write_text("2\nadded\nHe 0 0 0\nHe 0 0 1\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        v._browser_idx = [e.name for e in v._browser_entries].index("added.xyz")
        v._dispatch([KeyEvent("enter")])
        assert v._mode == "normal"
        assert len(v.structures) == 2
        assert v.structures[1].path == str(tmp_path / "added.xyz")
        assert "added" in v._msg
    finally:
        os.close(fd)


def test_browser_escape_cancels_without_touching_the_structure_set(tmp_path):
    (tmp_path / "ignored.xyz").write_text("1\nignored\nHe 0 0 0\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        v._dispatch([KeyEvent("escape")])
        assert v._mode == "normal"
        assert len(v.structures) == 1
    finally:
        os.close(fd)


def test_browser_cursor_clamps_at_both_ends(tmp_path):
    for name in ("a.xyz", "b.xyz"):
        (tmp_path / name).write_text("1\nx\nHe 0 0 0\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        for _ in range(10):
            v._dispatch([KeyEvent("up")])
        assert v._browser_idx == 0
        for _ in range(10):
            v._dispatch([KeyEvent("down")])
        assert v._browser_idx == len(v._browser_entries) - 1
    finally:
        os.close(fd)


def test_browser_tilde_jumps_home(tmp_path):
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        v._dispatch([KeyEvent("~")])
        assert v._browser_dir == os.path.expanduser("~")
    finally:
        os.close(fd)


def test_browser_keeps_the_cursor_on_screen_in_a_long_directory(tmp_path):
    for i in range(60):
        (tmp_path / f"f{i:02d}.xyz").write_text("1\nx\nHe 0 0 0\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        _top, _left, _width, height = v._file_browser_geometry()
        visible = height - 3
        assert visible < len(v._browser_entries)      # really does need scrolling
        for _ in range(40):
            v._dispatch([KeyEvent("down")])
        assert v._browser_scroll <= v._browser_idx < v._browser_scroll + visible
        assert v._file_browser_geometry()[0] >= 0
    finally:
        os.close(fd)


def test_browser_remembers_the_last_directory_within_a_session(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        v._browser_idx = 1
        v._dispatch([KeyEvent("enter")])          # into 'sub'
        v._dispatch([KeyEvent("escape")])
        v._dispatch([KeyEvent("A")])
        assert v._browser_dir == str(nested)
    finally:
        os.close(fd)


def test_browser_opening_does_not_spawn_any_subprocess(tmp_path, monkeypatch):
    """The whole point of replacing the native dialog: no osascript, so no
    macOS-only restriction and nothing that can block over SSH."""
    import subprocess

    def explode(*a, **k):                          # pragma: no cover - must not run
        raise AssertionError("the browser must not shell out")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(subprocess, "Popen", explode)
    (tmp_path / "x.xyz").write_text("1\nx\nHe 0 0 0\n")
    v, fd = _browser_viewer(tmp_path, str(tmp_path))
    try:
        v._dispatch([KeyEvent("A")])
        v._browser_idx = [e.name for e in v._browser_entries].index("x.xyz")
        v._dispatch([KeyEvent("enter")])
        assert len(v.structures) == 2
    finally:
        os.close(fd)
