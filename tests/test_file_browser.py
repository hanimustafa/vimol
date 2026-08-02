import os

from vimol.file_browser import Entry, list_directory


def _touch(path, text="1\nx\nHe 0 0 0\n"):
    path.write_text(text)


def test_listing_puts_parent_first_then_directories_then_files(tmp_path):
    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha").mkdir()
    _touch(tmp_path / "b.xyz")
    _touch(tmp_path / "a.pdb")

    entries = list_directory(str(tmp_path))
    assert [e.name for e in entries] == ["..", "alpha", "zeta", "a.pdb", "b.xyz"]
    assert [e.is_dir for e in entries] == [True, True, True, False, False]


def test_listing_hides_formats_vimol_cannot_open(tmp_path):
    _touch(tmp_path / "keep.xyz")
    _touch(tmp_path / "job.out")
    _touch(tmp_path / "wave.gbw")
    _touch(tmp_path / "keep.pdb")

    names = [e.name for e in list_directory(str(tmp_path))]
    assert names == ["..", "keep.pdb", "keep.xyz"]


def test_listing_matches_extensions_case_insensitively(tmp_path):
    _touch(tmp_path / "SHOUT.XYZ")
    _touch(tmp_path / "Mixed.Pdb")

    names = [e.name for e in list_directory(str(tmp_path))]
    assert names == ["..", "Mixed.Pdb", "SHOUT.XYZ"]


def test_listing_hides_dotfiles_and_dot_directories(tmp_path):
    _touch(tmp_path / ".hidden.xyz")
    (tmp_path / ".git").mkdir()
    _touch(tmp_path / "shown.xyz")

    names = [e.name for e in list_directory(str(tmp_path))]
    assert names == ["..", "shown.xyz"]


def test_listing_sorts_case_insensitively_within_each_group(tmp_path):
    for name in ("Beta", "alpha"):
        (tmp_path / name).mkdir()
    for name in ("B.xyz", "a.xyz"):
        _touch(tmp_path / name)

    names = [e.name for e in list_directory(str(tmp_path))]
    assert names == ["..", "alpha", "Beta", "a.xyz", "B.xyz"]


def test_root_directory_offers_no_parent_entry():
    assert [e.name for e in list_directory("/")][:1] != [".."]


def test_unreadable_directory_lists_as_empty_rather_than_raising(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    _touch(locked / "inside.xyz")
    os.chmod(locked, 0o000)
    try:
        # A permission error must not escape into the event loop.
        assert list_directory(str(locked)) == [
            Entry("..", os.path.dirname(str(locked)), True)]
    finally:
        os.chmod(locked, 0o755)


def test_missing_directory_lists_as_empty_rather_than_raising(tmp_path):
    gone = tmp_path / "not-there"
    assert [e.name for e in list_directory(str(gone))] == [".."]


def test_entries_carry_a_usable_absolute_path(tmp_path):
    _touch(tmp_path / "mol.xyz")
    entry = [e for e in list_directory(str(tmp_path)) if e.name == "mol.xyz"][0]
    assert entry.path == str(tmp_path / "mol.xyz")
    assert os.path.isfile(entry.path)
