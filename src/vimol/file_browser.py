"""Directory listings for the viewer's file browser.

Pure filesystem logic, deliberately free of terminal or viewer concerns: the
viewer owns drawing and keys, this owns what is in a directory and in what
order. That split is what lets the listing rules be tested without a pty.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from .parsers import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class Entry:
    name: str
    path: str
    is_dir: bool


def list_directory(path: str) -> List[Entry]:
    """Directories and openable files in *path*, in display order.

    ``..`` leads, then directories, then files vimol can actually parse --
    a calculation directory is mostly ``.out``/``.gbw``/``.tmp``, and listing
    those would bury the handful of structures among hundreds of dead ends.
    Dotfiles are omitted. Each group sorts case-insensitively.

    A directory that cannot be read lists as empty rather than raising: this
    runs inside the event loop, where an unreadable directory is something to
    show as empty, not a reason to take the viewer down.
    """
    directory = os.path.abspath(path)
    entries: List[Entry] = []
    parent = os.path.dirname(directory)
    if parent != directory:                      # anything but the filesystem root
        entries.append(Entry("..", parent, True))

    directories: List[Entry] = []
    files: List[Entry] = []
    try:
        with os.scandir(directory) as scan:
            for item in scan:
                if item.name.startswith("."):
                    continue
                try:
                    is_dir = item.is_dir()
                except OSError:                  # a broken symlink, say
                    continue
                if is_dir:
                    directories.append(Entry(item.name, item.path, True))
                elif os.path.splitext(item.name)[1].lower() in SUPPORTED_EXTENSIONS:
                    files.append(Entry(item.name, item.path, False))
    except OSError:
        return entries

    key = lambda entry: entry.name.lower()       # noqa: E731 - one expression
    directories.sort(key=key)
    files.sort(key=key)
    return entries + directories + files


__all__ = ["Entry", "list_directory"]
