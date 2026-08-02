"""Native file selection used by the interactive viewer."""
from __future__ import annotations

import subprocess
import sys
from typing import Optional


class FileDialogError(RuntimeError):
    """The native file picker could not be opened."""


def choose_structure_file() -> Optional[str]:
    """Open the macOS file picker and return its selected POSIX path.

    ``None`` means the user cancelled.  The subprocess is deliberately
    invoked without a shell: paths are returned over stdout, so spaces and
    shell metacharacters remain ordinary filename characters.
    """
    if sys.platform != "darwin":
        raise FileDialogError("the native add-file picker currently requires macOS")

    script = 'POSIX path of (choose file with prompt "Add a molecular structure")'
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise FileDialogError(f"could not open the macOS file picker: {exc}") from exc

    if result.returncode == 0:
        path = result.stdout.rstrip("\r\n")
        return path or None
    if "-128" in result.stderr or "User canceled" in result.stderr:
        return None
    detail = result.stderr.strip() or f"osascript exited with status {result.returncode}"
    raise FileDialogError(f"could not open the macOS file picker: {detail}")
