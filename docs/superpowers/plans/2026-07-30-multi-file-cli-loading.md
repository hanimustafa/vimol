# Multi-file CLI loading with auto-overlay (VIM-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `vimol` accept 2+ files on the command line, load them all into one interactive session, and automatically overlay one structure per file (the first model of each) so they're visually comparable with no extra keypresses.

**Architecture:** A new pure helper, `_build_structure_set()`, loads a list of paths into one `StructureSet` — cross-file label disambiguation, per-file `ensure_bonds`, per-file warn-and-skip on failure, marking each file's first model, and setting `overlay = True`. `app.py:main()` branches on `len(files)`: exactly one file keeps today's code path completely unchanged (same `Viewer(frames=...)` call, same error codes); 2+ files calls the new helper and constructs `Viewer(structures=...)` instead. Both branches converge on one shared terminal/Kitty-support gate, factored out of the existing inline check so it isn't duplicated.

**Tech Stack:** Python 3.8+, argparse, pytest (existing test suite in `tests/test_editing.py`).

## Global Constraints

- `file` positional argument becomes `nargs="*"` (was `nargs="?"`).
- Zero files → today's bundled-C60 demo fallback, unchanged.
- Exactly one file → the existing single-file code path is untouched: missing file is a hard error (stderr `error: no such file: <path>`, exit 2); a file that fails to parse is a hard error (stderr `error: failed to parse <path>: <err>`, exit 3); `--frame` unchanged.
- 2+ files: a file that's missing, fails to parse, or parses to zero molecules prints `warning: skipping <path>: <reason>` to stderr and loading continues with the rest. If every file fails: stderr `error: no molecules parsed`, exit 3.
- Cross-file labels: each file's label stem is its basename; if two files share a basename, later ones get `~2`, `~3`, … appended to the stem. Within a file, multiple models keep the existing `#1`, `#2`, … suffixing, applied after the stem is disambiguated.
- For 2+ files: `overlay = True`, and only the first-loaded model of each file has `marked = True` — every other model stays loaded and reachable (browse/list/`opt+click`) but starts unmarked.
- `--frame K` clamps into the *whole* combined set in load order, default `0` (first model of the first file) — a superset of today's single-file-only meaning.
- `--render`, `--kitty`, `--info`, `--size`, `--supersample` were already removed from the CLI in a prior commit (`ef3406f`) — nothing in this plan touches them.
- No `--active LABEL` flag; no change to how a single multi-model file behaves when opened alone (`overlay` stays `False`, nothing gets auto-marked).
- Full spec: `docs/superpowers/specs/2026-07-30-multi-file-cli-design.md`.

---

### Task 1: `_build_structure_set()` — load 2+ files into one `StructureSet`

**Files:**
- Modify: `src/vimol/app.py` (add imports, add `_build_structure_set`)
- Test: `tests/test_editing.py` (new tests near the existing `# -- CLI: no-arg default opens the bundled demo --` section, around line 3055)

**Interfaces:**
- Consumes: `vimol.structures.StructureSet` (`.append(molecule, label, path) -> Structure`, `.labels: List[str]`, `.overlay: bool`, iteration yields `Structure` with `.molecule`, `.label`, `.path`, `.marked`); `vimol.parsers.load_all(path) -> List[Molecule]`; `vimol.bonds.ensure_bonds(mol, tolerance=...)`.
- Produces: `_build_structure_set(paths: List[str], no_bonds: bool, tolerance: float) -> StructureSet`, used by Task 2's `main()`. On return, the set already has `overlay = True` and each file's first model `marked = True` — Task 2 does not need to set either.

- [ ] **Step 1: Write the failing tests**

Add this block to `tests/test_editing.py`, directly after the existing `test_cli_with_no_file_uses_demo_default_not_help` test (around line 3069):

```python
# -- CLI: multi-file loading (VIM-1) -----------------------------------
def test_build_structure_set_loads_each_file_in_order(tmp_path):
    a = tmp_path / "a.xyz"
    a.write_text("2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    b = tmp_path / "b.xyz"
    b.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(a), str(b)], no_bonds=False, tolerance=0.45)
    assert sset.labels == ["a.xyz", "b.xyz"]
    assert [e.path for e in sset] == [str(a), str(b)]


def test_build_structure_set_sets_overlay_true(tmp_path):
    a = tmp_path / "a.xyz"
    a.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    b = tmp_path / "b.xyz"
    b.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(a), str(b)], no_bonds=False, tolerance=0.45)
    assert sset.overlay is True


def test_build_structure_set_labels_multi_model_file_with_hash_suffix(tmp_path):
    traj = tmp_path / "traj.xyz"
    traj.write_text(
        "1\nf1\nH 0.0 0.0 0.0\n"
        "1\nf2\nH 0.0 0.0 1.0\n"
        "1\nf3\nH 0.0 0.0 2.0\n"
    )
    single = tmp_path / "single.xyz"
    single.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(traj), str(single)], no_bonds=False, tolerance=0.45)
    assert sset.labels == ["traj.xyz#1", "traj.xyz#2", "traj.xyz#3", "single.xyz"]


def test_build_structure_set_marks_only_first_model_of_each_file(tmp_path):
    traj = tmp_path / "traj.xyz"
    traj.write_text("1\nf1\nH 0.0 0.0 0.0\n1\nf2\nH 0.0 0.0 1.0\n")
    single = tmp_path / "single.xyz"
    single.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(traj), str(single)], no_bonds=False, tolerance=0.45)
    assert [e.marked for e in sset] == [True, False, True]


def test_build_structure_set_disambiguates_duplicate_basenames(tmp_path):
    d1 = tmp_path / "d1"
    d1.mkdir()
    d2 = tmp_path / "d2"
    d2.mkdir()
    p1 = d1 / "mol.xyz"
    p1.write_text("1\na\nH 0.0 0.0 0.0\n")
    p2 = d2 / "mol.xyz"
    p2.write_text("1\nb\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(p1), str(p2)], no_bonds=False, tolerance=0.45)
    assert sset.labels == ["mol.xyz", "mol.xyz~2"]


def test_build_structure_set_skips_missing_file_with_warning(tmp_path, capsys):
    missing = tmp_path / "nope.xyz"
    ok = tmp_path / "ok.xyz"
    ok.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(missing), str(ok)], no_bonds=False, tolerance=0.45)
    assert sset.labels == ["ok.xyz"]
    err = capsys.readouterr().err
    assert f"warning: skipping {missing}: no such file" in err


def test_build_structure_set_skips_unparseable_file_with_warning(tmp_path, capsys):
    bad = tmp_path / "bad.foo"
    bad.write_text("nonsense")
    ok = tmp_path / "ok.xyz"
    ok.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(bad), str(ok)], no_bonds=False, tolerance=0.45)
    assert sset.labels == ["ok.xyz"]
    err = capsys.readouterr().err
    assert f"warning: skipping {bad}:" in err


def test_build_structure_set_skips_file_with_zero_molecules(tmp_path, capsys):
    empty = tmp_path / "empty.xyz"
    empty.write_text("   \n\n")
    ok = tmp_path / "ok.xyz"
    ok.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    sset = vimol_app._build_structure_set([str(empty), str(ok)], no_bonds=False, tolerance=0.45)
    assert sset.labels == ["ok.xyz"]
    err = capsys.readouterr().err
    assert f"warning: skipping {empty}: no molecules parsed" in err


def test_build_structure_set_all_files_fail_returns_empty_set(tmp_path):
    m1 = tmp_path / "m1.xyz"
    m2 = tmp_path / "m2.xyz"
    sset = vimol_app._build_structure_set([str(m1), str(m2)], no_bonds=False, tolerance=0.45)
    assert len(sset) == 0


def test_build_structure_set_respects_no_bonds(tmp_path):
    a = tmp_path / "a.xyz"
    a.write_text("2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    sset = vimol_app._build_structure_set([str(a)], no_bonds=True, tolerance=0.45)
    assert len(sset[0].molecule.bonds) == 0


def test_build_structure_set_perceives_bonds_by_default(tmp_path):
    a = tmp_path / "a.xyz"
    a.write_text("2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    sset = vimol_app._build_structure_set([str(a)], no_bonds=False, tolerance=0.45)
    assert len(sset[0].molecule.bonds) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source .venv-bench/bin/activate
python -m pytest tests/test_editing.py -k build_structure_set -v
```

Expected: every test errors with `AttributeError: module 'vimol.app' has no attribute '_build_structure_set'`.

- [ ] **Step 3: Implement `_build_structure_set`**

In `src/vimol/app.py`, change the import block at the top from:

```python
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .parsers import load_all, SUPPORTED_EXTENSIONS
from .bonds import ensure_bonds
from .render import Style
from . import kitty
```

to:

```python
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple

from .parsers import load_all, SUPPORTED_EXTENSIONS
from .bonds import ensure_bonds
from .render import Style
from .structures import StructureSet
from . import kitty
```

Then add this function right after `_default_demo_path` (i.e. immediately before `def main(`):

```python
def _build_structure_set(paths: List[str], no_bonds: bool, tolerance: float) -> StructureSet:
    """Load 2+ files into one StructureSet, in load order (VIM-1). A file
    that's missing, fails to parse, or parses to zero molecules is skipped
    with a warning instead of aborting the whole session. Sets ``overlay =
    True`` and marks each file's first model, so the caller gets a
    ready-to-render auto-overlaid set with no further setup -- see
    docs/superpowers/specs/2026-07-30-multi-file-cli-design.md.
    """
    sset = StructureSet()
    basenames = [os.path.basename(p) for p in paths]
    dupe_counts = Counter(basenames)
    seen: Counter = Counter()
    for path, base in zip(paths, basenames):
        if not os.path.exists(path):
            print(f"warning: skipping {path}: no such file", file=sys.stderr)
            continue
        try:
            mols = load_all(path)
        except Exception as e:  # noqa: BLE001
            print(f"warning: skipping {path}: {e}", file=sys.stderr)
            continue
        if not mols:
            print(f"warning: skipping {path}: no molecules parsed", file=sys.stderr)
            continue

        if dupe_counts[base] > 1:
            seen[base] += 1
            stem = base if seen[base] == 1 else f"{base}~{seen[base]}"
        else:
            stem = base
        multi = len(mols) > 1
        for i, m in enumerate(mols):
            if not no_bonds:
                ensure_bonds(m, tolerance=tolerance)
            label = f"{stem}#{i + 1}" if multi else stem
            entry = sset.append(m, label=label, path=path)
            if i == 0:
                entry.marked = True
    sset.overlay = True
    return sset
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_editing.py -k build_structure_set -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vimol/app.py tests/test_editing.py
git commit -m "Add _build_structure_set for multi-file CLI loading (VIM-1)"
```

---

### Task 2: Wire multi-file loading into `main()`

**Files:**
- Modify: `src/vimol/app.py` (`make_parser`, module docstring, `main`; add `_check_kitty_terminal`)
- Modify: `README.md` (multi-file example)
- Test: `tests/test_editing.py` (new tests appended after Task 1's block)

**Interfaces:**
- Consumes: `_build_structure_set` from Task 1; `vimol.viewer.Viewer.__init__(molecule, frames=None, structures=None, style=None, autospin=False, backend="auto", source_path=None, editable=False, probe=None)` (existing signature, unchanged — `structures=` takes priority over `frames=`/`molecule` when not `None`).
- Produces: `main(argv) -> int`, unchanged public signature; `args.file` is now always a `list` (was previously a single `str | None`).

- [ ] **Step 1: Write the failing tests**

Append this block to `tests/test_editing.py`, after Task 1's tests:

```python
def test_make_parser_accepts_multiple_files():
    args = vimol_app.make_parser().parse_args(["a.xyz", "b.xyz"])
    assert args.file == ["a.xyz", "b.xyz"]


def test_make_parser_with_no_file_gives_empty_list():
    args = vimol_app.make_parser().parse_args([])
    assert args.file == []


def test_cli_single_file_reaches_interactive_gate_unchanged(tmp_path):
    a = tmp_path / "a.xyz"
    a.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    rc = vimol_app.main([str(a)])
    assert rc == 4


def test_cli_single_missing_file_is_still_a_hard_error(tmp_path, capsys):
    missing = tmp_path / "nope.xyz"
    rc = vimol_app.main([str(missing)])
    assert rc == 2
    assert f"error: no such file: {missing}" in capsys.readouterr().err


def test_cli_multiple_files_reaches_interactive_gate(tmp_path):
    a = tmp_path / "a.xyz"
    a.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    b = tmp_path / "b.xyz"
    b.write_text("1\nne\nNe 0.0 0.0 0.0\n")
    rc = vimol_app.main([str(a), str(b)])
    assert rc == 4


def test_cli_multiple_files_one_missing_still_reaches_interactive_gate(tmp_path, capsys):
    a = tmp_path / "a.xyz"
    a.write_text("1\nhe\nHe 0.0 0.0 0.0\n")
    missing = tmp_path / "missing.xyz"
    rc = vimol_app.main([str(a), str(missing)])
    assert rc == 4
    assert f"warning: skipping {missing}: no such file" in capsys.readouterr().err


def test_cli_multiple_files_all_missing_errors_out(tmp_path, capsys):
    m1 = tmp_path / "m1.xyz"
    m2 = tmp_path / "m2.xyz"
    rc = vimol_app.main([str(m1), str(m2)])
    assert rc == 3
    assert "error: no molecules parsed" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_editing.py -k "make_parser or cli_single or cli_multiple" -v
```

Expected: the two `make_parser` tests fail on `assert args.file == ["a.xyz", "b.xyz"]` (currently a bare string / `None` because `nargs="?"`); the `cli_multiple_files_*` tests fail because `main()` still treats `args.file` as a single path and mishandles a list.

- [ ] **Step 3: Wire it up**

In `src/vimol/app.py`, update the module docstring:

```python
"""Command-line driver for vimol.

    vimol                           # opens the bundled C60 demo (checkout only)
    vimol file.pdb                 # interactive viewer (opens editable: a=append)
    vimol file.xyz --spin          # autospinning
    vimol a.xyz b.pdb              # load both, auto-overlaid for comparison
"""
```

In `make_parser()`, change:

```python
    p.add_argument("file", nargs="?", help="structure file (xyz/pdb/mol/sdf)")
```

to:

```python
    p.add_argument("file", nargs="*", help="one or more structure files (xyz/pdb/mol/sdf)")
```

Add this helper right after `_probe_terminal_raw` (before `_default_demo_path`):

```python
def _check_kitty_terminal() -> Tuple[Optional["kitty.TerminalProbe"], int]:
    """Guard for the interactive path: stdout must be a real terminal, and it
    must speak the Kitty graphics protocol (checked via env heuristics, then
    a raw-mode probe if those come up empty). Returns ``(probe, 0)`` to
    proceed -- reusing the probe avoids a second round trip -- or
    ``(None, exit_code)`` to abort.
    """
    if not sys.stdout.isatty():
        print("error: interactive mode needs a terminal", file=sys.stderr)
        return None, 4
    if kitty.supports_kitty():
        return None, 0
    # The environment says nothing (common over SSH) -- ask the terminal
    # itself. Only an answered probe that lacks graphics support, or no
    # terminal to ask, refuses; a confirmed terminal proceeds normally.
    probe = _probe_terminal_raw()
    if probe is None or probe.graphics is not True:
        print("warning: this terminal does not appear to support the Kitty "
              "graphics protocol.", file=sys.stderr)
        print("         Set VIMOL_FORCE_KITTY=1 to try anyway.", file=sys.stderr)
        return None, 5
    return probe, 0
```

Replace the entire body of `main()` from `if not args.file:` down to the end with:

```python
    files = args.file
    if not files:
        demo = _default_demo_path()
        if not demo:
            make_parser().print_help()
            return 1
        files = [demo]

    if len(files) == 1:
        path = files[0]
        if not os.path.exists(path):
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2

        try:
            mols = load_all(path)
        except Exception as e:  # noqa: BLE001
            print(f"error: failed to parse {path}: {e}", file=sys.stderr)
            return 3
        if not mols:
            print("error: no molecules parsed", file=sys.stderr)
            return 3

        for m in mols:
            if not args.no_bonds:
                ensure_bonds(m, tolerance=args.bond_tolerance)

        idx = max(0, min(args.frame, len(mols) - 1))
        mol = mols[idx]
        frames, structures, source_path = mols, None, path
    else:
        structures = _build_structure_set(files, args.no_bonds, args.bond_tolerance)
        if len(structures) == 0:
            print("error: no molecules parsed", file=sys.stderr)
            return 3

        idx = max(0, min(args.frame, len(structures) - 1))
        mol = structures[0].molecule
        frames, source_path = None, None

    style = build_style(args)

    # -- interactive viewer ----------------------------------------------
    probe, rc = _check_kitty_terminal()
    if rc:
        return rc

    # interactive defaults to a terminal-matching transparent background
    if not args.opaque and args.background is None:
        style.transparent = True

    from .viewer import Viewer
    viewer = Viewer(mol, frames=frames, structures=structures, style=style,
                    autospin=args.spin, backend=args.backend,
                    source_path=source_path, editable=True,
                    probe=probe)   # reuse the detection probe: no second round trip
    viewer.frame_index = idx
    # apply initial rotation
    viewer.widget.scene.camera.orbit(args.rotate[0], args.rotate[1])
    viewer.run()
    return 0
```

The lines above `if not args.file:` (the `--version`/`--list-formats` handling) are unchanged.

In `README.md`, in the "In the terminal" section, change:

```bash
vimol traj.xyz --spin --style spacefill   # spin a trajectory, space-filling
```

to:

```bash
vimol traj.xyz --spin --style spacefill   # spin a trajectory, space-filling
vimol a.xyz b.pdb                         # load both, auto-overlaid for comparison
```

And add one sentence after the existing "Editing is on by default..." paragraph, before the code block:

```markdown
Pass more than one file (`vimol a.xyz b.pdb`) and they all load into one
session, auto-overlaid — the first structure of each file shown together,
tinted to tell them apart — so you can compare shapes immediately; `opt+click`
a row in the structure list to add another loaded frame into the overlay or
drop one out.
```

- [ ] **Step 4: Run the tests to verify they pass, then run the full suite**

```bash
python -m pytest tests/test_editing.py -k "make_parser or cli_single or cli_multiple or build_structure_set" -v
python -m pytest tests/ -q
```

Expected: the targeted run shows all PASS; the full run shows the same count as before this plan started (`456 passed, 1 xfailed`) plus this plan's new tests, with none failing.

- [ ] **Step 5: Commit**

```bash
git add src/vimol/app.py README.md tests/test_editing.py
git commit -m "Wire multi-file CLI loading with auto-overlay into main() (VIM-1)"
```

---

## After both tasks land

Move VIM-1 to **Requires Approval** in Linear (per the project's issue workflow: In Progress while developing, Requires Approval before design/code review and commit) and note in the issue that the scope shipped is interactive-only, with `--render`/`--kitty`/`--info` removed rather than extended — matching `docs/superpowers/specs/2026-07-30-multi-file-cli-design.md`.
