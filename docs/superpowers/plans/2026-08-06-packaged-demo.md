# Ship the C60 Demo Inside the Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vimol` with no file argument open the bundled C60 demo on a real `pip install`, not just from a checkout.

**Architecture:** `examples/build_examples.py` (the existing deterministic-geometry generator) writes a second copy of `c60.xyz` into `src/vimol/data/`, which `setuptools` ships as package-data. `_default_demo_path()` in `src/vimol/app.py` prefers that packaged copy and falls back to the checkout's `examples/c60.xyz`.

**Tech Stack:** Python 3.8+, setuptools (`[tool.setuptools.package-data]`), pytest, the existing `scripts/build-pypi-artifacts` / `scripts/lib.sh` release tooling.

## Global Constraints

- `examples/c60.xyz` remains the canonical source referenced by `examples/inset.py`, `examples/embed_demo.py`, and the README — do not change how those consume it.
- No `importlib.resources` migration — resolve the packaged path the same way `py.typed` is implicitly handled: a plain join next to `__file__`.
- Only C60 gets a packaged copy — no other `examples/*.xyz` file is a zero-arg default.
- Design doc: `docs/superpowers/specs/2026-08-06-packaged-demo-design.md`.

---

### Task 1: Generate the packaged demo copy

**Files:**
- Modify: `examples/build_examples.py:1-14` (module-level `HERE`, `write_xyz`), `examples/build_examples.py:47-68` (`buckyball`)
- Create (via running the script, not by hand): `src/vimol/data/c60.xyz`

**Interfaces:**
- Produces: `src/vimol/data/c60.xyz`, an XYZ file with the same 60-atom buckminsterfullerene geometry as `examples/c60.xyz` — later tasks (2, 3) depend on this file existing on disk.

- [ ] **Step 1: Add a second output directory and let `write_xyz` fan out to multiple dirs**

Replace lines 1-14 of `examples/build_examples.py`:

```python
"""Generate example molecule files with exact geometry (no external data)."""
import math
import os

HERE = os.path.dirname(__file__)
PKG_DATA = os.path.join(HERE, "..", "src", "vimol", "data")


def write_xyz(name, comment, atoms, dirs=(HERE,)):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(f"{len(atoms)}\n{comment}\n")
            for sym, (x, y, z) in atoms:
                f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
        print("wrote", path, len(atoms), "atoms")
```

This is backward compatible: every existing call site (`water()`, `methane()`, `benzene()`, `threonine()`) omits `dirs` and keeps writing only to `HERE`, same as today.

- [ ] **Step 2: Make `buckyball()` write to both directories**

In `examples/build_examples.py`, change the last line of `buckyball()` (currently line 68):

```python
    write_xyz("c60.xyz", "buckminsterfullerene C60", atoms, dirs=(HERE, PKG_DATA))
```

- [ ] **Step 3: Run the generator and verify only the expected files changed**

```bash
python3 examples/build_examples.py
```

Expected output includes two "wrote ... c60.xyz 60 atoms" lines — one for `examples/c60.xyz`, one for `.../src/vimol/data/c60.xyz` — plus the usual lines for `water.xyz`, `methane.xyz`, `benzene.xyz`, `threonine.xyz`.

```bash
git status --porcelain examples/ src/vimol/data/
```

Expected: `examples/*.xyz` show **no changes** (the geometry is deterministic, so re-running the generator reproduces byte-identical files — if any existing file shows a diff, stop and investigate before continuing); `src/vimol/data/c60.xyz` appears as a new untracked file; `examples/build_examples.py` shows the edit from steps 1-2.

- [ ] **Step 4: Confirm the new file matches the checkout's copy exactly**

```bash
diff examples/c60.xyz src/vimol/data/c60.xyz && echo "identical"
```

Expected: `identical`

- [ ] **Step 5: Commit**

```bash
git add examples/build_examples.py examples/c60.xyz src/vimol/data/c60.xyz
git commit -m "Generate a packaged copy of the C60 demo alongside the examples/ one"
```

(If step 3 showed zero diff for `examples/c60.xyz`, `git add` on it is a no-op — that's expected, include it anyway for clarity.)

---

### Task 2: Prefer the packaged demo path in `_default_demo_path()`

**Files:**
- Modify: `src/vimol/app.py:1-7` (module docstring), `src/vimol/app.py:162-171` (`_default_demo_path`)
- Test: `tests/test_editing.py` (new test near line 3167, alongside the existing `test_default_demo_path_resolves_to_bundled_c60`)

**Interfaces:**
- Consumes: `src/vimol/data/c60.xyz` (produced by Task 1) must exist on disk in this checkout for the new test to exercise the real packaged-path branch.
- Produces: `_default_demo_path() -> Optional[str]`, unchanged signature — still called from `main()` at `src/vimol/app.py:232` with no changes needed there.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_editing.py`, directly after `test_default_demo_path_resolves_to_bundled_c60` (after line 3170):

```python
def test_default_demo_path_prefers_packaged_copy_when_examples_dir_absent(monkeypatch):
    # Simulates a real `pip install`: no examples/ checkout directory exists,
    # only the packaged copy shipped inside the vimol package.
    packaged = os.path.join(os.path.dirname(vimol_app.__file__), "data", "c60.xyz")
    real_exists = os.path.exists

    def fake_exists(path):
        if os.path.normpath(path).endswith(os.path.join("examples", "c60.xyz")):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    path = vimol_app._default_demo_path()
    assert path == packaged
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_editing.py::test_default_demo_path_prefers_packaged_copy_when_examples_dir_absent -v
```

Expected: FAIL — current `_default_demo_path()` only ever checks the checkout path, which `fake_exists` hides, so it returns `None` instead of `packaged`.

- [ ] **Step 3: Implement the preferred-packaged-path resolution**

Replace `src/vimol/app.py:162-171`:

```python
def _default_demo_path() -> Optional[str]:
    """Path to the bundled C60 demo, for `vimol` with no file argument.

    Prefers the copy shipped inside the package (src/vimol/data/c60.xyz,
    included as package-data so it survives `pip install`); falls back to
    the checkout's examples/c60.xyz for a source tree where that data file
    hasn't been generated yet (see examples/build_examples.py).
    """
    packaged = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "c60.xyz")
    if os.path.exists(packaged):
        return packaged
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    checkout = os.path.join(root, "examples", "c60.xyz")
    return checkout if os.path.exists(checkout) else None
```

- [ ] **Step 4: Update the module docstring's stale "(checkout only)" note**

`src/vimol/app.py:1-7` currently reads:

```python
"""Command-line driver for vimol.

    vimol                           # opens the bundled C60 demo (checkout only)
    vimol file.pdb                 # interactive viewer (opens editable: a=append)
    vimol file.xyz --spin          # autospinning
    vimol a.xyz b.pdb              # load both, auto-overlaid for comparison
"""
```

Change the first example line to:

```python
    vimol                           # opens the bundled C60 demo
```

- [ ] **Step 5: Run the new test and the two existing demo-path tests to verify they pass**

```bash
pytest tests/test_editing.py -k "default_demo_path or demo_default" -v
```

Expected: 3 PASS — `test_default_demo_path_resolves_to_bundled_c60`, `test_default_demo_path_prefers_packaged_copy_when_examples_dir_absent`, `test_cli_with_no_file_uses_demo_default_not_help`.

- [ ] **Step 6: Run the full test file to check for regressions**

```bash
pytest tests/test_editing.py -q
```

Expected: all tests PASS (same pass count as before this task, plus the one new test).

- [ ] **Step 7: Commit**

```bash
git add src/vimol/app.py tests/test_editing.py
git commit -m "Prefer the packaged C60 demo over the examples/ checkout path"
```

---

### Task 3: Ship `data/c60.xyz` in the wheel and sdist

**Files:**
- Modify: `pyproject.toml:59-60`
- Modify: `scripts/build-pypi-artifacts:58-60`

**Interfaces:**
- Consumes: `src/vimol/data/c60.xyz` (Task 1), `$ver` (already set earlier in `scripts/build-pypi-artifacts` by `ver="$(read_version)"`), `$py` (the release venv's Python, already set by `ensure_release_venv`).

- [ ] **Step 1: Add `data/c60.xyz` to package-data**

Replace `pyproject.toml:59-60`:

```toml
[tool.setuptools.package-data]
vimol = ["py.typed", "data/c60.xyz"]
```

- [ ] **Step 2: Add a wheel-content assertion to the build script**

In `scripts/build-pypi-artifacts`, insert this block between the existing line 59 (`"$tw" check dist/*`) and line 61 (`echo "==> smoke test: ..."`):

```bash

echo "==> checking data/c60.xyz landed in the wheel"
"$py" -c "
import sys, zipfile
with zipfile.ZipFile('dist/vimol-$ver-py3-none-any.whl') as zf:
    sys.exit(0 if 'vimol/data/c60.xyz' in zf.namelist() else 1)
" || { echo "!!  vimol/data/c60.xyz is missing from the wheel -- check [tool.setuptools.package-data] in pyproject.toml" >&2; exit 1; }
```

- [ ] **Step 3: Run the real build script end-to-end to verify**

```bash
scripts/build-pypi-artifacts
```

Expected: build succeeds, prints `==> checking data/c60.xyz landed in the wheel` with no error after it, then continues through the existing smoke test (`ok: vimol <version>`) and lists `dist/` at the end. (`.venv-release/` already exists in this checkout, so this doesn't need network access.)

- [ ] **Step 4: Prove the check actually catches a missing file (negative test)**

Temporarily break the config to confirm the assertion isn't a no-op:

```bash
git stash push -- pyproject.toml
scripts/build-pypi-artifacts
```

Expected: FAILS with `!!  vimol/data/c60.xyz is missing from the wheel -- check [tool.setuptools.package-data] in pyproject.toml`, exit code 1.

Restore the real config:

```bash
git stash pop
```

- [ ] **Step 5: Clean up build artifacts**

```bash
rm -rf dist build src/vimol.egg-info
```

(`scripts/build-pypi-artifacts` itself does this at the start of every run, so leaving them around is harmless, but there's no reason to leave generated `dist/` output sitting in a clean working tree.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml scripts/build-pypi-artifacts
git commit -m "Ship the packaged C60 demo in the wheel and sdist"
```

---

## Self-Review Notes

- **Spec coverage:** design doc's five change points — `build_examples.py` (Task 1), `pyproject.toml` (Task 3), `app.py` `_default_demo_path()` (Task 2), `scripts/build-pypi-artifacts` (Task 3), new test simulating "installed wheel, no examples/" (Task 2) — each has a task. The design doc's "out of scope" items (other example files, `inset.py`/`embed_demo.py`/README, `importlib.resources`) are untouched by every task above.
- **Ordering:** Task 1 must run before Task 2's Step 1-2 (the new test needs the real packaged file on disk to exercise a genuine pass) and before Task 3 (nothing to package otherwise).
- **Type/signature consistency:** `_default_demo_path() -> Optional[str]` signature unchanged; `write_xyz`'s new `dirs` parameter defaults to the old single-directory behavior so no other call site needs touching.
