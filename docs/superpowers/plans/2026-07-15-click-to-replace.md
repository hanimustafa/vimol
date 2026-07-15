# Click-to-Replace Editor Gesture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clicking a heavy atom in append mode replaces it with the selected element/geometry (snapping terminal H's, topping up valency with H's) instead of growing a new fragment onto it.

**Architecture:** One new public operation `replace_atom` in `src/vimol/editor.py`; `grow_at_atom` keeps its signature but its heavy-atom branch dispatches to `replace_atom` instead of `_grow_onto` (which is deleted). No widget/viewer code changes — only their help/docs text.

**Tech Stack:** Python ≥3.8, numpy. Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-click-to-replace-design.md`

## Global Constraints

- Only dependency is `numpy>=1.20`; do not add others.
- Run tests as `python3 -m pytest` from the repo root (`python` is not on PATH).
- Tests live in `tests/test_editing.py`, which already imports `editor`, `templates`, `elements`, `Molecule`, `ensure_bonds`, `np`, `pytest` — reuse those imports.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

### Task 1: `replace_atom` editor operation

**Files:**
- Modify: `src/vimol/editor.py` (add helpers + `replace_atom`; nothing removed yet)
- Test: `tests/test_editing.py` (new section after the `# -- editor: grow` tests, around line 120)

**Interfaces:**
- Consumes: `templates.free_direction`, `templates.default_template`, existing private helpers `_neighbors`, `_bond_length`, `_cap`, `_reperceive` in `editor.py`.
- Produces: `replace_atom(mol: Molecule, idx: int, element: str = "C", template: Optional[AtomTemplate] = None) -> int` — returns `idx`. Task 2 calls this from `grow_at_atom`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_editing.py` after `test_lone_hydrogen_becomes_methane` (line ~120):

```python
# -- editor: replace --------------------------------------------------------
def test_replace_relabels_in_place_hypervalent():
    # CH4 carbon -> N (valence 3): coordination 4 >= 3, so relabel only
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4
    editor.replace_atom(mol, c, "N", templates.TEMPLATES[("N", 3)])
    assert mol.symbols[c] == "N"
    assert mol.n_atoms == 5                    # nothing added, nothing removed
    np.testing.assert_allclose(mol.positions[c], [0.0, 0.0, 0.0], atol=1e-9)


def test_replace_snaps_terminal_hydrogens():
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0])          # CH4 at C-H length
    editor.replace_atom(mol, c, "N", templates.TEMPLATES[("N", 3)])
    nh = elements.covalent_radius("N") + elements.covalent_radius("H")
    for h in range(1, 5):
        assert np.linalg.norm(mol.positions[h] - mol.positions[c]) == pytest.approx(nh, abs=1e-6)


def test_replace_keeps_heavy_neighbors_and_fills_valency():
    # C-C stub; replace atom 0 with O (default: bent, 2 bonds) -> one H added
    mol = Molecule(symbols=["C", "C"],
                   positions=np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0]]))
    ensure_bonds(mol)
    editor.replace_atom(mol, 0, "O")
    assert mol.symbols[0] == "O"
    np.testing.assert_allclose(mol.positions[1], [1.52, 0.0, 0.0], atol=1e-9)
    assert mol.symbols.count("H") == 1
    oh = elements.covalent_radius("O") + elements.covalent_radius("H")
    assert np.linalg.norm(mol.positions[2] - mol.positions[0]) == pytest.approx(oh, abs=1e-6)


def test_replace_fills_multiple_hydrogens_without_overlap():
    # water O -> C (valence 4): the 2 existing H snap, 2 more are added
    mol = Molecule()
    o = editor.birth_molecule(mol, [0.0, 0.0, 0.0], element="O")   # H2O
    editor.replace_atom(mol, o, "C")
    assert mol.formula() == "CH4"
    ch = elements.covalent_radius("C") + elements.covalent_radius("H")
    hs = [i for i, s in enumerate(mol.symbols) if s == "H"]
    for h in hs:
        assert np.linalg.norm(mol.positions[h] - mol.positions[o]) == pytest.approx(ch, abs=1e-6)
    # the naive free_direction loop would stack the last two H on top of
    # each other; no H pair may come close to colliding
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            assert np.linalg.norm(mol.positions[hs[i]] - mol.positions[hs[j]]) > 1.0


def test_replace_same_element_repairs_valency():
    # sp2 CH3 clicked with default (sp3) carbon selected -> gains one H
    mol = Molecule()
    c = editor.birth_molecule(mol, [0.0, 0.0, 0.0],
                              template=templates.TEMPLATES[("C", 3)])   # CH3
    editor.replace_atom(mol, c, "C")
    assert mol.formula() == "CH4"


def test_replace_lone_atom_caps_fully():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.replace_atom(mol, 0, "O")
    assert mol.formula() == "H2O"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_editing.py -k replace_ -v`
Expected: 6 FAILED / ERROR with `AttributeError: module 'vimol.editor' has no attribute 'replace_atom'`

- [ ] **Step 3: Implement `replace_atom`**

In `src/vimol/editor.py`, add after `_cap` (line ~56):

```python
def _fibonacci_sphere(n: int) -> np.ndarray:
    """*n* roughly-evenly distributed unit vectors."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def _fill_directions(existing, n: int) -> np.ndarray:
    """Pick *n* unit directions, one at a time, each far from all others.

    ``templates.free_direction`` handles the generic VSEPR-ish case, but it
    degenerates on symmetric arrangements (it can return a direction that is
    already occupied). When its answer lies within 60 degrees of an occupied
    direction we instead take the sphere point maximizing the minimum angle
    to everything placed so far.
    """
    dirs = [templates._normalize(d) for d in existing]
    out = []
    for _ in range(n):
        d = templates.free_direction(dirs)
        if dirs and max(float(np.dot(d, e)) for e in dirs) > 0.5:   # cos 60 deg
            pts = _fibonacci_sphere(256)
            worst = (pts @ np.array(dirs).T).max(axis=1)  # closest occupied, per point
            d = pts[int(np.argmin(worst))]
        out.append(d)
        dirs.append(d)
    return np.array(out)


def replace_atom(mol: Molecule, idx: int, element: str = "C",
                 template: Optional[AtomTemplate] = None) -> int:
    """Replace atom *idx* with *element*, keeping it anchored in place.

    The atom is relabeled where it stands. Terminal hydrogen neighbors are
    snapped to the new element-H bond length along their existing directions;
    heavy neighbors never move. If coordination is below the template's
    valence, hydrogens are added on free sites until it is met; excess
    coordination is left alone (hypervalency is the user's business).

    Returns *idx*.
    """
    tmpl = template or templates.default_template(element)
    element = elements.normalize_symbol(element)
    pos = mol.positions[idx].copy()
    neigh = _neighbors(mol, idx)
    mol.symbols[idx] = element
    for j in neigh:
        if mol.symbols[j] == "H" and _neighbors(mol, j) == [idx]:
            d = templates._normalize(mol.positions[j] - pos)
            mol.positions[j] = pos + d * _bond_length(element, "H")
    n_add = tmpl.valence - len(neigh)
    if n_add > 0:
        if not neigh:       # lone atom: full template, like birth_molecule
            _cap(mol, pos, tmpl.free_directions(), element, tmpl.cap)
        else:
            existing = [templates._normalize(mol.positions[j] - pos) for j in neigh]
            _cap(mol, pos, _fill_directions(existing, n_add), element, tmpl.cap)
    _reperceive(mol)
    return idx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_editing.py -k replace_ -v`
Expected: 6 PASSED

Then the full suite: `python3 -m pytest -q`
Expected: all pass (98 existing + 6 new), 1 skipped

- [ ] **Step 5: Commit**

```bash
git add src/vimol/editor.py tests/test_editing.py
git commit -m "Add replace_atom editor operation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Rewire the heavy-atom click gesture to replace

**Files:**
- Modify: `src/vimol/editor.py` (module docstring lines 1–16, `grow_at_atom` lines 122–136, delete `_grow_onto` lines 73–89)
- Modify: `tests/test_editing.py` (`test_grow_on_heavy_atom_attaches_methyl` line ~109, `test_widget_undo_reverts_edits` line ~174)

**Interfaces:**
- Consumes: `replace_atom(mol, idx, element, template) -> int` from Task 1.
- Produces: `grow_at_atom` keeps its exact signature `(mol, idx, element="C", template=None) -> int` — `widget.py:274` calls it unchanged. `_grow_onto` no longer exists.

- [ ] **Step 1: Update the two behavior tests to the new gesture**

In `tests/test_editing.py`, replace `test_grow_on_heavy_atom_attaches_methyl` (whole function, line ~109) with:

```python
def test_click_heavy_atom_replaces_it():
    # lone carbon, oxygen selected -> the C itself becomes a capped O (water)
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    editor.grow_at_atom(mol, 0, element="O")
    assert mol.formula() == "H2O"
    assert mol.symbols[0] == "O"          # replaced in place, not grown onto
```

In `test_widget_undo_reverts_edits` (line ~174), the second click currently lands on the methane's carbon; under replace semantics a same-element saturated click adds no atoms. Point it at empty space instead — replace the line

```python
    _click(w, 100, 100)                     # second click grows onto the methane
```

with

```python
    _click(w, 180, 20)                      # second click: empty corner, new methane
```

- [ ] **Step 2: Run the two tests to verify they fail**

Run: `python3 -m pytest tests/test_editing.py::test_click_heavy_atom_replaces_it -v`
Expected: FAIL — `grow_at_atom` still grows a new O *onto* the lone C (capped with one H), so the molecule is C + O + H (`"CHO"`) with `symbols[0] == "C"`, not `"H2O"`.

(`test_widget_undo_reverts_edits` still passes at this point — the coordinate change is forward-compatible with both behaviors.)

- [ ] **Step 3: Rewire `grow_at_atom`, delete `_grow_onto`, refresh docstrings**

In `src/vimol/editor.py`:

1. Delete the whole `_grow_onto` function (lines 73–89 in the pre-task file).
2. Replace `grow_at_atom` with:

```python
def grow_at_atom(mol: Molecule, idx: int, element: str = "C",
                 template: Optional[AtomTemplate] = None) -> int:
    """Edit the structure at atom *idx* with the selected *element*.

    * A hydrogen is *promoted*: it becomes the new element (moved to the right
      bond length) and its freed valences are capped -- "click an H, it turns
      into a carbon with three new hydrogens".
    * Any heavier atom is *replaced* in place -- see :func:`replace_atom`.

    Returns the index of the resulting central atom.
    """
    tmpl = template or templates.default_template(element)
    if mol.symbols[idx] == "H":
        return _promote_hydrogen(mol, idx, element, tmpl)
    return replace_atom(mol, idx, element, tmpl)
```

3. Replace the module docstring's gesture summary (the paragraph starting `The three public operations` through the sentence ending `three new hydrogens".`, lines 8–15) with:

```python
The public operations mirror the interactive gestures:

* :func:`birth_molecule` -- click empty space -> a fresh capped atom (methane).
* :func:`grow_at_atom`   -- click an atom     -> edit the structure there.
* :func:`replace_atom`   -- the heavy-atom half of ``grow_at_atom``.

``grow_at_atom`` splits on what was clicked: a hydrogen is *promoted* to the
building element and its freed valences capped ("click an H, it becomes a
carbon with three new hydrogens"); a heavier atom is *replaced* in place by
the building element, snapping its terminal hydrogens and topping up its
valency with new ones.
```

- [ ] **Step 4: Run the full suite to verify everything passes**

Run: `python3 -m pytest -q`
Expected: all pass, 1 skipped. If `test_widget_undo_reverts_edits` fails with `n2 > n1` being false, the (180, 20) click landed on an atom — move it further into the corner, e.g. `(190, 10)`.

- [ ] **Step 5: Commit**

```bash
git add src/vimol/editor.py tests/test_editing.py
git commit -m "Make heavy-atom clicks replace the atom instead of growing onto it

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Update user-facing help text and README

**Files:**
- Modify: `README.md:153` and `README.md:161-164`
- Modify: `src/vimol/viewer.py:69` (the `_HELP_EDIT` list)

**Interfaces:**
- Consumes: nothing from other tasks (text only).
- Produces: nothing consumed downstream.

- [ ] **Step 1: Update the README key table row**

Replace line 153:

```markdown
| Append / edit | `a` toggles; then click an atom → grow a carbon, click empty space → new methane |
```

with:

```markdown
| Append / edit | `a` toggles; then click an H → grow, click a heavy atom → replace it, click empty space → new methane |
```

- [ ] **Step 2: Update the README gesture description**

Replace lines 161–164:

```markdown
- **click an atom** → grow the structure there. Clicking a **hydrogen** promotes
  it to the element you've selected and caps its freed valences with hydrogens
  (so an H becomes a `–CH3` by default); clicking a heavier atom attaches a
  new group at a free site.
```

with:

```markdown
- **click a hydrogen** → grow there: the H is promoted to the element you've
  selected and its freed valences are capped with hydrogens (so an H becomes
  a `–CH3` by default).
- **click a heavier atom** → *replace* it with the selected element/geometry.
  Terminal hydrogens snap to the new bond length; an under-coordinated atom
  is topped up with hydrogens to meet the chosen valence, and excess bonds
  are left alone (hypervalent atoms are yours to make).
```

- [ ] **Step 3: Update the in-app help line**

In `src/vimol/viewer.py`, replace line 69:

```python
    "     click atom -> grow · click empty space -> new molecule",
```

with:

```python
    "     click H -> grow · heavy atom -> replace · empty space -> new molecule",
```

- [ ] **Step 4: Run the full suite (guards against accidental syntax damage)**

Run: `python3 -m pytest -q`
Expected: all pass, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add README.md src/vimol/viewer.py
git commit -m "Document the click-to-replace gesture in README and in-app help

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
