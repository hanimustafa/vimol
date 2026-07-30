# Multi-structure data model

**Issue:** VIM-8 · **Date:** 2026-07-28 · **Status:** Revision 2 (all open questions resolved)
**Blocks:** VIM-9 (single-file sets), VIM-1 (CLI), VIM-2 (switch), VIM-3 (overlay),
VIM-4 (align), VIM-6 (measure), VIM-7 (API) · **Related:** VIM-10 (name-based selection)

## Summary

Add one container — `StructureSet` — that holds N molecules with stable
identity, per-structure transform/tint/visibility/mark, and an active index.
Everything downstream flattens it into a single throwaway `Molecule` for
rendering, so **`camera.py`, `arrows.py`, `bonds.py` and `editor.py` need no
changes at all**, and the two renderers need exactly one additive change
between them (a per-atom flat-shading flag, §4). Alignment writes a rigid
transform into the container and never touches source coordinates.

New modules: `src/vimol/structures.py`, `src/vimol/align.py`,
`src/vimol/select.py`.
Changed: `scene.py`, `widget.py`, `viewer.py`, `app.py`, `__init__.py`,
and — for flat shading only — `render.py`, `gl_render.py`, `gl_adapter.py`.

**Revision 2** resolves all six open questions from revision 1 (§12), makes
single-file multi-molecule support the first story (§2), and adds the structure
list UI and overlay colouring (§4).

---

## 1. Container

`src/vimol/structures.py`. Three dataclasses and one class, in vimol's
existing "parallel numpy arrays plus plain Python lists" idiom.

```python
@dataclass
class Transform:
    """A rigid body transform in world space: x' = x @ rotation.T + translation.

    Kabsch produces exactly this pair, so nothing has to be repacked. 3x3 + (3,)
    rather than a 4x4 because Camera already works this way and vimol has no
    4x4 anywhere outside the GL projection matrix.
    """
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, positions: np.ndarray) -> np.ndarray:      # (N,3) -> (N,3)
    def apply_directions(self, vectors: np.ndarray) -> np.ndarray:  # rotation only
    def compose(self, other: "Transform") -> "Transform":      # self ∘ other
    def inverse(self) -> "Transform":
    @property
    def is_identity(self) -> bool:                             # exact-eye/zero test
    def key(self) -> bytes:                                    # cache key material


@dataclass
class AlignmentResult:
    """What an align call produced, kept for reporting and reproducibility."""
    rmsd: float
    n_fitted: int                       # atoms that entered the fit
    transform: Transform
    ref_label: str
    method: str                         # "index" | "subset" | "permute"
    select: Optional[np.ndarray] = None      # mobile indices used for the fit
    ref_select: Optional[np.ndarray] = None  # reference indices, if different
    mapping: Optional[np.ndarray] = None     # mobile idx -> reference idx (permute only)
    stale: bool = False                      # source geometry edited since the fit


@dataclass
class Structure:
    """One loaded molecule plus the per-set state that belongs to it."""
    molecule: Molecule
    label: str                          # unique within the set; shown in the UI
    path: Optional[str] = None          # source file, None for in-memory
    transform: Transform = field(default_factory=Transform)
    visible: bool = True                # 'h' hide; excluded from the composite
    marked: bool = False                # 'space' mark; the overlay/align multi-select
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # assigned at append time
    alignment: Optional[AlignmentResult] = None
    revision: int = 0                   # bumped by touch(); drives composite caching
    undo_stack: List = field(default_factory=list)
    saved_sig: Optional[tuple] = None   # per-structure [MODIFIED] tracking

    def touch(self) -> None:            # call after ANY in-place molecule mutation
        self.revision += 1


class StructureSet:
    entries: List[Structure]
    active_index: int = 0
    overlay: bool = False               # False: draw active only. True: draw the marked set.
```

`tint` is assigned once at `append` time from a fixed palette
(`structures.TINTS[entry_index % len(TINTS)]`) rather than being recomputed:
a structure must keep its colour when others are hidden or reordered, because
the same colour identifies it in the scene, the list swatch and the
measurement table. High-contrast, colour-blind-safe, and deliberately not near
any common CPK colour (which is reserved for the active structure, §4):

```python
TINTS = [(0.20, 0.85, 0.45),   # green
         (1.00, 0.55, 0.10),   # orange
         (0.65, 0.45, 0.95),   # purple
         (0.20, 0.75, 0.95),   # cyan
         (0.95, 0.35, 0.55),   # pink
         (0.85, 0.80, 0.25),   # gold
         (0.45, 0.60, 1.00),   # periwinkle
         (0.55, 0.85, 0.75)]   # sea
```

`StructureSet` API:

```python
    # sequence protocol -- iteration yields Structure, not Molecule.
    __len__, __iter__, __getitem__(int | str)       # str looks up by label
    molecules -> List[Molecule]                     # convenience for library users
    labels -> List[str]
    active -> Structure                             # entries[active_index]
    set_active(int | str) -> None
    cycle_active(step: int = 1) -> None             # wraps; drives n/p and ]/[
    append(molecule, label=None, path=None) -> Structure
    extend(molecules, ...) ; remove(int | str)
    invalidate() -> None                            # drop the composite cache
    composite() -> Composite                        # see §3
    drawn_indices() -> List[int]                    # what the composite contains, §3
    marked -> List[Structure] ; clear_marks() ; toggle_mark(i)
    solo(i) / unsolo() ; toggle_visible(i)          # see §4
    align(mobile, onto, **kw) -> AlignmentResult    # see §7
    align_marked(onto=None, **kw) -> List[AlignmentResult]
    measure(indices) -> List[Tuple[str, Optional[float]]]   # see §8
```

**Labels** are the stable identity. Derived from the source path's basename;
multi-model files (a 20-frame `traj.xyz`, an NMR PDB) get `traj.xyz#3`,
1-based. Collisions across directories get disambiguated by appending the
parent directory (`a/mol.xyz`, `b/mol.xyz`), then by a numeric suffix.
Labels never change once assigned — the render path, status bar, measurement
table and `sset["traj.xyz#3"]` all key off them.

*Rejected:* identity by list index (renumbers on `remove`), and by `id(molecule)`
(not printable, not scriptable).

---

## 2. Files × frames: one axis, not two — and VIM-9 is its first consumer

`Viewer` already has a `frames: List[Molecule]` axis with `n`/`p`, and the
locked `xyz-multi-struc-toggle` worktree adds a clickable `struc N/total` pill
for it. `vimol a.xyz b.xyz` where `a.xyz` holds 5 models would otherwise mean
two competing concepts.

**Decision: collapse them.** `load_all()` on a multi-model file yields N
molecules → N `Structure` entries with `#k` labels. `n`/`p`, the worktree's
`opt+up`/`opt+down` and the strip's row activation (§4.3) all call
`cycle_active`.

`Viewer` keeps both names as live properties, not read-only aliases —
`app.py:240` already does `viewer.frame_index = idx`:

```python
    @property
    def frames(self) -> List[Molecule]:      # deprecated; use .structures
        return self.structures.molecules

    @property
    def frame_index(self) -> int:
        return self.structures.active_index

    @frame_index.setter
    def frame_index(self, i: int) -> None:
        self.structures.set_active(i)        # also refreshes the widget
```

`Viewer.__init__(molecule, frames=[...], ...)` keeps its exact signature: a
`frames` list is converted into entries, and `molecule` alone becomes a
one-entry set. A new `structures=` keyword takes a `StructureSet` directly and
is what `app.py` will pass.

### VIM-9 — single-file multi-molecule is the first story

**A file that already contains several molecules is the first consumer of the
collapse, and ships before any multi-file CLI work.** Multi-model `.xyz`
trajectories, multi-`MODEL` PDBs and multi-record SDFs all already parse into
`List[Molecule]` (`parsers/__init__.py:load_all`), and `app.py:163-176,237-240`
already carries them to `Viewer(frames=...)` with `--frame` picking an index.
So VIM-9 is `StructureSet` exercised end to end against **working code and real
fixtures**, with the parser, CLI and file-format axes all held constant — the
cheapest possible proving ground for the container before multi-file loading
adds per-file failure handling, mixed formats and cross-file labels on top.

VIM-9 delivers: `load_all` → `StructureSet` with `#k` labels, the structure
list strip (§4), `n`/`p`/digit navigation, overlay of a marked subset,
and per-structure tint. Everything in §§1, 3, 4 is exercised; nothing in §§6–8
is required.

The `xyz-multi-struc-toggle` worktree (`cc51164`) is the **ancestor of VIM-9**,
not a later merge: its `opt+up`/`opt+down` navigation and clickable `struc N`
pill are the first two features of the strip. Rebase it as the opening commit
of VIM-9 and grow it onto `StructureSet` — `self.frames` becomes
`self.structures`, `self.frame_index` becomes the property above, and the
`struc N/total` pill becomes the strip's header plus the active-row highlight.
Its `input.py` alt-arrow decoding is independent of all of this and should land
on `main` first, on its own, because §4's keymap needs the same mechanism.

**Does VIM-9 change the `frames`/`frame_index` shim?** No — it is the reason
the shim exists. VIM-9 is precisely the case where `frames` was the *only* API,
so the property pair must be live (readable and settable) from VIM-9's first
commit, not deferred to the multi-file work. `app.py:240`'s
`viewer.frame_index = idx` must keep working unchanged through the whole story.

Consequences for `app.py`:

- `file` becomes `nargs="*"`. Zero files → the bundled C60 demo, as today.
- A file that fails to parse prints `warning: skipping <path>: <err>` to stderr
  and loading continues; exit code 3 only if *nothing* loaded (VIM-1).
- `--frame K` selects the initial `active_index` **into the whole set**, in load
  order — unchanged meaning for the single-file case, and unambiguous for many.
  `--active LABEL` is the readable form for multi-file sessions.
- ~~`--render` / `--kitty` with 2+ structures render the **overlay**~~ and
  ~~`--info` prints today's block per structure~~ — **superseded 2026-07-30.**
  `--render`, `--kitty` and `--info` (and the `--size` / `--supersample` flags
  that only configured them) were **removed from the CLI entirely** when VIM-1
  shipped, rather than extended to multi-file. There is no batch-still or
  batch-inspection flag left for a multi-structure rule to apply to. The
  underlying `Scene.to_png()` / `Scene.to_kitty()` library API is untouched.
  See `docs/superpowers/specs/2026-07-30-multi-file-cli-design.md`.
- **Auto-overlay (VIM-1, as shipped).** 2+ files → `overlay = True` with the
  **first model of each file** marked, so the overlay opens showing one
  structure per file; every other model stays loaded and reachable (`n`/`p`,
  the list strip, `opt+click` to add or drop). A single file opened alone is
  unaffected: `overlay` stays `False` and nothing is auto-marked.

The worktree branch should be **rebased onto this API as part of VIM-2**, not
merged first: its `self.frames` / `self.frame_index` references become
`self.structures`, and its status-bar pill becomes the label pill. Its
`input.py` alt-arrow decoding is independent and can land on `main` as-is.

---

## 3. Render path

`Scene` renders a **composite**: one flattened `Molecule` in world/display
coordinates, built from the visible entries, handed to the *existing*
single-molecule renderer. Both backends are served by one code path because
`GLRenderer.render()` already takes single concatenated batches and the CPU
`Renderer._framebuffers()` clears per call — any "loop over N molecules"
design collapses back into concatenation at the GL side anyway.

### Which entries are drawn

One rule, used by the composite, the strip's highlighting and the legend:

```python
    def drawn_indices(self) -> List[int]:
        """Entry indices in the composite, active first. Active-first matters:
        the first drawn entry is the one that keeps CPK colours (§4)."""
        if not self.overlay:
            return [self.active_index] if self.active.visible else []
        marked = [i for i, e in enumerate(self.entries) if e.marked and e.visible]
        rest = marked if marked else [i for i, e in enumerate(self.entries) if e.visible]
        rest = [i for i in rest if i != self.active_index]
        return ([self.active_index] if self.active.visible else []) + rest
```

`overlay` off → the active structure alone, which is byte-for-byte today's
behaviour. `overlay` on with nothing marked → every visible structure.
`overlay` on with marks → the active plus the marked ones, which is what
opt-click builds.

```python
@dataclass
class Composite:
    molecule: Molecule          # flattened; positions are POST-transform
    offsets: np.ndarray         # (K+1,) int; atom-index base per drawn slot
    sources: np.ndarray         # (K,) int; index into StructureSet.entries
    base_colors: np.ndarray     # (N,3) see "overlay colouring", §4
    flat: np.ndarray            # (N,) bool; per-atom flat-shading flag, §4

    def locate(self, i: int) -> Tuple[int, int]:
        """Composite atom index -> (entry index, local atom index)."""
        k = int(np.searchsorted(self.offsets, i, side="right")) - 1
        return int(self.sources[k]), i - int(self.offsets[k])

    def globalize(self, entry_index: int, local: np.ndarray) -> np.ndarray:
        """Entry-local atom indices -> composite indices (inverse of locate)."""
```

Build (`StructureSet.composite()`):

1. positions: `np.concatenate([e.transform.apply(e.molecule.positions) ...])`
2. symbols: list concat; bonds: `(i + off, j + off, order)`
3. `base_colors` and `flat`: per the colouring rule in §4
4. vector fields: concatenated, each padded to composite length, vectors
   rotated by `transform.apply_directions` (no translation — they are free
   vectors)

**Transforms are applied here and nowhere else.** Not in `Camera` (one camera
serves the whole view), not in the renderers (both would duplicate the work),
and never by mutating `molecule.positions` — that is VIM-4's explicit
acceptance criterion.

Because the composite carries *world* coordinates, picking, `unproject`,
measurement and the fog depth range all keep working unmodified, and a
cross-structure distance read off the screen is the distance you see.

**Fast path.** `len(drawn_indices()) == 1 and entry.transform.is_identity` →
`composite.molecule` **is** that entry's `Molecule` object, zero copy;
`base_colors` is plain `element_colors()` and `flat` is all-`False`, so
`style.flat_mask` stays `None` and `render.py` / `gl_render.py` take today's
exact path.

The condition keys on **drawn state, not on tint**. Every entry carries a tint
from the moment it is appended (§1), so a tint-based test would disable the fast
path for every multi-structure session — including the common one of viewing a
20-frame trajectory one frame at a time, where exactly one structure is drawn.
It is sound because §4.4 gives the *first drawn* entry CPK colours and ignores
its tint: a lone drawn entry is by definition the CPK one, so its tint is not
part of the output. This is what keeps `mol is scene.molecule` true (§10) and
keeps single-structure rendering byte-for-byte what happens today, whether or
not other structures are loaded alongside.

### Caching — a requirement at protein scale, not an optimisation

`StructureSet` memoizes the composite against a key of `(len(entries),
tuple((id(e.molecule), e.revision, e.visible, e.marked, e.tint,
e.transform.key()) for e in drawn))`, and caches **topology** (symbols, bonds,
base colors, flat mask) separately from **positions**, so a transform-only
change re-concatenates positions only. In-place mutation of a `Molecule` by
anything outside vimol requires `entry.touch()` or `sset.invalidate()`; this is
documented on `Structure.molecule`.

Measured (Apple M2, numpy 1.24.4, median of 11 after warmup):

| composite | full rebuild | positions only | one CPU raycast frame |
| --- | --- | --- | --- |
| 4 × 60 = 240 atoms | 0.15 ms | 0.01 ms | 89 ms (240 atoms, 640×480) |
| 16 × 60 = 960 atoms | 0.58 ms | — | |
| 4 × 500 = 2000 atoms | 1.08 ms | — | 570 ms (1500 atoms, 640×480) |
| **3 × 1500 = 4500 atoms** | **2.74 ms** | **0.10 ms** | 1348 ms (4500 atoms, 640×480) |
| 5 × 1500 = 7500 atoms | 5.03 ms | 0.09 ms | 1842 ms (4500 atoms, 900×700) |

Against the **CPU** raycaster a rebuild is 0.2% of a frame at every size, so it
would not matter. Against the **GL** backend at Hani's apo/holo/mutant scale a
frame is single-digit milliseconds, and an unconditional 2.74 ms Python rebuild
would then be a third to a half of the frame budget — a visible cost during a
drag, for work that produced an identical array. **The cache is therefore a
requirement.** What makes it effective is the split:

- **camera drag** (the hot path): nothing in the key changes → cache hit, 0 ms.
  This is the case that would otherwise pay 2.74 ms per frame for nothing.
- **align / tint / transform change**: positions only → 0.10 ms at 4500 atoms.
  The 27× gap between full rebuild and positions-only is the whole reason the
  topology cache is specified separately.
- **edit** (`touch()`) or a visibility/mark change: full rebuild, ≤ 3 ms, and
  only on the frame where the user actually changed something.

The full rebuild's cost is dominated by the Python-level bond re-indexing loop,
which is exactly the part the topology cache skips.

**The topology cache must also hold the per-atom derived arrays.**
`Molecule.vdw_radii()`, `element_colors()` and `atomic_numbers()` are Python
list comprehensions over `symbols` that recompute on every call
(`molecule.py:127-137`), and they are called on the *hot* path, not the rebuild
path: `render.py:_atom_radii` runs `vdw_radii()` every frame,
`Scene._max_atom_radius` runs it on every `fit`, and `widget.pick` runs it on
every mouse-move. Measured on the same machine:

| composite atoms | `vdw_radii()` | `element_colors()` | `atomic_numbers()` |
| --- | --- | --- | --- |
| 240 | 0.08 ms | 0.10 ms | 0.08 ms |
| 1500 | 0.47 ms | 0.62 ms | 0.50 ms |
| **4500** | **1.37 ms** | **1.86 ms** | 1.54 ms |
| 7500 | 2.24 ms | 3.09 ms | 2.55 ms |

At 4500 atoms a render-plus-pick frame pays ~4.6 ms for these — **larger than
the 2.74 ms composite rebuild this section exists to eliminate**, and it is a
cost the *current* single-molecule code already pays at that size. They are pure
functions of `symbols`, so the topology cache key already covers them: cache
`vdw_radii` and `element_colors` alongside the flattened symbols/bonds and hand
them out as read-only views. (Fixing this for plain single-molecule rendering
too is a small standalone win, but it is not in this design's scope — note it
and move on.)

### What reads the composite vs. the active structure

| reads the **composite** | reads the **active** structure |
| --- | --- |
| `Scene.render` / `to_kitty` / `to_png` | append / delete / relax / manual bond |
| `Scene.fit` — `centroid`, `radius_of_gyration_extent`, `vector_extent`, `_max_atom_radius` | undo, dirty flag, save |
| fog depth range (falls out of `render`) | `atom_info`, the status-bar formula/atom count |
| `widget.pick` / hover | measure pick list (`measure_sel`) |
| `widget.unproject` | `Scene.molecule`, `widget.molecule` |
| camera orbit / zoom / pan / roll | |

`Scene.fit` reading the composite is what makes overlay framing correct; if it
read the active structure the other files would sit off-screen.

`Scene.set_molecule(mol)` is redefined as "replace the set with a single
entry", which keeps `widget.set_molecule` and every existing embedder working.

### Highlighting

`style.color_override` is an (N,3) array consumed by both backends
(`render.py:137`, `gl_adapter.py:57`), so it must be **composite-sized**.
`widget._base_colors` becomes `composite.base_colors`.
`widget.hovered` / `selected` / `measure_sel` stay **active-local** (they feed
editing and measurement); `_apply_highlight` maps them through
`composite.globalize(active_index, idx)` before writing. `pick()` returns a
composite index and is converted with `composite.locate()` at the call site.

In append, delete **and measure** mode a pick that resolves to a non-active
structure is **ignored**, with `msg = "atom belongs to <label> — Tab to
activate"`. Silently editing a structure the user is not focused on is worse
than a no-op — and for measure the failure is quieter and worse: `pick()`
returns a *composite* index, so an unguarded cross-structure click would store
an index that means a different atom in the active structure and report a
confident, wrong number. `measure_sel` therefore holds **active-local** indices
only, which is exactly the invariant §8 depends on.

*Rejected:* letting `measure_sel` hold composite indices so a measurement can
span two structures. That is a real feature (an A-to-B contact distance) but a
different one: such a distance lives in world coordinates and is **not**
invariant under the per-structure transforms, so it would change whenever
something is aligned — the opposite of VIM-6's semantics. File it separately if
wanted.

---

## 4. Structure list, overlay colouring, flat shading

### 4.1 The strip

A vertical tab strip on the left, drawn **only when `len(sset) > 1`**. It is
ANSI text written straight into terminal cells — the same technique as
`_draw_periodic_table` and `_draw_geometry_picker`, not part of the Kitty
image — so it costs nothing to render and never touches the renderer.

```
 STRUCTURES 3

 █ 1 apo.pdb          <- active: full-width background, near-white label
 █ 2 holo.pdb
 ░ 3 mutant.pdb       <- hidden: hollow swatch, whole row dimmed
 ──────────────────
  1 - 9  jump to
  n  p  next/prev
  z  solo  h  hide
 overlay 1+2 · aligned
 camera shared
```

*(Rows carry no atom count: the number is noise next to the label, and the
strip is narrow enough that the label needs every column it can get.)*

**Structures are grouped by source file.** A file contributing more than one
structure gets a non-selectable header row naming it once, with its models
listed beneath as `frame 1`, `frame 2`, …; a file contributing exactly one
gets no header, just its own row labelled with the basename. Otherwise a
100-frame trajectory repeats its filename a hundred times.

```
 STRUCTURES 4
 traj.xyz
 1 █ frame 1
 2 █ frame 2
 3 █ frame 3
 4 █ apo.pdb        <- a second file, alone, so no header of its own
```

`Viewer._list_display_rows()` is the *only* place display rows and structure
indices are related: it returns `(kind, structure_index, text)` per row, and
`_draw_list` registers a click span (plus its structure index in
`_list_row_struct`) only for the rows it actually emitted. Keys, clicks and
`1`–`9` address **structures**, never display rows; group headers own no
structure, so clicking one is inert. Grouping runs over *consecutive*
structures — they load in file order, and grouping across a gap would reorder
the strip behind the user's back.

**The strip scrolls.** `Viewer._list_scroll` is the first visible display row.
`_list_capacity()` — terminal height minus the header above and the separator
and footer below — is the single source of truth for what fits: `_draw_list`
slices the rows with it and every scroll clamps against it, so a resize can
never strand the offset. The mouse wheel over the strip scrolls it (and never
reaches the widget's zoom); `Home`/`End` jump to the ends; `j`/`k`, `1`–`9`,
and `n`/`p` scroll the minimum needed to keep their target visible,
computed on **display** rows (`_list_ensure_visible`) — scrolling up to a
file's first frame brings its header along. A dim `↑`/`↓` in the header row
says there is more above/below.

**The panel's look.** Dark, calm and spacious; a row is
`<space><swatch> <index> <label>` and carries **no leader glyph at all**:

- **active row** — a full-panel-width background (`_LIST_ACTIVE_BG`, spanning
  the leading and trailing padding, not just the text) plus a near-white
  label. **cursor row** — the same idea in a subtler background, so the two
  stay tellable apart on the rows where they differ.
- **row in the overlay** — the label in the structure's own tint. Overlay
  membership still has to be visible once `✓` is gone, and colour already
  identifies a structure everywhere else (§4.4). With no key bound to
  membership it is the *only* on-screen reading of the overlay set, so it
  matters more, not less. The tint outranks the active row's near-white
  label: opt+clicking the active row must change something on screen, and
  the background already says which row is active.
- **swatch** — `█` in the structure's truecolor tint, `░` and dimmed when
  hidden (the whole row dims). **index** — 1-based, in a grey dimmer than the
  label.
- **header** — muted blue-grey, not bold-white, with a blank row beneath it.
  Plain `STRUCTURES N`: letter-spacing it would need ~24 of the 18 columns a
  default terminal gives the strip.
- **separator** — a dim rule inset one column each side, then a **legend** of
  key caps (padded key text on a lighter background) for `1`-`9`, `n`/`p`,
  `z`, `h`.
- Every row is emitted by `_list_line`, which lays out `(text, sgr)` segments
  to an exact *visible* width. SGR sequences are zero-width, so measuring a
  pre-decorated string with `len()` is precisely what corrupts the layout;
  segment text is escape-free by construction and each styled segment is
  closed with a reset plus a re-applied row background (a key cap carries its
  own background and must not leak into the rest of the row).
- The legend is sized to fit `_LIST_W_MIN` and is the last thing to go: on a
  short panel the status lines below it fall off first (`put()` simply
  refuses any row that would land on the status bar).
- Footer indicator: overlay membership as **1-based row
  numbers matching the rows above** (`overlay 1+2`) — never labels, which are
  too long and already shown on the rows; `aligned` when any drawn entry has a
  non-identity transform (`aligned*` when any is stale); and `camera shared`.
- Width `_LIST_W = min(28, max(18, cols // 5))` cells; labels are middle-
  truncated with `…` so the extension stays visible.

**Camera sharing is a stated fact, not a toggle.** One `Camera` per `Scene` is
a core invariant of §3 — per-structure transforms are applied in the composite
*precisely so* one camera serves everything. The indicator tells the user that
rotating moves all structures together; per-structure cameras are out of scope.

### 4.2 Viewport geometry and mouse mapping

`_update_geometry` reserves the strip's columns:

```python
    list_w = _LIST_W if len(self.structures) > 1 else 0
    self._img_cols = cols - list_w
    self._img_origin_px = (list_w * cw, 0)
```

`_draw` currently emits `_HOME + data`, which places the image at row 1 col 1.
It becomes `b"\x1b[1;%dH" % (list_w + 1) + data` — the image is placed at the
strip's right edge, exactly the cursor-positioning pattern the existing panels
use. The strip itself is written as `list_w`-wide segments on rows 0..rows-2.

**Mouse mapping needs no new machinery.** `widget.handle_event` /
`handle_mouse` already take `origin=(x_px, y_px)` and `_local_px` subtracts it
(`widget.py:158-165`) — the embedding API exists for exactly this. `_dispatch`
currently calls `self.widget.handle_event(ev)` with no origin; it passes
`origin=self._img_origin_px` instead. Two further pieces, both mirroring the
status bar's existing handling:

- a column hit test `_in_list_zone(col) -> col < list_w`, the horizontal twin
  of `_in_status_zone(row)`;
- a `_list_zone_press` latch, so a drag that begins on the strip never reaches
  the viewport even if the pointer strays over the molecule mid-drag — the
  identical bug `_status_zone_press` already guards against.

### 4.3 Keys — a focused strip, so no existing binding changes

`1`–`9`, `h` and `z` all collide with bindings that exist today: `1`–`4` set
the representation and `h` is the vim orbit-left key (`widget.handle_key`,
`widget.py:251-272`). Relocating established bindings to make room is the
expensive answer. (`[`/`]` — camera roll — are deliberately *not* in the
strip's keymap at all: next/prev is the global `n`/`p`, so the strip never
shadows the roll keys, focused or not.)

The overlay toggle is `v`, not `o`: `o` is claimed by `_EDIT_DRIVER_KEYS`
(`viewer.py`) as the editable-mode relocation of autospin, and the strip's
keymap only applies while it has focus — an editable viewer with the strip
focused would otherwise have `o` toggle the overlay instead of autospin. `v`
collides with nothing in `widget.handle_key` or the driver keys.

**Decision: the strip takes keyboard focus.** `Tab` toggles focus between the
viewport (default) and the strip; clicking any row also focuses it; `Esc`
returns focus to the viewport. While the strip has focus its keymap applies and
**not one existing binding changes**. One new global key buys the whole
namespace.

| key | focus | action |
| --- | --- | --- |
| `Tab` | global | toggle strip focus |
| `n` / `p` | global | next / prev structure (unchanged, already shipped) |
| `opt+↑` / `opt+↓` | global | next / prev structure (from the worktree branch) |
| `j` / `k` | strip | move the row cursor without activating |
| `↑` / `↓` | global | orbit the camera — the strip never claims the plain arrows |
| `1`–`9` | strip | jump to structure N |
| `Enter` | strip | activate the row cursor, clear the overlay set, return focus |
| `z` | strip | **solo** toggle |
| `h` | strip | **hide** toggle on the cursor row |
| `v` | strip | toggle `overlay` |
| `Esc` | strip | return focus to the viewport |

Mouse, focus-independent: **click** a row → `set_active(k)`, `clear_marks()`,
`overlay = False` — "that structure replaces the pane", identical to next/prev.
**opt+click** a row *toggles* its overlay membership (`Structure.marked`,
the internal flag) — "add it to / drop it from the overlay alongside the
current one" — and sets `overlay` to whether any member survives. **opt+click
is the only way to change membership**; there is no key binding for it, and
the UI never calls it a "mark". Dropping the last member therefore turns the
overlay off: overlay with an empty membership set means "draw every visible
structure" (§3 `drawn_indices`), which is not what dropping your last
selection asks for.

**The row cursor is not the active index.** `j`/`k` move a cursor without
changing what is rendered, so `z` / `h` need an unambiguous target:
they all act on **`Viewer._list_cursor`**, an int that starts at `active_index`
and is reset to it by every `set_active`. The two are drawn differently — `▸`
plus tint-coloured text marks the *active* row, a reverse-video background
marks the *cursor* row, and they coincide until `j`/`k` is pressed.
`StructureSet.toggle_mark(i)` / `solo(i)` / `toggle_visible(i)` (§1) take that
index explicitly and never read `active_index` themselves. `Enter` is the one
key that promotes cursor → active.

Semantics against `StructureSet`:

- **overlay membership** (opt+click only) — `Structure.marked`, the internal
  multi-select flag. It is the *one* subset concept, consumed by both overlay
  (`drawn_indices`, §3) and alignment (`align_marked(onto=active)` fits every
  member onto the active one). Rejected: separate overlay-set and align-set
  state; users would have to build the same selection twice. The flag keeps
  its `marked` name in the model; the **UI never says "mark"** — a row is
  either in the overlay or not, and opt+click is how it gets there.
- **solo** (`z`) — a **toggle**. `solo(i)` stores the current visibility vector
  in `_solo_restore` and sets `visible` True on `i` only; `z` again restores it.
  One-way solo would strand the user re-showing rows by hand.
- **hide** (`h`) — toggles `Structure.visible` on the cursor row. The last
  visible structure cannot be hidden (a blank pane reads as a crash); the
  attempt is refused with `at least one structure must stay visible`. Hiding
  the *active* structure does **not** auto-advance the active index — editing
  and measuring keep targeting it, and the row renders dimmed — so `h` twice is
  an exact round trip.

### 4.4 Overlay colouring

- The **first drawn** entry — always the active one (§3) — keeps normal
  per-element CPK colours and normal shading.
- **Every other** drawn entry renders **flat** (no diffuse/specular gradient)
  and uniformly in its own `tint`, no element variation at all.
- The palette is `structures.TINTS` (§1), shared by the scene, the strip
  swatches and the measurement table, so colour identifies a structure
  everywhere it appears.

```python
    # inside composite(): k is the position in drawn_indices()
    if k == 0:
        base_colors[lo:hi] = e.molecule.element_colors()
        flat[lo:hi] = False
    else:
        base_colors[lo:hi] = e.tint
        flat[lo:hi] = True
```

Depth cue still applies to flat atoms, so overlapping structures keep reading
front-to-back. Hover highlighting tints a flat atom's colour without unflatting
it.

*Note:* the mockup image renders **all three** structures uniformly tinted,
with no CPK structure. This design follows the written instruction — first CPK,
others flat+tinted — and treats the image as a sketch artifact.

### 4.5 Flat shading: verified expressible per-atom, in one pass, in both backends

This is **not** the two-pass z-buffer problem (§12.2). Mixing *representations*
would need two passes; mixing *shading models* does not, because both backends
already shade per fragment from per-primitive inputs. Checked both:

**CPU (`render.py`).** Shading happens in the `shade_write` closure
(`render.py:193-229`), called per primitive with that primitive's `albedo`, and
the atom loop already dispatches per atom in Python (`for a in order[in_band]`,
`render.py:273`). So a per-atom branch is free. Add `Style.flat_mask: object = None` —
a per-atom bool array, the exact idiom `color_override` already uses
(`render.py:39`) — and a `flat` argument to `shade_write` that short-circuits to
`rgb = albedo * fog` (skipping diffuse, fill and specular). Flat atoms are
*cheaper* than shaded ones. Bonds inherit flatness from their endpoints — bonds
never span structures in the composite — and `_draw_cylinder_segment` already
receives `shade_write` as a parameter (`render.py:386-388`), so it takes the
flat variant with no signature churn. **~15 lines.**

**GL (`gl_render.py`).** One extra per-instance float. The instance buffers are
interleaved format strings:

- spheres, `gl_render.py:691`: `"3f 1f 3f/i"` → `"3f 1f 3f 1f/i"`
- cylinders, `gl_render.py:712`: `"3f 3f 1f 3f 3f/i"` → `"3f 3f 1f 3f 3f 1f/i"`

plus `layout(location = 4) in float in_flat;` passed through to the fragment
shader, and one line in `_ATOM_FRAG` (and its bond twin) replacing

```glsl
    vec3 shaded = v_color * diff + vec3(spec);
```

with

```glsl
    vec3 shaded = mix(v_color * diff + vec3(spec), v_color, v_flat);
```

Branch-free, placed before the existing depth-cue block so flat fragments still
fog. `SphereBatch` / `CylinderBatch` gain a `flat` field defaulting to zeros, so
`.empty()` and every existing caller keep working unchanged; `gl_adapter`
forwards `style.flat_mask`. **~20 lines.**

**Arrows are never flat.** Arrow shafts are folded into the same cylinder batch
as bonds (`gl_adapter.py:86-93`), so the `flat` array is simply zero over the
arrow tail of that batch, and the cone shader (`gl_render.py:435`) is not
touched at all. Vector fields are a per-molecule annotation; flattening them
would lose the shape cue that makes an arrow read as an arrow.

Both changes are additive and default to today's behaviour when `flat_mask` is
`None`. No fallback or downgrade is needed.

## 5. Editing semantics and undo

- Edits always target `set.active.molecule`, in place, through the existing
  `editor` functions. `editor.py` keeps its `Molecule`-only signature.
- Every edit path ends with `entry.touch()` — the one new obligation.
- **Undo is per structure.** `Structure.undo_stack` replaces
  `widget._undo_stack` (limit 200, unchanged). Switching structures preserves
  both histories. *Rejected:* one global chronological stack — pressing `u`
  after `Tab` would silently modify a structure you are not looking at.
- Stack entries are tagged, so alignment is undoable too:
  - `("geometry", symbols, positions, bonds, manual_bonds, new_atoms)` — today's
    snapshot tuple
  - `("transform", old_transform, old_alignment)` — pushed by `align`
- `dirty` / `saved_sig` move onto `Structure`. The status bar shows
  `[MODIFIED]` for the active entry; the quit-confirm prompt fires if **any**
  entry is dirty and names them.
- An edit after an alignment sets `entry.alignment.stale = True`. The transform
  stays valid (it is rigid); only the reported RMSD is out of date, and the
  status bar shows `RMSD 0.42 (stale)`.
- **Save writes source coordinates**, not transformed ones — an alignment is a
  viewing aid, and round-tripping a file through vimol should not silently
  move it. `save(..., apply_transform=True)` bakes it in; the save prompt gains
  a toggle when the active entry has a non-identity transform.

---

## 6. Atom selection for alignment

**Representation:** a `np.ndarray` of ascending int64 atom indices into the
*mobile* structure. Bool masks and slices are normalized to that on entry.

**Where it lives:** it is an **argument to the align call**, not persistent
container state. The viewer keeps a transient `widget.selection: List[int]`
(a new select mode; `measure_sel` stays separate — one is a 2/3/4-atom ordered
pick list, the other an unordered set of arbitrary size). The selection that
was used is recorded on `AlignmentResult.select` so a fit can be reported and
reproduced. *Rejected:* storing a selection on `StructureSet` — it is
UI-scoped, and mobile and reference may need different ones, which the result
record already captures.

```python
sset.align("edited.xyz", onto="original.xyz", select=core)   # core: index array
```

`ref_select` gives the matching reference indices; when omitted, `select` is
reused (index correspondence between mobile and reference). Mismatched lengths
raise `ValueError`. The fit is computed over the selection; **the resulting
transform is applied to every atom** of the structure via `Structure.transform`.

**Interactive → API.** The status bar shows `sel 12`; the help overlay and the
`V` key print the Python equivalent — `sset.align(1, onto=0, select=[...])` —
into the message line, with the literal index list, so an interactive fit can
be pasted into a script verbatim.

**Helpers** (`src/vimol/select.py`, pure functions returning index arrays, no
state):

```python
select.all(mol) ; select.indices(mol, seq) ; select.heavy(mol)
select.symbols(mol, "C", "N")        # by element
select.within(mol, center, radius)   # geometric sphere
select.invert(mol, sel) ; select.union(a, b) ; select.intersect(a, b)
```

`select.backbone()` / `select.residue()` are **not** offered — they need atom
names the parsers discard. That is **VIM-10**, and it plugs in here without
changing anything: it adds `select.*` functions returning the same index arrays
(§12.1).

---

## 7. Alignment: algorithm and performance tiers

`src/vimol/align.py`.

```python
def kabsch(P, Q) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimal rotation+translation taking P onto Q (index correspondence).
    Returns (rotation, translation, rmsd). det is forced to +1 -- a reflection
    would 'fit' an enantiomer onto its mirror image."""

def superpose(mobile: Molecule, reference: Molecule, *,
              select=None, ref_select=None, permute: bool = False,
              trials: int = 2000, candidates: int = 64,
              ) -> AlignmentResult:
    """The single entry point. Tier is chosen by (select, permute)."""

def permutation_search(P, Q, symbols, trials, candidates
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Tier 3: optimal rotation + atomic permutation. (R, perm, rmsd)."""

def _assign(cost: np.ndarray, blocks) -> np.ndarray:
    """Element-blocked linear sum assignment. scipy if importable, else the
    bundled numpy LAPJV."""
```

### Licensing — decided, not open

vimol stays **MIT**. The algorithm is implemented from the published
description in Finkler & Goedecker, *J. Chem. Phys.* **152**, 164106 (2020)
("Funnel hopping Monte Carlo"), plus standard published methods: Kabsch
superposition (Kabsch, *Acta Cryst.* A32, 922 (1976)), the quaternion form
(Horn, *JOSA A* 4, 629 (1987)), and the Jonker–Volgenant shortest-augmenting-path
assignment algorithm (Jonker & Volgenant, *Computing* 38, 325 (1987); Crouse,
*IEEE Trans. AES* 52, 1679 (2016)). **RMSD-finder (GPL-3.0) is used only as a
test oracle** — its numbers are compared against, its source is not read,
ported, or paraphrased. Cite the paper in the module docstring.

**Fidelity bar:** agreement on the **final minimum RMSD value** across the
fixture set. The permutation and rotation that achieve it need not match — the
minimum is frequently degenerate (C60, benzene), and reproducing a specific
search trajectory is not a requirement. This buys the freedom to choose our own
rotation seeding and convergence test, validated by fixtures rather than by
imitation.

### Tier 1 — index correspondence (interactive default)

Plain Kabsch over all atoms, in index order. Covers trajectory frames,
conformers and edited copies — the common vimol case. Requires
`mobile.symbols == reference.symbols`; otherwise the call refuses rather than
returning a meaningless number.

### Tier 2 — explicit atom subset (interactive)

Kabsch over `mobile.positions[select]` vs `reference.positions[ref_select]`;
the transform is applied to all atoms. RMSD is reported over the selection,
with the all-atom RMSD alongside it when topologies match.

### Tier 3 — permutation-invariant (opt-in, `permute=True`)

1. **Centre both structures on their centroids.** For an unweighted RMSD
   objective the optimal translation superimposes unweighted centroids, so this
   is optimal, not a compromise; and with identical element multisets the
   centroid is permutation-invariant, so it is valid *before* the permutation is
   known. (`elements.py` carries no atomic masses, and does not need to.)
2. **Trial rotations.** `trials` (default 2000) deterministic, near-uniform
   quaternions from the Super-Fibonacci construction (Alexa, CVPR 2022), a
   closed-form four-line sequence. Deterministic beats a seeded RNG here: the
   same inputs give the same answer across runs and platforms, which the
   fixture tests depend on.
3. **Screen** all trial rotations with a batched **relaxed lower bound**: for
   each rotation, the element-blocked squared-distance matrix's per-row minimum,
   summed. Relaxing injectivity can only lower the cost, so this is a genuine
   lower bound on that rotation's optimal assignment cost — pruning any
   candidate whose bound exceeds the best cost found so far provably keeps the
   optimum *among the trial set*. Computed as one batched
   `(M,n,3)@(3,n)` matmul in float32, chunked to a ~60 MB working set.
4. **Refine** the best `candidates` (default 64) by alternating exact
   element-blocked assignment and Kabsch until the permutation stops changing
   (typically 2–4 iterations). Keep the lowest RMSD.

**Element blocking is correctness first, speed second:** permuting a carbon
onto a nitrogen is chemically meaningless, so the assignment is solved
independently per element. That it also turns one n³ problem into Σnₖ³ (≈8×
faster at a typical organic composition) is a bonus.

The method as a whole — RMSD-finder included — is a *heuristic global search*
over a finite trial set. "Matches RMSD-finder" therefore means agreement within
tolerance on fixtures, not provable identity.

### Measured cost

Apple M2, Python 3.9.6, numpy 1.24.4, scipy 1.11.0; median of 3–21 reps after
warmup; compositions ~half hydrogen. Scripts are throwaway (not in the repo).

**Tier 1 / 2 — plain Kabsch:**

| atoms | 20 | 60 | 200 | 1000 | **1462** | **1519** | 4441 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| time | 30 µs | 32 µs | 40 µs | 76 µs | **108 µs** | **104 µs** | 246 µs |

Comfortably below a frame at every size. Tier 2 adds one `(N,3)@(3,3)` matmul.
The bolded columns are Hani's apo/holo/mutant sizes: **~0.1 ms**, four orders of
magnitude inside the interactive budget. Proteins are a solved case here,
because apo/holo/mutant of one protein share a topology and therefore align by
index — tier 1, no permutation search anywhere near them.

**Tier 3, realistic case** (same molecule, rotated + permuted + 0.02 Å noise;
M=2000, K=32; the screen dominates because the first candidate converges and
the bound prunes the rest):

| atoms | 20 | 60 | 200 |
| --- | --- | --- | --- |
| numpy | 12.8 ms | 25.8 ms | 192 ms |

**Tier 3, worst case** (two unrelated conformers of the same formula — a
shallow, near-degenerate landscape where the bound prunes nothing):

| atoms | numpy | scipy | Hungarian solves |
| --- | --- | --- | --- |
| 20 | 54–92 ms | — | 91 |
| 60 | 338–430 ms | 33 ms | 131 |
| 200 | 3.6–4.2 s | 275 ms | 243 |

**Component costs.** Assignment step alone, dense vs element-blocked:

| atoms | dense numpy | dense scipy | blocked numpy | blocked scipy |
| --- | --- | --- | --- | --- |
| 20 | 0.77 ms | 0.003 ms | 0.44 ms | 0.021 ms |
| 60 | 2.44 ms | 0.052 ms | 1.32 ms | 0.030 ms |
| 200 | 14.1 ms | 0.71 ms | 12.9 ms | 0.37 ms |
| 1000 | 281 ms | 47.7 ms | 126 ms | 10.8 ms |

Screen alone (float32, M trial rotations):

| atoms | M=256 | M=2000 |
| --- | --- | --- |
| 20 | 0.5 ms | 3.5 ms |
| 60 | 1.8 ms | 15.5 ms |
| 200 | 13.8 ms | 88.6 ms |
| 1000 | 3.6–4.2 s | 31.7–33.0 s |
| 1500 | — | **42–52 s** (measured at M=16/64, extrapolated) |

The bundled numpy LAPJV was validated against
`scipy.optimize.linear_sum_assignment` on 30 random cost matrices (identical
optimal cost, identical permutation).

### Tier 3 at protein scale — unusable, and refused rather than attempted

At 1500 atoms the screen alone costs **42–52 s** (it is O(M·n²) and n² has grown
5600× from n=20), and a *single* blocked assignment solve costs **288 ms** numpy
/ 28.9 ms scipy — times the ~130 solves a K=32 refinement needs, that is another
**~37 s** numpy or ~3.8 s scipy. Total: **minutes**, numpy or scipy. Reducing M
does not rescue it; M=16 already costs 420 ms of screen alone, and 16 trial
rotations is far too few to find a global minimum.

**Decision: `permute=True` refuses above `permute_max_atoms = 300`** (override
available) with an error naming the measured cost and pointing at the tiers that
do work:

```
permutation search is impractical above 300 atoms (1519 atoms ≈ 1 min+).
Use index correspondence or an atom subset: sset.align(1, onto=0)
```

Refusing beats a viewer that appears to hang. This costs nothing in practice:
permutation search answers "which atom is which when the files disagree", and
apo/holo/mutant of one protein have identical topology, so they align by index
in ~0.1 ms (tier 1). Structures large enough to need tier 3 *and* lacking any
index correspondence are outside vimol's scope.

### Position on scipy

**scipy is not a dependency, and not required for correctness.** `align.py`
attempts `from scipy.optimize import linear_sum_assignment` once at import and
falls back to the bundled numpy LAPJV. Justification from the numbers above:

- Tiers 1 and 2 — every interactive path — contain **no assignment step at
  all**. scipy is irrelevant to them.
- Tier 3 is opt-in and off the interactive path. Worst case at 60 atoms is
  ~0.4 s numpy-only, which is an acceptable "press a key, see a spinner" cost;
  the realistic case is 26 ms.
- Between ~200 and the 300-atom cap the numpy path degrades badly (seconds), and
  there scipy's 10–14× on the assignment step is worth having. Above the cap
  neither is usable, so scipy cannot be the thing that unlocks protein-scale
  permutation search — it is a constant-factor win inside the supported range,
  nothing more. That is precisely why it stays optional.

Add `align = ["scipy>=1.4"]` to `[project.optional-dependencies]`, documented
as *"speeds up the opt-in permutation search on large structures; nothing else
uses it"*. `pip install vimol` remains numpy + moderngl.
*Rejected:* making scipy a hard dependency for a feature 95% of sessions never
invoke, on a project whose selling point is a pure-numpy renderer.

### Defaults and their limits

The candidate-selection study (three unrelated-conformer pairs at each of
n=20/60/200, versus a 200-candidate reference) found **no selection rule that is
uniformly best**: top-K-by-bound, evenly-spread, and a 50/50 mix each missed the
reference minimum on some seed by 0.03–0.15 Å. Selecting candidates purely by
the relaxed bound is therefore *not* reliable on shallow landscapes — the bound
ranks the starting cost, not the basin the refinement converges into. The
defaults are `trials=2000, candidates=64` (double the 32 measured above), both
overridable, with `candidates` drawn as half top-by-bound and half evenly spread
across the trial set. A greedy pre-rank of all candidates followed by exact
refinement of only the best 4 was measured at ~4× faster with identical results
at n=20/60 but a worse minimum at n=200; it is available as `fast=True`, not the
default.

### Validation fixtures

`tests/fixtures/align/*.xyz` plus `expected.json`.

**Objective, stated before the tolerance:** both sides must minimize the same
thing — the **unweighted** RMSD, `sqrt(Σ|R·pᵢ + t − q_π(i)|² / n)`, with no mass
weighting and no per-atom scaling. Confirm this against the oracle's definition
before recording any expected value. A mass-weighted-versus-unweighted mismatch
would show up as a small *uniform* offset across every fixture and would be
misread as a stuck search under the tolerance below; a uniform offset is the
signature to look for.

**Tolerance:** `|rmsd_vimol − rmsd_oracle| ≤ 1e-4 Å`. Both are global searches;
when both find the true minimum they agree to solver precision (~1e-6), and the
observed failure mode is a *stuck search* off by 0.05–0.5 Å. A loose tolerance
would hide exactly the bug the test exists to catch. A one-sided pass is also
allowed: `rmsd_vimol ≤ rmsd_oracle + 1e-4` — finding a better minimum than the
oracle is a pass.

**Fixture set — 16 pairs**, sized 12/20/60/200 atoms, at least half with ~50%
hydrogen:

| n | kind | what it catches |
| --- | --- | --- |
| 3 | identical file, rotated + translated | RMSD → 0; sign/convention errors |
| 3 | conformer or trajectory pairs, same atom order | tier 1 correctness |
| 3 | the same pairs with atom order randomly shuffled | tier 3 must recover the tier-1 RMSD exactly |
| 2 | high symmetry (C60 vs rotated C60, benzene) | deep permutation degeneracy — the classic local-minimum trap |
| 2 | enantiomer pairs | improper rotations must be rejected (det = +1); RMSD stays large |
| 2 | unrelated molecules of the same formula | shallow landscape; the hardest search |
| 1 | flexible tail, fit restricted to the rigid core | tier 2 subset semantics |

**Search-quality guard (runs in CI, no oracle needed):** for every fixture,
assert `rmsd(trials=2000, candidates=64) ≤ rmsd(trials=20000, candidates=512) +
1e-4`. This is how the design claim "the cheap defaults still find the true
minimum" stays honest under refactoring, and it needs no GPL binary.

**Oracle provenance:** expected values are produced **out of tree** by running
the published RMSD-finder on the same fixture files, and checked in as numbers
in `expected.json` with a `provenance` block naming the upstream commit, the
date, and the machine. No GPL source, binary, or build step enters the repo or
CI.

---

## 8. Cross-structure measurement (VIM-6) — index correspondence

**Position: VIM-6 depends on index correspondence, not on the alignment
mapping.** It is unblocked and can be implemented now.

```python
sset.measure(indices) -> List[Tuple[str, Optional[float]]]
```

- `indices` is the active structure's ordered 2/3/4-atom pick list (`measure_sel`).
- An entry is evaluated iff `entry.molecule.symbols == active.molecule.symbols`
  (element-by-element, not just length — a length-only check reports confident
  garbage). Non-matching entries yield `None` and render as `— (topology differs)`
  in the table, satisfying VIM-6's "degrade gracefully" criterion.
- Measurement reads **source coordinates**, deliberately. Distances, angles and
  dihedrals are invariant under a rigid per-structure transform, so the numbers
  are identical whether or not the structures have been aligned. This decouples
  VIM-6 from VIM-4 entirely.

**Extension path, once VIM-4 lands:** `AlignmentResult.mapping` (mobile index →
reference index) supplies the atom correspondence for mismatched topologies.
`sset.measure(indices, via="mapping")` routes the active structure's indices
through each entry's stored mapping, falling back to `None` where no alignment
exists. Ship index correspondence first, as VIM-6's own description asks.

---

## 9. API surface

```python
mol  = vimol.load("caffeine.sdf")             # a str path  -> Molecule (unchanged)
mol  = vimol.load("data.txt", "xyz")          # fmt stays positional (unchanged)
sset = vimol.load(["a.xyz", "b.pdb"])         # a sequence  -> StructureSet
sset = vimol.load_set(["a.xyz"])              # ALWAYS a StructureSet, any count
mols = vimol.load_all("traj.xyz")             # unchanged: List[Molecule]

vimol.view(sset)                              # also accepts Molecule | str | list[str]
scene = vimol.Scene(sset, 640, 480)           # also accepts a bare Molecule
w = vimol.MoleculeWidget(sset, 800, 600)      # also accepts a bare Molecule

sset[0]; sset["b.pdb"]; len(sset); sset.labels; sset.molecules
sset.set_active(1); sset.overlay = True; sset[2].marked = True
sset.solo(0); sset.toggle_visible(1); sset.clear_marks()
res = sset.align(1, onto=0, select=core, permute=False)
print(res.rmsd, res.method)
for label, value in sset.measure([3, 7, 11]):
    print(label, value)
```

`vimol.load` dispatches on the *type* of its first argument — `str` → today's
code path verbatim, sequence → `StructureSet` — and its signature stays
`load(path, fmt=None)`. **It must not become varargs.** `fmt` is positional
today, so `vimol.load("data.txt", "xyz")` means "parse data.txt as xyz"; a
`*paths` form would silently reinterpret that as two files. `loads(text, fmt)`
is untouched and stays single-molecule. `load_set(paths, fmt=None)` is the
recommended form for scripts — no type dispatch to reason about.
*Rejected:* varargs `load(*paths)` (breaks `fmt`), always returning a
`StructureSet` (breaks the documented API), and a separate `load_multi` name
that nobody would find.

New exports in `__init__.py`: `StructureSet`, `Structure`, `Transform`,
`AlignmentResult`, `load_set`, `align`, `select`.

---

## 10. Backwards compatibility

Every README snippet, and why it still works:

| README code | still works because |
| --- | --- |
| `vimol.load("caffeine.sdf")` | str first arg → `Molecule`, unchanged code path |
| `vimol.load(path, "xyz")` (not in README, but public) | signature stays `load(path, fmt=None)`; `fmt` remains positional |
| `vimol.loads(text, "xyz")`, `vimol.save(mol, path)` | untouched, single-molecule |
| `Viewer(mol, frames=[...])`, `viewer.frame_index = k` | `frames=` still accepted; `frame_index` is a property with a setter |
| `vimol.view(mol, editable=True)` | `view` wraps a bare `Molecule` in a 1-entry set |
| `vimol.Scene(mol, 640, 480)` | `Scene.__init__` accepts `Molecule` or `StructureSet` |
| `scene.camera.orbit(30, -15)` | `Camera` is untouched |
| `scene.render()` → (H,W,3) uint8 | composite fast path returns the same object to the same renderer |
| `scene.to_kitty()` / `scene.to_png()` | wrap `render()`, unchanged |
| `MoleculeWidget(mol, w, h)` | accepts `Molecule` or `StructureSet` |
| `InputDecoder` / `MouseEvent` / `widget.handle_mouse(ev, origin=...)` | input layer untouched |
| `widget.to_kitty(cols=, rows=)` | unchanged |
| `examples/embed_demo.py` | uses only the above |
| `vimol file.xyz`, `--frame` | single-file behaviour byte-identical (`--render` / `--kitty` / `--info` were removed from the CLI in VIM-1 — see §2) |
| every existing keybinding (`1`–`4`, `[`/`]`, `h`/`j`/`k`/`l`, `n`/`p`, …) | the strip's colliding keys are focus-scoped, and `[`/`]` are not claimed at all; only `Tab` is new (§4.3) |
| host apps embedding `MoleculeWidget` | the strip lives in `Viewer`, not the widget — an embedder's layout is untouched |
| `Style(...)` constructed by a caller | `flat_mask` is a new field defaulting to `None`, which is exactly today's shading |
| `SphereBatch(...)` / `CylinderBatch(...)` built directly | new `flat` field defaults to zeros; `.empty()` unchanged |

Additionally: `scene.molecule` and `widget.molecule` keep returning the
*editable* `Molecule` (the active structure's), which for a single structure is
the exact object the caller passed in — so `mol is scene.molecule` still holds,
and in-place edits by an embedder still show up. `Viewer(molecule, frames=[...])`
keeps its signature.

Tests: `tests/test_vimol.py` and `tests/test_editing.py` must pass unchanged.
That is the acceptance gate for the container work before VIM-9 or anything
after it starts.

---

## 11. Implementation order

0. **Prerequisite, standalone on `main`:** land the `xyz-multi-struc-toggle`
   branch's `input.py` alt-key decoding by itself (§2, §12.6). Small, isolated,
   and §4.3 depends on it.
1. **`structures.py` + `Composite` + `Scene`/`widget` plumbing, single-entry
   only.** Existing tests pass unchanged — that is the gate. (The VIM-8
   deliverable's code half.)
2. **VIM-9 — single-file multi-molecule (first story).** `load_all` →
   `StructureSet`; the `frames`/`frame_index` property shim; the structure strip
   (§4.1–4.3); flat shading in both backends (§4.5) and overlay colouring
   (§4.4); mark / solo / hide. Real fixtures already exist in `examples/` and
   the PDB/SDF parsers. This is where the container earns its keep.
3. **VIM-1** — CLI `nargs="*"`, per-file parse-failure handling, cross-file
   label disambiguation. Everything the strip needs already exists by now.
4. **VIM-2** — the remaining switch-story polish (status bar, camera-preservation
   behaviour); largely delivered by step 2.
5. **VIM-3** — overlay acceptance criteria (legend, style-option coverage);
   largely delivered by step 2.
6. **`select.py` + `align.py` tiers 1–2 + VIM-4's interactive path.**
7. **VIM-6** — index correspondence. Independent of 6; can run in parallel.
8. **`align.py` tier 3** + the 300-atom cap + fixtures + oracle numbers.
9. **VIM-7** — `load_set`, README library section.

Steps 2–5 deliberately front-load the container's hardest consumer (rendering N
structures) into the story with the fewest moving parts around it.

---

## 12. Decisions taken (was: open questions)

Revision 1 left six questions open. All six are now decided, each to the
simplest available option. Nothing in this document is pending.

**12.1 Named selections → element and geometric only; names are VIM-10.**
`parsers/pdb.py` reads columns 12–16 only to guess the element and then discards
atom names, residues and chains, and `Molecule` carries `symbols` and nothing
else. `select.backbone()` / `select.residue("ALA")` would need new `Molecule`
fields (`atom_names`, `residue_ids`, `chain_ids`) plus parser changes in every
format. **Out of scope here.** `select.py` ships the element and geometric
helpers of §6 only. Name-based selection is **VIM-10**, and this design does not
constrain it: it adds new `select.*` functions returning the same index arrays,
so nothing in §6's plumbing changes when it lands.

**12.2 Per-structure representation → tint only.** One `Style` per `Scene`.
Genuinely mixing representations (one spacefill, one wireframe) needs two render
passes into a shared z-buffer, which neither backend does; not worth it for the
overlay's purpose. `Structure` therefore carries **no `style` field** (it is
dropped from the revision-1 dataclass) — only `tint`, plus the per-atom `flat`
flag of §4.4. Note that flat shading is a *different question* and is **not**
blocked by this: it needs no second pass and is verified expressible per atom in
both backends (§4.5).

**12.3 Editing while overlaid → allowed, active structure only.** Edit modes
stay armed with `overlay` on; picks that resolve to a non-active structure are
ignored with `atom belongs to <label> — Tab to activate` in the status bar
(§3, Highlighting). Disabling editing whenever overlay is on would make the
overlay a mode you have to leave to work.

**12.4 Tier-3 budget → opt-in key with a progress message, no threading.**
`permute=True` is never on the interactive path. The viewer draws
`aligning <label> (permutation search)…` before the call and the result after.
No thread, no cancellation, no partial-progress protocol — a hard 300-atom cap
(§7) bounds the wait to ~1 s numpy-only, and threading the search would drag the
frame-pacing and fence machinery into it for no user-visible gain.

**12.5 Saving aligned coordinates → source coordinates by default.** `s` writes
the structure's own frame. When the active entry has a non-identity transform,
the save prompt shows an extra toggle line — `[ ] bake alignment into
coordinates` — defaulting to off. Round-tripping a file through vimol must not
silently move it.

**12.6 The `xyz-multi-struc-toggle` worktree → it is the seed of VIM-9.** Not a
later rebase during VIM-2: the branch *is* single-file multi-molecule
navigation, which is now the first story (§2). Rebase `cc51164` as the opening
commit of VIM-9 and grow it onto `StructureSet`. Its `input.py` alt-key decoding
lands on `main` first and separately, because §4.3's keymap needs the same
mechanism.

### Notes for the reviewer (not blocking, no decision required)

- **Keymap.** §4.3 introduces exactly one new global binding (`Tab`, strip
  focus) and changes none. That was chosen over relocating `1`–`4`
  (representation) and `h` (orbit-left), which the
  requested list-key set collides with. If a bare, unfocused `1`–`9` is wanted
  instead, four existing bindings need new homes — say so and it is a small
  change to this section, not to the model.
- **Mockup vs. text.** The mockup tints all three structures uniformly; the
  written instruction keeps the first in CPK. This design follows the text
  (§4.4).
