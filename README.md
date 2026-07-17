# vimol

A molecular viewer that runs **in your terminal**. Open an `.xyz`, `.pdb`,
`.mol`, or `.sdf` file and get a shaded, rotatable 3D structure right where
you're already working — over SSH, in tmux, next to your editor — without a
windowing system, a GPU, or a browser.

```bash
pip install vimol
vimol caffeine.sdf          # rotate with the mouse, hover an atom to identify it
```

vimol renders with the
[Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/),
so it needs a terminal that speaks it — **kitty, Ghostty, or WezTerm**. The
image is drawn entirely in software (pure Python + numpy); OpenGL is an
optional accelerator, not a requirement.

**What you can do with it:**

- **Look at a structure** without leaving the terminal — mouse-drag to rotate,
  wheel to zoom, hover to read off an element and coordinates.
- **Measure** distances, angles, and dihedrals by clicking atoms (`m`).
- **Edit** a molecule interactively — grow fragments, swap elements, draw
  bonds, delete atoms, and relax clashes — then save back to `.xyz`.
- **Render stills to PNG** in scripts or on a headless box (no terminal
  needed for `--render`).
- **Embed the viewer** in your own terminal app — it's a library first, a
  CLI second.

```bash
vimol protein.pdb --render out.png --size 1200x900   # batch still, works headless
vimol trajectory.xyz --spin --style spacefill        # spin a space-filling model
```

```python
import vimol
mol = vimol.load("caffeine.sdf")
vimol.view(mol, editable=True)   # full-screen: mouse rotate/pan/zoom + hover-to-identify
```

## Why impostor raycasting?

To work in *any* Kitty-capable terminal without a GPU, the renderer treats each
atom as an analytic **sphere** and each bond as an analytic **cylinder**, and
solves the ray/surface intersection per pixel under an orthographic camera. A
shared z-buffer resolves occlusion, so spheres and cylinders interpenetrate
correctly. Everything is vectorized over each primitive's screen-space bounding
box, which keeps interactive frame-rates for small/medium molecules in pure
numpy. It is the same "impostor" trick that GPU molecular viewers (PyMOL, VMD,
NGL) use — just done on the CPU.

## Features

- **Formats:** XYZ / extended-XYZ (+ trajectories), PDB (ATOM/HETATM/CONECT,
  multi-model), MOL / SDF (V2000, multi-record).
- **Representations:** ball-and-stick, space-filling (CPK), licorice, wireframe.
- **Chemistry:** covalent/van-der-Waals radii and CPK colors for the whole
  periodic table; automatic distance-based bond perception (spatial-hash, scales
  to large structures).
- **Interaction:** mouse-drag rotate, right-drag pan, wheel zoom, arrow/vim keys,
  roll, autospin, live representation switching, trajectory frame stepping.
- **Pixel-precise mouse:** uses SGR-Pixels (DECSET 1016) when the terminal
  supports it — probed at startup, with automatic fallback to cell coordinates —
  for smooth dragging and accurate **hover-to-identify** atom picking.
- **Rendering:** Phong lighting (key + fill + specular), depth cueing, and an
  optional **transparent background** — RGBA cutout with premultiplied-alpha
  edge anti-aliasing, so the molecule composites straight onto your terminal
  background (any theme, light or dark). On by default in the interactive viewer.
- **Interactive by default:** the software renderer drops resolution while you
  drag and supersamples once you stop, so motion stays responsive on the CPU and
  the resting frame is crisp. An optional OpenGL backend (`vimol[gl]`) renders on
  the GPU where one is available.
- **Fast startup, SSH-aware:** the first image appears in tens of milliseconds —
  capability probing (graphics, pixel mouse, cell size) happens in a single
  round trip *after* the first paint, and detection asks the terminal itself, so
  it works over SSH where environment variables lie. The link's measured latency
  and throughput set the resting-frame resolution automatically: full quality on
  a local terminal, adaptively reduced on slow connections.
- **Output:** live terminal, one-shot Kitty frame to stdout (pipeable), or PNG
  (via a built-in stdlib PNG encoder — no Pillow needed).

## Install

```bash
pip install vimol            # core viewer + CLI (numpy only)
pip install "vimol[gl]"      # + optional OpenGL backend (adds moderngl)
```

Requires Python ≥ 3.8; the only runtime dependency is **numpy**. The base
install does **not** pull in `moderngl` — the GPU backend is opt-in via the
`[gl]` extra, because it needs a working OpenGL driver that headless servers
and many clusters lack, and vimol renders fully on the CPU without it. A
terminal that speaks the Kitty graphics protocol (kitty, Ghostty, WezTerm) is
needed for *interactive* use; `--render` to PNG works anywhere, including
headless.

From a checkout, for development:

```bash
pip install -e ".[test,gl]"      # editable install with test + GL extras
# or run without installing:
PYTHONPATH=src python -m vimol examples/c60.xyz
```

## Library usage

```python
import vimol

mol = vimol.load("caffeine.sdf")     # parse (bonds perceived if absent)
scene = vimol.Scene(mol, 640, 480)   # bind camera + renderer, own the size
scene.camera.orbit(30, -15)            # rotate (degrees)

img = scene.render()                   # -> (H, W, 3) uint8 numpy array
os.write(1, scene.to_kitty())          # ...or paint it in a Kitty terminal
scene.to_png("out.png")                # ...or save a PNG

vimol.view(mol)                      # ...or launch the full interactive widget
```

### Embedding with your own input loop

When you embed the viewer, *your* app owns the terminal and the input loop. Use
the two building blocks directly and forward only the events you want — so you
can **intercept the mouse in your own region** and hand it to the molecule:

```python
from vimol import MoleculeWidget, InputDecoder, MouseEvent

widget = MoleculeWidget(mol, width_px, height_px)   # no terminal, no input loop
decoder = InputDecoder(pixel=True)

# ...in your loop, on bytes you read from the tty:
for ev in decoder.feed(data):
    if isinstance(ev, MouseEvent) and inside_my_region(ev):
        widget.handle_mouse(ev, origin=(region_x_px, region_y_px))  # -> rotate/pan/zoom/pick
    else:
        my_app_handles(ev)                          # the event stays yours

os.write(1, widget.to_kitty(cols=region_cols, rows=region_rows))
idx = widget.hovered   # atom index under the cursor, or None
```

`Viewer` (the full-screen driver) is itself just a thin wrapper over
`MoleculeWidget` + `InputDecoder`. See `examples/embed_demo.py` for a complete
host app that puts its own chrome above the molecule and routes the mouse.

## Interactive controls

| Action | Keys |
|---|---|
| Rotate | mouse drag · arrows · `h j k l` |
| Pan | right-drag · middle-drag |
| Zoom | wheel · `+` / `-` |
| Identify atom | hover the cursor over it |
| Roll | `[` / `]` |
| Representation | `1` ball · `2` space · `3` licorice · `4` wire · `s` cycle |
| Autospin | `a` |
| Measure | `m` toggles; click 2 atoms → distance (Å), 3 → angle, 4 → dihedral; click empty space to reset |
| Trajectory frame | `n` / `p` |
| Transparent background | `t` |
| Depth cue / hi-quality / re-fit / reset | `d` / `g` / `f` / `r` |
| Help / quit | `?` / `q` or `Esc` |

### Editing

The `vimol` CLI always opens with editing on. The Python API (`vimol.view()`,
`Viewer`, `MoleculeWidget`) still defaults to `editable=False` and needs
`editable=True` to opt in — that way a host application that embeds the
widget doesn't inherit these keybindings unless it asks for them. When
editing is on, a few keys change meaning from the classic bindings above
(`a` = autospin, `s` = cycle style):

| Action | Keys |
|---|---|
| Append / edit | `a` toggles; then click an H → grow, click a heavy atom → replace it, click empty space → new methane |
| Delete | `x` toggles; then click an atom to remove it (its terminal hydrogens go with it) |
| Manual bond | option/alt-drag from one atom to another — draws a bond kept even beyond the auto-perception distance |
| Cleanup | `c` — relaxes steric clashes and stretched manual bonds |
| Undo | `u` (reverts the last edit) |
| Save | `s` (prompts for a filename; asks before overwriting an existing file) |
| Autospin | `o` (relocated from `a` while editing) |

In **append** mode (status bar shows `✎APPEND`) a left *click* — not a drag,
which still rotates — builds structure:

- **click a hydrogen** → grow there: the H is promoted to the element you've
  selected and its freed valences are capped with hydrogens (so an H becomes
  a `–CH3` by default).
- **click a heavier atom** → *replace* it with the selected element/geometry.
  Terminal hydrogens snap to the new bond length; an under-coordinated atom
  is topped up with hydrogens to meet the chosen valence, and excess bonds
  are left alone (hypervalent atoms are yours to make).
- **click empty space** → a fresh, fully-capped molecule of the selected
  element is born on the plane through the molecule's center (carbon → methane,
  oxygen → water, etc).

The status bar shows what you're placing as two colored, button-styled tokens —
`adding [ C ] [ tetrahedral ]`. Both are clickable, and each opens a picker
anchored right above the button you clicked (not centered on screen) so it
appears where you're already looking; click the same pill again, click
anywhere outside the panel, or press `Esc`, to dismiss it without changing
anything — or click the *other* pill to switch straight to that picker.

- **Click the element pill** (`[ C ]`) to open a **periodic-table picker**: a
  bordered overlay colored by each element's CPK color. Drive it with the arrow
  keys / `h j k l`, the mouse, or both — click a cell or move to it and press
  Enter. The info line shows the full element name and the geometry it'll build
  with; the lanthanide/actinide gap cells (`**`) jump you down to their own row.
  Choosing a new element resets the geometry to that element's default.
- **Click the geometry pill** (`[ tetrahedral ]`) to open a **geometry picker**
  listing the hybridization/valence options for the current element — e.g.
  carbon offers `tetrahedral · sp3 · 4 bonds`, `trigonal · sp2 · 3 bonds`, and
  `linear · sp · 2 bonds`; nitrogen and oxygen list their own. The active choice
  is marked `●`. Pick one with the arrows/mouse + Enter, and subsequent builds
  use that geometry (so an sp² carbon grows a `–CH2`/`=CH` centre instead of a
  full tetrahedron).

Press `x` to toggle **delete** mode (status bar shows `✗DELETE`; on Kitty-family
terminals the mouse pointer becomes a **crosshair** while it's armed, and the
atom under the cursor previews in red). A left click then removes the atom it
lands on — and, if that atom is a heavy one, its own terminal hydrogens go with
it, so deleting a methyl carbon takes its three H too. Clicking a hydrogen
directly removes just that hydrogen; clicking empty space does nothing. Deletion
is a plain erase — a heavy neighbor that loses a bond is left dangling, with no
hydrogen added to cap it (unlike replace, which tops up valence). `x` and `a`
are mutually exclusive: arming one disarms the other.

Bond perception is distance-based, so freshly-built atoms can appear "bonded"
to whatever they merely landed near, and an option-drag manual bond can be
far longer than a real one. Press `c` to **cleanup**: it relaxes those two
cases with simple springs — pushing steric-clash bonds apart until the false
bond disappears, pulling over-long manual bonds toward a real bond length,
and nudging the angles around the affected atoms back toward their template
geometry — moving the atoms you just placed far more than the rest of the
structure. The relaxation animates over roughly half a second, so you watch
the molecule settle rather than teleport. The status bar hints `⚠ c cleanup`
whenever there's something for it to fix, and one `u` undoes the whole thing
like any other edit.

Any edit marks the model `[MODIFIED]`; `u` undoes step by step. Press `s` to
save — the prompt is pre-filled with the source path, saves straight to a new
file, and asks before replacing an existing one. Pressing `Esc` with unsaved
changes asks `save before quitting? (y/n/Esc)` instead of quitting silently
(`q` stays an immediate quit). Geometry for each element comes
from a small template registry (`templates.py`) keyed by element and valence —
tetrahedral carbon, pyramidal nitrogen, bent oxygen, … — so new templates are a
one-line addition.

## Project layout

```
src/vimol/
├── __init__.py             # public library API
├── widget.py               # MoleculeWidget: embeddable interaction core (+ picking)
├── input.py                # InputDecoder + MouseEvent/KeyEvent, SGR/pixel mouse
├── viewer.py               # full-screen driver (thin wrapper over widget + input)
├── app.py                  # batch/CLI implementation (vimol command)
├── scene.py                # Scene: molecule + camera + renderer + supersampling
├── render.py               # numpy impostor raycaster (spheres + cylinders)
├── camera.py               # orthographic trackball camera
├── kitty.py                # Kitty graphics protocol encoder + terminal geometry
├── molecule.py             # Molecule data model
├── bonds.py                # distance-based bond perception
├── elements.py             # periodic-table radii + CPK colors
└── parsers/                # xyz / pdb / mol-sdf
examples/                   # water, methane, benzene, C60 + embed/host demos
tests/                      # pytest suite
```

## License

MIT.
