# mviewer

A **fully-featured molecular viewer that runs in the terminal**, using the
[Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/).
It renders real 3D molecular structures — shaded spheres and cylinders with a
depth buffer, specular highlights and depth cueing — entirely in software
(pure Python + numpy). No GPU, no OpenGL, no windowing system.

It is built as **two layers**:

* **`mviewer` — an embeddable library.** Drop it into any terminal application
  to render molecules to pixels or a Kitty escape stream, and forward
  intercepted mouse/key events to it to make them interactive.
* **`main.py` (a.k.a. *main.app*) — the driver application.** Deliberately
  tiny: it resolves a file and opens the molecule full-screen with mouse
  control. All real logic lives in the library.

```
./main.py                             # opens the bundled C60, full-screen
./main.py examples/benzene.xyz        # drag to rotate, right-drag to pan, wheel to zoom
```

`main.py` really is this small — everything else is in `src/mviewer/`:

```python
import sys, mviewer
mviewer.view(sys.argv[1])   # full-screen: mouse rotate/pan/zoom + hover-to-identify
```

For batch rendering and flags (stills, styles, PNG), use the `mviewer` CLI
(`python -m mviewer` / the installed console script):

```
mviewer examples/c60.xyz --spin --style spacefill
mviewer protein.pdb --render out.png --size 1200x900
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
- **Rendering:** Phong lighting (key + fill + specular), depth cueing, adaptive
  supersampling (crisp when idle, fast while dragging), and an optional
  **transparent background** — RGBA cutout with premultiplied-alpha edge
  anti-aliasing, so the molecule composites straight onto your terminal
  background (any theme, light or dark). On by default in the interactive viewer.
- **Output:** live terminal, one-shot Kitty frame to stdout (pipeable), or PNG
  (via a built-in stdlib PNG encoder — no Pillow needed).

## Install

```bash
pip install -e .          # then: mviewer examples/c60.xyz
# or run straight from the checkout with no install:
./main.py examples/c60.xyz
```

Requires Python ≥ 3.8 and numpy. A terminal that speaks the Kitty graphics
protocol (kitty, Ghostty, WezTerm) is needed for *interactive* use; `--render`
to PNG works anywhere.

## Library usage

```python
import mviewer

mol = mviewer.load("caffeine.sdf")     # parse (bonds perceived if absent)
scene = mviewer.Scene(mol, 640, 480)   # bind camera + renderer, own the size
scene.camera.orbit(30, -15)            # rotate (degrees)

img = scene.render()                   # -> (H, W, 3) uint8 numpy array
os.write(1, scene.to_kitty())          # ...or paint it in a Kitty terminal
scene.to_png("out.png")                # ...or save a PNG

mviewer.view(mol)                      # ...or launch the full interactive widget
```

### Embedding with your own input loop

When you embed the viewer, *your* app owns the terminal and the input loop. Use
the two building blocks directly and forward only the events you want — so you
can **intercept the mouse in your own region** and hand it to the molecule:

```python
from mviewer import MoleculeWidget, InputDecoder, MouseEvent

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
| Trajectory frame | `n` / `p` |
| Transparent background | `t` |
| Depth cue / hi-quality / re-fit / reset | `d` / `g` / `f` / `r` |
| Help / quit | `?` / `q` or `Esc` |

### Editing

`./main.py` — the standalone app — always opens with editing on. Everywhere
else, editing is **off by default**: the `mviewer` CLI needs `--edit`
(`mviewer --edit mol.xyz`), and the Python API (`mviewer.view()`, `Viewer`,
`MoleculeWidget`) needs `editable=True`. That way a host application that
embeds the widget doesn't inherit these keybindings unless it opts in. When
editing is on, a few keys change meaning from the classic bindings above
(`a` = autospin, `s` = cycle style):

| Action | Keys |
|---|---|
| Append / edit | `a` toggles; then click an atom → grow a carbon, click empty space → new methane |
| Undo | `u` (reverts the last edit) |
| Save | `s` (prompts for a filename; asks before overwriting an existing file) |
| Autospin | `o` (relocated from `a` while editing) |

In **append** mode (status bar shows `✎APPEND`) a left *click* — not a drag,
which still rotates — builds structure:

- **click an atom** → grow the structure there. Clicking a **hydrogen** promotes
  it to a carbon and caps its freed valences with hydrogens (so an H becomes a
  `–CH3`); clicking a heavier atom attaches a new methyl at a free site.
- **click empty space** → a fresh methane is born on the plane through the
  molecule's center.

The status bar shows what you're placing as colored, button-styled tokens —
`adding [ C ] [ tetrahedral ]`. (Clicking those buttons to change the element or
geometry is coming in a later commit; for now they're the visual indicator.)

Any edit marks the model `[MODIFIED]`; `u` undoes step by step. Press `s` to
save — the prompt is pre-filled with the source path, saves straight to a new
file, and asks before replacing an existing one. Geometry for each element comes
from a small template registry (`templates.py`) keyed by element and valence —
tetrahedral carbon, pyramidal nitrogen, bent oxygen, … — so new templates are a
one-line addition.

## Project layout

```
main.py                     # main.app — very thin full-screen launcher
src/mviewer/
├── __init__.py             # public library API
├── widget.py               # MoleculeWidget: embeddable interaction core (+ picking)
├── input.py                # InputDecoder + MouseEvent/KeyEvent, SGR/pixel mouse
├── viewer.py               # full-screen driver (thin wrapper over widget + input)
├── app.py                  # batch/CLI implementation (mviewer command)
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
