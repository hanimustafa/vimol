# mviewer

A **fully-featured molecular viewer that runs in the terminal**, using the
[Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/).
It renders real 3D molecular structures — shaded spheres and cylinders with a
depth buffer, specular highlights and depth cueing — entirely in software
(pure Python + numpy). No GPU, no OpenGL, no windowing system.

It is built as **two layers**:

* **`mviewer` — an embeddable library.** Drop it into any terminal application
  to render molecules to pixels or to a Kitty escape stream.
* **`main.py` (a.k.a. *main.app*) — the driver application.** A full CLI that
  opens files, renders stills, and runs the interactive viewer.

```
./main.py examples/c60.xyz            # interactive buckyball
./main.py examples/benzene.xyz --spin --style spacefill
./main.py protein.pdb --render out.png --size 1200x900
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
- **Rendering:** Phong lighting (key + fill + specular), depth cueing, adaptive
  supersampling (crisp when idle, fast while dragging), flicker-free
  double-buffered animation.
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

See `examples/embed_demo.py` for embedding inside your own terminal UI.

## Interactive controls

| Action | Keys |
|---|---|
| Rotate | mouse drag · arrows · `h j k l` |
| Pan | right-drag |
| Zoom | wheel · `+` / `-` |
| Roll | `[` / `]` |
| Representation | `1` ball · `2` space · `3` licorice · `4` wire · `s` cycle |
| Autospin | `a` |
| Trajectory frame | `n` / `p` |
| Depth cue / hi-quality / re-fit / reset | `d` / `g` / `f` / `r` |
| Help / quit | `?` / `q` or `Esc` |

## Project layout

```
main.py                     # main.app — the driver CLI launcher
src/mviewer/
├── __init__.py             # public library API
├── app.py                  # CLI implementation
├── viewer.py               # interactive terminal widget (raw mode, input loop)
├── scene.py                # Scene: molecule + camera + renderer + supersampling
├── render.py               # numpy impostor raycaster (spheres + cylinders)
├── camera.py               # orthographic trackball camera
├── kitty.py                # Kitty graphics protocol encoder + terminal geometry
├── molecule.py             # Molecule data model
├── bonds.py                # distance-based bond perception
├── elements.py             # periodic-table radii + CPK colors
└── parsers/                # xyz / pdb / mol-sdf
examples/                   # water, methane, benzene, C60 + demos
tests/                      # pytest suite
```

## License

MIT.
