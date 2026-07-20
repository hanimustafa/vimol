# vimol

A molecular viewer and editor that runs **in your terminal**. Open an `.xyz`,
`.pdb`, `.mol`, or `.sdf` file and get a shaded, rotatable, editable 3D
structure right where you're already working — over SSH, in tmux, next to
your editor.

```bash
pip install vimol
vimol molecule.xyz
```

Requires Python ≥ 3.8 and a terminal that speaks the
[Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)
— **kitty, Ghostty, or WezTerm**. Rendering is pure software (numpy impostor
raycasting), with an OpenGL fast path used automatically where a GPU is
available. MIT licensed.

## In the terminal

![C60 spinning in the terminal, rendered by vimol](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/spin.gif)

`vimol` with no file opens this bundled C60 demo. Drag to rotate, scroll to
zoom, hover an atom to identify it, `m` to measure distances/angles/dihedrals.
Editing is on by default: `a` to append (grow fragments, swap elements — the
status-bar pills pick the element and geometry), option-drag to draw bonds,
`x` to delete, `c` to relax clashes, `u` to undo, `s` to save. `?` lists
every binding.

```bash
vimol traj.xyz --spin --style spacefill   # spin a trajectory, space-filling
vimol protein.pdb --render out.png        # batch still to PNG (works headless)
```

## Library usage

```python
import vimol

mol = vimol.load("caffeine.sdf")      # parse; bonds perceived if absent
vimol.view(mol, editable=True)        # full-screen interactive viewer

scene = vimol.Scene(mol, 640, 480)    # ...or drive the renderer yourself
scene.camera.orbit(30, -15)           # rotate (degrees)
img = scene.render()                  # (H, W, 3) uint8 numpy array
os.write(1, scene.to_kitty())         # paint it into a Kitty terminal
scene.to_png("out.png")               # or save a PNG (stdlib encoder, no Pillow)
```

To embed the viewer in your own terminal app, keep the input loop and hand
only the events you want to the widget:

```python
from vimol import MoleculeWidget, InputDecoder, MouseEvent

widget = MoleculeWidget(mol, width_px, height_px)   # no terminal, no loop
decoder = InputDecoder(pixel=True)

for ev in decoder.feed(data):                       # bytes you read from the tty
    if isinstance(ev, MouseEvent) and inside_my_region(ev):
        widget.handle_mouse(ev, origin=(region_x_px, region_y_px))
    else:
        my_app_handles(ev)

os.write(1, widget.to_kitty(cols=region_cols, rows=region_rows))
```

See `examples/embed_demo.py` for a complete host app.
