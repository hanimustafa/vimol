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
zoom, hover an atom to identify it, `m` to measure distances/angles/dihedrals
— with more than one structure loaded, a live column tracks it in a
comparison table next to the structure list, evaluated for every loaded
structure at once; move on to a different measurement (or a different frame)
and it locks in as its own column (click a column's `×` to remove it).
With an overlay up, `r` rigidly aligns every tinted structure onto the
untinted main one and reports each RMSD (`r` and `R` only align in overlay
mode, and say so otherwise). To fit on part of a structure instead, press `R`
and click atoms on the main structure (or option-click an atom to jump
straight into picking without `R`), then `Enter` or `r`. vimol finds the
matching element-compatible subset in each tinted molecule and aligns the
whole molecule from that fit (`Esc` cancels). Picking works before an overlay
exists too — select atoms, `opt+click` a row to overlay it, then `r`.
The fit stays beside the structure list as a `⊂RMSD #select1` column: hover
its header to see which atoms it uses, click it to arm that selection again
(clicking a second time disarms it), and press `R` to recalculate the same
column — `×` removes it. Editing the main frame marks a saved column `*` and
disarms it, since its numbers no longer describe the geometry on screen.
Below the structure list, click the highlighted `select` hint (or press
`Shift+S`) for the **Backbone** and **Backbone + Cβ** presets, or **Manual**
to keep clicking atoms yourself. Presets act on the untinted main frame only:
PDB `ATOM` names are used directly, other formats fall back to a bond-graph
`N–Cα–C(–O)` motif detector. Two PDB structures that share residue/atom
identity skip the search and fit directly. `pip install vimol[align]` adds
scipy, which speeds up the permutation search; without it a numpy fallback is
used.
Editing is on by default: `a` to append (grow fragments, swap elements — the
status-bar pills pick the element and geometry), option-drag to draw bonds,
`x` to delete, `c` to relax clashes, `u` to undo, `s` to save. `?` lists
every binding.

`1`–`4` switch between ball-and-stick, space-filling, licorice and wireframe.
On a protein, `5` reads it the way a figure does instead: the backbone runs as
a ribbon through the Cα atoms, and each residue links out from its Cα by way
of its Cβ to a solid carrying its one-letter code — a plate cut to the shape of
the ring for the aromatics, a rounded volume built from the real side-chain
carbons for everything else. Only the carbon skeleton is abstracted: the
carboxylate, the hydroxyl, the indole N–H and any hydrogen sitting on one of
those stay real atoms in their element colours, at the exact coordinates in the
file, with hairlines between the backbone amides close enough to be hydrogen
bonded. Overlaid structures tint flat, as in every other style. It needs
residue names, so it wants a PDB; anything else stays ball-and-stick and says
so.

This is the one style that looks materially better on a GPU. There the ribbon
is a real swept tube with a rounded edge and the tablets are chamfered solids
with their letters printed onto the faces — and it draws in about 2 ms where
the raycaster takes 100. Without a GL context it still works: the raycaster
intersects the same shapes analytically and fakes the smooth shading, which
costs the polish rather than the picture.

Pass more than one file (`vimol a.xyz b.pdb`) and they all load into one
session, auto-overlaid — the first structure of each file shown together, the
active one in normal element colours and the rest flat-tinted so you can tell
them apart. Multi-model files keep every frame loaded; `opt+click` a row in the
structure list to add one into the overlay or drop it out.

```bash
vimol traj.xyz --spin --style spacefill   # spin a trajectory, space-filling
vimol protein.pdb --style glyph           # lettered residues on a ribbon
vimol a.xyz b.pdb                         # load both, auto-overlaid for comparison
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
