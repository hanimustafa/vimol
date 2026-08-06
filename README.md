# vimol

A molecular viewer and editor that runs **in your terminal**. Open an `.xyz`,
`.pdb`, `.mol` or `.sdf` file and get a shaded, rotatable, editable 3D
structure right where you are already working — over SSH, in tmux, beside your
editor.

```bash
pip install vimol
vimol molecule.xyz
```

![C60 spinning in the terminal, rendered by vimol](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/spin.gif)

That is the C60 bundled with vimol, which `vimol` opens with no arguments.
Drag to rotate, scroll to zoom, hover an atom to identify it, `?` for every
binding.

You need Python 3.8 or newer and a terminal that speaks the
[Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) —
**kitty, Ghostty or WezTerm**. Rendering is pure software (numpy impostor
raycasting); if a GPU context can be created, an OpenGL path is used
automatically instead. MIT licensed.

## Open a protein that does not know it is one

![vimol opening a bare xyz protein and switching to the lettered residue view](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/protein.gif)

An `.xyz` file carries elements and coordinates and nothing else — no residue
names, no atom names, no chains. Hand vimol one anyway. It looks for the
`N–Cα–C(=O)` motif in the bond graph, and if it finds a backbone the file opens
as a cartoon ribbon rather than as a thicket of sticks.

Press `6` and each residue is drawn the way a figure would draw it: the
backbone runs as a ribbon through the Cα atoms, and every side chain links out
from its Cα to a solid carrying its one-letter code — a plate cut to the shape
of the ring for the aromatics, a rounded volume built from the real side-chain
carbons for the rest. Only the carbon skeleton is abstracted. The carboxylate,
the hydroxyl, the indole N–H and any hydrogen on one of those stay real atoms
in their element colours, at the exact coordinates in your file.

Residue identity is recovered from geometry alone. Walking outward from each Cα
gives every atom its bond distance from that Cα, which is precisely what the
Greek letter in a PDB atom name records; the resulting (element, distance)
signature identifies all twenty residues uniquely. The trp-cage above is PDB
1RIJ stripped to bare coordinates, and vimol reads it back as
`ALQELLGQWLKDGGPSSGRPPPS` — character for character the sequence in that
entry's `SEQRES`. Something with no peptide backbone in it stays
ball-and-stick and says so.

## Line one structure up against a conformer set

![vimol overlaying 72 CREST conformers on a reference structure and reporting each RMSD](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/align.gif)

Pass more than one file and they load into a single session, auto-overlaid: the
active structure in normal element colours, the rest flat-tinted so you can
tell them apart. Multi-model files — trajectories, NMR ensembles, multi-record
SDFs — keep every frame, listed down the left. Several files always open in
ball-and-stick, so add `--style ribbon` when they are proteins, as above.

Clicking a file's `ALL` control pulls its whole ensemble into the overlay.
Then `r` rigidly aligns every tinted structure onto the untinted one and
reports each RMSD in a `∀RMSD` column beside the list. Above, that is a
chignolin crystal structure and 72 CREST conformers of it, landing between
2.1 Å and 3.1 Å. Press `f` to re-fit the view once they are all in one place.

To fit on part of a structure instead, press `R` and click atoms on the main
structure, then `Enter`. vimol finds the matching element-compatible subset in
each tinted molecule and aligns the whole molecule from that fit, saving it as
a `⊂RMSD` column you can re-arm later. `Shift+S` offers ready-made selections —
backbone, backbone + Cβ, heavy atoms, largest ring system — instead of clicking
atoms yourself.
`pip install vimol[align]` adds scipy, which speeds up the permutation search;
without it a numpy fallback does the same work.

## Build on a structure

![vimol growing a methyl group onto threonine's side-chain hydroxyl](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/methyl.gif)

Editing is on by default. `a` enters append mode, and from there clicking a
hydrogen grows a new atom in its place, with its own filling hydrogens at
sensible geometry — one click turns threonine into O-methylthreonine. Clicking
a heavy atom replaces it, and clicking empty space starts a new molecule. The
pills in the status bar choose which element and which geometry you are
building with.

Option-drag from one atom to another draws a bond the automatic perception
would not have made, `x` deletes, `c` relaxes clashes, `u` undoes and `s`
saves.

## Use it as a library

![a C60 painted into the middle of a terminal by three lines of vimol](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/library.png)

Three lines put a molecule anywhere on the screen. Build a `Scene` at the pixel
size you want, park the cursor where its top-left corner belongs, and write:

```python
import os, vimol

scene = vimol.Scene(vimol.load("c60.xyz"), 640, 380)   # render it off-screen
scene.style.transparent = True                         # let the terminal through
os.write(1, b"\x1b[6;19H" + scene.to_kitty())          # paint at row 6, col 19
```

That is `examples/inset.py`, and the picture above is what it prints.
`scene.render()` hands you the pixels as a uint8 array instead — `(H, W, 3)`,
or `(H, W, 4)` with a transparent background as above — and
`scene.to_png("out.png")` writes a PNG with a stdlib encoder, no Pillow.

For a molecule the user can actually turn, keep your own input loop and give
the widget only the events you want it to have:

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

`examples/embed_demo.py` is a complete host application built that way.

## Reference

| Key | Does |
|---|---|
| Drag / arrows / `hjkl` | rotate |
| Wheel, `+` `-` | zoom |
| Right or middle drag | pan |
| `[` `]` | roll |
| `1` `2` `3` `4` | ball-and-stick, space-filling, licorice, wireframe |
| `5` `6` | ribbon, lettered residues (proteins) |
| `f` `z` | re-fit, reset the view |
| `m` | measure — click 2, 3 or 4 atoms for distance, angle, dihedral |
| `a` `x` `c` `u` `s` | append, delete, cleanup clashes, undo, save |
| `r` `R` | align an overlay, or align on picked atoms |
| `A` `n` `p` | add a file, next and previous structure |
| `Tab` | focus the structure list |
| `d` `g` `t` `ctrl-t` | depth cue, hi-quality, transparent background, theme |
| `?` `q` | help, quit |

```bash
vimol traj.xyz --spin --style spacefill   # spin a trajectory, space-filling
vimol protein.pdb --style glyph           # lettered residues on a ribbon
vimol a.xyz b.pdb                         # load both, auto-overlaid
```

Also `--backend cpu|gl|auto`, `--theme dark|light|auto`, `--rotate YAW PITCH`,
`--frame N`, `--background`, `--transparent`/`--opaque`, `--atom-scale`,
`--bond-radius`, `--no-depth-cue`, `--no-bonds`, `--bond-tolerance`,
`--list-formats`.

The protein, alignment and editing animations are recorded by
`scripts/record_demos.py`, which drives a real viewer and rasterizes the bytes
it writes; `docs/media/demo` holds their inputs.
