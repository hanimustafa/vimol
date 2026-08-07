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
Drag to rotate, scroll to zoom, option-drag empty space to pan, hover an atom
to identify it, `?` for every binding.

You need Python 3.8 or newer and a terminal that speaks the
[Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) —
**kitty, Ghostty or WezTerm**. Rendering is pure software, with an OpenGL path
used automatically where a GPU is available. MIT licensed.

## Proteins out of bare coordinates

![vimol opening a bare xyz protein and switching to the lettered residue view](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/protein.gif)

An `.xyz` file has no residue names, no atom names and no chains. vimol finds
the backbone anyway, opens the file as a ribbon, and on `6` letters every
residue onto it. The trp-cage above is PDB 1RIJ stripped to bare coordinates,
which vimol reads back as `ALQELLGQWLKDGGPSSGRPPPS` — character for character
the sequence in that entry.

## Overlays and RMSD

![vimol overlaying 72 CREST conformers on a reference structure and reporting each RMSD](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/align.gif)

Open several files, or a trajectory, and every structure lands in one session,
overlaid and colour-coded down the left. `r` aligns the whole overlay onto the
active structure and reports each RMSD in a column beside the list — above, a
chignolin crystal structure and 72 CREST conformers of it. `R` runs the same
fit on only the atoms you pick.

## Editing

![vimol growing a methyl group onto threonine's side-chain hydroxyl](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/methyl.gif)

Editing is on by default. Click a hydrogen to grow a fragment in its place, a
heavy atom to swap the element, empty space to start something new — one click
turns threonine into O-methylthreonine. Bonds, deletion, clash cleanup, undo
and save are a keystroke each.

## As a library

![a C60 painted into the middle of a terminal by three lines of vimol](https://raw.githubusercontent.com/hanimustafa/vimol/main/docs/media/library.png)

Three lines put a molecule anywhere on the screen:

```python
import os, vimol

scene = vimol.Scene(vimol.load("c60.xyz"), 640, 380)   # render it off-screen
scene.style.transparent = True                         # let the terminal through
os.write(1, b"\x1b[6;19H" + scene.to_kitty())          # paint at row 6, col 19
```

`scene.render()` hands you the pixels as a numpy array instead and
`scene.to_png()` writes a file. To let the reader turn the molecule,
`MoleculeWidget` takes the input events you hand it and leaves the loop to you;
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
| `Shift+S` | ready-made selections: backbone, heavy atoms, ring system |
| `A` `n` `p` | add a file, next and previous structure |
| `Tab` | focus the structure list |
| `d` `g` `t` `ctrl-t` | depth cue, hi-quality, transparent background, theme |
| `?` `q` | help, quit |

```bash
vimol traj.xyz --spin --style spacefill   # spin a trajectory, space-filling
vimol protein.pdb --style glyph           # lettered residues on a ribbon
vimol a.xyz b.pdb --style ribbon          # overlay two proteins
```

Several files always open in ball-and-stick, so pass `--style ribbon` for
proteins. Also `--backend cpu|gl|auto`, `--theme dark|light|auto`,
`--rotate YAW PITCH`, `--frame N`, `--background`,
`--transparent`/`--opaque`, `--atom-scale`, `--bond-radius`,
`--no-depth-cue`, `--no-bonds`, `--bond-tolerance`, `--list-formats`.
`pip install vimol[align]` adds scipy, which speeds up subset alignment.

The protein, alignment and editing animations are recorded by
`scripts/record_demos.py`.
