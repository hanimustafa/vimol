# A rewritten README, with recorded terminal demos

The README today is one long block of prose. It documents the program
accurately and reads like a changelog: everything vimol can do, in the order it
was built, with no picture of what using it is actually like. This replaces it
with a short introduction and four recorded demonstrations — three animated,
one still — each showing one thing end to end.

## The demos

**A. A protein arrives as an `.xyz`.** `vimol 1rij.xyz` — 334 atoms, no residue
names, no atom names, nothing but elements and coordinates. It opens as a green
backbone ribbon (style `5`) because vimol recovered the peptide backbone from
the bond graph, and `6` turns it into the lettered side-chain view. This is the
capability that is hardest to believe from prose and the easiest to show.

**B. One structure against a conformer set.** `vimol 5awl_prepared.xyz
crest_ensemble.xyz` loads chignolin's prepared crystal geometry alongside a
CREST ensemble, the `ALL` control on the structure list pulls every conformer
into the overlay, and `r` rigidly aligns them all onto the reference and reports
each RMSD in the table beside the list.

**C. Growing a methyl group.** Threonine, `a` for append mode, click the
hydroxyl-bearing carbon's hydrogen to grow a carbon in its place, and the three
filling hydrogens arrive with it.

**D. Library usage.** Three lines that put the viewer in the middle of a host
program's screen, and a still of the result.

## Why the recordings have to be synthesized

vimol paints its molecule with the Kitty graphics protocol — the pixels travel
as escape codes, not as text cells. That rules out the ordinary terminal
recorders: asciinema captures the bytes faithfully but `agg` renders only
cells, and VHS's built-in emulator has no graphics protocol at all. Screen
recording the real thing is also unavailable here: `screencapture` fails with
*could not create image from display*, because the process holds no Screen
Recording permission and cannot grant itself one.

So `scripts/termshot.py` renders the frames itself, from the bytes the program really
emits:

1. **Drive.** Construct a `Viewer` with `fd_out` pointed at a file rather than a
   tty, with `COLUMNS`/`LINES` set (`kitty.terminal_size_px` falls back to the
   environment when the `TIOCGWINSZ` ioctl fails on a non-tty) and `_cell_px`
   pinned. Feed it synthetic `KeyEvent`/`MouseEvent` objects through
   `Viewer._dispatch`, exactly the objects `InputDecoder` would have produced,
   and call `_draw()` for each frame we want.
2. **Replay.** Split the captured stream into kitty graphics APC blocks and
   everything else. The everything-else is a very small dialect — a spike over
   a real frame found only `CUP`, `EL`, and `SGR` 0/1/22/38;2/48;2 — which
   `pyte` already implements correctly, so the text grid comes from `pyte`
   rather than from a parser of our own. The APC blocks are parsed here:
   `a=T,f=32,o=z` means a zlib-compressed RGBA payload, chunked with `m=1`,
   placed at the cursor and scaled into `c`×`r` cells.
3. **Rasterize.** Paint the image layer first (it carries `z=-1200000000`, so it
   sits behind the text), then the cells over it. A cell with a default
   background must stay transparent or it will punch a hole in the molecule;
   only cells carrying a real background or a glyph get painted. Text is Menlo
   via Pillow, and the cell metrics we hand the `Viewer` are derived from
   Menlo's own advance and line height so that the image lands flush against
   the status bar.
4. **Encode.** PNG frames to GIF through ffmpeg's `palettegen`/`paletteuse`.

Three details decide whether the output is honest or subtly wrong:

- **Pin the resolution controllers.** `_interact_scale`, `_idle_scale` and the
  supersample factor adapt to measured frame times and to fence
  acknowledgements that a file descriptor will never send. Left alone they
  record the blur ramp instead of the picture. Pin all three and set
  `_paced = False`.
- **Resample the payload.** `encode_image` is called with `c`/`r`, which asks
  the terminal to scale the image into that cell box; the payload's own pixel
  size varies with the render scale. Resample to `c*cell_w × r*cell_h` rather
  than blitting one to one.
- **Synthesize clicks by projection, not by guess.** Demos B and C click on
  particular atoms. Project the atom through `scene.camera` to a pixel and send
  a pixel-mode `MouseEvent`, so the click lands where the atom is rather than
  where it looked like it was.

The recorder also draws the parts a terminal session has that the program does
not emit: a window frame with a title, a shell prompt with the command typed
one character at a time, and a pointer sprite for the steps that use the mouse.

## Demo inputs

`1rij.xyz` and `5awl_prepared.xyz` are conversions of PDB entries 1RIJ and 5AWL
and are vendored into `docs/media/demo/`, so demos A and B can be re-recorded
from a fresh checkout. Threonine is built by `examples/build_examples.py`,
which exists precisely to generate exact geometry with no external data.

The CREST ensemble in demo B is unpublished output from the author's own work
and is **not** vendored. `scripts/record_demos.py` reads its path from `VIMOL_DEMO_ENSEMBLE`
and skips demo B when it is unset, so the repository stays reproducible without
the recorder making a data release on the author's behalf.

## Library usage

The three lines in section D use the public API as it stands — `load`, `Scene`,
a cursor position, `to_kitty` — rather than a new `center=` convenience added to
make a snippet shorter. Centring is two integers of arithmetic against
`kitty.terminal_size_px`, and inventing public API to win a line in a README is
the wrong trade to make unattended.
