# The `glyph` skin: a lettered, diagrammatic protein representation

Date: 2026-08-05
Status: implemented, then revised repeatedly against further references. Where
the build differs from the plan below, this document records what was built and
why -- see the numbered passes at the end, which supersede the body.

## Goal

A fifth representation, selected with `5` alongside the existing `1`–`4`, that
reads a protein the way a figure in a paper reads it rather than the way a
ball-and-stick model does:

- the peptide backbone drawn as a **ribbon**,
- one **stick** per residue from its Cα out to a solid,
- that solid a **shape that keeps the geometry of the side chain**, carrying the
  residue's one-letter code,
- aromatic side chains getting a **flat plate cut to the shape of the ring**,
  everything else a **rounded volume** built from the real atom positions,
- tiny **red and blue nodes** at the exact file coordinates of the atoms that can
  accept or donate a hydrogen bond,
- hairline **links** between backbone donor/acceptor pairs that are actually
  within hydrogen-bonding distance.

## What has to exist that does not exist yet

vimol's raycaster knows three primitives: sphere, cylinder, cone. Nothing in it
is flat, and nothing carries text. The skin therefore needs two new primitives
and one new source of chemistry.

### 1. Residue names are not parsed

`parsers/pdb.py` captures `atom_names`, `atom_is_hetatm` and `atom_keys` but
never columns 18–20, the residue name. Both the letter and the
aromatic/non-aromatic branch depend on it.

Add a new atom-aligned `Molecule.atom_resnames: List[str]`. Do **not** widen
`atom_keys`: `align.py` and `structures.py` use its exact format for the
residue-identity fast path, and reformatting it would break that and its tests.
Every site that already carries `atom_names` must carry the new list too —
`Molecule.add_atom`, `editor._delete`, `widget`'s undo snapshot, and
`StructureSet._build_topology` — or the field goes ragged the first time a user
deletes an atom.

Formats without residue names (xyz, mol, sdf) get no glyph scene. The skin
degrades to ball-and-stick and says so in the status bar rather than drawing
garbage.

### 2. A convex-polyhedron primitive

One primitive covers both the ribbon and the ring plates: a convex solid given
as a set of half-spaces `n·(p − c) ≤ d`.

Under this renderer's orthographic camera every ray is axis-aligned in view
space, which collapses the intersection to something a single vectorized pass
over a bounding box can do. For a pixel at view-space `(X, Y)` the solid covers
the depths `Z` satisfying every plane. A plane with `n_z > 0` gives an upper
bound on `Z`, one with `n_z < 0` a lower bound, and one with `n_z ≈ 0` is a
pure 2-D mask on `(X, Y)`. The visible surface is `Z = min(upper bounds)`, valid
where it clears every lower bound and every mask; the plane that produced that
minimum is the surface normal. No tessellation, no per-triangle loop.

Storing a solid as `(center_world, normals_world, offsets)` keeps the per-frame
cost to one `(M,3) @ (3,3)` rotation of all planes in the scene at once —
offsets are rotation-invariant.

An oriented box is the six-plane case, so ribbon segments and ring plates share
one rasterizer.

A half-space set does not hand over a bounding box, and the rasterizer needs
one. Each solid therefore carries the oriented box that encloses it — three
orthonormal axes and a half-extent along each — which projects to a tight
screen rectangle. A bounding sphere would be simpler but asks for a square
several times the pixels a ribbon segment actually covers, and every one of
them gets shaded.

### 3. Screen-aligned letter sprites

Letters go on billboards at each solid's centroid, not as decals mapped onto the
plate faces. One mechanism serves every shape, the letter stays legible from any
camera angle — which is the whole point of a labelled diagram — and it needs no
UV math.

The glyphs are stroked, not bitmapped. Each is a few polylines in a unit box and
the rasterizer inks every pixel within a stroke half-width of one, which keeps
the letters smooth and evenly weighted at any size. A 5×7 bitmap was tried first
and read as pixel art beside the shaded solids — the chunkiness is the grid, not
aliasing, so no amount of supersampling fixes it. Either way they are hand-drawn:
the `README` advertises a stdlib PNG encoder and no Pillow, so no font library.

## Geometry

A new `glyphs.py` builds a `GlyphScene` from a molecule: flat arrays of spheres,
cylinders, polyhedra and labels, in world space, cached on the molecule's
revision.

**Residues.** Atoms group by `(chain, resseq, icode)` read out of `atom_keys`.
Each residue exposes `N`, `CA`, `C`, `O` and a side-chain set (everything else
that is not hydrogen and not `OXT`). Residues without a Cα are skipped.

**Ribbon.** Per Carson–Bugg: for the peptide plane between Cα(i) and Cα(i+1),
the chain direction is `A = CA(i+1) − CA(i)` and the in-plane side vector is
`D = normalize(A × (O(i) − CA(i)))`. `D` flips almost 180° between consecutive
residues of a β-strand, so each `D` is negated when it opposes its predecessor.
Centers and sides are then run through a Catmull–Rom spline at eight samples per
residue, and each consecutive pair of samples becomes one solid: long axis along
the segment, 1.6 Å wide along the interpolated side vector, 0.26 Å thick.

The two end caps are mitred — their normals are the averaged tangents at the
joints rather than the segment's own — so a segment and its neighbour end on the
identical plane. Square caps leave a wedge-shaped notch on the outside of every
bend, and the notches do not shrink with sampling density the way one might
hope: at eight segments per residue they turned the ribbon into a row of loose
slats. A joint sharper than about 60° mitres to a plane nearly parallel to the
ribbon, which is unbounded rather than merely ugly, so those square off instead.

**Aromatic plates.** For `PHE`, `TYR`, `TRP` and `HIS` the ring atoms are the
named ring set for that residue. Their best-fit plane comes from the smallest
singular vector of the centered ring coordinates; a monotone-chain convex hull
in that plane (no scipy — it is an optional extra, `vimol[align]`) gives the
outline, which is pushed out 0.30 Å and extruded ±0.11 Å. The resulting solid is
the hull edges as side planes plus two caps, so a six-ring reads as a hexagon
and tryptophan's fused system as its real fused outline. Pushing two edges out
and intersecting them moves a sharp corner further than the inflation itself, so
the enclosing box is measured on the inflated outline's own corners rather than
on the ring atoms plus the inflation.

**Rounded volumes.** Every other side chain is a union of spheres of radius
0.80 Å at its real *carbon* positions (see "Second pass" — originally at every
heavy atom, at 0.85 Å) — over half a bond length, so neighbouring spheres merge
into one solid, but only just, so a two-carbon side chain still reads as two
lobes rather than a ball. The shape is then literally the geometry of
the side chain, and it costs nothing new in the renderer. Glycine, which has no
side chain, gets a single small sphere at the position its Cβ would occupy
(built from the tetrahedral completion of N, C and Cα); alanine gets one sphere
at Cβ.

**Sticks.** A rod from Cα to the solid's anchor: the plate center for
aromatics, the side chain's centroid otherwise. (Superseded — see "Second pass":
the link now runs Cα bead → rod → Cβ bead → rod → glyph.)

**Nodes.** *Superseded — see "Second pass": the abstract nodes are gone and the
atoms themselves are drawn instead.* They were classified per `(residue, atom
name)` rather than from attached hydrogens, since most PDB files have none:
backbone `N` donates (except proline), backbone `O` and `OXT` accept, and a
side-chain table covers the rest. That table survives, and still decides which
backbone pairs the hairlines join.

**Links.** Backbone `N`···`O` pairs closer than 3.35 Å and at least two residues
apart get a 0.05 Å cylinder in a muted gold. Distance alone, on the heavy atoms:
these files usually have no hydrogens, so there is no N–H···O angle to test.

## Renderer plumbing

- `Style.representation` gains `"glyph"`. `_atom_radii` returns zeros for it and
  `draw_bonds` is false, so the underlying atoms and bonds disappear; the glyph
  scene is drawn in their place.
- The numba fast path in `render.render` returns before `draw_band` and knows
  nothing about polyhedra or labels. Its gate must exclude `"glyph"`, or the
  skin would render on machines where numba failed to compile and vanish on
  every other one.
- The GL backend converts a molecule to spheres/cylinders/cones and would
  silently drop everything else. `Scene.render` therefore routes `"glyph"` to a
  lazily built CPU `Renderer`, kept at the same size as the GL one — and that
  branch must apply `_downsample`, which the GL path skips because the GPU does
  it.
- `Scene._max_atom_radius` decides the fit extent and branches on
  representation; ribbon and plate geometry reaches well past `vdw × 0.25`, so
  `"glyph"` needs its own value or activating the skin clips the structure. It
  asks the scene for its actual reach from the centroid rather than using a
  padding constant, which makes the framing exact and leaves no number to guess.
  A molecule with no residues must fall back to the ball-and-stick value too, or
  the fallback quietly reframes itself.
- Flat faces need a matte finish. A sphere's normals turn through the specular
  highlight, so it stays a small bright spot; a planar facet crosses it all at
  once and flares to white together, which turned the ribbon into a row of
  mirrors. `shade_write` takes a per-primitive specular scale for this.
- The thinnest cylinders vimol draws are here. Every rasterizer shades its whole
  bounding box and writes only the pixels that pass coverage and depth; outside
  a cylinder the "normal" grows as segment length over radius, and raising that
  to the shininess overflows float32 — in numbers discarded a line later, but
  noisily enough to spam a terminal. The glyph band draw suppresses it.
- `widget.REPRESENTATIONS` gains a fifth entry and `widget.handle_key` accepts
  `"5"`. Nothing else claims that key: the structure strip deliberately leaves
  the digits alone.
- `app.py`'s `--style` choices and the help panel's `1 2 3 4` row both grow.

## Verification

`examples/` has no protein, so a β-hairpin with real residue names goes in
`tests/data/` — tryptophan zipper 2 (PDB `1LE1`, model 1, hydrogens stripped),
twelve residues covering four aromatics, a glycine and a turn. Verification is visual as well as structural:
render to PNG and look at it, because an assertion on array shape passes just as
happily with the ribbon inside out. Render once with the numba kernel warm and
once with it disabled, since the fast-path gate is invisible in a single run.

Unit tests cover residue grouping, the donor/acceptor table, ribbon frame
continuity (no 180° flip between strand residues), the plane-set of a box, the
convex hull, and the fallback when a molecule has no residue names. Two
properties earn tests of their own because breaking them fails invisibly rather
than loudly: that every solid fits inside the box the renderer shades, and that
each letter is lifted clear of the whole reach of the solid it names.


## Second pass

A second reference and three notes changed four things.

**The ribbon shades smoothly.** Each segment carries the averaged normals of
the joints at its two ends and interpolates between them along its length. That
is vertex-normal averaging: the geometry stays faceted, the shading runs
continuously across the joins, and the band reads as one curved surface. Two
traps, both of which draw a hairline dark crease at every joint and neither of
which is obvious from the code:

- The interpolation must be scaled by the segment's true half-length, not by the
  enclosing box's half-extent along the same axis. The mitre makes the second
  larger, which leaves the interpolation short of 0 and 1 at the real ends, so
  normals either side of a joint disagree.
- The end caps need the same treatment as the faces. They are internal — one
  segment's cap sits against the next one's — and left with their own edge-on
  normals they shade nearly black. Which side of the ribbon a hit is on comes
  from which side of the slab it landed, not from comparing normals: a cap is
  perpendicular to the surface it has to blend into.

The GPU was on the table for this and is not needed for it. Smooth shading is a
per-pixel normal, not a mesh pipeline. A GL port remains the way to make the
skin *fast* — it is the outstanding speed work, and it is not done.

**The ribbon runs through the Cα atoms.** Peptide-plane midpoints smooth
marginally better and sit up to an angstrom off every Cα, so each residue's link
launched from beside the ribbon rather than out of it. The side vector is still
per peptide plane, now averaged onto each residue.

**The link runs Cα bead → rod → Cβ bead → rod → glyph**, tinted by residue
class, rather than one rod skewering the Cβ on the way past.

**Atoms the glyphs do not stand for are drawn as themselves.** A solid is built
from the side chain's *carbon* skeleton only; the carboxylate, the hydroxyl, the
indole N–H and any hydrogen bonded to one of those is a real atom in its element
colour, bonded, at the exact file coordinates. This replaces the abstract
red/blue hydrogen-bond nodes, which sat at those same coordinates in those same
colours — the reference's legend relabelled them "oxygen" and "nitrogen", which
is the whole change. The gold hairlines stay: they are detected interaction, not
atoms, and nothing else conveys them.

Two boundaries had to be drawn by rendering it and looking, not by reasoning:

- The backbone amide stays the ribbon's business. Drawn as atoms, N and O put a
  blue and a red dot on every residue of the ribbon.
- Hydrogens show only on the drawn non-carbon atoms. An NMR model carries every
  C–H, and drawing those buries the skin under a hundred white spheres.

**Overlaid structures tint.** A residue whose atoms are flagged flat paints every
primitive it emits in its entry's tint, exactly as the other representations do.
Two things this needed: the cache key had to learn about tint and flatness, since
marking a structure into an overlay moves no atom and changes no count; and
residues had to become runs of consecutive atoms sharing an identity rather than
a dictionary keyed on it, or two overlaid copies of one file fold together and
the second gets no glyphs at all.


## Third pass: proline, and the GPU

**Proline is a tablet.** It is the one residue whose side chain closes a ring
without being aromatic, so the plate treatment now keys on "is a ring" and the
aromatic/aliphatic distinction only picks the colour. Its ring closes back onto
the backbone — N and Cα are two of the five — so its true centroid lands on the
ribbon and the tablet would be half-buried with nowhere for the rod to go. The
outline and the plane stay real; the tablet slides out to the centre of the side
chain proper.

**The GPU draws the skin as geometry.** `gl_render.py` gains a mesh program
alongside the impostors, and `glyph_mesh.py` builds the ribbon as a swept tube
with a rounded cross-section and the tablets as chamfered prisms, both with real
per-vertex normals. Everything else in the skin — volumes, beads, rods, bonds,
atoms — is a sphere or a cylinder, and those stay analytic impostors, which are
exact and cheaper than any tessellation. A 46-residue protein at 640x480 draws
in 2.2 ms against the raycaster's 97 ms.

The raycaster keeps its own path. The README's headline is pure-software
rendering and that path already worked; "GPU only" means the *polish* needs the
GPU. **The GPU rendering is canonical** — the two are allowed to differ, and a
test asserts they do rather than that they match.

Letters are the one part that cannot be cached with the geometry, because both
things that make one readable depend on the camera. A letter on a tablet is
printed onto the face and stands upright *on screen*; pinning it to the tablet's
own axes leaves it lying at whatever angle the ring happens to sit at. A letter
on a rounded volume has no face to print on, so it is squared to the viewer and
cut out of its quad. A tablet turned nearly edge-on falls back to the second.

Three things about the letter quads, each of which was a bug first:

- A printed letter blends ink into the face it sits on, so its quad has to carry
  *that face's* colour. Coloured with the ink itself, the blend has nothing to
  blend into and the whole quad comes out solid black.
- A cut-out letter has nothing behind it, so it discards below a coverage
  threshold instead of blending. Supersampling is what smooths its edge, the
  same as for every impostor silhouette.
- The in-plane basis has to be right-handed with respect to the face being
  viewed, or the letter is mirrored — which for a P reads as a 9 and takes a
  while to recognize as a handedness error rather than a rotation.

Two plumbing details worth keeping: the mesh fragment shader must write
`alpha = 1`, because `style.transparent` is the viewer's default and every PNG
check uses an opaque background, so the omission would be invisible until the
terminal; and mesh depth from `gl_Position` agrees exactly with the impostors'
analytic `gl_FragDepth`, because the projection's third row is what both use.


## Fourth pass

**The hydrogen-bond hairlines are gone.** They joined backbone N to backbone O,
and once the backbone amide was abstracted into the ribbon those two endpoints
stopped being drawn — so a hairline ran between two invisible points and read as
a stray tube rather than as a bond. The donor/acceptor table in `residues.py`
stays, unused and marked as such: it is the part that would be tedious to
rebuild if the interactions are ever drawn again, and the fix if they are is a
visual that reads as annotation rather than as structure.

**Tablets are thicker** (0.34 Å, from 0.22) so the chamfered rim has room to
show, and **letters are smaller** (0.78 Å cap height) now that they carry more
text.

**Labels are the residue's code and its number**, the number at 60% of the cap
height and sharing the code's baseline, like a subscript. Both renderers lay a
label out through one `glyph_font.layout`, so the terminal and the GPU put the
same text in the same place and only the rasterizing differs.

**The round letters are generated from arcs** rather than hand-listed at eight
or nine points, which is what a tablet filling a quarter of the frame needs: the
fix for an O that reads as a polygon is more points, not better-chosen ones. The
digits are new. Two of them were wrong in a way worth noting, because it is the
same mistake twice: a glyph whose bowl is a closed loop and whose tail is a
separate stroke has to be *two* polylines. Appending the tail to the loop draws
the connecting line as well, which turned the 6 into an epsilon and put a bar
through the G.

**The palette is reworked.** The two solids are the subject, so they take the
only two near-saturated values — brass for the aromatic tablets, bone for
everything else — and sit a clear step apart. The ribbon is warm graphite rather
than black, so its shading has somewhere to go. The beads and rods are fittings:
they share one lightness so they read as a family, muted enough not to compete
with the tablets, light enough to be visible against the ribbon they land on.


## Fifth pass

**Letters are printed onto the residue and stay there.** They were briefly
placed per frame so they always stood upright on screen; that made them tags
pointing at the structure rather than markings on it. Now each label's plane and
its in-plane baseline are fixed to the residue, so turning the camera
foreshortens a letter and eventually takes it out of sight, the way a marking on
a real object behaves. A tablet's plane is its own, printed on both faces so it
is readable from either side and lost only edge-on; a rounded volume gets the
single plane facing away from the backbone. A letter stands on its stem where
there is one to stand on -- feet toward the Cα -- and along the chain otherwise.

Because nothing about a letter now depends on the camera, the whole skin is
cached again: the per-frame letter pass is gone, and the raycaster maps a glyph
onto its plane with one 2x2 solve per pixel, which under an orthographic camera
is exact. Both renderers put the same marking on the same surface.

**The font grew serifs, derived rather than drawn.** Every free stroke terminal
gets a short tick perpendicular to the stroke that finishes there; a closed
polyline has no free end and so gets none. Doing it by rule keeps thirty-three
glyphs consistent, and it is what stops a run of capitals reading as marker pen.
The stroke is lighter to suit them.

**Tablets are thicker again** (0.42 Å) and their edge is a rounded bevel in four
steps rather than a single chamfer -- one step catches a single hard highlight,
a few make it a moulded edge. **The Cα bead is wider than the ribbon is thick**,
so the rod grows out of a swelling of the backbone instead of being planted on
it. **Side-chain spheres are 0.92 Å**, well over half a bond length, so a
two-carbon side chain runs together with a soft waist -- a snowman rather than a
string of beads.


## Sixth pass: reading a protein with no names

The skin needed a PDB, because it needed residue names. It no longer does.

`select.py` already had a bond-graph `N–Cα–C(=O)` motif detector, written for
the selection presets on unnamed formats, but it returned a flat set of indices.
It now exposes `peptide_motifs`, which returns the same motifs as `(N, Cα, C, O)`
tuples grouped into runs linked by real peptide bonds — the shape both callers
want, since a ribbon must not leap a chain break and neither must residue
numbering. The preset path is a thin consumer of it.

`residues.infer_residues` hangs a side chain off each Cα by walking outward
breadth first. The insight that makes this cheap: **a PDB side-chain atom name
already encodes the two facts the walk produces.** The Greek letter is the bond
distance from the Cα, and the leading character is the element — so `CD1` is
"carbon, three bonds out". The multiset of (element, distance) over a side chain
is unique across all twenty residues, which turns identification into a lookup
against `SIDE_CHAIN_ATOMS` rather than a graph search. Sorting the matched
residue's canonical names and the walked atoms into the same order then hands
every atom the name it would have had, so `RING_ATOMS` and the side-chain splits
work unchanged and the whole thing returns ordinary `Residue` objects.

Verified against three structures by stripping their names and checking the
recovered sequence *and* every atom name against the PDB they came from. Two
things the walk has to refuse to cross, both found that way:

- **Disulfides.** The one covalent bond between two side chains a protein
  routinely has; crossing one fuses both cysteines into a fragment matching
  nothing, which is what turned crambin's six cysteines into UNK. No residue
  owns two sulfurs, so refusing to step from one to another costs nothing.
- **Any other motif's backbone**, so a mis-perceived bond cannot leak the walk
  along the chain into the next residue.


## Seventh pass: text on a ball

Three complaints about the letters on the rounded volumes, all one cause: the
letter's plane was measured out from the blob's *centroid* along the outward
direction. A blob is not a ball centred on its centroid, so that plane cuts
through whichever lobe happens to lie in front of it -- and on glycine, whose
anchor **is** its only sphere, the plane fell inside it and the letter never
appeared at all.

The letter is now printed on the lobe that reaches furthest along the outward
direction, tangent to that sphere. A regression test asserts the print point of
every volume label lies outside every sphere in the scene, which is the property
that was actually violated.

It is also **wrapped onto the sphere** rather than floated on a tangent plane.
A point *d* from the print centre in tangent direction *w* maps to
``radius * (normal cos(d/radius) + w sin(d/radius))`` -- where you get to by
walking that far across the surface -- so proportions hold along the strokes
instead of stretching toward the edges, the way a decal on a ball behaves. On
a quarter-angstrom ball a flat quad visibly lifts off at the corners and reads
as a card stuck on; this reads as printed.

One consequence worth its own guard: wrapped text that reaches the ball's
horizon folds under and disappears, so the run has to be scaled to fit. On the
*width* of the whole run, not the cap height -- "G10" is more than twice as wide
as it is tall, which is exactly the case that overflowed a small glycine marker.

The raycaster still prints these flat, on the same corrected plane. The wrap is
GPU-only; the CPU path gets the placement right and loses only the curvature.
