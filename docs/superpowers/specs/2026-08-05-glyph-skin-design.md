# The `glyph` skin: a lettered, diagrammatic protein representation

Date: 2026-08-05
Status: approved for implementation

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

### 3. Screen-aligned letter sprites

Letters go on billboards at each solid's centroid, not as decals mapped onto the
plate faces. One mechanism serves every shape, the letter stays legible from any
camera angle — which is the whole point of a labelled diagram — and it needs no
UV math. Glyphs are hand-coded 5×7 bitmaps for the twenty one-letter codes;
`README` advertises a stdlib PNG encoder and no Pillow, so no font library.

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
Centers and sides are then run through a Catmull–Rom spline at six samples per
residue, and each consecutive pair of samples becomes one oriented box: long
axis along the segment, 1.5 Å wide along the interpolated side vector, 0.18 Å
thick. At that sampling the notches between boxes on a bend are sub-pixel.

**Aromatic plates.** For `PHE`, `TYR`, `TRP` and `HIS` the ring atoms are the
named ring set for that residue. Their best-fit plane comes from the smallest
singular vector of the centered ring coordinates; a monotone-chain convex hull
in that plane (no scipy — it is an optional extra, `vimol[align]`) gives the
outline, which is pushed out 0.45 Å and extruded ±0.20 Å. The resulting solid is
the hull edges as side planes plus two caps, so a six-ring reads as a hexagon
and tryptophan's fused system as its real fused outline.

**Rounded volumes.** Every other side chain is a union of spheres of radius
0.72 Å at the real heavy-atom positions. At a 1.5 Å bond length neighbouring
spheres merge into one smooth solid, so the shape is literally the geometry of
the side chain, and it costs nothing new in the renderer. Glycine, which has no
side chain, gets a single small sphere at the position its Cβ would occupy
(built from the tetrahedral completion of N, C and Cα); alanine gets one sphere
at Cβ.

**Sticks.** A cylinder from Cα to the solid's anchor: the plate center for
aromatics, the first side-chain atom otherwise.

**Nodes.** Classified per `(residue, atom name)` rather than from attached
hydrogens, since most PDB files have none: backbone `N` donates (except
proline), backbone `O` and `OXT` accept, and a side-chain table covers the rest.
Hydroxyls, histidine ring nitrogens and cysteine sulfur do both, and draw one
node of each colour offset ±0.15 Å along the atom's own bond axis so they do not
z-fight. Every other node sits at the atom's exact file coordinates. Radius
0.13 Å.

**Links.** Backbone `N`···`O` pairs closer than 3.35 Å and at least two residues
apart get a 0.035 Å cylinder in a muted gold.

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
  `"glyph"` needs its own value or activating the skin clips the structure.
- `widget.REPRESENTATIONS` gains a fifth entry and `widget.handle_key` accepts
  `"5"`. Nothing else claims that key: the structure strip deliberately leaves
  the digits alone.
- `app.py`'s `--style` choices and the help panel's `1 2 3 4` row both grow.

## Verification

`examples/` has no protein, so a hand-written β-hairpin PDB with real residue
names goes in `tests/data/`. Verification is visual as well as structural:
render to PNG and look at it, because an assertion on array shape passes just as
happily with the ribbon inside out. Render once with the numba kernel warm and
once with it disabled, since the fast-path gate is invisible in a single run.

Unit tests cover residue grouping, the donor/acceptor table, ribbon frame
continuity (no 180° flip between strand residues), the plane-set of a box, the
convex hull, and the fallback when a molecule has no residue names.
