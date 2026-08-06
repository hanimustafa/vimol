import os

import numpy as np
import pytest

import vimol
from vimol import _render_fast as _fast
from vimol import editor, glyphs, residues
from vimol.parsers import pdb
from vimol.render import Renderer, Style
from vimol.structures import StructureSet
from vimol.widget import REPRESENTATIONS, MoleculeWidget

HAIRPIN = os.path.join(os.path.dirname(__file__), "data", "hairpin.pdb")

# Two residues of an ideal extended chain, enough to exercise the backbone
# paths without depending on the bundled structure.
DIPEPTIDE = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.988  -0.773   1.199  1.00  0.00           C
ATOM      6  N   SER A   2       3.332   1.550   0.000  1.00  0.00           N
ATOM      7  CA  SER A   2       3.970   2.858   0.000  1.00  0.00           C
ATOM      8  C   SER A   2       5.480   2.700   0.000  1.00  0.00           C
ATOM      9  O   SER A   2       6.000   1.580   0.000  1.00  0.00           O
ATOM     10  CB  SER A   2       3.560   3.680   1.220  1.00  0.00           C
ATOM     11  OG  SER A   2       4.100   4.990   1.190  1.00  0.00           O
"""

# The same, with the hydrogens an NMR model would carry: one on the hydroxyl,
# the rest on carbons.
DIPEPTIDE_WITH_H = DIPEPTIDE + """\
ATOM     12  HB2 SER A   2       2.480   3.760   1.240  1.00  0.00           H
ATOM     13  HB3 SER A   2       3.860   3.170   2.140  1.00  0.00           H
ATOM     14  HG  SER A   2       3.840   5.470   1.980  1.00  0.00           H
ATOM     15  HA  SER A   2       3.700   3.410  -0.900  1.00  0.00           H
"""


@pytest.fixture
def hairpin():
    return vimol.load(HAIRPIN)


# -- residue parsing ------------------------------------------------------

def test_pdb_parser_keeps_residue_names_without_widening_atom_keys(hairpin):
    assert len(hairpin.atom_resnames) == hairpin.n_atoms
    assert hairpin.atom_resnames[0] == "SER"
    # align.py and StructureSet match structures on this key's exact shape.
    assert all(len(k.split("|")) == 6 for k in hairpin.atom_keys)


def test_residues_group_into_the_real_sequence(hairpin):
    res = residues.protein_residues(hairpin)
    assert "".join(r.letter for r in res) == "SWTWENGKWTWK"
    assert res[0].name == "SER"
    assert res[1].is_aromatic and not res[0].is_aromatic


def test_a_protein_with_no_names_at_all_is_read_from_its_structure(hairpin):
    """An xyz file has neither residue names nor atom names, so both come from
    the bond graph: the backbone from the same motif detector the selection
    presets use, the identities from each side chain's (element, bond-distance)
    signature. It has to agree with the PDB it was stripped from."""
    from vimol.molecule import Molecule
    bare = Molecule(symbols=list(hairpin.symbols), positions=hairpin.positions.copy())
    assert not bare.atom_resnames and not bare.atom_names
    named = residues.protein_residues(hairpin)
    guessed = residues.protein_residues(bare)
    assert [r.letter for r in guessed] == [r.letter for r in named]
    # And every atom got the name it would have had, which is what lets the
    # ring lookups and the side-chain splits work unchanged.
    for a, b in zip(named, guessed):
        for name, index in a.atoms.items():
            assert b.atoms.get(name) == index, f"{a.name} {name}"


def test_the_side_chain_signatures_tell_all_twenty_apart():
    assert len(residues._BY_SIGNATURE) == len(residues.SIDE_CHAIN_ATOMS) == 20


def test_a_disulfide_does_not_merge_two_cysteines():
    """The one covalent bond between two side chains a protein routinely has.
    Walking across it fuses both into a fragment matching nothing, which is
    what turned crambin's six cysteines into UNK. No residue owns two sulfurs,
    so refusing to step from one to another costs nothing."""
    #  CA - CB - SG ~~ SG - CB - CA, the two halves of a disulfide bridge.
    symbols = ["C", "C", "S", "S", "C", "C"]
    neighbors = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]]
    walked = residues._walk_side_chain(0, {0}, neighbors, symbols)
    assert sorted(walked) == [(1, 1), (2, 2)]
    # Without the rule the walk runs all the way into the partner residue.
    assert (4, 4) not in walked and (5, 5) not in walked


def test_a_glyph_scene_builds_from_a_nameless_protein(hairpin):
    from vimol.molecule import Molecule
    bare = Molecule(symbols=list(hairpin.symbols), positions=hairpin.positions.copy())
    scene = glyphs.build_scene(bare, "light")
    assert scene is not None
    assert "".join(scene.label_char) == "SWTWENGKWTWK"


def test_a_molecule_that_is_not_a_peptide_has_no_residues():
    mol = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                  "examples", "water.xyz"))
    assert residues.protein_residues(mol) == []
    assert glyphs.build_scene(mol) is None


def test_chain_runs_split_where_the_backbone_is_actually_broken():
    mol = pdb.parse(DIPEPTIDE)[0]
    res = residues.protein_residues(mol)
    assert len(residues.chain_runs(res, mol.positions)) == 1
    # Push the second residue far away: same numbering, no peptide bond.
    moved = mol.positions.copy()
    moved[5:] += 40.0
    assert len(residues.chain_runs(res, moved)) == 2


def test_side_chain_excludes_the_backbone():
    mol = pdb.parse(DIPEPTIDE)[0]
    ser = residues.protein_residues(mol)[1]
    names = {n for n, i in ser.atoms.items() if i in ser.side_chain_indices()}
    assert names == {"CB", "OG"}


# -- labels ---------------------------------------------------------------

def test_labels_carry_the_residue_number(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    assert scene.label_number == [str(n) for n in range(1, 13)]


def test_every_residue_gets_a_letter_clear_of_its_own_solid(hairpin):
    """A volume's letter is printed on the lobe that sticks out furthest, not
    on a plane measured from the blob's centroid: the centroid sits *inside*
    the spheres, so a letter placed from it is swallowed -- and on glycine,
    whose anchor is its only sphere, it never showed at all."""
    scene = glyphs.build_scene(hairpin, "light")
    assert len(scene.label_char) == 12
    for k in range(len(scene.label_char)):
        if scene.label_on_tablet[k]:
            continue
        printed = scene.label_center[k] + scene.label_normal[k] * scene.label_offset[k]
        inside = (np.linalg.norm(scene.sphere_center - printed, axis=1)
                  < scene.sphere_radius - 1e-6)
        assert not inside.any(), f"{scene.label_char[k]} is printed inside a sphere"


def test_a_wrapped_run_fits_on_the_ball_it_is_printed_on():
    """Wrapped text reaching the horizon folds under and disappears, so the
    run has to be scaled on its width -- "G10" is more than twice as wide as
    it is tall, which is what overflowed the small glycine marker."""
    from vimol.glyph_font import run_width
    mol = pdb.parse(DIPEPTIDE.replace("ALA A   1", "GLY A   1"))[0]
    scene = glyphs.build_scene(mol, "light")
    for k, char in enumerate(scene.label_char):
        if scene.label_on_tablet[k]:
            continue
        radius = float(scene.label_offset[k] - glyphs.glyph_mesh.LETTER_LIFT)
        assert run_width(char, scene.label_number[k],
                         float(scene.label_size[k])) <= radius * glyphs.LETTER_SPAN + 1e-9


def test_a_letter_stays_on_the_face_it_is_printed_on(hairpin):
    """It is a marking on the residue, not a tag pointing at it: the plane it
    lies in is fixed to the structure, so turning the camera foreshortens it
    and eventually takes it out of sight."""
    scene = glyphs.build_scene(hairpin, "light")
    assert np.allclose(np.linalg.norm(scene.label_normal, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(scene.label_down, axis=1), 1.0)
    # The letter's baseline lies in the face, so its down is perpendicular to
    # the normal -- otherwise the glyph would be sheared off the surface.
    assert np.allclose(np.einsum("ij,ij->i", scene.label_normal,
                                 scene.label_down), 0.0, atol=1e-9)


def test_the_font_is_a_real_face_with_no_runtime_font_machinery():
    """Outlines are baked out of DejaVu Sans Bold by a script, so vimol needs
    no fontTools, no freetype and no Pillow at run time -- and draws the same
    letterforms everywhere rather than whatever the host has installed."""
    from vimol import glyph_outlines
    assert set("ACDEFGHIKLMNPQRSTVWY0123456789") <= set(glyph_outlines.OUTLINES)
    assert "Bitstream Vera" in (glyph_outlines.__doc__ or "")
    # A D has an outer contour and its counter; a T has just the one.
    assert len(glyph_outlines.OUTLINES["D"]) == 2
    assert len(glyph_outlines.OUTLINES["T"]) == 1


def test_a_filled_glyph_is_solid_not_hollow():
    """The counter of a D is empty and the stroke around it is not. An
    even-odd fill gets this right; a stroke test would ink only the outline."""
    from vimol.glyph_font import atlas
    image, boxes = atlas()
    h, w = image.shape
    u0, v0, u1, v1 = boxes["D"]
    cell = image[int(v0 * h):int(v1 * h), int(u0 * w):int(u1 * w)]
    rows = cell.shape[0]
    def runs(row):
        return int(np.count_nonzero(np.diff(np.r_[False, row > 128, False].astype(int)) > 0))
    # A row through the waist crosses the stem, the counter, then the bowl:
    # two runs of ink. An outline would give the same there, so also check a
    # row through the top bar, where a filled D is one solid run and an
    # outlined one is two.
    assert runs(cell[rows // 2]) == 2
    assert runs(cell[int(rows * 0.18)]) == 1


def test_a_label_lays_its_number_out_beside_its_code():
    from vimol.glyph_font import layout
    run = layout("Y", "12", 1.0)
    assert [c for c, _dx, _dy, _h in run] == ["Y", "1", "2"]
    (_c, code_x, code_y, code_h), *digits = run
    assert code_y == 0.0 and code_h == 1.0
    # Smaller, to the right, and dropped to share the code's baseline.
    for _c, dx, dy, h in digits:
        assert dx > code_x and dy > 0 and h < code_h
    assert digits[0][1] < digits[1][1]


def test_a_residue_with_no_number_still_lays_out():
    from vimol.glyph_font import layout
    assert [c for c, *_ in layout("G", "", 1.0)] == ["G"]


def test_the_skin_no_longer_draws_hydrogen_bond_hairlines(hairpin):
    """Dropped on request: with the backbone amides abstracted into the ribbon
    they ran between two invisible points and read as stray tubes. The
    donor/acceptor table stays, unused, for whenever they come back."""
    scene = glyphs.build_scene(hairpin, "light")
    assert len(scene.cyl_a)
    # The hairlines were the only cylinder thinner than a rod.
    assert scene.cyl_radius.min() >= min(glyphs.ROD_RADIUS, 0.075)
    assert not hasattr(glyphs, "LINK_RADIUS")


# -- residue chemistry ----------------------------------------------------

def test_hbond_roles_follow_the_chemistry():
    assert residues.hbond_role("ALA", "N") == "donor"
    assert residues.hbond_role("PRO", "N") is None      # ring N, no hydrogen
    assert residues.hbond_role("ALA", "O") == "acceptor"
    assert residues.hbond_role("ASP", "OD1") == "acceptor"
    assert residues.hbond_role("LYS", "NZ") == "donor"
    assert residues.hbond_role("SER", "OG") == "both"
    assert residues.hbond_role("ALA", "CB") is None


def test_side_chain_oxygens_and_nitrogens_are_drawn_at_their_file_coordinates(hairpin):
    """The whole point of drawing them as atoms instead of as abstract markers
    is that they land exactly where the file puts them."""
    scene = glyphs.build_scene(hairpin, "light")
    for res in residues.protein_residues(hairpin):
        for atom in res.side_chain_polar():
            d = np.linalg.norm(scene.sphere_center - hairpin.positions[atom], axis=1)
            assert d.min() < 1e-9, f"{res.name} atom {atom} is not drawn where it is"


def test_side_chain_polar_atoms_keep_their_element_colours(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    colors = hairpin.element_colors()
    for res in residues.protein_residues(hairpin):
        for atom in res.side_chain_polar():
            k = int(np.argmin(np.linalg.norm(scene.sphere_center
                                             - hairpin.positions[atom], axis=1)))
            assert np.allclose(scene.sphere_color[k], colors[atom])


def test_the_backbone_amide_is_the_ribbons_business_not_an_atom(hairpin):
    """Drawing backbone N and O as atoms stipples a blue and a red dot onto
    every residue of the ribbon, which is what the ribbon is there to replace."""
    scene = glyphs.build_scene(hairpin, "light")
    for res in residues.protein_residues(hairpin):
        for name in ("N", "O"):
            if name not in res.atoms:
                continue
            d = np.linalg.norm(scene.sphere_center
                               - hairpin.positions[res.atoms[name]], axis=1)
            assert d.min() > 1e-9, f"backbone {name} drawn as an atom"


# -- geometry -------------------------------------------------------------

def test_box_planes_bound_exactly_the_box():
    u, v, w = np.eye(3)
    normals, offsets = glyphs.box_planes(u, v, w, (2.0, 1.0, 0.5))
    inside = np.array([1.9, 0.9, 0.4])
    outside = np.array([1.9, 0.9, 0.6])
    assert np.all(normals @ inside <= offsets + 1e-12)
    assert np.any(normals @ outside > offsets + 1e-12)


def test_every_solid_fits_inside_the_box_the_renderer_shades(hairpin):
    """The screen bounding box comes from `poly_axes`/`poly_half`. If a solid
    pokes out of it the rasterizer never visits those pixels, so the shape
    gets sliced off at an invisible edge -- and an inflated ring outline does
    poke past its own ring atoms plus the inflation, at every sharp corner."""
    scene = glyphs.build_scene(hairpin, "light")
    rng = np.random.default_rng(0)
    for p in range(len(scene.poly_center)):
        lo, hi = int(scene.poly_slice[p]), int(scene.poly_slice[p + 1])
        normals, offsets = scene.plane_normal[lo:hi], scene.plane_offset[lo:hi]
        half = scene.poly_half[p]
        reach = float(np.linalg.norm(half))
        pts = rng.uniform(-reach, reach, size=(3000, 3))
        inside = pts[np.all(pts @ normals.T <= offsets + 1e-12, axis=1)]
        assert len(inside), "sampling found no interior -- solid is degenerate"
        local = np.abs(inside @ scene.poly_axes[p].T)
        assert np.all(local <= half + 1e-9)


def test_convex_hull_drops_interior_points():
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1.5, 0.5]], float)
    hull = glyphs.convex_hull_2d(pts)
    assert sorted(hull) == [0, 1, 2, 3]
    # counter-clockwise, which the plate builder relies on for outward normals
    loop = pts[hull]
    area = np.cross(loop[1] - loop[0], loop[2] - loop[0])
    assert area > 0


def test_plane_frame_finds_the_ring_normal():
    ring = np.array([[np.cos(t), np.sin(t), 0.0]
                     for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)])
    center, normal, e1, e2 = glyphs.plane_frame(ring)
    assert np.allclose(center, 0, atol=1e-9)
    assert abs(abs(normal[2]) - 1.0) < 1e-9
    assert abs(np.dot(e1, e2)) < 1e-9


def test_the_ribbon_runs_through_the_alpha_carbons(hairpin):
    """Exactly through them -- no smoothing of the path at all. What read as
    crinkle was the ribbon's twist, not its route (see
    `test_the_ribbon_does_not_corkscrew_along_a_helix`)."""
    res = residues.protein_residues(hairpin)
    run = residues.chain_runs(res, hairpin.positions)[0]
    centers, _sides = glyphs._ribbon_frames(run, hairpin.positions)
    expected = np.array([hairpin.positions[r.atoms["CA"]] for r in run])
    assert np.allclose(centers, expected)


def test_the_link_beads_sit_on_the_real_alpha_and_beta_carbons(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    for res in residues.protein_residues(hairpin):
        for name in ("CA", "CB"):
            if name not in res.atoms:
                continue
            d = np.linalg.norm(scene.sphere_center
                               - hairpin.positions[res.atoms[name]], axis=1)
            assert d.min() < 1e-9, f"no bead on {res.name} {name}"


def test_hydrogens_show_only_where_they_say_something():
    """On a hydroxyl or an amide, not on every carbon: an NMR model carries
    every C–H, and drawing those buries the skin under white spheres."""
    mol = pdb.parse(DIPEPTIDE_WITH_H)[0]
    scene = glyphs.build_scene(mol, "light")
    res = residues.protein_residues(mol)[1]
    on_oxygen = mol.positions[res.atoms["HG"]]
    on_carbon = mol.positions[res.atoms["HB2"]]
    assert np.linalg.norm(scene.sphere_center - on_oxygen, axis=1).min() < 1e-9
    assert np.linalg.norm(scene.sphere_center - on_carbon, axis=1).min() > 1e-9


def test_ribbon_side_vectors_never_flip_between_neighbours(hairpin):
    res = residues.protein_residues(hairpin)
    run = residues.chain_runs(res, hairpin.positions)[0]
    _centers, sides = glyphs._ribbon_frames(run, hairpin.positions)
    dots = np.einsum("ij,ij->i", sides[:-1], sides[1:])
    # Carbonyls alternate along a strand; without the sign correction these
    # would be near -1 and the ribbon would corkscrew once per residue.
    assert dots.min() > 0.0


def test_virtual_cbeta_lands_at_a_real_bond_length():
    mol = pdb.parse(DIPEPTIDE)[0]
    res = residues.protein_residues(mol)[0]
    cb = glyphs.virtual_cbeta(mol.positions[res.atoms["N"]],
                              mol.positions[res.atoms["CA"]],
                              mol.positions[res.atoms["C"]])
    assert 1.4 < np.linalg.norm(cb - mol.positions[res.atoms["CA"]]) < 1.7


# -- the scene ------------------------------------------------------------

def test_scene_labels_every_residue_once(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    assert "".join(scene.label_char) == "SWTWENGKWTWK"
    assert len(scene.cyl_a) >= len(scene.label_char)      # one stick each, plus links


def test_aromatics_get_a_plate_and_everything_else_a_volume(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    plate = np.all(np.isclose(scene.poly_color, glyphs.LIGHT.plate), axis=1)
    # trpzip2 has four tryptophans and nothing else aromatic
    assert plate.sum() == 4
    volume = np.all(np.isclose(scene.sphere_color, glyphs.LIGHT.volume), axis=1)
    assert volume.sum() > 0


def test_glycine_still_gets_a_marker_and_a_letter():
    text = DIPEPTIDE.replace("ALA A   1", "GLY A   1").replace(
        "ATOM      5  CB  GLY A   1       1.988  -0.773   1.199  1.00  0.00           C\n", "")
    mol = pdb.parse(text)[0]
    scene = glyphs.build_scene(mol, "light")
    assert scene.label_char[0] == "G"
    volume = np.all(np.isclose(scene.sphere_color, glyphs.LIGHT.volume), axis=1)
    assert volume.sum() >= 1


def test_a_letter_is_lifted_clear_of_its_own_volume():
    """The lift must cover the whole reach of the shape the letter names. A
    letter biased by one sphere radius is swallowed by its own side chain the
    moment the camera turns, which is how it first went wrong."""
    mol = pdb.parse(DIPEPTIDE)[0]
    scene = glyphs.build_scene(mol, "light")
    volume = np.all(np.isclose(scene.sphere_color, glyphs.LIGHT.volume), axis=1)
    centers, radii = scene.sphere_center[volume], scene.sphere_radius[volume]
    for k, anchor in enumerate(scene.label_center):
        own = np.linalg.norm(centers - anchor, axis=1) < 3.0
        reach = (np.linalg.norm(centers[own] - anchor, axis=1) + radii[own]).max()
        assert scene.label_bias[k] >= reach


def test_a_letter_is_lifted_clear_of_its_own_plate(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    plate = np.all(np.isclose(scene.poly_color, glyphs.LIGHT.plate), axis=1)
    bounds = np.linalg.norm(scene.poly_half[plate], axis=1)
    for center, bound in zip(scene.poly_center[plate], bounds):
        k = int(np.argmin(np.linalg.norm(scene.label_center - center, axis=1)))
        # Centred on the plaque's outline rather than on the solid's origin --
        # for proline those differ, and the letter used to hang off the edge.
        assert np.linalg.norm(scene.label_center[k] - center) < bound
        assert scene.label_bias[k] >= bound


def test_scene_is_cached_per_molecule_and_busted_by_an_edit(hairpin):
    first = glyphs.cached_scene(hairpin, "light")
    assert glyphs.cached_scene(hairpin, "light") is first
    assert glyphs.cached_scene(hairpin, "dark") is not first
    hairpin.positions = hairpin.positions + 1.0
    assert glyphs.cached_scene(hairpin, "light") is not first


def test_the_cache_notices_a_structure_being_marked_into_an_overlay(hairpin):
    """Tint and flatness are geometry inputs, and neither moves an atom or
    changes the count -- so a key built from coordinates alone would keep
    serving the untinted scene after an overlay is switched on."""
    plain = glyphs.cached_scene(hairpin, "light")
    flat = np.ones(hairpin.n_atoms, bool)
    tinted = glyphs.cached_scene(hairpin, "light", flat_mask=flat,
                                 atom_colors=np.tile([1.0, 0.5, 0.0],
                                                     (hairpin.n_atoms, 1)))
    assert tinted is not plain
    assert not plain.poly_flat.any()
    assert tinted.poly_flat.all()


def test_an_overlaid_structure_renders_in_its_own_flat_tint(hairpin):
    other = vimol.load(HAIRPIN)
    other.positions = other.positions + np.array([3.0, 1.0, 0.0])
    structures = StructureSet()
    structures.append(hairpin, label="main")
    structures.append(other, label="other")
    for entry in structures.entries:
        entry.marked = True
    structures.overlay = True
    scene = vimol.Scene(structures, 160, 120, style=Style(representation="glyph"),
                        backend="cpu")
    composite = structures.composite()
    glyph_scene = glyphs.glyph_scene_for(composite.molecule,
                                         scene._effective_style(composite))
    # Both copies are drawn, and exactly one of them is flat-tinted.
    assert "".join(glyph_scene.label_char) == "SWTWENGKWTWK" * 2
    for flags in (glyph_scene.poly_flat, glyph_scene.label_flat):
        assert 0 < flags.sum() < len(flags)
    tint = np.asarray(structures.entries[1].tint)
    flat_solids = glyph_scene.poly_color[glyph_scene.poly_flat]
    assert np.allclose(flat_solids, tint)


# -- rendering ------------------------------------------------------------

def test_glyph_frame_is_not_a_ball_and_stick_frame(hairpin):
    a = vimol.Scene(hairpin, 160, 120, style=Style(representation="glyph"),
                    backend="cpu").render()
    b = vimol.Scene(hairpin, 160, 120, style=Style(), backend="cpu").render()
    assert not np.array_equal(a, b)
    assert (a != a[0, 0]).any(), "glyph frame is empty"


def test_a_non_protein_falls_back_to_ball_and_stick_pixel_for_pixel():
    mol = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                  "examples", "benzene.xyz"))
    a = vimol.Scene(mol, 160, 120, style=Style(representation="glyph"),
                    backend="cpu").render()
    b = vimol.Scene(mol, 160, 120, style=Style(), backend="cpu").render()
    assert np.array_equal(a, b)


def test_the_compiled_kernel_is_skipped_for_glyph_but_not_for_atoms(hairpin, monkeypatch):
    """The numba path returns before the band pass and knows nothing about
    polyhedra or letters, so leaving it armed would drop the whole skin -- and
    only on machines where numba compiled."""
    calls = []
    monkeypatch.setattr(_fast, "ready", lambda: True)
    monkeypatch.setattr(_fast, "render_frame",
                        lambda *a, **k: calls.append("kernel"))
    Renderer(80, 60).render(hairpin, vimol.Scene(hairpin, 80, 60).camera,
                            Style(representation="glyph"))
    assert calls == []
    Renderer(80, 60).render(hairpin, vimol.Scene(hairpin, 80, 60).camera, Style())
    assert calls == ["kernel"]


def test_the_gpu_draws_the_skin_as_real_geometry(hairpin):
    """The polished look is the GPU's: a swept ribbon and chamfered tablets as
    triangles, where the raycaster intersects half-spaces and fakes the smooth
    shading. The two are allowed to differ -- the GPU one is canonical."""
    try:
        gl = vimol.Scene(hairpin, 200, 160, style=Style(representation="glyph"),
                         backend="gl")
    except Exception:
        pytest.skip("no GL backend available")
    frame = gl.render()
    assert gl.backend == "gl"
    assert (frame != frame[0, 0]).any(), "GPU glyph frame is empty"
    cpu = vimol.Scene(hairpin, 200, 160, style=Style(representation="glyph"),
                      backend="cpu").render()
    assert not np.array_equal(frame, cpu)


def test_the_gpu_mesh_carries_the_ribbon_and_the_tablets(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    assert len(scene.mesh.vertices) > 0
    assert len(scene.mesh.indices) % 3 == 0
    assert int(scene.mesh.indices.max()) < len(scene.mesh.vertices)
    # Letters are printed into the mesh too, and are the only part of it that
    # samples the glyph atlas.
    lettered = scene.mesh.uv[:, 0] >= 0
    assert lettered.any() and not lettered.all()


def test_a_non_protein_still_falls_back_on_the_gpu():
    mol = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                  "examples", "benzene.xyz"))
    try:
        gl = vimol.Scene(mol, 160, 120, style=Style(representation="glyph"),
                         backend="gl")
    except Exception:
        pytest.skip("no GL backend available")
    plain = vimol.Scene(mol, 160, 120, style=Style(), backend="gl")
    assert np.array_equal(gl.render(), plain.render())


def test_the_camera_frames_the_glyph_geometry_not_the_atoms(hairpin):
    """`fit` adds this to the radius of gyration, so the sum has to be the
    reach of what is actually drawn -- ribbon edges and ring plates included,
    none of which a van der Waals radius knows about."""
    scene = vimol.Scene(hairpin, 160, 120, style=Style(representation="glyph"),
                        backend="cpu")
    glyph_scene = glyphs.cached_scene(hairpin, scene.style.glyph_theme)
    reach = glyph_scene.reach_from(hairpin.centroid())
    assert reach > hairpin.radius_of_gyration_extent()
    assert np.isclose(hairpin.radius_of_gyration_extent() + scene._max_atom_radius(),
                      reach)


def test_a_molecule_with_no_residues_is_framed_as_ball_and_stick():
    mol = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                  "examples", "benzene.xyz"))
    glyph = vimol.Scene(mol, 160, 120, style=Style(representation="glyph"),
                        backend="cpu")
    plain = vimol.Scene(mol, 160, 120, style=Style(), backend="cpu")
    assert glyph._max_atom_radius() == plain._max_atom_radius()


def test_a_letter_too_small_to_read_is_dropped_rather_than_smeared(hairpin):
    scene = vimol.Scene(hairpin, 60, 45, style=Style(representation="glyph"),
                        backend="cpu")
    scene.camera.zoom = 0.5           # letters far under MIN_GLYPH_PX
    scene.render()                    # must not raise


# -- the key and the fallback message -------------------------------------

def test_five_is_the_bare_ribbon_and_six_the_glyph_skin(hairpin):
    assert REPRESENTATIONS[4:] == ["ribbon", "glyph"]
    widget = MoleculeWidget(hairpin, 160, 120)
    assert widget.handle_key("5")
    assert widget.style.representation == "ribbon"
    assert widget.handle_key("6")
    assert widget.style.representation == "glyph"
    assert widget.rep_note == ""


def test_the_ribbon_mode_draws_the_backbone_and_nothing_else(hairpin):
    scene = glyphs.build_scene(hairpin, "light", ribbon_only=True)
    assert len(scene.poly_center)                     # ribbon segments
    assert not len(scene.sphere_center) and not len(scene.cyl_a)
    assert not scene.label_char
    assert np.allclose(scene.poly_color, glyphs.RIBBON_GREEN)


def test_the_ribbon_does_not_corkscrew_along_a_helix():
    """The width runs along the carbonyl squared up against the chain, which on
    a helix is very nearly the helix axis and so barely turns from one residue
    to the next. Carson-Bugg's other vector, the peptide plane's normal, is
    radial there and sweeps a full turn every 3.6 residues -- use it as the
    width and the ribbon corkscrews, presenting its edge to the outside instead
    of the flat face a cartoon shows."""
    # An ideal alpha helix: 100 degrees and 1.5 A of rise per residue, with the
    # carbonyl pointing along the axis.
    n = 12
    angle = np.radians(100.0) * np.arange(n)
    ca = np.column_stack([2.3 * np.cos(angle), 2.3 * np.sin(angle),
                          1.5 * np.arange(n)])
    residues_ = []
    positions = []
    for i in range(n):
        base = len(positions)
        positions += [ca[i], ca[i] + [0.0, 0.0, 1.0]]     # CA, then O up the axis
        res = residues.Residue(key=("", str(i + 1), ""), name="ALA", letter="A")
        res.atoms.update({"CA": base, "O": base + 1})
        res.elements.update({"CA": "C", "O": "O"})
        residues_.append(res)
    _centers, sides = glyphs._ribbon_frames(residues_, np.array(positions))
    turn = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", sides[:-1], sides[1:]), -1.0, 1.0)))
    assert turn.max() < 25.0, f"ribbon rolls {turn.max():.0f} deg per residue"


def test_the_skin_says_why_it_did_nothing_for_a_non_protein():
    mol = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                  "examples", "benzene.xyz"))
    widget = MoleculeWidget(mol, 160, 120)
    widget.handle_key("5")
    assert "backbone" in widget.rep_note
    widget.handle_key("1")
    assert widget.rep_note == ""


# -- the new metadata survives everything that touches an atom list --------

def test_residue_names_survive_a_delete_and_an_undo(hairpin):
    widget = MoleculeWidget(hairpin, 160, 120, editable=True)
    mol = widget.scene.molecule
    before = list(mol.atom_resnames)
    widget._push_undo()
    editor.delete_atom(mol, 3)
    assert len(mol.atom_resnames) == mol.n_atoms
    assert mol.atom_resnames[3] == before[4]
    widget.undo()
    assert widget.scene.molecule.atom_resnames == before


def test_an_added_atom_keeps_the_metadata_lists_aligned(hairpin):
    hairpin.add_atom("C", 0.0, 0.0, 0.0)
    assert len(hairpin.atom_resnames) == hairpin.n_atoms
    assert len(hairpin.atom_names) == hairpin.n_atoms


def test_an_overlay_still_knows_its_residues(hairpin):
    water = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                    "examples", "water.xyz"))
    structures = StructureSet()
    structures.append(hairpin, label="hairpin")
    structures.append(water, label="water")
    for entry in structures.entries:
        entry.marked = True
    structures.overlay = True
    composite = structures.composite().molecule
    assert len(composite.atom_resnames) == composite.n_atoms
    assert len(residues.protein_residues(composite)) == 12


def test_a_composite_of_plain_files_carries_no_empty_identity():
    structures = StructureSet()
    for name in ("water.xyz", "methane.xyz"):
        structures.append(
            vimol.load(os.path.join(os.path.dirname(__file__), "..", "examples", name)))
    for entry in structures.entries:
        entry.marked = True
    structures.overlay = True
    composite = structures.composite().molecule
    assert composite.atom_resnames == []
    assert composite.atom_names == []
