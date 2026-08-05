import os

import numpy as np
import pytest

import vimol
from vimol import _render_fast as _fast
from vimol import editor, glyphs, residues
from vimol.molecule import Molecule
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


def test_a_molecule_without_residue_names_has_no_residues():
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


# -- hydrogen bonding -----------------------------------------------------

def test_hbond_roles_follow_the_chemistry():
    assert residues.hbond_role("ALA", "N") == "donor"
    assert residues.hbond_role("PRO", "N") is None      # ring N, no hydrogen
    assert residues.hbond_role("ALA", "O") == "acceptor"
    assert residues.hbond_role("ASP", "OD1") == "acceptor"
    assert residues.hbond_role("LYS", "NZ") == "donor"
    assert residues.hbond_role("SER", "OG") == "both"
    assert residues.hbond_role("ALA", "CB") is None


def test_pure_donor_and_acceptor_nodes_sit_on_the_file_coordinates(hairpin):
    scene = glyphs.build_scene(hairpin, "light")
    nodes = {"donor": np.asarray(glyphs.LIGHT.donor),
             "acceptor": np.asarray(glyphs.LIGHT.acceptor)}
    placed = {k: scene.sphere_center[np.all(np.isclose(scene.sphere_color, c), axis=1)]
              for k, c in nodes.items()}
    for res in residues.protein_residues(hairpin):
        for name, atom in res.atoms.items():
            role = residues.hbond_role(res.name, name)
            if role in ("donor", "acceptor"):
                d = np.linalg.norm(placed[role] - hairpin.positions[atom], axis=1)
                assert d.min() < 1e-9, f"{res.name} {name} node moved off its atom"


def test_an_atom_that_both_donates_and_accepts_gets_one_node_of_each():
    mol = pdb.parse(DIPEPTIDE)[0]
    res = residues.protein_residues(mol)[1]
    og = res.atoms["OG"]
    placed = glyphs._node_positions(res, og, "both", mol.positions)
    kinds = sorted(k for _p, k in placed)
    assert kinds == ["acceptor", "donor"]
    # The acceptor keeps the real coordinates; the donor steps aside by one
    # node width so the two do not fight for the same pixels.
    (p_acc, _), (p_don, _) = placed
    assert np.allclose(p_acc, mol.positions[og])
    assert np.isclose(np.linalg.norm(p_don - p_acc), glyphs.NODE_SPLIT)


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
        assert np.allclose(scene.label_center[k], center)
        assert scene.label_bias[k] >= bound


def test_scene_is_cached_per_molecule_and_busted_by_an_edit(hairpin):
    first = glyphs.cached_scene(hairpin, "light")
    assert glyphs.cached_scene(hairpin, "light") is first
    assert glyphs.cached_scene(hairpin, "dark") is not first
    hairpin.positions = hairpin.positions + 1.0
    assert glyphs.cached_scene(hairpin, "light") is not first


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


def test_glyph_renders_through_the_raycaster_even_on_the_gl_backend(hairpin):
    try:
        gl = vimol.Scene(hairpin, 160, 120, style=Style(representation="glyph"),
                         backend="gl")
    except Exception:
        pytest.skip("no GL backend available")
    cpu = vimol.Scene(hairpin, 160, 120, style=Style(representation="glyph"),
                      backend="cpu")
    assert gl.backend == "gl"
    # The GL path converts a molecule to spheres/cylinders/cones and would drop
    # ribbons, plates and letters without a trace.
    assert np.array_equal(gl.render(), cpu.render())


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

def test_five_selects_the_glyph_skin(hairpin):
    assert REPRESENTATIONS[4] == "glyph"
    widget = MoleculeWidget(hairpin, 160, 120)
    assert widget.handle_key("5")
    assert widget.style.representation == "glyph"
    assert widget.rep_note == ""


def test_the_skin_says_why_it_did_nothing_for_a_non_protein():
    mol = vimol.load(os.path.join(os.path.dirname(__file__), "..",
                                  "examples", "benzene.xyz"))
    widget = MoleculeWidget(mol, 160, 120)
    widget.handle_key("5")
    assert "protein" in widget.rep_note
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
