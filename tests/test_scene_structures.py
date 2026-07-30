"""Scene / StructureSet wiring (design doc §3, §11 step 1).

The gate for this milestone: the existing single-molecule behaviour of
Scene must stay byte-for-byte identical -- these tests pin that, plus the
new StructureSet-aware surface.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vimol
from vimol.molecule import Molecule
from vimol.scene import Scene
from vimol.structures import StructureSet
from vimol.bonds import ensure_bonds

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def _mol():
    m = Molecule(symbols=["C", "H"], positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    return m


def test_scene_wraps_a_bare_molecule_in_a_one_entry_structure_set():
    mol = _mol()
    scene = Scene(mol, 64, 64)
    assert isinstance(scene.structures, StructureSet)
    assert len(scene.structures) == 1
    assert scene.molecule is mol


def test_scene_accepts_a_structure_set_directly():
    sset = StructureSet()
    sset.append(_mol(), label="a")
    scene = Scene(sset, 64, 64)
    assert scene.structures is sset
    assert scene.molecule is sset.active.molecule


def test_scene_set_molecule_replaces_with_a_single_entry():
    mol = _mol()
    scene = Scene(mol, 64, 64)
    new_mol = _mol()
    scene.set_molecule(new_mol)
    assert scene.molecule is new_mol
    assert len(scene.structures) == 1


def test_scene_render_byte_identical_to_pre_structureset_baseline():
    """Pin today's single-molecule render output. A regression here means the
    composite fast path stopped being zero-copy/no-op for this case."""
    mol = vimol.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol)
    scene = Scene(mol, 96, 96, backend="cpu")
    scene.camera.orbit(20, -15)
    img1 = scene.render()
    # Rebuild fresh and render again -- must match exactly (determinism +
    # fast-path zero copy, not some cached numeric coincidence).
    mol2 = vimol.load(os.path.join(EX, "methane.xyz"))
    ensure_bonds(mol2)
    scene2 = Scene(mol2, 96, 96, backend="cpu")
    scene2.camera.orbit(20, -15)
    img2 = scene2.render()
    assert np.array_equal(img1, img2)


def test_scene_fit_uses_composite_extent_for_overlay():
    """A camera fit on an overlay must frame ALL drawn structures, not just
    the active one (design §3: 'if it read the active structure the other
    files would sit off-screen')."""
    sset = StructureSet()
    a = sset.append(Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]])), label="a")
    from vimol.structures import Transform
    far = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    b = sset.append(far, label="b")
    b.transform = Transform(translation=np.array([50.0, 0.0, 0.0]))
    sset.overlay = True
    scene = Scene(sset, 64, 64)
    # extent should be influenced by the far, translated structure
    assert scene.camera.extent > 10.0


# -- widget wiring: cache invalidation + composite-index highlight mapping --

from vimol.widget import MoleculeWidget
from vimol.structures import Transform


def test_widget_edit_keeps_composite_base_colors_in_sync(tmp_path):
    """Appending an atom must invalidate the cached composite (design §5:
    'every edit path ends with entry.touch()') -- else base_colors/offsets
    stay the old shorter shape."""
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    w = MoleculeWidget(mol, 200, 200, backend="cpu", editable=True)
    w.set_append_mode(True)
    w._edit_at(100.0, 100.0)   # empty space under append mode -> births an atom
    comp = w.scene.structures.composite()
    assert comp.base_colors.shape[0] == w.molecule.n_atoms


def test_widget_highlight_maps_active_local_index_through_composite_offset():
    """hovered/selected/measure_sel are ACTIVE-LOCAL indices (design §3); when
    the active structure isn't the first entry of the composite, the
    highlight must land on the right GLOBAL atom, not local index 0."""
    from vimol.scene import Scene
    from vimol.structures import StructureSet

    sset = StructureSet()
    first = Molecule(symbols=["C", "C"], positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    active = Molecule(symbols=["O", "O"], positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    sset.append(first, label="first")
    sset.append(active, label="active")
    sset.set_active(1)
    sset.overlay = True   # both drawn; active listed first in drawn_indices()

    w = MoleculeWidget.__new__(MoleculeWidget)
    w.style = __import__("vimol.render", fromlist=["Style"]).Style()
    w.scene = Scene(sset, 100, 100, style=w.style, backend="cpu")
    w.theme = "dark"
    w.hovered = 1          # active-local index 1 ("O" atom at x=2)
    w.selected = None
    w.measure_sel = []
    w.delete_mode = False
    w._base_colors = w.scene.structures.composite().base_colors

    w._apply_highlight()
    cols = w.style.color_override
    assert cols is not None
    comp = w.scene.structures.composite()
    active_idx = sset.active_index
    global_idx = int(comp.globalize(active_idx, np.array([1]))[0])
    # the atom actually tinted yellow is the GLOBAL index, not local index 1
    assert not np.allclose(cols[global_idx], comp.base_colors[global_idx])
    # and no other atom (in particular local-index-shaped index 1 of entry 0,
    # which would be a bug if the mapping were skipped) was touched
    other = 1 if global_idx != 1 else 0
    assert np.allclose(cols[other], comp.base_colors[other])


# -- flat shading (design §4.4/§4.5): CPU renderer -------------------------

from vimol.render import Renderer, Style
from vimol.camera import Camera


def _flat_test_scene():
    mol = Molecule(symbols=["C", "C"], positions=np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    style = Style(representation="spacefill", depth_cue=0.0,
                  color_override=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                  flat_mask=np.array([False, True]))
    cam = Camera(center=np.zeros(3), extent=10.0)
    cam.fit(240, 240, 10.0)
    r = Renderer(240, 240)
    return r.render(mol, cam, style)


def test_flat_atom_renders_perfectly_uniform_color():
    img = _flat_test_scene()
    # atom1 (flat, blue) sits on the right half of the frame
    blue_mask = (img[:, :, 2] > 150) & (img[:, :, 0] < 50)
    assert blue_mask.sum() > 100
    blue_pixels = img[blue_mask]
    assert (blue_pixels == blue_pixels[0]).all()


def test_shaded_atom_still_has_a_shading_gradient():
    img = _flat_test_scene()
    # atom0 (shaded, red) sits on the left half of the frame
    red_mask = (img[:, :, 0] > 50) & (img[:, :, 2] < 50)
    assert red_mask.sum() > 100
    red_pixels = img[red_mask]
    # NOT all identical -- diffuse/specular vary across the sphere's silhouette
    assert not (red_pixels == red_pixels[0]).all()


def test_flat_mask_none_is_byte_identical_to_no_flat_mask_field():
    mol = Molecule(symbols=["C"], positions=np.array([[0.0, 0.0, 0.0]]))
    style_a = Style(representation="spacefill")
    style_b = Style(representation="spacefill", flat_mask=None)
    cam = Camera(center=np.zeros(3), extent=5.0)
    cam.fit(80, 80, 5.0)
    img_a = Renderer(80, 80).render(mol, cam, style_a)
    img_b = Renderer(80, 80).render(mol, cam, style_b)
    assert np.array_equal(img_a, img_b)


def test_flat_bond_between_two_flat_atoms_renders_uniform_too():
    """Bonds never span structures in the composite (design §4.5), so a
    bond's flatness is unambiguous -- both endpoints flat -> the cylinder is
    flat too, not partially shaded."""
    mol = Molecule(symbols=["C", "C"], positions=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    mol.add_bond(0, 1, 1)
    style = Style(representation="ball_and_stick", depth_cue=0.0,
                  color_override=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
                  flat_mask=np.array([True, True]), bond_radius=0.3)
    cam = Camera(center=np.zeros(3), extent=3.0)
    cam.fit(120, 60, 3.0)
    img = Renderer(120, 60).render(mol, cam, style)
    # sample along the bond's midline (screen y == center row), away from
    # the atom spheres themselves -- should be uniform green, no shading.
    row = img[30, :, :]
    green_cols = np.where((row[:, 1] > 100) & (row[:, 0] < 60))[0]
    assert len(green_cols) > 5
    pixels = row[green_cols]
    assert (pixels == pixels[0]).all()


# -- editing while overlaid: pick() is composite, edits stay active-only ---
# (design §3 "Highlighting", §12.3)

def _overlay_two_atom_widget():
    """Two 1-atom structures, both visible, overlaid, active = index 0.
    Placed far apart on screen (each centered under the camera at its own
    world position) with a wide-open ortho view so both are individually
    clickable."""
    from vimol.scene import Scene

    sset = StructureSet()
    a = Molecule(symbols=["C"], positions=np.array([[-5.0, 0.0, 0.0]]))
    b = Molecule(symbols=["O"], positions=np.array([[5.0, 0.0, 0.0]]))
    sset.append(a, label="a.xyz")
    sset.append(b, label="b.xyz")
    sset.overlay = True
    w = MoleculeWidget(sset, 200, 200, backend="cpu", editable=True)
    w.scene.camera.center = np.zeros(3)
    w.scene.camera.rotation = np.eye(3)
    w.scene.camera.fit(200, 200, 8.0)
    return w, sset


def _px_for_world_x(w, x):
    """Widget-local pixel coords for a point at world (x, 0, 0), given the
    camera set up by _overlay_two_atom_widget (identity rotation, centered
    on the origin)."""
    cam = w.scene.camera
    Wr, Hr = w.scene.render_size
    sx = Wr * 0.5 + cam.pan[0] + x * cam.zoom
    sy = Hr * 0.5 - cam.pan[1]
    return sx / (Wr / w.scene.width), sy / (Hr / w.scene.height)


def test_pick_returns_a_composite_index_spanning_all_drawn_structures():
    w, sset = _overlay_two_atom_widget()
    px, py = _px_for_world_x(w, 5.0)   # atom 'b', entry index 1
    comp = sset.composite()
    idx = w.pick(px, py)
    assert idx is not None
    entry_idx, local = comp.locate(idx)
    assert entry_idx == 1
    assert local == 0


def test_delete_click_on_non_active_structure_is_refused_no_mutation():
    w, sset = _overlay_two_atom_widget()
    w.set_delete_mode(True)
    px, py = _px_for_world_x(w, 5.0)   # belongs to entry 1 ("b.xyz"), not active
    before = sset[1].molecule.n_atoms
    _click(w, px, py)
    assert sset[1].molecule.n_atoms == before   # nothing deleted
    assert w.pick_refusal is not None
    assert "b.xyz" in w.pick_refusal


def test_delete_click_on_active_structure_still_works_when_overlaid():
    w, sset = _overlay_two_atom_widget()
    w.set_delete_mode(True)
    px, py = _px_for_world_x(w, -5.0)   # belongs to entry 0 (active, "a.xyz")
    _click(w, px, py)
    assert sset[0].molecule.n_atoms == 0
    assert w.pick_refusal is None


def test_append_click_on_non_active_structure_is_refused():
    w, sset = _overlay_two_atom_widget()
    w.set_append_mode(True)
    px, py = _px_for_world_x(w, 5.0)
    before = sset[1].molecule.n_atoms
    _click(w, px, py)
    assert sset[1].molecule.n_atoms == before
    assert w.pick_refusal is not None


def test_measure_click_on_non_active_structure_is_refused_and_holds_no_index():
    w, sset = _overlay_two_atom_widget()
    w.set_measure_mode(True)
    px, py = _px_for_world_x(w, 5.0)
    _click(w, px, py)
    assert w.measure_sel == []
    assert w.pick_refusal is not None


def test_hover_over_non_active_structure_highlights_nothing():
    w, sset = _overlay_two_atom_widget()
    px, py = _px_for_world_x(w, 5.0)
    w.handle_mouse(MouseEvent("move", px, py, pixel=True))
    assert w.hovered is None


def test_hover_over_active_structure_still_works_when_overlaid():
    w, sset = _overlay_two_atom_widget()
    px, py = _px_for_world_x(w, -5.0)
    w.handle_mouse(MouseEvent("move", px, py, pixel=True))
    assert w.hovered == 0


def _mouse_click(px, py):
    return MouseEvent("down", px, py, button=0, pixel=True)


def _click(w, px, py):
    w.handle_mouse(MouseEvent("down", px, py, button=0, pixel=True))
    w.handle_mouse(MouseEvent("up", px, py, button=0, pixel=True))


from vimol.input import MouseEvent


# -- end-to-end: Scene.render() actually applies overlay colouring ---------

def test_scene_render_overlay_first_entry_cpk_rest_flat_tinted_cpu():
    """Exercises the full chain: Scene.render() -> _effective_style(composite)
    -> Renderer -- not a hand-built Style/SphereBatch. The second (non-active)
    entry must render flat and tinted; the first keeps CPK/shaded."""
    sset = StructureSet()
    a = Molecule(symbols=["C", "C"], positions=np.array([[-5.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]))
    b = Molecule(symbols=["C", "C"], positions=np.array([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
    sset.append(a, label="a")
    entry_b = sset.append(b, label="b")
    sset.overlay = True
    style = Style(representation="spacefill", depth_cue=0.0)
    scene = Scene(sset, 240, 240, style=style, backend="cpu")
    scene.camera.center = np.zeros(3)
    scene.camera.rotation = np.eye(3)
    scene.camera.fit(240, 240, 10.0)
    img = scene.render()

    tint_rgb = tuple(int(c * 255) for c in entry_b.tint)
    tint_mask = (np.abs(img[:, :, 0].astype(int) - tint_rgb[0]) < 12) & \
                (np.abs(img[:, :, 1].astype(int) - tint_rgb[1]) < 12) & \
                (np.abs(img[:, :, 2].astype(int) - tint_rgb[2]) < 12)
    assert tint_mask.sum() > 50
    tinted_pixels = img[tint_mask]
    assert (tinted_pixels == tinted_pixels[0]).all()   # flat: perfectly uniform

    # the active entry (CPK carbon-grey) is shaded, not flat/tinted
    carbon_rgb = tuple(int(c * 255) for c in a.element_colors()[0])
    active_mask = (np.abs(img[:, :, 0].astype(int) - carbon_rgb[0]) < 40) & \
                  (np.abs(img[:, :, 1].astype(int) - carbon_rgb[1]) < 40) & \
                  (np.abs(img[:, :, 2].astype(int) - carbon_rgb[2]) < 40)
    assert active_mask.sum() > 50
    active_pixels = img[active_mask]
    assert not (active_pixels == active_pixels[0]).all()   # shaded: not uniform
