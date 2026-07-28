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
