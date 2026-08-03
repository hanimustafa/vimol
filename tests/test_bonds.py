"""Bond perception: correctness the vectorized rewrite must preserve."""
import os

import numpy as np
import pytest

from vimol.bonds import perceive_bonds, ensure_bonds
from vimol.molecule import Molecule
from vimol.parsers import load_all

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(name):
    return load_all(os.path.join(EX, f"{name}.xyz"))[0]


def _reference(mol, tolerance=0.45):
    """Brute-force O(N^2) ground truth, written for obviousness not speed."""
    pos = np.asarray(mol.positions, dtype=np.float64)
    cov = np.asarray(mol.covalent_radii(), dtype=np.float64)
    out = set()
    for a in range(mol.n_atoms):
        for b in range(a + 1, mol.n_atoms):
            d2 = float(((pos[a] - pos[b]) ** 2).sum())
            cut = cov[a] + cov[b] + tolerance
            if 0.16 <= d2 <= cut * cut:
                out.add((a, b))
    return out


# Counts locked in from the implementation that shipped before the rewrite.
@pytest.mark.parametrize("name, n_atoms, n_bonds", [
    ("water", 3, 2),
    ("methane", 5, 4),
    ("benzene", 12, 12),
    ("hydrocarbon", 35, 34),
    ("c60", 60, 90),
])
def test_known_molecules_keep_their_chemistry(name, n_atoms, n_bonds):
    mol = _load(name)
    assert mol.n_atoms == n_atoms
    assert len(perceive_bonds(mol)) == n_bonds


@pytest.mark.parametrize("name", ["water", "methane", "benzene",
                                  "hydrocarbon", "c60"])
def test_matches_brute_force_ground_truth(name):
    mol = _load(name)
    got = {(a, b) for a, b, _ in perceive_bonds(mol)}
    assert got == _reference(mol)


@pytest.mark.parametrize("offset", [0.0, 50.0, 1000.0, 5000.0])
def test_bonds_are_translation_invariant(offset):
    """Connectivity cannot depend on where the molecule sits in space.

    This is the guard on the working precision: float16 silently invents and
    drops bonds once coordinates get far from the origin (it loses all 339
    bonds of a real structure at +5000 A), which is why the distance block is
    built in float32.
    """
    mol = _load("hydrocarbon")
    ref = {(a, b) for a, b, _ in perceive_bonds(mol)}
    moved = Molecule()
    for sym, p in zip(mol.symbols, np.asarray(mol.positions) + offset):
        moved.add_atom(sym, float(p[0]), float(p[1]), float(p[2]))
    assert {(a, b) for a, b, _ in perceive_bonds(moved)} == ref


def test_result_is_independent_of_chunk_size(monkeypatch):
    """Chunking bounds peak memory; it must not change a single bond."""
    from vimol import bonds as bonds_mod
    mol = _load("c60")
    ref = perceive_bonds(mol)
    for cap in (1 << 10, 1 << 14, 1 << 30):     # tiny -> many chunks, huge -> one
        monkeypatch.setattr(bonds_mod, "_CHUNK_BYTES", cap)
        assert perceive_bonds(mol) == ref


def test_peak_block_stays_bounded_for_big_inputs():
    """A large structure must not try to allocate an N^2 block up front."""
    from vimol.bonds import _chunk_rows, _CHUNK_BYTES
    for n in (100, 5_000, 100_000):
        rows = _chunk_rows(n)
        assert 1 <= rows <= n
        assert rows * n * 3 * 4 <= max(_CHUNK_BYTES, n * 3 * 4)


def test_degenerate_inputs():
    assert perceive_bonds(Molecule()) == []
    one = Molecule()
    one.add_atom("C", 0.0, 0.0, 0.0)
    assert perceive_bonds(one) == []


def test_coincident_atoms_do_not_bond():
    """Duplicate/overlapping atoms are rejected, not bonded to each other."""
    mol = Molecule()
    mol.add_atom("C", 0.0, 0.0, 0.0)
    mol.add_atom("C", 0.0, 0.0, 0.0)
    assert perceive_bonds(mol) == []


def test_max_bonds_per_atom_is_enforced():
    """A pathologically crowded atom does not exceed the cap."""
    mol = Molecule()
    mol.add_atom("C", 0.0, 0.0, 0.0)
    for k in range(12):                       # 12 H at bonding distance
        ang = 2 * np.pi * k / 12
        mol.add_atom("H", float(np.cos(ang)), float(np.sin(ang)), 0.0)
    bonds = perceive_bonds(mol, max_bonds_per_atom=4)
    counts = np.zeros(mol.n_atoms, int)
    for a, b, _ in bonds:
        counts[a] += 1
        counts[b] += 1
    assert counts.max() <= 4


def test_ensure_bonds_does_not_overwrite_existing():
    mol = _load("water")
    mol.bonds = [(0, 1, 1)]
    assert ensure_bonds(mol).bonds == [(0, 1, 1)]


# -- deferred perception (bonds are computed when a structure is drawn) -----
def _set_of(name, n=3):
    from vimol.structures import StructureSet
    sset = StructureSet()
    for i in range(n):
        sset.append(_load(name), label=f"f{i}")
    return sset


def test_loading_does_not_perceive_bonds():
    """Appending a frame must cost nothing: that is the whole point."""
    sset = _set_of("c60")
    assert all(not e.molecule.bonds for e in sset.entries)
    assert all(not e.molecule.bonds_perceived for e in sset.entries)


def test_drawing_perceives_only_what_is_drawn():
    sset = _set_of("c60")
    sset.composite()                       # draws the active entry only
    assert sset.entries[0].molecule.bonds
    assert not sset.entries[1].molecule.bonds
    assert not sset.entries[2].molecule.bonds


def test_selecting_a_frame_perceives_it_on_the_spot():
    """The 'user selects a frame we have not bonded yet' case."""
    sset = _set_of("c60")
    sset.composite()
    sset.set_active(2)
    sset.invalidate()
    sset.composite()
    assert sset.entries[2].molecule.bonds


def test_overlay_perceives_every_marked_frame():
    sset = _set_of("c60")
    sset.overlay = True
    for e in sset.entries:
        e.marked = True
    sset.composite()
    assert all(e.molecule.bonds for e in sset.entries)


def test_auto_bonds_off_never_perceives():
    """--no-bonds must survive the move to draw-time perception."""
    sset = _set_of("c60")
    sset.auto_bonds = False
    sset.composite()
    assert not sset.entries[0].molecule.bonds


def test_bond_tolerance_is_honoured_at_draw_time():
    """The CLI's --bond-tolerance has to reach the deferred call."""
    tight, loose = _set_of("hydrocarbon", 1), _set_of("hydrocarbon", 1)
    tight.bond_tolerance = -0.4                 # nothing is close enough
    loose.bond_tolerance = 0.45
    tight.composite()
    loose.composite()
    assert len(tight.entries[0].molecule.bonds) < len(loose.entries[0].molecule.bonds)


def test_bondless_molecule_is_perceived_only_once(monkeypatch):
    """A molecule with no bonds in range must not re-perceive every redraw."""
    from vimol import bonds as bonds_mod
    from vimol.molecule import Molecule
    from vimol.structures import StructureSet

    far = Molecule()                            # two atoms far past any cutoff
    far.add_atom("He", 0.0, 0.0, 0.0)
    far.add_atom("He", 40.0, 0.0, 0.0)
    sset = StructureSet()
    sset.append(far, label="far")

    calls = []
    real = bonds_mod.perceive_bonds
    monkeypatch.setattr(bonds_mod, "perceive_bonds",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for _ in range(5):
        sset.invalidate()
        sset.composite()
    assert far.bonds == []
    assert len(calls) == 1, f"re-perceived {len(calls)} times"
