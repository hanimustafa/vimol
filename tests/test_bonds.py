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

