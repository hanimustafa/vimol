"""Bond perception: correctness the vectorized rewrite must preserve."""
import os
import time

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


# -- large overlays slice their bond perception ------------------------------
def _make_chain(n_atoms, seed=0):
    """A synthetic, densely-bonded chain -- real work for the perceiver,
    without needing a big example file on disk."""
    rng = np.random.default_rng(seed)
    positions = np.cumsum(rng.normal(scale=1.2, size=(n_atoms, 3)), axis=0)
    return Molecule(symbols=["C"] * n_atoms, positions=positions.astype(float))


def _big_overlay(n_frames=400, n_atoms=120, marked=True):
    from vimol.structures import StructureSet
    sset = StructureSet()
    sset.overlay = True
    for i in range(n_frames):
        e = sset.append(_make_chain(n_atoms, seed=i), label=f"f{i}")
        e.marked = marked
    return sset


def _drain(sset, limit=2000):
    """Run slices the way the render loop does, until the queue empties."""
    for _ in range(limit):
        if not sset.bonds_pending():
            return
        sset.poll_bonds()
    raise AssertionError("bond queue never drained")


def test_large_overlay_does_not_perceive_everything_in_one_call():
    """Marking hundreds of frames at once (the structure-list ALL button)
    used to perceive bonds for every one of them inside a single composite()
    call, blocking the whole single-threaded UI for as long as that took --
    ~17s for a real 5000-frame trajectory. Perception is now sliced: one
    call does a bounded amount of work and queues the rest.

    Asserted as "most frames are still unperceived and queued" rather than a
    wall-clock bound -- the contract is that the work is bounded per call,
    and a timing assertion would flake under load."""
    sset = _big_overlay()
    sset.composite()
    assert sset.bonds_pending(), "a large overlay must leave frames queued"
    perceived = sum(1 for e in sset.entries if e.molecule.bonds_perceived)
    assert perceived < len(sset.entries) / 2, (
        f"{perceived}/{len(sset.entries)} perceived in one call -- not sliced")


def test_sliced_overlay_eventually_bonds_every_frame():
    sset = _big_overlay()
    sset.composite()
    _drain(sset)
    assert all(e.molecule.bonds_perceived for e in sset.entries)
    assert all(e.molecule.bonds for e in sset.entries)


def test_drained_bonds_reach_the_composite():
    """The screen -- not just the molecules -- has to end up with the bonds.

    Every frame can be correctly perceived while the composite still serves
    a cached, bond-less flattening, since the cache key (_entry_state) does
    not include bonds. Only a drain-triggered invalidate closes that gap, so
    assert on the rendered artefact rather than on the molecules."""
    sset = _big_overlay()
    before = len(sset.composite().molecule.bonds)   # only the first slice's frames
    _drain(sset)
    after = len(sset.composite().molecule.bonds)
    total = sum(len(e.molecule.bonds) for e in sset.entries)
    assert after > before, "bonds landed on the molecules but never reached the composite"
    assert after == total, f"composite has {after} bonds, molecules have {total}"


def test_a_frame_that_cannot_be_perceived_does_not_wedge_the_queue(monkeypatch):
    """One unperceivable frame must not starve the frames behind it, or be
    retried forever."""
    from vimol import bonds as bonds_mod

    sset = _big_overlay(n_frames=20, n_atoms=60)
    victim = sset.entries[5].molecule
    real = bonds_mod.perceive_bonds

    def explode(mol, *a, **k):
        if mol is victim:
            raise RuntimeError("simulated perception failure")
        return real(mol, *a, **k)

    monkeypatch.setattr(bonds_mod, "perceive_bonds", explode)
    sset.composite()
    _drain(sset)
    assert not sset.bonds_pending()
    others = [e.molecule for e in sset.entries if e.molecule is not victim]
    assert all(m.bonds for m in others), "a failed frame starved the rest of the queue"


def test_editing_a_queued_frame_is_not_overwritten_by_its_own_perception():
    """A queued frame that gets edited must never be handed bonds computed
    from its pre-edit geometry: those indices point past the shortened atom
    list and raise straight out of both renderers.

    Perception is on the same thread as the edit and reads geometry when it
    runs, so there is no snapshot to go stale. Here the edit re-perceives
    (every edit ends in editor._reperceive), which leaves mol.bonds
    non-empty, so the drain skips the frame entirely -- the sibling test
    below covers the case where the slice does run on edited geometry."""
    from vimol import editor

    sset = _big_overlay(n_frames=200, n_atoms=120)
    sset.composite()
    victim = sset.entries[-1].molecule        # far enough back to still be queued
    assert not victim.bonds_perceived, "sanity: victim must still be queued"

    victim.positions = victim.positions[:40]
    victim.symbols = victim.symbols[:40]
    editor._reperceive(victim)                # every edit ends here
    _drain(sset)

    assert victim.n_atoms == 40
    highest = max((max(a, b) for a, b, _ in victim.bonds), default=-1)
    assert highest < victim.n_atoms, (
        f"stale bond index {highest} against {victim.n_atoms} atoms")


def test_a_queued_frame_is_perceived_from_its_current_geometry():
    """The case above's sibling: shrink a queued frame WITHOUT re-perceiving,
    so the slice is the thing that bonds it. It must bond what the molecule
    is now, not what it was when it was queued."""
    sset = _big_overlay(n_frames=200, n_atoms=120)
    sset.composite()
    victim = sset.entries[-1].molecule
    assert not victim.bonds_perceived and not victim.bonds, "sanity: still queued"

    victim.positions = victim.positions[:40]
    victim.symbols = victim.symbols[:40]
    _drain(sset)

    assert victim.bonds_perceived and victim.bonds
    highest = max(max(a, b) for a, b, _ in victim.bonds)
    assert highest < victim.n_atoms == 40, (
        f"bond index {highest} against {victim.n_atoms} atoms -- perceived pre-edit")


def test_closing_a_file_stops_the_screen_waiting_on_its_frames():
    """The queue is work the screen is waiting on, so it must follow the
    drawn set. A file closed mid-catch-up otherwise leaves its frames queued
    (and alive), and the render loop keeps spinning at a zero read timeout,
    holding off the supersampled settle, for structures that are gone."""
    sset = _big_overlay()
    sset.composite()
    assert sset.bonds_pending()
    sset.remove_range(1, len(sset))
    sset.composite()
    assert len(sset) == 1
    assert not sset.bonds_pending(), (
        f"{len(sset._bond_queue)} frames still queued after their file closed")


def test_clearing_the_overlay_stops_the_screen_waiting_on_its_frames():
    """Same, for clicking the strip's ALL button a second time."""
    sset = _big_overlay()
    sset.composite()
    assert sset.bonds_pending()
    for e in sset.entries:
        e.marked = False
    sset.active.marked = True
    sset.composite()
    assert not sset.bonds_pending(), (
        f"{len(sset._bond_queue)} frames still queued after the overlay cleared")


def test_one_tick_spends_one_slice_budget_across_every_caller():
    """composite() slices too, above its own cache check, and a drawing tick
    calls composite() more than once (highlight, then scene). Those must
    share the tick's allowance rather than stacking -- unshared, a tick that
    claims to cost one frame of latency measured 3.1x that."""
    from vimol.structures import _BOND_SLICE_SECONDS

    sset = _big_overlay(n_frames=600, n_atoms=120)
    sset.composite()

    def perceived():
        return sum(1 for e in sset.entries if e.molecule.bonds_perceived)

    before = perceived()
    sset.poll_bonds()          # opens the tick, spends the budget
    after_poll = perceived()
    sset.composite()           # same tick: allowance already spent
    sset.composite()
    after_draw = perceived()

    assert after_poll > before, "sanity: the tick's slice must do some work"
    assert after_draw == after_poll, (
        f"composite() perceived {after_draw - after_poll} more frames on a tick "
        "whose budget was already spent")
    # A fresh tick gets a fresh allowance.
    sset.poll_bonds()
    assert perceived() > after_draw, "next tick must get a fresh budget"


def test_small_overlay_still_perceives_in_the_one_call():
    """The ordinary case must not change: a handful of drawn frames still
    finishes inside composite(), with nothing left queued."""
    sset = _set_of("c60")
    sset.overlay = True
    for e in sset.entries:
        e.marked = True
    sset.composite()
    assert not sset.bonds_pending()
    assert all(e.molecule.bonds for e in sset.entries)


def test_active_frame_is_never_left_queued():
    """The active frame is what the editor acts on, and editor._neighbors
    reads mol.bonds with no fallback -- so it must be bonded on return from
    composite() however big the overlay behind it is."""
    sset = _big_overlay()
    sset.set_active(3)
    sset.composite()
    assert sset.active.molecule.bonds, "active frame left unbonded"


def test_auto_bonds_off_queues_nothing_on_a_large_overlay():
    """--no-bonds has to survive slicing too: nothing perceived, nothing
    queued, no catch-up work invented for a viewer that asked for none."""
    sset = _big_overlay()
    sset.auto_bonds = False
    sset.composite()
    assert not sset.bonds_pending()
    assert not any(e.molecule.bonds for e in sset.entries)


def test_bond_tolerance_reaches_the_sliced_path():
    tight = _big_overlay(n_frames=60, n_atoms=80)
    loose = _big_overlay(n_frames=60, n_atoms=80)
    tight.bond_tolerance = -0.4          # nothing is close enough
    loose.bond_tolerance = 0.45
    for s in (tight, loose):
        s.composite()
        _drain(s)
    tight_bonds = sum(len(e.molecule.bonds) for e in tight.entries)
    loose_bonds = sum(len(e.molecule.bonds) for e in loose.entries)
    assert tight_bonds < loose_bonds
