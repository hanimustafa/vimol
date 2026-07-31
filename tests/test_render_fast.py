"""The numba kernel (_render_fast) must stay bit-identical to the numpy path.

The shading math exists twice -- render.py's vectorized numpy and
_render_fast's compiled per-pixel loops -- under a manual mirroring contract
(see _render_fast's module docstring). These tests are the net that catches
drift: every eligible frame is rendered through both paths and compared
byte-for-byte. They skip when numba is not installed (the numpy path is then
the only path, exercised by the rest of the suite).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vimol import _render_fast as _fast
from vimol import parsers, render
from vimol.bonds import ensure_bonds
from vimol.camera import Camera
from vimol.molecule import Molecule
from vimol.render import Renderer, Style

EX = os.path.join(os.path.dirname(__file__), "..", "examples")

needs_numba = pytest.mark.skipif(not _fast.available(),
                                 reason="numba not installed")


@pytest.fixture(scope="module")
def kernel_ready():
    """Force the kernel to be compiled (or definitively failed) before the
    module's tests run. Goes through warm_sync(), never a direct _run_once()
    call -- an earlier test's Renderer() may already have kicked off
    warm_async()'s background compile, and calling numba's compile machinery
    a second time concurrently is what actually caused the crash this test
    module exists to guard against (see test_kernel_ready_is_race_safe)."""
    if not _fast.available():
        pytest.skip("numba not installed")
    assert _fast.warm_sync(), "numba present but kernel failed to compile"


def _c60():
    mol = parsers.load(os.path.join(EX, "c60.xyz"))
    ensure_bonds(mol)
    return mol


def _c60_arrows():
    mol = _c60()
    rng = np.random.default_rng(0)
    mol.add_vector_field(rng.normal(size=(mol.n_atoms, 3)) * 0.5,
                         color=(1.0, 0.5, 0.2), scale=1.0, radius=0.05)
    return mol


def _camera(mol):
    cam = Camera(center=mol.centroid(), extent=mol.radius_of_gyration_extent())
    cam.fit(640, 480, mol.radius_of_gyration_extent() + 0.5)
    return cam


def _flat_mask(mol):
    flat = np.zeros(mol.n_atoms, bool)
    flat[::3] = True
    return flat


# (label, molecule factory, style kwargs, W, H, kernel_eligible)
# 640x480 C60 is >250k px with 150 primitives, so those frames run
# multi-banded -- the prange path is exercised, not just band 0.
CASES = [
    ("ball_and_stick", _c60, {}, 640, 480, True),
    ("transparent", _c60, {"transparent": True}, 640, 480, True),
    ("spacefill", _c60, {"representation": "spacefill"}, 320, 200, True),
    ("licorice_odd_size", _c60, {"representation": "licorice"}, 517, 311, True),
    ("wireframe", _c60, {"representation": "wireframe"}, 200, 150, True),
    ("flat_mask", _c60, {"flat_mask": "MASK"}, 640, 480, True),
    ("flat_mask_transparent", _c60,
     {"flat_mask": "MASK", "transparent": True}, 640, 480, True),
    ("no_depth_cue", _c60, {"depth_cue": 0.0}, 640, 480, True),
    ("custom_bg", _c60, {"background": (0.9, 0.1, 0.3)}, 640, 480, True),
    ("non_pow2_shininess", _c60, {"shininess": 17.0}, 640, 480, False),
    ("arrows", _c60_arrows, {}, 640, 480, False),
    ("arrows_transparent", _c60_arrows, {"transparent": True}, 640, 480, False),
    ("empty_molecule", lambda: Molecule([], np.zeros((0, 3))), {}, 100, 80, False),
]


def _build(case):
    label, mol_factory, style_kwargs, W, H, eligible = case
    mol = mol_factory()
    kwargs = dict(style_kwargs)
    if kwargs.get("flat_mask") == "MASK":
        kwargs["flat_mask"] = _flat_mask(mol)
    return mol, _camera(mol), Style(**kwargs), W, H, eligible


def _render_numpy(mol, cam, style, W, H, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(_fast, "ready", lambda: False)
        return Renderer(W, H).render(mol, cam, style)


@needs_numba
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_kernel_bit_identical_to_numpy(case, kernel_ready, monkeypatch):
    mol, cam, style, W, H, eligible = _build(case)
    frame_numpy = _render_numpy(mol, cam, style, W, H, monkeypatch)

    # count kernel entries so an always-false gate can't fake a green parity
    calls = []
    real = _fast.render_frame
    monkeypatch.setattr(_fast, "render_frame",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    frame_fast = Renderer(W, H).render(mol, cam, style)

    assert bool(calls) == eligible, (
        f"kernel {'skipped an eligible' if eligible else 'ran an ineligible'} frame")
    assert frame_fast.shape == frame_numpy.shape
    assert np.array_equal(frame_fast, frame_numpy), (
        f"max abs diff {np.abs(frame_fast.astype(int) - frame_numpy.astype(int)).max()}")


def test_transparent_alpha_semantics():
    """RGBA frames: alpha is 0/255 coverage; rgb matches an opaque render
    on a black background (undrawn premultiplied-zero, drawn shaded
    identically). Locks output while the buffer layout changes underneath."""
    mol = _c60()
    cam = _camera(mol)
    rgba = Renderer(640, 480).render(mol, cam, Style(transparent=True))
    rgb_black = Renderer(640, 480).render(mol, cam, Style(background=(0, 0, 0)))
    assert rgba.shape == (480, 640, 4) and rgba.dtype == np.uint8
    alpha = rgba[..., 3]
    assert set(np.unique(alpha)) == {0, 255}
    assert np.array_equal(rgba[..., :3], rgb_black)
    assert not rgba[..., :3][alpha == 0].any()


def test_renderer_reuse_across_style_switches():
    """One Renderer alternating transparent/opaque must match fresh
    renderers -- persistent buffers may not leak state across styles."""
    mol = _c60()
    cam = _camera(mol)
    st_t, st_o = Style(transparent=True), Style()
    r = Renderer(320, 240)
    got = [r.render(mol, cam, st_t), r.render(mol, cam, st_o),
           r.render(mol, cam, st_t)]
    want = [Renderer(320, 240).render(mol, cam, s) for s in (st_t, st_o, st_t)]
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


# Cold-cache numba compile measured at ~30 s/attempt on an M-series laptop.
# This bound is deliberately ~10x that: it exists to catch a deadlock, not
# to police how fast someone's machine compiles.
_COLD_COMPILE_TIMEOUT = 300


def test_kernel_ready_is_race_safe(tmp_path):
    """A second, unsynchronized caller forcing readiness while warm_async's
    background thread is already mid-compile must not crash the process.

    This is a regression test for a real crash: numba's dispatcher compile
    is not safe to enter from two threads at once for the same
    not-yet-compiled function. It reproduced reliably (SIGABRT/SIGSEGV, ~75%
    of runs) with a cold cache when something called the old private
    ``_warm()`` directly (as this test module's own ``kernel_ready`` fixture
    used to) while warm_async's thread -- started by an earlier test's
    Renderer() -- was still compiling. warm_sync() fixes it by waiting for
    whoever claimed the compile first instead of re-entering it.

    Needs a fresh NUMBA_CACHE_DIR per attempt: a warm cache compiles too
    fast to open the race window at all, which is why every fixture/bench
    caller elsewhere in this repo deletes it before measuring compile time.

    That cold compile is the whole cost of this test: measured at ~30 s per
    attempt on an M-series laptop (0.2 s once cached), NOT the "~1-3 s"
    warm_async's docstring quotes. warm_sync() is therefore called with no
    timeout of its own -- exactly as the kernel_ready fixture above and
    bench/speed.py call it. An inner deadline only adds a second, tighter
    bound that turns a merely slow machine into a failure claiming the
    concurrent-compile crash came back. subprocess.run's timeout is the one
    real bound, set an order of magnitude above the measured cost so that
    reaching it means something is genuinely wrong (a deadlock in warm_sync
    is a plausible regression of this very code, so it must fail rather than
    skip).
    """
    if not _fast.available():
        pytest.skip("numba not installed")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    script = (
        "import time\n"
        "from vimol import _render_fast as _fast\n"
        "_fast.warm_async()\n"
        "time.sleep(0.05)\n"           # give the background thread a head start
        "assert _fast.warm_sync(), 'warm_sync reported the kernel not ready'\n"
    )
    for i in range(3):
        env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / f"cache{i}"))
        try:
            out = subprocess.run([sys.executable, "-c", script],
                                 env=env, cwd=src, capture_output=True, text=True,
                                 timeout=_COLD_COMPILE_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"attempt {i}: cold compile did not finish within "
                f"{_COLD_COMPILE_TIMEOUT}s -- warm_sync is deadlocked, or this "
                f"machine is far slower than the ~30s this is budgeted for")
        assert out.returncode >= 0, (
            f"attempt {i}: killed by signal {-out.returncode} -- this is the "
            f"concurrent-compile crash regressing\n{out.stderr}")
        assert out.returncode == 0, (
            f"attempt {i}: exit {out.returncode}\n{out.stderr}")


def test_numba_cache_dir_kept_out_of_source_tree():
    """Importing _render_fast must point numba's disk cache outside the
    package dir (the source tree is Dropbox-synced; cache churn there is
    noise). setdefault only -- an explicit user setting wins."""
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, vimol._render_fast; print(os.environ['NUMBA_CACHE_DIR'])"],
        env=env, cwd=src, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    cache_dir = out.stdout.strip()
    assert cache_dir
    pkg_dir = os.path.abspath(os.path.join(src, "vimol"))
    assert not os.path.abspath(cache_dir).startswith(pkg_dir)
