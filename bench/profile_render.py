"""One-off render profiler: CPU vs GL, per-stage timing, cProfile hotspots."""
import cProfile, io, os, pstats, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vimol import parsers
from vimol.scene import Scene
from vimol.render import Style

mol = parsers.load(os.path.join(os.path.dirname(__file__), "..", "examples", "c60.xyz"))
print(f"molecule: {mol.n_atoms} atoms, {len(mol.bonds)} bonds")

# A full-screen retina terminal viewport, interactive quality (ss=1) and
# settle quality (ss=2), at render_scale 1.0.
for W, H, ss, label in [(2560, 1440, 1, "interactive 2560x1440 ss1"),
                        (2560, 1440, 2, "settle      2560x1440 ss2")]:
    for backend in ("cpu", "gl"):
        try:
            scene = Scene(mol, W, H, supersample=ss, backend=backend)
        except Exception as e:
            print(f"{label} [{backend}]: UNAVAILABLE ({type(e).__name__}: {e})")
            continue
        scene.render()  # warmup
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            img = scene.render()
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000
        print(f"{label} [{backend}] actual={scene.backend}: "
              f"median {np.median(ts):7.1f} ms  min {ts.min():7.1f} ms  -> {img.shape}")

# cProfile the CPU renderer at settle resolution
scene = Scene(mol, 2560, 1440, supersample=1, backend="cpu")
scene.render()
pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    scene.render()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
print("\n=== cProfile: CPU backend, 5 frames @ 2560x1440 ss1 ===")
print(s.getvalue())

# Time the kitty encode step at display size
from vimol import kitty
img = scene.render()
for lvl in (1, 6):
    t0 = time.perf_counter()
    for _ in range(5):
        data = kitty.encode_image(img, image_id=1, compress_level=lvl, transmit="direct")
    dt = (time.perf_counter() - t0) / 5 * 1000
    print(f"kitty encode direct level={lvl}: {dt:7.1f} ms  ({len(data)//1024} KiB)")
t0 = time.perf_counter()
for _ in range(20):
    data = kitty.encode_image(img, image_id=1, transmit="shm")
dt = (time.perf_counter() - t0) / 20 * 1000
print(f"kitty encode shm:              {dt:7.1f} ms  ({len(data)//1024} KiB)")
kitty.shm_cleanup()
