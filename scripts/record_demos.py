#!/usr/bin/env python3
"""Record the README's terminal demos.

    pip install pyte Pillow
    python3 scripts/record_demos.py               # all of them
    python3 scripts/record_demos.py protein align # or just these

Each demo drives a real viewer through :mod:`termshot` and writes a GIF into
``docs/media``. Demo inputs live in ``docs/media/demo`` and ``examples``,
except the CREST ensemble of the alignment demo, which is not in the
repository -- point ``VIMOL_DEMO_ENSEMBLE`` at one to record that demo.

ffmpeg does the GIF encoding; without it the frames are left as PNGs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import termshot as ts                                            # noqa: E402
import vimol                                                     # noqa: E402
from vimol import app                                            # noqa: E402

COLS, ROWS, FONT_SIZE = 108, 30, 15
MEDIA = os.path.join(ROOT, "docs", "media")
DEMO = os.path.join(MEDIA, "demo")
EXAMPLES = os.path.join(ROOT, "examples")
BACKEND = os.environ.get("VIMOL_DEMO_BACKEND", "auto")
PROMPT_DIR = "~/proteins"


# -- assembling frames -----------------------------------------------------

class Recorder:
    """Frames with per-frame durations, encoded through ffmpeg's concat
    demuxer so a two-second pause costs one PNG rather than thirty."""

    def __init__(self, canvas: ts.Canvas):
        self.canvas = canvas
        self.frames = []            # (image, seconds)

    def add(self, image, seconds=0.08):
        self.frames.append((image, seconds))

    def shot(self, term, seconds=0.08, **kw):
        self.add(self.canvas.render(term, **kw), seconds)

    def hold(self, seconds=1.0):
        if self.frames:
            image, _ = self.frames[-1]
            self.frames.append((image, seconds))

    def save(self, path, fps=16, colors=160):
        tmp = tempfile.mkdtemp(prefix="vimol-rec-")
        listing = os.path.join(tmp, "frames.txt")
        with open(listing, "w") as f:
            for i, (image, seconds) in enumerate(self.frames):
                name = os.path.join(tmp, "f%05d.png" % i)
                image.save(name)
                f.write("file '%s'\nduration %.3f\n" % (name, seconds))
            f.write("file '%s'\n" % name)      # concat wants the last one twice
        if not shutil.which("ffmpeg"):
            print("  ffmpeg not found; frames left in", tmp)
            return
        vf = ("fps=%d,split[a][b];[a]palettegen=max_colors=%d:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
              % (fps, colors))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", listing, "-filter_complex", vf,
                        "-loop", "0", path], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        print("  wrote %s (%.1f MB, %d frames, %.1fs)"
              % (os.path.relpath(path, ROOT), os.path.getsize(path) / 1e6,
                 len(self.frames), sum(s for _, s in self.frames)))


def type_command(rec, command, cwd=PROMPT_DIR):
    """The shell prompt, typed one character at a time."""
    term = ts.Terminal(COLS, ROWS)
    prefix = "\x1b[38;2;120;160;230m%s\x1b[0m \x1b[38;2;110;200;170m$\x1b[0m " % cwd
    plain = len(cwd) + 3
    rec.shot(term, 0.5, cursor=(plain, 0))
    for i in range(1, len(command) + 1):
        term.reset()
        term.feed((prefix + command[:i]).encode())
        rec.shot(term, 0.045, cursor=(plain + i, 0))
    rec.hold(0.7)
    term.reset()
    rec.shot(term, 0.25)


def glide(rec, session, frm, to, steps=10, seconds=0.05, click_at_end=True,
          badge=None):
    """Walk the pointer from one screen pixel to another, then click."""
    for i in range(steps + 1):
        t = i / steps
        ease = t * t * (3 - 2 * t)
        at = (frm[0] + (to[0] - frm[0]) * ease, frm[1] + (to[1] - frm[1]) * ease)
        rec.shot(session.terminal, seconds, pointer=at, badge=badge)
    if click_at_end:
        return to
    return to


def click_bloom(rec, session, at, badge=None):
    """The click itself: the ring expands over four frames."""
    for k in (0.15, 0.45, 0.75, 1.0):
        rec.shot(session.terminal, 0.045, pointer=at, click=k, badge=badge)


def zoom(rec, session, notches, direction="up", seconds=0.05):
    """Scroll over the viewport, the way a user zooms."""
    w = session.viewer.widget
    ox, oy = session.viewer._img_origin_px
    at = (ox + w.scene.width / 2, oy + w.scene.height / 2)
    for _ in range(notches):
        session.mouse("scroll", at, button=None, scroll=direction)
        session.capture()
        rec.shot(session.terminal, seconds)


def orbit(rec, session, dx, dy, steps=12, seconds=0.05, start=None):
    """Drag inside the viewport to rotate, capturing every step."""
    w = session.viewer.widget
    ox, oy = session.viewer._img_origin_px
    cx = ox + w.scene.width / 2 if start is None else start[0]
    cy = oy + w.scene.height / 2 if start is None else start[1]
    session.mouse("down", (cx, cy))
    for i in range(1, steps + 1):
        t = i / steps
        session.mouse("drag", (cx + dx * t, cy + dy * t))
        session.capture()
        rec.shot(session.terminal, seconds)
    session.mouse("up", (cx + dx, cy + dy))
    session.capture()
    rec.shot(session.terminal, seconds)


# -- the demos -------------------------------------------------------------

def demo_protein(canvas, font):
    """A. A bare .xyz turns out to be a protein, and 6 letters its residues."""
    rec = Recorder(canvas)
    type_command(rec, "vimol trpcage.xyz")

    mol = vimol.load(os.path.join(DEMO, "trpcage.xyz"))
    vimol.ensure_bonds(mol)
    style = app._default_representation(mol)
    assert style == "ribbon", "trpcage.xyz no longer opens as a protein"
    s = ts.Session(COLS, ROWS, font, molecule=mol, source_path="trpcage.xyz",
                   style=style, backend=BACKEND)
    s.capture()
    rec.shot(s.terminal, 0.1)
    rec.hold(1.8)
    orbit(rec, s, 150, 40, steps=14)
    rec.hold(0.9)

    s.key("6")
    s.capture()
    rec.shot(s.terminal, 0.1, badge="6")
    rec.hold(2.6)
    zoom(rec, s, 4)
    rec.hold(1.4)
    orbit(rec, s, 150, 30, steps=16)
    rec.hold(2.8)
    s.close()
    return rec


def demo_align(canvas, font):
    """B. One structure against a whole conformer set."""
    ensemble = os.environ.get("VIMOL_DEMO_ENSEMBLE")
    if not ensemble or not os.path.exists(ensemble):
        print("  skipped: set VIMOL_DEMO_ENSEMBLE to a multi-frame .xyz")
        return None
    rec = Recorder(canvas)
    # --style ribbon because the protein auto-detection only runs on the
    # single-file path; opening several files always starts ball-and-stick,
    # and 73 superimposed proteins as sticks is a thicket.
    type_command(rec, "vimol 5awl_prepared.xyz crest_ensemble.xyz --style ribbon")

    sset = app._build_structure_set(
        [os.path.join(DEMO, "5awl_prepared.xyz"), ensemble],
        no_bonds=False, tolerance=0.45)
    s = ts.Session(COLS, ROWS, font, structures=sset, style="ribbon",
                   source_path="5awl_prepared.xyz", backend=BACKEND)
    s.capture()
    rec.shot(s.terminal, 0.1)
    rec.hold(2.0)

    # The ALL control of the ensemble's group: the viewer records where it
    # drew each one, so the pointer is aimed at the real thing.
    span = next(sp for sp in s.viewer._list_group_all_spans if sp[3] == 1)
    row, c0, c1, first, end = span
    target = s.cell_px((c0 + c1 - 1) / 2, row)
    start = (target[0] + 210, target[1] + 150)
    glide(rec, s, start, target, steps=12)
    click_bloom(rec, s, target)
    s.viewer._toggle_list_group_all(first, end)
    s.capture()
    rec.shot(s.terminal, 0.1, pointer=target)
    rec.hold(2.2)

    s.key("r")
    s.capture()
    rec.shot(s.terminal, 0.1, pointer=target, badge="r")
    rec.hold(3.0)
    zoom(rec, s, 4)
    rec.hold(0.8)
    orbit(rec, s, 150, 30, steps=16)
    rec.hold(2.4)
    s.close()
    return rec


def demo_methyl(canvas, font):
    """C. Growing a methyl group onto threonine's side-chain hydroxyl."""
    HYDROXYL_H = 8          # examples/build_examples.py names the index
    rec = Recorder(canvas)
    type_command(rec, "vimol threonine.xyz")

    mol = vimol.load(os.path.join(EXAMPLES, "threonine.xyz"))
    vimol.ensure_bonds(mol)
    s = ts.Session(COLS, ROWS, font, molecule=mol, source_path="threonine.xyz",
                   backend=BACKEND)
    s.capture()
    rec.shot(s.terminal, 0.1)
    rec.hold(1.4)
    zoom(rec, s, 3)
    rec.hold(0.8)

    s.key("a")
    s.capture()
    rec.shot(s.terminal, 0.1, badge="a")
    rec.hold(2.0)

    target = s.to_screen(s.screen_of(HYDROXYL_H))
    assert s.viewer.widget.pick(*s.screen_of(HYDROXYL_H)) == HYDROXYL_H, \
        "the hydroxyl hydrogen is not where the pointer is aimed"
    start = (target[0] - 230, target[1] + 160)
    glide(rec, s, start, target, steps=14)
    s.mouse("move", target)
    s.capture()
    rec.shot(s.terminal, 0.1, pointer=target)
    rec.hold(1.2)
    click_bloom(rec, s, target)

    before = s.viewer.structures.active.molecule.n_atoms
    s.click(target)
    after = s.viewer.structures.active.molecule
    assert after.n_atoms == before + 3, "append did not add a carbon and 3 H"
    assert after.formula() == "C5H11NO3", after.formula()
    s.capture()
    rec.shot(s.terminal, 0.1, pointer=target)
    rec.hold(2.6)
    orbit(rec, s, 120, 20, steps=12)
    rec.hold(2.4)
    s.close()
    return rec


def demo_library(canvas, font):
    """D. The three-line snippet, and a still of what it paints."""
    snippet = os.path.join(EXAMPLES, "inset.py")
    term = ts.Terminal(COLS, ROWS)
    prefix = ("\x1b[38;2;120;160;230m%s\x1b[0m \x1b[38;2;110;200;170m$\x1b[0m "
              % PROMPT_DIR)
    term.feed((prefix + "python3 inset.py").encode())
    out = subprocess.run([sys.executable, snippet], capture_output=True,
                         check=True, cwd=EXAMPLES,
                         env={**os.environ, "COLUMNS": str(COLS),
                              "LINES": str(ROWS)})
    term.feed(b"\r\n")
    term.feed(out.stdout)
    image = canvas.render(term)
    path = os.path.join(MEDIA, "library.png")
    image.save(path)
    print("  wrote %s (%.2f MB)" % (os.path.relpath(path, ROOT),
                                    os.path.getsize(path) / 1e6))
    return None


DEMOS = {
    "protein": (demo_protein, "protein.gif"),
    "align": (demo_align, "align.gif"),
    "methyl": (demo_methyl, "methyl.gif"),
    "library": (demo_library, None),
}


def main(argv):
    wanted = argv or list(DEMOS)
    unknown = [n for n in wanted if n not in DEMOS]
    if unknown:
        print("unknown demo(s): %s\nknown: %s" % (", ".join(unknown),
                                                  ", ".join(DEMOS)))
        return 2
    font = ts.Font(FONT_SIZE)
    canvas = ts.Canvas(font)
    print("terminal %dx%d cells of %dx%d px -> %dx%d image"
          % (COLS, ROWS, font.cell_w, font.cell_h, *canvas.size(COLS, ROWS)))
    for name in wanted:
        fn, out = DEMOS[name]
        print(name + ":")
        rec = fn(canvas, font)
        if rec is not None and out:
            rec.save(os.path.join(MEDIA, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
