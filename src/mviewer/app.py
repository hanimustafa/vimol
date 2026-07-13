"""Command-line driver for mviewer.

    mviewer file.pdb                 # interactive viewer
    mviewer file.xyz --spin          # autospinning
    mviewer file.mol --render out.png --size 800x800
    mviewer file.pdb --kitty         # emit one frame to stdout (for pipes/embeds)
    mviewer file.xyz --info          # print structure info and exit
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .parsers import load_all, SUPPORTED_EXTENSIONS
from .bonds import ensure_bonds
from .render import Style
from .scene import Scene
from . import kitty
from .molecule import Molecule


def _parse_size(s: str):
    if "x" in s.lower():
        w, h = s.lower().split("x")
        return int(w), int(h)
    v = int(s)
    return v, v


def build_style(args) -> Style:
    st = Style(representation=args.style)
    if args.atom_scale is not None:
        st.atom_scale = args.atom_scale
    if args.bond_radius is not None:
        st.bond_radius = args.bond_radius
    if args.background is not None:
        st.background = _parse_color(args.background)
        st.transparent = False
    if args.transparent:
        st.transparent = True
    if args.opaque:
        st.transparent = False
    if args.no_depth_cue:
        st.depth_cue = 0.0
    return st


def _parse_color(s: str):
    s = s.strip().lstrip("#")
    if len(s) == 6:
        return tuple(int(s[i:i + 2], 16) / 255 for i in (0, 2, 4))
    parts = s.split(",")
    if len(parts) == 3:
        return tuple(float(p) for p in parts)
    raise argparse.ArgumentTypeError(f"bad color {s!r}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mviewer",
        description="A terminal molecular viewer using the Kitty graphics protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS),
    )
    p.add_argument("file", nargs="?", help="structure file (xyz/pdb/mol/sdf)")
    p.add_argument("--style", default="ball_and_stick",
                   choices=["ball_and_stick", "spacefill", "licorice", "wireframe"])
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "gl"],
                   help="rendering backend: numpy CPU raycaster, GPU (OpenGL, needs "
                        "mviewer[gl]), or auto (GPU if available, else CPU)")
    p.add_argument("--size", default="0x0", help="pixel size WxH for --render/--kitty (0=auto)")
    p.add_argument("--supersample", type=int, default=2, help="anti-aliasing factor for stills")
    p.add_argument("--rotate", nargs=2, type=float, metavar=("YAW", "PITCH"),
                   default=(20.0, -15.0), help="initial rotation in degrees")
    p.add_argument("--spin", action="store_true", help="autospin in interactive mode")
    p.add_argument("--render", metavar="PNG", help="render a still image to a PNG file and exit")
    p.add_argument("--kitty", action="store_true", help="emit one frame to stdout as Kitty graphics and exit")
    p.add_argument("--info", action="store_true", help="print structure info and exit")
    p.add_argument("--frame", type=int, default=0, help="frame/model index for multi-model files")
    p.add_argument("--atom-scale", type=float, default=None)
    p.add_argument("--bond-radius", type=float, default=None)
    p.add_argument("--background", type=str, default=None, help="hex or r,g,b background color (implies --opaque)")
    p.add_argument("--transparent", action="store_true", help="transparent background (RGBA cutout)")
    p.add_argument("--opaque", action="store_true", help="solid background (default for --render)")
    p.add_argument("--no-depth-cue", action="store_true")
    p.add_argument("--no-bonds", action="store_true", help="do not auto-perceive bonds")
    p.add_argument("--bond-tolerance", type=float, default=0.45)
    p.add_argument("--list-formats", action="store_true")
    p.add_argument("--version", action="store_true")
    return p


def _print_info(mol: Molecule):
    print(f"name:    {mol.name}")
    print(f"formula: {mol.formula()}")
    print(f"atoms:   {mol.n_atoms}")
    print(f"bonds:   {len(mol.bonds)}")
    ext = mol.radius_of_gyration_extent()
    print(f"extent:  {ext:.2f} A (max atom-centroid distance)")
    from collections import Counter
    comp = Counter(mol.symbols)
    print("composition: " + ", ".join(f"{k}:{v}" for k, v in sorted(comp.items())))


def main(argv: List[str] | None = None) -> int:
    args = make_parser().parse_args(argv)

    if args.version:
        from . import __version__
        print(f"mviewer {__version__}")
        return 0
    if args.list_formats:
        print("Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS))
        return 0
    if not args.file:
        make_parser().print_help()
        return 1
    if not os.path.exists(args.file):
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 2

    try:
        mols = load_all(args.file)
    except Exception as e:  # noqa: BLE001
        print(f"error: failed to parse {args.file}: {e}", file=sys.stderr)
        return 3
    if not mols:
        print("error: no molecules parsed", file=sys.stderr)
        return 3

    for m in mols:
        if not args.no_bonds:
            ensure_bonds(m, tolerance=args.bond_tolerance)

    idx = max(0, min(args.frame, len(mols) - 1))
    mol = mols[idx]

    if args.info:
        _print_info(mol)
        if len(mols) > 1:
            print(f"models:  {len(mols)}")
        return 0

    style = build_style(args)
    w, h = _parse_size(args.size)

    # -- still render to PNG ---------------------------------------------
    if args.render:
        if not w:
            w, h = 900, 700
        scene = Scene(mol, w, h, style=style, supersample=max(1, args.supersample),
                     backend=args.backend)
        scene.camera.orbit(args.rotate[0], args.rotate[1])
        scene.to_png(args.render)
        print(f"wrote {args.render} ({w}x{h})")
        return 0

    # -- single kitty frame to stdout ------------------------------------
    if args.kitty:
        if not w:
            cols, rows, xpx, ypx = kitty.terminal_size_px(1)
            cw, ch = kitty.cell_size_px(1)
            w = int(min(cols, 60) * cw)
            h = int(min(rows - 2, 30) * ch)
            w, h = max(w, 200), max(h, 200)
        scene = Scene(mol, w, h, style=style, supersample=max(1, args.supersample),
                     backend=args.backend)
        scene.camera.orbit(args.rotate[0], args.rotate[1])
        sys.stdout.write("\n")
        sys.stdout.flush()
        os.write(1, scene.to_kitty(move_cursor=True))
        sys.stdout.write("\n")
        return 0

    # -- interactive viewer ----------------------------------------------
    if not sys.stdout.isatty():
        print("error: interactive mode needs a terminal (use --render or --kitty)", file=sys.stderr)
        return 4
    if not kitty.supports_kitty():
        print("warning: this terminal may not support the Kitty graphics protocol.",
              file=sys.stderr)
        print("         Set MVIEWER_FORCE_KITTY=1 to try anyway, or use --render out.png.",
              file=sys.stderr)
        return 5

    # interactive defaults to a terminal-matching transparent background
    if not args.opaque and args.background is None:
        style.transparent = True

    from .viewer import Viewer
    viewer = Viewer(mol, frames=mols, style=style, autospin=args.spin, backend=args.backend)
    viewer.frame_index = idx
    # apply initial rotation
    viewer.widget.scene.camera.orbit(args.rotate[0], args.rotate[1])
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
