"""Command-line driver for vimol.

    vimol                           # opens the bundled C60 demo (checkout only)
    vimol file.pdb                 # interactive viewer (opens editable: a=append)
    vimol file.xyz --spin          # autospinning
    vimol a.xyz b.pdb              # load both, auto-overlaid for comparison
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple

from .parsers import load_all, SUPPORTED_EXTENSIONS
from .render import Style
from .structures import StructureSet
from . import kitty


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
        prog="vimol",
        description="A terminal molecular viewer using the Kitty graphics protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS),
    )
    p.add_argument("file", nargs="*", help="one or more structure files (xyz/pdb/mol/sdf)")
    p.add_argument("--style", default="ball_and_stick",
                   choices=["ball_and_stick", "spacefill", "licorice", "wireframe"])
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "gl"],
                   help="rendering backend: numpy CPU raycaster, GPU (OpenGL), "
                        "or auto (GPU if a context can be created, else CPU)")
    p.add_argument("--theme", default="auto", choices=["auto", "dark", "light"],
                   help="chrome color theme; auto detects the terminal's own background")
    p.add_argument("--rotate", nargs=2, type=float, metavar=("YAW", "PITCH"),
                   default=(20.0, -15.0), help="initial rotation in degrees")
    p.add_argument("--spin", action="store_true", help="autospin in interactive mode")
    p.add_argument("--frame", type=int, default=0, help="frame/model index for multi-model files")
    p.add_argument("--atom-scale", type=float, default=None)
    p.add_argument("--bond-radius", type=float, default=None)
    p.add_argument("--background", type=str, default=None, help="hex or r,g,b background color (implies --opaque)")
    p.add_argument("--transparent", action="store_true", help="transparent background (RGBA cutout)")
    p.add_argument("--opaque", action="store_true", help="solid background")
    p.add_argument("--no-depth-cue", action="store_true")
    p.add_argument("--no-bonds", action="store_true", help="do not auto-perceive bonds")
    p.add_argument("--bond-tolerance", type=float, default=0.45)
    p.add_argument("--list-formats", action="store_true")
    p.add_argument("--version", action="store_true")
    return p


def _apply_theme_arg(args) -> None:
    """--theme is the highest-precedence source in theme.resolve()'s ladder
    (see docs/design/theme-and-aesthetics.md sec 2) -- implemented by setting
    VIMOL_THEME before the Viewer is constructed, rather than adding a
    parallel parameter to Viewer.__init__ for what VIMOL_THEME already
    covers.

    "auto" is the DEFAULT, i.e. "the user passed no --theme at all", so it
    must leave VIMOL_THEME alone rather than clearing it -- clearing would
    mean the env var only ever worked for direct Viewer() construction and
    never for the CLI, silently deleting the ladder's second rung on every
    plain `vimol file.xyz`.
    """
    if args.theme != "auto":
        os.environ["VIMOL_THEME"] = args.theme


def _probe_terminal_raw() -> Optional["kitty.TerminalProbe"]:
    """Ask the terminal itself about Kitty graphics support, briefly raw.

    Used when the environment heuristics (kitty.supports_kitty) come up
    empty -- typically over SSH, where TERM is rewritten and the kitty/
    ghostty env vars are stripped even though the terminal at the other end
    renders graphics fine. The probe needs raw mode to read the reply;
    restore the tty exactly as found. Returns None when stdin isn't a tty
    (nothing to ask).
    """
    import termios
    import tty
    try:
        old = termios.tcgetattr(0)
    except (OSError, termios.error):
        return None
    try:
        tty.setraw(0)
        return kitty.probe_terminal(0, 1)
    finally:
        termios.tcsetattr(0, termios.TCSADRAIN, old)


def _check_kitty_terminal() -> Tuple[Optional["kitty.TerminalProbe"], int]:
    """Guard for the interactive path: stdout must be a real terminal, and it
    must speak the Kitty graphics protocol (checked via env heuristics, then
    a raw-mode probe if those come up empty). Returns ``(probe, 0)`` to
    proceed -- reusing the probe avoids a second round trip -- or
    ``(None, exit_code)`` to abort.
    """
    if not sys.stdout.isatty():
        print("error: interactive mode needs a terminal", file=sys.stderr)
        return None, 4
    if kitty.supports_kitty():
        return None, 0
    # The environment says nothing (common over SSH) -- ask the terminal
    # itself. Only an answered probe that lacks graphics support, or no
    # terminal to ask, refuses; a confirmed terminal proceeds normally.
    probe = _probe_terminal_raw()
    if probe is None or probe.graphics is not True:
        print("warning: this terminal does not appear to support the Kitty "
              "graphics protocol.", file=sys.stderr)
        print("         Set VIMOL_FORCE_KITTY=1 to try anyway.", file=sys.stderr)
        return None, 5
    return probe, 0


def _default_demo_path() -> Optional[str]:
    """Path to the bundled C60 demo, for `vimol` with no file argument.

    Only resolvable from a checkout or editable install -- `examples/` sits
    next to `src/`, not inside the installed package -- so a plain `vimol`
    still falls back to --help rather than crashing on a real install.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "examples", "c60.xyz")
    return path if os.path.exists(path) else None


def _build_structure_set(paths: List[str], no_bonds: bool, tolerance: float) -> StructureSet:
    """Load 2+ files into one StructureSet, in load order (VIM-1). A file
    that's missing, fails to parse, or parses to zero molecules is skipped
    with a warning instead of aborting the whole session. Sets ``overlay =
    True`` and marks each file's first model, so the caller gets a
    ready-to-render auto-overlaid set with no further setup -- see
    docs/superpowers/specs/2026-07-30-multi-file-cli-design.md.
    """
    sset = StructureSet()
    # Bonds are perceived per drawn frame by StructureSet.composite, not here:
    # only each file's first model is on screen initially, and bonding a whole
    # ensemble up front cost seconds for frames the user may never open.
    sset.auto_bonds = not no_bonds
    sset.bond_tolerance = tolerance
    basenames = [os.path.basename(p) for p in paths]
    dupe_counts = Counter(basenames)
    seen: Counter = Counter()
    for path, base in zip(paths, basenames):
        if not os.path.exists(path):
            print(f"warning: skipping {path}: no such file", file=sys.stderr)
            continue
        try:
            mols = load_all(path)
        except Exception as e:  # noqa: BLE001
            print(f"warning: skipping {path}: {e}", file=sys.stderr)
            continue
        if not mols:
            print(f"warning: skipping {path}: no molecules parsed", file=sys.stderr)
            continue

        if dupe_counts[base] > 1:
            seen[base] += 1
            stem = base if seen[base] == 1 else f"{base}~{seen[base]}"
        else:
            stem = base
        multi = len(mols) > 1
        for i, m in enumerate(mols):
            label = f"{stem}#{i + 1}" if multi else stem
            entry = sset.append(m, label=label, path=path)
            if i == 0:
                entry.marked = True
    sset.overlay = True
    return sset


def main(argv: List[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    _apply_theme_arg(args)

    if args.version:
        from . import __version__
        print(f"vimol {__version__}")
        return 0
    if args.list_formats:
        print("Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS))
        return 0
    files = args.file
    if not files:
        demo = _default_demo_path()
        if not demo:
            make_parser().print_help()
            return 1
        files = [demo]

    if len(files) == 1:
        path = files[0]
        if not os.path.exists(path):
            print(f"error: no such file: {path}", file=sys.stderr)
            return 2

        try:
            mols = load_all(path)
        except Exception as e:  # noqa: BLE001
            print(f"error: failed to parse {path}: {e}", file=sys.stderr)
            return 3
        if not mols:
            print("error: no molecules parsed", file=sys.stderr)
            return 3

        # No bonding pass here either -- the Viewer's StructureSet perceives
        # each frame as it is drawn (see StructureSet.composite).
        idx = max(0, min(args.frame, len(mols) - 1))
        mol = mols[idx]
        frames, structures, source_path = mols, None, path
    else:
        structures = _build_structure_set(files, args.no_bonds, args.bond_tolerance)
        if len(structures) == 0:
            print("error: no molecules parsed", file=sys.stderr)
            return 3

        idx = max(0, min(args.frame, len(structures) - 1))
        mol = structures[0].molecule
        frames, source_path = None, None

    style = build_style(args)

    # -- interactive viewer ----------------------------------------------
    probe, rc = _check_kitty_terminal()
    if rc:
        return rc

    # interactive defaults to a terminal-matching transparent background
    if not args.opaque and args.background is None:
        style.transparent = True

    from .viewer import Viewer
    viewer = Viewer(mol, frames=frames, structures=structures, style=style,
                    autospin=args.spin, backend=args.backend,
                    source_path=source_path, editable=True,
                    auto_bonds=not args.no_bonds,
                    bond_tolerance=args.bond_tolerance,
                    probe=probe)   # reuse the detection probe: no second round trip
    viewer.frame_index = idx
    # apply initial rotation
    viewer.widget.scene.camera.orbit(args.rotate[0], args.rotate[1])
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
