#!/usr/bin/env python3
"""Embedding demo: use mviewer as a library inside your own terminal app.

Two patterns are shown:

1. One-shot: render a molecule to a frame and paint it at the cursor.
2. Widget: run the full interactive Viewer from inside your app.

Run in a Kitty-capable terminal:

    python3 examples/embed_demo.py            # static frame
    python3 examples/embed_demo.py --live     # interactive widget
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import mviewer
from mviewer.render import Style

EX = os.path.dirname(os.path.abspath(__file__))


def static_frame():
    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    mviewer.ensure_bonds(mol)

    # A Scene is the embeddable rendering object. You own the pixel size.
    scene = mviewer.Scene(mol, 480, 360, style=Style(representation="ball_and_stick"),
                          supersample=2)
    scene.camera.orbit(30, -20)

    # Your app's own chrome around the image:
    print("\x1b[1m┌─ my terminal app ────────────────────────────┐\x1b[0m")
    print(f"  molecule: {mol.name}  ({mol.formula()}, {mol.n_atoms} atoms)")
    print()
    sys.stdout.flush()

    # Paint the molecule at the cursor. move_cursor=True lets normal text flow after it.
    os.write(1, scene.to_kitty(move_cursor=True))
    print("\n\x1b[2m(rendered with mviewer — a numpy software raycaster)\x1b[0m")


def live_widget():
    mol = mviewer.load(os.path.join(EX, "c60.xyz"))
    mviewer.view(mol, autospin=True)  # blocks until the user quits


if __name__ == "__main__":
    if "--live" in sys.argv:
        live_widget()
    else:
        static_frame()
