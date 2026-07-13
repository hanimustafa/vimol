"""mviewer — an embeddable terminal molecular viewer using the Kitty graphics protocol.

Quick start
-----------
    import mviewer

    mol = mviewer.load("caffeine.pdb")     # parse a structure
    scene = mviewer.Scene(mol, 640, 480)    # bind camera + renderer
    scene.camera.orbit(30, 15)              # rotate
    img = scene.render()                    # (H, W, 3) uint8 numpy array

    # display in a Kitty-capable terminal:
    import sys, os
    os.write(1, scene.to_kitty())

    # or run the full interactive viewer:
    mviewer.view(mol)

Everything here is pure Python + numpy, no GPU or windowing system required.
"""
from __future__ import annotations

from .molecule import Molecule
from .camera import Camera
from .render import Renderer, Style
from .scene import Scene
from .widget import MoleculeWidget
from .input import InputDecoder, MouseEvent, KeyEvent, enable_mouse, disable_mouse
from .parsers import load, load_all, loads, SUPPORTED_EXTENSIONS
from .bonds import perceive_bonds, ensure_bonds
from . import kitty
from . import elements

__version__ = "0.1.0"

__all__ = [
    "Molecule", "Camera", "Renderer", "Style", "Scene", "MoleculeWidget",
    "InputDecoder", "MouseEvent", "KeyEvent", "enable_mouse", "disable_mouse",
    "load", "load_all", "loads", "SUPPORTED_EXTENSIONS",
    "perceive_bonds", "ensure_bonds", "kitty", "elements",
    "view", "__version__",
]


def view(molecule_or_path, **kwargs):
    """Launch the interactive terminal viewer.

    Accepts a :class:`Molecule` or a path to a structure file. Extra keyword
    arguments are forwarded to :class:`mviewer.viewer.Viewer`.
    """
    from .viewer import Viewer

    if isinstance(molecule_or_path, str):
        mol = load(molecule_or_path)
    else:
        mol = molecule_or_path
    ensure_bonds(mol)
    v = Viewer(mol, **kwargs)
    v.widget.scene.camera.orbit(20, -15)  # a pleasant default 3/4 view
    v.run()
