#!/usr/bin/env python3
"""Embedding demo: intercept the mouse in your own UI and drive the molecule.

The point of this file: when you embed the viewer, *your* app owns the terminal
and the input loop. You decode input yourself and forward only the events you
want into the molecule region — everything else stays yours. vimol exposes
exactly the two pieces you need for that:

    vimol.MoleculeWidget   # the interaction core (no terminal, no input loop)
    vimol.InputDecoder     # bytes -> MouseEvent / KeyEvent

Here the top two rows are the host app's own chrome; the molecule lives below
them in a sub-region. Mouse events are routed to the widget ONLY when the
pointer is inside that region (that's the "interception"); a click up in the
chrome is handled by the host instead.

Run in a Kitty-capable terminal:

    python3 examples/embed_demo.py            # this interactive host demo
    python3 examples/embed_demo.py --static   # just paint one frame inline
"""
import os
import select
import sys
import termios
import tty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import vimol  # noqa: E402
from vimol import kitty, input as minput  # noqa: E402

EX = os.path.dirname(os.path.abspath(__file__))
HEADER_ROWS = 2  # the host app's own top chrome


def static_frame():
    mol = vimol.load(os.path.join(EX, "c60.xyz"))
    vimol.ensure_bonds(mol)
    scene = vimol.Scene(mol, 480, 360, supersample=2)
    scene.camera.orbit(30, -20)
    print("\x1b[1m┌─ my terminal app ─────────────────────────────┐\x1b[0m")
    print(f"  {mol.name}  ({mol.formula()}, {mol.n_atoms} atoms)\n")
    sys.stdout.flush()
    os.write(1, scene.to_kitty(move_cursor=True))
    print("\n\x1b[2m(rendered with vimol)\x1b[0m")


def interactive_host():
    fd = 0
    mol = vimol.load(os.path.join(EX, "c60.xyz"))
    vimol.ensure_bonds(mol)

    cols, rows, xpx, ypx = kitty.terminal_size_px(1)
    cw, ch = kitty.cell_size_px(1)
    region_rows = max(rows - HEADER_ROWS - 1, 1)
    region_y0_px = HEADER_ROWS * ch                 # widget origin (pixels)

    widget = vimol.MoleculeWidget(mol, int(cols * cw), int(region_rows * ch),
                                    supersample=1)
    widget.set_cell_metrics(cw, ch)
    widget.scene.camera.orbit(25, -18)
    decoder = minput.InputDecoder(pixel=False)

    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    os.write(1, b"\x1b[?1049h\x1b[?25l\x1b[2J")
    os.write(1, minput.enable_mouse(pixel=True, hover=True))
    decoder.pixel = minput.supports_pixel_mouse(fd, 1)

    img_id = kitty.unique_id_base() + 5
    prev = None
    running = True
    host_msg = "click the [SPIN] button up here ↑ (host handles it); drag the model below"
    spin = False

    def in_region(ev) -> bool:
        py = ev.y if ev.pixel else ev.y * ch
        return py >= region_y0_px

    try:
        while running:
            # ---- draw host chrome ----
            out = bytearray(b"\x1b[H\x1b[2K")
            out += b"\x1b[1;1H\x1b[44;97m  MyApp   [SPIN]   [QUIT]  \x1b[0m"
            out += b"\x1b[2;1H\x1b[2K\x1b[90m  " + host_msg.encode() + b"\x1b[0m"
            # ---- draw the embedded molecule below the chrome ----
            out += b"\x1b[%d;1H" % (HEADER_ROWS + 1)
            img = widget.render()
            out += kitty.encode_image(img, image_id=img_id, placement_id=img_id,
                                      cols=cols, rows=region_rows, move_cursor=False)
            if prev is not None:
                out += kitty.delete_image(prev)
            prev = img_id
            img_id = img_id + 1 if img_id < kitty.unique_id_base() + 8 else kitty.unique_id_base() + 5
            # ---- status line ----
            info = widget.atom_info(widget.hovered) or "hover an atom to identify it"
            out += b"\x1b[%d;1H\x1b[2K\x1b[100m %s \x1b[0m" % (rows, info.encode())
            os.write(1, bytes(out))

            r, _, _ = select.select([fd], [], [], 1 / 60)
            if not r:
                if spin:
                    widget.scene.camera.orbit(1.4, 0)
                continue
            data = os.read(fd, 4096)
            for ev in decoder.feed(data):
                if isinstance(ev, minput.KeyEvent):
                    if ev.key in ("q", "escape"):
                        running = False
                    else:
                        widget.handle_key(ev.key)
                elif isinstance(ev, minput.MouseEvent):
                    # THE INTERCEPTION: only the molecule region gets the mouse
                    if in_region(ev):
                        widget.handle_mouse(ev, origin=(0, region_y0_px))
                    elif ev.action == "down":
                        # host chrome click: crude hit-test on row 1
                        col = ev.x / cw if ev.pixel else ev.x
                        if 9 <= col <= 16:      # [SPIN]
                            spin = not spin
                            host_msg = f"spin {'on' if spin else 'off'} (host state, not the widget's)"
                        elif 18 <= col <= 25:   # [QUIT]
                            running = False
    finally:
        os.write(1, minput.disable_mouse(pixel=True))
        os.write(1, kitty.clear_all_images() + b"\x1b[?25h\x1b[?1049l")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    if "--static" in sys.argv:
        static_frame()
    else:
        interactive_host()
