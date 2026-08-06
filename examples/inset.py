"""Put a molecule in the middle of a terminal screen -- the README's snippet.

Everything below the imports is the whole of it: build a Scene at the pixel
size you want, park the cursor where you want its top-left corner, and write.
Run it in kitty, Ghostty or WezTerm."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import vimol

scene = vimol.Scene(vimol.load("c60.xyz"), 640, 380)   # render it off-screen
scene.style.transparent = True                         # let the terminal through
os.write(1, b"\x1b[6;19H" + scene.to_kitty())          # paint at row 6, col 19
