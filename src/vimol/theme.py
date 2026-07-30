"""Chrome color palettes for the interactive viewer, dark and light.

Every field here replaces a module-level color constant that used to live in
viewer.py -- see docs/design/theme-and-aesthetics.md sec 1 for the mapping.
DARK's values are exactly those old constants (a no-op refactor); LIGHT is
new content, tuned for a white/light terminal background.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    panel_bg: RGB
    panel_fg: RGB
    edit_prefix_fg: RGB
    warn_bg: RGB
    warn_fg: RGB
    input_bg: RGB
    input_fg: RGB
    help_bg: RGB
    help_fg: RGB
    list_header_fg: RGB
    list_muted_fg: RGB
    list_label_fg: RGB
    list_dim_fg: RGB
    list_active_bg: RGB
    list_cursor_bg: RGB
    list_rule_fg: RGB
    list_cap_bg: RGB
    cleanup_hint_fg: RGB
    pt_bg: RGB
    pt_border_fg: RGB
    pt_text_fg: RGB
    pt_dim_fg: RGB
    pt_gap_bg: RGB


DARK = Theme(
    name="dark", panel_bg=(30, 33, 44), panel_fg=(230, 232, 240),
    edit_prefix_fg=(150, 155, 170), warn_bg=(60, 30, 30), warn_fg=(250, 230, 230),
    input_bg=(44, 40, 30), input_fg=(240, 236, 220),
    help_bg=(20, 22, 30), help_fg=(220, 220, 230),
    list_header_fg=(139, 146, 165), list_muted_fg=(110, 118, 135),
    list_label_fg=(232, 236, 244), list_dim_fg=(200, 206, 216),
    list_active_bg=(37, 45, 64), list_cursor_bg=(28, 33, 46),
    list_rule_fg=(60, 66, 84), list_cap_bg=(42, 49, 66),
    cleanup_hint_fg=(255, 170, 60),
    pt_bg=(18, 20, 26), pt_border_fg=(60, 200, 180),
    pt_text_fg=(220, 220, 230), pt_dim_fg=(110, 114, 126), pt_gap_bg=(40, 42, 50),
)

LIGHT = Theme(
    name="light", panel_bg=(225, 227, 232), panel_fg=(30, 32, 38),
    edit_prefix_fg=(90, 95, 110), warn_bg=(255, 225, 225), warn_fg=(120, 20, 20),
    input_bg=(255, 247, 214), input_fg=(90, 70, 10),
    help_bg=(238, 239, 243), help_fg=(35, 37, 44),
    list_header_fg=(70, 76, 95), list_muted_fg=(120, 126, 142),
    list_label_fg=(20, 22, 28), list_dim_fg=(55, 60, 72),
    list_active_bg=(202, 210, 230), list_cursor_bg=(216, 220, 230),
    list_rule_fg=(190, 195, 206), list_cap_bg=(206, 211, 222),
    cleanup_hint_fg=(170, 90, 0),
    pt_bg=(238, 240, 244), pt_border_fg=(0, 140, 125),
    pt_text_fg=(30, 32, 38), pt_dim_fg=(120, 125, 138), pt_gap_bg=(220, 223, 230),
)


def luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def from_colorfgbg(value: str) -> Optional[Theme]:
    """COLORFGBG is "fg;bg" (basic-16-color indices). 7 or 15 (white/
    bright-white) in the background field means a light terminal; any other
    parseable value means dark. Unparseable or empty -> None (skip to the
    next fallback)."""
    if not value:
        return None
    parts = value.split(";")
    if not parts:
        return None
    try:
        bg_code = int(parts[-1])
    except ValueError:
        return None
    return LIGHT if bg_code in (7, 15) else DARK


def resolve(explicit: Optional[str], osc11_rgb: Optional[Tuple[int, int, int]],
            colorfgbg: Optional[str]) -> Theme:
    """explicit ("dark"/"light") -> OSC 11 background color -> COLORFGBG -> DARK."""
    if explicit == "light":
        return LIGHT
    if explicit == "dark":
        return DARK
    if osc11_rgb is not None:
        return LIGHT if luminance(osc11_rgb) > 140 else DARK
    guess = from_colorfgbg(colorfgbg) if colorfgbg else None
    if guess is not None:
        return guess
    return DARK
