"""Standard 18-column periodic-table layout, derived from mviewer.elements.

Grid coordinates here drive the interactive element picker in the viewer:
cursor navigation, click hit-testing, and rendering all key off :data:`GRID`.
The layout follows the common convention of pulling the lanthanides and
actinides out of the main table into two extra rows below, leaving a
non-selectable placeholder cell in the gap each leaves behind.

Element placement is computed from atomic-number ranges against
:data:`mviewer.elements.SYMBOLS` rather than typed out by hand, so it can't
drift out of sync with that table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import elements

N_COLS = 18


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    symbol: Optional[str]           # None for a non-selectable placeholder
    z: Optional[int] = None
    label: Optional[str] = None     # display text; defaults to the symbol
    note: Optional[str] = None      # shown in the info line for a placeholder
    jump_row: Optional[int] = None  # row Enter/click jumps to, for a placeholder

    @property
    def text(self) -> str:
        if self.label is not None:
            return self.label
        return self.symbol or ""


def _add_row(rows: List[List[Optional[Cell]]], row_idx: int, placements) -> None:
    """placements: (col, z) for an element, or (col, None, label, note, jump_row)
    for a non-selectable placeholder."""
    cells: List[Optional[Cell]] = [None] * N_COLS
    for item in placements:
        if len(item) == 2:
            col, z = item
            cells[col] = Cell(row_idx, col, elements.SYMBOLS[z], z)
        else:
            col, _, label, note, jump_row = item
            cells[col] = Cell(row_idx, col, None, None, label, note, jump_row)
    rows.append(cells)


def _build() -> List[List[Optional[Cell]]]:
    rows: List[List[Optional[Cell]]] = []
    # Period 1: H(1) .. He(2)
    _add_row(rows, 0, [(0, 1), (17, 2)])
    # Period 2: Li(3) Be(4) ... B(5)..Ne(10)
    _add_row(rows, 1, [(0, 3), (1, 4)] + [(12 + i, 5 + i) for i in range(6)])
    # Period 3: Na(11) Mg(12) ... Al(13)..Ar(18)
    _add_row(rows, 2, [(0, 11), (1, 12)] + [(12 + i, 13 + i) for i in range(6)])
    # Period 4: K(19)..Kr(36), 18 consecutive
    _add_row(rows, 3, [(c, 19 + c) for c in range(18)])
    # Period 5: Rb(37)..Xe(54), 18 consecutive
    _add_row(rows, 4, [(c, 37 + c) for c in range(18)])
    # Period 6: Cs(55) Ba(56) [gap: La-Lu below] Hf(72)..Rn(86)
    _add_row(rows, 5, [(0, 55), (1, 56), (2, None, "**", "Lanthanides — see the row below", 7)]
             + [(3 + c, 72 + c) for c in range(15)])
    # Period 7: Fr(87) Ra(88) [gap: Ac-Lr below] Rf(104)..Og(118)
    _add_row(rows, 6, [(0, 87), (1, 88), (2, None, "**", "Actinides — see the row below", 8)]
             + [(3 + c, 104 + c) for c in range(15)])
    # Lanthanides: La(57)..Lu(71), indented under the gap above
    _add_row(rows, 7, [(3 + c, 57 + c) for c in range(15)])
    # Actinides: Ac(89)..Lr(103)
    _add_row(rows, 8, [(3 + c, 89 + c) for c in range(15)])
    return rows


GRID: List[List[Optional[Cell]]] = _build()

POSITIONS: Dict[str, Tuple[int, int]] = {
    cell.symbol: (r, c)
    for r, row in enumerate(GRID)
    for c, cell in enumerate(row)
    if cell is not None and cell.symbol is not None
}


def cell_at(row: int, col: int) -> Optional[Cell]:
    if 0 <= row < len(GRID) and 0 <= col < N_COLS:
        return GRID[row][col]
    return None


def position_of(symbol: str) -> Tuple[int, int]:
    """Grid (row, col) for *symbol*, defaulting to Carbon if not placed."""
    return POSITIONS.get(elements.normalize_symbol(symbol), POSITIONS["C"])
