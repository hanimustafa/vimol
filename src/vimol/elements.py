"""Periodic-table reference data: symbols, covalent/van-der-Waals radii and CPK colors.

All radii are in angstrom. Colors are (r, g, b) floats in [0, 1].
The tables are deliberately self-contained (no external data files) so that the
library stays dependency-light and trivially embeddable.
"""
from __future__ import annotations

# Atomic number -> symbol (1..118). Index 0 is a dummy so Z indexes directly.
SYMBOLS = [
    "X",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

SYMBOL_TO_Z = {s.upper(): z for z, s in enumerate(SYMBOLS)}

# Full element names, aligned with SYMBOLS (index 0 is the same dummy).
NAMES = [
    "",
    "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen",
    "Oxygen", "Fluorine", "Neon", "Sodium", "Magnesium", "Aluminium", "Silicon",
    "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
    "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron",
    "Cobalt", "Nickel", "Copper", "Zinc", "Gallium", "Germanium", "Arsenic",
    "Selenium", "Bromine", "Krypton", "Rubidium", "Strontium", "Yttrium",
    "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium",
    "Palladium", "Silver", "Cadmium", "Indium", "Tin", "Antimony", "Tellurium",
    "Iodine", "Xenon", "Caesium", "Barium", "Lanthanum", "Cerium",
    "Praseodymium", "Neodymium", "Promethium", "Samarium", "Europium",
    "Gadolinium", "Terbium", "Dysprosium", "Holmium", "Erbium", "Thulium",
    "Ytterbium", "Lutetium", "Hafnium", "Tantalum", "Tungsten", "Rhenium",
    "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium", "Lead",
    "Bismuth", "Polonium", "Astatine", "Radon", "Francium", "Radium",
    "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium",
    "Plutonium", "Americium", "Curium", "Berkelium", "Californium",
    "Einsteinium", "Fermium", "Mendelevium", "Nobelium", "Lawrencium",
    "Rutherfordium", "Dubnium", "Seaborgium", "Bohrium", "Hassium",
    "Meitnerium", "Darmstadtium", "Roentgenium", "Copernicium", "Nihonium",
    "Flerovium", "Moscovium", "Livermorium", "Tennessine", "Oganesson",
]

# Covalent radii (Cordero 2008), angstrom. Fallback 0.77 (carbon-ish).
_COVALENT = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46,
}

# Van der Waals radii (Bondi + extensions), angstrom. Fallback 1.7.
_VDW = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39, "Ga": 1.87,
    "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02, "I": 1.98,
    "Xe": 2.16, "Fe": 2.05, "Pd": 1.63, "Ag": 1.72, "Cd": 1.58, "Sn": 2.17,
    "Pt": 1.75, "Au": 1.66, "Hg": 1.55, "Pb": 2.02,
}

# CPK / Jmol-inspired colors, (r, g, b) in [0, 1]. Fallback pink for unknowns.
_COLORS = {
    "H": (1.00, 1.00, 1.00), "He": (0.85, 1.00, 1.00), "Li": (0.80, 0.50, 1.00),
    "Be": (0.76, 1.00, 0.00), "B": (1.00, 0.71, 0.71), "C": (0.35, 0.35, 0.38),
    "N": (0.19, 0.31, 0.97), "O": (1.00, 0.05, 0.05), "F": (0.56, 0.88, 0.31),
    "Ne": (0.70, 0.89, 0.96), "Na": (0.67, 0.36, 0.95), "Mg": (0.54, 1.00, 0.00),
    "Al": (0.75, 0.65, 0.65), "Si": (0.94, 0.78, 0.63), "P": (1.00, 0.50, 0.00),
    "S": (1.00, 0.86, 0.19), "Cl": (0.12, 0.94, 0.12), "Ar": (0.50, 0.82, 0.89),
    "K": (0.56, 0.25, 0.83), "Ca": (0.24, 1.00, 0.00), "Fe": (0.88, 0.40, 0.20),
    "Cu": (0.78, 0.50, 0.20), "Zn": (0.49, 0.50, 0.69), "Br": (0.65, 0.16, 0.16),
    "I": (0.58, 0.00, 0.58), "Au": (1.00, 0.82, 0.14), "Ag": (0.75, 0.75, 0.75),
    "Pt": (0.82, 0.82, 0.88), "Hg": (0.72, 0.72, 0.82), "Ni": (0.31, 0.82, 0.31),
    "Mn": (0.61, 0.48, 0.78), "Se": (1.00, 0.63, 0.00), "Pb": (0.34, 0.35, 0.38),
}

_DEFAULT_COLOR = (1.00, 0.41, 0.71)  # hot pink for unknown elements


def normalize_symbol(sym: str) -> str:
    """Normalize an element token to canonical capitalization (e.g. 'FE' -> 'Fe')."""
    sym = sym.strip()
    if not sym:
        return "X"
    return sym[0].upper() + sym[1:].lower()


def symbol_to_z(sym: str) -> int:
    return SYMBOL_TO_Z.get(normalize_symbol(sym).upper(), 0)


def covalent_radius(sym: str) -> float:
    return _COVALENT.get(normalize_symbol(sym), 0.77)


def vdw_radius(sym: str) -> float:
    return _VDW.get(normalize_symbol(sym), 1.70)


def element_color(sym: str) -> tuple:
    return _COLORS.get(normalize_symbol(sym), _DEFAULT_COLOR)


def element_name(sym: str) -> str:
    """Full element name, e.g. 'Carbon' for 'C'. Falls back to the symbol itself."""
    z = symbol_to_z(sym)
    return NAMES[z] if z else normalize_symbol(sym)
