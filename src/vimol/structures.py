"""The multi-structure container: ``StructureSet`` holds N molecules with
stable identity, per-structure transform/tint/visibility/mark, and an active
index. See ``docs/design/multi-structure.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .molecule import Molecule


@dataclass
class Transform:
    """A rigid body transform in world space: x' = x @ rotation.T + translation.

    Kabsch produces exactly this pair, so nothing has to be repacked. 3x3 + (3,)
    rather than a 4x4 because Camera already works this way and vimol has no
    4x4 anywhere outside the GL projection matrix.
    """
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions) @ self.rotation.T + self.translation

    def apply_directions(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors) @ self.rotation.T

    def compose(self, other: "Transform") -> "Transform":
        """self ∘ other: apply *other* first, then self."""
        return Transform(
            rotation=self.rotation @ other.rotation,
            translation=self.rotation @ other.translation + self.translation,
        )

    def inverse(self) -> "Transform":
        rot_inv = self.rotation.T
        return Transform(rotation=rot_inv, translation=-(rot_inv @ self.translation))

    @property
    def is_identity(self) -> bool:
        return bool(np.array_equal(self.rotation, np.eye(3))
                    and np.array_equal(self.translation, np.zeros(3)))

    def key(self) -> bytes:
        return self.rotation.tobytes() + self.translation.tobytes()


@dataclass
class AlignmentResult:
    """What an align call produced, kept for reporting and reproducibility."""
    rmsd: float
    n_fitted: int                       # atoms that entered the fit
    transform: Transform
    ref_label: str
    method: str                         # "index" | "subset" | "permute"
    select: Optional[np.ndarray] = None      # mobile indices used for the fit
    ref_select: Optional[np.ndarray] = None  # reference indices, if different
    mapping: Optional[np.ndarray] = None     # mobile idx -> reference idx (permute only)
    stale: bool = False                      # source geometry edited since the fit


@dataclass
class Structure:
    """One loaded molecule plus the per-set state that belongs to it."""
    molecule: Molecule
    label: str                          # unique within the set; shown in the UI
    path: Optional[str] = None          # source file, None for in-memory
    transform: Transform = field(default_factory=Transform)
    visible: bool = True                # 'h' hide; excluded from the composite
    marked: bool = False                # 'space' mark; the overlay/align multi-select
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # assigned at append time
    alignment: Optional[AlignmentResult] = None
    revision: int = 0                   # bumped by touch(); drives composite caching
    undo_stack: List = field(default_factory=list)
    saved_sig: Optional[tuple] = None   # per-structure [MODIFIED] tracking

    def touch(self) -> None:
        """Call after ANY in-place mutation of ``self.molecule``."""
        self.revision += 1


# High-contrast, colour-blind-safe, deliberately not near any common CPK
# colour (which is reserved for the active structure, see composite()).
# Assigned once at append() time from TINTS[entry_index % len(TINTS)] --
# never recomputed -- so a structure keeps its colour when others are
# hidden or reordered.
TINTS = [
    (0.20, 0.85, 0.45),   # green
    (1.00, 0.55, 0.10),   # orange
    (0.65, 0.45, 0.95),   # purple
    (0.20, 0.75, 0.95),   # cyan
    (0.95, 0.35, 0.55),   # pink
    (0.85, 0.80, 0.25),   # gold
    (0.45, 0.60, 1.00),   # periwinkle
    (0.55, 0.85, 0.75),   # sea
]


class StructureSet:
    """N molecules with stable identity, per-structure state, and one active
    index. Everything downstream flattens it into a single throwaway
    ``Molecule`` (see :meth:`composite`) for rendering."""

    def __init__(self) -> None:
        self.entries: List[Structure] = []
        self.active_index: int = 0
        self.overlay: bool = False   # False: draw active only. True: draw the marked set.
        self._solo_restore: Optional[List[bool]] = None

    # -- sequence protocol --------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def _index_of_label(self, label: str) -> int:
        for i, e in enumerate(self.entries):
            if e.label == label:
                return i
        raise KeyError(label)

    def __getitem__(self, key) -> Structure:
        if isinstance(key, str):
            return self.entries[self._index_of_label(key)]
        return self.entries[key]

    @property
    def molecules(self) -> List[Molecule]:
        return [e.molecule for e in self.entries]

    @property
    def labels(self) -> List[str]:
        return [e.label for e in self.entries]

    # -- active index ---------------------------------------------------
    @property
    def active(self) -> Structure:
        return self.entries[self.active_index]

    def set_active(self, key) -> None:
        self.active_index = self._index_of_label(key) if isinstance(key, str) else int(key)

    def cycle_active(self, step: int = 1) -> None:
        if not self.entries:
            return
        self.active_index = (self.active_index + step) % len(self.entries)

    # -- membership -------------------------------------------------------
    def append(self, molecule: Molecule, label: Optional[str] = None,
               path: Optional[str] = None) -> Structure:
        label = label if label is not None else f"structure{len(self.entries)}"
        tint = TINTS[len(self.entries) % len(TINTS)]
        entry = Structure(molecule=molecule, label=label, path=path, tint=tint)
        self.entries.append(entry)
        return entry

    def extend(self, molecules, labels=None, path: Optional[str] = None) -> List[Structure]:
        labels = labels or [None] * len(molecules)
        return [self.append(m, label=lbl, path=path) for m, lbl in zip(molecules, labels)]

    def remove(self, key) -> None:
        idx = self._index_of_label(key) if isinstance(key, str) else int(key)
        del self.entries[idx]
        if self.active_index >= len(self.entries):
            self.active_index = max(0, len(self.entries) - 1)

    # -- marks / visibility ------------------------------------------------
    @property
    def marked(self) -> List[Structure]:
        return [e for e in self.entries if e.marked]

    def clear_marks(self) -> None:
        for e in self.entries:
            e.marked = False

    def toggle_mark(self, i: int) -> None:
        self.entries[i].marked = not self.entries[i].marked

    def toggle_visible(self, i: int) -> None:
        self.entries[i].visible = not self.entries[i].visible

    def solo(self, i: int) -> None:
        """Toggle solo: show only entry *i*, or restore prior visibility on a
        second call (see design §4.3)."""
        if self._solo_restore is not None:
            for e, v in zip(self.entries, self._solo_restore):
                e.visible = v
            self._solo_restore = None
            return
        self._solo_restore = [e.visible for e in self.entries]
        for k, e in enumerate(self.entries):
            e.visible = (k == i)

    def unsolo(self) -> None:
        if self._solo_restore is not None:
            for e, v in zip(self.entries, self._solo_restore):
                e.visible = v
            self._solo_restore = None

    # -- what the composite draws -------------------------------------------
    def drawn_indices(self) -> List[int]:
        """Entry indices in the composite, active first. Active-first matters:
        the first drawn entry is the one that keeps CPK colours (design §4)."""
        if not self.overlay:
            return [self.active_index] if self.active.visible else []
        marked = [i for i, e in enumerate(self.entries) if e.marked and e.visible]
        rest = marked if marked else [i for i, e in enumerate(self.entries) if e.visible]
        rest = [i for i in rest if i != self.active_index]
        return ([self.active_index] if self.active.visible else []) + rest
