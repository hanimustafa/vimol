"""The multi-structure container: ``StructureSet`` holds N molecules with
stable identity, per-structure transform/tint/visibility/mark, and an active
index. See ``docs/design/multi-structure.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


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
