import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vimol
from vimol.molecule import Molecule
from vimol.structures import Transform


def test_transform_identity_apply_is_noop():
    t = Transform()
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = t.apply(pos)
    assert np.allclose(out, pos)


def test_transform_identity_is_identity_true():
    assert Transform().is_identity is True


def test_transform_rotation_translation_apply():
    # 90 degree rotation about z: (x,y,z) -> (-y,x,z)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = Transform(rotation=rot, translation=np.array([10.0, 0.0, 0.0]))
    pos = np.array([[1.0, 0.0, 0.0]])
    out = t.apply(pos)
    assert np.allclose(out, [[10.0, 1.0, 0.0]])
    assert t.is_identity is False


def test_transform_apply_directions_ignores_translation():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = Transform(rotation=rot, translation=np.array([100.0, 100.0, 100.0]))
    vec = np.array([[1.0, 0.0, 0.0]])
    out = t.apply_directions(vec)
    assert np.allclose(out, [[0.0, 1.0, 0.0]])


def test_transform_inverse_round_trips():
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = Transform(rotation=rot, translation=np.array([5.0, -2.0, 1.0]))
    pos = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    out = t.inverse().apply(t.apply(pos))
    assert np.allclose(out, pos)


def test_transform_compose_applies_other_first():
    t1 = Transform(translation=np.array([1.0, 0.0, 0.0]))
    t2 = Transform(translation=np.array([0.0, 1.0, 0.0]))
    composed = t1.compose(t2)
    pos = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(composed.apply(pos), t1.apply(t2.apply(pos)))


def test_transform_key_differs_for_different_transforms():
    a = Transform()
    b = Transform(translation=np.array([1.0, 0.0, 0.0]))
    assert a.key() != b.key()
    assert a.key() == Transform().key()
