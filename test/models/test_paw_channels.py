# -*- coding: utf-8 -*-
"""Tests for PAW projector metadata layout and block matrix assembly."""

import numpy as np
import pytest

from mandacaru.models.paw_channels import assemble_block_matrix, projector_blocks


class DummyProjector:
    def __init__(self, atom_index: int, channel: tuple[int, int], index: int = 0, real: bool = True):
        self.atom_index = atom_index
        self.channel = channel
        self.l, self.m = channel
        self.index = index
        self.real = real


def test_projector_blocks_ordering_and_grouping():
    p0 = DummyProjector(0, (0, 0), index=1)
    p1 = DummyProjector(0, (0, 0), index=0)
    p2 = DummyProjector(1, (1, -1), index=0)
    p3 = DummyProjector(1, (1, -1), index=1)

    blocks = projector_blocks([p0, p1, p2, p3])
    assert (0, 0, 0) in blocks
    assert blocks[(0, 0, 0)] == [1, 0]  # p1 (idx 0) then p0 (idx 1)
    assert (1, 1, -1) in blocks
    assert blocks[(1, 1, -1)] == [2, 3]


def test_projector_blocks_duplicate_radial_index_raises():
    p0 = DummyProjector(0, (0, 0), index=0)
    p1 = DummyProjector(0, (0, 0), index=0)
    with pytest.raises(ValueError, match="carry duplicate radial indices"):
        projector_blocks([p0, p1])


def test_assemble_block_matrix_diagonal():
    p0 = DummyProjector(0, (0, 0), index=0)
    p1 = DummyProjector(0, (1, 0), index=0)
    mat = assemble_block_matrix([p0, p1], blocks=None, diagonal=[2.0, 3.5])
    np.testing.assert_allclose(mat, np.diag([2.0, 3.5]))


def test_assemble_block_matrix_with_blocks():
    p0 = DummyProjector(0, (0, 0), index=0)
    p1 = DummyProjector(0, (0, 0), index=1)
    p2 = DummyProjector(1, (1, 0), index=0)

    b0 = np.array([[1.0, 0.5], [0.5, 2.0]])
    b1 = np.array([[3.0]])

    mat = assemble_block_matrix(
        [p0, p1, p2],
        blocks={(0, 0, 0): b0, (1, 1, 0): b1},
    )
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(mat[:2, :2], b0)
    assert mat[2, 2] == 3.0
    assert mat[0, 2] == 0.0


def test_assemble_block_matrix_mismatched_channels():
    p0 = DummyProjector(0, (0, 0), index=0)
    with pytest.raises(ValueError, match="missing"):
        assemble_block_matrix([p0], blocks={})

    with pytest.raises(ValueError, match="unmatched"):
        assemble_block_matrix([p0], blocks={(0, 0, 0): np.eye(1), (1, 0, 0): np.eye(1)})
