# -*- coding: utf-8 -*-
# file: test/algorithms/test_active_space_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The finite-difference gradient of a reduced active space.

Only internal directions are displaced: an isolated molecule's gradient has
no component along a rigid translation or rotation.  The end-to-end check
against energy differences lives with the PAW forces
(``test_paw_forces.test_reduced_active_space_force_tracks_optimized_energy``).
"""

import numpy as np
import pytest

from mandacaru.algorithms.active_space_forces import internal_directions


def rigid_motions(positions):
    """The six rigid-motion vectors (translations, rotations about the centroid)."""
    relative = positions - positions.mean(axis=0)
    out = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = 1.0
        unit = np.zeros(3)
        unit[axis] = 1.0
        out += [translation.ravel(), np.cross(unit, relative).ravel()]
    return np.array(out)


@pytest.mark.parametrize("positions, count", [
    ([[0, 0, 0], [0, 0, 1.2]], 1),                                  # diatomic
    ([[0, 0, 0], [0, 0, 1.1], [0, 0, 2.3]], 4),                     # linear
    ([[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]], 3),            # bent
    ([[0, 0, 0], [0.94, 0, -0.38], [-0.47, 0.81, -0.38],
      [-0.47, -0.81, -0.38]], 6),
    ([[1.0, 2.0, 3.0]], 0),                                         # one atom
])
def test_the_internal_directions_count_3n_minus_rigid_motions(positions,
                                                               count):
    positions = np.asarray(positions, dtype=float)
    directions = internal_directions(positions)
    assert directions.shape == (positions.size, count)
    assert np.allclose(directions.T @ directions, np.eye(count))
    assert np.allclose(rigid_motions(positions) @ directions, 0.0,
                       atol=1e-12)


def test_a_gradient_free_of_rigid_motions_is_recovered_exactly():
    # What the finite differences assemble: the directional derivatives
    # along the internal directions, projected back.
    rng = np.random.default_rng(5)
    positions = rng.normal(size=(4, 3))
    directions = internal_directions(positions)
    gradient = directions @ rng.normal(size=directions.shape[1])
    rebuilt = directions @ (directions.T @ gradient)
    assert np.allclose(rebuilt, gradient)
