# -*- coding: utf-8 -*-
# file: test/test_ewald.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Ewald summation against known Madelung constants.

The periodic Hamiltonian removes the divergent ``G = 0`` electron terms
(jellium), so its ion-ion energy must use the same convention.  These are the
standard references for that sum.
"""

import numpy as np
import pytest

from carcara.core.ewald import ewald_energy

#: Wigner's constant: one point charge per cubic cell in a neutralizing
#: background has energy ``-zeta q^2 / 2L``.
WIGNER = 2.8372974795
#: The rocksalt Madelung constant (energy per ion pair, ``-alpha/d``).
NACL = 1.7475645946


def rocksalt(d=1.0):
    """Eight alternating charges on a cube of edge ``2d``."""
    positions, charges = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                positions.append([i * d, j * d, k * d])
                charges.append((-1.0) ** (i + j + k))
    return positions, charges, np.eye(3) * 2 * d


class TestMadelungConstants:
    @pytest.mark.parametrize("length", [1.0, 2.0, 3.7])
    def test_single_charge_with_background(self, length):
        energy = ewald_energy([[0.0, 0.0, 0.0]], [1.0], np.eye(3) * length)
        assert energy == pytest.approx(-WIGNER / (2 * length), rel=1e-9)

    def test_rocksalt(self):
        positions, charges, cell = rocksalt()
        energy = ewald_energy(positions, charges, cell) / 4      # 4 ion pairs
        assert energy == pytest.approx(-NACL, rel=1e-9)

    def test_caesium_chloride(self):
        """Two interpenetrating simple cubic lattices, alpha = 1.762675."""
        cell = np.eye(3) * 2.0 / np.sqrt(3.0)          # nearest neighbour = 1
        energy = ewald_energy([[0, 0, 0], list(cell.sum(axis=0) / 2)],
                              [1.0, -1.0], cell)
        assert energy == pytest.approx(-1.762674773, rel=1e-8)


class TestInvariance:
    @pytest.mark.parametrize("eta", [0.4, 1.0, 2.5, 4.0])
    def test_independent_of_the_splitting_parameter(self, eta):
        positions, charges, cell = rocksalt()
        reference = ewald_energy(positions, charges, cell)
        assert ewald_energy(positions, charges, cell,
                            eta=eta) == pytest.approx(reference, rel=1e-8)

    def test_a_neutral_pair_approaches_its_isolated_energy(self):
        """With no net charge the finite-size terms vanish as the cell grows."""
        previous = None
        for length in (20.0, 40.0, 80.0):
            energy = ewald_energy([[0, 0, 0], [0, 0, 1.0]], [1.0, -1.0],
                                  np.eye(3) * length)
            error = abs(energy - (-1.0))          # isolated: -q^2/d = -1 Ha
            assert previous is None or error < previous
            previous = error
        assert previous < 1e-3

    def test_translation_invariance(self):
        positions, charges, cell = rocksalt()
        shifted = (np.asarray(positions) + np.array([0.3, -0.7, 1.1])).tolist()
        assert ewald_energy(shifted, charges, cell) == pytest.approx(
            ewald_energy(positions, charges, cell), rel=1e-9)


class TestContract:
    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValueError, match="positions"):
            ewald_energy([[0, 0, 0]], [1.0, 1.0], np.eye(3))
        with pytest.raises(ValueError, match="zero volume"):
            ewald_energy([[0, 0, 0]], [1.0], np.zeros((3, 3)))

    def test_no_charges_is_zero(self):
        assert ewald_energy(np.zeros((0, 3)), [], np.eye(3)) == 0.0
