# -*- coding: utf-8 -*-
# file: test/test_ewald.py

# This code is part of Mandacaru.
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

from mandacaru.core.ewald import (ewald_energy, ewald_potential,
                                  madelung_constant)

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
        cell = np.eye(3) * 2.0 / np.sqrt(3.0)          # nearest neighbor = 1
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


class TestTheEwaldPotential:
    """``ewald_potential`` is the potential ``ewald_energy`` is the energy of.

    The zero-mean convention it uses is the same one a reciprocal-space Coulomb
    kernel makes when it drops ``G = 0``, which is what lets the two be mixed in
    one Hamiltonian.  These checks pin that they really do agree.
    """

    CELL = np.eye(3) * 8.0

    def test_it_approaches_the_madelung_constant_at_a_charge(self):
        """``v(r) - 1/r -> v_M``, and quadratically, since erfc is even."""
        target = madelung_constant(self.CELL)
        errors = []
        for radius in (0.10, 0.05):
            value = ewald_potential([[0.0, 0.0, 0.0]], [1.0], self.CELL,
                                    [[radius, 0.0, 0.0]])[0]
            errors.append(abs(value - 1.0 / radius - target))
        assert errors[0] < 1e-4
        assert errors[1] == pytest.approx(errors[0] / 4.0, rel=0.2)

    def test_the_madelung_constant_matches_the_cubic_value(self):
        """``v_M = -2.837297 / L`` for a simple cubic lattice."""
        for length in (6.0, 8.0, 11.0):
            assert madelung_constant(np.eye(3) * length) == pytest.approx(
                -2.8372974794 / length, rel=1e-6)

    def test_it_reproduces_the_energy_it_is_the_potential_of(self):
        """``E = 1/2 sum_A q_A (v_others(tau_A) + q_A v_M)``.

        The self-term is the Madelung constant, which is exactly the piece a
        ``G = 0``-dropped kernel leaves out.
        """
        positions = np.array([[0.0, 0.0, 0.0], [3.1, 2.3, 1.7]])
        charges = np.array([1.0, -1.0])
        constant = madelung_constant(self.CELL)
        total = 0.0
        for index, (site, charge) in enumerate(zip(positions, charges)):
            others = [j for j in range(charges.size) if j != index]
            elsewhere = ewald_potential(positions[others], charges[others],
                                        self.CELL, [site])[0]
            total += 0.5 * charge * (elsewhere + charge * constant)
        assert total == pytest.approx(
            ewald_energy(positions, charges, self.CELL), abs=1e-9)

    def test_its_mean_over_the_cell_is_zero(self):
        """The background term's whole job; the residue is quadrature on 1/r."""
        nodes, length = 60, 8.0
        axis = (np.arange(nodes) + 0.5) * (length / nodes)
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"),
                          axis=-1).reshape(-1, 3)
        mean = ewald_potential([[0.4, 0.3, 0.2]], [1.0], self.CELL, points).mean()
        assert mean == pytest.approx(0.0, abs=5e-5)

    def test_it_does_not_depend_on_the_splitting_parameter(self):
        """``eta`` divides the work between the two sums, not the answer."""
        site, charge = [[1.0, 2.0, 3.0]], [2.0]
        probe = [[5.0, 1.0, 4.0]]
        values = [ewald_potential(site, charge, self.CELL, probe, eta=eta)[0]
                  for eta in (0.35, 0.5, 0.7)]
        assert values[1] == pytest.approx(values[0], abs=1e-8)
        assert values[2] == pytest.approx(values[0], abs=1e-8)

    def test_a_mismatched_charge_count_is_refused(self):
        with pytest.raises(ValueError, match="positions for"):
            ewald_potential([[0.0, 0.0, 0.0]], [1.0, -1.0], self.CELL,
                            [[1.0, 0.0, 0.0]])
