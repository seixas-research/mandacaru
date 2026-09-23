# -*- coding: utf-8 -*-
# file: test_onecenter.py

"""The PAW-LCAO one-center two-body machinery (:mod:`mandacaru.pseudopotentials.onecenter`).

The module is validated machinery that is deliberately **not** wired into the
Hamiltonian: it measured the term the linearized one-center treatment drops
(0.08-0.33 eV on oxygen) and thereby ruled it out as the cause of the p-valence
binding failure.  These tests pin the pieces against closed forms so the
measurement stays reproducible.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mandacaru.pseudopotentials.onecenter import (angular_coupling,
                                                  one_center_coulomb,
                                                  radial_coulomb,
                                                  reference_density_matrix)


def orbital(l, m):
    return SimpleNamespace(l=l, m=m)


class TestRadialCoulomb:
    def test_uniform_sphere_monopole(self):
        """int int r_>^-1 r^2 r'^2 over a unit-density sphere is 2 R^5 / 15."""
        R = 2.0
        r = np.linspace(0.0, R, 4001)
        rho = np.ones_like(r)
        assert radial_coulomb(r, rho, rho, 0) == pytest.approx(2 * R ** 5 / 15,
                                                               rel=1e-7)

    @pytest.mark.parametrize("L", [0, 1, 2])
    def test_symmetric_in_the_two_densities(self, L):
        r = np.linspace(0.0, 6.0, 6001)
        a, b = np.exp(-r ** 2), r * np.exp(-0.5 * r)
        assert radial_coulomb(r, a, b, L) == pytest.approx(
            radial_coulomb(r, b, a, L), rel=1e-6)


class TestAngularCoupling:
    def test_s_densities_carry_only_the_monopole(self):
        s = orbital(0, 0)
        assert angular_coupling(0, s, s, s, s) == pytest.approx(1.0)
        assert angular_coupling(1, s, s, s, s) == 0
        assert angular_coupling(2, s, s, s, s) == 0

    def test_sp_transition_density_is_a_dipole(self):
        """4 pi / (2L+1) sum_M G G for the s-p pair density: 1/3 at L = 1."""
        s, p = orbital(0, 0), orbital(1, 0)
        assert angular_coupling(1, s, p, s, p) == pytest.approx(1.0 / 3.0)
        assert angular_coupling(0, s, p, s, p) == 0          # no monopole


class TestOnAPAWDataset:
    @pytest.fixture(scope="class")
    def oxygen(self):
        from mandacaru.pseudopotentials import get_paw
        from mandacaru.pseudopotentials.paw import paw_projectors
        try:
            dataset = get_paw("O")
        except FileNotFoundError:
            pytest.skip("the PAW-LCAO library is not linked")
        projectors = paw_projectors(["O"], np.zeros((1, 3)), {"O": dataset})
        return dataset, projectors

    def test_reference_density_matrix_holds_the_valence_electrons(self, oxygen):
        dataset, projectors = oxygen
        D0 = reference_density_matrix(dataset, projectors)
        assert np.allclose(D0, np.diag(np.diag(D0)))          # diagonal
        assert np.trace(D0).real == pytest.approx(dataset.valence_charge)

    def test_one_center_correction_is_symmetric_and_finite(self, oxygen):
        dataset, projectors = oxygen
        dW = one_center_coulomb(dataset, projectors)
        n = len(projectors)
        assert dW.shape == (n, n, n, n) and np.all(np.isfinite(dW))
        # (ab|cd) = (cd|ab): exchanging the two electrons.  The inner integral
        # is a cumulative trapezoid and the outer one Simpson, so the symmetry
        # holds to the quadrature error (8e-7 Ha here), not to round-off.
        assert np.allclose(dW, dW.transpose(2, 3, 0, 1), atol=1e-5)
        assert np.abs(dW).max() > 1e-3            # a real correction, not zero
