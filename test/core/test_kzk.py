# -*- coding: utf-8 -*-
# file: test/core/test_kzk.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The size-dependent LDA: :mod:`mandacaru.core.kzk`.

Kwee, Zhang and Krakauer, Phys. Rev. Lett. **100**, 126404 (2008).  This is the
one correction in Mandacaru built on a **fitted** parametrization, so the tests
are mostly about whether the transcription is faithful rather than whether the
code runs.  Four of them check the implementation against the paper itself:

* ``a_0`` against its stated ``-0.916 Ry``;
* the density boundary ``gamma = r_s(N = 2)`` against the three exchange
  discontinuities visible in Fig. 1, at ``L = 7.5, 10.3, 18`` Bohr;
* the ``a_1/L^2`` cancellation between exchange and correlation, which the
  paper puts there on purpose so that ``e_xc`` scales as ``O(1/L^3)``;
* that scaling itself, which is the sharpest of the four because it is a
  consequence of the design rather than a number read off a plot.

The remaining structural check is that ``f(r_s)``, the intermediate-density
cubic, is continuous in **value and slope** at both ends -- the four conditions
that determine it, carrying no fitted parameters of its own.

Unit trap: the parametrization is Rydberg throughout while Mandacaru is
Hartree, and the repository's own Perdew-Zunger routine is the Hartree form.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.basis.atomic_solver import lda_correlation
from mandacaru.core import kzk
from mandacaru.core.kzk import (KZKCorrection, effective_length, fs_correlation,
                                fs_exchange, fs_exchange_correlation,
                                infinite_exchange_correlation, kzk_correction,
                                leading_term_cancellation)
from mandacaru.optimizers import Optimizer

SLSQP = Optimizer(method="SLSQP", maxiter=1000, tol=1e-12)


class TestItMatchesThePaper:
    def test_table_one_is_transcribed(self):
        """Table I, in Rydberg atomic units."""
        assert (kzk.A1, kzk.A2, kzk.A3) == (-2.2037, 0.4710, -0.0150)
        assert (kzk.G1, kzk.G2, kzk.G3, kzk.G4) == (0.1182, 1.1656,
                                                    -5.2884, -1.1233)

    def test_a0_is_the_slater_constant(self):
        """The paper: "the term with ``a_0 ~= -0.916 Ry``"."""
        assert kzk.A0 == pytest.approx(-0.916, abs=5e-4)
        # Computed, not rounded: -3/4 (9/4 pi^2)^(1/3) Ha, doubled to Rydberg.
        exact = 2.0 * (-0.75 * (9.0 / (4.0 * np.pi ** 2)) ** (1 / 3))
        assert kzk.A0 == pytest.approx(exact, rel=1e-14)

    @pytest.mark.parametrize("length, seen", [(7.5, 3.7), (10.3, 5.0),
                                              (18.0, 8.9)])
    def test_gamma_matches_the_figure_one_jumps(self, length, seen):
        """``gamma = r_s(N = 2)`` is where Fig. 1's exchange curves jump.

        Three independent points read off the published figure, which is the
        only way to check this boundary without the authors' own code.
        """
        assert kzk._rs_at(2, length) == pytest.approx(seen, abs=0.15)

    def test_the_boundaries_are_ordered(self):
        """``gamma_h < gamma < gamma_l``: more electrons means smaller r_s."""
        L = 12.0
        assert kzk._rs_at(12, L) < kzk._rs_at(2, L) < kzk._rs_at(0.5, L)

    def test_the_infinite_exchange_is_the_lda_one(self):
        """``a_0 / r_s`` must be Slater exchange at that density."""
        rs = np.array([1.0, 2.0, 5.0])
        density = 3.0 / (4.0 * np.pi * rs ** 3)
        slater = -0.75 * (3.0 / np.pi) ** (1 / 3) * density ** (1 / 3)
        assert (kzk.A0 / rs) * kzk.HARTREE_PER_RYDBERG == pytest.approx(
            slater, rel=1e-12)

    def test_the_correlation_helper_is_rydberg(self):
        """The repository's Perdew-Zunger routine is Hartree; this is double."""
        rs = np.array([0.5, 2.0, 8.0])
        density = 3.0 / (4.0 * np.pi * rs ** 3)
        hartree, _v = lda_correlation(density)
        rydberg, _d = kzk._pz_correlation_rydberg(rs)
        assert rydberg == pytest.approx(2.0 * hartree, rel=1e-12)

    def test_the_correlation_derivative_is_exact(self):
        """From ``v_c = e_c - (r_s/3) de_c/dr_s``, not a finite difference."""
        rs = np.array([0.7, 3.0])
        _value, slope = kzk._pz_correlation_rydberg(rs)
        step = 1e-6
        numeric = ((kzk._pz_correlation_rydberg(rs + step)[0]
                    - kzk._pz_correlation_rydberg(rs - step)[0]) / (2 * step))
        assert slope == pytest.approx(numeric, rel=1e-6)


class TestTheDesignedCancellation:
    r"""The ``a_1/L^2`` terms cancel between exchange and correlation.

    The paper adds ``+a_1 r_s/L^2`` to exchange and ``-a_1 r_s/L^2`` to
    correlation so the pair vanishes in ``e_xc``, leaving the physical
    ``O(1/L^3)``.  That makes the scaling a *consequence* of the
    transcription, and therefore the sharpest available test of it.
    """

    @pytest.mark.parametrize("length", [7.5, 18.0, 50.0])
    def test_the_terms_cancel_exactly(self, length):
        rs = np.linspace(0.2, 0.9 * kzk._rs_at(12, length), 40)
        assert leading_term_cancellation(rs, length) < 1e-10

    def test_the_residual_scales_as_one_over_l_cubed(self):
        """``e_xc^FS - e_xc^inf`` must fall by 125 when ``L`` grows by 5."""
        rs = np.array(2.0)
        infinite = float(infinite_exchange_correlation(rs))
        errors = [abs(float(fs_exchange_correlation(rs, L)) - infinite)
                  for L in (10.0, 50.0, 250.0)]
        assert errors[0] / errors[1] == pytest.approx(125.0, rel=0.02)
        assert errors[1] / errors[2] == pytest.approx(125.0, rel=0.02)

    def test_it_reaches_the_infinite_limit(self):
        rs = np.array([0.8, 2.0, 4.0])
        assert fs_exchange_correlation(rs, 1e6) == pytest.approx(
            infinite_exchange_correlation(rs), rel=1e-8)


class TestTheIntermediateCubic:
    """``f`` is fixed by four continuity conditions and has no free parameters."""

    @pytest.mark.parametrize("length", [10.0, 20.0])
    def test_the_value_is_continuous_at_both_ends(self, length):
        _a, _b, gamma_h, gamma_l = kzk._intermediate_cubic(length)
        step = 1e-6
        for boundary in (gamma_h, gamma_l):
            below = float(fs_correlation(np.array(boundary - step), length))
            above = float(fs_correlation(np.array(boundary + step), length))
            assert below == pytest.approx(above, abs=1e-7)

    @pytest.mark.parametrize("length", [10.0, 20.0])
    def test_the_slope_is_continuous_at_both_ends(self, length):
        _a, _b, gamma_h, gamma_l = kzk._intermediate_cubic(length)
        step = 1e-5

        def slope(at, side):
            # A one-sided difference taken *away* from the boundary; dividing
            # by `side * 2 * step` keeps the direction consistent, which the
            # first version of this helper got wrong for side = -1.
            near = np.array(at + side * step)
            far = np.array(at + side * 3 * step)
            return (float(fs_correlation(far, length))
                    - float(fs_correlation(near, length))) / (side * 2 * step)

        for boundary in (gamma_h, gamma_l):
            assert slope(boundary, -1) == pytest.approx(slope(boundary, +1),
                                                        abs=1e-6)

    def test_correlation_vanishes_beyond_gamma_l(self):
        length = 10.0
        gamma_l = kzk._rs_at(0.5, length)
        assert fs_correlation(np.array(gamma_l * 1.5), length) == 0.0

    def test_exchange_decays_beyond_gamma(self):
        """``a_3 L^5 / r_s^6`` -- the self-interaction of a fractional electron."""
        length = 10.0
        gamma = kzk._rs_at(2, length)
        far = np.array(gamma * 3.0)
        assert abs(float(fs_exchange(far, length))) < abs(
            float(fs_exchange(np.array(gamma * 1.01), length)))


class TestTheIntegratedCorrection:
    def test_the_effective_length_is_the_equal_volume_cube(self):
        assert effective_length(27.0) == pytest.approx(3.0)

    def test_a_cubic_cell_has_no_deviation(self):
        density = np.full(8, 0.05)
        result = kzk_correction(density, 1.0, np.diag([2.0, 2.0, 2.0]))
        assert result.cubic_deviation == pytest.approx(0.0, abs=1e-12)

    def test_a_slab_cell_reports_its_deviation(self):
        """The fit is for cubic cells; a flat cell must say so."""
        density = np.full(8, 0.05)
        result = kzk_correction(density, 1.0, np.diag([1.0, 5.0, 5.0]))
        assert result.cubic_deviation > 0.3

    def test_an_empty_density_is_refused(self):
        with pytest.raises(ValueError, match="below the vacuum floor"):
            kzk_correction(np.zeros(8), 1.0, np.eye(3))

    def test_the_scopes_are_kept_apart(self):
        density = np.full(64, 0.02)
        result = kzk_correction(density, 0.5, np.diag([4.0] * 3), n_cells=4,
                                energy_per_cell=-2.0)
        assert result.correction_per_cell == pytest.approx(
            result.correction / 4)
        assert result.corrected_energy_per_cell == pytest.approx(
            -2.0 + result.correction_per_cell)

    def test_the_correction_is_the_difference_of_the_two_integrals(self):
        density = np.full(64, 0.02)
        result = kzk_correction(density, 0.5, np.diag([4.0] * 3))
        assert result.correction == pytest.approx(
            result.xc_infinite - result.xc_finite, rel=1e-12)

    def test_a_huge_cell_needs_almost_no_correction(self):
        """``e_xc^FS -> e_xc^inf``, so the integral of the difference dies."""
        density = np.full(64, 0.02)
        small = kzk_correction(density, 0.5, np.diag([4.0] * 3))
        large = kzk_correction(density, 0.5, np.diag([80.0] * 3))
        assert abs(large.correction) < abs(small.correction)


class TestOnARun:
    @pytest.fixture(scope="class")
    def solved_chain(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=np.diag([2.0, 10.0, 10.0]),
                      pbc=[True, False, False])
        atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                               kpts={"size": (2, 1, 1), "gamma": True},
                               basis="HAO", h=0.35, trace=False,
                               optimizer=SLSQP)
        atoms.get_potential_energy()
        return atoms

    def test_it_is_selectable(self, solved_chain):
        assert isinstance(solved_chain.calc.finite_size_correction("kzk"),
                          KZKCorrection)

    def test_the_density_holds_the_right_electron_count(self, solved_chain):
        """``int n`` over the grid is the check that the density is the run's."""
        result = solved_chain.calc.finite_size_correction("kzk")
        assert result.n_electrons == pytest.approx(2.0, rel=1e-6)

    def test_it_flags_a_non_cubic_cell(self, solved_chain):
        """2 x 10 x 10 Angstrom is nothing like the cube the fit assumes."""
        result = solved_chain.calc.finite_size_correction("kzk")
        assert result.cubic_deviation > 0.4

    def test_the_summary_names_the_assumptions(self, solved_chain):
        text = solved_chain.calc.finite_size_correction("kzk").summary()
        assert "cubic deviation" in text and "r_s in" in text

    def test_all_three_schemes_are_reachable(self, solved_chain):
        import warnings

        from mandacaru.core.ccmh import CCMHCorrection
        from mandacaru.core.mpc import FiniteSizeCorrection

        assert isinstance(solved_chain.calc.finite_size_correction("mpc"),
                          FiniteSizeCorrection)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert isinstance(solved_chain.calc.finite_size_correction("ccmh"),
                              CCMHCorrection)
        assert isinstance(solved_chain.calc.finite_size_correction("kzk"),
                          KZKCorrection)

    def test_an_unknown_scheme_names_all_three(self, solved_chain):
        with pytest.raises(ValueError, match="mpc.*ccmh.*kzk"):
            solved_chain.calc.finite_size_correction("nonesuch")
