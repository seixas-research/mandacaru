# -*- coding: utf-8 -*-
# file: test_onecenter.py

"""The PAW-LCAO one-center two-body machinery (:mod:`mandacaru.pseudopotentials.onecenter`).

The Coulomb correction is validated machinery that is deliberately **not**
wired into the Hamiltonian: it measured the term the linearized one-center
treatment drops (0.08-0.33 eV on oxygen) and thereby ruled it out as the cause
of the p-valence binding failure.  These tests pin the pieces against closed
forms so the measurement stays reproducible.  Its short-range counterpart and
the frozen semilocal exchange are the one-center terms of the HSE06 hybrid
(:class:`~mandacaru.pseudopotentials.onecenter.OneCenterHybrid`), which are
wired in.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mandacaru.pseudopotentials.onecenter import (OneCenterHybrid,
                                                  angular_coupling,
                                                  frozen_exchange_terms,
                                                  long_range_radial_kernel,
                                                  long_range_shape_potential,
                                                  one_center_coulomb,
                                                  one_center_exchange_tensor,
                                                  radial_coulomb)


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
            pytest.skip("MANDACARU_PAW_PATH is not configured")
        projectors = paw_projectors(["O"], np.zeros((1, 3)), {"O": dataset})
        return dataset, projectors

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


class TestLongRangeKernel:
    r"""The multipole components of :math:`\operatorname{erf}(\omega r)/r`."""

    def test_small_omega_is_the_series(self):
        r""":math:`2\omega/\sqrt\pi - (2\omega^3/3\sqrt\pi)(r^2 + r'^2)` at
        :math:`L = 0`, :math:`(4\omega^3/3\sqrt\pi)\,rr'` at :math:`L = 1`,
        up to the :math:`\omega^5 R^4` term (< 3e-6 here)."""
        omega = 0.05
        r = np.linspace(0.0, 1.5, 7)
        f = long_range_radial_kernel(r, r, [0, 1, 2], omega)
        c = 2.0 * omega ** 3 / (3.0 * np.sqrt(np.pi))
        R1, R2 = np.meshgrid(r, r, indexing="ij")
        constant = 2 * omega / np.sqrt(np.pi)
        assert np.abs(f[0] - constant).max() > 1e-4
        assert np.allclose(f[0], constant - c * (R1 ** 2 + R2 ** 2), atol=3e-6)
        assert np.allclose(f[1], 2.0 * c * R1 * R2, atol=3e-6)
        assert np.abs(f[2]).max() < 3e-6

    def test_large_omega_is_the_coulomb_kernel(self):
        """Away from ``r = r'`` the error function is one there:
        ``r_<^L / r_>^(L+1)``."""
        r1, r2 = np.array([0.5, 1.0]), np.array([2.0, 3.0])
        f = long_range_radial_kernel(r1, r2, [0, 1, 2], 20.0, points=64)
        for L in (0, 1, 2):
            assert np.allclose(f[L], r1[:, None] ** L / r2[None, :] ** (L + 1),
                               rtol=1e-6)

    def test_the_shape_potential_tends_to_its_screened_monopole(self):
        r"""Far from a unit monopole shape the potential is
        :math:`4\pi\operatorname{erf}(\omega r)/r` up to the shape's finite
        size; at small :math:`\omega` it is the constant
        :math:`4\pi\cdot 2\omega/\sqrt\pi` everywhere."""
        from scipy.special import erf
        r = np.array([20.0, 30.0])
        v = long_range_shape_potential(r, 1.45, 0, 0.11)
        assert np.allclose(v, 4 * np.pi * erf(0.11 * r) / r, rtol=1e-4)
        omega = 1e-4
        v = long_range_shape_potential(np.array([0.0, 1.0]), 1.45, 0, omega)
        assert np.allclose(v, 8.0 * np.sqrt(np.pi) * omega, rtol=1e-6)


@pytest.fixture(scope="module")
def water_projectors():
    """The projectors of H2O's three atoms and their datasets, by atom."""
    from mandacaru.pseudopotentials import get_paw
    from mandacaru.pseudopotentials.paw import paw_projectors
    try:
        datasets = {"O": get_paw("O"), "H": get_paw("H")}
    except FileNotFoundError:
        pytest.skip("MANDACARU_PAW_PATH is not configured")
    symbols = ["O", "H", "H"]
    positions = np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.48],
                          [0.0, -0.76, -0.48]])
    projectors = paw_projectors(symbols, positions, datasets)
    return projectors, [datasets[s] for s in symbols]


def _oxygen(water_projectors):
    projectors, datasets = water_projectors
    return datasets[0], [p for p in projectors if p.atom_index == 0]


class TestOneCenterExchange:
    """The PAW-LCAO one-center terms of the screened hybrid."""

    def test_without_screening_it_is_the_coulomb_correction(
            self, water_projectors):
        dataset, own = _oxygen(water_projectors)
        W = one_center_coulomb(dataset, own)
        symmetric = 0.5 * (W + W.transpose(2, 3, 0, 1))
        assert np.allclose(one_center_exchange_tensor(dataset, own, 0.0),
                           symmetric, atol=1e-12)

    def test_the_screening_is_a_small_symmetric_correction(
            self, water_projectors):
        r"""Inside a sphere :math:`\omega r < 0.2`: the erfc kernel raises
        the reference atom's one-center exchange by ~5e-4 relative, and the
        tensor keeps both of its symmetries."""
        dataset, own = _oxygen(water_projectors)
        bare = one_center_exchange_tensor(dataset, own, 0.0)
        screened = one_center_exchange_tensor(dataset, own, 0.11)
        assert np.allclose(screened, screened.transpose(2, 3, 0, 1),
                           atol=1e-12)
        assert np.allclose(screened.conj(), screened.transpose(1, 0, 3, 2),
                           atol=1e-10)
        D = np.zeros((len(own), len(own)))
        for i, a in enumerate(own):          # the reference atom, raw basis
            for j, b in enumerate(own):
                if (a.l, a.m) == (b.l, b.m):
                    B = np.asarray(dataset.channels[a.l].vanderbilt)
                    D[i, j] = (dataset.channels[a.l].occupation
                               / (2 * a.l + 1) * B[0, a.index] * B[0, b.index])

        def exchange(W):
            return -0.25 * np.real(np.einsum("li,jk,ijkl->", D, D, W))
        assert exchange(bare) < 0.0
        assert 0.0 < exchange(screened) - exchange(bare) < 1e-3 * abs(
            exchange(bare))

    def test_the_frozen_semilocal_terms_do_not_depend_on_the_basis(
            self, water_projectors):
        r"""The raw blocks are the dual ones transformed, the constant is the
        same, and the all-electron valence exchange is the deeper one
        (:math:`X^0 < 0`)."""
        dataset, _own = _oxygen(water_projectors)
        constant_raw, raw = frozen_exchange_terms(dataset, 0.11, "raw")
        constant_dual, dual = frozen_exchange_terms(dataset, 0.11, "dual")
        assert constant_raw == pytest.approx(constant_dual, abs=1e-14)
        for l, block in dual.items():
            B_inv = np.linalg.inv(np.asarray(dataset.channels[l].vanderbilt))
            assert np.allclose(raw[l], B_inv @ block @ B_inv.T, atol=1e-12)
        linear = sum(dataset.channels[l].occupation * dual[l][0, 0]
                     for l in dual)
        assert constant_dual + linear < 0.0

    @pytest.mark.parametrize("spins", [1, 2])
    def test_the_operator_is_the_derivative_of_the_energy(
            self, water_projectors, spins):
        projectors, datasets = water_projectors
        hybrid = OneCenterHybrid(projectors, datasets, 0.11, 0.25)
        P = len(projectors)
        rng = np.random.default_rng(spins)

        def hermitian():
            A = rng.normal(size=(P, P)) + 1j * rng.normal(size=(P, P))
            return 0.5 * (A + A.conj().T)
        D = [hermitian() for _ in range(spins)]
        dD = [hermitian() for _ in range(spins)]
        terms = hybrid.evaluate(*D)
        step = 1e-5
        plus = hybrid.evaluate(*[d + step * x for d, x in zip(D, dD)]).energy
        minus = hybrid.evaluate(*[d - step * x for d, x in zip(D, dD)]).energy
        directional = sum(float(np.real(np.sum(O.T * x)))
                          for O, x in zip(terms.operators, dD))
        assert (plus - minus) / (2 * step) == pytest.approx(directional,
                                                             rel=1e-8)

    def test_the_spin_channels_and_the_limits(self, water_projectors):
        """Two equal channels are the closed shell; no fraction, no term."""
        projectors, datasets = water_projectors
        hybrid = OneCenterHybrid(projectors, datasets, 0.11, 0.25)
        P = len(projectors)
        A = np.random.default_rng(3).normal(size=(P, P))
        D = A @ A.T
        closed = hybrid.evaluate(D)
        split = hybrid.evaluate(0.5 * D, 0.5 * D)
        assert split.energy == pytest.approx(closed.energy, rel=1e-12)
        assert np.allclose(split.operators[0], closed.operators[0])
        assert np.allclose(split.operators[1], closed.operators[0])
        assert closed.energy == pytest.approx(closed.exact_exchange
                                              + closed.semilocal)
        off = OneCenterHybrid(projectors, datasets, 0.11, 0.0).evaluate(D)
        assert off.energy == 0.0
        assert not np.any(off.operators[0])
