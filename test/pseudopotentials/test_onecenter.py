# -*- coding: utf-8 -*-
# file: test_onecenter.py

"""The PAW-LCAO one-center two-body machinery (:mod:`mandacaru.pseudopotentials.onecenter`).

The Coulomb correction is the two-body operator of every wave-function
Hamiltonian's spheres (:class:`~mandacaru.pseudopotentials.onecenter.OneCenterExact`,
since 2026-10-12); before that it was measured machinery only (0.08-0.33 eV
on oxygen, the term the linearized treatment drops).  These tests pin the
pieces against closed forms.  Its short-range counterpart and the frozen
semilocal exchange are the one-center terms of the HSE06 hybrid
(:class:`~mandacaru.pseudopotentials.onecenter.OneCenterHybrid`).
"""

from fractions import Fraction
from math import factorial
from types import SimpleNamespace

from pathlib import Path

import numpy as np
import pytest

from mandacaru import _native
from mandacaru.pseudopotentials.native_context import context
from mandacaru.pseudopotentials.onecenter import (OneCenterExact,
                                                  OneCenterHybrid,
                                                  core_orbitals,
                                                  exact_one_center_terms,
                                                  frozen_exchange_terms,
                                                  one_center_coulomb,
                                                  one_center_exchange_tensor)


def orbital(l, m, real=True):
    return SimpleNamespace(l=l, m=m, real=real)


# The native radial and angular kernels have no Python route; these helpers
# only hand arrays to them.

def radial_coulomb(r, rho_1, rho_2, L):
    return _native.paw_radial_coulomb(
        context(), *(np.ascontiguousarray(x, dtype=float).ravel()
                     for x in (r, rho_1, rho_2)), int(L))


def long_range_radial_kernel(r1, r2, levels, omega, points=None):
    levels = [int(L) for L in levels]
    kernel = _native.paw_long_range_kernel(
        context(), np.atleast_1d(np.ascontiguousarray(r1, dtype=float)),
        np.atleast_1d(np.ascontiguousarray(r2, dtype=float)), levels,
        float(omega), None if points is None else int(points))
    return {L: kernel[i] for i, L in enumerate(levels)}


def long_range_shape_potential(radius, r_g, L, omega):
    radius = np.asarray(radius, dtype=float)
    return _native.paw_long_range_shape(
        context(), np.ascontiguousarray(radius.ravel()), float(r_g), int(L),
        float(omega)).reshape(radius.shape)[()]


def angular_coupling(L, a, b, c, d):
    real = bool(a.real)
    key = np.array([[int(L), a.l, a.m, b.l, b.m, c.l, c.m, d.l, d.m]],
                   dtype=np.int64)
    value = _native.paw_angular_coupling(key, real)[0]
    return float(value.real) if real else complex(value)


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


    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_points_raise(self, n):
        """An empty grid panicked inside the native code (a PanicException);
        one point has no interval to integrate.  Both are a ValueError."""
        r = np.full(n, 0.5)
        with pytest.raises(ValueError, match="at least two grid points"):
            radial_coulomb(r, np.ones(n), np.ones(n), 0)

    def test_two_points_are_the_smallest_grid(self):
        assert np.isfinite(radial_coulomb(np.array([0.0, 1.0]), np.ones(2),
                                          np.ones(2), 0))


class TestAngularCoupling:
    def test_s_densities_carry_only_the_monopole(self):
        s = orbital(0, 0)
        assert angular_coupling(0, s, s, s, s) == pytest.approx(1.0)
        assert angular_coupling(1, s, s, s, s) == 0
        assert angular_coupling(2, s, s, s, s) == 0

    @pytest.mark.parametrize("real", [True, False])
    def test_sp_transition_density_is_a_dipole(self, real):
        """4 pi / (2L+1) sum_M G G for the s-p pair density: 1/3 at L = 1."""
        s, p = orbital(0, 0, real), orbital(1, 0, real)
        assert angular_coupling(1, s, p, s, p) == pytest.approx(1.0 / 3.0)
        assert angular_coupling(0, s, p, s, p) == 0          # no monopole

    def test_the_real_coupling_is_the_complex_one_transformed(self):
        """Every four-orbital coupling of s, p and d functions, the real
        harmonics against the complex ones rotated shell by shell."""
        from mandacaru.basis._angular import real_harmonic_transform

        shells = [0, 1, 2]
        U = {l: real_harmonic_transform(l) for l in shells}
        rng = np.random.default_rng(4)
        for _ in range(12):
            la, lb, lc, ld = rng.choice(shells, 4)
            ma, mb, mc, md = (int(rng.integers(-l, l + 1))
                              for l in (la, lb, lc, ld))
            for L in range(0, 5):
                real = angular_coupling(L, *(orbital(l, m) for l, m in
                                             zip((la, lb, lc, ld),
                                                 (ma, mb, mc, md))))
                # (ab|cd) carries conj on a and c: S_a = sum U_am Y_m, so
                # the complex couplings are summed with conj(U) on a, c.
                total = 0.0 + 0.0j
                for na in range(-la, la + 1):
                    for nb in range(-lb, lb + 1):
                        for nc in range(-lc, lc + 1):
                            for nd in range(-ld, ld + 1):
                                weight = (np.conj(U[la][ma + la, na + la])
                                          * U[lb][mb + lb, nb + lb]
                                          * np.conj(U[lc][mc + lc, nc + lc])
                                          * U[ld][md + ld, nd + ld])
                                if weight == 0:
                                    continue
                                total += weight * angular_coupling(
                                    L, orbital(la, na, False),
                                    orbital(lb, nb, False),
                                    orbital(lc, nc, False),
                                    orbital(ld, nd, False))
                assert abs(real - total) < 1e-12


@pytest.fixture(scope="module")
def oxygen():
    from mandacaru.pseudopotentials import get_paw
    from mandacaru.pseudopotentials.paw import paw_projectors
    try:
        dataset = get_paw("O")
    except FileNotFoundError:
        pytest.skip("MANDACARU_PAW_PATH is not configured")
    projectors = paw_projectors(["O"], np.zeros((1, 3)), {"O": dataset})
    return dataset, projectors


class TestOnAPAWDataset:
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

    def test_spinor_blocks_are_the_derivative_and_the_channel_form(
            self, water_projectors):
        """Two-component blocks: the ``(2P, 2P)`` operator is the derivative
        of the energy in every block (the off-diagonal ones included), and
        spin-diagonal blocks give the two channels' energy and operators."""
        projectors, datasets = water_projectors
        hybrid = OneCenterHybrid(projectors, datasets, 0.11, 0.25)
        P = len(projectors)
        rng = np.random.default_rng(5)
        A = rng.normal(size=(2 * P, 2 * P)) + 1j * rng.normal(
            size=(2 * P, 2 * P))
        D = A @ A.conj().T / (2 * P)
        dD = rng.normal(size=(2 * P, 2 * P)) + 1j * rng.normal(
            size=(2 * P, 2 * P))
        dD = 0.5 * (dD + dD.conj().T)

        def blocks(X):
            return [[X[s * P:(s + 1) * P, t * P:(t + 1) * P]
                     for t in range(2)] for s in range(2)]
        terms = hybrid.evaluate_spinor(blocks(D))
        step = 1e-5
        plus = hybrid.evaluate_spinor(blocks(D + step * dD)).energy
        minus = hybrid.evaluate_spinor(blocks(D - step * dD)).energy
        directional = float(np.real(np.sum(terms.operators[0].T * dD)))
        assert (plus - minus) / (2 * step) == pytest.approx(directional,
                                                             rel=1e-8)
        up, down = blocks(D)[0][0], blocks(D)[1][1]
        zero = np.zeros_like(up)
        diagonal = hybrid.evaluate_spinor([[up, zero], [zero, down]])
        channels = hybrid.evaluate(up, down)
        assert diagonal.energy == pytest.approx(channels.energy, rel=1e-13)
        O = diagonal.operators[0]
        assert np.allclose(O[:P, :P], channels.operators[0], atol=1e-13)
        assert np.allclose(O[P:, P:], channels.operators[1], atol=1e-13)
        assert not np.any(O[:P, P:])

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


class TestOneCenterExact:
    """The exact one-center terms of a Hartree-Fock PAW-LCAO Hamiltonian."""

    def test_three_j_symbols(self):
        three_j = _native.paw_three_j_squared
        assert three_j(1, 1, 0) == pytest.approx(1 / 3)
        assert three_j(0, 1, 1) == pytest.approx(1 / 3)
        assert three_j(1, 1, 2) == pytest.approx(2 / 15)
        assert three_j(2, 2, 2) == pytest.approx(2 / 35)
        assert three_j(1, 1, 1) == 0.0          # odd sum
        assert three_j(0, 0, 2) == 0.0          # triangle

    @pytest.mark.parametrize("abc", [(60, 60, 60), (100, 100, 100),
                                     (128, 128, 128), (128, 128, 2),
                                     (64, 100, 128)])
    def test_three_j_symbols_at_large_momenta(self, abc):
        """Against the exact rational value; the float factorial form returned
        0 at (60, 60, 60) and NaN at (100, 100, 100)."""
        a, b, c = abc
        j, g = a + b + c, (a + b + c) // 2
        exact = (Fraction(factorial(j - 2 * a) * factorial(j - 2 * b)
                          * factorial(j - 2 * c), factorial(j + 1))
                 * Fraction(factorial(g), factorial(g - a) * factorial(g - b)
                            * factorial(g - c)) ** 2)
        value = _native.paw_three_j_squared(a, b, c)
        assert value == pytest.approx(float(exact), rel=1e-12)

    def test_the_channels_are_charged_to_the_context(self):
        """The factors, the stack and the scratch are reserved: a context too
        small for them raises MemoryError instead of allocating."""
        c = np.ones((3, 4), dtype=complex)                       # 192 bytes
        factors = [np.ones((1, 2, 2))]                           # 32 bytes
        members = [[0, 1]]
        for limit in (200, 300):        # the factors do not fit; the scratch does not
            small = _native.Context(threads=1, memory_limit_bytes=limit)
            with pytest.raises(MemoryError):
                _native.paw_exact_channels(small, c, members, factors, True)
            with pytest.raises(MemoryError):
                _native.paw_exact_channel_derivatives(small, c, c, members,
                                                      factors, True)
        roomy = _native.Context(threads=1, memory_limit_bytes=1 << 20)
        stack = _native.paw_exact_channels(roomy, c, members, factors, True)
        assert stack.shape == (1, 3, 3) and stack.dtype == float
        assert np.all(stack == 4.0)             # (1, 1) U (1, 1)^T, U all ones

    def test_three_j_symbols_beyond_the_range_raise(self):
        three_j = _native.paw_three_j_squared
        assert three_j(128, 128, 1) == 0.0      # odd sum, inside the range
        with pytest.raises(ValueError, match="up to 128"):
            three_j(129, 129, 2)
        with pytest.raises(ValueError, match="up to 128"):
            three_j(1, 1, 10 ** 6)

    def test_the_factors_reproduce_the_tensor(self, water_projectors):
        """sum_k lambda_k U^k (x) U^k is Delta W, with the antisymmetric pairs
        as the only null space: rank 33 of each atom's 64 pairs.
        Hydrogen had 3 of 4 (an s channel only) until 2026-10-25, when its
        dataset gained an empty p channel: 8 projectors, as oxygen has."""
        projectors, datasets = water_projectors
        exact = OneCenterExact(projectors, datasets)
        ranks = [len(entry["values"]) for entry in exact.atoms]
        assert ranks == [33, 33, 33] and exact.rank == 99
        for entry in exact.atoms:
            U, lam, W = entry["factors"], entry["values"], entry["W"]
            rebuilt = np.einsum("k,kab,kcd->abcd", lam, U, U)
            assert np.abs(rebuilt - W).max() < 1e-9 * np.abs(W).max()
            assert np.allclose(U, U.transpose(0, 2, 1))

    def test_the_one_body_blocks_are_spherical(self, water_projectors):
        """-V - Delta W.D0 + K couples only equal (l, m), with the same block
        for every m of a channel."""
        projectors, datasets = water_projectors
        exact = OneCenterExact(projectors, datasets)
        oxygen = exact.atoms[0]
        own = oxygen["projectors"]
        coupling = oxygen["coupling"]
        for i, a in enumerate(own):
            for j, b in enumerate(own):
                if (a.l, a.m) != (b.l, b.m):
                    assert abs(coupling[i, j]) < 1e-9
        blocks = exact.coupling_blocks()
        p = [blocks[(0, 1, m)] for m in (-1, 0, 1)]
        assert np.allclose(p[0], p[1], atol=1e-10)
        assert np.allclose(p[1], p[2], atol=1e-10)

    def test_the_core_and_its_exchange(self, water_projectors):
        """The re-solved reference atom gives back the stored core density,
        and the core exchange is attractive on the bound partial waves."""
        _projectors, datasets = water_projectors
        oxygen = datasets[0]
        core = core_orbitals(oxygen)
        assert [(n, l, f) for n, l, f, _R in core] == [(1, 0, 2.0)]
        assert oxygen._core_density_error < 1e-3
        assert core_orbitals(datasets[1]) == []            # hydrogen
        terms = exact_one_center_terms(oxygen, "dual")
        for l, (V, K, D0) in terms.items():
            assert np.allclose(V, V.T) and np.allclose(K, K.T)
            assert K[0, 0] < 0.0
        # The reference occupations: 2 in the 2s, 4 / 3 per 2p orbital.
        assert terms[0][2][0, 0] == pytest.approx(2.0)
        assert terms[1][2][0, 0] == pytest.approx(4.0 / 3.0)

    def test_the_spheres_cancel_a_lone_electron(self, water_projectors):
        """Hartree and exchange of one electron in one sphere cancel exactly:
        1/2 D D DeltaW - 1/2 D D DeltaW for a pure-state D."""
        projectors, datasets = water_projectors
        exact = OneCenterExact(projectors, datasets)
        entry = exact.atoms[0]
        W = entry["W"]
        c = np.random.default_rng(5).normal(size=W.shape[0])
        D = np.outer(c, c)
        hartree = 0.5 * np.einsum("ab,cd,abcd->", D, D, W)
        exchange = -0.5 * np.einsum("ad,cb,abcd->", D, D, W)
        assert hartree + exchange == pytest.approx(0.0, abs=1e-12)


class TestTheDShellAgainstItsRadialIntegrals:
    """``DeltaW`` of one d orbital, ``(d_xy d_xy | d_xy d_xy)``, is the
    all-electron minus smooth-plus-compensation Slater combination
    ``F0 + 4/49 F2 + 36/441 F4`` of the same radial pair, from the radial
    integrals alone (vanadium's 3d, whose augmentation shape differs from
    the all-electron wave by 0.4 to 0.9 eV in ``F^L``; the semicore
    dataset's 1.67 Bohr d sphere holds 1.5 eV of it, the old 1.156 Bohr one
    0.56 eV)."""

    def test_the_tensor_is_the_slater_combination(self):
        from mandacaru.pseudopotentials import get_paw
        from mandacaru.pseudopotentials.multipoles import (partial_waves,
                                                           shape_function)

        vanadium = get_paw("V")
        r_g = float(vanadium.compensation_radius)
        projectors = [SimpleNamespace(l=2, m=m, real=True, index=0,
                                      atom_index=0, projector_basis="dual")
                      for m in range(-2, 3)]
        tensor = one_center_coulomb(vanadium, projectors, "dual")
        # real d_xy is m = -2
        r = np.asarray(vanadium.r)
        inside = r <= vanadium.channels[2].r_cut
        r = r[inside]
        ae, smooth = (w[0][inside] for w in partial_waves(vanadium, 2))
        density = ae * ae
        smooth_density = smooth * smooth
        expected = 0.0
        for L, factor in ((0, 1.0), (2, 4.0 / 49.0), (4, 36.0 / 441.0)):
            moment = np.trapezoid((density - smooth_density) * r ** (L + 2),
                                  r)
            compensated = smooth_density + moment * shape_function(r, r_g, L)
            expected += factor * (radial_coulomb(r, density, density, L)
                                  - radial_coulomb(r, compensated,
                                                   compensated, L))
        assert tensor[0, 0, 0, 0] == pytest.approx(expected, rel=2e-3)
        assert 0.05 < tensor[0, 0, 0, 0] < 0.06        # about 1.5 eV


REFERENCE = (Path(__file__).resolve().parents[1] / "migration" / "reference"
             / "paw" / "onecenter.npz")
#: Native against the pre-port Python: the measured worst difference is 6e-11
#: (the Gaussian-weighted H sphere), so 1e-9 holds with room.
RTOL = 1e-9


def close(value, frozen, rtol=RTOL):
    """The largest difference, relative to the largest frozen entry."""
    frozen = np.asarray(frozen)
    scale = max(float(np.abs(frozen).max()), 1e-300)
    assert float(np.abs(np.asarray(value) - frozen).max()) <= rtol * scale


@pytest.fixture(scope="module")
def frozen():
    return np.load(REFERENCE)


@pytest.fixture(scope="module")
def datasets():
    from mandacaru.pseudopotentials import get_paw
    try:
        return {symbol: get_paw(symbol) for symbol in ("O", "H", "V", "Cu")}
    except FileNotFoundError:
        pytest.skip("MANDACARU_PAW_PATH is not configured")


class TestFrozenPythonOutputs:
    """The Rust routines against the outputs of the Python they replaced
    (``test/migration/reference/paw/onecenter.npz``; the inputs are in the JSON
    beside it).  ``radial_coulomb`` was frozen on its SciPy branch: the C-ABI
    kernel integrates one interval short for an odd number of points."""

    @staticmethod
    def own(dataset, symbol, basis, real=True, only=None):
        from mandacaru.pseudopotentials.paw import paw_projectors
        projectors = paw_projectors([symbol], np.zeros((1, 3)),
                                    {symbol: dataset}, projector_basis=basis,
                                    real=real)
        return [p for p in projectors if only is None or p.l in only]

    def test_the_radial_slater_integral(self, frozen):
        grids = {"uniform_600": np.linspace(0, 6, 600),
                 "uniform_601": np.linspace(0, 6, 601),
                 "quadratic_500": 6 * np.linspace(0, 1, 500) ** 2,
                 "quadratic_501": 6 * np.linspace(0, 1, 501) ** 2}
        values = []
        for r in grids.values():
            a, b, c = np.exp(-r ** 2), r * np.exp(-0.5 * r), np.exp(-r) * (1 + r)
            for L in range(4):
                values += [radial_coulomb(r, f, s, L)
                           for f, s in ((a, a), (a, b), (b, c))]
        close(values, frozen["radial_coulomb"], 1e-12)

    def test_the_screened_kernel_and_its_shape_potential(self, frozen):
        r1, r2 = np.linspace(0, 1.5, 7), np.array([0.2, 0.9, 1.4, 2.5, 3.0])
        kernels = []
        for omega, points in ((0.11, None), (1.0, None), (20.0, None),
                              (20.0, 64), (0.05, 24)):
            f = long_range_radial_kernel(r1, r2, [0, 1, 2, 3], omega,
                                         points=points)
            kernels.append(np.stack([f[L] for L in range(4)]))
        close(np.stack(kernels), frozen["long_range_kernel"])
        radius = np.linspace(0, 30, 61)
        shapes = [long_range_shape_potential(radius, r_g, L, omega)
                  for r_g, L, omega in [(1.45, 0, 0.11), (1.45, 1, 0.11),
                                        (1.45, 2, 1.0), (1.7, 0, 3.0)]]
        close(np.stack(shapes), frozen["long_range_shape"])

    @pytest.mark.parametrize("real", [True, False])
    def test_the_angular_couplings(self, frozen, real):
        keys = frozen["angular_keys"]
        values = [angular_coupling(
            int(k[0]), *(orbital(int(k[1 + 2 * i]), int(k[2 + 2 * i]), real)
                         for i in range(4))) for k in keys]
        close(values, frozen["angular_real" if real else "angular_complex"],
              1e-12)

    @pytest.mark.parametrize("name,symbol,basis,real,only", [
        ("O_raw", "O", "raw", True, None), ("O_dual", "O", "dual", True, None),
        ("H_raw", "H", "raw", True, None),
        ("V_d_dual", "V", "dual", True, (2,)),
        ("Cu_d_raw", "Cu", "raw", True, (2,)),
        ("O_raw_complex", "O", "raw", False, None)])
    def test_the_coulomb_tensor(self, frozen, datasets, name, symbol, basis,
                                real, only):
        dataset = datasets[symbol]
        tensor = one_center_coulomb(
            dataset, self.own(dataset, symbol, basis, real, only), basis)
        assert np.iscomplexobj(tensor) == (not real)
        close(tensor, frozen[f"tensor_{name}"])

    @pytest.mark.parametrize("name,symbol,basis,omega", [
        ("O_raw_w011", "O", "raw", 0.11), ("O_dual_w011", "O", "dual", 0.11),
        ("H_raw_w5", "H", "raw", 5.0), ("O_raw_w0", "O", "raw", 0.0)])
    def test_the_screened_exchange_tensor(self, frozen, datasets, name, symbol,
                                          basis, omega):
        dataset = datasets[symbol]
        tensor = one_center_exchange_tensor(
            dataset, self.own(dataset, symbol, basis), omega, basis)
        close(tensor, frozen[f"exchange_{name}"])

    @pytest.mark.parametrize("symbol", ["O", "H", "Cu"])
    @pytest.mark.parametrize("basis", ["raw", "dual"])
    def test_the_frozen_exchange_terms(self, frozen, datasets, symbol, basis):
        constant, blocks = frozen_exchange_terms(datasets[symbol], 0.11, basis)
        assert constant == pytest.approx(
            float(frozen[f"frozen_{symbol}_{basis}_constant"]), rel=RTOL)
        for l, block in blocks.items():
            close(block, frozen[f"frozen_{symbol}_{basis}_{l}"])

    @pytest.mark.parametrize("functional,symbol", [
        ("lda", "O"), ("lda", "H"), ("lda", "V"), ("lda", "Cu"),
        ("pbe", "O"), ("pbe", "Cu")])
    @pytest.mark.parametrize("basis", ["raw", "dual"])
    def test_the_core_and_the_exact_terms(self, frozen, datasets, functional,
                                          symbol, basis):
        """The V of a PBE dataset uses the atom solver's radial scheme (the
        one the datasets were unscreened with), not the variational one."""
        from mandacaru.pseudopotentials import get_paw
        try:
            dataset = (datasets[symbol] if functional == "lda"
                       else get_paw(symbol, xc="pbe"))
        except FileNotFoundError:
            pytest.skip("the pbe-sr library is not configured")
        tag = f"{functional}_{symbol}"
        for n, l, _f, R in core_orbitals(dataset):
            close(R, frozen[f"core_{tag}_{n}_{l}"])
        terms = exact_one_center_terms(dataset, basis)
        for l, (V, K, D0) in terms.items():
            for name, array in (("V", V), ("K", K), ("D0", D0)):
                close(array, frozen[f"exact_{tag}_{basis}_{name}_{l}"])

    def test_the_hybrid_on_water(self, frozen, datasets):
        from mandacaru.pseudopotentials.paw import paw_projectors
        symbols = ["O", "H", "H"]
        positions = np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.48],
                              [0.0, -0.76, -0.48]])
        pair = {"O": datasets["O"], "H": datasets["H"]}
        projectors = paw_projectors(symbols, positions, pair)
        hybrid = OneCenterHybrid(projectors, [pair[s] for s in symbols],
                                 0.11, 0.25)
        assert hybrid.constant == pytest.approx(
            float(frozen["hybrid_constant"]), rel=RTOL)
        P = len(projectors)
        spinor = frozen["hybrid_spinor_density"]
        runs = {
            "closed_real": hybrid.evaluate(frozen["hybrid_closed_real_density"]),
            "closed_complex": hybrid.evaluate(frozen["hybrid_up_density"]),
            "channels": hybrid.evaluate(frozen["hybrid_up_density"],
                                        frozen["hybrid_down_density"]),
            "spinor": hybrid.evaluate_spinor(
                [[spinor[:P, :P], spinor[:P, P:]],
                 [spinor[P:, :P], spinor[P:, P:]]])}
        for name, terms in runs.items():
            close([terms.energy, terms.exact_exchange, terms.semilocal],
                  frozen[f"hybrid_{name}_energy"])
            for s, operator in enumerate(terms.operators):
                close(operator, frozen[f"hybrid_{name}_operator_{s}"])
        assert runs["closed_real"].operators[0].dtype == float

        exact = OneCenterExact(projectors, [pair[s] for s in symbols])
        assert exact.rank == int(frozen["exact_rank"])
        for k, entry in enumerate(exact.atoms[:2]):
            for key in ("W", "V", "K", "D0", "coupling", "values"):
                close(entry[key], frozen[f"exactclass_{k}_{key}"])
        C, dC = frozen["exactclass_C"], frozen["exactclass_dC"]
        G, lam = exact.channels(C)
        close(np.einsum("k,kab,kcd->abcd", lam, G, G),
              frozen["exactclass_contraction"])
        dG = exact.channel_derivatives(C, dC)
        close(2.0 * np.einsum("k,kab,kcd->abcd", lam, dG, G),
              frozen["exactclass_derivative_contraction"])
        for key, block in exact.coupling_blocks().items():
            name = "exactclass_block_" + "_".join(str(x) for x in key)
            if name in frozen.files:
                close(block, frozen[name])
