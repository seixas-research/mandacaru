# -*- coding: utf-8 -*-
# file: test/integrals/test_poisson.py

# This code is part of Mandacaru.
# MIT License

"""The isolated Coulomb kernels of :mod:`mandacaru.integrals.poisson`: bare
and screened, against the closed-form interaction of two Gaussians."""

import numpy as np
import pytest
from scipy.special import erf

from scipy import fft as sfft

from mandacaru.integrals.poisson import (SPLIT_REACH, PoissonFFTSolver,
                                         short_range_coulomb_kernel)
from mandacaru.integrals import poisson as poisson_mod

H = 0.25
N = 40


def _gaussian(exponent, center):
    """A unit Gaussian charge on an N^3 grid of spacing H (Bohr)."""
    axis = (np.arange(N) - (N - 1) / 2) * H
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    return ((exponent / np.pi) ** 1.5 * np.exp(-exponent * r2)).reshape(-1)


def _interaction(solver, a, b):
    return float(np.real(np.sum(a * solver.solve(b))) * H ** 3)


class TestTheIsolatedSolver:
    @pytest.mark.parametrize("omega", [0.11, 0.6, 2.0])
    @pytest.mark.parametrize("a, b, distance", [(3.0, 2.0, 1.5),
                                                (1.0, 1.0, 0.0),
                                                (4.0, 4.0, 0.0)])
    def test_two_gaussians_interact_as_the_closed_form(self, omega, a, b,
                                                       distance):
        r"""Unit Gaussians of exponents ``a`` and ``b`` a distance ``R``
        apart interact through :math:`\operatorname{erfc}(\omega r)/r` as
        :math:`[\operatorname{erf}(\sqrt\mu R) - \operatorname{erf}(\sqrt p
        R)]/R`, :math:`\mu = ab/(a+b)`, :math:`1/p = 1/a + 1/b + 1/\omega^2`
        (:math:`2\sqrt{\mu/\pi} - 2\sqrt{p/\pi}` at ``R = 0``) -- to
        1e-8.  omega = 2 lies above the split, where the whole kernel is
        spectral."""
        rho_a = _gaussian(a, [0.0, 0.0, -0.5 * distance])
        rho_b = _gaussian(b, [0.0, 0.0, 0.5 * distance])
        solver = PoissonFFTSolver(N, spacing=H, omega=omega)
        assert (solver.split == omega) == (omega == 2.0)
        mu = a * b / (a + b)
        p = 1.0 / (1.0 / a + 1.0 / b + 1.0 / omega ** 2)
        exact = (2.0 * (np.sqrt(mu) - np.sqrt(p)) / np.sqrt(np.pi)
                 if distance == 0.0
                 else (erf(np.sqrt(mu) * distance)
                       - erf(np.sqrt(p) * distance)) / distance)
        assert _interaction(solver, rho_a, rho_b) == pytest.approx(exact,
                                                                   rel=1e-8)

    @pytest.mark.parametrize("a, b, distance", [(3.0, 2.0, 1.5),
                                                (1.0, 1.0, 0.0),
                                                (4.0, 4.0, 0.0)])
    def test_the_bare_kernel_is_exact(self, a, b, distance):
        """The default (omega = 0): the bare Coulomb interaction of two
        Gaussians, erf(sqrt(mu) R)/R (2 sqrt(mu/pi) at R = 0), to 1e-8,
        self interaction included."""
        rho_a = _gaussian(a, [0.0, 0.0, -0.5 * distance])
        rho_b = _gaussian(b, [0.0, 0.0, 0.5 * distance])
        mu = a * b / (a + b)
        exact = (2.0 * np.sqrt(mu / np.pi) if distance == 0.0
                 else erf(np.sqrt(mu) * distance) / distance)
        solver = PoissonFFTSolver(N, spacing=H)
        assert solver.omega == 0.0
        assert _interaction(solver, rho_a, rho_b) == pytest.approx(exact,
                                                                   rel=1e-8)

    def test_the_padding_does_not_depend_on_omega(self):
        """Nothing long-ranged is spectral, so a small omega costs what the
        bare kernel does: 2N - 1, rounded up to an FFT-friendly length."""
        solver = PoissonFFTSolver(N, spacing=H, omega=0.01)
        assert solver.L == PoissonFFTSolver(N, spacing=H).L
        assert solver.L == (sfft.next_fast_len(2 * N - 1),) * 3

    def test_a_small_grid_is_padded_past_the_spectral_reach(self):
        """On a grid shorter than the spectral part's reach (~14 nodes),
        2N - 1 would put its images inside the box."""
        solver = PoissonFFTSolver(6, spacing=H)
        reach = int(np.ceil(SPLIT_REACH / (solver.split * H)))
        assert all(L >= 6 + reach > 2 * 6 - 1 for L in solver.L)

    def test_omega_must_not_be_negative(self):
        with pytest.raises(ValueError, match="omega"):
            PoissonFFTSolver(8, spacing=H, omega=-0.1)


class TestKernels:
    def test_the_short_range_kernel_in_reciprocal_space(self):
        omega = 0.11
        g2 = np.array([0.0, 1e-12, 0.01, 1.0, 100.0])
        kernel = short_range_coulomb_kernel(g2, omega)
        assert kernel[0] == pytest.approx(np.pi / omega ** 2)
        assert kernel[1] == pytest.approx(np.pi / omega ** 2, rel=1e-9)
        expected = 4 * np.pi / g2[2:] * (1 - np.exp(-g2[2:] / (4 * omega ** 2)))
        np.testing.assert_allclose(kernel[2:], expected, rtol=1e-14)


def _gap(shape, L):
    """Mask of the padded mesh points with an offset in the gap on some axis
    (``n <= m <= L - n``): no box-to-box convolution reads them."""
    masks = []
    for n, m in zip(shape, L):
        index = np.arange(m)
        masks.append((index >= n) & (index <= m - n))
    return (masks[0][:, None, None] | masks[1][None, :, None]
            | masks[2][None, None, :])


class TestTheEvenKernel:
    """The isolated kernel is filled symmetrically in the gap, which makes
    its transform real; the box never sees the gap."""

    SHAPE = (9, 11, 7)
    STEP = np.diag([0.4, 0.35, 0.5])

    def _densities(self, real=False):
        rng = np.random.default_rng(11)
        G = int(np.prod(self.SHAPE))
        rho = rng.normal(size=(4, G))
        return rho if real else rho + 1j * rng.normal(size=(4, G))

    @pytest.mark.parametrize("omega", [0.0, 0.3])
    def test_the_transform_is_real_on_an_orthogonal_grid(self, omega):
        solver = PoissonFFTSolver(self.SHAPE, step=self.STEP, omega=omega)
        assert solver.even and solver._Gk.dtype == np.float64

    def test_gap_values_never_reach_the_box(self):
        """Replace every gap value of the real-space kernel by noise: the
        box potentials do not move (to round-off)."""
        solver = PoissonFFTSolver(self.SHAPE, step=self.STEP)
        kernel = np.fft.ifftn(solver._Gk)
        noisy = kernel.copy()
        gap = _gap(self.SHAPE, solver.L)
        noisy[gap] = np.random.default_rng(3).normal(size=int(gap.sum()))
        rho = self._densities()
        reference = solver.solve_stack(rho)
        nx, ny, nz = self.SHAPE
        transform = np.fft.fftn(noisy)
        for p, row in enumerate(rho):
            pad = np.zeros(solver.L, dtype=complex)
            pad[:nx, :ny, :nz] = row.reshape(self.SHAPE)
            value = np.fft.ifftn(np.fft.fftn(pad) * transform)
            box = value[:nx, :ny, :nz].reshape(-1) * solver.dV
            assert np.abs(box - reference[p]).max() < 1e-12 * np.abs(
                reference[p]).max()

    def test_real_densities_take_the_real_route_to_the_same_potentials(self):
        solver = PoissonFFTSolver(self.SHAPE, step=self.STEP)
        rho = self._densities(real=True)
        as_complex = solver.solve_stack(rho.astype(complex))
        out = np.empty_like(rho)
        assert solver.solve_stack(rho, out=out) is out
        assert np.abs(out - as_complex.real).max() < 1e-13 * np.abs(
            out).max()

    def test_a_skewed_grid_keeps_its_complex_transform(self):
        """On a skewed grid the spectral part's Nyquist planes are not
        symmetric, so the kernel is not even and nothing is symmetrized."""
        step = np.array([[0.4, 0.1, 0.0], [0.0, 0.38, 0.05],
                         [0.02, 0.0, 0.41]])
        solver = PoissonFFTSolver((8, 9, 10), step=step)
        assert not solver.even and np.iscomplexobj(solver._Gk)
        with pytest.raises(ValueError, match="even"):
            solver.self_energies(np.ones((1, 720)))

    def test_self_energies_by_parseval(self):
        solver = PoissonFFTSolver(self.SHAPE, step=self.STEP)
        rho = self._densities(real=True)
        direct = np.sum(rho * solver.solve_stack(rho).real, axis=1) * solver.dV
        np.testing.assert_allclose(solver.self_energies(rho), direct,
                                   rtol=1e-12)

    def test_out_must_be_a_contiguous_buffer_of_the_stack_shape(self):
        solver = PoissonFFTSolver(self.SHAPE, step=self.STEP)
        rho = self._densities()
        with pytest.raises(ValueError, match="C-contiguous"):
            solver.solve_stack(rho, out=np.empty((rho.shape[1], 4),
                                                 complex).T)
        with pytest.raises(ValueError, match="real out"):
            solver.solve_stack(rho, out=np.empty(rho.shape))

    def test_the_kernel_cache_is_bounded_in_bytes(self, monkeypatch):
        monkeypatch.setattr(poisson_mod, "_KERNELS",
                            type(poisson_mod._KERNELS)())
        monkeypatch.setattr(poisson_mod, "KERNEL_CACHE_BYTES", 1)
        first = PoissonFFTSolver(6, spacing=H)
        PoissonFFTSolver(7, spacing=H)
        assert len(poisson_mod._KERNELS) == 1
        assert PoissonFFTSolver(6, spacing=H)._Gk is not first._Gk
