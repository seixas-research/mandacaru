# -*- coding: utf-8 -*-
# file: test/integrals/test_exchange_correlation.py

# This code is part of Mandacaru.
# MIT License

"""The exchange-correlation functionals evaluated on the 3D grid."""

from types import SimpleNamespace

import numpy as np
import pytest

from mandacaru.basis.xc import xc_energy_density
from mandacaru.integrals import Grid
from mandacaru.integrals import exchange_correlation as xc_grid

ALPHA = 0.8


def _grid(h=0.15, box=6.0):
    return Grid(center=np.zeros(3), box_size=box, h=h, units="bohr")


def _gaussian(grid, alpha=ALPHA, electrons=2.0):
    r2 = (grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2).reshape(-1)
    return electrons * (alpha / np.pi) ** 1.5 * np.exp(-alpha * r2), r2


def _tau(grid, rho, r2):
    """A kinetic-energy density above the von Weizsaecker bound everywhere."""
    grad = xc_grid.gradient(grid, rho)
    bound = np.sum(grad * grad, axis=0) / (8.0 * np.maximum(rho, 1e-30))
    return bound + 0.3 * rho * np.exp(-0.2 * r2)


class TestSpectralDerivatives:
    def test_gradient_of_a_gaussian_is_analytic(self):
        grid = _grid()
        rho, _r2 = _gaussian(grid, electrons=1.0)
        grad = xc_grid.gradient(grid, rho)
        expected = -2.0 * ALPHA * np.stack(
            [grid.X.reshape(-1), grid.Y.reshape(-1), grid.Z.reshape(-1)]) * rho
        assert np.max(np.abs(grad - expected)) < 1e-5 * np.max(np.abs(expected))

    def test_a_complex_stack_is_differentiated_row_by_row(self):
        grid = _grid()
        rho, _r2 = _gaussian(grid, electrons=1.0)
        stack = np.stack([rho, 1j * rho])
        grad = xc_grid.gradient(grid, stack)
        assert grad.shape == (3, 2, rho.size)
        assert np.allclose(grad[:, 1], 1j * xc_grid.gradient(grid, rho))

    def test_divergence_of_a_gradient_is_the_laplacian(self):
        grid = _grid()
        rho, r2 = _gaussian(grid, electrons=1.0)
        laplacian = xc_grid.divergence(grid, xc_grid.gradient(grid, rho))
        expected = (4.0 * ALPHA ** 2 * r2 - 6.0 * ALPHA) * rho
        assert np.max(np.abs(laplacian - expected)) < 1e-6 * np.max(np.abs(expected))


class TestEnergies:
    @pytest.mark.parametrize("functional", ["lda", "pbe"])
    @pytest.mark.parametrize("relativistic", [False, True])
    def test_a_spherical_density_matches_the_radial_quadrature(
            self, functional, relativistic):
        """The 3D grid and the radial functional agree on the same density."""
        grid = _grid()
        rho, _r2 = _gaussian(grid)
        energy = xc_grid.evaluate(grid, rho, functional,
                                  relativistic=relativistic).energy
        r = np.linspace(1e-6, 12.0, 40001)
        rho_r = 2.0 * (ALPHA / np.pi) ** 1.5 * np.exp(-ALPHA * r * r)
        drho = -2.0 * ALPHA * r * rho_r
        f = xc_energy_density(rho_r, drho, functional=functional,
                              relativistic=relativistic)
        radial = np.trapezoid(4.0 * np.pi * r * r * f, r)
        assert energy == pytest.approx(radial, rel=1e-6)

    def test_pbe_is_below_lda_for_an_inhomogeneous_density(self):
        grid = _grid()
        rho, _r2 = _gaussian(grid)
        assert (xc_grid.evaluate(grid, rho, "pbe").energy
                < xc_grid.evaluate(grid, rho, "lda").energy)

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_the_potential_is_the_functional_derivative(self, functional):
        r"""``dE/de = int v delta`` along a smooth perturbation of the density."""
        grid = _grid()
        rho, r2 = _gaussian(grid)
        tau = _tau(grid, rho, r2)
        delta = 0.05 * np.exp(-0.5 * r2) * (1.0 + 0.3 * grid.X.reshape(-1))
        eps = 1e-4

        def energy(density):
            return xc_grid.evaluate(grid, density, functional, tau=tau).energy

        potential = xc_grid.evaluate(grid, rho, functional, tau=tau).potential
        directional = np.sum(potential * delta) * grid.dV
        assert (energy(rho + eps * delta) - energy(rho - eps * delta)) \
            / (2.0 * eps) == pytest.approx(directional, rel=1e-5)

    def test_the_tau_potential_is_the_tau_derivative(self):
        grid = _grid()
        rho, r2 = _gaussian(grid)
        tau = _tau(grid, rho, r2)
        delta = 0.02 * rho * np.exp(-0.3 * r2)
        eps = 1e-4

        def energy(t):
            return xc_grid.evaluate(grid, rho, "r2scan", tau=t).energy

        terms = xc_grid.evaluate(grid, rho, "r2scan", tau=tau)
        directional = np.sum(terms.tau_potential * delta) * grid.dV
        assert (energy(tau + eps * delta) - energy(tau - eps * delta)) \
            / (2.0 * eps) == pytest.approx(directional, rel=1e-5)

    def test_a_meta_gga_needs_tau(self):
        grid = _grid(h=0.4)
        rho, _r2 = _gaussian(grid)
        with pytest.raises(ValueError, match="kinetic-energy density"):
            xc_grid.evaluate(grid, rho, "r2scan")


def _dataset(**fields):
    defaults = {"symbol": "X", "r": np.linspace(0.0, 10.0, 2001), "xc": "lda",
                "relativistic_exchange": False}
    return SimpleNamespace(**{**defaults, **fields})


class TestCoreDensities:
    def test_a_norm_conserving_dataset_uses_its_partial_core(self):
        r = np.linspace(0.0, 10.0, 2001)
        core = np.exp(-4.0 * r * r)
        dataset = _dataset(core_density=core, nlcc={"applied": True})
        assert xc_grid.xc_core_density(dataset) is core or np.array_equal(
            xc_grid.xc_core_density(dataset), core)

    def test_a_paw_dataset_uses_its_smooth_core_not_the_true_one(self):
        r = np.linspace(0.0, 10.0, 2001)
        dataset = _dataset(core_density=np.exp(-50.0 * r * r),
                           smooth_core_density=np.exp(-4.0 * r * r),
                           nlcc={"applied": True, "source": "smooth_core"})
        assert np.array_equal(xc_grid.xc_core_density(dataset),
                              dataset.smooth_core_density)

    def test_a_paw_core_re_pseudized_elsewhere_is_refused(self):
        r = np.linspace(0.0, 10.0, 2001)
        dataset = _dataset(core_density=np.exp(-50.0 * r * r),
                           smooth_core_density=np.exp(-4.0 * r * r),
                           nlcc={"applied": True, "r_nlcc": 1.2})
        with pytest.raises(NotImplementedError, match="does not store"):
            xc_grid.xc_core_density(dataset)

    def test_without_a_core_correction_there_is_nothing_to_add(self):
        dataset = _dataset(core_density=np.ones(2001), nlcc={"applied": False})
        assert xc_grid.xc_core_density(dataset) is None
        assert xc_grid.core_density_on_grid(_grid(h=0.4), [dataset],
                                            [np.zeros(3)]) is None
        assert xc_grid.core_correction_offset(dataset) == 0.0

    def test_the_core_lands_on_the_grid_at_its_atom(self):
        grid = _grid(h=0.4)
        r = np.linspace(0.0, 10.0, 2001)
        dataset = _dataset(core_density=np.exp(-r * r), nlcc={"applied": True})
        center = np.array([1.0, 0.0, 0.0])
        values = xc_grid.core_density_on_grid(grid, [dataset], [center])
        nearest = np.argmax(values)
        position = np.array([grid.X.reshape(-1)[nearest],
                             grid.Y.reshape(-1)[nearest],
                             grid.Z.reshape(-1)[nearest]])
        assert np.linalg.norm(position - center) < 0.4


class TestOptions:
    @pytest.mark.parametrize("spelling, canonical",
                             [("LDA", "lda"), ("pz81", "lda"), ("PBE", "pbe"),
                              ("gga-pbe", "pbe"), ("r2SCAN", "r2scan"),
                              ("r²SCAN", "r2scan")])
    def test_spellings_resolve(self, spelling, canonical):
        assert xc_grid.resolve_functional(spelling) == canonical

    def test_an_unknown_functional_is_refused(self):
        with pytest.raises(ValueError, match="unknown exchange-correlation"):
            xc_grid.resolve_functional("b3lyp")


class TestSpinPolarized:
    """`evaluate_spin`: the potentials of each channel on the grid."""

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    def test_equal_channels_give_the_unpolarized_terms(self, functional):
        grid = _grid()
        rho, r2 = _gaussian(grid)
        tau = _tau(grid, rho, r2)
        spin = xc_grid.evaluate_spin(grid, 0.5 * rho, 0.5 * rho, functional,
                                     tau_up=0.5 * tau, tau_dn=0.5 * tau)
        plain = xc_grid.evaluate(grid, rho, functional, tau=tau)
        assert spin.energy == pytest.approx(plain.energy, rel=1e-12)
        # A channel at rho/2 crosses the density floor before the total does:
        # below it (rho ~ 1e-12) the two conventions differ by v itself.
        live = rho > 1e-10
        for potential in (spin.potential_up, spin.potential_dn):
            assert np.allclose(potential[live], plain.potential[live],
                               rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
    @pytest.mark.parametrize("channel", ["up", "dn"])
    def test_each_potential_is_its_functional_derivative(self, functional,
                                                         channel):
        r"""``dE/de = int v_sigma delta`` perturbing one channel, on a
        density that is fully polarized away from the center (a spin-up tail
        with no spin-down)."""
        grid = _grid()
        rho, r2 = _gaussian(grid)
        up = rho
        dn = 0.5 * rho * np.exp(-0.8 * r2)
        tau_up, tau_dn = _tau(grid, up, r2), _tau(grid, dn, r2)
        # The perturbation lives where its channel does: added to an empty
        # channel, E_x ~ rho^(4/3) is not differentiable at zero and the
        # difference quotient picks up an eps^(1/3) term.
        shape = 1.0 + 0.3 * grid.Y.reshape(-1)
        delta = 0.1 * (up if channel == "up" else dn) * shape
        eps = 1e-4

        def energy(u, d):
            return xc_grid.evaluate_spin(grid, u, d, functional,
                                         tau_up=tau_up, tau_dn=tau_dn).energy

        terms = xc_grid.evaluate_spin(grid, up, dn, functional,
                                      tau_up=tau_up, tau_dn=tau_dn)
        if channel == "up":
            numeric = (energy(up + eps * delta, dn)
                       - energy(up - eps * delta, dn)) / (2 * eps)
            potential = terms.potential_up
        else:
            numeric = (energy(up, dn + eps * delta)
                       - energy(up, dn - eps * delta)) / (2 * eps)
            potential = terms.potential_dn
        assert numeric == pytest.approx(np.sum(potential * delta) * grid.dV,
                                        rel=1e-5)
