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

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan",
                                            "r2scan-rvv10"])
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
        dataset = _dataset(core_density=np.exp(-9.0 * r * r),
                           smooth_core_density=np.exp(-r * r),
                           nlcc={"applied": True, "source": "smooth_core"})
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
                              ("r²SCAN", "r2scan"),
                              ("r2SCAN+rVV10", "r2scan-rvv10"),
                              ("r2scan_rvv10", "r2scan-rvv10")])
    def test_spellings_resolve(self, spelling, canonical):
        assert xc_grid.resolve_functional(spelling) == canonical

    def test_an_unknown_functional_is_refused(self):
        with pytest.raises(ValueError, match="unknown exchange-correlation"):
            xc_grid.resolve_functional("b3lyp")


class TestSpinPolarized:
    """`evaluate_spin`: the potentials of each channel on the grid."""

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan",
                                            "r2scan-rvv10"])
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

    @pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan",
                                            "r2scan-rvv10"])
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


class TestHybrid:
    """The semilocal part of HSE06 (``screening=``)."""

    SCREENING = xc_grid.HYBRIDS["hse06"]

    def test_without_its_exact_exchange_it_is_refused(self):
        grid = _grid(h=0.4)
        rho, _r2 = _gaussian(grid)
        with pytest.raises(ValueError, match="hybrid"):
            xc_grid.evaluate(grid, rho, "hse06")
        with pytest.raises(ValueError, match="hybrid"):
            xc_grid.evaluate_spin(grid, 0.5 * rho, 0.5 * rho, "hse06")

    def test_screening_removes_part_of_the_exchange(self):
        """``fraction`` of the short-range exchange is taken out: the
        semilocal part rises, and by less than the full exchange would."""
        grid = _grid()
        rho, _r2 = _gaussian(grid)
        full = xc_grid.evaluate(grid, rho, "hse06",
                                screening=(0.11, 0.0)).energy
        hybrid = xc_grid.evaluate(grid, rho, "hse06",
                                  screening=self.SCREENING).energy
        pbe = xc_grid.evaluate(grid, rho, "pbe").energy
        assert full == pytest.approx(pbe, rel=5e-3)
        assert full < hybrid < full + 0.25 * abs(full)

    @pytest.mark.parametrize("exchange", ["same", "valence"])
    def test_the_potential_is_the_functional_derivative(self, exchange):
        r"""``dE/de = int v delta``, the short-range exchange taken on the
        whole density or on a valence part of it (a fixed core added to the
        rest, the way a partial core enters)."""
        grid = _grid()
        rho, r2 = _gaussian(grid)
        core = 0.4 * np.exp(-3.0 * r2) if exchange == "valence" else 0.0
        delta = 0.05 * np.exp(-0.5 * r2) * (1.0 + 0.3 * grid.X.reshape(-1))
        eps = 1e-4

        def terms(valence):
            return xc_grid.evaluate(
                grid, valence + core, "hse06", screening=self.SCREENING,
                exchange_density=valence if exchange == "valence" else None)

        directional = np.sum(terms(rho).potential * delta) * grid.dV
        assert (terms(rho + eps * delta).energy
                - terms(rho - eps * delta).energy) / (2.0 * eps) \
            == pytest.approx(directional, rel=1e-5)

    def test_equal_channels_give_the_unpolarized_terms(self):
        grid = _grid()
        rho, _r2 = _gaussian(grid)
        spin = xc_grid.evaluate_spin(grid, 0.5 * rho, 0.5 * rho, "hse06",
                                     screening=self.SCREENING)
        plain = xc_grid.evaluate(grid, rho, "hse06", screening=self.SCREENING)
        assert spin.energy == pytest.approx(plain.energy, rel=1e-10)
        live = rho > 1e-10
        for potential in (spin.potential_up, spin.potential_dn):
            assert np.allclose(potential[live], plain.potential[live],
                               rtol=1e-8, atol=1e-11)

    def test_each_spin_potential_is_its_functional_derivative(self):
        grid = _grid()
        rho, r2 = _gaussian(grid)
        up, dn = rho, 0.5 * rho * np.exp(-0.8 * r2)
        delta = 0.1 * dn * (1.0 + 0.3 * grid.Y.reshape(-1))
        eps = 1e-4

        def energy(d):
            return xc_grid.evaluate_spin(grid, up, d, "hse06",
                                         screening=self.SCREENING).energy

        potential = xc_grid.evaluate_spin(grid, up, dn, "hse06",
                                          screening=self.SCREENING
                                          ).potential_dn
        assert (energy(dn + eps * delta) - energy(dn - eps * delta)) \
            / (2 * eps) == pytest.approx(
                np.sum(potential * delta) * grid.dV, rel=1e-5)

    def test_hse_spellings_resolve(self):
        assert xc_grid.resolve_functional("HSE06") == "hse06"
        assert xc_grid.resolve_functional("hse") == "hse06"
        assert xc_grid.is_hybrid("hse06") and not xc_grid.is_hybrid("pbe")
        assert not xc_grid.takes_relativistic_exchange("hse06")


class TestCrystalSymmetry:
    """On a crystal's grid the spectral derivatives commute with every
    operation that maps the grid onto itself -- hexagonal ones included,
    which do not map the FFT box onto itself -- so a gradient-dependent
    functional is exactly as symmetric as LDA (HISTORY 2026-10-07, K13)."""

    #: Wurtzite AlN (Angstrom) on an even grid, so the c-glide is kept too.
    NODES = (12, 12, 20)

    @pytest.fixture(scope="class")
    def wurtzite(self):
        from ase.build import bulk

        from mandacaru.integrals import reciprocal as rc
        from mandacaru.pseudopotentials.periodic_paw import grid_symmetry

        atoms = bulk("AlN", "wurtzite", a=3.11, c=4.98, u=0.382)
        cell = atoms.cell.array
        grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0,
                    h=np.linalg.norm(cell, axis=1) / np.asarray(self.NODES),
                    units="angstrom", cell=cell, periodic=True)
        symmetry = grid_symmetry(atoms, grid)
        assert symmetry.n_operations == 12
        x = np.stack([grid.X.ravel(), grid.Y.ravel(), grid.Z.ravel()], axis=1)
        rho = np.zeros(len(x))
        bohr = 1.0 / 0.529177210903
        lattice = rc.lattice_vectors(grid)
        for position, alpha in zip(atoms.positions * bohr, (0.6, 0.6, 0.9, 0.9)):
            for R in rc.lattice_translations(lattice, 9.0):
                d = x - position - R
                rho += np.exp(-alpha * np.sum(d * d, axis=1))
        rho = 0.05 * rho / rho.max() + 1e-4
        # A density without the crystal's symmetry: its images differ.
        lumpy = rho * (1.0 + 0.3 * np.cos(0.7 * x[:, 0] + 0.2 * x[:, 2]))
        grad = xc_grid.gradient(grid, lumpy)
        tau = np.sum(grad * grad, axis=0) / (8.0 * lumpy) + 0.3 * lumpy
        return grid, symmetry, lumpy, tau

    @pytest.mark.parametrize("functional", ["pbe", "r2scan", "hse06",
                                            "r2scan-rvv10"])
    def test_an_image_has_the_same_energy_and_the_image_potential(
            self, wurtzite, functional):
        grid, symmetry, rho, tau = wurtzite
        screening = xc_grid.HYBRIDS.get(functional)

        def terms(perm):
            return xc_grid.evaluate(grid, rho[perm], functional, tau=tau[perm],
                                    screening=screening)

        identity = np.arange(rho.size)
        plain = terms(identity)
        scale = np.max(np.abs(plain.potential))
        for perm in symmetry.permutations:
            image = terms(perm)
            assert image.energy == pytest.approx(plain.energy, rel=1e-13)
            assert np.max(np.abs(image.potential - plain.potential[perm])) \
                < 1e-10 * scale

    def test_the_spin_channels_are_as_symmetric(self, wurtzite):
        grid, symmetry, rho, tau = wurtzite
        dn = 0.4 * rho[symmetry.permutations[3]]

        def energy(perm):
            return xc_grid.evaluate_spin(grid, rho[perm], dn[perm], "pbe"
                                         ).energy

        plain = energy(np.arange(rho.size))
        for perm in symmetry.permutations:
            assert energy(perm) == pytest.approx(plain, rel=1e-13)

    @pytest.mark.parametrize("functional", ["pbe", "r2scan"])
    def test_the_potential_is_the_functional_derivative(self, wurtzite,
                                                        functional):
        """The gradient and the divergence of the potential share the
        averaged wave-vectors, so ``v`` stays the exact derivative of the
        grid energy on a skewed, even grid."""
        grid, _symmetry, rho, tau = wurtzite
        delta = 0.2 * rho * np.cos(0.5 * grid.Y.reshape(-1))
        # r2SCAN's difference quotient carries 4e-7 of eps^2 at eps = 1e-4.
        eps = 1e-5

        def energy(density):
            return xc_grid.evaluate(grid, density, functional, tau=tau).energy

        potential = xc_grid.evaluate(grid, rho, functional, tau=tau).potential
        directional = np.sum(potential * delta) * grid.dV
        assert (energy(rho + eps * delta) - energy(rho - eps * delta)) \
            / (2.0 * eps) == pytest.approx(directional, rel=5e-8)

    def test_the_energy_is_smooth_in_a_shear(self, wurtzite):
        """A shear that keeps the lattice's operations moves the averaged
        wave-vectors linearly, so E_xc of fixed nodal values is as smooth
        in the strain as with the box: central differences at 1e-4 and
        1e-3 agree.  The shortest alias (Wigner-Seitz cell), as symmetric,
        switched aliases where the shear lifts a tie: the two slopes had
        opposite signs here."""
        from mandacaru.integrals import reciprocal as rc

        grid, _symmetry, rho, _tau = wurtzite
        cell = rc.lattice_vectors(grid).T * 0.529177210903
        shear = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                          [0.0, 0.0, 0.0]])

        def energy(eps):
            moved = cell @ (np.eye(3) + 0.5 * eps * shear).T
            strained = Grid(center=0.5 * moved.sum(axis=0), box_size=0.0,
                            h=np.linalg.norm(moved, axis=1)
                            / np.asarray(self.NODES),
                            units="angstrom", cell=moved, periodic=True)
            assert tuple(strained.shape) == self.NODES
            return xc_grid.evaluate(strained, rho, "pbe").energy

        slopes = [(energy(t) - energy(-t)) / (2.0 * t) for t in (1e-4, 1e-3)]
        assert abs(slopes[0]) > 1e-4
        assert slopes[1] == pytest.approx(slopes[0], rel=1e-5)

    def test_a_real_field_on_a_box_is_differentiated_as_before(self):
        """On an orthorhombic grid the averaged wave-vectors are the box's but
        for an even axis' Nyquist plane, whose derivative along that axis a
        real field never kept: a molecule's gradients are unchanged."""
        from mandacaru.integrals import reciprocal as rc

        grid = _grid(h=12.0 / 31.0)                         # 32 nodes
        assert all(n % 2 == 0 for n in grid.shape)
        rng = np.random.default_rng(3)
        field = rng.standard_normal(grid.size)
        transform = np.fft.fftn(field.reshape(grid.shape))
        box = np.stack([np.real(np.fft.ifftn(1j * k * transform)).ravel()
                        for k in rc.wavevectors(grid)])
        assert np.max(np.abs(xc_grid.gradient(grid, field) - box)) < 1e-12
