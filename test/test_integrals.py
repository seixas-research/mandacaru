# -*- coding: utf-8 -*-
# file: test_integrals.py

import numpy as np
import pytest

from carcara.basis import FullAtomicOrbital
from carcara.integrals import Grid, IntegralEngine, PoissonFFTSolver, Potentials
from carcara.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, HARTREE_TO_EV

# The physics core is validated in atomic units (Bohr, Hartree); the grids below
# therefore pass units="bohr" and the engine energy_units="Ha".


@pytest.fixture
def h1s_engine():
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=12.0, h=0.51, units="bohr")
    basis = [FullAtomicOrbital(1, 0, 0, Z=1.0, units="bohr")]
    return IntegralEngine(basis, grid)


class TestPoissonFFT:
    def test_1s_self_repulsion_matches_exact(self, h1s_engine):
        # (1s 1s | 1s 1s) = 5/8 Ha exactly for the hydrogen 1s orbital.
        J = h1s_engine.two_body(method="fft", energy_units="Ha")[0, 0, 0, 0].real
        assert abs(J - 0.625) < 0.02

    def test_converges_toward_exact_with_resolution(self):
        errs = []
        for h in (1.04, 0.51):  # coarse -> fine spacing (Bohr)
            grid = Grid(center=[0, 0, 0], box_size=12.0, h=h, units="bohr")
            eng = IntegralEngine([FullAtomicOrbital(1, 0, 0, Z=1.0, units="bohr")],
                                 grid)
            J = eng.two_body(method="fft", energy_units="Ha")[0, 0, 0, 0].real
            errs.append(abs(J - 0.625))
        assert errs[1] < errs[0]  # finer grid is closer to the exact value

    def test_eri_is_real_and_positive_diagonal(self, h1s_engine):
        J = h1s_engine.two_body(method="fft")[0, 0, 0, 0]
        assert abs(J.imag) < 1e-10
        assert J.real > 0

    def test_eight_fold_symmetry(self):
        grid = Grid(center=[0, 0, 0], box_size=8.0, h=0.84, units="bohr")
        basis = [FullAtomicOrbital(1, 0, 0, Z=1.0, units="bohr"),
                 FullAtomicOrbital(2, 1, 0, Z=1.0, units="bohr")]
        eri = IntegralEngine(basis, grid).two_body(method="fft")
        # Physicists' <ab|cd>: the full 8-fold symmetry of a real basis set.
        # The electron-1/2 bra-swaps <ab|cd>==<cb|ad>==<ad|cb> are the ones that
        # distinguish the physicists' convention from the chemists' (ab|cd).
        assert np.allclose(eri, np.transpose(eri, (2, 1, 0, 3)))   # e-1 bra swap
        assert np.allclose(eri, np.transpose(eri, (0, 3, 2, 1)))   # e-2 bra swap
        assert np.allclose(eri, np.transpose(eri, (2, 3, 0, 1)))   # <ab|cd>=<cd|ab>
        assert np.allclose(eri, np.conj(np.transpose(eri, (1, 0, 3, 2))))  # e- swap


class TestUnits:
    def test_engine_returns_eV_by_default(self, h1s_engine):
        # Default frontend unit is eV; explicit Ha recovers the atomic value.
        J_ev = h1s_engine.two_body(method="fft")[0, 0, 0, 0].real
        J_ha = h1s_engine.two_body(method="fft", energy_units="Ha")[0, 0, 0, 0].real
        assert abs(J_ev - 0.625 * HARTREE_TO_EV) < 0.02 * HARTREE_TO_EV
        assert np.isclose(J_ev, J_ha * HARTREE_TO_EV)

    def test_grid_angstrom_and_bohr_agree(self):
        # The same physical box, specified in Angstrom or Bohr, is identical.
        g_bohr = Grid(center=[0, 0, 0], box_size=10.0, h=0.5, units="bohr")
        g_ang = Grid(center=[0, 0, 0], box_size=10.0 * BOHR_TO_ANGSTROM,
                     h=0.5 * BOHR_TO_ANGSTROM, units="angstrom")
        assert g_ang.points == g_bohr.points
        assert np.isclose(g_ang.dx, g_bohr.dx)
        assert np.allclose(g_ang.X, g_bohr.X)

    def test_orbital_center_converts_to_bohr(self):
        orb = FullAtomicOrbital(1, 0, 0, center=[0.0, 0.0, 1.0])  # Angstrom
        assert np.allclose(orb.center, [0.0, 0.0, ANGSTROM_TO_BOHR])
        orb_b = FullAtomicOrbital(1, 0, 0, center=[0.0, 0.0, 1.0], units="bohr")
        assert np.allclose(orb_b.center, [0.0, 0.0, 1.0])

    def test_potentials_center_converts_to_bohr(self):
        pot = Potentials([(1.0, [0.0, 0.0, 1.0])])  # Angstrom
        assert np.allclose(pot.nuclei[0][1], [0.0, 0.0, ANGSTROM_TO_BOHR])


class TestSlater:
    @pytest.mark.parametrize("atomic_number, n, l, expected", [
        (1, 1, 0, 1.00),   # H 1s
        (2, 1, 0, 1.70),   # He 1s: 0.30 screening from the other 1s
        (3, 1, 0, 2.70),   # Li 1s
        (3, 2, 0, 1.30),   # Li 2s
        (3, 2, 1, 1.30),   # Li 2p (same sp group as 2s)
        (6, 2, 1, 3.25),   # C 2p
        (26, 4, 0, 3.75),  # Fe 4s
        (26, 3, 2, 6.25),  # Fe 3d
    ])
    def test_effective_charge_values(self, atomic_number, n, l, expected):
        Z = FullAtomicOrbital.slater_effective_charge(atomic_number, n, l)
        assert np.isclose(Z, expected)

    def test_from_slater_sets_Z(self):
        orb = FullAtomicOrbital.from_slater(2, 0, 0, atomic_number=3)
        assert np.isclose(orb.Z, 1.30)
        assert orb.state == (2, 0, 0)

    def test_unoccupied_orbital_raises(self):
        # He has no 2s in its ground state -> screening undefined.
        with pytest.raises(ValueError):
            FullAtomicOrbital.slater_effective_charge(2, 2, 0)

    def test_invalid_atomic_number_raises(self):
        with pytest.raises(ValueError):
            FullAtomicOrbital.slater_effective_charge(0, 1, 0)


class TestPotentials:
    def test_single_charge_is_minus_z_over_r(self):
        # V(r) = -Z/|r - R|; a unit charge at the origin gives -1 at 1 Bohr.
        pot = Potentials([(1.0, np.array([0.0, 0.0, 0.0]))], units="bohr")
        r = np.array([1.0, 2.0])
        v = pot.nuclear_potential(r, np.zeros_like(r), np.zeros_like(r))
        assert np.allclose(v, [-1.0, -0.5])

    def test_charges_superpose(self):
        # Two wells add: at the midpoint each contributes -Z/(R/2).
        R = 2.0
        pot = Potentials([(1.0, np.array([0.0, 0.0, -R / 2])),
                          (1.0, np.array([0.0, 0.0, +R / 2]))], units="bohr")
        v = pot.nuclear_potential(np.array([0.0]), np.array([0.0]),
                                  np.array([0.0]))
        assert np.allclose(v, [-2.0 / (R / 2)])

    def test_softening_bounds_the_singularity(self):
        # On top of the nucleus the potential is finite: -Z/softening.
        pot = Potentials([(1.0, np.array([0.0, 0.0, 0.0]))], softening=1e-3,
                         units="bohr")
        z = np.array([0.0])
        v = pot.nuclear_potential(z, z, z)
        assert np.isfinite(v).all()
        assert np.allclose(v, [-1.0 / 1e-3])

    def test_bound_method_drives_the_engine(self):
        # The method is directly consumable as the one_body potential callable.
        grid = Grid(center=[0, 0, 0], box_size=10.0, h=0.645, units="bohr")
        pot = Potentials([(1.0, np.array([0.0, 0.0, 0.0]))], units="bohr")
        eng = IntegralEngine([FullAtomicOrbital(1, 0, 0, Z=1.0, units="bohr")],
                             grid)
        T, V = eng.one_body(pot.nuclear_potential)
        assert V[0, 0].real < 0  # attractive well


class TestPoissonSolverDirect:
    def test_solver_matches_direct_convolution_on_tiny_grid(self):
        # On a tiny grid, compare the FFT solver to the explicit O(N^2) sum
        # with the SAME self-energy on the diagonal cell -> must agree.
        # h = 1.2 a0 over box 3.0 a0 keeps this at a tiny 6^3 grid.
        grid = Grid(center=[0, 0, 0], box_size=3.0, h=1.2, units="bohr")
        rng = np.random.default_rng(0)
        rho = (rng.standard_normal(grid.size)
               + 1j * rng.standard_normal(grid.size))

        solver = PoissonFFTSolver(grid.points, grid.dx)
        phi_fft = solver.solve(rho)

        xg, yg, zg = grid.flat_coords()
        n = grid.size
        phi_ref = np.zeros(n, dtype=complex)
        G0 = 2.3800756 / grid.dx
        for i in range(n):
            d = np.sqrt((xg[i] - xg) ** 2 + (yg[i] - yg) ** 2
                        + (zg[i] - zg) ** 2)
            kern = np.where(d > 0, np.divide(1.0, d, where=d > 0, out=None), G0)
            phi_ref[i] = np.sum(rho * kern) * grid.dV

        assert np.allclose(phi_fft, phi_ref, atol=1e-10)
