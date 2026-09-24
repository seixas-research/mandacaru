# -*- coding: utf-8 -*-
# file: test/integrals/test_slab_kernel.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The two-dimensionally truncated Coulomb kernel, for slabs with a dipole.

A three-dimensional kernel makes a slab interact with an infinite stack of
copies of itself.  For a neutral non-polar slab that error decays with the
vacuum and can be converged away; for a slab carrying a **dipole** it cannot,
because two dipole sheets a distance ``L`` apart have an interaction energy per
area that does not vanish as ``L`` grows.  The property this kernel exists to
give is therefore an energy that does not depend on the vacuum thickness at all,
and that is what ``TestTheDipoleArtifactIsGone`` measures.

The kernel's algebra is checked two independent ways: the two branches
(``G_par != 0`` and ``G_par = 0``) must be one function, which pins the
prefactors; and an exact real-space reference built from the sheet potential
``-2 pi |dz|`` -- sharing no code with the kernel -- must converge to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandacaru.integrals.poisson import (KERNEL_TRUNCATIONS,
                                         PeriodicPoissonSolver,
                                         slab_occupancy,
                                         slab_truncated_kernel)


# --------------------------------------------------------------------------- #
# Helpers: a dipolar slab, and an independent exact energy for it.
# --------------------------------------------------------------------------- #

def dipolar_slab(nz: int, length: float, side: float = 6.0, nxy: int = 4,
                 separation: float = 2.0, width: float = 0.45):
    """A neutral pair of in-plane-uniform Gaussian sheets, centered in the cell.

    Returns ``(shape, cell, area, profile, rho)`` with ``profile`` the
    plane-averaged charge density along the axis and ``rho`` the flat grid array.
    """
    shape = (nxy, nxy, nz)
    cell = np.diag([side, side, length])
    area = side * side
    z = np.arange(nz) * (length / nz)
    profile = np.zeros(nz)
    for center, charge in ((length / 2 - separation / 2, +1.0),
                           (length / 2 + separation / 2, -1.0)):
        profile += charge * np.exp(-0.5 * ((z - center) / width) ** 2) \
            / (width * np.sqrt(2.0 * np.pi))
    density = profile / area
    rho = np.repeat(density[None, None, :], nxy, 0).repeat(nxy, 1)
    return shape, cell, area, density, np.ascontiguousarray(rho.ravel())


def exact_sheet_energy(density, length: float, area: float) -> float:
    r"""``E = 1/2 int rho Phi`` for an in-plane-uniform slab, isolated in ``z``.

    Only ``G_par = 0`` contributes, and there a uniform plane at ``z'`` produces
    the potential :math:`-2\pi|z-z'|`.  Evaluated by direct quadrature: no FFT,
    no period, no kernel -- which is what makes it an independent reference.
    """
    nz = density.size
    dz = length / nz
    z = np.arange(nz) * dz
    kernel = -2.0 * np.pi * np.abs(z[:, None] - z[None, :])
    potential = kernel @ density * dz
    return 0.5 * float(np.sum(density * potential)) * dz * area


def hartree_energy(shape, cell, rho, **kwargs) -> float:
    """``1/2 int rho Phi dV`` through the FFT solver."""
    step = cell @ np.diag(1.0 / np.array(shape, dtype=float))
    solver = PeriodicPoissonSolver(shape, step=step, **kwargs)
    potential = np.real(solver.solve(rho))
    return 0.5 * float(np.sum(rho * potential)) * abs(np.linalg.det(step))


# --------------------------------------------------------------------------- #
# The kernel itself.
# --------------------------------------------------------------------------- #

class TestTheKernel:
    def test_it_is_finite_everywhere(self):
        kernel = slab_truncated_kernel((4, 4, 32), np.diag([6.0, 6.0, 24.0]), 2)
        assert np.all(np.isfinite(kernel))

    def test_the_g_zero_mode_is_the_neutrality_convention(self):
        kernel = slab_truncated_kernel((4, 4, 32), np.diag([6.0, 6.0, 24.0]), 2)
        assert kernel[0, 0, 0] == 0.0

    def test_the_in_plane_zero_axial_nonzero_modes_survive(self):
        # These are exactly where a dipole lives.  Dropping them along with
        # G = 0 would discard the effect the kernel exists to produce, so this
        # is the assertion that the implementation kept them.
        kernel = slab_truncated_kernel((4, 4, 32), np.diag([6.0, 6.0, 24.0]), 2)
        assert np.any(np.abs(kernel[0, 0, 1:]) > 0.0)

    def test_it_is_even_in_the_axial_wavevector(self):
        # Which is what lets |G_z| be used: the two Nyquist aliases are the same
        # mode and must give the same kernel.
        kernel = slab_truncated_kernel((4, 4, 32), np.diag([6.0, 6.0, 24.0]), 2)
        assert np.allclose(kernel[:, :, 1:16], kernel[:, :, 31:16:-1])

    def test_the_two_branches_are_one_function(self):
        r"""The check that pins the prefactors.

        Subtracting the sheet-charge pole :math:`4\pi\sin(G_zL/2)/(G_zG_\parallel)`
        from the ``G_par != 0`` branch and letting :math:`G_\parallel \to 0` must
        give the ``G_par = 0`` branch.  Evaluated here as a numerical limit, with
        both branches written out independently of the module.
        """
        length, g_z = 24.0, 2.0 * np.pi * 3 / 24.0
        phase = g_z * length / 2.0

        def branch_general(g_par):
            g2 = g_par ** 2 + g_z ** 2
            return (4.0 * np.pi / g2) * (
                1.0 + np.exp(-g_par * length / 2.0)
                * ((g_z / g_par) * np.sin(phase) - np.cos(phase)))

        def pole(g_par):
            return 4.0 * np.pi * np.sin(phase) / (g_z * g_par)

        branch_sheet = 2.0 * np.pi * (2.0 - 2.0 * np.cos(phase)
                                      - g_z * length * np.sin(phase)) / g_z ** 2
        residuals = [abs(branch_general(g) - pole(g) - branch_sheet)
                     for g in (1e-2, 1e-3, 1e-4)]
        # What to assert is the *rate*.  The remainder after removing the pole is
        # O(G_par), so each tenfold reduction in G_par must take a factor of ten
        # off it -- measured 2.31, 0.243, 0.0244, ratios 0.105 and 0.100.  A
        # wrong prefactor in either branch leaves a constant difference instead,
        # which would plateau however small G_par got; no tolerance is needed to
        # tell those two apart.  (An absolute bound would be the wrong test: the
        # kernel itself is ~40 here, so a converging residual is not a small one.)
        ratios = [residuals[i + 1] / residuals[i] for i in range(2)]
        assert all(0.08 < r < 0.13 for r in ratios), (residuals, ratios)
        assert residuals[-1] < 1e-3 * abs(branch_sheet), residuals

    def test_the_sheet_branch_matches_the_module(self):
        # The same closed form, against what the module actually built.
        length, nz = 24.0, 32
        kernel = slab_truncated_kernel((4, 4, nz), np.diag([6.0, 6.0, length]),
                                       2)
        for k in (1, 2, 3, 5):
            g_z = 2.0 * np.pi * k / length
            phase = g_z * length / 2.0
            expected = 2.0 * np.pi * (2.0 - 2.0 * np.cos(phase)
                                      - g_z * length * np.sin(phase)) / g_z ** 2
            assert kernel[0, 0, k] == pytest.approx(expected, rel=1e-12)

    def test_a_tilted_vacuum_axis_is_refused(self):
        # The kernel splits G into in-plane and axial parts, which exists only
        # for a perpendicular axis; a sheared one would be silently wrong.
        cell = np.diag([6.0, 6.0, 24.0]).astype(float)
        cell[0, 2] = 1.0                       # tilt the vacuum vector
        with pytest.raises(ValueError, match="perpendicular to the periodic"):
            slab_truncated_kernel((4, 4, 32), cell, 2)

    def test_a_hexagonal_plane_is_accepted(self):
        # Only the *vacuum* axis has to be perpendicular; the plane may be
        # sheared, which is the common case for a hexagonal surface.
        cell = np.array([[6.0, -3.0, 0.0],
                         [0.0, 5.196, 0.0],
                         [0.0, 0.0, 24.0]])
        kernel = slab_truncated_kernel((6, 6, 32), cell, 2)
        assert np.all(np.isfinite(kernel))

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_any_axis_may_be_the_vacuum_direction(self, axis):
        lengths = [6.0, 6.0, 6.0]
        lengths[axis] = 24.0
        shape = [4, 4, 4]
        shape[axis] = 32
        kernel = slab_truncated_kernel(tuple(shape), np.diag(lengths), axis)
        assert np.all(np.isfinite(kernel))
        assert kernel[0, 0, 0] == 0.0

    def test_a_bad_axis_is_refused(self):
        with pytest.raises(ValueError, match="axis must be 0, 1 or 2"):
            slab_truncated_kernel((4, 4, 8), np.diag([6.0, 6.0, 24.0]), 3)


# --------------------------------------------------------------------------- #
# The solver option.
# --------------------------------------------------------------------------- #

class TestTheSolverOption:
    def test_the_default_is_the_fully_periodic_kernel(self):
        solver = PeriodicPoissonSolver((4, 4, 8), spacing=0.5)
        assert solver.truncation == "none"
        assert solver.axis is None

    def test_the_untruncated_kernel_is_unchanged(self):
        # The feature must not move a single existing periodic number.
        from mandacaru.integrals.poisson import fft_g_squared

        solver = PeriodicPoissonSolver((6, 6, 10), spacing=0.4)
        g_squared = fft_g_squared(solver.shape, solver.cell)
        expected = np.where(g_squared > 0.0, 4.0 * np.pi / np.where(
            g_squared > 0.0, g_squared, 1.0), 0.0)
        assert np.allclose(solver._kernel, expected)

    def test_slab_needs_an_axis(self):
        with pytest.raises(ValueError, match="needs axis="):
            PeriodicPoissonSolver((4, 4, 8), spacing=0.5, truncation="slab")

    def test_an_axis_without_slab_is_refused(self):
        # It would be silently ignored otherwise, which reads as a truncated run.
        with pytest.raises(ValueError, match="means nothing with truncation"):
            PeriodicPoissonSolver((4, 4, 8), spacing=0.5, axis=2)

    def test_an_unknown_truncation_is_refused(self):
        with pytest.raises(ValueError, match="unknown truncation"):
            PeriodicPoissonSolver((4, 4, 8), spacing=0.5, truncation="2d")

    def test_every_name_in_the_registry_builds(self):
        for name in KERNEL_TRUNCATIONS:
            axis = 2 if name == "slab" else None
            solver = PeriodicPoissonSolver((4, 4, 16), spacing=0.5,
                                           truncation=name, axis=axis)
            assert np.all(np.isfinite(solver._kernel))


# --------------------------------------------------------------------------- #
# The physics: the dipole artifact.
# --------------------------------------------------------------------------- #

class TestTheDipoleArtifactIsGone:
    """The reason the kernel exists, stated as a measurement."""

    THICKNESSES = ((16.0, 128), (24.0, 192), (32.0, 256), (48.0, 384))

    def test_the_slab_energy_does_not_depend_on_the_vacuum(self):
        energies = [hartree_energy(*dipolar_slab(nz, length)[:2],
                                   dipolar_slab(nz, length)[4],
                                   truncation="slab", axis=2)
                    for length, nz in self.THICKNESSES]
        spread = max(energies) - min(energies)
        assert spread < 1e-9, energies

    def test_the_periodic_kernel_does_depend_on_it(self):
        # The control.  Without this the test above could pass for a kernel that
        # simply returned a constant.
        energies = [hartree_energy(*dipolar_slab(nz, length)[:2],
                                   dipolar_slab(nz, length)[4])
                    for length, nz in self.THICKNESSES]
        spread = max(energies) - min(energies)
        assert spread > 1e-2, energies
        # And it is still moving at the widest vacuum, i.e. not converged.
        assert abs(energies[-1] - energies[-2]) > 1e-3, energies

    def test_the_two_agree_on_a_reference_the_kernel_never_sees(self):
        # An exact real-space sheet sum converges to the truncated-kernel
        # energy at second order in the spacing, while the truncated-kernel
        # energy does not move at all -- it is already the band-limited answer.
        length = 24.0
        differences = []
        for nz in (96, 192, 384):
            shape, cell, area, density, rho = dipolar_slab(nz, length)
            truncated = hartree_energy(shape, cell, rho, truncation="slab",
                                       axis=2)
            reference = exact_sheet_energy(density, length, area)
            differences.append(abs(truncated - reference))
        ratios = [differences[i + 1] / differences[i] for i in range(2)]
        assert all(0.2 < r < 0.3 for r in ratios), (differences, ratios)

    def test_the_truncated_energy_is_independent_of_the_spacing(self):
        length = 24.0
        energies = [hartree_energy(*dipolar_slab(nz, length)[:2],
                                   dipolar_slab(nz, length)[4],
                                   truncation="slab", axis=2)
                    for nz in (96, 192, 384)]
        assert max(energies) - min(energies) < 1e-9, energies


# --------------------------------------------------------------------------- #
# The confinement the energy relies on.
# --------------------------------------------------------------------------- #

class TestTheOccupancyMeasure:
    def test_a_confined_slab_occupies_little_of_the_axis(self):
        shape, _cell, _area, _density, rho = dipolar_slab(256, 48.0)
        assert slab_occupancy(rho, shape, 2) < 0.3

    def test_a_thin_cell_leaves_the_density_over_half_the_axis(self):
        # The regime where the truncation starts cutting real interactions, and
        # the number a caller is meant to check before trusting the energy.
        shape, _cell, _area, _density, rho = dipolar_slab(64, 8.0)
        assert slab_occupancy(rho, shape, 2) > 0.5

    def test_an_empty_density_occupies_nothing(self):
        shape = (4, 4, 16)
        assert slab_occupancy(np.zeros(4 * 4 * 16), shape, 2) == 0.0

    def test_it_measures_the_wrapped_extent(self):
        # A slab centered on the cell boundary is contiguous through the wrap,
        # and reporting it as filling the whole axis would be wrong.
        shape = (2, 2, 32)
        profile = np.zeros(32)
        profile[[30, 31, 0, 1]] = 1.0
        rho = np.repeat(profile[None, None, :], 2, 0).repeat(2, 1).ravel()
        assert slab_occupancy(rho, shape, 2) == pytest.approx(4 / 32)


# --------------------------------------------------------------------------- #
# What it refuses to be combined with.
# --------------------------------------------------------------------------- #

class TestTheHamiltonianRefusesItForNow:
    def test_the_periodic_hamiltonian_says_why(self):
        # Truncating only the Hartree term would leave the electron-ion and
        # ion-ion Ewald sums three-dimensional: a total energy assembled from
        # two boundary conditions, every term plausible and the sum meaningless.
        from mandacaru.core.hamiltonian import minimal_hao_basis
        from mandacaru.core.periodic import PeriodicIntegrals
        from mandacaru.integrals import Grid

        cell = np.diag([6.0, 6.0, 12.0])
        nuclei = [(1.0, np.array([3.0, 3.0, 6.0]))]
        grid = Grid(center=[3, 3, 6], box_size=0.0, h=0.6, units="bohr",
                    cell=cell, periodic=True)
        with pytest.raises(NotImplementedError, match="one boundary condition"):
            PeriodicIntegrals(nuclei, minimal_hao_basis(nuclei, "bohr"), grid,
                              cell, units="bohr", truncation="slab")
