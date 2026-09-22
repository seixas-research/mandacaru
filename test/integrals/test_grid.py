# -*- coding: utf-8 -*-
# file: test/integrals/test_grid.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The real-space grid: anisotropic boxes, resolution, and the ASE cell.

:class:`~mandacaru.integrals.Grid` accepts an anisotropic ``box_size`` or a
full ``cell`` tensor, and the integral engine must still recover the reference
physics (H 1s on-site repulsion = 5/8 Ha) on a non-cubic box.  The grid a
calculator uses is cut from the geometry's own cell
(:func:`~mandacaru.algorithms._hamiltonian_from_atoms.grid_from_cell`), which
is also why where a molecule sits inside that cell must not matter.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import grid_from_cell
from mandacaru.basis import HydrogenicAtomicOrbital
from mandacaru.integrals import Grid, IntegralEngine


def _eri_00(grid):
    """H 1s on-site repulsion <00|00> on ``grid`` (Hartree)."""
    orb = HydrogenicAtomicOrbital(1, 0, 0, Z=1.0, center=[0.0, 0.0, 0.0],
                                  units="bohr")
    eng = IntegralEngine([orb], grid)
    return float(np.real(eng.two_body(method="fft", energy_units="Ha")[0, 0, 0, 0]))


class TestNonCubicGrid:
    def test_cubic_backward_compatible(self):
        g = Grid(center=[0, 0, 0], box_size=8.0, h=0.20, units="bohr")
        assert g.is_cubic
        assert g.shape == (g.points, g.points, g.points)
        assert g.size == g.points ** 3

    def test_anisotropic_box_is_non_cubic(self):
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        assert not g.is_cubic
        nx, ny, nz = g.shape
        assert nx != ny and ny != nz
        # Uniform spacing across axes is required by the finite-difference core.
        assert g.X.shape == (nx, ny, nz)

    def test_uniform_spacing_on_non_cubic(self):
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        dx = np.diff(g.X[:, 0, 0])
        dy = np.diff(g.Y[0, :, 0])
        dz = np.diff(g.Z[0, 0, :])
        np.testing.assert_allclose(dx, g.dx)
        np.testing.assert_allclose(dy, g.dx)
        np.testing.assert_allclose(dz, g.dx)

    def test_cell_tensor_orthorhombic(self):
        cell = np.diag([16.0, 18.0, 14.0])
        g = Grid(center=[0, 0, 0], box_size=0.0, h=0.20, units="bohr", cell=cell)
        assert not g.is_cubic
        assert g.shape[0] < g.shape[1]     # 16 < 18 -> fewer x nodes than y

    def test_cell_tensor_non_orthogonal_bounding_box(self):
        # A skewed (non-orthogonal) cell is still enclosed by the grid.
        cell = np.array([[16.0, 0.0, 0.0],
                         [2.0, 18.0, 0.0],
                         [0.0, 0.0, 14.0]])
        g = Grid(center=[0, 0, 0], box_size=0.0, h=0.25, units="bohr", cell=cell)
        assert g.size > 0
        assert g.dx == pytest.approx(0.25)

    def test_eri_matches_cubic_reference(self):
        # The H 1s self-repulsion must be 5/8 Ha independent of box shape.
        cubic = _eri_00(Grid(center=[0, 0, 0], box_size=8.0, h=0.20, units="bohr"))
        rect = _eri_00(Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0],
                            h=0.20, units="bohr"))
        cell = _eri_00(Grid(center=[0, 0, 0], box_size=0.0, h=0.20, units="bohr",
                            cell=np.diag([16.0, 18.0, 14.0])))
        assert cubic == pytest.approx(0.625, abs=2e-3)
        assert rect == pytest.approx(0.625, abs=2e-3)
        assert cell == pytest.approx(0.625, abs=2e-3)
        assert rect == pytest.approx(cubic, abs=1e-4)

    def test_kinetic_energy_non_cubic(self):
        # <1s|-1/2 nabla^2|1s> = 1/2 Ha for hydrogen, on a non-cubic grid.
        orb = HydrogenicAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], h=0.20, units="bohr")
        eng = IntegralEngine([orb], g)
        zero_v = lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape)
        T, _ = eng.one_body(zero_v, energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=1e-2)


class TestVaryingResolution:
    """Per-axis (varying) resolution and non-orthogonal cells (C-backend req)."""

    def test_per_axis_resolution_grid(self):
        # A length-3 h sets a different spacing on each axis.
        g = Grid(center=[0, 0, 0], box_size=8.0, h=[0.20, 0.25, 0.30],
                 units="bohr")
        assert (g.dx, g.dy, g.dz) == pytest.approx((0.20, 0.25, 0.30))
        assert g.is_orthorhombic and not g.is_cubic
        assert g.dV == pytest.approx(0.20 * 0.25 * 0.30)
        # Inverse metric is diagonal 1/d^2 for an orthorhombic grid.
        np.testing.assert_allclose(
            g.metric_inverse(),
            np.diag([1 / 0.20 ** 2, 1 / 0.25 ** 2, 1 / 0.30 ** 2]), atol=1e-12)

    def test_kinetic_energy_varying_resolution(self):
        # The generalized (per-axis) Laplacian still recovers <1s|T|1s> = 1/2 Ha.
        orb = HydrogenicAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        g = Grid(center=[0, 0, 0], box_size=8.0, h=[0.20, 0.25, 0.30],
                 units="bohr")
        T, _ = IntegralEngine([orb], g).one_body(
            lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape),
            energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=2e-2)

    def test_skewed_non_orthogonal_grid(self):
        # skew=True samples the actual (non-orthogonal) lattice; the step matrix
        # is non-diagonal and dV = |det(step)|.
        cell = np.array([[16.0, 0.0, 0.0],
                         [3.0, 16.0, 0.0],
                         [0.0, 1.0, 16.0]])
        g = Grid(center=[0, 0, 0], cell=cell, h=0.4, units="bohr", skew=True)
        assert not g.is_orthorhombic
        assert g.dV == pytest.approx(abs(np.linalg.det(g.step)))
        # Cross-term Laplacian recovers the hydrogen kinetic energy.
        orb = HydrogenicAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0], units="bohr")
        T, _ = IntegralEngine([orb], g).one_body(
            lambda x, y, z: np.zeros(np.broadcast(x, y, z).shape),
            energy_units="Ha")
        assert float(np.real(T[0, 0])) == pytest.approx(0.5, abs=3e-2)

    def test_c_and_numpy_kernels_agree(self):
        # The C general kernel and the NumPy fallback must stay in lockstep on
        # anisotropic and skewed grids.
        import mandacaru.integrals._backend as backend
        if not backend.HAS_C_BACKEND:
            pytest.skip("C backend not built")
        orb = HydrogenicAtomicOrbital(1, 0, 0, Z=1.0, center=[0, 0, 0.2], units="bohr")
        for g in (Grid(center=[0, 0, 0], box_size=6.0, h=[0.25, 0.30, 0.35],
                       units="bohr"),
                  Grid(center=[0, 0, 0],
                       cell=np.array([[12., 0, 0], [3., 12., 0], [0, 1., 12.]]),
                       h=0.4, units="bohr", skew=True)):
            psi = np.stack([orb.sample(g)])
            vext = np.zeros(g.size)
            saved = backend.HAS_C_BACKEND
            try:
                backend.HAS_C_BACKEND = True
                Tc, _ = backend.one_body_matrices(psi, vext, g)
                backend.HAS_C_BACKEND = False
                Tn, _ = backend.one_body_matrices(psi, vext, g)
            finally:
                backend.HAS_C_BACKEND = saved
            np.testing.assert_allclose(Tc, Tn, atol=1e-10)


class TestGridFromCell:
    def test_grid_spans_the_cell_and_is_non_cubic(self):
        cell = [[10.0, 0, 0], [0, 12.0, 0], [0, 0, 8.0]]  # Angstrom, orthorhombic
        atoms = Atoms("H", positions=[[5, 6, 4]], cell=cell, pbc=True)
        grid = grid_from_cell(atoms, h=0.25)
        assert not grid.is_cubic          # 10 != 12 != 8
        # h is Angstrom; the grid spacing is that value converted to Bohr.
        assert grid.dx == pytest.approx(0.25 * 1.8897259886)

    def test_grid_requires_a_cell(self):
        atoms = Atoms("H", positions=[[0, 0, 0]])          # no unit cell
        with pytest.raises(ValueError, match="no unit cell"):
            grid_from_cell(atoms, h=0.2)


class TestPlacementInvariance:
    def test_energy_independent_of_position_in_cell(self):
        # PBC-aware grid: the molecule is centered on itself, so placing H2 at the
        # cell corner vs. the cell center gives the same Hamiltonian and energy.
        def energy(pos):
            atoms = Atoms("H2", positions=pos,
                          cell=[[8, 0, 0], [0, 8, 0], [0, 0, 8]], pbc=True)
            atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                                   h=0.35, trace=False, max_iterations=6,
                                   gradient_tolerance=1e-3)
            return atoms.get_total_energy()

        corner = energy([[0, 0, -0.37], [0, 0, 0.37]])
        center = energy([[4, 4, 3.63], [4, 4, 4.37]])
        assert corner == pytest.approx(center, abs=1e-6)
