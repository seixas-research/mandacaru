# -*- coding: utf-8 -*-
# file: test/integrals/test_grid.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""The real-space grid: anisotropic boxes, resolution, and the ASE cell.

:class:`~mandacaru.integrals.Grid` accepts an anisotropic ``box_size`` or a
full ``cell`` tensor.  The grid a calculator uses is cut from the geometry's
own cell, which is also why where a molecule sits inside that cell must not
matter.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.integrals import Grid


class TestGridCarriesNoPrecision:
    """The precision is the constructor's of ``Mandacaru``; a grid is geometry."""

    def test_the_grid_has_no_precision_member(self):
        g = Grid(center=[0, 0, 0], box_size=4.0, grid_spacing=0.5, units="bohr")
        for name in ("precision", "real_dtype", "complex_dtype"):
            assert not hasattr(g, name)

    def test_a_precision_argument_is_refused(self):
        with pytest.raises(TypeError, match="precision"):
            Grid(center=[0, 0, 0], box_size=4.0, grid_spacing=0.5, units="bohr",
                 precision="float32")


class TestNonCubicGrid:
    def test_cubic_backward_compatible(self):
        g = Grid(center=[0, 0, 0], box_size=8.0, grid_spacing=0.20, units="bohr")
        assert g.is_cubic
        assert g.shape == (g.points, g.points, g.points)
        assert g.size == g.points ** 3

    def test_anisotropic_box_is_non_cubic(self):
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], grid_spacing=0.20, units="bohr")
        assert not g.is_cubic
        nx, ny, nz = g.shape
        assert nx != ny and ny != nz
        # Uniform spacing across axes is required by the finite-difference core.
        assert g.X.shape == (nx, ny, nz)

    def test_uniform_spacing_on_non_cubic(self):
        g = Grid(center=[0, 0, 0], box_size=[8.0, 9.0, 7.0], grid_spacing=0.20, units="bohr")
        dx = np.diff(g.X[:, 0, 0])
        dy = np.diff(g.Y[0, :, 0])
        dz = np.diff(g.Z[0, 0, :])
        np.testing.assert_allclose(dx, g.dx)
        np.testing.assert_allclose(dy, g.dx)
        np.testing.assert_allclose(dz, g.dx)

    def test_cell_tensor_orthorhombic(self):
        cell = np.diag([16.0, 18.0, 14.0])
        g = Grid(center=[0, 0, 0], box_size=0.0, grid_spacing=0.20, units="bohr", cell=cell)
        assert not g.is_cubic
        assert g.shape[0] < g.shape[1]     # 16 < 18 -> fewer x nodes than y

    def test_cell_tensor_non_orthogonal_bounding_box(self):
        # A skewed (non-orthogonal) cell is still enclosed by the grid.
        cell = np.array([[16.0, 0.0, 0.0],
                         [2.0, 18.0, 0.0],
                         [0.0, 0.0, 14.0]])
        g = Grid(center=[0, 0, 0], box_size=0.0, grid_spacing=0.25, units="bohr", cell=cell)
        assert g.size > 0
        assert g.dx == pytest.approx(0.25)




class TestVaryingResolution:
    """Per-axis (varying) resolution and non-orthogonal cells (C-backend req)."""

    def test_per_axis_resolution_grid(self):
        # A length-3 grid_spacing sets a different spacing on each axis.
        g = Grid(center=[0, 0, 0], box_size=8.0, grid_spacing=[0.20, 0.25, 0.30],
                 units="bohr")
        assert (g.dx, g.dy, g.dz) == pytest.approx((0.20, 0.25, 0.30))
        assert g.is_orthorhombic and not g.is_cubic
        assert g.dV == pytest.approx(0.20 * 0.25 * 0.30)
        # Inverse metric is diagonal 1/d^2 for an orthorhombic grid.
        np.testing.assert_allclose(
            g.metric_inverse(),
            np.diag([1 / 0.20 ** 2, 1 / 0.25 ** 2, 1 / 0.30 ** 2]), atol=1e-12)


class TestPlacementInvariance:
    def test_energy_independent_of_position_in_cell(self):
        # The grid is cut from the cell and centered on the molecule, so placing
        # H2 at the cell corner vs. the cell center gives the same Hamiltonian
        # and energy.  (A periodic molecular adapt-vqe run is no longer offered:
        # periodic geometries are native DFT only.)
        def energy(pos):
            atoms = Atoms("H2", positions=pos,
                          cell=[[8, 0, 0], [0, 8, 0], [0, 0, 8]], pbc=False)
            atoms.calc = Mandacaru(method="adapt-vqe", pool="ceo", basis="HAO",
                                   grid_spacing=0.35, trace=False, max_iterations=6,
                                   convergence={"gradient": 1e-3})
            return atoms.get_total_energy()

        corner = energy([[0, 0, -0.37], [0, 0, 0.37]])
        center = energy([[4, 4, 3.63], [4, 4, 4.37]])
        assert corner == pytest.approx(center, abs=1e-6)
