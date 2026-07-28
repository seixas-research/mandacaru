# -*- coding: utf-8 -*-
# file: test/test_pes.py

"""Potential-energy-curve helpers (examples/pes_utils.py): grids, references.

The physical contract pinned here is the one the dissociation examples rely on:
every curve is referenced to the **sum of isolated-atom UHF energies** computed
on the same grid with the same Coulomb softening, so ``E = 0`` is the
separated-atom limit and a molecule near equilibrium is *bound* (negative)
against it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "examples"))

pes_utils = pytest.importorskip("pes_utils")
from pes_utils import (GridSpec, atom_energy, atomic_reference,  # noqa: E402
                       commensurate_distances, cusp_softening,
                       molecule_positions, rhf_total_energy)
from carcara.basis import BasisSet  # noqa: E402
from carcara.integrals import Grid  # noqa: E402


class TestCommensurateDistances:
    def test_step_is_twice_the_spacing(self):
        d = commensurate_distances(1.0, 3.5, GridSpec(box_size=10.0, spacing=0.16))
        assert d[0] == pytest.approx(1.0)
        assert np.allclose(np.diff(d), 2 * 0.16)
        assert d[-1] <= 3.5 + 1e-9

    def test_single_point_when_range_below_step(self):
        d = commensurate_distances(1.0, 1.1, GridSpec(box_size=8.0, spacing=0.2))
        assert len(d) == 1


class TestGridSpec:
    def test_build_matches_the_spec(self):
        from carcara.units import BOHR_TO_ANGSTROM

        grid = GridSpec(box_size=6.0, spacing=0.22).build()
        assert isinstance(grid, Grid)
        # Grid steps are stored in Bohr (the numerical core's units).
        assert grid.dx * BOHR_TO_ANGSTROM == pytest.approx(0.22, abs=0.02)

    def test_softening_is_half_a_grid_step(self):
        grid = GridSpec(box_size=6.0, spacing=0.22).build()
        assert cusp_softening(grid) == pytest.approx(
            0.5 * min(grid.dx, grid.dy, grid.dz))


@pytest.fixture(scope="module")
def grid():
    return Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.22)


@pytest.fixture(scope="module")
def sto3g():
    return BasisSet.build("GTO", n_gaussians=3)


class TestEnergies:
    def test_h2_rhf_energy_is_bound(self, grid, sto3g):
        pos = molecule_positions(0.74)
        e = rhf_total_energy(["H", "H"], pos, sto3g, grid)
        # H2 near equilibrium is well below two isolated H atoms (2 * -0.5).
        assert -1.3 < e < -1.0

    def test_isolated_hydrogen_atom(self, grid, sto3g):
        e = atom_energy("H", sto3g, grid, position=[0.03, 0.02, -0.37])
        assert e == pytest.approx(-0.5, abs=0.06)

    def test_h2_binds_against_the_atomic_reference(self, grid, sto3g):
        """The contract of the dissociation examples: E(molecule) < E(atoms)."""
        pos = molecule_positions(0.74)
        e_mol = rhf_total_energy(["H", "H"], pos, sto3g, grid)
        e_atoms = atomic_reference(["H", "H"], sto3g, grid, pos)
        binding = e_mol - e_atoms
        # Bound by a chemically sensible amount (experimental D_e ~ -0.17 Ha).
        assert -0.35 < binding < -0.05

    def test_atomic_reference_is_the_sum_of_atoms(self, grid, sto3g):
        pos = molecule_positions(1.0)
        total = atomic_reference(["H", "H"], sto3g, grid, pos)
        parts = sum(atom_energy("H", sto3g, grid, p) for p in pos)
        assert total == pytest.approx(parts, abs=1e-12)
