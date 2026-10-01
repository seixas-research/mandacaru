# -*- coding: utf-8 -*-
# file: test/pseudopotentials/test_periodic_paw.py

# This code is part of Mandacaru.
# MIT License

"""PAW-LCAO in a crystal: Bloch sums, k-point reduction, lattice energies."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mandacaru.algorithms.periodic_dft import PeriodicKohnSham
from mandacaru.integrals import exchange_correlation as xc_grid
from mandacaru.integrals import reciprocal as rc
from mandacaru.pseudopotentials import periodic_paw as pp

#: Diamond silicon on a grid commensurate with both the primitive cell and
#: its 2x1x1 supercell (10 nodes per primitive lattice vector).
SILICON = bulk("Si", "diamond", a=5.43)
SILICON_H = float(np.linalg.norm(SILICON.cell[0])) / 10
SILICON_BASIS = {"size": "SZ", "filter": 200}


def _energy(atoms, h, options, kpts=None):
    """Extrapolated energy (Hartree per cell) of a periodic ``atoms``."""
    crystal, context = pp.build_crystal(atoms, h, options, kpts=kpts)
    constant = (sum(d.one_center_energy for d in crystal.datasets)
                + sum(xc_grid.core_correction_offset(d)
                      for d in crystal.datasets))
    result = PeriodicKohnSham(crystal, context["n_electrons"], "lda",
                              smearing={"method": "fermi-dirac",
                                        "width": 0.01},
                              constant=constant).run()
    assert result.converged
    return result


class TestKPoints:
    def test_a_gamma_centered_even_mesh_is_its_own_time_reverse(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        _size, _gamma, mesh = monkhorst_pack_kpts({"size": (2, 2, 2),
                                                   "gamma": True})
        reduced, weights = pp.time_reversal_reduce(mesh)
        assert len(reduced) == 8 and np.allclose(weights, 1 / 8)

    def test_an_odd_mesh_pairs_k_with_minus_k(self):
        from mandacaru.algorithms._hamiltonian_from_atoms import \
            monkhorst_pack_kpts
        _size, _gamma, mesh = monkhorst_pack_kpts((3, 3, 3))
        reduced, weights = pp.time_reversal_reduce(mesh)
        assert len(reduced) == 14
        assert weights.sum() == pytest.approx(1.0)
        gamma = np.argmin(np.linalg.norm(reduced, axis=1))
        assert np.allclose(reduced[gamma], 0.0)
        assert weights[gamma] == pytest.approx(1 / 27)       # its own partner


class TestBlochSums:
    def test_a_lattice_translation_multiplies_by_the_phase(self):
        crystal, _context = pp.build_crystal(SILICON, SILICON_H,
                                             SILICON_BASIS)
        A = crystal.lattice
        k = rc.reciprocal_vectors(A) @ np.array([0.25, -0.5, 0.125])
        points = (np.array([0.3, 1.1]), np.array([0.7, -0.4]),
                  np.array([1.9, 0.2]))
        shifted = tuple(points[i] + A[i, 0] for i in range(3))
        center = np.zeros(3)
        here = pp.bloch_values(crystal.basis, A, k[None], points, center, 12.0)
        there = pp.bloch_values(crystal.basis, A, k[None], shifted, center,
                                12.0)
        assert np.allclose(there, np.exp(1j * k @ A[:, 0]) * here, atol=1e-12)

    def test_a_support_is_where_the_table_vanishes_not_where_it_ends(self):
        crystal, _context = pp.build_crystal(SILICON, SILICON_H,
                                             SILICON_BASIS)
        for function in crystal.basis:
            assert pp._support(function) < 20.0


class TestEnergies:
    def test_a_mesh_equals_its_supercell(self):
        """Born-von Karman: a 2x1x1 mesh is the 2x1x1 supercell at Gamma.

        Every term -- Bloch phases, projections, short-range spheres,
        lattice electrostatics -- has to agree for this to hold per cell.
        """
        mesh = _energy(SILICON, SILICON_H, SILICON_BASIS,
                       kpts={"size": (2, 1, 1), "gamma": True})
        supercell = _energy(SILICON.repeat((2, 1, 1)), SILICON_H,
                            SILICON_BASIS)
        assert mesh.extrapolated_energy == pytest.approx(
            supercell.extrapolated_energy / 2, abs=1e-7)

    def test_a_molecule_in_a_box_is_the_molecular_limit(self):
        """H2 at Gamma in an 8 Angstrom cell: the converged molecular energy.

        The molecular PAW-LCAO-DZP LDA energy, with the spectral kinetic
        operator, extrapolates as h^2 from h = 0.25 ... 0.10 Angstrom to
        -1.1269 Ha; the periodic, Fourier-filtered one gives it at h = 0.25
        (HISTORY.md, 2026-10-01).
        """
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                      cell=[8.0] * 3, pbc=True)
        atoms.center()
        result = _energy(atoms, 0.25, {"size": "DZP"})
        assert result.extrapolated_energy == pytest.approx(-1.1269, abs=5e-4)
