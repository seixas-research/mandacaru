# -*- coding: utf-8 -*-
# file: test/test_bloch.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""BlochADAPTVQE -- k-point band structure and total energy for crystals.

Covers the general (1-/2-/3-D) crystal driver: the Bloch Hamiltonian / band
structure solved per k-point, and the total energy using all k-points via the
Born-von Karman supercell equivalence.
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import BandStructure, BlochADAPTVQE


@pytest.fixture(scope="module")
def h_chain():
    """1-D hydrogen chain, one atom per cell (fast, coarse settings)."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                  pbc=[True, False, False])
    return BlochADAPTVQE(atoms, basis="FAO", mapping="jordan_wigner",
                         n_cells=3, n_images=5, h=0.30)


# --------------------------------------------------------------------------- #
# Construction / geometry.
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_dimension_and_bands(self, h_chain):
        assert h_chain.dimension == 1
        assert h_chain.n_bands == 1                 # one FAO 1s per cell

    def test_requires_a_periodic_direction(self):
        molecule = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                         cell=[10, 10, 10], pbc=False)
        with pytest.raises(ValueError):
            BlochADAPTVQE(molecule)

    def test_n_images_at_least_n_cells(self):
        atoms = Atoms("H", positions=[[0, 0, 0]],
                      cell=[[1.0, 0, 0], [0, 10, 0], [0, 0, 10]],
                      pbc=[True, False, False])
        b = BlochADAPTVQE(atoms, n_cells=6, n_images=2)
        assert b.n_images >= b.n_cells


# --------------------------------------------------------------------------- #
# Bloch Hamiltonian and bands.
# --------------------------------------------------------------------------- #
class TestBands:
    def test_bloch_matrices_hermitian(self, h_chain):
        H, S = h_chain.bloch_hamiltonian([0.3, 0.0, 0.0])
        assert np.allclose(H, H.conj().T)
        assert np.allclose(S, S.conj().T)

    def test_band_time_reversal_symmetry(self, h_chain):
        assert np.isclose(h_chain.bands([0.27, 0, 0])[0, 0],
                          h_chain.bands([-0.27, 0, 0])[0, 0])

    def test_bonding_minimum_at_gamma(self, h_chain):
        # negative hopping -> the band bottom is the bonding state at Gamma.
        assert h_chain.bands([0, 0, 0])[0, 0] < h_chain.bands([0.5, 0, 0])[0, 0]

    def test_bands_shape(self, h_chain):
        kpts = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
        assert h_chain.bands(kpts).shape == (3, h_chain.n_bands)

    def test_band_structure_path(self, h_chain):
        bs = h_chain.band_structure("GX", npoints=10)
        assert isinstance(bs, BandStructure)
        assert bs.energies.shape == (10, h_chain.n_bands)
        assert bs.x.shape == (10,)
        assert "G" in bs.labels and "X" in bs.labels

    def test_monkhorst_pack_gamma_centred(self, h_chain):
        mesh = h_chain.monkhorst_pack((10, 1, 1), gamma=True)
        assert mesh.shape == (10, 3)
        assert np.isclose(np.abs(mesh[:, 0]).min(), 0.0)   # Gamma is on the mesh


# --------------------------------------------------------------------------- #
# Generality: a 2-D crystal.
# --------------------------------------------------------------------------- #
class TestTwoDimensional:
    @pytest.fixture(scope="class")
    def square(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 10.0]],
                      pbc=[True, True, False])
        return BlochADAPTVQE(atoms, basis="FAO", n_cells=2, n_images=3, h=0.40)

    def test_dimension(self, square):
        assert square.dimension == 2

    def test_bands_disperse_and_symmetric(self, square):
        gamma = square.bands([0.0, 0.0, 0.0])[0, 0]
        corner = square.bands([0.5, 0.5, 0.0])[0, 0]
        assert gamma < corner                              # dispersion
        assert np.isclose(square.bands([0.3, 0.2, 0])[0, 0],
                          square.bands([-0.3, -0.2, 0])[0, 0])


# --------------------------------------------------------------------------- #
# Total energy using all k-points (Born-von Karman supercell).
# --------------------------------------------------------------------------- #
class TestTotalEnergy:
    def test_supercell_size(self, h_chain):
        sc = h_chain.supercell((4, 1, 1))
        assert len(sc) == 4                                # 4 cells -> 4 atoms

    def test_total_energy_lowers_and_per_cell(self, h_chain):
        e_cell, result = h_chain.total_energy((2, 1, 1), h=0.40,
                                              max_iterations=6,
                                              gradient_tolerance=1e-3)
        assert e_cell < 0.0
        assert result.optimal_energy < result.reference_energy   # ADAPT lowered E
