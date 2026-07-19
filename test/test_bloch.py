# -*- coding: utf-8 -*-
# file: test/test_bloch.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Bloch / k-point variational drivers -- band structure and total energy.

Covers the shared :class:`BlochVariationalDriver` base and its three concrete
crystal drivers (:class:`BlochVQE`, :class:`BlochADAPTVQE`, :class:`BlochVASQE`):
the single-particle Bloch Hamiltonian / band structure (solver-independent, on the
base), and the total energy using all k-points via the Born-von Karman supercell
equivalence (per molecular solver).
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import (
    ADAPTVQEResult,
    BandStructure,
    BlochADAPTVQE,
    BlochVASQE,
    BlochVQE,
    BlochVariationalDriver,
    VASQEResult,
    VQEResult,
)
from carcara.optimizers import Optimizer

BLOCH_DRIVERS = [BlochVQE, BlochADAPTVQE, BlochVASQE]


def _chain(cls, **kwargs):
    """1-D hydrogen chain, one atom per cell (fast, coarse settings)."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                  pbc=[True, False, False])
    kwargs.setdefault("basis", "FAO")
    kwargs.setdefault("mapping", "jordan_wigner")
    kwargs.setdefault("n_cells", 3)
    kwargs.setdefault("n_images", 5)
    kwargs.setdefault("h", 0.30)
    return cls(atoms, **kwargs)


@pytest.fixture(scope="module")
def band_driver():
    # The band structure is solver-independent, so any concrete driver serves.
    return _chain(BlochVQE)


# total_energy kwargs differ per calculator (VQE has no adaptive controls).
_TOTAL_ENERGY_KWARGS = {
    BlochVQE: dict(h=0.40, optimizer=Optimizer("L-BFGS-B", maxiter=2000)),
    BlochADAPTVQE: dict(h=0.40, optimizer=Optimizer("L-BFGS-B", maxiter=2000),
                        max_iterations=6, gradient_tolerance=1e-3),
    BlochVASQE: dict(h=0.40, optimizer=Optimizer("L-BFGS-B", maxiter=2000),
                     temperature=1.0, max_iterations=6, gradient_tolerance=1e-3),
}
_RESULT_TYPES = {BlochVQE: VQEResult, BlochADAPTVQE: ADAPTVQEResult,
                 BlochVASQE: VASQEResult}


# --------------------------------------------------------------------------- #
# Construction / geometry.
# --------------------------------------------------------------------------- #
class TestConstruction:
    @pytest.mark.parametrize("cls", BLOCH_DRIVERS)
    def test_dimension_and_bands(self, cls):
        d = _chain(cls)
        assert d.dimension == 1
        assert d.n_bands == 1                       # one FAO 1s per cell

    @pytest.mark.parametrize("cls", BLOCH_DRIVERS)
    def test_requires_a_periodic_direction(self, cls):
        molecule = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                         cell=[10, 10, 10], pbc=False)
        with pytest.raises(ValueError):
            cls(molecule)

    def test_n_images_at_least_n_cells(self):
        atoms = Atoms("H", positions=[[0, 0, 0]],
                      cell=[[1.0, 0, 0], [0, 10, 0], [0, 0, 10]],
                      pbc=[True, False, False])
        b = BlochVQE(atoms, n_cells=6, n_images=2)
        assert b.n_images >= b.n_cells


# --------------------------------------------------------------------------- #
# Abstract base.
# --------------------------------------------------------------------------- #
class TestAbstractBase:
    def test_base_total_energy_is_abstract(self):
        driver = _chain(BlochVariationalDriver)
        with pytest.raises(NotImplementedError):
            driver.total_energy((2, 1, 1))

    def test_base_still_does_bands(self):
        # The band structure lives on the base and needs no calculator.
        driver = _chain(BlochVariationalDriver)
        assert driver.bands([0.0, 0, 0]).shape == (1, 1)


# --------------------------------------------------------------------------- #
# Bloch Hamiltonian and bands (solver-independent).
# --------------------------------------------------------------------------- #
class TestBands:
    def test_bloch_matrices_hermitian(self, band_driver):
        H, S = band_driver.bloch_hamiltonian([0.3, 0.0, 0.0])
        assert np.allclose(H, H.conj().T)
        assert np.allclose(S, S.conj().T)

    def test_band_time_reversal_symmetry(self, band_driver):
        assert np.isclose(band_driver.bands([0.27, 0, 0])[0, 0],
                          band_driver.bands([-0.27, 0, 0])[0, 0])

    def test_bonding_minimum_at_gamma(self, band_driver):
        # negative hopping -> the band bottom is the bonding state at Gamma.
        assert (band_driver.bands([0, 0, 0])[0, 0]
                < band_driver.bands([0.5, 0, 0])[0, 0])

    def test_bands_shape(self, band_driver):
        kpts = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
        assert band_driver.bands(kpts).shape == (3, band_driver.n_bands)

    def test_band_structure_path(self, band_driver):
        bs = band_driver.band_structure("GX", npoints=10)
        assert isinstance(bs, BandStructure)
        assert bs.energies.shape == (10, band_driver.n_bands)
        assert bs.x.shape == (10,)
        assert "G" in bs.labels and "X" in bs.labels

    def test_monkhorst_pack_gamma_centred(self, band_driver):
        mesh = band_driver.monkhorst_pack((10, 1, 1), gamma=True)
        assert mesh.shape == (10, 3)
        assert np.isclose(np.abs(mesh[:, 0]).min(), 0.0)   # Gamma is on the mesh

    def test_bands_identical_across_drivers(self):
        # Bands are single-particle: all three drivers must agree exactly.
        kpts = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
        ref = _chain(BlochVQE).bands(kpts)
        for cls in (BlochADAPTVQE, BlochVASQE):
            np.testing.assert_allclose(_chain(cls).bands(kpts), ref)


# --------------------------------------------------------------------------- #
# Generality: a 2-D crystal.
# --------------------------------------------------------------------------- #
class TestTwoDimensional:
    @pytest.fixture(scope="class")
    def square(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 10.0]],
                      pbc=[True, True, False])
        return BlochVQE(atoms, basis="FAO", n_cells=2, n_images=3, h=0.40)

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
    def test_supercell_size(self, band_driver):
        sc = band_driver.supercell((4, 1, 1))
        assert len(sc) == 4                                # 4 cells -> 4 atoms

    @pytest.mark.parametrize("cls", BLOCH_DRIVERS)
    def test_total_energy_lowers_and_per_cell(self, cls):
        driver = _chain(cls)
        e_cell, result = driver.total_energy((2, 1, 1), **_TOTAL_ENERGY_KWARGS[cls])
        assert isinstance(result, _RESULT_TYPES[cls])
        assert e_cell < 0.0
        # the solver lowered the energy below the Hartree-Fock reference.
        assert result.optimal_energy < result.reference_energy

    def test_all_drivers_agree_on_total_energy(self):
        # Same Hamiltonian, different solvers -> same ground-state energy per cell.
        energies = []
        for cls in BLOCH_DRIVERS:
            e_cell, _ = _chain(cls).total_energy((2, 1, 1),
                                                 **_TOTAL_ENERGY_KWARGS[cls])
            energies.append(e_cell)
        assert max(energies) - min(energies) < 1e-3        # eV
