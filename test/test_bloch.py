# -*- coding: utf-8 -*-
# file: test/test_bloch.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Bloch / k-point variational calculator -- band structure and total energy.

Covers :class:`BlochCalculator` for every variational ``method`` (``"vqe"``,
``"adapt-vqe"``, ``"vasqe"``): the single-particle Bloch Hamiltonian / band
structure (method-independent), and the total energy using all k-points via the
Born-von Karman supercell equivalence (per method).
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import (
    ADAPTVQEResult,
    BandStructure,
    BlochCalculator,
    VASQEResult,
    VQEResult,
)
from carcara.optimizers import Optimizer

METHODS = ["vqe", "adapt-vqe", "vasqe"]


def _chain(method="vqe", **kwargs):
    """1-D hydrogen chain, one atom per cell (fast, coarse settings)."""
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                  cell=[[1.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                  pbc=[True, False, False])
    kwargs.setdefault("basis", "FAO")
    kwargs.setdefault("mapping", "jordan_wigner")
    kwargs.setdefault("n_cells", 3)
    kwargs.setdefault("n_images", 5)
    kwargs.setdefault("h", 0.30)
    return BlochCalculator(atoms, method=method, **kwargs)


@pytest.fixture(scope="module")
def band_driver():
    # The band structure is method-independent, so any method serves.
    return _chain("vqe")


# total_energy kwargs differ per method (VQE has no adaptive controls).
_TOTAL_ENERGY_KWARGS = {
    "vqe": dict(h=0.40, optimizer=Optimizer("L-BFGS-B", maxiter=2000)),
    "adapt-vqe": dict(h=0.40, optimizer=Optimizer("L-BFGS-B", maxiter=2000),
                      max_iterations=6, gradient_tolerance=1e-3),
    "vasqe": dict(h=0.40, optimizer=Optimizer("L-BFGS-B", maxiter=2000),
                  temperature=1.0, max_iterations=6, gradient_tolerance=1e-3),
}
_RESULT_TYPES = {"vqe": VQEResult, "adapt-vqe": ADAPTVQEResult,
                 "vasqe": VASQEResult}


# --------------------------------------------------------------------------- #
# Construction / geometry.
# --------------------------------------------------------------------------- #
class TestConstruction:
    @pytest.mark.parametrize("method", METHODS)
    def test_dimension_and_bands(self, method):
        d = _chain(method)
        assert d.dimension == 1
        assert d.n_bands == 1                       # one FAO 1s per cell

    @pytest.mark.parametrize("method", METHODS)
    def test_requires_a_periodic_direction(self, method):
        molecule = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                         cell=[10, 10, 10], pbc=False)
        with pytest.raises(ValueError):
            BlochCalculator(molecule, method=method)

    def test_unknown_method_rejected(self):
        atoms = Atoms("H", positions=[[0, 0, 0]],
                      cell=[[1.0, 0, 0], [0, 10, 0], [0, 0, 10]],
                      pbc=[True, False, False])
        with pytest.raises(ValueError, match="unknown method"):
            BlochCalculator(atoms, method="qaoa")

    def test_n_images_at_least_n_cells(self):
        atoms = Atoms("H", positions=[[0, 0, 0]],
                      cell=[[1.0, 0, 0], [0, 10, 0], [0, 0, 10]],
                      pbc=[True, False, False])
        b = BlochCalculator(atoms, n_cells=6, n_images=2)
        assert b.n_images >= b.n_cells


# --------------------------------------------------------------------------- #
# Bloch Hamiltonian and bands (method-independent).
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

    def test_monkhorst_pack_gamma_centered(self, band_driver):
        mesh = band_driver.monkhorst_pack((10, 1, 1), gamma=True)
        assert mesh.shape == (10, 3)
        assert np.isclose(np.abs(mesh[:, 0]).min(), 0.0)   # Gamma is on the mesh

    def test_bands_identical_across_methods(self):
        # Bands are single-particle: every method must agree exactly.
        kpts = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
        ref = _chain("vqe").bands(kpts)
        for method in ("adapt-vqe", "vasqe"):
            np.testing.assert_allclose(_chain(method).bands(kpts), ref)


# --------------------------------------------------------------------------- #
# Generality: a 2-D periodic system.
# --------------------------------------------------------------------------- #
class TestTwoDimensional:
    @pytest.fixture(scope="class")
    def square(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 10.0]],
                      pbc=[True, True, False])
        return BlochCalculator(atoms, basis="FAO", n_cells=2, n_images=3, h=0.40)

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

    @pytest.mark.parametrize("method", METHODS)
    def test_total_energy_lowers_and_per_cell(self, method):
        driver = _chain(method)
        e_cell, result = driver.total_energy((2, 1, 1),
                                             **_TOTAL_ENERGY_KWARGS[method])
        assert isinstance(result, _RESULT_TYPES[method])
        assert e_cell < 0.0
        # the solver lowered the energy below the Hartree-Fock reference.
        assert result.optimal_energy < result.reference_energy

    def test_all_methods_agree_on_total_energy(self):
        # Same Hamiltonian, different solvers -> same ground-state energy per cell.
        energies = []
        for method in METHODS:
            e_cell, _ = _chain(method).total_energy(
                (2, 1, 1), **_TOTAL_ENERGY_KWARGS[method])
            energies.append(e_cell)
        assert max(energies) - min(energies) < 1e-3        # eV
