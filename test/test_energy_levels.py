# -*- coding: utf-8 -*-
# file: test_energy_levels.py

"""Molecular energy levels (ground + excited states) via variational deflation.

Both :class:`VQE` and :class:`ADAPTVQE` expose ``energy_levels``, which computes
the low-lying spectrum with variational quantum deflation (VQD).  On H2 (MO
basis) every level it returns must coincide with a true eigenvalue of the qubit
Hamiltonian; the ground level must match exact diagonalization.
"""

import numpy as np
import pytest

from carcara.algorithms import ADAPTVQE, VQE, EnergyLevels
from carcara.algorithms._energy_levels import spectral_width_beta
from carcara.circuits import UCCSD
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid
from carcara.optimizers import Optimizer


# --------------------------------------------------------------------------- #
# Shared H2 fixtures (MO basis) -- mirrors test_adapt_vqe.py.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def h2_hamiltonian():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.25)
    mints = MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid)
    return mints.molecular_hamiltonian(mo_basis=True, n_electrons=2)


@pytest.fixture(scope="module")
def h2_spectrum(h2_hamiltonian):
    m = h2_hamiltonian.map_to_qubits("jordan_wigner").to_matrix()
    return np.sort(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).real)


def _is_eigenvalue(energy, spectrum, tol=1e-5):
    return float(np.min(np.abs(spectrum - energy))) < tol


# --------------------------------------------------------------------------- #
# VQE.energy_levels
# --------------------------------------------------------------------------- #

class TestVQEEnergyLevels:
    def _vqe(self, h2_hamiltonian):
        ansatz = UCCSD(2, (1, 1), mapping="jordan_wigner")
        return VQE(h2_hamiltonian, ansatz,
                   optimizer=Optimizer("L-BFGS-B", maxiter=2000), verbose=False)

    def test_ground_level_matches_exact(self, h2_hamiltonian, h2_spectrum):
        levels = self._vqe(h2_hamiltonian).energy_levels(1)
        assert levels.num_states == 1
        assert levels.ground_state_energy == pytest.approx(h2_spectrum[0], abs=1e-6)

    def test_levels_are_true_eigenvalues(self, h2_hamiltonian, h2_spectrum):
        levels = self._vqe(h2_hamiltonian).energy_levels(2, restarts=4)
        for e in levels.energies:
            assert _is_eigenvalue(e, h2_spectrum)

    def test_levels_ascending_and_distinct(self, h2_hamiltonian):
        levels = self._vqe(h2_hamiltonian).energy_levels(2, restarts=4)
        assert np.all(np.diff(levels.energies) > 1e-6)

    def test_excitation_energies(self, h2_hamiltonian):
        levels = self._vqe(h2_hamiltonian).energy_levels(2, restarts=4)
        assert levels.excitation_energies[0] == pytest.approx(0.0, abs=1e-12)
        assert levels.excitation_energies[1] > 0.0
        # eV view is the Hartree gap times the conversion factor.
        ev = levels.excitation_energies_in_units("eV")
        assert ev[1] == pytest.approx(levels.excitation_energies[1] * 27.211386,
                                      rel=1e-4)

    def test_states_stored_and_orthogonal(self, h2_hamiltonian):
        levels = self._vqe(h2_hamiltonian).energy_levels(2, restarts=4)
        assert len(levels.states) == 2
        overlap = abs(np.vdot(levels.states[0], levels.states[1]))
        assert overlap < 1e-4        # deflation makes the levels orthogonal


# --------------------------------------------------------------------------- #
# ADAPTVQE.energy_levels
# --------------------------------------------------------------------------- #

class TestADAPTEnergyLevels:
    def _adapt(self, h2_hamiltonian):
        return ADAPTVQE(h2_hamiltonian, "fermionic", num_particles=(1, 1),
                        n_spatial_orbitals=2,
                        optimizer=Optimizer("L-BFGS-B", maxiter=2000),
                        verbose=False, profile=False, gradient_tolerance=1e-6)

    def test_ground_level_matches_exact(self, h2_hamiltonian, h2_spectrum):
        levels = self._adapt(h2_hamiltonian).energy_levels(1)
        assert levels.ground_state_energy == pytest.approx(h2_spectrum[0], abs=1e-6)

    def test_levels_are_true_eigenvalues(self, h2_hamiltonian, h2_spectrum):
        levels = self._adapt(h2_hamiltonian).energy_levels(2)
        for e in levels.energies:
            assert _is_eigenvalue(e, h2_spectrum)

    def test_excited_state_lifts_above_ground(self, h2_hamiltonian, h2_spectrum):
        levels = self._adapt(h2_hamiltonian).energy_levels(2)
        assert levels.energies[1] > levels.energies[0] + 1e-6
        # ADAPT records how many operators were grown for each level.
        assert levels.num_operators is not None
        assert len(levels.num_operators) == 2

    def test_reference_energy_populated(self, h2_hamiltonian):
        levels = self._adapt(h2_hamiltonian).energy_levels(1)
        assert levels.reference_energy is not None


# --------------------------------------------------------------------------- #
# Shared helpers / result container.
# --------------------------------------------------------------------------- #

class TestEnergyLevelsHelpers:
    def test_num_states_validated(self, h2_hamiltonian):
        ansatz = UCCSD(2, (1, 1), mapping="jordan_wigner")
        vqe = VQE(h2_hamiltonian, ansatz, verbose=False)
        with pytest.raises(ValueError):
            vqe.energy_levels(0)

    def test_requires_configuration(self):
        vqe = VQE(basis="FAO", verbose=False)          # calculator mode, unconfigured
        with pytest.raises(RuntimeError):
            vqe.energy_levels(2)

    def test_spectral_width_beta_positive(self, h2_hamiltonian):
        qh = h2_hamiltonian.map_to_qubits("jordan_wigner")
        assert spectral_width_beta(qh) > 0.0

    def test_container_views(self):
        lv = EnergyLevels(energies=np.array([-1.0, -0.5, 0.25]))
        assert lv.num_states == 3
        assert lv.ground_state_energy == -1.0
        np.testing.assert_allclose(lv.excitation_energies, [0.0, 0.5, 1.25])
        np.testing.assert_allclose(lv.gaps, [0.5, 0.75])
        np.testing.assert_allclose(lv.in_units("Ha"), [-1.0, -0.5, 0.25])
