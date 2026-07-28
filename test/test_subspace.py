# -*- coding: utf-8 -*-
# file: test_subspace.py

"""Subspace-search VQE / ADAPT-VQE: ground + excited states in one optimization.

SSVQE sends several orthogonal reference determinants through one shared unitary
and minimizes a weighted energy sum.  The rigorous, ansatz-independent properties
checked here are the Hylleraas-Undheim-MacDonald upper bounds (the sorted trial
energies bound the exact eigenvalues from above) and the orthonormality of the
returned states.  Validated on H2 (MO basis).
"""

import numpy as np
import pytest

from carcara.algorithms import (
    SubspaceADAPTVQE,
    SubspaceADAPTVQEResult,
    SubspaceVQE,
    SubspaceVQEResult,
)
from carcara.algorithms.subspace import (
    reference_matrix,
    resolve_weights,
    subspace_determinants,
)
from carcara.circuits import UCCSD
from carcara.core import MolecularIntegrals, minimal_fao_basis
from carcara.integrals import Grid
from carcara.optimizers import Optimizer


# --------------------------------------------------------------------------- #
# Shared H2 fixtures (MO basis).
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


def _ssvqe(h2_hamiltonian, k, **kwargs):
    ansatz = UCCSD(2, (1, 1), mapping="jordan_wigner")
    return SubspaceVQE(h2_hamiltonian, ansatz, num_states=k,
                       optimizer=Optimizer("L-BFGS-B", maxiter=4000),
                       verbose=False, **kwargs)


def _ss_adapt(h2_hamiltonian, k, **kwargs):
    return SubspaceADAPTVQE(h2_hamiltonian, "fermionic", num_states=k,
                            num_particles=(1, 1), n_spatial_orbitals=2,
                            optimizer=Optimizer("L-BFGS-B", maxiter=4000),
                            verbose=False, profile=False,
                            gradient_tolerance=1e-6, max_iterations=30, **kwargs)


# --------------------------------------------------------------------------- #
# Reference determinants and weights.
# --------------------------------------------------------------------------- #

class TestReferences:
    def test_determinants_start_at_hartree_fock(self):
        dets = subspace_determinants(4, [0, 2], 4)
        assert dets[0] == (0, 2)                       # HF determinant first
        assert len(dets) == 4                          # full Sz=0 sector
        assert len(set(dets)) == 4                     # all distinct

    def test_reference_matrix_orthonormal(self):
        refs = reference_matrix("jordan_wigner", 4, [0, 2], 3)
        assert refs.shape == (16, 3)
        gram = refs.conj().T @ refs
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-12)

    def test_too_many_states_raises(self):
        with pytest.raises(ValueError):
            subspace_determinants(4, [0, 2], 99)

    def test_default_weights_strictly_decreasing(self):
        w = resolve_weights(None, 3)
        np.testing.assert_allclose(w, [3.0, 2.0, 1.0])

    @pytest.mark.parametrize("bad", [[1, 1, 1], [1, 2, 3], [1, -1, -2], [1, 2]])
    def test_bad_weights_rejected(self, bad):
        with pytest.raises(ValueError):
            resolve_weights(bad, 3)


# --------------------------------------------------------------------------- #
# SubspaceVQE
# --------------------------------------------------------------------------- #

class TestSubspaceVQE:
    def test_single_state_recovers_ground(self, h2_hamiltonian, h2_spectrum):
        r = _ssvqe(h2_hamiltonian, 1).run()
        assert r.num_states == 1
        assert r.optimal_energy == pytest.approx(h2_spectrum[0], abs=1e-6)

    def test_hylleraas_undheim_upper_bounds(self, h2_hamiltonian, h2_spectrum):
        # The sorted SSVQE energies bound the exact eigenvalues from above.
        r = _ssvqe(h2_hamiltonian, 3).run()
        for i, e in enumerate(r.energies):
            assert e >= h2_spectrum[i] - 1e-6

    def test_ground_is_exact_from_default_start(self, h2_hamiltonian, h2_spectrum):
        r = _ssvqe(h2_hamiltonian, 2).run()          # init = zeros (near HF)
        assert r.energies[0] == pytest.approx(h2_spectrum[0], abs=1e-5)

    def test_levels_ascending(self, h2_hamiltonian):
        r = _ssvqe(h2_hamiltonian, 3).run()
        assert np.all(np.diff(r.energies) >= -1e-9)

    def test_states_orthonormal(self, h2_hamiltonian):
        r = _ssvqe(h2_hamiltonian, 3).run()
        gram = np.array([[np.vdot(a, b) for b in r.states] for a in r.states])
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-8)

    def test_weights_used_and_stored(self, h2_hamiltonian):
        r = _ssvqe(h2_hamiltonian, 2, weights=[5.0, 1.0]).run()
        np.testing.assert_allclose(r.weights, [5.0, 1.0])

    def test_result_views(self, h2_hamiltonian):
        r = _ssvqe(h2_hamiltonian, 2).run()
        assert r.excitation_energies[0] == pytest.approx(0.0, abs=1e-12)
        ev = r.in_units("eV")
        assert ev[0] == pytest.approx(r.energies[0] * 27.211386, rel=1e-4)
        assert r.levels.num_states == 2               # EnergyLevels view

    def test_is_ase_calculator(self, h2_spectrum):
        from ase import Atoms
        atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
                      cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)
        atoms.calc = SubspaceVQE(basis="FAO", h=0.30, num_states=2, verbose=False)
        energy_ev = atoms.get_potential_energy()
        result = atoms.calc.result
        assert isinstance(result, SubspaceVQEResult)
        assert result.num_states == 2
        # ASE energy is the ground state (eV).
        assert energy_ev == pytest.approx(result.in_units("eV")[0], abs=1e-6)


# --------------------------------------------------------------------------- #
# SubspaceADAPTVQE
# --------------------------------------------------------------------------- #

class TestSubspaceADAPTVQE:
    def test_single_state_recovers_ground(self, h2_hamiltonian, h2_spectrum):
        r = _ss_adapt(h2_hamiltonian, 1).run()
        assert r.optimal_energy == pytest.approx(h2_spectrum[0], abs=1e-6)

    def test_hylleraas_undheim_upper_bounds(self, h2_hamiltonian, h2_spectrum):
        r = _ss_adapt(h2_hamiltonian, 2).run()
        for i, e in enumerate(r.energies):
            assert e >= h2_spectrum[i] - 1e-6

    def test_grows_and_converges(self, h2_hamiltonian):
        r = _ss_adapt(h2_hamiltonian, 2).run()
        assert isinstance(r, SubspaceADAPTVQEResult)
        assert r.num_operators >= 1                   # grew at least one operator
        assert r.converged
        assert len(r.operators) == r.num_operators

    def test_states_orthonormal(self, h2_hamiltonian):
        r = _ss_adapt(h2_hamiltonian, 2).run()
        gram = np.array([[np.vdot(a, b) for b in r.states] for a in r.states])
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-8)

    def test_is_ase_calculator(self):
        from ase import Atoms
        atoms = Atoms("H2", positions=[[4.0, 4.0, 3.63], [4.0, 4.0, 4.37]],
                      cell=[[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], pbc=True)
        atoms.calc = SubspaceADAPTVQE(basis="FAO", h=0.30, num_states=2,
                                      pool="fermionic", verbose=False,
                                      profile=False, gradient_tolerance=1e-4,
                                      max_iterations=20)
        atoms.get_potential_energy()
        assert isinstance(atoms.calc.result, SubspaceADAPTVQEResult)
        assert atoms.calc.result.num_states == 2


# --------------------------------------------------------------------------- #
# Construction guards.
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_num_states_validated_vqe(self):
        with pytest.raises(ValueError):
            SubspaceVQE(num_states=0)

    def test_num_states_validated_adapt(self):
        with pytest.raises(ValueError):
            SubspaceADAPTVQE(num_states=0)

    def test_run_requires_configuration(self):
        with pytest.raises(RuntimeError):
            SubspaceVQE(basis="FAO", num_states=2, verbose=False).run()
