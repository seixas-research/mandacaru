# -*- coding: utf-8 -*-
# file: test_planewave.py

"""Plane-wave basis: periodic one-/two-body integrals and driver integration.

Covers the reciprocal-space plane-wave engine (kinetic diagonal, Hermitian
potential, momentum-conserving electron repulsion, jellium G=0 handling) and its
selection through the ``{"name": "PW", "energy_cutoff": ...}`` basis dict.
"""

import numpy as np
import pytest
from ase import Atoms

from carcara.core import PlaneWaveIntegrals, plane_wave_vectors
from carcara.algorithms import ADAPTVQE, VQE
from carcara.algorithms._hamiltonian_from_atoms import (build_basis_hamiltonian,
                                                        resolve_basis)

EV = 1.0 / 27.211386245988   # eV -> Hartree
B2A = 0.52917721             # Bohr -> Angstrom


class TestPlaneWaveVectors:
    def test_cubic_shells(self):
        cell = np.eye(3) * 4.0                     # 4 Bohr cube
        # A cubic reciprocal lattice fills the shells 1, 7, 27, ...
        assert len(plane_wave_vectors(cell, 30 * EV)[0]) == 1        # Gamma only
        assert len(plane_wave_vectors(cell, 60 * EV)[0]) == 7        # + 6 faces
        assert len(plane_wave_vectors(cell, 120 * EV)[0]) == 27

    def test_gamma_is_first_and_included(self):
        G, miller = plane_wave_vectors(np.eye(3) * 5.0, 80 * EV)
        np.testing.assert_allclose(G[0], [0, 0, 0])
        assert tuple(miller[0]) == (0, 0, 0)

    def test_cutoff_respected(self):
        cell = np.eye(3) * 4.0
        ecut = 90 * EV
        G, _ = plane_wave_vectors(cell, ecut)
        assert np.all(0.5 * np.sum(G * G, axis=1) <= ecut + 1e-9)


class TestPlaneWaveIntegrals:
    @pytest.fixture
    def pw(self):
        cell = np.eye(3) * 4.0                     # Bohr
        nuclei = [(1.0, np.array([2.0, 2.0, 1.6])),
                  (1.0, np.array([2.0, 2.0, 2.4]))]
        return PlaneWaveIntegrals(nuclei, cell, energy_cutoff=60, units="bohr")

    def test_kinetic_is_diagonal_g_squared_over_two(self, pw):
        h = pw.one_body()
        expected = 0.5 * np.sum(pw.G * pw.G, axis=1)
        # The diagonal is purely kinetic (the G=0 external term is dropped).
        np.testing.assert_allclose(np.diag(h).real, expected, atol=1e-10)

    def test_one_body_hermitian(self, pw):
        h = pw.one_body()
        np.testing.assert_allclose(h, h.conj().T, atol=1e-10)

    def test_two_body_physicist_symmetry(self, pw):
        eri = pw.two_body()
        # <pq|rs> == <rs|pq> (real, Hermitian in reciprocal space).
        np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10)

    def test_two_body_conserves_momentum(self, pw):
        eri = pw.two_body()
        mill = pw.miller
        nz = np.argwhere(np.abs(eri) > 1e-12)
        for p, q, r, s in nz:
            # Non-zero entries satisfy G_p + G_q == G_r + G_s and G_p != G_r.
            assert np.array_equal(mill[p] + mill[q], mill[r] + mill[s])
            assert not np.array_equal(mill[p], mill[r])

    def test_nuclear_repulsion(self, pw):
        # Two protons 0.8 Bohr apart -> 1/0.8 = 1.25 Ha.
        assert pw.nuclear_repulsion == pytest.approx(1.25, abs=1e-9)

    def test_overlap_is_identity(self, pw):
        np.testing.assert_allclose(pw.overlap(), np.eye(pw.npw))

    def test_molecular_hamiltonian_is_hermitian(self, pw):
        H = pw.molecular_hamiltonian()             # 7 PWs -> 14-qubit Fermion
        assert H.map_to_qubits("jordan_wigner").is_hermitian()

    def test_too_many_plane_waves_raises(self):
        cell = np.eye(3) * 20.0                     # large cell -> many PWs
        with pytest.raises(ValueError, match="plane waves"):
            PlaneWaveIntegrals([(1.0, np.zeros(3))], cell, energy_cutoff=300,
                               units="bohr")

    def test_integration_profile_reports_pw_stages(self, pw):
        pw.molecular_hamiltonian()
        prof = pw.integration_profile()
        assert any("PW" in k for k in prof["stages_s"])


class TestBasisDict:
    def test_resolve_basis_forms(self):
        assert resolve_basis("FAO") == ("FAO", {})
        assert resolve_basis({"name": "FAO"}) == ("FAO", {})
        assert resolve_basis({"name": "NAO", "energy_shift": 0.03}) == (
            "NAO", {"energy_shift": 0.03})
        assert resolve_basis({"name": "PW", "energy_cutoff": 300}) == (
            "PW", {"energy_cutoff": 300})

    def test_dict_without_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            resolve_basis({"energy_cutoff": 300})

    def test_localized_dict_passes_options(self):
        # A GTO dict routes n_gaussians through to BasisSet.build.
        atoms = Atoms("H2", positions=[[3, 3, 2.63], [3, 3, 3.37]],
                      cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
        H, npart, norb, _, _ctx = build_basis_hamiltonian(
            atoms, {"name": "GTO", "n_gaussians": 3}, None, 0.4, 0, None)
        assert norb == 2 and npart == (1, 1)       # STO-3G H2 -> 2 orbitals

    def test_pw_dict_builds_periodic_hamiltonian(self):
        cell = np.diag([4 * B2A, 4 * B2A, 4 * B2A])
        atoms = Atoms("H2", positions=[[2 * B2A, 2 * B2A, 2 * B2A - 0.37],
                                       [2 * B2A, 2 * B2A, 2 * B2A + 0.37]],
                      cell=cell, pbc=True)
        H, npart, norb, prof, _ctx = build_basis_hamiltonian(
            atoms, {"name": "PW", "energy_cutoff": 60}, None, 0.2, 0, None)
        assert norb == 7                            # 4 Bohr cube, 60 eV -> 7 PWs
        assert H.map_to_qubits("jordan_wigner").is_hermitian()


class TestPlaneWaveDrivers:
    """A tiny anisotropic cell keeps the plane-wave count small enough to run."""

    def _atoms(self):
        cell = np.diag([16 * B2A, 3 * B2A, 3 * B2A])
        pos = [[8 * B2A, 1.5 * B2A, 1.5 * B2A - 0.37],
               [8 * B2A, 1.5 * B2A, 1.5 * B2A + 0.37]]
        return Atoms("H2", positions=pos, cell=cell, pbc=True)

    def test_vqe_runs_with_plane_wave_basis(self):
        atoms = self._atoms()
        atoms.calc = VQE(basis={"name": "PW", "energy_cutoff": 8},
                         optimizer="COBYLA", verbose=False)
        energy = atoms.get_total_energy()
        assert np.isfinite(energy)
        assert atoms.calc.n_qubits == 6            # 3 PWs -> 6 qubits

    def test_adapt_runs_with_plane_wave_basis(self):
        atoms = self._atoms()
        atoms.calc = ADAPTVQE(pool="ceo", basis={"name": "PW", "energy_cutoff": 8},
                              verbose=False, max_iterations=6,
                              gradient_tolerance=1e-3)
        assert np.isfinite(atoms.get_total_energy())
        assert atoms.calc.adapt_result.integration_profile is not None

    def test_pw_requires_cell(self):
        atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]])  # no cell
        atoms.calc = VQE(basis={"name": "PW", "energy_cutoff": 8}, verbose=False)
        with pytest.raises(ValueError, match="unit cell"):
            atoms.get_total_energy()
