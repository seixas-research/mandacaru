# -*- coding: utf-8 -*-
# file: test_hartree_fock.py

"""Restricted and unrestricted Hartree-Fock on the real-space integral engine."""

import numpy as np
import pytest

from mandacaru.algorithms import GHF, RHF, UHF, transform_integrals
from mandacaru.core import MolecularIntegrals, minimal_hao_basis
from mandacaru.integrals import Grid


@pytest.fixture(scope="module")
def h2_integrals():
    R = 0.74
    nuclei = [(1.0, np.array([0.0, 0.0, -R / 2])),
              (1.0, np.array([0.0, 0.0, +R / 2]))]
    grid = Grid(center=[0.0, 0.0, 0.0], box_size=6.0, h=0.20)
    return MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)


class TestRHF:
    def test_converges(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        assert rhf.converged
        assert rhf.n_occupied == 1
        assert rhf.mo_coefficients.shape == (2, 2)

    def test_is_variational_above_fci(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        total = rhf.electronic_energy + h2_integrals.nuclear_repulsion
        H = h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        m = H.map_to_qubits("jordan_wigner").to_matrix()
        fci = float(np.linalg.eigvalsh(0.5 * (m + m.conj().T)).min())
        # The mean-field energy lies above the correlated ground state.
        assert total >= fci - 1e-9

    def test_odd_electron_count_rejected(self, h2_integrals):
        with pytest.raises(ValueError):
            RHF(h2_integrals.one_body(), h2_integrals.two_body(), 3)


class TestUHF:
    def test_hydrogen_atom_energy(self):
        # A single electron: UHF is self-interaction-free, E -> the exact H 1s.
        c = np.array([0.03, 0.017, 0.023])          # offset from any grid node
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=8.0, h=0.15)
        ig = MolecularIntegrals([(1.0, c)],
                                 minimal_hao_basis([(1.0, c)]), grid)
        e = UHF(ig.one_body(), ig.two_body(), 1, 0).run()
        assert e == pytest.approx(-0.5, abs=0.05)

    def test_matches_rhf_for_closed_shell(self, h2_integrals):
        rhf = RHF(h2_integrals.one_body(), h2_integrals.two_body(), 2).run()
        uhf = UHF(h2_integrals.one_body(), h2_integrals.two_body(), 1, 1).run()
        assert uhf == pytest.approx(rhf.electronic_energy, abs=1e-7)


class TestTransform:
    def test_identity_transform_is_noop(self, h2_integrals):
        h, eri = h2_integrals.one_body(), h2_integrals.two_body()
        C = np.eye(h.shape[0])
        h_new, eri_new = transform_integrals(h, eri, C)
        assert np.allclose(h_new, h)
        assert np.allclose(eri_new, eri)

    def test_mo_hamiltonian_reference_is_hf(self, h2_integrals):
        rhf = h2_integrals.hartree_fock(2)
        H = h2_integrals.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        m = H.map_to_qubits("jordan_wigner").to_matrix()
        # HF determinant |0101>-type reference (occ spin-orbitals {0, 2}).
        hf_index = (1 << (4 - 1 - 0)) | (1 << (4 - 1 - 2))
        e_ref = float(np.real(m[hf_index, hf_index]))
        assert e_ref == pytest.approx(
            rhf.electronic_energy + h2_integrals.nuclear_repulsion, abs=1e-9)


class TestGHF:
    """Generalized (spinor) Hartree-Fock on a random 4-orbital problem."""

    M = 4

    @pytest.fixture(scope="class")
    def problem(self):
        M = self.M
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(M, M)), rng.normal(size=(M, M))
        c = b + b.T
        eri = 0.3 * (np.einsum("pr,qs->pqrs", c, c)
                     + np.einsum("pr,qs->pqrs", np.eye(M), np.eye(M)))
        coupling = rng.normal(size=(2 * M, 2 * M)) \
            + 1j * rng.normal(size=(2 * M, 2 * M))
        coupling = 0.15 * (coupling + coupling.conj().T)
        coupling[:M, :M] = coupling[M:, M:] = 0.0
        return a + a.T, eri, coupling

    def _spin_blocked(self, h, coupling=None):
        M = self.M
        out = np.zeros((2 * M, 2 * M), dtype=complex)
        out[:M, :M] = out[M:, M:] = h
        return out if coupling is None else out + coupling

    def _exact(self, h_so, g_so, n):
        from mandacaru.core import Fermion

        matrix = Fermion.from_integrals(h_so, g_so).map_to_qubits(
            "jordan_wigner", n_modes=2 * self.M).to_matrix()
        states = [i for i in range(4 ** self.M) if bin(i).count("1") == n]
        return np.linalg.eigvalsh(matrix[np.ix_(states, states)])[0]

    def test_a_closed_shell_without_coupling_is_rhf(self, problem):
        h, eri, _c = problem
        ghf = GHF(self._spin_blocked(h), eri, 4).solve()
        assert ghf.converged
        assert ghf.electronic_energy == pytest.approx(
            RHF(h, eri, 4).run().electronic_energy, abs=1e-9)
        assert ghf.kramers_pairing < 1e-8

    def test_an_open_shell_is_at_or_below_uhf(self, problem):
        """GHF may break collinearity, so it can only be lower."""
        h, eri, _c = problem
        ghf = GHF(self._spin_blocked(h), eri, 3).solve()
        uhf = UHF(h, eri, 2, 1).solve()
        assert ghf.electronic_energy <= uhf.electronic_energy + 1e-9

    @pytest.mark.parametrize("n", [3, 4])
    def test_with_coupling_it_lies_between_exact_and_the_uhf_determinant(
            self, problem, n):
        from mandacaru.core.hamiltonian import spin_block_integrals

        h, eri, coupling = problem
        h_so = self._spin_blocked(h, coupling)
        solver = GHF(h_so, eri, n)
        uhf = UHF(h, eri, (n + 1) // 2, n // 2).solve()
        start = GHF.collinear_spinors(uhf.mo_coefficients_alpha,
                                      uhf.mo_coefficients_beta,
                                      (n + 1) // 2, n // 2)
        ghf = solver.solve(guesses=[start])
        _h, g_so = spin_block_integrals(h, eri)
        exact = self._exact(h_so, g_so, n)
        assert exact <= ghf.electronic_energy + 1e-9
        assert ghf.electronic_energy <= solver.energy_of(start) + 1e-9

    def test_the_spinor_hamiltonian_keeps_the_spectrum_and_the_reference(
            self, problem):
        """Mode order puts the GHF determinant where the reference state of
        (ceil(N/2), floor(N/2)) particles is."""
        from mandacaru.core import Fermion
        from mandacaru.core.hamiltonian import spin_block_integrals

        h, eri, coupling = problem
        n, M = 3, self.M
        h_so = self._spin_blocked(h, coupling)
        ghf = GHF(h_so, eri, n).solve()
        _h, g_so = spin_block_integrals(h, eri)
        assert self._exact(ghf.h_mo, ghf.eri_mo, n) == pytest.approx(
            self._exact(h_so, g_so, n), abs=1e-9)
        matrix = Fermion.from_integrals(ghf.h_mo, ghf.eri_mo).map_to_qubits(
            "jordan_wigner", n_modes=2 * M).to_matrix()
        occupied = [0, 1, M]
        reference = sum(1 << (2 * M - 1 - q) for q in occupied)
        assert matrix[reference, reference].real == pytest.approx(
            ghf.electronic_energy, abs=1e-9)
