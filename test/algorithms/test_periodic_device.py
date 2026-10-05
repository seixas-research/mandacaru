# -*- coding: utf-8 -*-
# file: test/algorithms/test_periodic_device.py

# This code is part of Mandacaru.
# MIT License

"""The per-k-point operations of the periodic SCF behind a device."""

import numpy as np
import pytest
from ase.build import bulk
from scipy.linalg import eigh
from threadpoolctl import threadpool_info

from mandacaru.algorithms import periodic_device as pdev
from mandacaru.algorithms.periodic_dft import PeriodicKohnSham

SILICON = bulk("Si", "diamond", a=5.43)


@pytest.fixture(scope="module")
def converged():
    """``(solver, result)``: diamond Si, SZ, coarse grid, a 2x2x1 mesh, LDA."""
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    crystal, context = build_crystal(
        SILICON, float(np.linalg.norm(SILICON.cell[0])) / 8,
        {"size": "SZ", "filter": 200},
        kpts={"size": (2, 2, 1), "gamma": True})
    solver = PeriodicKohnSham(crystal, context["n_electrons"], "lda")
    return solver, solver.run()


@pytest.fixture
def solver(converged):
    return converged[0]


def _dense_moments(crystal, data):
    """``{(A, L, M): C_A blk C_A^dagger}`` -- the dense operators the
    projector-space ones replace."""
    C = data.projections
    return {ch: C[:, crystal.projector_columns(ch[0])] @ blk
            @ C[:, crystal.projector_columns(ch[0])].conj().T
            for ch, blk in crystal.multipole_blocks.items()}


class _Stack:
    """Random Hermitian ``H`` and positive definite ``S`` per k-point."""

    def __init__(self, nk=3, M=7, seed=5):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(nk, M, M)) + 1j * rng.normal(size=(nk, M, M))
        B = rng.normal(size=(nk, M, M)) + 1j * rng.normal(size=(nk, M, M))
        self.H = A + np.swapaxes(A, -1, -2).conj()
        self.S = B @ np.swapaxes(B, -1, -2).conj() + M * np.eye(M)

    def device(self):
        from types import SimpleNamespace
        return pdev.CPUDevice(None, [SimpleNamespace(overlap=S)
                                     for S in self.S])


class TestTheEigensolver:
    def test_it_matches_the_generalized_lapack_solve(self):
        stack = _Stack()
        eps, C = stack.device().eigensolve(stack.H)
        for k in range(len(stack.H)):
            assert np.allclose(eps[k], eigh(stack.H[k], stack.S[k],
                                            eigvals_only=True), atol=1e-12)

    def test_the_vectors_are_s_orthonormal_eigenvectors(self):
        stack = _Stack()
        eps, C = stack.device().eigensolve(stack.H)
        for k in range(len(stack.H)):
            Ck = C[k]
            assert np.allclose(Ck.conj().T @ stack.S[k] @ Ck, np.eye(len(Ck)),
                               atol=1e-12)
            assert np.allclose(stack.H[k] @ Ck, stack.S[k] @ Ck * eps[k],
                               atol=1e-10)

    def test_a_leading_spin_axis_is_solved_channel_by_channel(self):
        stack = _Stack()
        device = stack.device()
        both = np.stack([stack.H, 2.0 * stack.H])
        eps, _C = device.eigensolve(both)
        assert eps.shape == (2,) + stack.H.shape[:2]
        assert np.allclose(eps[1], 2.0 * device.eigensolve(stack.H)[0])


class TestTheDensityMatrices:
    def test_they_are_c_f_c_dagger(self):
        rng = np.random.default_rng(1)
        C = rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4))
        f = rng.uniform(0, 2, size=(2, 4))
        P = _Stack().device().density_matrices(C, f)
        for k in range(2):
            assert np.allclose(P[k], C[k] @ np.diag(f[k]) @ C[k].conj().T)


class TestTheHamiltonians:
    def test_they_are_the_dense_kohn_sham_matrices(self, solver):
        crystal = solver.crystal
        V, v_tau, w = solver.potentials
        H = solver.device.hamiltonians(V, v_tau, w)
        for k, data in enumerate(crystal.kpoint_data):
            psi = data.psi
            dense = (data.fixed + ((psi.conj() * V) @ psi.T) * crystal.grid.dV
                     + sum(w[ch] * Q for ch, Q in
                           _dense_moments(crystal, data).items()))
            dense = 0.5 * (dense + dense.conj().T)
            assert np.allclose(H[k], dense, atol=1e-12)

    def test_the_scf_eigenvalues_solve_them(self, converged):
        solver, result = converged
        V, v_tau, w = solver.potentials
        device = solver.device
        eps, _C = device.eigensolve(device.hamiltonians(V, v_tau, w))
        assert np.allclose(eps, np.array(result.eigenvalues)
                           - solver.eigenvalue_reference(), atol=1e-12)


class TestTheDensity:
    def test_a_device_on_some_k_points_uses_their_bloch_sums(self, solver):
        """A device on a subset of the mesh sums over that subset -- with
        its own weights and Bloch sums, not the crystal's first points."""
        crystal = solver.crystal
        everything = crystal.kpoint_data
        assert len(everything) > 1
        last = [everything[-1]]
        P = solver.density_matrices[-1:]
        rho, q = pdev.make_device(crystal, last).density(P)
        expected_rho, expected_q = crystal.density(P, last)
        assert np.allclose(rho, expected_rho, atol=1e-14)
        first_rho, _q = crystal.density(P, everything[:1])
        assert not np.allclose(rho, first_rho)

    def test_matrices_and_k_points_must_pair(self, solver):
        with pytest.raises(ValueError, match="density matrices"):
            solver.crystal.density(solver.density_matrices[:1])


class TestThreads:
    def test_the_grid_products_use_the_physical_cores(self, solver,
                                                      monkeypatch):
        seen = []
        original = pdev.kohn_sham_matrix

        def spy(*args, **kwargs):
            seen.append({p["num_threads"] for p in threadpool_info()
                         if p["user_api"] == "blas"})
            return original(*args, **kwargs)

        monkeypatch.setattr(pdev, "kohn_sham_matrix", spy)
        V, v_tau, w = solver.potentials
        solver.device.hamiltonians(V, v_tau, w)
        assert seen and all(s == {solver.device.grid_threads} for s in seen)
