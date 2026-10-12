# -*- coding: utf-8 -*-
# file: test/integrals/test_cholesky.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Cholesky-factored two-electron integrals against the dense tensor.

The factors are generated matrix-free on the grid
(:func:`~mandacaru.integrals.cholesky.cholesky_two_body`); here they are
reconstructed and compared with the FFT tensor of the same grid, entry by
entry, and through the contractions the SCF and the active space use.
"""

import numpy as np
import pytest
from ase.build import molecule

from mandacaru.basis import HydrogenicAtomicOrbital
from mandacaru.integrals import Grid, IntegralEngine, PoissonFFTSolver
from mandacaru.integrals.cholesky import CholeskyERI, cholesky_two_body
from mandacaru.integrals.direct import augmentation_tensor


def _pair_matrix(eri):
    """``A[(c,a),(b,d)] = <ab|cd>``: the Hermitian pair matrix."""
    M = eri.shape[0]
    return eri.transpose(2, 0, 1, 3).reshape(M * M, M * M)


def _hydrogenic(closed=True, real=True):
    """A small all-electron basis of real harmonics: full p and d shells, or
    two p functions alone."""
    c = (0.6, 0.2, -0.3)
    fns = [HydrogenicAtomicOrbital(1, 0, 0, real=real),
           HydrogenicAtomicOrbital(2, 0, 0, center=c, real=real),
           HydrogenicAtomicOrbital(2, 1, 1, center=c, real=real),
           HydrogenicAtomicOrbital(2, 1, 0, center=c, real=real)]
    if closed:
        fns += [HydrogenicAtomicOrbital(2, 1, -1, center=c),
                HydrogenicAtomicOrbital(3, 2, 2), HydrogenicAtomicOrbital(3, 2, 1),
                HydrogenicAtomicOrbital(3, 2, 0), HydrogenicAtomicOrbital(3, 2, -1),
                HydrogenicAtomicOrbital(3, 2, -2)]
    engine = IntegralEngine(fns, Grid(box_size=6.0, h=0.5, center=(0, 0, 0)))
    solver = PoissonFFTSolver(engine.grid.shape, step=engine.grid.step)
    return engine, solver


@pytest.fixture(scope="module", params=[True, False],
                ids=["full-shells", "partial-shell"])
def hydrogenic(request):
    engine, solver = _hydrogenic(request.param)
    eri = engine.two_body(energy_units="Ha", solver=solver)
    return engine, solver, eri


@pytest.fixture(scope="module")
def water():
    """Water, PAW-LCAO SZP on a coarse grid, with its augmentation."""
    from mandacaru.algorithms._hamiltonian_from_atoms import \
        build_basis_hamiltonian

    atoms = molecule("H2O")
    atoms.center(vacuum=3.0)
    _none, particles, _M, _profile, context = build_basis_hamiltonian(
        atoms, {"name": "PAW-LCAO", "size": "SZP"}, None, 0.35, 0, None,
        hamiltonian=False)
    integrals = context["integrals"]
    engine = integrals._engine
    solver = PoissonFFTSolver(engine.grid.shape, step=engine.grid.step)
    augmentation = integrals.direct_coulomb().augmentation()
    eri = engine.two_body(energy_units="Ha", solver=solver)
    eri = eri + augmentation_tensor(augmentation, np.eye(eri.shape[0]))
    return integrals, solver, augmentation, eri, int(sum(particles))


def _density(M, rank, seed=3):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(M, rank)) + 1j * rng.normal(size=(M, rank))
    C, _ = np.linalg.qr(C)
    return 2.0 * C @ C.conj().T


class TestTheFactors:
    @pytest.mark.parametrize("tolerance", [1e-4, 1e-6, 1e-9])
    def test_every_integral_is_within_the_tolerance(self, hydrogenic,
                                                    tolerance):
        """The residual pair matrix is positive semidefinite, so each entry is
        bounded by the largest residual diagonal (Cauchy-Schwarz), which the
        factorization drives below the tolerance."""
        engine, solver, eri = hydrogenic
        chol = cholesky_two_body(engine._psi, solver, tolerance)
        assert chol.max_residual <= tolerance
        error = np.abs(chol.dense() - eri).max()
        assert error <= tolerance * (1.0 + 1e-6) + 1e-12

    def test_the_reconstruction_is_hermitian_and_its_residual_positive(
            self, hydrogenic):
        engine, solver, eri = hydrogenic
        chol = cholesky_two_body(engine._psi, solver, 1e-5)
        A = _pair_matrix(chol.dense())
        assert np.abs(A - A.conj().T).max() < 1e-13
        residual = _pair_matrix(eri) - A
        residual = 0.5 * (residual + residual.conj().T)
        assert np.linalg.eigvalsh(residual).min() > -1e-10
        assert np.real(np.diag(residual)).min() > -1e-12

    def test_the_rank_grows_as_the_tolerance_tightens(self, hydrogenic):
        engine, solver, _eri = hydrogenic
        ranks = [cholesky_two_body(engine._psi, solver, t).rank
                 for t in (1e-1, 1e-2, 1e-4, 1e-6, 1e-9)]
        assert ranks == sorted(ranks) and ranks[0] < ranks[-1]

    def test_the_factor_matrices_are_real_symmetric(self, hydrogenic):
        engine, solver, _eri = hydrogenic
        B = cholesky_two_body(engine._psi, solver, 1e-6).matrices()
        assert B.dtype == np.float64
        assert np.abs(B - B.transpose(0, 2, 1)).max() == 0.0


class TestTheConsumers:
    def test_coulomb_and_exchange_match_the_dense_contraction(self,
                                                              hydrogenic):
        engine, solver, eri = hydrogenic
        chol = cholesky_two_body(engine._psi, solver, 1e-10)
        D = _density(eri.shape[0], 2)
        J = np.einsum("sr,prqs->pq", D, eri, optimize=True)
        K = np.einsum("sr,prsq->pq", D, eri, optimize=True)
        Jc, Kc = chol.jk(D)
        assert np.abs(Jc - J).max() < 1e-9
        assert np.abs(Kc - K).max() < 1e-9

    def test_orbital_blocks_match_the_transformed_tensor(self, hydrogenic):
        engine, solver, eri = hydrogenic
        chol = cholesky_two_body(engine._psi, solver, 1e-10)
        M = eri.shape[0]
        rng = np.random.default_rng(5)
        C = np.linalg.qr(rng.normal(size=(M, M))
                         + 1j * rng.normal(size=(M, M)))[0]
        reference = np.einsum("ap,bq,cr,ds,abcd->pqrs", C.conj(), C.conj(),
                              C, C, eri, optimize=True)
        n, top = min(4, M), min(5, M)
        assert np.abs(chol.active_tensor(C[:, :n])
                      - reference[:n, :n, :n, :n]).max() < 1e-9
        occ, vir = C[:, :2], C[:, 2:top]
        ovov = chol.ovov(occ, vir)                        # (ia|jb)
        assert np.abs(ovov - reference[:2, :2, 2:top, 2:top].transpose(
            0, 2, 1, 3)).max() < 1e-9
        pair = chol.pair_exchange(occ, vir)               # <ii|aa>
        expected = np.array([[reference[i, i, a, a] for a in range(2, top)]
                             for i in range(2)])
        assert np.abs(pair - expected).max() < 1e-9
        blocks = chol.exchange_blocks(occ, vir, {(0, 1): [0, 1]})
        assert np.abs(blocks[(0, 1)]
                      - reference[0, 1, 2:4, 2:4]).max() < 1e-9
        mo = chol.to_mo(C)
        assert mo.shape == (chol.rank, M, M)


class TestPAWWater:
    """PAW-LCAO water: the augmentation is carried structured and exactly."""

    def test_the_augmentation_is_exact_and_apart(self, water):
        integrals, solver, augmentation, eri, _n = water
        psi = integrals._engine._psi
        chol = cholesky_two_body(psi, solver, 1e-10,
                                 augmentation=augmentation)
        bare = cholesky_two_body(psi, solver, 1e-10)
        assert np.abs(chol.dense() - eri).max() < 1e-9
        M = eri.shape[0]
        exact = augmentation_tensor(augmentation, np.eye(M))
        assert np.abs((chol.dense() - bare.dense()) - exact).max() < 1e-12
        assert chol.nbytes > bare.nbytes
        # The structured part reaches every consumer, the pair blocks too.
        C = np.linalg.qr(np.random.default_rng(2).normal(size=(M, 4)))[0]
        blocks = chol.exchange_blocks(C[:, :2], C[:, 2:], {(0, 1): [0, 1]})
        reference = np.einsum("ai,bj,cp,dq,abcd->ijpq", C[:, :2].conj(),
                              C[:, :2].conj(), C[:, 2:], C[:, 2:], eri,
                              optimize=True)
        assert np.abs(blocks[(0, 1)] - reference[0, 1]).max() < 1e-9

    def test_a_fixed_density_energy_at_the_default_tolerance(self, water):
        """The two-electron energy of the core-guess RHF density in the
        orthonormal basis, ``1/2 tr(D J) - 1/4 tr(D K)``, within 1e-6 Ha."""
        from mandacaru.integrals.cholesky import DEFAULT_TOLERANCE

        integrals, solver, augmentation, eri, n = water
        chol = cholesky_two_body(integrals._engine._psi, solver,
                                 DEFAULT_TOLERANCE, augmentation=augmentation)
        X = integrals._lowdin_x()
        h = integrals.one_body()
        _e, C = np.linalg.eigh(h)
        Co = X @ C[:, :n // 2]
        D = 2.0 * Co @ Co.conj().T                       # AO density
        J = np.einsum("sr,prqs->pq", D, eri, optimize=True)
        K = np.einsum("sr,prsq->pq", D, eri, optimize=True)
        exact = np.real(0.5 * np.sum(D.T * J) - 0.25 * np.sum(D.T * K))
        Jc, Kc = chol.jk(D)
        approx = np.real(0.5 * np.sum(D.T * Jc) - 0.25 * np.sum(D.T * Kc))
        assert abs(approx - exact) < 1e-6


class TestRefusals:
    def test_a_skewed_grid_is_refused(self):
        step = np.array([[0.4, 0.1, 0.0], [0.0, 0.38, 0.05],
                         [0.02, 0.0, 0.41]])
        solver = PoissonFFTSolver((8, 9, 10), step=step)
        with pytest.raises(ValueError, match="orthogonal grid"):
            cholesky_two_body(np.ones((1, 720), complex), solver)

    def test_a_periodic_solver_is_refused(self):
        from mandacaru.integrals.poisson import PeriodicPoissonSolver

        solver = PeriodicPoissonSolver((6, 6, 6), spacing=0.5)
        with pytest.raises(ValueError, match="isolated solver"):
            cholesky_two_body(np.ones((1, 216), complex), solver)

    def test_complex_samples_are_refused(self):
        """A basis of complex harmonics (a crystal's) has complex pair
        densities: the factorization is for a molecule's real ones."""
        engine, solver = _hydrogenic(True, real=False)
        with pytest.raises(ValueError, match="complex"):
            cholesky_two_body(engine._psi, solver)

    def test_the_tolerance_must_be_positive(self, hydrogenic):
        engine, solver, _eri = hydrogenic
        with pytest.raises(ValueError, match="positive"):
            cholesky_two_body(engine._psi, solver, 0.0)


def test_the_class_is_exported():
    from mandacaru.integrals import cholesky

    assert cholesky.CholeskyERI is CholeskyERI
