# -*- coding: utf-8 -*-
# file: test/algorithms/test_periodic_exchange.py

# This code is part of Mandacaru.
# MIT License

"""Short-range exact exchange in a crystal (screened hybrids)."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk

from mandacaru.algorithms import periodic_exchange as px
from mandacaru.algorithms.periodic_device import make_device
from mandacaru.algorithms.periodic_dft import PeriodicKohnSham, occupation
from mandacaru.integrals import reciprocal as rc
from mandacaru.integrals.poisson import short_range_coulomb_kernel

SILICON = bulk("Si", "diamond", a=5.43)
OMEGA = 0.11
MESH = 2


@pytest.fixture(scope="module")
def converged():
    """Si, SZ, a 2x2x2 mesh on an **odd** grid (9^3: no Nyquist plane, so
    the grid keeps its symmetry), LDA; and a solver for any k-points at
    the converged potential."""
    from mandacaru.pseudopotentials.periodic_paw import build_crystal
    crystal, context = build_crystal(
        SILICON, float(np.linalg.norm(SILICON.cell[0])) / 9,
        {"size": "SZ", "filter": 200},
        kpts={"size": (MESH,) * 3, "gamma": True})
    solver = PeriodicKohnSham(crystal, context["n_electrons"], "lda")
    result = solver.run()
    V, v_tau, w = solver.potentials
    mu = result.fermi_level - solver.eigenvalue_reference()

    def solve(kpoint_data):
        device = make_device(crystal, kpoint_data)
        eps, C = device.eigensolve(device.hamiltonians(V, v_tau, w))
        return C, 2.0 * occupation((eps - mu) / solver.width, solver.method)

    return crystal, solve


def _direct_states(crystal, solve):
    """The occupied states solved at every point of the mesh -- no images."""
    B = rc.reciprocal_vectors(crystal.lattice)
    fractional = np.array(list(itertools.product(range(MESH), repeat=3))) \
        / MESH
    full = crystal.kpoint_matrices(fractional @ B.T,
                                   np.full(MESH ** 3, 1.0 / MESH ** 3))
    vectors, occupations = solve(full)
    states = px.MeshStates()
    for data, C, f in zip(full, vectors, occupations):
        occupied = f > px.OCCUPATION_FLOOR
        states.kpoints.append(data.k)
        states.weights.append(data.weight)
        states.occupations.append(f[occupied])
        states.orbitals.append(C[:, occupied].T @ data.psi)
        states.projections.append(C[:, occupied].conj().T @ data.projections)
    return full, vectors, occupations, states


def _exchange(crystal):
    return px.PeriodicExchange(crystal, OMEGA)


class TestTheMeshStates:
    def test_the_stars_fill_the_mesh_with_equal_weights(self, converged):
        crystal, solve = converged
        exchange = _exchange(crystal)
        states = exchange.mesh_states(crystal.kpoint_data,
                                      *solve(crystal.kpoint_data))
        assert len(states.kpoints) == MESH ** 3
        assert np.allclose(states.weights, 1.0 / MESH ** 3)

    def test_a_wedge_its_operations_did_not_reduce_is_refused(self,
                                                              converged):
        crystal, solve = converged
        C, f = solve(crystal.kpoint_data)
        skewed = [SimpleNamespace(k=d.k, psi=d.psi, weight=1.0 / len(C))
                  for d in crystal.kpoint_data]
        with pytest.raises(ValueError, match="share one weight"):
            _exchange(crystal).mesh_states(skewed, C, f)

    def test_the_images_are_the_states_solved_there(self, converged):
        """The exchange over the wedge's images equals the exchange over
        states solved at every mesh point.  The direct solves themselves
        break the star's symmetry by ~3e-7 Ha in eigenvalue (the sphere
        quadratures and the spectral kinetic term), which bounds this."""
        crystal, solve = converged
        exchange = _exchange(crystal)
        states = exchange.mesh_states(crystal.kpoint_data,
                                      *solve(crystal.kpoint_data))
        K = exchange.matrices(crystal.kpoint_data, states)
        *_rest, direct = _direct_states(crystal, solve)
        reference = exchange.matrices(crystal.kpoint_data, direct)
        assert np.abs(K - reference).max() < 5e-6 * np.abs(reference).max()

    def test_the_projected_density_is_the_direct_solves(self, converged):
        """The spheres' density matrix from the images' projections equals
        the one of states solved at every mesh point."""
        crystal, solve = converged
        states = _exchange(crystal).mesh_states(crystal.kpoint_data,
                                                *solve(crystal.kpoint_data))
        *_rest, direct = _direct_states(crystal, solve)
        R, reference = states.projected_density(), direct.projected_density()
        assert np.allclose(R, R.conj().T, atol=1e-13)
        assert np.abs(R - reference).max() < 5e-6 * np.abs(reference).max()


class TestTheExchangeMatrices:
    def test_they_are_hermitian(self, converged):
        crystal, solve = converged
        exchange = _exchange(crystal)
        states = exchange.mesh_states(crystal.kpoint_data,
                                      *solve(crystal.kpoint_data))
        for K in exchange.matrices(crystal.kpoint_data, states):
            assert np.abs(K - K.conj().T).max() < 1e-13

    def test_at_gamma_they_contract_the_four_index_integrals(self,
                                                             converged):
        r"""The q = Gamma term at k = Gamma against
        :math:`\sum_{\lambda\sigma} P_{\lambda\sigma}(\mu\lambda|\sigma\nu)`,
        every basis pair density solved separately."""
        crystal, solve = converged
        full, vectors, occupations, direct = _direct_states(crystal, solve)
        assert np.allclose(full[0].k, 0.0)
        gamma = px.MeshStates(direct.kpoints[:1], [1.0],
                              direct.occupations[:1], direct.orbitals[:1],
                              direct.projections[:1])
        exchange = _exchange(crystal)
        K = exchange.matrices(full[:1], gamma)[0]
        psi, C, M = full[0].psi, full[0].projections, crystal.M
        # Augmented basis pairs chi_l^* chi_n: transforms T[l, n, G] of the
        # smooth pair (FFT) plus its compensation charges.
        grid = crystal.grid
        G = rc.wavevectors(grid)
        kernel = short_range_coulomb_kernel(np.sum(G ** 2, axis=0),
                                            OMEGA).ravel()
        origin = np.exp(-1j * np.einsum("c,cxyz->xyz", rc.grid_origin(grid),
                                        G)).ravel()
        pairs = (psi.conj()[:, None, :] * psi[None, :, :]).reshape(
            M, M, *grid.shape)
        T = np.fft.fftn(pairs, axes=(2, 3, 4)).reshape(M, M, -1) \
            * origin * grid.dV
        shapes = crystal.compensation_transforms(G).reshape(
            len(crystal.channels), -1)
        moments = np.stack([C[:, crystal.projector_columns(ch[0])] @ blk
                            @ C[:, crystal.projector_columns(ch[0])].conj().T
                            for ch, blk in crystal.multipole_blocks.items()])
        T = T + np.einsum("cln,cg->lng", moments, shapes)
        # (mu lam|sig nu) = <pair(lam, mu)|v|pair(sig, nu)> + dense part
        eri = np.einsum("lmg,g,sng->mlsn", T.conj(), kernel, T) \
            / crystal.volume
        correction = exchange._dense_correction(
            np.zeros(3), "gamma-test", kernel, shapes)
        eri += np.einsum("clm,cd,dsn->mlsn", moments.conj(), correction,
                         moments)
        P = (vectors[0] * occupations[0]) @ vectors[0].conj().T
        expected = np.einsum("ls,mlsn->mn", P, eri)
        assert np.abs(K - expected).max() < 1e-11

    def test_the_exchange_is_symmetric_between_two_densities(self,
                                                             converged):
        r""":math:`\mathrm{tr}(P_1K[P_2]) = \mathrm{tr}(P_2K[P_1])` summed
        over the wedge: the pairing of k and q, the images and the weights
        all enter."""
        crystal, solve = converged
        C, f1 = solve(crystal.kpoint_data)
        f2 = [np.where(f > 1.0, 0.5 * f, 0.0) for f in f1]
        traces = []
        exchange = _exchange(crystal)
        for fa, fb in ((f1, f2), (f2, f1)):
            states = exchange.mesh_states(crystal.kpoint_data, C, fb)
            K = exchange.matrices(crystal.kpoint_data, states)
            traces.append(sum(
                d.weight * np.sum(((Ck * fk) @ Ck.conj().T) * Kk.T)
                for d, Ck, fk, Kk in zip(crystal.kpoint_data, C, fa, K)))
        assert traces[0] == pytest.approx(traces[1], abs=1e-9)
