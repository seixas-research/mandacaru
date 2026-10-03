# -*- coding: utf-8 -*-
# file: test/integrals/test_direct.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Integral-direct Coulomb, exchange and RHF against the two-body tensor."""

import numpy as np
import pytest
from ase.build import molecule


def _integrals(basis):
    from mandacaru.algorithms._hamiltonian_from_atoms import \
        build_basis_hamiltonian

    # NH3 tilted off every grid direction: complex p functions off any
    # symmetry plane, where a wrong conjugation pattern shows (the tensor's
    # X-transform once shifted water's energy by 1.2 Ha this way).
    atoms = molecule("NH3")
    atoms.center(vacuum=3.0)
    atoms.rotate(41, (1, 2, 3), center="COM")
    _none, particles, _M, _profile, context = build_basis_hamiltonian(
        atoms, basis, None, 0.3, 0, None, hamiltonian=False)
    return context["integrals"], sum(particles)


@pytest.fixture(scope="module", params=["paw", "hao"])
def system(request):
    basis = ({"name": "PAW-LCAO", "size": "DZP"} if request.param == "paw"
             else "HAO")
    integrals, n_electrons = _integrals(basis)
    return request.param, integrals, n_electrons


def _random_density(M, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
    return A @ A.conj().T / M


class TestDirectCoulomb:
    def test_coulomb_and_exchange_match_the_tensor(self, system):
        kind, integrals, _n = system
        eri = integrals.ao_two_body()
        D = _random_density(eri.shape[0])
        direct = integrals.direct_coulomb()
        J = np.einsum("sr,prqs->pq", D, eri, optimize=True)
        K = np.einsum("sr,prsq->pq", D, eri, optimize=True)
        assert np.abs(direct.coulomb(D) - J).max() < 1e-10
        assert np.abs(direct.exchange(D) - K).max() < 1e-10
        if kind == "paw":
            # The PAW compensation charges really are in play.
            assert direct.augmentation() is not None

    def test_orbital_integrals_match_the_transformed_tensor(self, system):
        _kind, integrals, _n = system
        eri = integrals.ao_two_body()
        rng = np.random.default_rng(2)
        M = eri.shape[0]
        C = rng.normal(size=(M, 4)) + 1j * rng.normal(size=(M, 4))
        reference = np.einsum("ap,bq,cr,ds,abcd->pqrs", C.conj(), C.conj(),
                              C, C, eri, optimize=True)
        direct = integrals.direct_coulomb().orbital_integrals(C)
        assert np.abs(direct - reference).max() < 1e-9 * np.abs(
            reference).max()


class TestDirectHartreeFock:
    def test_it_reaches_the_tensor_solution(self, system):
        _kind, integrals, n_electrons = system
        direct = integrals.direct_hartree_fock(n_electrons)
        tensor = integrals.hartree_fock(n_electrons)
        assert direct.converged
        assert direct.electronic_energy == pytest.approx(
            tensor.electronic_energy, abs=1e-10)
        assert np.allclose(direct.mo_energies, tensor.mo_energies,
                           atol=1e-9)
        # No tensor was formed in the MO basis, and the tensor path's cache
        # was not handed the direct result.
        assert direct.eri_mo is None
        assert tensor.eri_mo is not None

    def test_local_exchange_finishes_on_the_exact_solution(self):
        # Local boxes for the bulk of the SCF, the exact exchange for the
        # last iterations: the result is the exact SCF's, not the local one.
        from mandacaru.integrals.direct import LOCAL_BOX_THRESHOLD

        integrals, n_electrons = _integrals({"name": "PAW-LCAO",
                                             "size": "DZP"})
        integrals.direct_box_threshold = LOCAL_BOX_THRESHOLD
        staged = integrals.direct_hartree_fock(n_electrons)
        tensor = integrals.hartree_fock(n_electrons)
        assert staged.converged
        assert staged.coarse_iterations > 0
        assert staged.electronic_energy == pytest.approx(
            tensor.electronic_energy, abs=1e-9)

    def test_a_periodic_kernel_is_refused(self):
        from mandacaru.integrals.direct import DirectCoulomb

        class Periodic:
            periodic = True

        with pytest.raises(NotImplementedError, match="molecular only"):
            DirectCoulomb(Periodic())


class TestDirectHamiltonian:
    """``molecular_hamiltonian(direct=True)`` equals the tensor build."""

    @pytest.mark.parametrize("basis, spec", [
        ({"name": "PAW-LCAO", "size": "DZP"},
         {"orbitals": 8, "correlating_pairs": True, "symmetry": True}),
        ("HAO", {"frozen": 1, "orbitals": 6}),
    ])
    def test_it_matches_the_tensor_build(self, basis, spec):
        built = []
        for direct in (False, True):
            integrals, n_electrons = _integrals(basis)
            n = n_electrons // 2
            H = integrals.molecular_hamiltonian(
                mo_basis=True, n_electrons=n_electrons, num_particles=(n, n),
                active_space=spec, direct=direct)
            built.append((H, integrals.active_space))
        (tensor, tensor_space), (direct, direct_space) = built
        assert direct_space.active == tensor_space.active
        assert direct_space.frozen == tensor_space.frozen
        difference = tensor + (-1.0) * direct
        worst = max((abs(c) for c in difference.terms.values()), default=0.0)
        assert worst < 1e-9

    def test_canonical_mp2_is_refused_without_the_tensor(self):
        integrals, n_electrons = _integrals("HAO")
        n = n_electrons // 2
        with pytest.raises(ValueError, match="never forms"):
            integrals.molecular_hamiltonian(
                mo_basis=True, n_electrons=n_electrons, num_particles=(n, n),
                active_space={"orbitals": 6, "method": "mp2"}, direct=True)


class TestLocalExchange:
    """Exchange on local boxes around localized factors of the density."""

    @pytest.fixture(scope="class")
    def scf_density(self):
        integrals, n_electrons = _integrals({"name": "PAW-LCAO",
                                             "size": "DZP"})
        rhf = integrals.direct_hartree_fock(n_electrons)
        n = n_electrons // 2
        C = integrals._lowdin_x() @ rhf.mo_coefficients[:, :n]
        return integrals, 2.0 * C @ C.conj().T

    def test_pivoted_cholesky_reproduces_the_density(self, scf_density):
        from mandacaru.integrals.direct import pivoted_cholesky

        _integrals_, D = scf_density
        L = pivoted_cholesky(D)
        assert L.shape[1] == 4                      # NH3: the occupied rank
        assert np.allclose(L @ L.conj().T, D, atol=1e-10)

    def test_the_boys_factors_reproduce_the_density(self, scf_density):
        integrals, D = scf_density
        factors = integrals.direct_coulomb().local_factors(D)
        assert np.allclose(factors @ factors.conj().T, D, atol=1e-10)

    def test_with_no_threshold_it_is_the_global_exchange(self, scf_density):
        integrals, D = scf_density
        direct = integrals.direct_coulomb()
        assert np.abs(direct.exchange_local(D, threshold=1e-30)
                      - direct.exchange(D)).max() < 1e-10
