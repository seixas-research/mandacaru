# -*- coding: utf-8 -*-
# file: test/algorithms/test_mp2.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Second-order Moller-Plesset theory and the natural orbitals it defines.

The spin-summed closed-shell formulas in :mod:`mandacaru.algorithms.mp2` carry
prefactors that a *selection* would never expose -- ranking the virtuals uses
only the ordering of the eigenvalues, which any positive multiple of the density
reproduces -- but that a reported occupation number would be wrong by.  So the
spin-orbital expressions are implemented independently here and the two are
pinned against each other.

A note on the numbers below.  These fixtures put a Gaussian basis on a coarse
real-space grid (``h = 0.3``), which under-resolves a compact core badly -- for
LiH the Li 1s resolution ratio is ~0.63, so absolute Hartree-Fock and full-CI
energies here are Hartree away from published 6-31G values.  That is harmless
for what is being tested, because every comparison is made between two
calculations sharing the same integrals, and it is a pre-existing property of
building a Gaussian basis on a grid rather than anything to do with the active
space.  It does mean **no absolute energy in this file is literature-comparable**
and none should be quoted as one.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.core import MolecularIntegrals
from mandacaru.core.hamiltonian import molecular_orbital_integrals
from mandacaru.integrals import Grid
from mandacaru.algorithms.mp2 import (MIN_DENOMINATOR, fock_matrix,
                                      mp2_amplitudes, mp2_density, mp2_energy,
                                      mp2_natural_orbitals, rotate_integrals,
                                      semicanonical_rotation)


# --------------------------------------------------------------------------- #
# Fixtures: one grid-based problem, built once.
# --------------------------------------------------------------------------- #

def _h2_integrals(basis="6-31G", distance=0.74, box=8.0, h=0.3):
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, distance)])
    positions = atoms.get_positions()
    from mandacaru.basis import BasisSet

    bset = BasisSet.build(basis)
    functions, nuclei = [], []
    for Z, symbol, position in zip(atoms.get_atomic_numbers(),
                                   atoms.get_chemical_symbols(), positions):
        functions += bset.atom(symbol, center=position, units="angstrom")
        nuclei.append((float(Z), position))
    grid = Grid(center=positions.mean(axis=0), box_size=box, h=h,
                units="angstrom")
    return MolecularIntegrals(nuclei, functions, grid,
                              softening=0.5 * min(grid.dx, grid.dy, grid.dz))


@pytest.fixture(scope="module")
def h2():
    """``(h_mo, eri_mo, n_occ)`` of H2 in a 4-orbital Gaussian basis."""
    integrals = _h2_integrals()
    h_mo, eri_mo, _orbitals = molecular_orbital_integrals(integrals, 2, (1, 1),
                                                          None)
    return np.real(h_mo), np.real(eri_mo), 1


# --------------------------------------------------------------------------- #
# An independent spin-orbital MP2, used only to pin the closed-shell prefactors.
# --------------------------------------------------------------------------- #

def _spin_orbital_reference(h_mo, eri_mo, n_occ):
    r"""``(E_MP2, D_oo, D_vv)`` from the spin-orbital expressions.

    Blocks the spatial integrals into spin orbitals (alpha then beta), forms the
    antisymmetrized amplitudes ``t_IJ^AB = <IJ||AB> / D``, and evaluates

        E = 1/4 sum_IJAB <IJ||AB> t_IJ^AB
        D_AB = +1/2 sum_IJC t_IJ^AC t_IJ^BC
        D_IJ = -1/2 sum_ABK t_IK^AB t_JK^AB

    then spin-sums each spatial block.  Deliberately written from the textbook
    formulas rather than from the module under test.
    """
    M = h_mo.shape[0]
    fock = fock_matrix(h_mo, eri_mo, n_occ)
    eps = np.real(np.diag(fock))
    # Spin orbital P = p + spin * M; occupations 0..n_occ-1 in each spin block.
    occupied = [p for p in range(n_occ)] + [p + M for p in range(n_occ)]
    virtual = [p for p in range(n_occ, M)] + [p + M for p in range(n_occ, M)]

    def spatial(P):
        return P % M

    def spin(P):
        return P // M

    def anti(I, J, A, B):
        """``<IJ||AB>`` for spin orbitals, from the spatial integrals."""
        direct = (eri_mo[spatial(I), spatial(J), spatial(A), spatial(B)]
                  if spin(I) == spin(A) and spin(J) == spin(B) else 0.0)
        exchange = (eri_mo[spatial(I), spatial(J), spatial(B), spatial(A)]
                    if spin(I) == spin(B) and spin(J) == spin(A) else 0.0)
        return direct - exchange

    no, nv = len(occupied), len(virtual)
    t = np.zeros((no, no, nv, nv))
    for i, I in enumerate(occupied):
        for j, J in enumerate(occupied):
            for a, A in enumerate(virtual):
                for b, B in enumerate(virtual):
                    denominator = (eps[spatial(I)] + eps[spatial(J)]
                                   - eps[spatial(A)] - eps[spatial(B)])
                    t[i, j, a, b] = anti(I, J, A, B) / denominator

    integrals = np.array([[[[anti(I, J, A, B) for B in virtual]
                            for A in virtual] for J in occupied]
                          for I in occupied], dtype=float)
    energy = 0.25 * np.einsum("ijab,ijab->", integrals, t)

    d_vv_so = 0.5 * np.einsum("ijac,ijbc->ab", t, t)
    d_oo_so = -0.5 * np.einsum("ikab,jkab->ij", t, t)
    # Spin-sum: the spatial block is the sum of the two spin blocks.
    n_spatial_v, n_spatial_o = M - n_occ, n_occ
    d_vv = (d_vv_so[:n_spatial_v, :n_spatial_v]
            + d_vv_so[n_spatial_v:, n_spatial_v:])
    d_oo = (d_oo_so[:n_spatial_o, :n_spatial_o]
            + d_oo_so[n_spatial_o:, n_spatial_o:])
    return float(energy), d_oo, d_vv


# --------------------------------------------------------------------------- #
# The Fock matrix and semicanonicalization.
# --------------------------------------------------------------------------- #

class TestTheFockMatrix:
    def test_it_is_diagonal_in_the_canonical_basis(self, h2):
        h_mo, eri_mo, n_occ = h2
        fock = fock_matrix(h_mo, eri_mo, n_occ)
        off_diagonal = fock - np.diag(np.diag(fock))
        assert np.max(np.abs(off_diagonal)) < 1e-8

    def test_its_diagonal_is_the_orbital_energies(self):
        # The one genuinely independent check: the RHF solver reports orbital
        # energies of its own, and the contraction here has to reproduce them.
        integrals = _h2_integrals()
        rhf = integrals.hartree_fock(2)
        fock = fock_matrix(np.real(rhf.h_mo), np.real(rhf.eri_mo), 1)
        assert np.allclose(np.real(np.diag(fock)),
                           np.real(rhf.mo_energies), atol=1e-8)

    def test_it_is_hermitian(self, h2):
        h_mo, eri_mo, n_occ = h2
        fock = fock_matrix(h_mo, eri_mo, n_occ)
        assert np.allclose(fock, fock.conj().T)


class TestSemicanonicalization:
    def test_a_canonical_fock_matrix_gives_the_identity(self, h2):
        h_mo, eri_mo, n_occ = h2
        rotation, energies = semicanonical_rotation(
            fock_matrix(h_mo, eri_mo, n_occ), n_occ)
        assert np.allclose(rotation, np.eye(h_mo.shape[0]))
        assert np.allclose(energies,
                           np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ))))

    def test_it_block_diagonalizes_and_never_mixes_occupied_with_virtual(self):
        rng = np.random.default_rng(7)
        n, n_occ = 6, 2
        fock = rng.normal(size=(n, n))
        fock = 0.5 * (fock + fock.T)
        fock[:n_occ, n_occ:] = 0.0            # a Fock matrix has no o-v block
        fock[n_occ:, :n_occ] = 0.0
        rotation, energies = semicanonical_rotation(fock, n_occ)
        assert np.allclose(rotation.T @ rotation, np.eye(n), atol=1e-12)
        assert np.allclose(rotation[:n_occ, n_occ:], 0.0)
        assert np.allclose(rotation[n_occ:, :n_occ], 0.0)
        transformed = rotation.T @ fock @ rotation
        assert np.allclose(transformed, np.diag(np.diag(transformed)),
                           atol=1e-10)
        assert np.allclose(np.sort(energies),
                           np.sort(np.linalg.eigvalsh(fock)))


class TestRotatingIntegrals:
    def test_an_orthogonal_rotation_preserves_the_mean_field_energy(self, h2):
        # A rotation inside the occupied block leaves the determinant, and so
        # the Hartree-Fock energy, exactly where it was.
        h_mo, eri_mo, n_occ = h2
        M = h_mo.shape[0]
        rng = np.random.default_rng(11)
        block, _ = np.linalg.qr(rng.normal(size=(M - n_occ, M - n_occ)))
        rotation = np.eye(M)
        rotation[n_occ:, n_occ:] = block
        h_new, eri_new = rotate_integrals(h_mo, eri_mo, rotation)
        # The occupied-occupied part of the Fock energy is untouched.
        assert np.allclose(h_new[:n_occ, :n_occ], h_mo[:n_occ, :n_occ])
        assert np.allclose(fock_matrix(h_new, eri_new, n_occ)[:n_occ, :n_occ],
                           fock_matrix(h_mo, eri_mo, n_occ)[:n_occ, :n_occ])


# --------------------------------------------------------------------------- #
# The correlation energy.
# --------------------------------------------------------------------------- #

class TestTheCorrelationEnergy:
    def test_it_is_negative(self, h2):
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        assert mp2_energy(eri_mo, eps, n_occ) < 0.0

    def test_it_matches_the_spin_orbital_expression(self, h2):
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        reference, _d_oo, _d_vv = _spin_orbital_reference(h_mo, eri_mo, n_occ)
        assert mp2_energy(eri_mo, eps, n_occ) == pytest.approx(reference,
                                                               rel=1e-12)

    def test_it_is_invariant_under_a_virtual_rotation(self, h2):
        # E_MP2 depends on the virtual *space*, not on the basis chosen inside
        # it -- which is exactly what makes a natural-orbital rotation free.
        h_mo, eri_mo, n_occ = h2
        M = h_mo.shape[0]
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        before = mp2_energy(eri_mo, eps, n_occ)
        rng = np.random.default_rng(3)
        block, _ = np.linalg.qr(rng.normal(size=(M - n_occ, M - n_occ)))
        rotation = np.eye(M)
        rotation[n_occ:, n_occ:] = block
        h_new, eri_new = rotate_integrals(h_mo, eri_mo, rotation)
        fock = fock_matrix(h_new, eri_new, n_occ)
        rotated, energies = semicanonical_rotation(fock, n_occ)
        h_new, eri_new = rotate_integrals(h_new, eri_new, rotated)
        after = mp2_energy(eri_new, energies, n_occ)
        assert after == pytest.approx(before, rel=1e-8)

    def test_a_vanishing_gap_is_refused(self, h2):
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ))).copy()
        eps[n_occ] = eps[n_occ - 1] + 0.5 * MIN_DENOMINATOR
        with pytest.raises(ValueError, match="smallest MP2 denominator"):
            mp2_amplitudes(eri_mo, eps, n_occ)

    def test_it_recovers_most_of_the_exact_correlation(self, h2):
        # MP2 is the *selector*, not the answer, so what is pinned here is only
        # that it is the right order of magnitude: a sign error or a factor of
        # two would show, a 5 % inaccuracy is the method working as intended.
        from mandacaru.core.mapping import Fermion
        from mandacaru.core.hamiltonian import spin_block_integrals

        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        h_so, g_so = spin_block_integrals(h_mo, eri_mo, None)
        H = Fermion.from_integrals(h_so, g_so)
        matrix = H.map_to_qubits("jordan_wigner",
                                 n_modes=H.n_modes()).to_sparse_matrix()
        from mandacaru.core.sector import ParticleSector

        sector = ParticleSector(H.n_modes(), (n_occ, n_occ), "jordan_wigner")
        restricted = sector.restrict(
            H.map_to_qubits("jordan_wigner", n_modes=H.n_modes()))
        exact = float(np.min(np.linalg.eigvalsh(restricted.toarray())))
        del matrix
        reference = 2.0 * float(np.sum(eps[:n_occ])) - _mean_field_double_count(
            h_mo, eri_mo, n_occ)
        exact_correlation = exact - reference
        mp2 = mp2_energy(eri_mo, eps, n_occ)
        assert exact_correlation < 0.0
        assert 0.4 < mp2 / exact_correlation < 1.0


def _mean_field_double_count(h_mo, eri_mo, n_occ):
    """``sum_ij (2<ij|ij> - <ij|ji>)``, the term double-counted by ``2 sum eps``."""
    occupied = slice(0, n_occ)
    return float(np.real(
        2.0 * np.einsum("ijij->", eri_mo[occupied, occupied, occupied, occupied])
        - np.einsum("ijji->", eri_mo[occupied, occupied, occupied, occupied])))


# --------------------------------------------------------------------------- #
# The second-order density.
# --------------------------------------------------------------------------- #

class TestTheSecondOrderDensity:
    def test_the_two_blocks_are_traceless_together(self, h2):
        # Second-order correlation moves charge out of the occupied space into
        # the virtual space; it does not create or destroy electrons.  The two
        # blocks are one sum with relabelled dummies, so this is exact.
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        d_oo, d_vv = mp2_density(eri_mo, eps, n_occ)
        assert np.trace(d_oo) + np.trace(d_vv) == pytest.approx(0.0, abs=1e-14)

    def test_the_virtual_block_is_positive_and_the_occupied_one_negative(self, h2):
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        d_oo, d_vv = mp2_density(eri_mo, eps, n_occ)
        assert np.all(np.linalg.eigvalsh(d_vv) > -1e-14)
        assert np.all(np.linalg.eigvalsh(d_oo) < 1e-14)

    def test_both_blocks_are_symmetric(self, h2):
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        d_oo, d_vv = mp2_density(eri_mo, eps, n_occ)
        assert np.allclose(d_oo, d_oo.T)
        assert np.allclose(d_vv, d_vv.T)

    def test_it_matches_the_spin_orbital_expression(self, h2):
        # This is what pins the prefactors: a selection would be blind to a
        # constant factor, an occupation number is not.
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        d_oo, d_vv = mp2_density(eri_mo, eps, n_occ)
        _e, ref_oo, ref_vv = _spin_orbital_reference(h_mo, eri_mo, n_occ)
        assert np.allclose(d_vv, ref_vv, atol=1e-12)
        assert np.allclose(d_oo, ref_oo, atol=1e-12)


# --------------------------------------------------------------------------- #
# The natural orbitals.
# --------------------------------------------------------------------------- #

class TestTheNaturalOrbitals:
    def test_the_rotation_is_orthogonal(self, h2):
        result = mp2_natural_orbitals(*h2)
        M = h2[0].shape[0]
        assert np.allclose(result.rotation.T @ result.rotation, np.eye(M),
                           atol=1e-12)

    def test_it_leaves_the_occupied_block_alone(self, h2):
        # The deliberate half of the frozen-natural-orbital scheme: rotating the
        # occupied block would rename the orbitals `frozen_core="auto"`
        # identifies by index.
        h_mo, _eri, n_occ = h2
        result = mp2_natural_orbitals(*h2)
        assert np.allclose(result.rotation[:n_occ, :n_occ], np.eye(n_occ),
                           atol=1e-12)
        assert np.allclose(result.rotation[:n_occ, n_occ:], 0.0, atol=1e-12)
        assert np.allclose(result.rotation[n_occ:, :n_occ], 0.0, atol=1e-12)

    def test_the_virtual_occupations_are_positive_and_descending(self, h2):
        result = mp2_natural_orbitals(*h2)
        occupations = result.virtual_occupations
        assert np.all(occupations > -1e-14)
        assert np.all(np.diff(occupations) <= 1e-14)

    def test_the_occupied_occupations_are_just_below_two(self, h2):
        result = mp2_natural_orbitals(*h2)
        occupations = result.occupied_occupations
        assert np.all(occupations < 2.0)
        assert np.all(occupations > 1.9)

    def test_the_occupations_sum_to_the_electron_count(self, h2):
        # 2 * n_occ, because the two blocks are traceless together.
        _h, _eri, n_occ = h2
        result = mp2_natural_orbitals(*h2)
        assert float(np.sum(result.occupations)) == pytest.approx(2 * n_occ,
                                                                 abs=1e-12)

    def test_the_rotation_diagonalizes_the_virtual_block(self, h2):
        h_mo, eri_mo, n_occ = h2
        result = mp2_natural_orbitals(*h2)
        h_new, eri_new = rotate_integrals(h_mo, eri_mo, result.rotation)
        fock = fock_matrix(h_new, eri_new, n_occ)
        # The rotated basis is no longer canonical in the virtual block, so the
        # density has to be re-derived semicanonically before comparing; what is
        # pinned is that the *occupations* come back.
        rotated, energies = semicanonical_rotation(fock, n_occ)
        h_new, eri_new = rotate_integrals(h_new, eri_new, rotated)
        _d_oo, d_vv = mp2_density(eri_new, energies, n_occ)
        assert np.allclose(np.sort(np.linalg.eigvalsh(d_vv)),
                           np.sort(result.virtual_occupations), atol=1e-8)

    def test_a_basis_with_no_virtual_orbital_is_refused(self):
        h_mo = np.eye(1)
        eri_mo = np.zeros((1, 1, 1, 1))
        with pytest.raises(ValueError, match="at least one occupied and one"):
            mp2_natural_orbitals(h_mo, eri_mo, 1)

    def test_the_leading_virtual_is_not_always_the_lowest_one(self, h2):
        # The reason this module exists.  If the largest-occupation natural
        # orbital were always the lowest canonical virtual, ranking by orbital
        # energy would be free and correct, and it is not: in a basis with
        # diffuse functions the low virtuals are spatially wrong for
        # correlation, so the leading natural orbital mixes them.
        _h, _eri, n_occ = h2
        result = mp2_natural_orbitals(*h2)
        leading = result.rotation[n_occ:, n_occ]
        assert abs(leading[0]) < 0.99, (
            "the leading natural orbital is the lowest canonical virtual, so "
            "this basis cannot show what the selector is for")


# --------------------------------------------------------------------------- #
# The prefactors, from first principles.
# --------------------------------------------------------------------------- #

class TestThePrefactorsAgainstExactPerturbationTheory:
    r"""The decisive check on ``mp2_density``'s factors of two.

    Two weaker checks come first and neither is enough on its own.
    ``Tr(D_oo) + Tr(D_vv) = 0`` holds *identically* -- the blocks are one sum
    with relabelled dummies -- so it catches an index error and is blind to a
    common factor.  The spin-orbital cross-check above catches a common factor
    but shares an author with the code, so both could carry the same mistake.

    A third check suggests itself and **is wrong**: comparing the MP2
    occupations against the exact 1-RDM occupations from a sector
    diagonalization.  Those disagree by a factor that is not even constant
    (1.5x to 2.6x across the virtuals of H2 / 6-31G), because a raw MP2
    occupation is a second-order quantity and the exact one is not.  That is
    higher-order contamination, not a prefactor error.

    What settles it is the coupling constant.  With :math:`H(\mu) = H_0 + \mu V`
    for :math:`H_0` the diagonal Fock operator and :math:`V` the fluctuation
    potential, Moller-Plesset theory *is* the expansion in :math:`\mu`, so

    .. math::

        \lim_{\mu \to 0}
          \frac{\gamma_{pq}(\mu) - \gamma^{\rm HF}_{pq}}{\mu^2}
        = D^{(2)}_{pq}

    exactly, and :math:`\gamma(\mu)` is available from exact diagonalization.
    Nothing on the right-hand side of that limit shares code with `mp2.py`.
    """

    @staticmethod
    def _scaled_density(h_mo, eri_mo, n_occ, mu):
        """Exact spatial 1-RDM of ``H_0 + mu V``'s ground state, spin-summed."""
        from mandacaru.algorithms.rdm import one_rdm
        from mandacaru.core.hamiltonian import spin_block_integrals
        from mandacaru.core.mapping import Fermion
        from mandacaru.core.sector import ParticleSector

        M = h_mo.shape[0]
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        # H = H_0 + V with H_0 the (diagonal) Fock operator.  Scaling V by mu
        # and the two-body part with it is the Moller-Plesset partition.
        h_scaled = mu * h_mo + (1.0 - mu) * np.diag(eps)
        h_so, g_so = spin_block_integrals(h_scaled, mu * eri_mo, None)
        H = Fermion.from_integrals(h_so, g_so)
        n_modes = H.n_modes()
        sector = ParticleSector(n_modes, (n_occ, n_occ), "jordan_wigner")
        matrix = sector.restrict(
            H.map_to_qubits("jordan_wigner", n_modes=n_modes)).toarray()
        values, vectors = np.linalg.eigh(matrix)
        psi = vectors[:, int(np.argmin(values))]
        gamma = one_rdm(psi, n_modes, "jordan_wigner", sector)
        # Spin-sum the spin-blocked RDM back to spatial orbitals.
        return np.real(gamma[:M, :M] + gamma[M:, M:])

    def test_the_second_order_density_is_the_mu_squared_coefficient(self, h2):
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        d_oo, d_vv = mp2_density(eri_mo, eps, n_occ)
        reference = np.zeros_like(np.real(h_mo))
        for i in range(n_occ):
            reference[i, i] = 2.0

        errors = []
        for mu in (0.2, 0.1, 0.05):
            gamma = self._scaled_density(h_mo, eri_mo, n_occ, mu)
            coefficient = (gamma - reference) / mu ** 2
            errors.append(max(
                float(np.max(np.abs(coefficient[:n_occ, :n_occ] - d_oo))),
                float(np.max(np.abs(coefficient[n_occ:, n_occ:] - d_vv)))))

        # What to assert is the *rate*, not a tolerance.  The residual is the
        # first neglected order, O(mu), so halving mu must halve it -- measured
        # 1.98e-3, 9.36e-4, 4.55e-4 at mu = 0.2, 0.1, 0.05, ratios 0.47 and
        # 0.49.  A wrong prefactor leaves a *constant* discrepancy instead: the
        # error would plateau rather than converge, however small mu got, which
        # is a qualitative difference no choice of tolerance is needed to see.
        ratios = [errors[i + 1] / errors[i] for i in range(len(errors) - 1)]
        assert all(0.4 < r < 0.6 for r in ratios), (errors, ratios)

    def test_a_doubled_density_would_not_survive_the_same_limit(self, h2):
        # The guard on the guard: if the limit above were loose enough to accept
        # any prefactor, this would pass too.  It must not.
        h_mo, eri_mo, n_occ = h2
        eps = np.real(np.diag(fock_matrix(h_mo, eri_mo, n_occ)))
        d_oo, d_vv = mp2_density(eri_mo, eps, n_occ)
        gamma = self._scaled_density(h_mo, eri_mo, n_occ, 0.05)
        reference = np.zeros_like(np.real(h_mo))
        for i in range(n_occ):
            reference[i, i] = 2.0
        coefficient = (gamma - reference) / 0.05 ** 2
        right = float(np.max(np.abs(coefficient[n_occ:, n_occ:] - d_vv)))
        doubled = float(np.max(np.abs(coefficient[n_occ:, n_occ:] - 2.0 * d_vv)))
        assert doubled > 10.0 * right, (right, doubled)
