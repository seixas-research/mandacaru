# -*- coding: utf-8 -*-
# file: test/core/test_spin_orbit.py

"""Spin-orbit coupling as a one-body term over spin-orbitals.

``L . S`` has a spectrum that is known exactly -- ``l/2`` on the ``2l + 2``
states of ``j = l + 1/2`` and ``-(l+1)/2`` on the ``2l`` states of
``j = l - 1/2`` -- so both the bare operator and the assembled matrix can be
checked against it rather than against a tolerance.  The other thing worth
pinning is what the term *costs*: it commutes with ``J_z`` and not with
``S_z``, which is exactly why the sector reduction and the excitation pools
cannot be left to assume a fixed ``(n_alpha, n_beta)``.
"""

import numpy as np
import pytest

from mandacaru.core.spin_orbit import (ALPHA, BETA, breaks_spin_symmetry,
                                       ls_matrix, spin_orbit_one_body)


class Projector:
    """The four labels the assembly reads off a KBProjector."""

    def __init__(self, atom_index, l, m, index):
        self.atom_index, self.l, self.m, self.index = atom_index, l, m, index


def channel(atom=0, l=1, n_radial=2):
    return [Projector(atom, l, m, i)
            for m in range(-l, l + 1) for i in range(n_radial)]


def angular_momentum_operators(projectors):
    """``(L_z, S_z)`` over the spin-orbital ordering ``P = p + sigma M``."""
    M = len(projectors)
    m_of = np.array([p.m for p in projectors], dtype=float)
    Lz = np.diag(np.concatenate([m_of, m_of]))
    Sz = np.diag(np.concatenate([np.full(M, 0.5), np.full(M, -0.5)]))
    return Lz, Sz


class TestTheBareOperator:
    @pytest.mark.parametrize("l", [1, 2, 3, 4])
    def test_its_spectrum_is_the_two_j_levels(self, l):
        eigenvalues = np.linalg.eigvalsh(ls_matrix(l))
        up, down = l / 2.0, -(l + 1) / 2.0
        assert np.sum(np.abs(eigenvalues - up) < 1e-12) == 2 * l + 2
        assert np.sum(np.abs(eigenvalues - down) < 1e-12) == 2 * l

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_it_is_hermitian_and_traceless(self, l):
        operator = ls_matrix(l)
        assert np.array_equal(operator, operator.conj().T)
        assert np.trace(operator) == pytest.approx(0.0, abs=1e-14)

    def test_an_s_channel_has_no_spin_orbit_coupling(self):
        assert np.count_nonzero(ls_matrix(0)) == 0

    @pytest.mark.parametrize("l", [1, 2])
    def test_it_conserves_jz_but_not_sz(self, l):
        operator = ls_matrix(l)
        size = 2 * l + 1
        m = np.arange(-l, l + 1, dtype=float)
        Lz = np.diag(np.concatenate([m, m]))
        Sz = np.diag(np.concatenate([np.full(size, 0.5), np.full(size, -0.5)]))
        Jz = Lz + Sz
        assert np.allclose(operator @ Jz, Jz @ operator, atol=1e-14)
        assert not np.allclose(operator @ Sz, Sz @ operator, atol=1e-9)

    def test_the_ladder_element_has_the_textbook_value(self):
        """<m+1, beta| L.S |m, alpha> = sqrt(l(l+1) - m(m+1)) / 2."""
        l, m = 2, 0
        size = 2 * l + 1
        element = ls_matrix(l)[BETA * size + (m + 1 + l), ALPHA * size + (m + l)]
        assert element == pytest.approx(
            0.5 * np.sqrt(l * (l + 1) - m * (m + 1)))


class TestTheAssembledOneBodyMatrix:
    def test_it_reproduces_the_j_splitting_of_its_coupling(self):
        """With the basis spanning the projectors, each eigenvalue d of D
        contributes d*l/2 and -d*(l+1)/2 with the j multiplicities."""
        l, projectors = 1, channel(l=1)
        M = len(projectors)
        D = np.diag([1.0, 0.3])
        h = spin_orbit_one_body(np.eye(M, dtype=complex), projectors,
                                {(0, l): D})
        eigenvalues = np.linalg.eigvalsh(h)
        for d in (1.0, 0.3):
            up, down = d * l / 2.0, -d * (l + 1) / 2.0
            assert np.sum(np.abs(eigenvalues - up) < 1e-9) == 2 * l + 2
            assert np.sum(np.abs(eigenvalues - down) < 1e-9) == 2 * l

    def test_it_is_hermitian_traceless_and_spin_coupling(self):
        projectors = channel()
        M = len(projectors)
        h = spin_orbit_one_body(np.eye(M, dtype=complex), projectors,
                                {(0, 1): np.diag([1.0, 0.3])})
        assert np.allclose(h, h.conj().T, atol=1e-14)
        assert np.trace(h) == pytest.approx(0.0, abs=1e-12)
        assert breaks_spin_symmetry(h)

    def test_it_conserves_jz(self):
        projectors = channel()
        M = len(projectors)
        h = spin_orbit_one_body(np.eye(M, dtype=complex), projectors,
                                {(0, 1): np.diag([1.0, 0.3])})
        Lz, Sz = angular_momentum_operators(projectors)
        assert np.allclose(h @ (Lz + Sz), (Lz + Sz) @ h, atol=1e-12)
        assert not np.allclose(h @ Sz, Sz @ h, atol=1e-9)

    def test_no_blocks_gives_the_zero_matrix_of_the_right_shape(self):
        projectors = channel()
        M = len(projectors)
        h = spin_orbit_one_body(np.eye(M, dtype=complex), projectors, {})
        assert h.shape == (2 * M, 2 * M)
        assert not np.any(h)
        assert not breaks_spin_symmetry(h)

    def test_an_s_channel_contributes_nothing(self):
        projectors = channel(l=0, n_radial=2)
        M = len(projectors)
        h = spin_orbit_one_body(np.eye(M, dtype=complex), projectors,
                                {(0, 0): np.diag([5.0, 5.0])})
        assert not np.any(h)
        assert not breaks_spin_symmetry(h)

    def test_two_atoms_contribute_independently(self):
        projectors = channel(atom=0) + channel(atom=1)
        M = len(projectors)
        C = np.eye(M, dtype=complex)
        D = np.diag([1.0, 0.3])
        both = spin_orbit_one_body(C, projectors, {(0, 1): D, (1, 1): D})
        first = spin_orbit_one_body(C, projectors, {(0, 1): D})
        second = spin_orbit_one_body(C, projectors, {(1, 1): D})
        assert np.allclose(both, first + second, atol=1e-12)

    def test_a_mismatched_block_is_refused_by_shape(self):
        projectors = channel()
        M = len(projectors)
        with pytest.raises(ValueError, match="must be"):
            spin_orbit_one_body(np.eye(M, dtype=complex), projectors,
                                {(0, 1): np.eye(3)})

    def test_it_scales_linearly_with_the_coupling(self):
        projectors = channel()
        M = len(projectors)
        C = np.eye(M, dtype=complex)
        D = np.diag([1.0, 0.3])
        assert np.allclose(spin_orbit_one_body(C, projectors, {(0, 1): 2 * D}),
                           2 * spin_orbit_one_body(C, projectors,
                                                   {(0, 1): D}), atol=1e-12)


class TestTheIntegralsHook:
    def test_an_ordinary_molecule_carries_no_spin_orbit_term(self):
        """Nothing that did not ask for it may acquire an alpha-beta block."""
        from mandacaru.core.hamiltonian import MolecularIntegrals

        assert not MolecularIntegrals.has_spin_orbit.fget(
            type("Fake", (), {"spin_orbit_coupling": {}})())

    def test_spin_orbit_blocks_need_projectors_to_act_on(self):
        from mandacaru.core.hamiltonian import (MolecularIntegrals,
                                                 minimal_hao_basis)
        from mandacaru.integrals.grid import Grid

        nuclei = [(1.0, np.zeros(3))]
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.4)
        with pytest.raises(ValueError, match="needs projectors"):
            MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid,
                               spin_orbit_coupling={(0, 1): np.eye(2)})

    def test_an_ordinary_molecule_gets_a_zero_spin_orbit_matrix(self):
        from mandacaru.core.hamiltonian import (MolecularIntegrals,
                                                 minimal_hao_basis)
        from mandacaru.integrals.grid import Grid

        nuclei = [(1.0, np.zeros(3))]
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=5.0, h=0.4)
        integrals = MolecularIntegrals(nuclei, minimal_hao_basis(nuclei), grid)
        assert not integrals.has_spin_orbit
        h = integrals.spin_orbit_matrix()
        assert h.shape == (2 * integrals.n_orbitals,) * 2
        assert not np.any(h)
