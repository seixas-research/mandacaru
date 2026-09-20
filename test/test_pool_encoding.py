# -*- coding: utf-8 -*-
# file: test/test_pool_encoding.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Every pool is built in the encoding the Hamiltonian uses.

The qubit-excitation pools (``qeb``, ``ceo``) are constructed from the
encoding's own update and flip sets (:func:`mandacaru.core.mapping.qubit_excitation`),
not by deleting Jordan-Wigner ``Z`` strings: reusing the JW strings under
parity or Bravyi-Kitaev gives operators that no longer commute with the mapped
number operator, and the ansatz would leave the physical sector.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru
from mandacaru.circuits.pools import build_pool
from mandacaru.core.mapping import Fermion, PauliSum, qubit_excitation
from mandacaru.core.sector import ParticleSector
from mandacaru.units import HARTREE_TO_EV

MAPPINGS = ["jordan_wigner", "parity", "bravyi_kitaev"]
CONSERVING = ["fermionic", "qeb", "ceo"]


def number_operator(n_modes: int, mapping: str) -> PauliSum:
    """``N = sum_p a_p^dagger a_p`` in the requested encoding."""
    terms = {((p, True), (p, False)): 1.0 for p in range(n_modes)}
    return Fermion(terms, n_modes=n_modes).map_to_qubits(mapping,
                                                         n_modes=n_modes)


def spin_z_operator(n_modes: int, mapping: str) -> PauliSum:
    """``Sz = (N_alpha - N_beta)/2`` in the requested encoding."""
    half = n_modes // 2
    terms = {((p, True), (p, False)): (0.5 if p < half else -0.5)
             for p in range(n_modes)}
    return Fermion(terms, n_modes=n_modes).map_to_qubits(mapping,
                                                         n_modes=n_modes)


def commutator_norm(a: PauliSum, b: PauliSum) -> float:
    """Frobenius norm of ``[A, B]`` through the sparse matrices."""
    ma, mb = a.to_sparse_matrix(), b.to_sparse_matrix()
    return float(np.sqrt(abs((ma @ mb - mb @ ma).power(2).sum())))


class TestSymmetryInEveryEncoding:
    @pytest.mark.parametrize("mapping", MAPPINGS)
    @pytest.mark.parametrize("name", CONSERVING)
    def test_generators_conserve_particle_number_and_spin(self, mapping, name):
        pool = build_pool(name, 3, (1, 1), mapping=mapping)
        N = number_operator(6, mapping)
        Sz = spin_z_operator(6, mapping)
        for op in pool.operators():
            assert commutator_norm(op.generator, N) < 1e-10, op.label
            assert commutator_norm(op.generator, Sz) < 1e-10, op.label

    @pytest.mark.parametrize("mapping", MAPPINGS)
    def test_the_qubit_pool_breaks_it_on_purpose(self, mapping):
        """Individual Pauli strings are not number conserving in any encoding."""
        pool = build_pool("qubit", 3, (1, 1), mapping=mapping)
        assert pool.conserves_particle_number is False
        N = number_operator(6, mapping)
        assert max(commutator_norm(op.generator, N)
                   for op in pool.operators()) > 1.0

    @pytest.mark.parametrize("mapping", MAPPINGS)
    @pytest.mark.parametrize("name", ["fermionic", "qeb", "ceo", "qubit"])
    def test_every_pool_builds_in_every_encoding(self, mapping, name):
        assert build_pool(name, 3, (1, 1), mapping=mapping).operators()


class TestJordanWignerIsUnchanged:
    """The general construction must reproduce the JW operators exactly."""

    def test_single_matches_the_textbook_form(self):
        # (i/2)(X_0 Y_1 - Y_0 X_1) on the two involved qubits.
        generator = qubit_excitation((1,), (0,), 4, "jordan_wigner")
        assert generator.terms == pytest.approx(
            {"XYII": -0.5j, "YXII": 0.5j})

    def test_double_has_the_eight_quarter_terms(self):
        generator = qubit_excitation((1, 3), (2, 0), 4, "jordan_wigner")
        assert len(generator.terms) == 8
        assert all(abs(abs(c) - 0.125) < 1e-12
                   for c in generator.terms.values())
        assert all(abs(c.real) < 1e-12 for c in generator.terms.values())


class TestCEOGrouping:
    def test_ceo_equals_qeb_in_jordan_wigner(self):
        """One excitation per support there, so every group is a singleton."""
        qeb = build_pool("qeb", 3, (1, 1)).operators()
        ceo = build_pool("ceo", 3, (1, 1)).operators()
        assert ([op.generator.terms for op in qeb]
                == [op.generator.terms for op in ceo])

    @pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
    def test_ceo_genuinely_groups_in_other_encodings(self, mapping):
        """Wider update/flip sets make distinct excitations share a support.

        Each CEO generator must be exactly the sum of the QEB generators on its
        support, and at least one group must hold more than one of them.
        """
        qeb = build_pool("qeb", 3, (1, 1), mapping=mapping).operators()
        ceo = build_pool("ceo", 3, (1, 1), mapping=mapping).operators()
        assert len(ceo) < len(qeb)

        members = 0
        for group in ceo:
            summed = PauliSum(num_qubits=6)
            count = 0
            for op in qeb:
                if op.support == group.support:
                    summed = summed + op.generator
                    count += 1
            members = max(members, count)
            summed = summed.simplify()
            assert set(summed.terms) == set(group.generator.terms)
            for label, coeff in summed.terms.items():
                assert group.generator.terms[label] == pytest.approx(coeff)
        assert members > 1


class TestSamePhysicsEverywhere:
    """H2 must reach the same ground state whatever pool and encoding."""

    @staticmethod
    def run(mapping, pool, reduction=False):
        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0] * 3)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.4,
                               pool=pool, mapping=("parity_reduced"
                                                   if reduction else mapping),
                               trace=False,
                               profile=False, gradient_tolerance=1e-7,
                               max_iterations=12)
        energy = atoms.get_potential_energy()
        matrix = atoms.calc._h_matrix
        exact = np.linalg.eigvalsh(
            matrix if isinstance(matrix, np.ndarray) else matrix.toarray())[0]
        return energy, exact * HARTREE_TO_EV

    @pytest.mark.parametrize("mapping", MAPPINGS)
    @pytest.mark.parametrize("pool", ["qeb", "ceo"])
    def test_reaches_the_exact_energy(self, mapping, pool):
        energy, exact = self.run(mapping, pool)
        assert energy == pytest.approx(exact, abs=1e-6)

    @pytest.mark.parametrize("pool", ["qeb", "ceo"])
    def test_the_two_qubit_reduction_applies_to_qubit_excitations(self, pool):
        """They commute with both tapered symmetries, so tapering is exact."""
        assert build_pool(pool, 2, (1, 1),
                          mapping="parity_reduced").n_qubits == 2
        energy, exact = self.run("parity", pool, reduction=True)
        assert energy == pytest.approx(exact, abs=1e-6)

    def test_the_qubit_pool_cannot_be_tapered(self):
        with pytest.raises(ValueError, match="do not commute"):
            build_pool("qubit", 2, (1, 1), mapping="parity_reduced")


class TestSectorLeakage:
    """Restricting an operator that leaves the sector would change the ansatz."""

    @pytest.mark.parametrize("mapping", MAPPINGS)
    def test_qubit_strings_leak_while_qubit_excitations_do_not(self, mapping):
        sector = ParticleSector(6, (1, 1), mapping)
        qubit = build_pool("qubit", 3, (1, 1), mapping=mapping).operators()
        qeb = build_pool("qeb", 3, (1, 1), mapping=mapping).operators()
        assert not all(sector.conserves(op.generator) for op in qubit)
        assert all(sector.conserves(op.generator) for op in qeb)

    def test_conserves_needs_a_matching_register(self):
        sector = ParticleSector(6, (1, 1))
        with pytest.raises(ValueError, match="qubits"):
            sector.conserves(PauliSum({"XX": 1.0}))

    def test_cancelling_terms_still_count_as_conserving(self):
        sector = ParticleSector(4, (1, 1))
        leaking = PauliSum({"XIII": 1j})
        assert not sector.conserves(leaking)
        assert sector.conserves(leaking + PauliSum({"XIII": -1j}))
