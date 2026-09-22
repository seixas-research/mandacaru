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
from mandacaru.optimizers import Optimizer
from mandacaru.backends.providers import (QiskitProvider, pauli_rotations, pauli_strings_commute)
from mandacaru.core import WavefunctionCheckpoint

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
COBYLA_OPT = Optimizer(method="COBYLA", maxiter=2000, tol=1e-12)

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
    """The CEO pool of Ramoa et al., npj Quantum Inf. 11, 86 (2025), Sec. II B.

    A CEO couples the qubit excitations acting on one set of spin-orbitals.
    Before 2026-09-20 the pool summed QEB generators grouped by their *mapped*
    support and, with the occupied-to-virtual enumeration, every group was a
    singleton: ``ceo`` was ``qeb`` under Jordan-Wigner and an encoding artifact
    under parity / Bravyi-Kitaev.  These tests pin the paper's construction.
    """

    def test_the_paper_equations(self):
        """Eqs. (11), (12), (25) and (26), sign for sign.

        Four qubits ordered alpha2 alpha1 beta2 beta1 from most to least
        significant, as the paper sets up before Eq. (11).
        """
        a2, a1, b2, b1 = 0, 1, 2, 3
        t1 = qubit_excitation((a2, b2), (a1, b1), 4, "jordan_wigner")
        t2 = qubit_excitation((a1, b2), (a2, b1), 4, "jordan_wigner")

        def pattern(op):
            return {k: ("+" if (complex(v) / 1j).real > 0 else "-")
                    for k, v in op.simplify().terms.items()}

        assert pattern(t1) == {"XXXY": "+", "XXYX": "-", "XYXX": "+",
                               "XYYY": "+", "YXXX": "-", "YXYY": "-",
                               "YYXY": "+", "YYYX": "-"}          # Eq. (11)
        assert pattern(t2) == {"XXXY": "+", "XXYX": "-", "XYXX": "-",
                               "XYYY": "-", "YXXX": "+", "YXYY": "+",
                               "YYXY": "+", "YYYX": "-"}          # Eq. (12)
        # Both QEs carry the same eight strings with coefficient 1/8, which is
        # why coupling them costs no extra entangling structure.
        assert all(abs(complex(v)) == pytest.approx(0.125)
                   for v in t1.simplify().terms.values())
        assert pattern(t1 + t2) == {"XXXY": "+", "XXYX": "-",
                                    "YYXY": "+", "YYYX": "-"}     # Eq. (25)
        assert pattern(t1 + t2 * -1.0) == {"XYXX": "+", "XYYY": "+",
                                           "YXXX": "-", "YXYY": "-"}  # Eq. (26)

    def test_ceo_is_not_qeb(self):
        """The whole point: a larger pool built from generalized excitations."""
        qeb = build_pool("qeb", 3, (1, 1))
        ceo = build_pool("ceo", 3, (1, 1))
        assert len(ceo) > len(qeb)
        assert ([op.generator.simplify().terms for op in ceo.operators()]
                != [op.generator.simplify().terms for op in qeb.operators()])

    @pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                         "bravyi_kitaev"])
    def test_sizes_follow_the_construction(self, mapping):
        """2 CEOs per opposite-spin set, 6 per same-spin set, 1 per single.

        Of the three ways to pair four spin-orbitals only the S_z-conserving
        ones are excitations: two when the set holds two alpha and two beta
        orbitals, three when all four share a spin.  Each set then contributes
        one CEO per ordered pair of its excitations and sign.
        """
        from math import comb
        for M in (2, 3, 4):
            pool = build_pool("ceo", M, (1, 1), mapping=mapping)
            singles = 2 * comb(M, 2)
            opposite = comb(M, 2) ** 2 * 2
            same = 2 * comb(M, 4) * 6
            assert len(pool) == singles + opposite + same

    def test_every_operator_couples_excitations_on_one_orbital_set(self):
        ceo = build_pool("ceo", 3, (1, 1)).operators()
        for op in ceo:
            assert 1 <= len(op.members) <= 2
            summed = PauliSum(num_qubits=6)
            for member, sign in zip(op.members, (1.0, 1.0)):
                summed = summed + member.generator * sign
            difference = PauliSum(num_qubits=6)
            for member, sign in zip(op.members, (1.0, -1.0)):
                difference = difference + member.generator * sign
            target = op.generator.simplify().terms
            assert (target == summed.simplify().terms
                    or target == difference.simplify().terms)

    def test_a_coupled_double_has_four_pauli_strings(self):
        """Eqs. (25)-(26): four strings where a QE needs eight.

        This is what the paper's 9-CNOT circuit exploits, and what makes the
        generic compilation cheaper here too.
        """
        ceo = build_pool("ceo", 2, (1, 1)).operators()
        doubles = [op for op in ceo if len(op.support) == 4]
        assert doubles and all(len(op.generator.simplify().terms) == 4
                               for op in doubles)

    def test_coupled_excitations_commute(self):
        """So appending them in sequence is exactly ``exp(sum_i theta_i T_i)``.

        The growth step relies on this to realize an MVP-CEO without a
        multi-parameter block in the ansatz.
        """
        for op in build_pool("ceo", 3, (1, 1)).operators():
            for x in range(len(op.members)):
                for y in range(x + 1, len(op.members)):
                    A = op.members[x].matrix()
                    B = op.members[y].matrix()
                    assert np.abs(A @ B - B @ A).max() == pytest.approx(0.0,
                                                                        abs=1e-12)

    @pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                         "bravyi_kitaev"])
    def test_generators_conserve_the_particle_numbers(self, mapping):
        """A CEO is a sum of qubit excitations, so it keeps N and S_z."""
        pool = build_pool("ceo", 3, (1, 1), mapping=mapping)
        sector = ParticleSector(pool.n_qubits, (1, 1), mapping)
        for op in pool.operators():
            assert sector.conserves(op.generator)

    def test_growth_expands_only_when_two_excitations_are_live(self):
        """The paper's modified step 3."""
        pool = build_pool("ceo", 3, (1, 1))
        coupled = next(op for op in pool.operators() if len(op.members) == 2)
        first, second = coupled.members

        assert pool.grown_operators(coupled, lambda o: 0.0) == [coupled]
        assert pool.grown_operators(
            coupled, lambda o: 1.0 if o is first else 0.0) == [coupled]
        assert pool.grown_operators(coupled, lambda o: 1.0) == [first, second]

        single = next(op for op in pool.operators() if len(op.members) == 1)
        assert pool.grown_operators(single, lambda o: 1.0) == [single]

    def test_ovp_variant_is_the_same_pool_without_the_mvp_expansion(self):
        """``ceo-ovp``: the paper's one-parameter variant (Supplementary I).

        Mandacaru has no MVP circuit synthesis, so an expanded step compiles as
        separate eight-string excitations and costs gates; keeping the
        one-parameter form is what actually halves the CNOT count here.
        """
        ceo = build_pool("ceo", 3, (2, 2))
        ovp = build_pool("ceo-ovp", 3, (2, 2))
        assert [op.label for op in ovp.operators()] == [op.label
                                                        for op in ceo.operators()]
        coupled = next(op for op in ovp.operators() if len(op.members) == 2)
        assert ovp.grown_operators(coupled, lambda o: 1.0) == [coupled]
        assert len(ceo.grown_operators(coupled, lambda o: 1.0)) == 2

    def test_ovp_halves_the_gate_count_of_qeb(self):
        """The reason ``examples/24_ADAPTVQE_LiH_IBM.py`` uses it."""
        atoms = Atoms("LiH", positions=[[7.5, 7.5, 6.7], [7.5, 7.5, 8.3]],
                      cell=[15.0] * 3, pbc=True)
        counts = {}
        for pool in ("qeb", "ceo-ovp"):
            work = atoms.copy()
            work.calc = Mandacaru(method="adapt-vqe", pool=pool,
                                  mapping="jordan_wigner",
                                  basis={"name": "GTO", "n_gaussians": 3},
                                  h=0.15, optimizer=COBYLA_OPT,
                                  max_iterations=14, gradient_tolerance=1e-3,
                                  profile=True, trace=False)
            energy = work.get_potential_energy()
            counts[pool] = (energy, work.calc.result.metrics.cnot_count)
        assert counts["ceo-ovp"][0] == pytest.approx(counts["qeb"][0], abs=1e-3)
        assert counts["ceo-ovp"][1] < 0.6 * counts["qeb"][1]


class TestSamePhysicsEverywhere:
    """H2 must reach the same ground state whatever pool and encoding."""

    @staticmethod
    def run(mapping, pool, reduction=False):
        atoms = Atoms("H2", positions=[[4, 4, 3.63], [4, 4, 4.37]],
                      cell=[8.0] * 3)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.4,
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

    def test_canceling_terms_still_count_as_conserving(self):
        sector = ParticleSector(4, (1, 1))
        leaking = PauliSum({"XIII": 1j})
        assert not sector.conserves(leaking)
        assert sector.conserves(leaking + PauliSum({"XIII": -1j}))


class TestCEOLabels:
    def test_ceo_labels_unique_and_descriptive(self):
        from mandacaru.circuits import build_pool
        for name in ("ceo", "ceo-ovp"):
            labels = [op.label for op in build_pool(name, 3, (2, 2)).operators()]
            # No collisions, and each label names the spin-orbital set it
            # couples plus the excitations it combines -- the grouping key is
            # the orbital set, so the label says "o", not "q".
            assert len(labels) == len(set(labels))
            assert all(lbl.startswith("CEO[o") for lbl in labels)
            assert all("{" in lbl and lbl.endswith("}") for lbl in labels)


class TestCommutingTermsOnly:
    def test_the_pairwise_rule(self):
        assert pauli_strings_commute("XX", "YY")           # two clashes
        assert not pauli_strings_commute("XI", "ZI")       # one clash
        assert pauli_strings_commute("XI", "IZ")           # disjoint

    def test_an_anticommuting_generator_is_refused(self):
        generator = PauliSum({"X": 0.3j, "Z": 0.2j})
        with pytest.raises(ValueError, match="anticommute"):
            pauli_rotations(generator)
        record = WavefunctionCheckpoint(1, [], [generator], [1.7])
        with pytest.raises(ValueError, match="anticommute"):   # fidelity 0.983
            QiskitProvider().statevector(*record.problem()[:4])
        assert np.linalg.norm(record.state_vector()) == pytest.approx(1.0)

    @pytest.mark.parametrize("mapping", ["jordan_wigner", "parity",
                                         "bravyi_kitaev"])
    @pytest.mark.parametrize("pool", ["fermionic", "qubit", "qeb"])
    def test_these_pools_are_exportable_in_every_mapping(self, pool, mapping):
        for operator in build_pool(pool, 3, (2, 1), mapping=mapping).operators():
            assert pauli_rotations(operator.generator)

    def test_every_pool_generator_is_exportable(self):
        """The guard's original case is gone: CEO generators now commute.

        When this check was written, ``ceo`` summed qubit excitations that
        happened to share a *mapped* support, and under parity / Bravyi-Kitaev
        those anticommute, so a circuit could not realize them.  The pool now
        couples excitations on one set of *spin-orbitals*, which commute in
        every encoding, so nothing is refused -- see
        ``test_pool_encoding.TestCEOGrouping``.
        """
        for mapping in ("jordan_wigner", "parity", "bravyi_kitaev"):
            for name in ("fermionic", "qeb", "ceo", "qubit"):
                for operator in build_pool(name, 3, (2, 1),
                                           mapping=mapping).operators():
                    assert pauli_rotations(operator.generator)

    def test_non_commuting_terms_are_still_refused(self):
        """The guard itself must stay: a product of rotations is not the
        exponential of a sum unless the terms commute."""
        # XY and YX differ on both qubits, so they commute; XX and XY differ on
        # one, so they anticommute.
        assert pauli_rotations(PauliSum({"XY": 0.5j, "YX": 0.5j}))
        with pytest.raises(ValueError, match="commut"):
            pauli_rotations(PauliSum({"XX": 0.5j, "XY": 0.5j}))
