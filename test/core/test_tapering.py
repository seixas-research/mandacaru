# -*- coding: utf-8 -*-
# file: test/core/test_tapering.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Z\ :sub:`2` symmetry tapering: one qubit per conserved parity.

The test that matters is that the **ground-state energy survives**.  Everything
else -- how many qubits come off, which strings the symmetry finder returns --
is bookkeeping that could be wrong in a way no eigenvalue notices, so the
eigenvalue is checked directly against the untapered operator on real molecular
Hamiltonians.

The independent landmark is H2 in a minimal basis, which is known to reduce to a
**single** qubit under the full Z2 group; anything else would mean the finder is
missing symmetries or inventing them.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mandacaru.basis import BasisSet
from mandacaru.core import MolecularIntegrals
from mandacaru.core.mapping import PauliSum
from mandacaru.core.tapering import (SymmetryLeakError, leaking_terms,
                                     reduce_generators,
                                     rotate_to_z_type, sector_signs,
                                     symmetry_generators, symplectic_form,
                                     taper, taper_problem,
                                     tapered_reference_bits)
from mandacaru.integrals import Grid


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #

def molecular_pauli(symbols, positions, basis, n_electrons, num_particles,
                    box=8.0, h=0.35):
    """``(PauliSum, reference_bits, num_particles)`` for a real molecule."""
    atoms = Atoms(symbols, positions=positions)
    coordinates = atoms.get_positions()
    bset = BasisSet.build(basis)
    functions, nuclei = [], []
    for Z, symbol, position in zip(atoms.get_atomic_numbers(),
                                   atoms.get_chemical_symbols(), coordinates):
        functions += bset.atom(symbol, center=position, units="angstrom")
        nuclei.append((float(Z), position))
    grid = Grid(center=coordinates.mean(axis=0), box_size=box, h=h,
                units="angstrom")
    integrals = MolecularIntegrals(
        nuclei, functions, grid,
        softening=0.5 * min(grid.dx, grid.dy, grid.dz))
    fermion = integrals.molecular_hamiltonian(
        mo_basis=True, n_electrons=n_electrons, num_particles=num_particles)
    n_modes = fermion.n_modes()
    operator = fermion.map_to_qubits("jordan_wigner",
                                     n_modes=n_modes).simplify()
    # Jordan-Wigner reference: the lowest orbitals of each spin block.
    n_alpha, n_beta = num_particles
    spatial = n_modes // 2
    bits = [0] * n_modes
    for k in range(n_alpha):
        bits[k] = 1
    for k in range(n_beta):
        bits[spatial + k] = 1
    return operator, bits


def ground_energy(operator: PauliSum) -> float:
    return float(np.min(np.linalg.eigvalsh(operator.to_matrix())))


@pytest.fixture(scope="module")
def h2_minimal():
    return molecular_pauli("H2", [(0, 0, 0), (0, 0, 0.74)], "STO-3G", 2, (1, 1))


@pytest.fixture(scope="module")
def h2_split():
    return molecular_pauli("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G", 2, (1, 1))


@pytest.fixture(scope="module")
def lih_minimal():
    return molecular_pauli("LiH", [(0, 0, 0), (0, 0, 1.60)], "STO-3G", 4,
                           (2, 2))


# --------------------------------------------------------------------------- #
# The symplectic bookkeeping.
# --------------------------------------------------------------------------- #

class TestTheSymplecticForm:
    def test_it_reads_the_supports_off_the_labels(self):
        operator = PauliSum({"XYZI": 1 + 0j})
        x, z = symplectic_form(operator)
        assert list(x[0]) == [1, 1, 0, 0]          # X and Y carry X support
        assert list(z[0]) == [0, 1, 1, 0]          # Y and Z carry Z support

    def test_an_empty_operator_has_no_rows(self):
        x, z = symplectic_form(PauliSum({}, num_qubits=4))
        assert x.size == 0 and z.size == 0


class TestReducingTheGenerators:
    def test_each_reduced_generator_owns_its_anchor(self):
        # The property the Cliffords depend on: tau_j acts on q_j and no other
        # generator does.  Without it the conjugation is not unitary or does not
        # commute with the other generators, and the first version of the module
        # got this wrong in exactly that way.
        reduced, anchors = reduce_generators(["ZZII", "IZZI", "ZIZI"])
        supports = [[1 if c == "Z" else 0 for c in tau] for tau in reduced]
        assert len(set(anchors)) == len(anchors)
        for j, q in enumerate(anchors):
            assert supports[j][q] == 1
            for i, other in enumerate(supports):
                if i != j:
                    assert other[q] == 0

    def test_it_drops_a_dependent_generator(self):
        # ZZII * IZZI = ZIZI, so only two of the three are independent.
        reduced, anchors = reduce_generators(["ZZII", "IZZI", "ZIZI"])
        assert len(reduced) == len(anchors) == 2

    def test_no_generators_is_no_anchors(self):
        assert reduce_generators([]) == ([], [])


# --------------------------------------------------------------------------- #
# The symmetries themselves.
# --------------------------------------------------------------------------- #

class TestFindingTheSymmetries:
    def test_every_generator_commutes_with_every_term(self, h2_minimal):
        operator, _bits = h2_minimal
        generators = symmetry_generators(operator)
        assert generators
        for tau in generators:
            pauli = PauliSum({tau: 1 + 0j})
            commutator = pauli.compose(operator).simplify()
            reverse = operator.compose(pauli).simplify()
            for label in set(commutator.terms) | set(reverse.terms):
                assert commutator.terms.get(label, 0j) == pytest.approx(
                    reverse.terms.get(label, 0j), abs=1e-12)

    def test_an_operator_with_no_symmetry_gives_none(self):
        # A single X on every qubit leaves no Z-type operator commuting with it.
        operator = PauliSum({"XX": 1 + 0j, "XI": 0.5 + 0j, "IX": 0.25 + 0j})
        assert symmetry_generators(operator) == []

    def test_a_diagonal_operator_is_symmetric_under_every_z(self):
        operator = PauliSum({"ZZ": 1 + 0j, "ZI": 0.5 + 0j})
        generators = symmetry_generators(operator)
        # Two independent Z-type generators on two qubits.
        assert len(reduce_generators(generators)[0]) == 2


# --------------------------------------------------------------------------- #
# The Clifford rotation.
# --------------------------------------------------------------------------- #

class TestTheRotation:
    def test_it_sends_each_generator_to_its_anchor_x(self, h2_minimal):
        operator, _bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        for tau, qubit in zip(reduced, anchors):
            n = len(tau)
            expected = "".join("X" if q == qubit else "I" for q in range(n))
            rotated = rotate_to_z_type(PauliSum({tau: 1 + 0j}), reduced,
                                      anchors)
            assert list(rotated.terms) == [expected]
            assert rotated.terms[expected] == pytest.approx(1 + 0j, abs=1e-12)

    def test_it_preserves_the_spectrum(self, h2_minimal):
        # A Clifford is unitary, so this is the check that it was built as one.
        operator, _bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        rotated = rotate_to_z_type(operator, reduced, anchors)
        before = np.linalg.eigvalsh(operator.to_matrix())
        after = np.linalg.eigvalsh(rotated.to_matrix())
        assert np.allclose(before, after, atol=1e-10)

    def test_the_rotated_operator_commutes_with_every_anchor_x(self, h2_minimal):
        operator, _bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        rotated = rotate_to_z_type(operator, reduced, anchors)
        # An X on the anchor commutes iff no term carries Y or Z there.
        for qubit in anchors:
            assert all(label[qubit] in ("I", "X") for label in rotated.terms)


# --------------------------------------------------------------------------- #
# The physics: the ground-state energy must survive.
# --------------------------------------------------------------------------- #

class TestTheGroundStateSurvives:
    @staticmethod
    def _taper(operator, bits):
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        signs = sector_signs(bits, reduced)
        return taper(operator, reduced, signs, anchors), reduced, anchors

    def test_hydrogen_in_a_minimal_basis_reduces_to_one_qubit(self, h2_minimal):
        # The independent landmark: H2/STO-3G is known to taper to a single
        # qubit.  Fewer would mean an invented symmetry, more a missed one.
        operator, bits = h2_minimal
        tapered, reduced, _anchors = self._taper(operator, bits)
        assert operator.num_qubits == 4
        assert len(reduced) == 3
        assert tapered.num_qubits == 1

    @pytest.mark.parametrize("case", ["h2_minimal", "h2_split", "lih_minimal"])
    def test_the_energy_is_unchanged(self, case, request):
        operator, bits = request.getfixturevalue(case)
        tapered, _reduced, _anchors = self._taper(operator, bits)
        assert ground_energy(tapered) == pytest.approx(ground_energy(operator),
                                                      abs=1e-10)

    @pytest.mark.parametrize("case", ["h2_minimal", "h2_split", "lih_minimal"])
    def test_it_removes_at_least_as_many_qubits_as_the_parity_reduction(
            self, case, request):
        # The parity two-qubit reduction is the special case this generalizes,
        # so the general taper must never do worse than it.
        operator, bits = request.getfixturevalue(case)
        tapered, _reduced, _anchors = self._taper(operator, bits)
        assert tapered.num_qubits <= operator.num_qubits - 2

    def test_the_wrong_sector_gives_a_different_energy(self, h2_minimal):
        # Which is why `sector_signs` reads the reference rather than guessing:
        # the tapered spectrum is a *subset* of the original, and the lowest
        # eigenvalue of the wrong subset is not the ground-state energy.
        operator, bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        right = sector_signs(bits, reduced)
        wrong = [-s for s in right]
        exact = ground_energy(operator)
        assert ground_energy(taper(operator, reduced, right, anchors)) \
            == pytest.approx(exact, abs=1e-10)
        assert ground_energy(taper(operator, reduced, wrong, anchors)) \
            > exact + 1e-6

    def test_every_tapered_eigenvalue_is_an_eigenvalue_of_the_original(
            self, h2_minimal):
        operator, bits = h2_minimal
        tapered, _reduced, _anchors = self._taper(operator, bits)
        full = np.linalg.eigvalsh(operator.to_matrix())
        for value in np.linalg.eigvalsh(tapered.to_matrix()):
            assert np.min(np.abs(full - value)) < 1e-9, value


# --------------------------------------------------------------------------- #
# The trap it shares with the particle-number sector.
# --------------------------------------------------------------------------- #

class TestItRefusesToProjectAnOperator:
    def test_a_leaking_operator_is_named(self, h2_minimal):
        operator, _bits = h2_minimal
        reduced, _anchors = reduce_generators(symmetry_generators(operator))
        # A single X anticommutes with any Z-type tau that acts on that qubit.
        stray = PauliSum({"X" + "I" * (operator.num_qubits - 1): 1 + 0j})
        assert leaking_terms(stray, reduced)

    def test_tapering_it_raises_rather_than_projecting(self, h2_minimal):
        # exp(PAP) != P exp(A) P: an ansatz built from projected generators
        # converges above the ground state and says nothing about it.
        operator, bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        signs = sector_signs(bits, reduced)
        stray = PauliSum({"X" + "I" * (operator.num_qubits - 1): 1 + 0j})
        with pytest.raises(SymmetryLeakError,
                           match="do not commute with the Z2"):
            taper(stray, reduced, signs, anchors)
        # Callers that catch ValueError keep working.
        assert issubclass(SymmetryLeakError, ValueError)

    def test_the_hamiltonian_itself_never_leaks(self, h2_split):
        # By construction: the generators were found as the operators commuting
        # with every term.  Worth asserting, because it is the premise.
        operator, _bits = h2_split
        reduced, _anchors = reduce_generators(symmetry_generators(operator))
        assert leaking_terms(operator, reduced) == []

    def test_a_mismatched_sign_count_is_refused(self, h2_minimal):
        operator, _bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        with pytest.raises(ValueError, match="signs for"):
            taper(operator, reduced, [1], anchors)


class TestTaperingAnOperatorAfterwards:
    def test_it_lands_where_the_problem_put_its_own_generators(self, lih_minimal):
        # ``taper_operator`` is how an operator that was not handed to
        # ``taper_problem`` joins the register it built; if it took a different
        # Clifford or sector, that operator would sit on a register of its own.
        # Terms of the Hamiltonian commute with every symmetry by construction,
        # so they stand in for pool generators that survive the taper.
        operator, bits = lih_minimal
        labels = [label for label in operator.terms if set(label) != {"I"}][:3]
        generators = [PauliSum({label: 1j}) for label in labels]
        info = taper_problem(operator, generators, bits)
        assert info.kept == (0, 1, 2)
        for original, tapered in zip(generators, info.generators):
            assert info.taper_operator(original).terms == pytest.approx(
                tapered.terms)

    def test_it_uses_the_sector_the_problem_chose(self, lih_minimal):
        # The wrong sector gives a different spectrum (see
        # ``test_the_wrong_sector_gives_a_different_energy``), so this is what
        # a mismatched sign or anchor would break.
        operator, bits = lih_minimal
        info = taper_problem(operator, [], bits)
        tapered = info.taper_operator(operator)
        assert tapered.num_qubits == info.n_qubits
        assert ground_energy(tapered) == pytest.approx(ground_energy(operator),
                                                      abs=1e-10)

    def test_it_refuses_to_project_an_operator(self, lih_minimal):
        operator, bits = lih_minimal
        info = taper_problem(operator, [], bits)
        stray = PauliSum({"X" + "I" * (operator.num_qubits - 1): 1 + 0j})
        with pytest.raises(ValueError, match="do not commute"):
            info.taper_operator(stray)

    def test_observable_projection_drops_only_symmetry_changing_terms(
            self, lih_minimal):
        """A sector observable may be projected; a pool generator may not."""
        operator, bits = lih_minimal
        info = taper_problem(operator, [], bits)
        commuting = next(label for label in operator.terms
                         if set(label) != {"I"})
        leaking = next("".join("X" if q == index else "I"
                               for q in range(operator.num_qubits))
                       for index in range(operator.num_qubits)
                       if leaking_terms(PauliSum({"".join(
                           "X" if q == index else "I"
                           for q in range(operator.num_qubits)): 1}),
                           info.symmetries))
        mixed = PauliSum({commuting: 2.0, leaking: 3.0})
        projected = info.taper_observable(mixed)
        expected = info.taper_operator(PauliSum({commuting: 2.0}))
        assert projected.num_qubits == info.n_qubits
        assert projected.terms == pytest.approx(expected.terms)


class TestTheReferenceState:
    def test_the_signs_are_plus_or_minus_one(self, lih_minimal):
        operator, bits = lih_minimal
        reduced, _anchors = reduce_generators(symmetry_generators(operator))
        assert all(s in (+1, -1) for s in sector_signs(bits, reduced))

    def test_a_reference_of_the_wrong_width_is_refused(self, h2_minimal):
        operator, _bits = h2_minimal
        reduced, _anchors = reduce_generators(symmetry_generators(operator))
        with pytest.raises(ValueError, match="bits and the symmetry acts on"):
            sector_signs([1, 0], reduced)

    def test_the_tapered_reference_drops_the_anchors(self, h2_minimal):
        operator, bits = h2_minimal
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        kept = tapered_reference_bits(bits, anchors)
        assert len(kept) == operator.num_qubits - len(anchors)


# --------------------------------------------------------------------------- #
# Through the calculator: taper=True.
# --------------------------------------------------------------------------- #

OPTIMIZER = {"method": "SLSQP", "maxiter": 400, "tol": 1e-11}


def _run(symbols, positions, basis, cell, **kwargs):
    from mandacaru import Mandacaru

    atoms = Atoms(symbols, positions=positions, cell=cell)
    atoms.center()
    atoms.calc = Mandacaru(method="adapt-vqe", basis=basis, h=0.35,
                           optimizer=OPTIMIZER, trace=False, **kwargs)
    energy = atoms.get_potential_energy()
    return atoms, energy, atoms.calc.solver


class TestThroughTheCalculator:
    """``taper=True`` must change the register and nothing else."""

    CASES = (("H2", [(0, 0, 0), (0, 0, 0.74)], "STO-3G", (8., 8., 8.), 4, 1),
             ("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G", (8., 8., 8.), 8, 5),
             ("LiH", [(0, 0, 0), (0, 0, 1.60)], "STO-3G", (9., 9., 10.), 6, 4))

    @pytest.mark.parametrize("symbols,positions,basis,cell,plain,tapered",
                             CASES)
    def test_the_energy_is_unchanged_and_the_register_smaller(
            self, symbols, positions, basis, cell, plain, tapered):
        _a0, e0, s0 = _run(symbols, positions, basis, cell, taper=False)
        _a1, e1, s1 = _run(symbols, positions, basis, cell, taper=True)
        assert s0.n_qubits == plain
        assert s1.n_qubits == tapered
        assert e1 == pytest.approx(e0, abs=1e-8)

    def test_dropping_symmetry_changing_generators_costs_nothing(self):
        # The design decision this validates: generators that change a Z2
        # symmetry are dropped rather than projected.  Eight of them go on
        # H2/6-31G and the energy does not move, because in the sector the
        # reference fixes they cannot contribute at all.
        _a0, e0, _s0 = _run("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G",
                            (8., 8., 8.), taper=False)
        _a1, e1, s1 = _run("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G",
                           (8., 8., 8.), taper=True)
        assert s1._taper_info.dropped
        assert e1 == pytest.approx(e0, abs=1e-8)

    def test_the_coupled_exchange_pool_is_tapered_through_its_members(
            self, monkeypatch):
        # A CEO operator carries the excitations it couples in ``members``, and
        # the growth step evaluates and appends those.  They were left on the
        # full register while the operator itself was tapered, so the first
        # gradient of a member multiplied a 2^n matrix into a 2^(n-k) state.
        from mandacaru.circuits.pools import CEOPool

        expanded = []
        grown_operators = CEOPool.grown_operators

        def spy(self, selected, gradient):
            if len(selected.members) > 1:
                expanded.append(selected.label)
            return grown_operators(self, selected, gradient)

        monkeypatch.setattr(CEOPool, "grown_operators", spy)
        args = ("LiH", [(0, 0, 0), (0, 0, 1.60)], "STO-3G", (9., 9., 10.))
        _a0, e0, _s0 = _run(*args, taper=False, pool="ceo")
        expanded.clear()
        _a1, e1, s1 = _run(*args, taper=True, pool="ceo")
        # The pool must hold coupled operators and the run must have reached the
        # branch that evaluates their members; without that the energy match
        # below would pass on a run that never touched them.
        assert any(len(op.members) > 1 for op in s1._pool_ops)
        assert expanded
        for op in s1._pool_ops:
            assert op.generator.num_qubits == s1.n_qubits
            for member in op.members:
                assert member.generator.num_qubits == s1.n_qubits
        assert e1 == pytest.approx(e0, abs=1e-8)

    def test_it_reports_what_it_did(self):
        _a, _e, solver = _run("LiH", [(0, 0, 0), (0, 0, 1.60)], "STO-3G",
                              (9., 9., 10.), taper=True)
        line = solver._taper_label()
        assert "2 qubit(s) removed" in line
        assert "4 on the register" in line

    def test_an_untapered_run_says_none(self):
        _a, _e, solver = _run("LiH", [(0, 0, 0), (0, 0, 1.60)], "STO-3G",
                              (9., 9., 10.), taper=False)
        assert solver._taper_label() == "none"

    def test_the_particle_number_sector_is_not_used_with_it(self):
        # A tapered register has no particle-number basis to enumerate: the
        # Clifford mixed the occupation bits into parities.
        _a, _e, solver = _run("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G",
                              (8., 8., 8.), taper=True)
        assert solver._sector is None


class TestWhatItRefuses:
    def test_it_needs_jordan_wigner(self):
        from mandacaru import Mandacaru

        with pytest.raises(ValueError, match="needs mapping='jordan_wigner'"):
            Mandacaru(basis="HAO", taper=True, mapping="parity_reduced")

    @pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
    def test_every_other_encoding_is_refused_too(self, mapping):
        from mandacaru import Mandacaru

        with pytest.raises(ValueError, match="needs mapping='jordan_wigner'"):
            Mandacaru(basis="HAO", taper=True, mapping=mapping)

    def test_tapered_rdms_reproduce_untapered_density_and_forces(self):
        """The original ladder observables are mapped through the same taper."""
        args = ("H2", [(0, 0, 0), (0, 0, 0.74)], "STO-3G", (8., 8., 8.))
        plain, _energy, _solver = _run(*args)
        tapered, _energy, _solver = _run(*args, taper=True)
        for atoms in (plain, tapered):
            atoms.get_forces()
        assert np.allclose(tapered.calc._state_rdms(tapered.calc.solver)[0],
                           plain.calc._state_rdms(plain.calc.solver)[0],
                           atol=1e-8)
        assert np.allclose(tapered.calc.force_result.unprojected,
                           plain.calc.force_result.unprojected, atol=1e-4)

    def test_measured_tapered_rdms_reproduce_exact_forces(self):
        """Measured Pauli labels have the tapered width and correct sector."""
        from mandacaru.backends.providers import QiskitProvider

        args = ("H2", [(0, 0, 0), (0, 0, 0.74)], "STO-3G", (8., 8., 8.))
        exact, _energy, _solver = _run(*args, taper=True)
        measured, _energy, _solver = _run(
            *args, taper=True,
            measurement_provider=QiskitProvider(device="statevector", shots=0))
        expected = exact.get_forces()
        observed = measured.get_forces()
        assert np.allclose(observed, expected, atol=1e-5)
        assert measured.calc.measurement["rdms"] is not None

    def test_the_dry_run_calls_its_count_an_upper_bound(self):
        # It cannot know how many symmetries there are without the Pauli terms.
        from mandacaru import Mandacaru

        atoms = Atoms("LiH", positions=[(0, 0, 0), (0, 0, 1.60)],
                      cell=(9., 9., 10.))
        atoms.center()
        estimate = Mandacaru(basis="STO-3G", taper=True, dry_run=True,
                             trace=False).estimate_qubits(atoms)
        assert any("UPPER BOUND" in note for note in estimate.notes)
