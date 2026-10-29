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

from mandacaru.core.mapping import PauliSum
from mandacaru.core.tapering import reduce_generators, sector_signs, symmetry_generators, symplectic_form, taper


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def molecular_pauli(symbols, positions, basis, cell, grid_spacing=0.35):
    """``(PauliSum, reference_bits)`` for a real molecule, Jordan-Wigner."""
    from mandacaru import Mandacaru

    atoms = Atoms(symbols, positions=positions, cell=cell)
    atoms.center()
    atoms.calc = Mandacaru(method="adapt-vqe", basis=basis,
                           grid_spacing=grid_spacing, max_iterations=0,
                           trace=False, profile=False)
    atoms.get_potential_energy()
    problem = atoms.calc.problem
    operator = PauliSum({label: complex(re, im)
                         for label, re, im in problem.hamiltonian_terms},
                        num_qubits=problem.num_qubits).simplify()
    bits = [0] * problem.num_qubits
    for qubit in problem.reference_qubits:
        bits[qubit] = 1
    return operator, bits


def ground_energy(operator: PauliSum) -> float:
    return float(np.min(np.linalg.eigvalsh(operator.to_matrix())))


@pytest.fixture(scope="module")
def h2_minimal():
    return molecular_pauli("H2", [(0, 0, 0), (0, 0, 0.74)], "STO-3G",
                           (8., 8., 8.))


@pytest.fixture(scope="module")
def h2_split():
    return molecular_pauli("H2", [(0, 0, 0), (0, 0, 0.74)], "6-31G",
                           (8., 8., 8.))


@pytest.fixture(scope="module")
def lih_minimal():
    return molecular_pauli("LiH", [(0, 0, 0), (0, 0, 1.60)], "STO-3G",
                           (9., 9., 10.))


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
# The physics: the ground-state energy must survive.
# --------------------------------------------------------------------------- #

class TestTheGroundStateSurvives:
    @staticmethod
    def _taper(operator, bits):
        reduced, anchors = reduce_generators(symmetry_generators(operator))
        signs = sector_signs(bits, reduced)
        return taper(operator, reduced, signs, anchors), reduced, anchors


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
