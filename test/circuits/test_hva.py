# -*- coding: utf-8 -*-
# file: test/circuits/test_hva.py

# This code is part of Mandacaru.
# MIT License

"""Physical and numerical contracts of the fixed HVA ansatz."""

import numpy as np
import pytest

from mandacaru.circuits.hva import (HamiltonianVariationalAnsatz,
                                   hamiltonian_groups)
from mandacaru.core.mapping import Fermion


def spin_mixing_model():
    """A small number-conserving model with noncommuting mapped terms."""
    n0 = Fermion.creation(0) * Fermion.annihilation(0)
    excitation = Fermion.creation(0) * Fermion.annihilation(1)
    conjugate = Fermion.creation(1) * Fermion.annihilation(0)
    return n0, excitation, conjugate


def test_exact_group_evolution_is_unitary_and_zero_is_the_reference():
    """An exact HVA block applies its entire noncommuting group."""
    n0, excitation, conjugate = spin_mixing_model()
    ansatz = HamiltonianVariationalAnsatz(
        n0 + excitation + conjugate, (1, 0), mapping="jordan_wigner",
        layers=1)
    assert np.array_equal(ansatz.state(np.zeros(1)), ansatz.reference_state())
    assert np.linalg.norm(ansatz.state(np.array([0.71]))) == pytest.approx(1.0)
    assert not ansatz.circuit_serializable


def test_strang_steps_converge_for_a_noncommuting_group():
    """The circuit ansatz approaches but differs from exact-group evolution."""
    n0, excitation, conjugate = spin_mixing_model()
    hamiltonian = n0 + excitation + conjugate
    exact = HamiltonianVariationalAnsatz(
        hamiltonian, (1, 0), mapping="jordan_wigner", layers=1)
    one = HamiltonianVariationalAnsatz(
        hamiltonian, (1, 0), mapping="jordan_wigner", layers=1,
        evolution="trotter", order=2, steps=1)
    eight = HamiltonianVariationalAnsatz(
        hamiltonian, (1, 0), mapping="jordan_wigner", layers=1,
        evolution="trotter", order=2, steps=8)
    angle = np.array([0.71])
    target = exact.state(angle)
    error_one = 1 - abs(np.vdot(target, one.state(angle)))
    error_eight = 1 - abs(np.vdot(target, eight.state(angle)))
    assert error_one > 1e-5
    assert error_eight < error_one / 20
    assert eight.circuit_serializable
    assert eight.circuit_parameters(angle).size == len(eight.pauli_generators)


def test_nonhermitian_custom_group_is_rejected():
    """A full Hermitian H does not make each split group unitary."""
    n0, excitation, conjugate = spin_mixing_model()
    hamiltonian = n0 + excitation + conjugate
    with pytest.raises(ValueError, match="Hermitian"):
        HamiltonianVariationalAnsatz(
            hamiltonian, (1, 0), mapping="jordan_wigner", layers=1,
            groups=(n0 + excitation, conjugate))


def test_parity_reduction_requires_separate_spin_conservation():
    """Two-qubit parity reduction cannot encode a spin-flip HVA block."""
    n0, excitation, conjugate = spin_mixing_model()
    with pytest.raises(ValueError, match="separately"):
        HamiltonianVariationalAnsatz(
            n0 + excitation + conjugate, (1, 0),
            mapping="parity_reduced", layers=1)


def test_invalid_group_coefficients_are_rejected_before_mapping():
    """No NaN coefficient may enter the sparse exponential or optimizer."""
    n0, _, _ = spin_mixing_model()
    invalid = Fermion({((0, True), (0, False)): np.nan}, n_modes=2)
    with pytest.raises(ValueError, match="finite"):
        hamiltonian_groups(n0 + invalid)
