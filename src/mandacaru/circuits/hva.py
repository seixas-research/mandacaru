# -*- coding: utf-8 -*-
# file: circuits/hva.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Hamiltonian variational ansatz with fixed, symmetry-respecting layers.

The default decomposition is the full one-body and two-body parts of a
number-conserving fermionic Hamiltonian.  One layer applies
``exp(-i theta_1 H_1) exp(-i theta_2 H_2)`` in that order.  Custom groups can
describe a different physical decomposition, provided their sum is the
nonconstant Hamiltonian.  Sparse exponential actions prepare exact local
state vectors without storing dense unitary matrices.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.sparse.linalg import expm_multiply

from ..core.mapping import (Fermion, FermionTerm, PauliSum,
                            reference_qubit_bits, resolve_mapping)


def hamiltonian_groups(
        hamiltonian: Fermion, groups: tuple[Fermion, ...] | None = None
) -> tuple[Fermion, ...]:
    """Validate or build a number-conserving decomposition of ``hamiltonian``.

    Constants are omitted because they contribute only a global phase.  Each
    supplied group must preserve particle number and their termwise sum must
    equal every nonconstant term of the Hamiltonian.  The default groups all
    quadratic terms together and all quartic terms together.
    """
    n_modes = hamiltonian.n_modes()
    if groups is None:
        by_order: dict[int, dict[FermionTerm, complex]] = {2: {}, 4: {}}
        for term, coefficient in hamiltonian.terms.items():
            if not term:
                continue
            if len(term) not in by_order:
                raise ValueError(
                    "default HVA groups support one- and two-body "
                    "number-conserving Hamiltonians; pass hva_groups= for "
                    "another decomposition")
            by_order[len(term)][term] = coefficient
        groups = tuple(Fermion(terms, n_modes=n_modes)
                       for terms in by_order.values() if terms)
    else:
        groups = tuple(groups)
        if not groups:
            raise ValueError("hva_groups must contain at least one operator")

    combined: dict[FermionTerm, complex] = {}
    for group in groups:
        if not isinstance(group, Fermion):
            raise TypeError("every HVA group must be a Fermion operator")
        if group.n_modes() != n_modes:
            raise ValueError("every HVA group must act on all Hamiltonian modes")
        for term, coefficient in group.terms.items():
            if not term:
                raise ValueError("HVA groups omit scalar terms (global phases)")
            if sum(bool(dagger) for _, dagger in term) * 2 != len(term):
                raise ValueError("HVA groups must conserve particle number")
            combined[term] = combined.get(term, 0j) + coefficient
    keys = set(combined) | {term for term in hamiltonian.terms if term}
    if any(abs(combined.get(term, 0j)
               - hamiltonian.terms.get(term, 0j)) > 1e-9 for term in keys):
        raise ValueError("HVA groups must sum to the nonconstant Hamiltonian")
    return groups


class HamiltonianVariationalAnsatz:
    r"""Fixed product of exponentials of Hamiltonian groups.

    Parameters
    ----------
    hamiltonian : Fermion
        Number-conserving model in the same orbital basis as the reference.
    num_particles : tuple[int, int]
        Alpha and beta occupations of the reference determinant.
    mapping : str
        Fermion-to-qubit encoding, including ``parity_reduced``.
    layers : int
        Positive number of repetitions of the ordered group sequence (default
        two; one layer cannot correlate some symmetry-adapted HF references).
    groups : tuple[Fermion, ...], optional
        Explicit physical decomposition; defaults to one-body then two-body.

    Notes
    -----
    ``state`` and ``evolve`` use exact sparse exponential actions.  The ansatz
    implements the state-vector protocol, but deliberately does not claim a
    gate-level serialization: generic noncommuting Pauli terms inside a group
    require a specified Trotter formula to become a circuit.
    """

    def __init__(self, hamiltonian: Fermion,
                 num_particles: tuple[int, int], *, mapping: str,
                 layers: int = 2,
                 groups: tuple[Fermion, ...] | None = None) -> None:
        if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
            raise ValueError("HVA layers must be a positive integer")
        self.layers = int(layers)
        self.mapping = resolve_mapping(mapping)
        self.n_modes = hamiltonian.n_modes()
        if self.n_modes < 2 or self.n_modes % 2:
            raise ValueError("HVA needs an even number of spin orbitals")
        self.n_spatial_orbitals = self.n_modes // 2
        self.num_particles = tuple(int(n) for n in num_particles)
        if len(self.num_particles) != 2 or any(
                n < 0 or n > self.n_spatial_orbitals
                for n in self.num_particles):
            raise ValueError("HVA particle counts do not fit the orbital basis")
        self.n_qubits = self.n_modes - (
            2 if self.mapping == "parity_reduced" else 0)
        self.groups = hamiltonian_groups(hamiltonian, groups)
        if not self.groups:
            raise ValueError("HVA needs at least one nonconstant Hamiltonian group")
        self._matrices = []
        for group in self.groups:
            mapped = group.map_to_qubits(
                self.mapping, n_modes=self.n_modes,
                num_particles=self.num_particles if self.mapping == "parity_reduced"
                else None)
            self._check_hermitian(mapped)
            self._matrices.append((-1j * mapped.to_sparse_matrix()).tocsr())
        occupied = (tuple(range(self.num_particles[0]))
                    + tuple(range(self.n_spatial_orbitals,
                                  self.n_spatial_orbitals + self.num_particles[1])))
        bits = reference_qubit_bits(self.mapping, self.n_modes, occupied)
        index = sum(int(bit) << (self.n_qubits - 1 - qubit)
                    for qubit, bit in enumerate(bits))
        self._reference = np.zeros(1 << self.n_qubits, dtype=complex)
        self._reference[index] = 1.0

    @staticmethod
    def _check_hermitian(operator: PauliSum) -> None:
        """Reject a group that cannot generate unitary real-time evolution."""
        scale = max((abs(c) for c in operator.terms.values()), default=1.0)
        residual = max((abs(c.imag) for c in operator.terms.values()), default=0.0)
        if residual > 1e-9 * max(scale, 1.0):
            raise ValueError("each HVA Hamiltonian group must be Hermitian")

    @property
    def num_parameters(self) -> int:
        """One angle per group per layer."""
        return self.layers * len(self.groups)

    def reference_state(self) -> np.ndarray:
        """The occupation-ordered reference determinant in this mapping."""
        return self._reference.copy()

    def state(self, theta: np.ndarray) -> np.ndarray:
        """Prepare the HVA state from the mean-field reference determinant."""
        return self.evolve(theta, self._reference)

    def evolve(self, theta: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Apply all ordered HVA layers to one or more reference states."""
        angles = np.asarray(theta, dtype=float).ravel()
        if angles.size != self.num_parameters or not np.all(np.isfinite(angles)):
            raise ValueError(f"HVA needs {self.num_parameters} finite angles")
        out = np.asarray(references, dtype=complex)
        if out.ndim not in (1, 2) or out.shape[0] != self._reference.size:
            raise ValueError("reference state has the wrong register dimension")
        out = out.copy()
        for angle, generator in zip(angles, self._matrices * self.layers):
            if angle:
                out = expm_multiply(angle * generator, out,
                                    traceA=angle * generator.diagonal().sum())
        return out
