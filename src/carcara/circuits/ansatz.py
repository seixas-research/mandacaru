# -*- coding: utf-8 -*-
# file: circuits/ansatz.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Parameterized ansätze for variational quantum algorithms.

:class:`UCCSD` implements the Unitary Coupled-Cluster with Singles and Doubles
ansatz as a state-vector generator:

.. math::

    |\psi(\vec\theta)\rangle
        = e^{\sum_k \theta_k (\hat T_k - \hat T_k^\dagger)}\,|\text{HF}\rangle,

the exponential of the full anti-Hermitian cluster operator built from the single
and double excitation generators (:mod:`carcara.circuits.gates`), acting on the
Hartree-Fock reference determinant.  Each generator is mapped to qubits with the
same fermion-to-qubit mapping as the Hamiltonian (Jordan-Wigner by default) and
the exponential is evaluated exactly on the :math:`2^N` state vector -- an exact,
simulator-oriented reference implementation for small systems.

With ``trotter=True`` the state is instead the first-order Trotter product
:math:`\prod_k e^{\theta_k (\hat T_k - \hat T_k^\dagger)}|\text{HF}\rangle`, which
is the form realized as a quantum circuit but only approximates the exact UCC
unitary.

Spin-orbitals follow the spin-blocked ordering used throughout Carcará: the
first ``n_spatial_orbitals`` indices are the :math:`\alpha` orbitals, the next
``n_spatial_orbitals`` the :math:`\beta` orbitals.  The reference fills the
lowest ``n_alpha`` / ``n_beta`` orbitals of each block.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..core.mapping import Fermion
from .gates import double_excitation, single_excitation

# UCCSD conforms to the carcara.circuits.base.Ansatz protocol
# (num_parameters, n_qubits, reference_state, state, evolve).


class UCCSD:
    """UCCSD ansatz state-vector generator over ``2 * n_spatial_orbitals`` qubits.

    Parameters
    ----------
    n_spatial_orbitals : int
        Number of spatial orbitals ``M``; the ansatz acts on ``N = 2M`` qubits.
    num_particles : (int, int)
        ``(n_alpha, n_beta)`` electrons defining the Hartree-Fock reference.
    mapping : str
        Fermion-to-qubit mapping for the generators (default ``"jordan_wigner"``);
        must match the Hamiltonian's mapping.
    include_singles : bool
        Include single excitations (default ``True``).
    trotter : bool
        If ``True``, prepare the state as the first-order Trotter product of
        single-generator exponentials (circuit-faithful but approximate); if
        ``False`` (default), exponentiate the full cluster operator exactly.
    provider : CircuitProvider, optional
        Execute the ansatz as a real quantum circuit on an SDK's local
        state-vector simulator (Qiskit / Amazon Braket / Cirq; see
        :mod:`carcara.backends.providers`).  A circuit realizes the **Trotter
        product** form, so ``trotter=True`` is required when a provider is given.
    """

    def __init__(self, n_spatial_orbitals: int, num_particles: tuple[int, int],
                 mapping: str = "jordan_wigner", include_singles: bool = True,
                 trotter: bool = False, provider=None):
        self.n_spatial_orbitals = int(n_spatial_orbitals)
        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.mapping = mapping
        self.include_singles = include_singles
        self.trotter = trotter
        self.provider = provider
        self.n_qubits = 2 * self.n_spatial_orbitals
        if provider is not None and not trotter:
            raise ValueError(
                "a circuit backend realizes the Trotter product form of UCCSD; "
                "build the ansatz with trotter=True to execute it on "
                f"{getattr(provider, 'name', provider)!r} (the exact "
                "exponential of the summed cluster operator is not a circuit)")

        self._occupied, self._virtual = self._reference_partition()
        self.excitations = self._build_excitations()
        # Qubit (Pauli) form of each anti-Hermitian generator: the matrices drive
        # the state-vector backends, the PauliSums the circuit backends.
        self._pauli_generators = [
            g.map_to_qubits(self.mapping, n_modes=self.n_qubits)
            for g in self.excitations]
        # Pre-materialize each generator matrix once (skipped for circuit
        # execution, where the 2^N matrices are never needed).
        self._generators = ([] if provider is not None else
                            [g.to_matrix() for g in self._pauli_generators])
        self._hf = self._reference_vector()

    # -- reference determinant -------------------------------------------- #

    def _reference_partition(self) -> tuple[list[int], list[int]]:
        """Occupied and virtual spin-orbital indices of the HF reference."""
        M = self.n_spatial_orbitals
        na, nb = self.num_particles
        occ = list(range(na)) + list(range(M, M + nb))
        virt = ([a for a in range(M) if a not in occ]
                + [a for a in range(M, 2 * M) if a not in occ])
        return occ, virt

    def _reference_vector(self) -> np.ndarray:
        """Hartree-Fock computational-basis state (qubit 0 is the MSB)."""
        index = 0
        for j in self._occupied:
            index |= 1 << (self.n_qubits - 1 - j)
        vec = np.zeros(2 ** self.n_qubits, dtype=complex)
        vec[index] = 1.0
        return vec

    # -- excitation pool -------------------------------------------------- #

    def _spin(self, p: int) -> int:
        return p // self.n_spatial_orbitals

    def _build_excitations(self) -> list[Fermion]:
        """Spin-conserving single and double excitation generators."""
        occ, virt = self._occupied, self._virtual
        gens: list[Fermion] = []

        if self.include_singles:
            for i in occ:
                for a in virt:
                    if self._spin(i) == self._spin(a):
                        gens.append(single_excitation(i, a))

        for x in range(len(occ)):
            for y in range(x + 1, len(occ)):
                i, j = occ[x], occ[y]
                for p in range(len(virt)):
                    for q in range(p + 1, len(virt)):
                        a, b = virt[p], virt[q]
                        # Conserve total spin projection Sz.
                        if self._spin(i) + self._spin(j) == self._spin(a) + self._spin(b):
                            gens.append(double_excitation(i, j, a, b))
        return gens

    # -- public interface ------------------------------------------------- #

    @property
    def num_parameters(self) -> int:
        return len(self.excitations)

    @property
    def pauli_generators(self):
        """The cluster generators as qubit operators (for circuit backends)."""
        return list(self._pauli_generators)

    def reference_qubits(self) -> list[int]:
        """Qubit indices set to ``|1>`` in the Hartree-Fock determinant."""
        from ..backends.providers import _occupied_qubits, basis_state_index
        return _occupied_qubits(basis_state_index(self._hf), self.n_qubits)

    def reference_state(self) -> np.ndarray:
        """The Hartree-Fock reference state vector (a copy)."""
        return self._hf.copy()

    def state(self, theta) -> np.ndarray:
        r"""Prepared state ``|psi(theta)>`` for parameters ``theta``.

        Exact UCC (default): ``exp(sum_k theta_k G_k) |HF>``.  Trotter mode:
        the product ``prod_k exp(theta_k G_k) |HF>``.
        """
        return self.evolve(theta, self._hf)

    def evolve(self, theta, references) -> np.ndarray:
        r"""Apply the ansatz unitary ``U(theta)`` to arbitrary reference state(s).

        ``references`` is a single state vector (shape ``(2**N,)``) or a stack of
        column vectors (shape ``(2**N, k)``); the return has the matching shape.
        The unitary is the *same* for every column -- computed once and applied to
        all of them -- which is what the subspace-search solvers
        (:class:`~carcara.algorithms.SubspaceVQE`) need to send several orthogonal
        references through one shared ``U(theta)``.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {theta.size}")
        refs = np.asarray(references, dtype=complex)
        single = refs.ndim == 1
        if single:
            refs = refs[:, None]
        if self.provider is not None:
            out = self._evolve_on_provider(theta, refs)
            return out[:, 0] if single else out
        if self.trotter:
            out = refs.copy()
            for angle, gen in zip(theta, self._generators):
                out = expm(angle * gen) @ out
        else:
            cluster = sum(
                (angle * gen for angle, gen in zip(theta, self._generators)),
                np.zeros((2 ** self.n_qubits,) * 2, dtype=complex))
            out = expm(cluster) @ refs
        return out[:, 0] if single else out

    def _evolve_on_provider(self, theta, refs: np.ndarray) -> np.ndarray:
        """Run the Trotterized UCCSD circuit once per reference column.

        Mirrors :meth:`~carcara.circuits.adapt_ansatz.AdaptAnsatz._evolve_on_provider`:
        each column must be a computational-basis (Slater-determinant) state, as
        a circuit cannot be initialized in a superposition.
        """
        from ..backends.providers import basis_state_index, _occupied_qubits

        columns = []
        for j in range(refs.shape[1]):
            index = basis_state_index(refs[:, j])
            if index is None:
                raise ValueError(
                    f"the {self.provider.name} circuit backend can only evolve "
                    "computational-basis (Slater-determinant) reference states; "
                    f"column {j} of `references` is a superposition")
            columns.append(self.provider.statevector(
                self.n_qubits, _occupied_qubits(index, self.n_qubits),
                self._pauli_generators, theta))
        return np.asarray(columns, dtype=complex).T

    def __repr__(self) -> str:
        return (f"UCCSD(n_qubits={self.n_qubits}, "
                f"num_particles={self.num_particles}, "
                f"num_parameters={self.num_parameters})")
