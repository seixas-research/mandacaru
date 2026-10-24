# -*- coding: utf-8 -*-
# file: circuits/ansatz.py

# This code is part of Mandacaru.
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
and double excitation generators (:mod:`mandacaru.circuits.gates`), acting on the
Hartree-Fock reference determinant.  Each generator is mapped to qubits with the
same fermion-to-qubit mapping as the Hamiltonian (Jordan-Wigner by default) and
the exponential is evaluated exactly on the :math:`2^N` state vector -- an exact,
simulator-oriented reference implementation for small systems.

With ``trotter=True`` the state is instead the first-order Trotter product
:math:`\prod_k e^{\theta_k (\hat T_k - \hat T_k^\dagger)}|\text{HF}\rangle`, which
is the form realized as a quantum circuit but only approximates the exact UCC
unitary.

Spin-orbitals follow the spin-blocked ordering used throughout Mandacaru: the
first ``n_spatial_orbitals`` indices are the :math:`\alpha` orbitals, the next
``n_spatial_orbitals`` the :math:`\beta` orbitals.  The reference fills the
lowest ``n_alpha`` / ``n_beta`` orbitals of each block.

**Particle-number sector.**  Every generator conserves ``(n_alpha, n_beta)``,
so the state never leaves the ``C(M, n_alpha) C(M, n_beta)`` determinants of
that sector.  With ``sector=`` (:class:`~mandacaru.core.sector.ParticleSector`)
the state vectors have only those amplitudes and the generators are sparse
sector matrices: no ``2^N x 2^N`` array exists.  The exact form evaluates
:math:`e^{\sum_k\theta_k G_k}|\text{HF}\rangle` with
:func:`scipy.sparse.linalg.expm_multiply` on the summed sparse generator
(its sparsity pattern is built once, each evaluation only reweights it); the
Trotter form is the product of closed-form exponentials of
:class:`~mandacaru.circuits.adapt_ansatz.AdaptAnsatz`, which also gives it
the adjoint gradient.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..core.mapping import Fermion, resolve_mapping
from .gates import double_excitation, single_excitation
from .pools import PoolOperator

# UCCSD conforms to the mandacaru.circuits.base.Ansatz protocol
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
    sector : ParticleSector, optional
        Prepare states inside the ``(n_alpha, n_beta)`` sector instead of on
        the whole register (see the module docstring).  It can also be set
        later with :meth:`use_sector`, which is how a driver that resolves the
        sector from the Hamiltonian attaches it.  Not combinable with
        ``provider``.
    provider : CircuitProvider, optional
        Execute the ansatz as a real quantum circuit on an SDK's local
        state-vector simulator (Qiskit / Amazon Braket / Cirq; see
        :mod:`mandacaru.backends.providers`).  A circuit realizes the **Trotter
        product** form, so ``trotter=True`` is required when a provider is given.
    """

    def __init__(self, n_spatial_orbitals: int, num_particles: tuple[int, int],
                 mapping: str = "jordan_wigner", include_singles: bool = True,
                 trotter: bool = False, provider=None, sector=None):
        self.n_spatial_orbitals = int(n_spatial_orbitals)
        self.num_particles = (int(num_particles[0]), int(num_particles[1]))
        self.mapping = resolve_mapping(mapping)
        self.include_singles = include_singles
        self.trotter = trotter
        self.provider = provider
        self.n_modes = 2 * self.n_spatial_orbitals
        self.n_qubits = self.n_modes - (
            2 if self.mapping == "parity_reduced" else 0)
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
            g.map_to_qubits(self.mapping, n_modes=self.n_modes,
                            num_particles=self.num_particles)
            for g in self.excitations]
        # The 2^N generator matrices and the reference vector are built on
        # first use, not here: a shot-based run submits circuits and never
        # needs them (at 12 qubits each matrix is 256 MiB).
        self._generator_cache = None
        self._hf_cache = None
        #: The sector the states live in (``None``: the whole register).
        self.sector = None
        self._sector_cache = None
        if sector is not None:
            self.use_sector(sector)

    def use_sector(self, sector) -> None:
        """Prepare states inside ``sector`` (``None`` returns to the register).

        Drops whatever was cached for the previous representation.
        """
        if sector is not None:
            if self.provider is not None:
                raise ValueError(
                    "a circuit provider prepares full-register states; it "
                    "cannot be combined with a particle-number sector")
            if sector.n_qubits != self.n_qubits:
                raise ValueError(
                    f"the sector spans {sector.n_qubits} qubits, the ansatz "
                    f"{self.n_qubits}")
        self.sector = sector
        self._generator_cache = None
        self._hf_cache = None
        self._sector_cache = None

    @property
    def _generators(self) -> list:
        """Dense generator matrices, materialized once on first use."""
        if self._generator_cache is None:
            self._generator_cache = [g.to_matrix()
                                     for g in self._pauli_generators]
        return self._generator_cache

    @property
    def _hf(self) -> np.ndarray:
        if self._hf_cache is None:
            self._hf_cache = self._reference_vector()
        return self._hf_cache

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
        """Hartree-Fock computational-basis state (qubit 0 is the MSB).

        The bits follow the fermion-to-qubit map (occupations for
        Jordan-Wigner, parity sums otherwise), tapered when the two-qubit
        reduction is on.
        """
        from ..core.mapping import reference_qubit_bits
        bits = reference_qubit_bits(self.mapping, self.n_modes, self._occupied)
        index = 0
        for k, bit in enumerate(bits):
            if bit:
                index |= 1 << (self.n_qubits - 1 - k)
        if self.sector is not None:
            return self.sector.basis_vector(index)
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

    @property
    def preparation(self) -> str:
        """``"product"`` (Trotter) or ``"sum"`` (the exact UCC exponential) --
        the form a checkpoint of this ansatz has to record."""
        return "product" if self.trotter else "sum"

    def reference_qubits(self) -> list[int]:
        """Qubit indices set to ``|1>`` in the Hartree-Fock determinant --
        read off the mapped occupations, with no ``2^N`` vector."""
        from ..core.mapping import reference_qubit_bits
        bits = reference_qubit_bits(self.mapping, self.n_modes, self._occupied)
        return [k for k, bit in enumerate(bits) if bit]

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
        (:class:`~mandacaru.algorithms.SubspaceVQE`) need to send several orthogonal
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
        out = self._evolve_in_sector(theta, refs)
        return out[:, 0] if single else out

    # -- the sector representation ---------------------------------------- #

    def _sector_state(self):
        """What the sector representation keeps, built on first use.

        Trotter: an :class:`~mandacaru.circuits.adapt_ansatz.AdaptAnsatz` on
        the same sector holding every generator (its closed-form exponentials
        and adjoint gradient).  Exact: the generators' entries stacked once,
        sorted by position, as ``(values, owner, starts, indices, indptr)``,
        so that the summed generator for any ``theta`` is one weighted
        ``reduceat`` and a CSR constructor -- no per-evaluation matrix sum.
        """
        if self._sector_cache is not None:
            return self._sector_cache
        if self.trotter:
            from .adapt_ansatz import AdaptAnsatz

            ansatz = AdaptAnsatz(
                self.n_qubits, self._occupied, self.mapping, sparse=True,
                num_particles=self.num_particles, sector=self.sector)
            for k, generator in enumerate(self._pauli_generators):
                ansatz.append(PoolOperator(f"uccsd-{k}", generator, (),
                                           "uccsd"))
            self._sector_cache = ansatz
            return ansatz
        dim = self.sector.dim if self.sector is not None else (1 << self.n_qubits)
        rows, cols, values, owner = [], [], [], []
        for k, generator in enumerate(self._pauli_generators):
            matrix = (self.sector.restrict(generator) if self.sector is not None
                      else generator.to_sparse_matrix()).tocoo()
            rows.append(matrix.row.astype(np.int64))
            cols.append(matrix.col.astype(np.int64))
            values.append(matrix.data)
            owner.append(np.full(matrix.data.size, k, dtype=np.int32))
        if rows:
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            values, owner = np.concatenate(values), np.concatenate(owner)
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            values = np.zeros(0, dtype=complex)
            owner = np.zeros(0, dtype=np.int32)
        position = rows * np.int64(dim) + cols
        order = np.argsort(position, kind="stable")
        position, values, owner = position[order], values[order], owner[order]
        if position.size:
            starts = np.flatnonzero(np.concatenate(
                ([True], position[1:] != position[:-1])))
        else:
            starts = np.zeros(0, dtype=np.int64)
        unique = position[starts]
        indices = (unique % dim).astype(np.int32)
        indptr = np.zeros(dim + 1, dtype=np.int64)
        np.add.at(indptr, unique // dim + 1, 1)
        np.cumsum(indptr, out=indptr)
        self._sector_cache = (values, owner, starts, indices, indptr)
        return self._sector_cache

    def _cluster(self, theta: np.ndarray):
        """The summed generator ``sum_k theta_k G_k`` as a sector CSR matrix."""
        import scipy.sparse as sp

        values, owner, starts, indices, indptr = self._sector_state()
        dim = self.sector.dim if self.sector is not None else (1 << self.n_qubits)
        if not starts.size:
            return sp.csr_matrix((dim, dim), dtype=complex)
        data = np.add.reduceat(values * theta[owner], starts)
        return sp.csr_matrix((data, indices, indptr), shape=(dim, dim))

    def _evolve_in_sector(self, theta: np.ndarray, refs: np.ndarray
                          ) -> np.ndarray:
        """``U(theta)`` on sector vectors ``refs`` of shape ``(dim, k)``."""
        dim = self.sector.dim if self.sector is not None else (1 << self.n_qubits)
        if refs.shape[0] != dim:
            raise ValueError(
                f"expected sector vectors of {dim} amplitudes, "
                f"got {refs.shape[0]}")
        if self.trotter:
            return self._sector_state().evolve(theta, refs)
        if not theta.size or not np.any(theta):
            return refs.copy()
        cluster = self._cluster(theta)
        if cluster.nnz == 0:
            return refs.copy()
        row_nnz = np.diff(cluster.indptr)
        nonzero_rows = np.flatnonzero(row_nnz)
        norm1 = float(np.max(np.add.reduceat(
            np.abs(cluster.data), cluster.indptr[nonzero_rows])))
        if norm1 <= 8.0:
            steps = max(1, int(np.ceil(norm1)))
            scale = 1.0 / steps
            out = refs.copy()
            for _ in range(steps):
                term = out
                for k in range(1, 25):
                    term = (cluster @ term) * (scale / k)
                    out = out + term
                    if np.max(np.abs(term)) <= 1e-15 * np.max(np.abs(out)):
                        break
            return out
        from scipy.sparse.linalg import expm_multiply

        return expm_multiply(cluster, refs, traceA=0.0)

    @property
    def energy_and_gradient(self):
        r"""``(theta, apply_hamiltonian) -> (E, dE/dtheta)`` by the adjoint
        method (:meth:`AdaptAnsatz.energy_and_gradient
        <mandacaru.circuits.adapt_ansatz.AdaptAnsatz.energy_and_gradient>`).

        Only the ordered product of exponentials can sweep back through its
        factors, and only the sector representation applies it, so the
        attribute exists for ``trotter=True`` in a sector.  The exact
        exponential of a sum of non-commuting generators has no such sweep;
        for it the attribute is absent and the optimizer differences.
        """
        if self.sector is None or not self.trotter:
            raise AttributeError("energy_and_gradient")
        inner = self._sector_state().energy_and_gradient
        return lambda theta, apply_hamiltonian: inner(
            np.asarray(theta, dtype=float).ravel(), apply_hamiltonian)

    def _evolve_on_provider(self, theta, refs: np.ndarray) -> np.ndarray:
        """Run the Trotterized UCCSD circuit once per reference column.

        Mirrors :meth:`~mandacaru.circuits.adapt_ansatz.AdaptAnsatz._evolve_on_provider`:
        each column must be a computational-basis (Slater-determinant) state, as
        a circuit cannot be initialized in a superposition.
        """
        from ..backends.providers import evolve_determinants

        return evolve_determinants(self.provider, self.n_qubits,
                                   self._pauli_generators, theta, refs)

    def __repr__(self) -> str:
        return (f"UCCSD(n_qubits={self.n_qubits}, "
                f"num_particles={self.num_particles}, "
                f"num_parameters={self.num_parameters})")
