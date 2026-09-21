# -*- coding: utf-8 -*-
# file: algorithms/rdm.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Reduced density matrices of a variational state.

The one- and two-particle reduced density matrices (RDMs)

.. math::

    \gamma_{pq}      &= \langle\psi|\, a^\dagger_p a_q \,|\psi\rangle, \\
    \Gamma_{pqrs}    &= \langle\psi|\, a^\dagger_p a^\dagger_q a_s a_r \,|\psi\rangle,

are everything an observable built from one- and two-body integrals needs.  In
particular the electronic energy is

.. math::

    E = \sum_{pq} \gamma_{pq}\, h_{pq}
      + \tfrac12 \sum_{pqrs} \Gamma_{pqrs}\, \langle pq|rs\rangle ,

with exactly the index convention :meth:`~mandacaru.core.mapping.Fermion.from_integrals`
uses -- so contracting these RDMs with the integrals reproduces the driver's
energy.  That identity is what makes **nuclear gradients** possible: at the
variational minimum the RDMs are stationary, so differentiating the energy with
respect to the nuclear coordinates only has to differentiate the *integrals*
(see :mod:`mandacaru.algorithms.forces`).

Evaluation strategy
-------------------
Building each RDM element as a qubit operator and taking its expectation value
would cost :math:`O(M^4)` sparse matrix constructions.  Instead we use

.. math::

    \gamma_{pq} = \langle a_p\psi | a_q\psi \rangle, \qquad
    \Gamma_{pqrs} = \langle a_q a_p \psi | a_s a_r \psi \rangle ,

which needs only :math:`M` singly- and :math:`M(M-1)/2` doubly-annihilated
vectors; every RDM element is then an inner product, evaluated for all indices at
once as a single matrix product.  The ladder operators are applied in whatever
fermion-to-qubit encoding the driver used, so the RDMs come out in the *same*
spin-orbital basis as the Hamiltonian.
"""

from __future__ import annotations

import numpy as np

from ..core.mapping import Fermion


def _ladder_operators(n_modes: int, mapping: str):
    """``a_p`` for every mode as qubit :class:`~mandacaru.core.mapping.PauliSum`."""
    return [Fermion({((p, False),): 1.0}, n_modes=n_modes)
            .map_to_qubits(mapping, n_modes=n_modes) for p in range(n_modes)]


def _check_sector(psi, n_modes, sector):
    if sector.mapping == "parity_reduced":
        raise NotImplementedError(
            "RDMs need ladder operators, which do not survive the parity "
            "two-qubit reduction")
    if sector.n_qubits != int(n_modes):
        raise ValueError(f"sector register has {sector.n_qubits} qubits, "
                         f"expected {n_modes}")
    psi = np.asarray(psi, dtype=complex).ravel()
    if psi.size != sector.dim:
        raise ValueError(f"expected {sector.dim} sector amplitudes, got {psi.size}")
    return psi


def _stack_sparse(states):
    """Stack sparse states ``[(indices, amplitudes), ...]`` over their union."""
    if not states:
        return np.zeros((0, 0), dtype=complex)
    keys = np.unique(np.concatenate([idx for idx, _amp in states]))
    out = np.zeros((len(states), keys.size), dtype=complex)
    for row, (idx, amp) in enumerate(states):
        out[row, np.searchsorted(keys, idx)] = amp
    return out


def _sector_annihilated(psi, n_modes, mapping, sector):
    """``[a_p |psi>]`` as sparse states, for a sector state vector."""
    from ..core.sector import apply_pauli_sum
    psi = _check_sector(psi, n_modes, sector)
    ops = _ladder_operators(n_modes, mapping)
    return ops, [apply_pauli_sum(op, sector.indices, psi) for op in ops]


def _annihilator_matrices(n_modes: int, mapping: str):
    """Sparse matrices of ``a_p`` for every mode, in the given qubit encoding."""
    ops = []
    for p in range(n_modes):
        fermion = Fermion({((p, False),): 1.0}, n_modes=n_modes)
        pauli = fermion.map_to_qubits(mapping, n_modes=n_modes)
        ops.append(pauli.to_sparse_matrix())
    return ops


def annihilated_states(psi: np.ndarray, n_modes: int,
                       mapping: str = "jordan_wigner") -> np.ndarray:
    r"""Stack of ``a_p |psi>`` for every spin-orbital ``p``.

    Returns an ``(M, 2**n)`` array whose row ``p`` is :math:`a_p|\psi\rangle`.
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    ops = _annihilator_matrices(n_modes, mapping)
    return np.stack([op @ psi for op in ops])


def one_rdm(psi: np.ndarray, n_modes: int,
            mapping: str = "jordan_wigner", sector=None) -> np.ndarray:
    r"""One-particle RDM ``gamma_pq = <psi| a+_p a_q |psi>``.

    Parameters
    ----------
    psi : array_like
        The converged state vector on ``n_modes`` qubits.
    n_modes : int
        Number of spin-orbitals (= qubits) the state lives on.
    mapping : str
        The fermion-to-qubit mapping the Hamiltonian was built with.
    sector : ParticleSector, optional
        When the driver simulated a particle-number sector, ``psi`` holds that
        sector's amplitudes; the ladder operators are then applied to the sparse
        state directly (no full-register vector is formed).

    Returns
    -------
    numpy.ndarray
        The ``(M, M)`` Hermitian matrix.  Its trace is the electron number, and
        contracting it with the one-body integrals gives the one-body energy.
    """
    if sector is not None:
        _ops, singles = _sector_annihilated(psi, n_modes, mapping, sector)
        a_psi = _stack_sparse(singles)
        gamma = a_psi.conj() @ a_psi.T
        return 0.5 * (gamma + gamma.conj().T)
    a_psi = annihilated_states(psi, n_modes, mapping)
    # gamma_pq = <a_p psi | a_q psi>
    gamma = a_psi.conj() @ a_psi.T
    return 0.5 * (gamma + gamma.conj().T)


def two_rdm(psi: np.ndarray, n_modes: int,
            mapping: str = "jordan_wigner", sector=None) -> np.ndarray:
    r"""Two-particle RDM ``Gamma_pqrs = <psi| a+_p a+_q a_s a_r |psi>``.

    Uses ``Gamma_pqrs = <a_q a_p psi | a_s a_r psi>``, so only the
    :math:`M(M-1)/2` distinct doubly-annihilated vectors are built; the full
    tensor follows from one matrix product plus the antisymmetry
    :math:`\Gamma_{pqrs} = -\Gamma_{qprs} = -\Gamma_{pqsr}`.

    The index convention matches
    :meth:`~mandacaru.core.mapping.Fermion.from_integrals`: the two-body energy is
    ``0.5 * sum Gamma_pqrs <pq|rs>`` in physicists' notation.

    .. note::

       Memory scales as ``M**4``; for the active spaces Mandacaru targets
       (:math:`M \lesssim 20` spin-orbitals) that is a few MB at most.
    """
    pairs = [(p, q) for p in range(n_modes) for q in range(p + 1, n_modes)]
    if sector is not None:
        from ..core.sector import apply_pauli_sum
        ops, singles = _sector_annihilated(psi, n_modes, mapping, sector)
        chi = _stack_sparse([apply_pauli_sum(ops[q], *singles[p])
                             for p, q in pairs])
    else:
        psi = np.asarray(psi, dtype=complex).ravel()
        ops = _annihilator_matrices(n_modes, mapping)
        # |chi_{pq}> = a_q a_p |psi>  for p < q; the p > q entries follow by
        # antisymmetry and the diagonal vanishes (a_p a_p = 0).
        chi = np.stack([ops[q] @ (ops[p] @ psi) for p, q in pairs]) \
            if pairs else np.zeros((0, psi.size), dtype=complex)

    # <chi_{pq} | chi_{rs}> for the stored (p<q, r<s) pairs.
    block = chi.conj() @ chi.T

    gamma2 = np.zeros((n_modes,) * 4, dtype=complex)
    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            value = block[i, j]
            # Gamma_pqrs = <a_q a_p psi | a_s a_r psi>; antisymmetric in (p,q)
            # and in (r,s) separately.
            gamma2[p, q, r, s] = value
            gamma2[q, p, r, s] = -value
            gamma2[p, q, s, r] = -value
            gamma2[q, p, s, r] = value
    return gamma2


def electronic_energy(gamma: np.ndarray, gamma2: np.ndarray,
                      h_so: np.ndarray, g_so: np.ndarray) -> float:
    r"""Energy ``sum gamma_pq h_pq + 0.5 sum Gamma_pqrs <pq|rs>`` (real part).

    The consistency check between the RDMs and the integrals they were built
    from: this must reproduce the driver's electronic energy (everything except
    the constant nuclear-repulsion / frozen-core shift).
    """
    one = np.einsum("pq,pq->", gamma, h_so)
    two = 0.5 * np.einsum("pqrs,pqrs->", gamma2, g_so)
    return float(np.real(one + two))


def particle_number(gamma: np.ndarray) -> float:
    """Electron number ``tr(gamma)`` -- a cheap sanity check on the state."""
    return float(np.real(np.trace(gamma)))


# --------------------------------------------------------------------------- #
# RDMs from Pauli expectation values (tapered registers, measured states).
# --------------------------------------------------------------------------- #

#: Largest register whose RDMs are assembled from Pauli expectation values:
#: every spin-conserving RDM element is mapped to qubits separately, O(n^4).
MAX_PAULI_RDM_MODES = 25


def rdm_qubit_operators(n_modes: int, mapping: str = "jordan_wigner",
                        num_particles=None):
    r"""Qubit operators of the spin-conserving RDM elements.

    Returns ``(ones, twos)``: ``{(p, q): PauliSum}`` for
    :math:`a^\dagger_p a_q` and ``{(p, q, r, s): PauliSum}`` for
    :math:`a^\dagger_p a^\dagger_q a_s a_r` (spin-blocked modes, alpha first),
    mapped with ``mapping``.  With ``"parity_reduced"`` they are tapered by
    the two parity symmetries.  Only elements that conserve both spin populations are built:
    the others vanish for a state of definite :math:`(n_\alpha, n_\beta)`, and
    only those operators survive the tapering.  Their expectation values
    determine the RDMs -- which is how a tapered register, or a state measured
    on a processor, yields forces.
    """
    n_modes = int(n_modes)
    if n_modes > MAX_PAULI_RDM_MODES:
        raise ValueError(
            f"RDMs from Pauli expectation values are limited to "
            f"{MAX_PAULI_RDM_MODES} spin-orbitals (got {n_modes})")
    half = n_modes // 2

    def qubit(term):
        return Fermion({term: 1.0}, n_modes=n_modes).map_to_qubits(
            mapping, n_modes=n_modes, num_particles=num_particles)

    beta = [int(p >= half) for p in range(n_modes)]
    ones = {(p, q): qubit(((p, True), (q, False)))
            for p in range(n_modes) for q in range(n_modes)
            if beta[p] == beta[q]}
    twos = {}
    for p in range(n_modes):
        for q in range(n_modes):
            if p == q:
                continue
            for r in range(n_modes):
                for s in range(n_modes):
                    if r == s or beta[p] + beta[q] != beta[r] + beta[s]:
                        continue
                    if {beta[p], beta[q]} != {beta[r], beta[s]}:
                        continue
                    twos[(p, q, r, s)] = qubit(((p, True), (q, True),
                                                (s, False), (r, False)))
    return ones, twos


def rdms_from_expectations(n_modes: int, ones: dict, twos: dict,
                           expectations: dict):
    """``(gamma, gamma2)`` from Pauli expectation values ``{label: <P>}``."""
    n_modes = int(n_modes)

    def value(op):
        return sum(complex(c) * expectations[label]
                   for label, c in op.terms.items())

    gamma = np.zeros((n_modes, n_modes), dtype=complex)
    for (p, q), op in ones.items():
        gamma[p, q] = value(op)
    gamma2 = np.zeros((n_modes,) * 4, dtype=complex)
    for (p, q, r, s), op in twos.items():
        gamma2[p, q, r, s] = value(op)
    return gamma, gamma2


def pauli_expectations(psi, labels) -> dict:
    """Exact ``{label: <psi|P|psi>}`` for Pauli strings on a small register."""
    from ..core.mapping import PauliSum

    psi = np.asarray(psi, dtype=complex).ravel()
    out = {}
    for label in labels:
        matrix = PauliSum({label: 1.0}).to_sparse_matrix()
        out[label] = float(np.real(np.vdot(psi, matrix @ psi)))
    return out

def expand_frozen_core(gamma, gamma2, frozen, n_spatial_orbitals: int):
    r"""Lift active-space RDMs to the full orbital set, refilling the core.

    A frozen-core run solves for :math:`|\Psi\rangle = a^\dagger_{c_1}\cdots
    a^\dagger_{c_k}|\Psi_{\text{act}}\rangle`: the core spin-orbitals are
    doubly occupied and inert.  Its RDMs therefore carry only the active block,
    while a nuclear gradient contracts against the **full** integrals and needs
    the total density.  The core contributions are exact and closed form:

    .. math::

        \gamma_{pq} &= \delta_{pq}\ (p, q \in C), \qquad
        \gamma_{pq} = \gamma^{\text{act}}_{pq}\ (p, q \in A), \\
        \Gamma_{pqrs} &= \delta_{pr}\delta_{qs} - \delta_{ps}\delta_{qr}
            \quad (\text{all in } C), \\
        \Gamma_{pqrs} &= \delta_{pr}\gamma^{\text{act}}_{qs}
            - \delta_{ps}\gamma^{\text{act}}_{qr} \quad (p, r \in C),

    and their index-swapped partners; core-active cross blocks with an odd
    number of core indices vanish.  Spin-orbitals are spin-blocked
    (``p = spatial + spin * M``), the convention
    :func:`~mandacaru.core.hamiltonian.spin_block_integrals` uses.

    Parameters
    ----------
    gamma, gamma2 : ndarray
        Active-space RDMs, over ``2 * len(active)`` spin-orbitals.
    frozen : sequence of int
        Frozen (doubly occupied) **spatial** orbital indices.
    n_spatial_orbitals : int
        Total number of spatial orbitals.

    Returns
    -------
    (ndarray, ndarray)
        The one- and two-body RDMs over all ``2 * n_spatial_orbitals``
        spin-orbitals.
    """
    M = int(n_spatial_orbitals)
    core_spatial = sorted({int(f) for f in frozen})
    if not core_spatial:
        return gamma, gamma2
    active_spatial = [p for p in range(M) if p not in set(core_spatial)]
    n_act = len(active_spatial)
    gamma = np.asarray(gamma, dtype=complex)
    gamma2 = np.asarray(gamma2, dtype=complex)
    if gamma.shape != (2 * n_act, 2 * n_act):
        raise ValueError(f"expected active RDMs over {2 * n_act} spin-orbitals, "
                         f"got {gamma.shape}")

    # Spin-blocked maps: active index -> full index, and the core set.
    active = [active_spatial[i] + spin * M
              for spin in (0, 1) for i in range(n_act)]
    core = [c + spin * M for spin in (0, 1) for c in core_spatial]
    n_modes = 2 * M

    full_gamma = np.zeros((n_modes, n_modes), dtype=complex)
    full_gamma[np.ix_(active, active)] = gamma
    for c in core:
        full_gamma[c, c] += 1.0

    full_gamma2 = np.zeros((n_modes,) * 4, dtype=complex)
    full_gamma2[np.ix_(active, active, active, active)] = gamma2
    core_index = np.array(core, dtype=int)
    act_index = np.array(active, dtype=int)
    # Core-core: the two-electron density of a filled shell.
    for p in core_index:
        for q in core_index:
            if p == q:
                continue
            full_gamma2[p, q, p, q] += 1.0
            full_gamma2[p, q, q, p] -= 1.0
    # Core-active cross terms (both orderings, direct and exchange).
    for c in core_index:
        full_gamma2[np.ix_([c], act_index, [c], act_index)] += gamma[
            np.newaxis, :, np.newaxis, :]
        full_gamma2[np.ix_(act_index, [c], act_index, [c])] += gamma[
            :, np.newaxis, :, np.newaxis]
        full_gamma2[np.ix_([c], act_index, act_index, [c])] -= gamma.T[
            np.newaxis, :, :, np.newaxis]
        full_gamma2[np.ix_(act_index, [c], [c], act_index)] -= gamma.T[
            :, np.newaxis, np.newaxis, :]
    return full_gamma, full_gamma2
