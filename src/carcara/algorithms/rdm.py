# -*- coding: utf-8 -*-
# file: algorithms/rdm.py

# This code is part of Carcará.
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

with exactly the index convention :meth:`~carcara.core.mapping.Fermion.from_integrals`
uses -- so contracting these RDMs with the integrals reproduces the driver's
energy.  That identity is what makes **nuclear gradients** possible: at the
variational minimum the RDMs are stationary, so differentiating the energy with
respect to the nuclear coordinates only has to differentiate the *integrals*
(see :mod:`carcara.algorithms.forces`).

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
            mapping: str = "jordan_wigner") -> np.ndarray:
    r"""One-particle RDM ``gamma_pq = <psi| a+_p a_q |psi>``.

    Parameters
    ----------
    psi : array_like
        The converged state vector on ``n_modes`` qubits.
    n_modes : int
        Number of spin-orbitals (= qubits) the state lives on.
    mapping : str
        The fermion-to-qubit mapping the Hamiltonian was built with.

    Returns
    -------
    numpy.ndarray
        The ``(M, M)`` Hermitian matrix.  Its trace is the electron number, and
        contracting it with the one-body integrals gives the one-body energy.
    """
    a_psi = annihilated_states(psi, n_modes, mapping)
    # gamma_pq = <a_p psi | a_q psi>
    gamma = a_psi.conj() @ a_psi.T
    return 0.5 * (gamma + gamma.conj().T)


def two_rdm(psi: np.ndarray, n_modes: int,
            mapping: str = "jordan_wigner") -> np.ndarray:
    r"""Two-particle RDM ``Gamma_pqrs = <psi| a+_p a+_q a_s a_r |psi>``.

    Uses ``Gamma_pqrs = <a_q a_p psi | a_s a_r psi>``, so only the
    :math:`M(M-1)/2` distinct doubly-annihilated vectors are built; the full
    tensor follows from one matrix product plus the antisymmetry
    :math:`\Gamma_{pqrs} = -\Gamma_{qprs} = -\Gamma_{pqsr}`.

    The index convention matches
    :meth:`~carcara.core.mapping.Fermion.from_integrals`: the two-body energy is
    ``0.5 * sum Gamma_pqrs <pq|rs>`` in physicists' notation.

    .. note::

       Memory scales as ``M**4``; for the active spaces Carcará targets
       (:math:`M \lesssim 20` spin-orbitals) that is a few MB at most.
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    ops = _annihilator_matrices(n_modes, mapping)

    # |chi_{pq}> = a_q a_p |psi>  for p < q; the p > q entries follow by
    # antisymmetry and the diagonal vanishes (a_p a_p = 0).
    pairs = [(p, q) for p in range(n_modes) for q in range(p + 1, n_modes)]
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
