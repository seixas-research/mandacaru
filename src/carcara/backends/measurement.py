# -*- coding: utf-8 -*-
# file: backends/measurement.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Measuring :math:`\langle H\rangle` from shots -- the hardware energy path.

A state-vector simulator hands back the full amplitude vector, so the energy is
one inner product.  **Real quantum hardware cannot do that.**  A QPU returns
*samples* of computational-basis bit-strings, so the energy of a qubit
Hamiltonian :math:`H = \sum_j c_j P_j` has to be assembled term by term from
measured expectation values,

.. math::

    \langle H\rangle = \sum_j c_j \langle P_j\rangle ,

and each :math:`\langle P_j\rangle` needs the register measured in that Pauli's
own eigenbasis.  Measuring one term per circuit is correct but ruinously
expensive: a modest active space has :math:`10^2`--:math:`10^4` terms.

The standard remedy, implemented here, is **qubit-wise commuting (QWC)
grouping**.  Two Pauli strings are QWC when, on every qubit where both act
non-trivially, they carry the *same* Pauli.  A QWC set can be measured by a
single circuit: rotate each qubit once into the basis its group prescribes,
measure everything, and read every term's expectation value out of the same
bit-strings.  :func:`qubit_wise_commuting_groups` builds those groups with a
greedy largest-first colouring, typically cutting the circuit count by one to
two orders of magnitude.

This module is deliberately provider-independent -- it only produces *basis
labels* and combines counts -- so any SDK backend (see
:mod:`carcara.backends.providers`) can drive a QPU with it.
"""

from __future__ import annotations

import numpy as np

from ..core.mapping import PauliSum


def is_qubit_wise_commuting(a: str, b: str) -> bool:
    """True when Pauli strings ``a`` and ``b`` can be measured in one basis.

    The condition is per qubit: wherever both strings act non-trivially they must
    carry the *same* Pauli letter (identity always agrees).
    """
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def merge_basis(labels) -> str:
    """The single measurement basis covering a set of QWC Pauli strings.

    Qubit ``k`` takes the non-identity letter any member requires there, or
    ``"I"`` (measured in the ``Z`` basis, contributing nothing) if none does.

    Raises
    ------
    ValueError
        If the strings are not qubit-wise commuting.
    """
    labels = list(labels)
    if not labels:
        return ""
    basis = list(labels[0])
    for label in labels[1:]:
        for k, letter in enumerate(label):
            if letter == "I":
                continue
            if basis[k] in ("I", letter):
                basis[k] = letter
            else:
                raise ValueError(
                    f"Pauli strings {''.join(basis)!r} and {label!r} are not "
                    f"qubit-wise commuting on qubit {k}")
    return "".join(basis)


def qubit_wise_commuting_groups(hamiltonian: PauliSum,
                                drop_identity: bool = True):
    """Partition ``hamiltonian`` into QWC groups; return ``(groups, identity)``.

    ``groups`` is a list of ``(basis, [(pauli, coefficient), ...])`` -- one
    measurement circuit per entry, ``basis`` being the string of single-qubit
    bases to rotate into.  ``identity`` is the coefficient of the all-``I`` term,
    which needs no measurement and is added to the energy as a constant.

    The partition is greedy, largest-coefficient-first (so the terms that
    dominate the energy land in the earliest, largest groups).  Greedy colouring
    is not optimal -- finding the minimum number of groups is NP-hard -- but it
    is fast and close enough in practice.
    """
    simplified = hamiltonian.simplify()
    identity = 0.0
    terms = []
    for label, coeff in simplified.terms.items():
        if set(label) == {"I"}:
            identity += float(np.real(coeff))
            continue
        terms.append((label, complex(coeff)))
    if drop_identity is False and identity:
        terms.append(("I" * simplified.num_qubits, complex(identity)))

    # Largest |coefficient| first: the energetically important terms group early.
    terms.sort(key=lambda item: -abs(item[1]))

    groups: list[tuple[list[str], list[tuple[str, complex]]]] = []
    for label, coeff in terms:
        for members, payload in groups:
            if all(is_qubit_wise_commuting(label, other) for other in members):
                members.append(label)
                payload.append((label, coeff))
                break
        else:
            groups.append(([label], [(label, coeff)]))

    return [(merge_basis(members), payload) for members, payload in groups], \
        identity


def pauli_expectation_from_counts(label: str, counts: dict) -> float:
    r"""``<P>`` for one Pauli string from bit-string counts of its own basis.

    ``counts`` maps a measured bit-string (character ``k`` is qubit ``k``, the
    Carcará convention) to its number of occurrences, taken in a basis that
    diagonalizes ``label``.  Each shot contributes :math:`(-1)^{\text{parity}}`
    over the qubits where ``label`` acts non-trivially.
    """
    active = [k for k, letter in enumerate(label) if letter != "I"]
    if not active:
        return 1.0
    total = sum(counts.values())
    if total == 0:
        return 0.0
    accumulated = 0
    for bits, n in counts.items():
        parity = sum(int(bits[k]) for k in active) & 1
        accumulated += -n if parity else n
    return accumulated / total


def energy_from_group_counts(groups, identity: float, counts_per_group) -> float:
    r"""Assemble :math:`\langle H\rangle` from one count dictionary per QWC group.

    ``groups`` is the output of :func:`qubit_wise_commuting_groups` and
    ``counts_per_group`` the measured bit-string counts for each of them, in the
    same order.  Coefficients are Hermitian (real) for a physical Hamiltonian, so
    only the real part contributes.
    """
    energy = float(identity)
    for (_basis, payload), counts in zip(groups, counts_per_group):
        for label, coeff in payload:
            energy += float(np.real(coeff)) * \
                pauli_expectation_from_counts(label, counts)
    return energy


def shot_noise_estimate(hamiltonian: PauliSum, shots: int) -> float:
    r"""Rough standard error of a shot-based ``<H>`` at ``shots`` per group.

    Each :math:`\langle P_j\rangle` has variance at most 1, so summing
    independent terms bounds the error by
    :math:`\bigl(\sum_j |c_j|\bigr)/\sqrt{\text{shots}}` -- the usual
    coefficient-1-norm estimate.  Useful for choosing ``shots`` before paying for
    QPU time: chemical accuracy (1.6 mHa) on a 1-norm of 10 Ha needs
    :math:`\sim 4\times10^7` shots per group in the worst case, which is why
    hardware VQE needs error mitigation and smarter estimators.
    """
    one_norm = sum(abs(complex(c)) for label, c in
                   hamiltonian.simplify().terms.items() if set(label) != {"I"})
    return float(one_norm / np.sqrt(max(int(shots), 1)))
