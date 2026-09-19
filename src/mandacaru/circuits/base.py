# -*- coding: utf-8 -*-
# file: circuits/base.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The ansatz protocol shared by every parameterized state-vector ansatz.

An **ansatz** prepares a state ``|psi(theta)> = U(theta)|reference>`` on
``n_qubits`` qubits from ``num_parameters`` real parameters.  The variational
drivers (:mod:`mandacaru.algorithms`) consume any object satisfying this protocol,
so a new ansatz plugs in without touching a driver.  Both
:class:`~mandacaru.circuits.ansatz.UCCSD` and
:class:`~mandacaru.circuits.adapt_ansatz.AdaptAnsatz` conform to it.

Because it is a :class:`typing.Protocol` with ``runtime_checkable``, conformance
is structural -- an ansatz need not subclass it, and ``isinstance(obj, Ansatz)``
checks only that the methods exist.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Ansatz(Protocol):
    """Structural interface for a parameterized state-vector ansatz."""

    #: Number of qubits the ansatz acts on.
    n_qubits: int

    @property
    def num_parameters(self) -> int:
        """Number of variational parameters ``theta``."""
        ...

    def reference_state(self) -> np.ndarray:
        """The reference state ``|reference>`` (``U`` at ``theta = 0``)."""
        ...

    def state(self, theta) -> np.ndarray:
        """Prepared state ``U(theta)|reference>`` (a ``2**n_qubits`` vector)."""
        ...

    def evolve(self, theta, references) -> np.ndarray:
        """Apply ``U(theta)`` to arbitrary reference state(s).

        ``references`` is a single ``(2**n_qubits,)`` vector or a ``(2**n_qubits,
        k)`` stack of columns; the return matches.  Enables subspace-search
        solvers to send several orthogonal references through one shared unitary.
        """
        ...


@runtime_checkable
class SerializableAnsatz(Ansatz, Protocol):
    """An :class:`Ansatz` that can also be *described*, not only evaluated.

    Checkpoints, circuit export and provider measurement need the state as
    data: the generators as Pauli sums and the reference determinant in
    register terms.  This is **optional** -- an ansatz that only implements
    :class:`Ansatz` still runs through VQE; it just cannot be checkpointed or
    exported, and asking for either is refused before the run starts.
    """

    @property
    def pauli_generators(self) -> list:
        """The anti-Hermitian generators as :class:`~mandacaru.core.mapping.PauliSum`."""
        ...

    def reference_qubits(self) -> list[int]:
        """Qubits set to ``|1>`` in the reference determinant."""
        ...


def is_serializable(ansatz) -> bool:
    """Whether ``ansatz`` can be written to a checkpoint / exported as a circuit."""
    return (hasattr(ansatz, "pauli_generators")
            and callable(getattr(ansatz, "reference_qubits", None)))

