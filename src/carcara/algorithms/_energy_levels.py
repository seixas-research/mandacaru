# -*- coding: utf-8 -*-
# file: algorithms/_energy_levels.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared machinery for molecular energy levels via variational deflation.

Both :class:`~carcara.algorithms.vqe.VQE` and
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQE` compute the low-lying spectrum
(ground state **and** excited states) with **variational quantum deflation**
(VQD, Higgott *et al.* 2019): after the :math:`m` lowest states
:math:`\{|\psi_j\rangle\}_{j<m}` are found, the next one is obtained by
minimizing the *deflated* cost

.. math::

    L_m(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle
        + \sum_{j<m} \beta\,|\langle\psi_j|\psi(\vec\theta)\rangle|^2 ,

whose global minimum is the :math:`m`-th eigenstate provided the penalty weight
:math:`\beta` exceeds the spanned spectral gaps.  This module holds the pieces
shared by both drivers: the penalty term, an automatic :math:`\beta`, and the
:class:`EnergyLevels` result container.  The exact state-vector backend makes the
overlaps :math:`\langle\psi_j|\psi\rangle` and the penalty exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import from_hartree


def deflation_penalty(psi: np.ndarray, states, beta: float) -> float:
    r"""Overlap penalty :math:`\beta\sum_j |\langle\psi_j|\psi\rangle|^2`.

    ``states`` is the list of previously found (state-vector) eigenstates; the
    penalty is zero when ``psi`` is orthogonal to all of them, so it vanishes at
    the sought eigenstate and leaves the reported energy unbiased.
    """
    total = 0.0
    for sj in states:
        ov = np.vdot(sj, psi)
        total += ov.real * ov.real + ov.imag * ov.imag
    return beta * total


def spectral_width_beta(pauli_sum) -> float:
    r"""Automatic deflation weight from the Hamiltonian's coefficient 1-norm.

    The spectral radius is bounded by :math:`\sum_i |c_i|` over the Pauli
    coefficients, so :math:`2\sum_i |c_i|` safely exceeds every eigenvalue gap --
    a robust default penalty that guarantees deflation separates the levels.
    """
    total = sum(abs(complex(c)) for c in pauli_sum.simplify().terms.values())
    return 2.0 * float(total)


@dataclass
class EnergyLevels:
    """Molecular energy levels (ground + excited states) from a deflation run.

    Energies are stored in **Hartree**, ascending, mirroring the internal units of
    :class:`~carcara.algorithms.vqe.VQEResult` /
    :class:`~carcara.algorithms.adapt_vqe.ADAPTVQEResult`.
    """

    energies: np.ndarray                       # Hartree, ascending
    states: list = field(default_factory=list)  # optimal state vectors (optional)
    reference_energy: float | None = None      # ansatz reference (HF) energy, Ha
    num_evaluations: int = 0                   # total cost-function evaluations
    num_operators: list[int] | None = None     # operators per state (ADAPT only)

    @property
    def num_states(self) -> int:
        return int(len(self.energies))

    @property
    def ground_state_energy(self) -> float:
        return float(self.energies[0])

    @property
    def excitation_energies(self) -> np.ndarray:
        r"""Energies relative to the ground state, :math:`E_i - E_0` (Hartree)."""
        return np.asarray(self.energies, dtype=float) - float(self.energies[0])

    @property
    def gaps(self) -> np.ndarray:
        """Successive level spacings :math:`E_{i+1} - E_i` (Hartree)."""
        return np.diff(np.asarray(self.energies, dtype=float))

    def in_units(self, units: str = "eV") -> np.ndarray:
        """Return the energy levels converted to ``units`` (``"eV"`` or ``"Ha"``)."""
        return from_hartree(np.asarray(self.energies, dtype=float), units)

    def excitation_energies_in_units(self, units: str = "eV") -> np.ndarray:
        """Excitation energies :math:`E_i - E_0` converted to ``units``."""
        return from_hartree(self.excitation_energies, units)

    def __repr__(self) -> str:
        levels = ", ".join(f"{e:.6f}" for e in np.asarray(self.energies)[:6])
        more = "" if self.num_states <= 6 else f", ... (+{self.num_states - 6})"
        return f"EnergyLevels([{levels}{more}] Ha, num_states={self.num_states})"
