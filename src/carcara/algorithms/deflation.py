# -*- coding: utf-8 -*-
# file: algorithms/deflation.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Excited states by variational quantum deflation (VQD), as a driver mixin.

Variational quantum deflation (Higgott *et al.* 2019) finds excited states *one
after another*: after the :math:`m` lowest states
:math:`\{|\psi_j\rangle\}_{j<m}` are found, the next one minimizes the *deflated*
cost

.. math::

    L_m(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle
        + \beta\sum_{j<m}|\langle\psi_j|\psi(\vec\theta)\rangle|^2 ,

whose global minimum is the :math:`m`-th eigenstate provided :math:`\beta` exceeds
the spanned gaps.  :class:`DeflationMixin` implements the shared *outer* loop
(:meth:`~DeflationMixin.energy_levels`) once for every driver; each driver only
supplies :meth:`~DeflationMixin._deflated_ground`, the per-state solve on its own
ansatz (a fixed ansatz for :class:`~carcara.algorithms.vqe.VQE`, a freshly grown
one for :class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`).  The exact state-vector
backend makes the overlaps -- and hence the penalty -- exact.
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


class DeflationMixin:
    """Adds :meth:`energy_levels` (excited states by deflation) to a driver.

    The mixin owns the state-by-state outer loop, the automatic ``beta`` and the
    :class:`EnergyLevels` assembly; the concrete driver supplies
    :meth:`_deflated_ground`, which finds the lowest state orthogonal to the ones
    already found.  Expects the host to provide ``_configured``, ``_check_kpts``,
    ``hamiltonian`` and ``reference_energy`` (all on
    :class:`~carcara.algorithms.base.VariationalDriver`).
    """

    def energy_levels(self, num_states: int = 2, *, beta: float | None = None,
                      **solver_kwargs) -> EnergyLevels:
        r"""Molecular energy levels (ground + excited states) by deflation.

        Computes the ``num_states`` lowest eigenstates the driver's ansatz can
        reach.  Reported energies are the *bare* expectation values (the penalty
        vanishes at each eigenstate), so every returned level is a true eigenvalue
        of ``H`` within the reachable sector.

        Parameters
        ----------
        num_states : int
            Number of levels to return (``>= 1``); ``1`` is just the ground state.
        beta : float, optional
            Deflation penalty weight.  Defaults to a value from the Hamiltonian's
            coefficient 1-norm (guaranteed to exceed the spanned gaps).
        **solver_kwargs
            Forwarded to the driver's :meth:`_deflated_ground` -- e.g. ``restarts``
            / ``seed`` / ``initial_parameters`` (VQE) or ``max_iterations`` /
            ``gradient_tolerance`` (ADAPT-VQE).

        Returns
        -------
        EnergyLevels
            Ascending energies (Hartree), the optimal state vectors, and
            convenience views (``excitation_energies`` / ``in_units("eV")``).
        """
        if not getattr(self, "_configured", False):
            raise RuntimeError(
                f"{type(self).__name__} has no Hamiltonian; construct it with one, "
                "use it as an ASE calculator and evaluate an energy first, or "
                "call run()")
        self._check_kpts()
        if int(num_states) < 1:
            raise ValueError("num_states must be >= 1")
        if beta is None:
            beta = spectral_width_beta(self.hamiltonian)
        beta = float(beta)

        states: list[np.ndarray] = []
        energies: list[float] = []
        num_ops: list[int | None] = []
        total_evals = 0
        for m in range(int(num_states)):
            energy, psi, nev, nops = self._deflated_ground(
                states, beta, state_index=m, **solver_kwargs)
            energies.append(energy)
            states.append(psi)
            num_ops.append(nops)
            total_evals += nev

        order = np.argsort(energies)
        has_ops = any(n is not None for n in num_ops)
        return EnergyLevels(
            energies=np.asarray(energies, dtype=float)[order],
            states=[states[i] for i in order],
            reference_energy=self.reference_energy(),
            num_evaluations=total_evals,
            num_operators=([num_ops[i] for i in order] if has_ops else None))

    def _deflated_ground(self, states, beta, *, state_index, **kwargs):
        """Lowest state orthogonal to ``states``; ``(energy, psi, nfev, n_ops)``.

        ``n_ops`` is the operator count for adaptive drivers, ``None`` otherwise.
        """
        raise NotImplementedError
