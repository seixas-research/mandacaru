# -*- coding: utf-8 -*-
# file: algorithms/subspace.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Subspace-search VQE and ADAPT-VQE for simultaneous ground + excited states.

Where variational quantum deflation (:mod:`carcara.algorithms._energy_levels`)
finds excited states *one after another*, **subspace-search VQE** (SSVQE,
Nakanishi *et al.* 2019) finds the ground state and the first few excited states
**at once**, in a single optimization.

The idea: pick :math:`k` mutually orthogonal reference determinants
:math:`\{|\varphi_j\rangle\}`, send them all through the **same** parameterized
unitary :math:`U(\vec\theta)`, and minimize the *weighted* energy sum

.. math::

    L(\vec\theta) = \sum_{j=0}^{k-1} w_j\,
        \langle\varphi_j|U^\dagger(\vec\theta)\,H\,U(\vec\theta)|\varphi_j\rangle ,
    \qquad w_0 > w_1 > \dots > w_{k-1} > 0 .

Because :math:`U` is unitary the images :math:`U|\varphi_j\rangle` stay
orthonormal, and the descending weights force the largest weight onto the lowest
energy: at the optimum :math:`U|\varphi_0\rangle` is the ground state,
:math:`U|\varphi_1\rangle` the first excited state, and so on.  Each level's
reported energy is the *bare* expectation value
:math:`\langle\varphi_j|U^\dagger H U|\varphi_j\rangle`.

:class:`SubspaceVQE` uses a fixed ansatz; :class:`SubspaceADAPTVQE` grows one
shared adaptive ansatz whose pool-screening gradient is the weighted sum of the
per-reference gradients.  Both subclass the ground-state drivers and are ASE
calculators (the returned ASE energy is the ground state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter as _perf

import numpy as np

from ase.calculators.calculator import all_changes

from ..core.mapping import reference_qubit_bits
from ._energy_levels import EnergyLevels
from .adapt_vqe import ADAPTVQE, AdaptAnsatz, CircuitMetrics, profile_ansatz
from .vqe import VQE


# --------------------------------------------------------------------------- #
# Reference determinants for the search subspace.
# --------------------------------------------------------------------------- #

def subspace_determinants(n_qubits: int, occupied, num_states: int) -> list[tuple]:
    """The ``num_states`` lowest determinants of the Hartree-Fock particle sector.

    Enumerates Slater determinants with the same ``(n_alpha, n_beta)`` occupation
    as the Hartree-Fock reference ``occupied`` (spin-blocked spin-orbital
    indices), ordered by excitation level from HF then lexicographically, and
    returns the first ``num_states`` as tuples of occupied spin-orbital indices.
    These are mutually orthogonal and share the electron count, so the
    number-conserving ansatz keeps them in the physical sector.
    """
    M = n_qubits // 2
    occ = sorted(int(o) for o in occupied)
    n_alpha = sum(1 for o in occ if o < M)
    n_beta = len(occ) - n_alpha
    alpha_orbitals = range(M)
    beta_orbitals = range(M, n_qubits)
    hf = frozenset(occ)

    dets: list[tuple[int, tuple]] = []
    for a_sel in combinations(alpha_orbitals, n_alpha):
        for b_sel in combinations(beta_orbitals, n_beta):
            det = frozenset(a_sel) | frozenset(b_sel)
            level = len(det - hf)                 # number of promoted electrons
            dets.append((level, tuple(sorted(det))))
    dets.sort(key=lambda t: (t[0], t[1]))
    if num_states > len(dets):
        raise ValueError(
            f"requested {num_states} states but the particle-number sector only "
            f"has {len(dets)} determinants")
    return [d[1] for d in dets[:num_states]]


def _determinant_vector(mapping: str, n_qubits: int, occupied) -> np.ndarray:
    """Computational-basis state vector of a determinant (qubit 0 = MSB)."""
    bits = reference_qubit_bits(mapping, n_qubits, occupied)
    index = 0
    for i, bit in enumerate(bits):
        if bit:
            index |= 1 << (n_qubits - 1 - i)
    vec = np.zeros(2 ** n_qubits, dtype=complex)
    vec[index] = 1.0
    return vec


def reference_matrix(mapping: str, n_qubits: int, occupied,
                     num_states: int) -> np.ndarray:
    """Stack the ``num_states`` lowest determinants as columns ``(2**n, k)``."""
    dets = subspace_determinants(n_qubits, occupied, num_states)
    cols = [_determinant_vector(mapping, n_qubits, det) for det in dets]
    return np.column_stack(cols)


def resolve_weights(weights, num_states: int) -> np.ndarray:
    """Validate / default the SSVQE weights (strictly decreasing, positive)."""
    if weights is None:
        w = np.arange(num_states, 0, -1, dtype=float)     # k, k-1, ..., 1
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != num_states:
            raise ValueError(
                f"expected {num_states} weights, got {w.size}")
        if np.any(w <= 0):
            raise ValueError("weights must be positive")
        if num_states > 1 and np.any(np.diff(w) >= 0):
            raise ValueError("weights must be strictly decreasing")
    return w


# --------------------------------------------------------------------------- #
# Result containers.
# --------------------------------------------------------------------------- #

@dataclass
class SubspaceVQEResult:
    """Result of a :class:`SubspaceVQE` run (ground + excited states)."""

    energies: np.ndarray                 # per-level energy, ascending (Hartree)
    optimal_parameters: np.ndarray       # shared ansatz parameters
    weights: np.ndarray                  # SSVQE weights used
    states: list = field(default_factory=list)      # optimal state vectors
    reference_energy: float | None = None
    num_evaluations: int = 0
    success: bool = True
    timings: dict | None = None
    integration_profile: dict | None = None

    @property
    def optimal_energy(self) -> float:
        """Ground-state energy (lowest level) -- the ASE-facing energy."""
        return float(self.energies[0])

    @property
    def num_states(self) -> int:
        return int(len(self.energies))

    @property
    def excitation_energies(self) -> np.ndarray:
        return np.asarray(self.energies, float) - float(self.energies[0])

    @property
    def levels(self) -> EnergyLevels:
        """View as an :class:`~carcara.algorithms.EnergyLevels`."""
        return EnergyLevels(energies=np.asarray(self.energies, float),
                            states=list(self.states),
                            reference_energy=self.reference_energy,
                            num_evaluations=self.num_evaluations)

    def in_units(self, units: str = "eV") -> np.ndarray:
        return self.levels.in_units(units)

    def __repr__(self) -> str:
        levels = ", ".join(f"{e:.6f}" for e in np.asarray(self.energies))
        return (f"SubspaceVQEResult([{levels}] Ha, "
                f"num_states={self.num_states}, success={self.success})")


@dataclass
class SubspaceADAPTVQEResult:
    """Result of a :class:`SubspaceADAPTVQE` run (ground + excited states)."""

    energies: np.ndarray                 # per-level energy, ascending (Hartree)
    optimal_parameters: np.ndarray
    weights: np.ndarray
    converged: bool
    final_max_gradient: float
    operators: list = field(default_factory=list)   # selected operator labels
    states: list = field(default_factory=list)
    reference_energy: float | None = None
    num_evaluations: int = 0
    metrics: CircuitMetrics | None = None
    timings: dict | None = None
    integration_profile: dict | None = None

    @property
    def optimal_energy(self) -> float:
        return float(self.energies[0])

    @property
    def num_states(self) -> int:
        return int(len(self.energies))

    @property
    def num_operators(self) -> int:
        return len(self.operators)

    @property
    def excitation_energies(self) -> np.ndarray:
        return np.asarray(self.energies, float) - float(self.energies[0])

    @property
    def levels(self) -> EnergyLevels:
        return EnergyLevels(energies=np.asarray(self.energies, float),
                            states=list(self.states),
                            reference_energy=self.reference_energy,
                            num_evaluations=self.num_evaluations)

    def in_units(self, units: str = "eV") -> np.ndarray:
        return self.levels.in_units(units)

    def __repr__(self) -> str:
        levels = ", ".join(f"{e:.6f}" for e in np.asarray(self.energies))
        return (f"SubspaceADAPTVQEResult([{levels}] Ha, "
                f"num_states={self.num_states}, n_ops={self.num_operators}, "
                f"converged={self.converged})")


# --------------------------------------------------------------------------- #
# Subspace-search VQE (fixed ansatz).
# --------------------------------------------------------------------------- #

class SubspaceVQE(VQE):
    """Subspace-search VQE: ground + first excited states in one optimization.

    Extends :class:`~carcara.algorithms.VQE`; every constructor argument of
    ``VQE`` is accepted (including the ASE-calculator ``basis`` / ``h`` mode),
    plus:

    Parameters
    ----------
    num_states : int
        Number of levels (ground + excited) to compute simultaneously
        (default ``2``).
    weights : sequence of float, optional
        Strictly decreasing positive SSVQE weights, one per level.  Defaults to
        ``(k, k-1, ..., 1)``.

    Notes
    -----
    The reference determinants are the ``num_states`` lowest of the Hartree-Fock
    particle-number sector; the ansatz must supply ``evolve(theta, references)``
    (as :class:`~carcara.circuits.UCCSD` does) so one shared unitary can act on
    all of them.  As an ASE calculator the reported energy is the ground state;
    the full spectrum is on :attr:`subspace_result`.
    """

    def __init__(self, hamiltonian=None, ansatz=None, *, num_states: int = 2,
                 weights=None, **kwargs):
        if int(num_states) < 1:
            raise ValueError("num_states must be >= 1")
        self.num_states = int(num_states)
        self._weights_spec = weights
        super().__init__(hamiltonian, ansatz, **kwargs)

    # -- ASE calculator: also expose the spectrum on `subspace_result` ---- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.subspace_result = self.vqe_result

    # -- subspace evolution ---------------------------------------------- #

    def _references(self) -> np.ndarray:
        occupied = getattr(self.ansatz, "_occupied", None)
        if occupied is None:
            raise TypeError(
                "the ansatz does not expose its occupied orbitals; SubspaceVQE "
                "needs a determinant reference (e.g. a UCCSD ansatz)")
        return reference_matrix(self.mapping, self.n_qubits, occupied,
                                self.num_states)

    def _evolve(self, theta, references) -> np.ndarray:
        evolve = getattr(self.ansatz, "evolve", None)
        if evolve is None:
            raise TypeError(
                f"ansatz {type(self.ansatz).__name__} has no evolve(theta, "
                "references); SubspaceVQE needs it to share one unitary across "
                "the reference states")
        return evolve(theta, references)

    def run(self, initial_parameters=None) -> SubspaceVQEResult:
        """Optimize the shared unitary and return the ``num_states`` levels."""
        if not self._configured:
            raise RuntimeError(
                "SubspaceVQE has no Hamiltonian/ansatz; construct it with both, "
                "or use it as an ASE calculator with a `basis`")
        self._check_kpts()
        from ..integrals import _backend
        from ..utils.profiling import Timings

        k = self.num_states
        weights = resolve_weights(self._weights_spec, k)
        refs = self._references()
        H = self._h_matrix

        n = self.ansatz.num_parameters
        x0 = (np.zeros(n) if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        timings = Timings(n_cores=_backend.num_threads(),
                          backend="C (OpenMP)" if _backend.HAS_C_BACKEND
                          else "NumPy")
        run_t0 = self.__dict__.pop("_wall_start", None) or _perf()

        ref_energy = self.reference_energy()
        if self.verbose:
            from ..utils import banner
            banner.show()
            self._print_header(ref_energy)
            print(f"Subspace search: {k} states, weights = "
                  f"{np.array2string(weights, precision=3)}")

        def weighted_cost(theta):
            evolved = self._evolve(theta, refs)
            total = 0.0
            for j in range(k):
                psi = evolved[:, j]
                total += weights[j] * float(np.real(np.vdot(psi, H @ psi)))
            return total

        with timings.time("parameter optimization"):
            result = self.optimizer.minimize(weighted_cost, x0)

        evolved = self._evolve(result.x, refs)
        energies = np.array([float(np.real(np.vdot(evolved[:, j], H @ evolved[:, j])))
                             for j in range(k)])
        order = np.argsort(energies)
        states = [evolved[:, j].copy() for j in order]

        if self._integration_profile is not None:
            for name, secs in self._integration_profile.get("stages_s", {}).items():
                timings.add(f"integration: {name}", secs)
        timings.wall_time = _perf() - run_t0

        subspace = SubspaceVQEResult(
            energies=energies[order],
            optimal_parameters=np.asarray(result.x, float),
            weights=weights,
            states=states,
            reference_energy=ref_energy,
            num_evaluations=result.nfev,
            success=result.success,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile)

        if self.verbose:
            self._print_subspace_summary(subspace, timings)
        return subspace

    def _print_subspace_summary(self, result: SubspaceVQEResult, timings) -> None:
        rule = "=" * 70
        print(rule)
        status = "converged" if result.success else "did not converge"
        print(f"Subspace-VQE finished ({status}): {result.num_states} levels")
        for i, e in enumerate(result.energies):
            tag = "ground" if i == 0 else f"excited {i}"
            print(f"  E[{i}] ({tag:>9s}) = {e:+.8f} Ha")
        if timings is not None:
            print(timings.format_report())
        print(rule)


# --------------------------------------------------------------------------- #
# Subspace-search ADAPT-VQE (one shared, adaptively grown ansatz).
# --------------------------------------------------------------------------- #

class SubspaceADAPTVQE(ADAPTVQE):
    """Subspace-search ADAPT-VQE: grow one shared ansatz for several states.

    Extends :class:`~carcara.algorithms.ADAPTVQE`; accepts every ``ADAPTVQE``
    argument plus ``num_states`` / ``weights`` (as :class:`SubspaceVQE`).  A
    single adaptive ansatz :math:`U(\\vec\\theta)` is grown and applied to all
    reference determinants; the pool-screening gradient is the **weighted sum**
    of the per-reference gradients
    :math:`\\sum_j w_j\\,\\langle\\psi_j|[H, A_i]|\\psi_j\\rangle`, and the inner
    re-optimization minimizes the weighted energy.

    As an ASE calculator the reported energy is the ground state; the spectrum is
    on :attr:`subspace_result`.
    """

    def __init__(self, hamiltonian=None, pool="fermionic", *,
                 num_states: int = 2, weights=None, **kwargs):
        if int(num_states) < 1:
            raise ValueError("num_states must be >= 1")
        self.num_states = int(num_states)
        self._weights_spec = weights
        super().__init__(hamiltonian, pool, **kwargs)

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.subspace_result = self.adapt_result

    # -- weighted subspace gradient -------------------------------------- #

    def _weighted_gradients(self, evolved: np.ndarray,
                            weights: np.ndarray) -> np.ndarray:
        """Weighted-sum pool gradient over the reference states (columns)."""
        grads = np.zeros(len(self._pool_matrices))
        for j in range(evolved.shape[1]):
            grads += weights[j] * self._analytic_gradients(evolved[:, j])
        return grads

    def run(self, initial_parameters=None, *, geometry=None, cell=None,
            **_ignored) -> SubspaceADAPTVQEResult:
        """Grow the shared ansatz and return the ``num_states`` levels."""
        if not self._configured:
            raise RuntimeError(
                "SubspaceADAPTVQE has no Hamiltonian; construct it with one, or "
                "use it as an ASE calculator with a `basis`")
        self._check_kpts()
        from ..integrals import _backend
        from ..utils.profiling import Timings

        k = self.num_states
        weights = resolve_weights(self._weights_spec, k)
        refs = reference_matrix(self.mapping, self.n_qubits,
                                self.pool.occupied_orbitals, k)
        H = self._h_matrix
        max_iterations = self.max_iterations
        gradient_tol = self.gradient_tolerance

        timings = Timings(n_cores=_backend.num_threads(),
                          backend="C (OpenMP)" if _backend.HAS_C_BACKEND
                          else "NumPy")
        run_t0 = self.__dict__.pop("_wall_start", None) or _perf()

        ref_energy = self.reference_energy()
        if self.verbose:
            from ..utils import banner
            banner.show()
            self._print_header(ref_energy, self._energy_unit_label())
            print(f"Subspace search: {k} states, weights = "
                  f"{np.array2string(weights, precision=3)}")

        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping, sparse=self._sparse)
        params = (np.asarray(initial_parameters, float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        selected: list[str] = []
        total_evals = 0
        converged = False
        max_grad = np.inf

        for _ in range(max_iterations):
            with timings.time("gradient screening"):
                evolved = ansatz.evolve(params, refs)
                grads = self._weighted_gradients(evolved, weights)
            idx = int(np.argmax(np.abs(grads)))
            max_grad = float(abs(grads[idx]))
            if max_grad < gradient_tol:
                converged = True
                break

            op = self._pool_ops[idx]
            ansatz.append(op)
            selected.append(op.label)
            x0 = np.concatenate([params, [0.0]])

            def weighted_cost(theta, _ansatz=ansatz):
                ev = _ansatz.evolve(theta, refs)
                return sum(weights[j] * self.energy(ev[:, j]) for j in range(k))

            with timings.time("parameter optimization"):
                result = self.optimizer.minimize(weighted_cost, x0)
            params = np.asarray(result.x, float)
            total_evals += result.nfev

            if self.verbose:
                energies_now = [self.energy(ansatz.evolve(params, refs)[:, j])
                                for j in range(k)]
                print(f"\n[iter {len(selected)}] selected {op.label} "
                      f"(|grad|={max_grad:.6e})  "
                      f"E0={min(energies_now):+.8f} Ha")

        # Final per-level energies (bare expectation values).
        evolved = ansatz.evolve(params, refs)
        energies = np.array([self.energy(evolved[:, j]) for j in range(k)])
        order = np.argsort(energies)
        states = [evolved[:, j].copy() for j in order]

        metrics = (profile_ansatz(self.n_qubits, ansatz.occupied, ansatz.operators)
                   if self.profile else
                   CircuitMetrics(None, None, ansatz.num_parameters))

        if not converged and len(selected) == max_iterations:
            evolved = ansatz.evolve(params, refs)
            max_grad = float(np.max(np.abs(
                self._weighted_gradients(evolved, weights))))

        if self._integration_profile is not None:
            for name, secs in self._integration_profile.get("stages_s", {}).items():
                timings.add(f"integration: {name}", secs)
        timings.wall_time = _perf() - run_t0

        subspace = SubspaceADAPTVQEResult(
            energies=energies[order],
            optimal_parameters=params,
            weights=weights,
            converged=converged,
            final_max_gradient=max_grad,
            operators=selected,
            states=states,
            reference_energy=ref_energy,
            num_evaluations=total_evals,
            metrics=metrics,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile)

        if self.verbose:
            self._print_subspace_summary(subspace, timings)
        return subspace

    def _print_subspace_summary(self, result: SubspaceADAPTVQEResult,
                                timings) -> None:
        rule = "=" * 70
        print(rule)
        status = "converged" if result.converged else "not converged"
        print(f"Subspace-ADAPT-VQE finished ({status}): {result.num_states} "
              f"levels, {result.num_operators} operators, "
              f"final |grad| = {result.final_max_gradient:.6e}")
        for i, e in enumerate(result.energies):
            tag = "ground" if i == 0 else f"excited {i}"
            print(f"  E[{i}] ({tag:>9s}) = {e:+.8f} Ha")
        if timings is not None:
            print(timings.format_report())
        print(rule)
