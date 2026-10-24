# -*- coding: utf-8 -*-
# file: algorithms/adapt_vqe.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""ADAPT-VQE: adaptively grown variational ansatz.

:class:`ADAPTVQE` implements ADAPT-VQE (Grimsley *et al.*, 2019), which builds a
compact, problem-tailored ansatz one operator at a time instead of using a fixed
template.  Each macro-iteration:

1. evaluate the energy gradient of appending each pool operator :math:`A_i` at
   zero angle,
   :math:`g_i = \partial E/\partial\theta_i = \langle\psi(\vec\theta)|[H, A_i]
   |\psi(\vec\theta)\rangle = 2\,\mathrm{Re}\,\langle H\psi|A_i\psi\rangle`;
2. stop if :math:`\max_i |g_i| < \varepsilon`;
3. append :math:`e^{\theta_k A_{\mathrm{opt}}}` for the largest-gradient operator,
   initializing :math:`\theta_k = 0`;
4. re-optimize **all** parameters with the classical optimizer (warm-started from
   the previous optimum -- ADAPT's key efficiency property).

With ``shots = 0`` it is an exact state-vector implementation in the same
spirit as :class:`~mandacaru.algorithms.vqe.VQE`: the Hamiltonian and each
generator are materialized (dense, sparse, sector or matrix-free) and the
ansatz produces the exact state vector, so gradients and energies are
noiseless.  With ``shots > 0`` nothing is simulated: every energy is measured
on the device, every pool gradient after the first step is the measured
expectation value of its commutator observable ``[H, A_i]``
(:mod:`~mandacaru.backends.gradient_measurement`), the first step's are exact
at the reference determinant, and the classical memory stays polynomial in the
qubit count.

Each grown ansatz is also **profiled** for hardware cost: the parameterized
unitary is compiled to a native ``{CNOT, U}`` gate set with Qiskit and its CNOT
count and circuit depth are logged per iteration (see :class:`CircuitMetrics`).
The four operator pools (:mod:`mandacaru.circuits.pools`) can then be compared on
accuracy-per-CNOT.  Profiling is optional -- if Qiskit is unavailable the run
proceeds and metrics are reported as ``None``.

Beyond the ground state, :meth:`ADAPTVQE.energy_levels` returns the low-lying
**molecular energy levels** (ground + excited states) by variational quantum
deflation, growing a fresh deflated ansatz per level -- see
:mod:`mandacaru.algorithms.deflation`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..circuits.adapt_ansatz import AdaptAnsatz
from ..circuits.pools import GRADIENT_FLOOR, PoolOperator, _support_of
from ..circuits.profiling import CircuitMetrics
from ..backends.measurement import RunBudgetExceeded
from ..units import convert_energy, to_hartree
from .calculator import _method_key
from .convergence import Convergence
from .deflation import DeflationMixin, deflation_penalty
from .pool_driver import PoolDriver
from .proposal_data import EditRecorder, ProblemRecord, resolve_record

if TYPE_CHECKING:
    from ..optimizers.optim import Optimizer


#: Canonical names of the pool-screening gradient estimators.  Everything the
#: code writes, prints, documents and tests uses these spellings; the *input*
#: accepts any spelling of them (:func:`resolve_gradient_method`), the same way
#: ``method="ADAPT_VQE"`` names ``"adapt-vqe"``.
GRADIENT_METHODS = ("analytic", "finite_difference", "parameter_shift")

#: TETRIS-ADAPT-VQE (Anastasiou *et al.*, Phys. Rev. Research **6**, 013254,
#: 2024) grows the ansatz with every operator whose support is disjoint from
#: the ones already taken this iteration, in descending gradient order, instead
#: of only the largest.  The extra operators are free -- their gradients were
#: measured anyway -- and they compile into the same circuit layer, so the
#: ansatz gets denser and shallower.
#:
#: Pruned-ADAPT-VQE (Ramos-Calderer-style pruning, J. Chem. Theory Comput.
#: **21**, 8720, 2025) does the opposite at the other end: after each growth it
#: removes the one operator that has become irrelevant, ranked by a decision
#: factor f_i = F1(theta_i) F2(x_i) with F1 = 1 / theta^2 and
#: F2 = exp(-alpha x), x the relative position in the ansatz.  Only the
#: top-ranked operator is considered, and only when its parameter is below a
#: threshold set by the most recently added ones.
PRUNE_ALPHA = 10.0
#: Fraction of the recent mean amplitude that sets the removal threshold.
PRUNE_FRACTION = 0.1
#: How many of the most recently added operators define that mean.
PRUNE_RECENT = 4
#: Never prune below this many operators: the ranking needs a position scale.
PRUNE_MIN_OPERATORS = 2
#: Growth steps allowed per operator of ``max_iterations`` when pruning is on.
#: Pruning can remove the operator the next step re-adds -- the paper reports
#: exactly that near convergence -- and the loop counts *operators*, so without
#: a cap on the steps themselves such a cycle would never end.
PRUNE_STEP_BUDGET = 3

#: Step of the central difference used by ``gradient="finite_difference"``.  It
#: is that estimator's truncation-error scale -- the one number that says how
#: far its gradients can be from the exact ones -- so it is reported in the run
#: log rather than left as a default buried in the method.
FINITE_DIFFERENCE_STEP = 1e-4

#: What each estimator actually computes, one line each.  Written to the
#: ``[OPTIMIZATION SETUP]`` block as ``gradient_formula`` and printed in the
#: trace header, so the name in the log is never the reader's only clue.  No
#: value may contain a colon: the log parser splits ``key: value`` on the first
#: one.
GRADIENT_FORMULAS = {
    "analytic": "exact derivative, g = 2 Re<H psi|A psi>",
    "finite_difference": ("central difference of the energy, step = "
                          f"{FINITE_DIFFERENCE_STEP:g}"),
    "parameter_shift": ("parameter-shift rule, exact for the generator's "
                        "frequency set"),
    "measured": ("g = <psi|[H, A]|psi> from Pauli expectation values measured "
                 "on the device, exact (classical) at the reference "
                 "determinant"),
}

#: Spellings of the ``screening`` option other than ``{"operators": K}``.
SCREENING_MODES = ("auto", "full")


def resolve_screening(spec):
    """``"auto"``, ``"full"`` or ``{"operators": K}`` (``K >= 1``).

    Which pool operators each growth step screens: every one (``"full"``),
    the ``K`` with the largest gradients at the reference determinant
    (``{"operators": K}``), or ``"auto"`` -- every one on a local simulation,
    and on a shot-based run every one whose gradients fit **one** job of the
    measurement budget, else the largest such ranked subset
    (:meth:`ADAPTVQE._screened_operators`).
    """
    if isinstance(spec, str) and spec.strip().lower() in SCREENING_MODES:
        return spec.strip().lower()
    if isinstance(spec, Mapping):
        unknown = sorted(set(spec) - {"operators"})
        count = spec.get("operators")
        if not unknown and isinstance(count, (int, np.integer)) \
                and not isinstance(count, bool) and int(count) >= 1:
            return {"operators": int(count)}
    raise ValueError(f"screening must be 'auto', 'full' or "
                     f"{{'operators': K}} with K >= 1, got {spec!r}")


def resolve_gradient_method(name) -> str:
    """Canonical screening-gradient name; the input spelling does not matter.

    ``"parameter-shift"``, ``"parameter_shift"`` and ``"Parameter Shift"`` all
    name the same estimator, by the same rule that makes ``method="ADAPT_VQE"``
    name ``"adapt-vqe"`` (:func:`~mandacaru.algorithms.calculator._method_key`).
    Only the canonical underscore form of :data:`GRADIENT_METHODS` is ever
    written back out, so a log or a test never has two spellings of one method.

    Raises
    ------
    ValueError
        If ``name`` is not one of :data:`GRADIENT_METHODS` under any spelling.
    """
    canonical = {_method_key(method): method for method in GRADIENT_METHODS}
    resolved = canonical.get(_method_key(name))
    if resolved is None:
        raise ValueError(f"unknown gradient {name!r}; use one of "
                         f"{GRADIENT_METHODS}")
    return resolved


def _unique_frequencies(eigenvalues: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Unique positive eigenvalue differences (the frequencies of ``E(theta)``).

    ``E(theta) = <psi| e^{-theta A} H e^{theta A} |psi>`` for a generator with
    ``-iA`` eigenvalues ``{w_k}`` is a trigonometric polynomial with frequencies
    ``|w_k - w_l|``; this returns the distinct positive ones (clustered to
    ``tol``), used by the parameter-shift gradient.
    """
    w = np.asarray(eigenvalues, dtype=float)
    diffs = np.abs(w[:, None] - w[None, :]).ravel()
    diffs = diffs[diffs > tol]
    if diffs.size == 0:
        return np.empty(0)
    # Cluster near-equal differences so the frequency set stays small.
    rounded = np.round(diffs / tol) * tol
    return np.unique(rounded)


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #

@dataclass
class AdaptIteration:
    """Record of one ADAPT-VQE macro-iteration.

    ``energy`` is in the driver's output units (eV; Hartree with
    ``atomic_units=True``), like the enclosing :class:`ADAPTVQEResult`.
    """

    operator_label: str
    operator_kind: str
    max_gradient: float
    energy: float
    cnot_count: int | None
    depth: int | None
    num_parameters: int
    #: Steps the classical optimizer took to re-optimize the grown ansatz
    #: (parameter updates, not cost evaluations -- see
    #: :attr:`~mandacaru.optimizers.optim.OptimizeResult.nit`).
    optimizer_steps: int | None = None
    #: Cost evaluations the same re-optimization spent.
    num_evaluations: int | None = None


@dataclass
class ADAPTVQEResult:
    """Result of an :class:`ADAPTVQE` run.

    Every energy (``optimal_energy``, ``reference_energy``, the per-iteration
    ``energy_history``) is in :attr:`energy_unit` -- **eV** by default, Hartree
    when the driver was built with ``atomic_units=True``; :meth:`in_units`
    converts.
    """

    optimal_energy: float
    optimal_parameters: np.ndarray
    reference_energy: float
    converged: bool
    final_max_gradient: float
    operators: list[str]                      # selected operator labels, in order
    iterations: list[AdaptIteration] = field(default_factory=list)
    num_evaluations: int = 0                  # total inner cost evaluations
    #: Total classical-optimizer **steps** over every growth step (parameter
    #: updates; ``num_evaluations`` counts cost evaluations instead).
    optimizer_steps: int = 0
    #: ``(growth step, message)`` of every inner optimization that did not
    #: report convergence -- empty when every step converged.
    optimizer_failures: list = field(default_factory=list)
    #: ``(iteration, label)`` of every operator ``prune=True`` took back out.
    pruned_operators: list = field(default_factory=list)
    #: Where the run started: the empty ansatz, a checkpoint, or the
    #: previous geometry's ansatz (``transfer=True``), and why.
    start: str | None = None
    metrics: CircuitMetrics | None = None     # final compiled-circuit metrics
    timings: dict | None = None               # per-stage wall time / cores / memory
    integration_profile: dict | None = None   # real-space integration profile
    energy_unit: str = "eV"                   # unit of every energy above

    @property
    def num_operators(self) -> int:
        return len(self.operators)

    @property
    def energy_history(self) -> list[float]:
        return [it.energy for it in self.iterations]

    @property
    def correlation_energy(self) -> float:
        """Energy lowered relative to the reference (``E - E_ref``)."""
        return self.optimal_energy - self.reference_energy

    def in_units(self, units: str = "eV") -> float:
        """The optimal energy converted to ``units`` (``"eV"`` or ``"Ha"``)."""
        return float(convert_energy(self.optimal_energy, self.energy_unit, units))

    @property
    def gradient_history(self) -> list[float]:
        return [it.max_gradient for it in self.iterations]

    def __repr__(self) -> str:
        cnots = self.metrics.cnot_count if self.metrics else None
        return (f"ADAPTVQEResult(energy={self.optimal_energy:.6f}, "
                f"n_ops={self.num_operators}, cnots={cnots}, "
                f"converged={self.converged})")


# --------------------------------------------------------------------------- #
# ADAPT-VQE driver.
# --------------------------------------------------------------------------- #


def _max_abs(values) -> float:
    """``max |g|`` over the pool gradients -- ``0.0`` for an empty pool.

    A one-electron, one-orbital problem (the hydrogen atom in a minimal basis)
    has no excitation at all, so the pool is empty and the reference is already
    exact; an empty pool therefore reads as converged rather than as an error.
    """
    values = np.asarray(values, dtype=float).ravel()
    return float(np.max(np.abs(values))) if values.size else 0.0


#: Largest difference (Hartree) between a checkpoint's stored energy and the
#: energy its state evaluates to, for the *same* Hamiltonian, before the resume
#: says so.  It is a consistency check on the file, not a convergence threshold.
RESUME_ENERGY_TOLERANCE = 1e-8

#: Random parameter samples per expressivity estimate (two states each).
EXPRESSIVITY_SAMPLES = 400

#: Expressivity is **opt-in** (``run(log_expressivity=True)``).  The estimate
#: is ``2 * EXPRESSIVITY_SAMPLES`` state preparations and each one applies
#: *every* operator in the ansatz, so its cost is linear in the ansatz and
#: quadratic over a run: measured at 6 qubits, 0.010 / 0.031 / 0.059 / 0.125 s
#: at 1 / 4 / 8 / 16 operators.  It is a diagnostic, not a result, so nothing
#: pays for it unless it was asked for.


class ADAPTVQE(DeflationMixin, PoolDriver):
    """Adaptive VQE: an exact state-vector run, or a measured one on a device.

    **The internal layer**: reached only through the calculator,
    ``Mandacaru(method="adapt-vqe", ...)``, which forwards every option here.

    * **Direct** -- ``Mandacaru(method="adapt-vqe", hamiltonian=..., pool=...,
      num_particles=..., n_spatial_orbitals=...)`` and
      :meth:`~mandacaru.algorithms.Mandacaru.run`.
    * **ASE calculator** -- ``atoms.calc = Mandacaru(method="adapt-vqe", ...)``:
      ``atoms.get_total_energy()`` builds the Hamiltonian from the current
      geometry and runs.  ASE energies are returned in **eV**.

    Parameters
    ----------
    hamiltonian : PauliSum or Fermion, optional
        Qubit Hamiltonian, or a fermionic Hamiltonian mapped with ``mapping``.
        Omit it in calculator mode and let ``basis`` (or ``hamiltonian_builder``)
        build it from the geometry instead.
    pool : PoolBase or str
        The operator pool, or a name for :func:`~mandacaru.circuits.pools.build_pool`
        -- one of ``"ceo"``, ``"fermionic"``, ``"qubit"``, ``"qeb"``.  When a name
        is given ``n_spatial_orbitals`` and ``num_particles`` are required (in
        direct mode; in calculator mode the builder supplies them).
    basis : str or dict
        Basis set used to build the molecular Hamiltonian from an ASE geometry in
        calculator mode.  Either a name -- ``"HAO"`` (default; Full Atomic
        Orbitals), ``"NAO"``, ``"GTO"``/``"STO-3G"``, ``"6-31G(d)"`` (localized
        real-space families) or ``"PW"`` (periodic plane waves) -- or a
        ``{"name": ..., <options>}`` dict passing that family's options, e.g.
        ``{"name": "NAO", "energy_shift": 0.03}``, ``{"name": "GTO",
        "n_gaussians": 3}`` or ``{"name": "PW", "energy_cutoff": 300}``.
    num_particles : (int, int), optional
        ``(n_alpha, n_beta)``; required to build a pool from a name and to set the
        Hartree-Fock reference.  Inferred from the pool object otherwise.
    n_spatial_orbitals : int, optional
        Number of spatial orbitals; required to build a pool from a name.
    optimizer : str, dict or Optimizer
        Classical optimizer for the inner re-optimization.  A method name --
        one of ``"SPSA"``, ``"COBYLA"``, ``"Nelder-Mead"``, ``"SLSQP"``
        (default), ``"L-BFGS"``, ``"BFGS"`` -- takes the library's budget and
        tolerance; ``{"method": ..., "maxiter": ..., "tol": ...}`` sets them
        without importing anything; a pre-built
        :class:`~mandacaru.optimizers.optim.Optimizer` is what the dict builds.
    mapping : str
        Fermion-to-qubit mapping -- one of ``"jordan_wigner"`` (default),
        ``"parity"``, ``"bravyi_kitaev"`` -- used when ``hamiltonian`` is a
        ``Fermion`` and to build a named fermionic pool.
    gradient : str
        How the pool screening gradients are evaluated -- ``"analytic"``
        (default; the exact derivative ``g_i = 2 Re<H psi|A_i psi>``),
        ``"finite_difference"`` (a central difference of the energy at shifted
        parameters, step :data:`FINITE_DIFFERENCE_STEP`) or
        ``"parameter_shift"`` (the quantum parameter-shift rule).  The spelling
        does not matter on input -- ``"parameter-shift"`` and ``"Parameter
        Shift"`` name the same estimator -- but the canonical underscore form
        of :data:`GRADIENT_METHODS` is what is stored, logged and printed
        (:func:`resolve_gradient_method`).  All three converge to the same
        number; ``"analytic"`` is both the cheapest -- one matrix-vector product
        per pool operator, against ``2 x |pool|`` energy evaluations *plus*
        ``|pool|`` dense diagonalizations the shift estimators need -- and the
        only one free of a step-size truncation error, so the shift-based
        estimators are opt-in, for studying the estimator itself.  The method
        actually used is reported as ``gradient_method`` in the
        ``[OPTIMIZATION SETUP]`` block of the ``output.txt`` log.
        ``"analytic"`` also gives a gradient optimizer (SLSQP, BFGS, L-BFGS,
        CG) the exact gradient in the ansatz parameters on an exact objective
        (:meth:`~mandacaru.circuits.adapt_ansatz.AdaptAnsatz.energy_and_gradient`,
        the adjoint method); the other two leave it the optimizer's own finite
        difference.  The log's ``parameter_gradient`` line says which ran.
    device : str
        Execution device -- ``"AER_simulator"`` (default; ideal simulator),
        a noiseless fake backend (``"fake_kingston"``, ...), an IBM Quantum
        device (``"ibm-quantum"``, ``"ibm_torino"``, ...) or an Amazon Braket
        device (``"braket-sv1"``, ...).  See
        :mod:`mandacaru.backends.hardware`.
    max_iterations : int
        Maximum number of operators to append before stopping (default ``50``).
        Used as the default for :meth:`run` / the ASE-calculator evaluation.
    convergence : dict, optional
        When growth stops: ``{"gradient": ..., "energy": ...}``, thresholds on
        the largest pool gradient (Hartree) and on the energy change of the
        last growth step (eV; the log's ``dE``).  A criterion set to ``None`` or
        left out is not used; with both set, both must hold, and a vanished
        gradient (below ``1e-5``) stops the growth regardless.  Default
        ``{"gradient": 1e-3, "energy": 1e-3}``
        (:mod:`~mandacaru.algorithms.convergence`).
    txt : str, optional
        Path of the structured runtime log (``txt="output.txt"`` by
        convention).  The default, ``None``, writes no file -- a verbose run
        then prints **the same blocks** to standard output instead, so the
        screen and the file never disagree about what a run reported.  See
        :meth:`~mandacaru.algorithms.base.VariationalDriver.log_targets`.
    profile : bool
        Compile and profile the ansatz each iteration (default ``True``).
    verbose : bool
        Print a live trace of the quantum simulation to standard output (default
        ``True``): the qubit Hamiltonian as Pauli strings before the loop, and the
        selected operator's generator as Pauli strings at each iteration.
    sparse : bool or str
        Memory strategy for the operator pool (default ``"auto"``).  A dense pool
        materializes every operator's ``2^n x 2^n`` matrix and eigendecomposition,
        which is intractable beyond ~11 qubits (tens of GB for a 12-qubit water
        active space).  The sparse pool keeps the generators as sparse matrices and
        screens with the exact analytic gradient, densifying only the few
        *selected* operators; ``"auto"`` enables it for ``n_qubits >= 10``, ``True``
        / ``False`` force it.  ``"matrix-free"`` stores no matrix at all: the
        Hamiltonian, the pool and the ansatz generators are applied as Pauli
        strings (:mod:`mandacaru.core.matrix_free`), slower per product but
        bounded by the state vectors alone; ``"auto"`` switches to it when the
        stored matrices might not fit in a quarter of the memory limit (see
        :doc:`/guide/scalability`).  In sparse and matrix-free mode screening
        always uses the analytic gradient (the ``gradient`` argument's
        estimators need the dense eigendecompositions and are unavailable).
    sector : bool or str
        Simulate only the ``(n_alpha, n_beta)`` particle-number sector
        (default ``"auto"``).  The Hamiltonian and every pool generator are
        restricted to the sector's basis states
        (:class:`~mandacaru.core.sector.ParticleSector`) and state vectors carry
        ``C(M, n_alpha) C(M, n_beta)`` amplitudes instead of ``2^n`` -- 100
        instead of 1,048,576 for H2 or LiH in a DZP basis.  Exact, since the
        operators conserve particle number.  ``"auto"`` enables it for
        ``n_qubits >= 16`` (where a full-register sparse Hamiltonian no longer
        fits in memory) whenever states are prepared internally; ``True`` /
        ``False`` force it.  Incompatible with executing the ansatz as a
        circuit (``execute_circuits``), which prepares full-register states,
        and with ``shots > 0``, which simulates no state at all: there
        ``"auto"`` builds no sector and ``True`` is refused.
    atomic_units : bool
        Units used in the ``output.txt`` log.  ``False`` (default) logs energies
        in **eV** and lengths in **Angstrom**; ``True`` logs Hartree and Bohr.
        (ASE's ``get_total_energy`` always returns eV, per the ASE convention.)
    record : str, Path or bool, optional
        Record every growth step into an edit store
        (:mod:`~mandacaru.algorithms.proposal_data`), the training data of
        VALQA's sequence model: a path names the store, ``True`` the shared
        store (``MANDACARU_PROPOSAL_DATA``), and ``None`` or ``False`` (the
        default) records nothing.  Each row is the insertion the step made --
        the operators so far, the appended pool index, every pool gradient
        measured at that state and the energy change -- written as the step
        completes.  Not available with ``tetris`` or ``prune``, whose steps are
        not one appended operator.
    transfer : bool
        Along an ASE trajectory (a relaxation, a scan, dynamics), start each
        geometry from the ansatz the previous geometry ended with -- its
        operators and angles, followed through the molecular orbitals
        (:mod:`~mandacaru.algorithms.orbital_tracking`) -- re-optimize every
        angle here, and let the pool gradient decide whether it stands as it
        is or grows (default ``False``).  It is kept only if its relaxed
        energy lies below the reference, and refused, with the reason in the
        result's ``start``, for another system or pool, an orbital match below
        ``transfer_threshold``, or an operator the matching cannot follow.
    transfer_threshold : float, optional
        Smallest matched orbital overlap at which the ansatz is still carried
        (default
        :data:`~mandacaru.algorithms.orbital_tracking.DEFAULT_TRANSFER_THRESHOLD`).
    grid : Grid, optional
        Explicit real-space integration grid for the calculator-mode ``basis``
        builder.  When omitted the grid is generated automatically from the ASE
        ``atoms.cell`` at resolution ``grid_spacing``; a unit cell is then required.
    grid_spacing : float
        Target grid spacing in **Angstrom** (default ``0.20``) for the automatic
        cell-based grid used when ``grid`` is not given.  Finer ``grid_spacing`` (e.g.
        ``0.10``) gives a denser grid and more accurate one-/two-body integrals.
    kpts : (int, int, int) or dict, optional
        Monkhorst-Pack k-point mesh, resolved with ASE
        (:func:`ase.dft.kpoints.monkhorst_pack`); default ``None`` (a single
        Gamma point).  Accepts a size triple ``(n1, n2, n3)`` or the ASE dict
        ``{"size": (n1, n2, n3), "gamma": True}`` (``gamma=True`` centers the mesh
        on Gamma).  The real-space engine solves a Gamma-point (molecular)
        problem, so a denser mesh is generated and exposed on :attr:`kpoints` but
        raises ``NotImplementedError`` at run time.
    spin : bool
        Kept for compatibility; the reference spin state is read from the
        geometry's initial magnetic moments (``Atoms(..., magmoms=...)``): their
        rounded total is the number of unpaired electrons.  Without magnetic
        moments an even electron count is a closed-shell singlet and an odd
        count a doublet.  Odd-electron (open-shell) systems are built in the
        UHF natural-orbital basis, even counts in the RHF basis -- see
        :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.molecular_hamiltonian`.
    initial_state : str, optional
        The ansatz reference state; ``"hartree-fock"`` (default) is the
        Hartree-Fock determinant.  ``None`` is treated as ``"hartree-fock"``.
    charge : int
        Total charge, used to set the electron count in the ``basis`` builder.
    n_electrons : int, optional
        Explicit electron count for the ``basis`` builder (overrides ``charge``).
    active_space : dict, optional
        Which spatial molecular orbitals reach the register (default ``None``:
        all of them), as ``{"method": ..., "orbitals": ..., "threshold": ...,
        "frozen": ...}`` -- see
        :func:`~mandacaru.algorithms.active_space.resolve_active_space_spec`
        and :doc:`/guide/active_space`.  ``"frozen"`` is the frozen-core
        approximation: ``"auto"`` (or ``True``) the chemical noble-gas core, an
        integer that many lowest molecular orbitals, a list those explicit
        doubly occupied orbitals; they are replaced by their mean-field
        contribution -- a constant core energy plus an effective one-body
        potential.  ``"orbitals"`` (a count, ``{"occupied": n, "virtual":
        m}`` or explicit indices) and ``"threshold"`` (an occupation) drop
        virtual orbitals as well, ranked by ``"method"`` (``"energy"``,
        ``"mp2"`` or ``"natural"``).  The ansatz, pool and qubit count are
        built for what is left.
    hamiltonian_builder : callable, optional
        ``atoms -> (hamiltonian, num_particles, n_spatial_orbitals)``.  An
        explicit override for the built-in ``basis`` builder in calculator mode.
    save_hamiltonian : bool or str
        Write the qubit Hamiltonian (as Pauli strings) to disk once it has been
        built (default ``False``); ``True`` uses ``"hamiltonian"`` plus the
        extension of ``hamiltonian_format``, a string is used as the path.  See
        :mod:`mandacaru.core.serialization`.
    hamiltonian_format : {"parquet", "json"}
        Format written by ``save_hamiltonian`` (default ``"parquet"``).
        ``"json"`` is plain text and needs no Parquet engine.  Loading
        auto-detects the format, so the two are interchangeable.
    load_hamiltonian : str, optional
        Path of a Hamiltonian file written by ``save_hamiltonian``.  The qubit
        Hamiltonian is then read from disk and **the molecular integrals and the
        fermion-to-qubit mapping are skipped entirely**; because the file also
        records ``num_particles`` and ``n_spatial_orbitals``, the pool is rebuilt
        without a geometry, so a loaded driver runs directly (no ``Atoms``
        needed).  This makes a pool / optimizer / temperature sweep over the same
        molecule essentially free after the first build.
    backend_provider : str
        Quantum SDK used to construct the ansatz circuits -- ``"qiskit"``
        (default), ``"braket"`` (``amazon-braket-sdk``) or ``"cirq"``.  Circuit
        *profiling* always uses this SDK; circuit *execution* is governed by
        ``execute_circuits``.  See :mod:`mandacaru.backends.providers`.
    execute_circuits : bool, optional
        Prepare each ansatz state by executing the compiled circuit on the
        provider's local state-vector simulator, instead of the internal NumPy /
        sparse state-vector backend.  Defaults to ``True`` for ``"braket"`` and
        ``"cirq"`` and ``False`` for ``"qiskit"`` (whose fast default numerics are
        kept unless execution is requested).  The Pauli-rotation decomposition is
        exact for these generators, so all providers agree with the internal
        backend to machine precision -- only the runtime differs.
    quenching : bool
        Dynamic parametrization (default ``True``).  ``True`` re-optimizes **all**
        variational parameters at every growth step -- standard ADAPT-VQE, and
        what makes the method reach FCI.  ``False`` optimizes only the newly
        appended parameter and freezes every earlier one at its previous optimum:
        a much cheaper one-dimensional line search per step, at the cost of
        variational freedom (the energy is then an upper bound to the quenched
        result).
    screening : str or dict
        Which pool operators each growth step screens (default ``"auto"``).
        ``"full"`` screens every one; ``{"operators": K}`` the ``K`` with the
        largest gradients at the reference determinant (computed exactly and
        classically), a fixed candidate set; ``"auto"`` screens every one on a
        local simulation and, with ``shots > 0``, every one whose gradients
        fit **one** job of the measurement budget -- else the largest ranked
        subset that does, with a warning and the reason in the
        ``[GRADIENT MEASUREMENT]`` block.  With ``shots > 0`` every gradient
        after the first step is **measured on the device** as the
        expectation value of the commutator ``[H, A_k]``
        (:mod:`~mandacaru.backends.gradient_measurement`); the first step's,
        at the reference determinant, are exact and classical.  No state
        vector is ever built (see :doc:`/guide/measurement_cost`).
    run_options : dict, optional
        Extra keyword arguments forwarded to :meth:`run` on each calculator
        evaluation (e.g. ``{"log_expressivity": False}``).  Only arguments that
        :meth:`run` accepts are valid here -- the stopping controls
        (``max_iterations`` / ``convergence``), ``txt`` and ``verbose``
        are constructor arguments, not ``run`` arguments.
    """

    _default_sparse = "auto"
    #: The total-N sector and the ``"spin-orbit"`` pool (see
    #: :meth:`_resolve_spin_conservation`).
    _supports_spin_orbit = True
    #: With ``shots > 0`` the pool gradients are measured on the device.
    _measures_pool_gradients = True

    citation_method = "adapt-vqe"
    #: The name in the ``[SYSTEM]`` block's title.
    log_title = "ADAPT-VQE"
    #: ADAPT's ``run()`` writes the ``txt=`` log and honors checkpoints.
    writes_output_log = True

    def __init__(self,
                 hamiltonian=None,
                 pool="fermionic",
                 basis="HAO",
                 num_particles=None,
                 n_spatial_orbitals=None,
                 optimizer: str | Optimizer | None = None,
                 mapping: str = "jordan_wigner",
                 gradient: str = "analytic",
                 device: str = "AER_simulator",
                 max_iterations: int = 50,
                 convergence: Convergence | Mapping[str, float | None]
                 | None = None,
                 profile: bool = True,
                 verbose: bool = True,
                 sparse: bool | str = "auto",
                 sector: bool | str = "auto",
                 tetris: bool = False,
                 prune: bool = False,
                 atomic_units: bool = False,
                 record=None,
                 transfer: bool = False,
                 transfer_threshold: float | None = None,
                 screening="auto",
                 **driver_kwargs):
        # Everything else -- the problem setup, the Hamiltonian cache, the
        # circuit providers, checkpoints -- is a VariationalDriver option and is
        # forwarded untouched, so its name and default live in one place.  The
        # pool options are the PoolDriver's, shared with MCAS-VQE.
        super().__init__(pool=pool, profile=profile, sector=sector,
                         optimizer=optimizer, mapping=mapping, basis=basis,
                         device=device, verbose=verbose, sparse=sparse,
                         atomic_units=atomic_units, **driver_kwargs)

        # Validate the enumerated gradient option up front, and keep it in the
        # canonical spelling: `self.gradient` is what the log, the trace and the
        # tests read back, so it must not depend on how the user typed it.
        self.gradient = resolve_gradient_method(gradient)
        # Two growth strategies, independent of each other and of the pool.
        for name, value in (("tetris", tetris), ("prune", prune)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be True or False, got {value!r}")
        self.tetris = tetris
        self.prune = prune
        # Recording is opt-in for ADAPT-VQE: None does not mean the shared
        # store here, unlike the chain methods.
        self.record = (None if record is None or record is False
                       else resolve_record(record))
        if self.record is not None and (tetris or prune):
            raise ValueError("record= needs one appended operator per growth "
                             "step; it is not available with tetris or prune")

        from .orbital_tracking import DEFAULT_TRANSFER_THRESHOLD

        if not isinstance(transfer, bool):
            raise ValueError(f"transfer must be True or False, got "
                             f"{transfer!r}")
        #: Start each geometry of an ASE trajectory from the ansatz the
        #: previous one reported (:meth:`inherit_ansatz`).
        self.transfer = transfer
        self.transfer_threshold = float(
            DEFAULT_TRANSFER_THRESHOLD if transfer_threshold is None
            else transfer_threshold)
        self._inherited = None
        #: Where the last run started from, one line for the log and the
        #: result.
        self.start = None

        #: Which pool operators a growth step screens
        #: (:func:`resolve_screening`).
        self.screening = resolve_screening(screening)
        self._screened = None

        # Run defaults (also the defaults for the ASE-calculator evaluation).
        self.max_iterations = int(max_iterations)
        self.convergence = Convergence.resolve(convergence)

        # Seeded RNG for reproducible expressivity logging (the run log).
        self._expr_rng = np.random.default_rng(0)
        self._adopt_problem(hamiltonian, num_particles, n_spatial_orbitals)

    # -- setup helpers ---------------------------------------------------- #

    def _configure(self, hamiltonian, num_particles, n_spatial_orbitals):
        """Reset ADAPT's caches, then configure the pool (:class:`PoolDriver`)."""
        # The dense pool's per-operator eigendecomposition is built lazily
        # (``_pool_eigendecomposition``): only the shift-based estimators
        # need it, and it is |pool| dense diagonalizations of the full
        # register that the default analytic screening never looks at.
        self._pool_eig = None
        # The screened operators and their measurement plan belong to this
        # Hamiltonian and pool.
        self._screened = None
        #: Standard errors of the last measured screening (pool order), or
        #: ``None`` before one.
        self.gradient_errors = None
        super()._configure(hamiltonian, num_particles, n_spatial_orbitals)

    # -- energy / gradient ------------------------------------------------ #

    def _select_disjoint(self, grads: np.ndarray) -> list:
        """TETRIS-ADAPT-VQE: pool indices to grow with this iteration.

        The largest gradient, then the largest remaining one whose support does
        not touch any qubit already taken, and so on until the register is
        covered or nothing disjoint is left.  The gradients were measured for
        the whole pool anyway, so the extra operators cost no measurement; they
        act on different qubits, so they occupy the same circuit layer and the
        ansatz grows denser instead of deeper (Anastasiou *et al.*, Phys. Rev.
        Research 6, 013254, 2024).

        Falls back to the single largest gradient when the pool is not
        disjointly partitionable at this step, so it is never worse than the
        ordinary growth.
        """
        order = np.argsort(np.abs(grads))[::-1]
        chosen, covered = [], set()
        for idx in order:
            if abs(grads[idx]) <= GRADIENT_FLOOR:
                break
            support = set(self._pool_ops[int(idx)].support)
            if chosen and (support & covered):
                continue
            chosen.append(int(idx))
            covered |= support
            if len(covered) >= self.n_qubits:
                break
        return chosen or [int(np.argmax(np.abs(grads)))]

    def _prune_ansatz(self, ansatz, params: np.ndarray):
        """Pruned-ADAPT-VQE: drop the one operator that has become irrelevant.

        Ranks the ansatz by ``f_i = F1(theta_i) F2(x_i)`` -- ``F1 = 1/theta^2``
        favors near-zero amplitudes, ``F2 = exp(-alpha x)`` favors operators
        early in the ansatz, which are the ones a later operator has rendered
        redundant -- and removes the top-ranked one only if its amplitude is
        under ``PRUNE_FRACTION`` of the mean amplitude of the ``PRUNE_RECENT``
        most recent operators.  Recently added operators keep their parameters
        small while still doing work, which is why the threshold is measured
        against *them* rather than being absolute.

        Returns ``(params, removed_label)``; the parameters are not
        re-optimized, since the removed operator was by construction doing
        almost nothing (J. Chem. Theory Comput. 21, 8720, 2025).
        """
        n = int(ansatz.num_parameters)
        if n <= PRUNE_MIN_OPERATORS:
            return params, None
        theta = np.asarray(params, dtype=float).ravel()
        recent = np.abs(theta[-min(PRUNE_RECENT, n):])
        threshold = PRUNE_FRACTION * float(np.mean(recent))
        position = np.arange(1, n + 1) / n
        with np.errstate(divide="ignore", over="ignore"):
            factor = np.exp(-PRUNE_ALPHA * position) / np.square(theta)
        # The newest operator is never pruned: it was just selected on the
        # largest gradient, and removing it would make the loop cycle.
        factor[-1] = -np.inf
        candidate = int(np.argmax(factor))
        if abs(theta[candidate]) >= threshold:
            return params, None
        removed = ansatz.remove(candidate)
        return np.delete(theta, candidate), removed.label

    def _operator_gradient(self, op: PoolOperator, psi: np.ndarray) -> float:
        """``2 Re<H psi | A psi>`` for a single operator, pooled or not."""
        return 2.0 * float(np.real(np.vdot(self._h_matrix @ psi,
                                           self._operator_matrix(op) @ psi)))

    def _grow(self, ansatz, selected: list, op: PoolOperator, gradient) -> int:
        """Append what ``op`` grows into; returns the number of new parameters.

        One operator for every pool but the coupled-exchange one, where the
        selected element may expand into the several excitations it couples
        (the paper's MVP-CEO).  ``gradient`` is the energy derivative of a
        candidate at the current state, which differs between the ground-state,
        deflated and subspace loops, so each caller supplies its own.
        """
        grown = self.pool.grown_operators(op, gradient)
        for new in grown:
            ansatz.append(new)
            selected.append(new.label)
        return len(grown)

    def _pool_eigendecomposition(self):
        r"""Per-operator eigendecomposition of the dense pool, built on demand.

        For ``A`` anti-Hermitian ``-iA`` is Hermitian: ``-iA = V diag(w) V^dag``,
        so ``exp(theta A) = V diag(exp(i theta w)) V^dag`` and the energy along a
        pool direction is had without a matrix exponential.  The unique positive
        eigenvalue *differences* are the frequencies of ``E(theta)``, which the
        parameter-shift reconstruction needs.

        Only the shift-based estimators call this, and it diagonalizes every pool
        operator over the full register -- so it is not part of configuring a run.
        """
        if self._pool_eig is None:
            self._pool_eig = []
            for a in self._pool_matrices:
                dense = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
                w, V = np.linalg.eigh(-1j * dense)
                self._pool_eig.append((w, V, _unique_frequencies(w)))
        return self._pool_eig

    def _pool_energy_at(self, psi: np.ndarray, i: int, theta: float) -> float:
        r"""Energy after appending ``exp(theta A_i)`` to ``psi``.

        ``E_i(theta) = <psi| e^{-theta A_i} H e^{theta A_i} |psi>`` evaluated from
        the cached eigendecomposition of ``A_i`` (no matrix exponential).
        """
        w, V, _ = self._pool_eigendecomposition()[i]
        c = V.conj().T @ psi
        phi = V @ (np.exp(1j * theta * w) * c)
        return float(np.real(np.vdot(phi, self._h_matrix @ phi)))

    def _finite_difference_gradients(
            self, psi: np.ndarray,
            eps: float = FINITE_DIFFERENCE_STEP) -> np.ndarray:
        r"""Classical gradient: central finite difference in each pool direction.

        ``g_i ~= [E_i(+eps) - E_i(-eps)] / (2 eps)`` -- a purely classical
        estimate that evaluates the energy at *shifted parameter* values.  The
        step is :data:`FINITE_DIFFERENCE_STEP`, and the run log reports it
        (``gradient_formula``) because it is this estimator's truncation-error
        scale.
        """
        grads = np.empty(len(self._pool_matrices))
        for i in range(len(self._pool_matrices)):
            plus = self._pool_energy_at(psi, i, eps)
            minus = self._pool_energy_at(psi, i, -eps)
            grads[i] = (plus - minus) / (2.0 * eps)
        return grads

    def _parameter_shift_gradients(self, psi: np.ndarray) -> np.ndarray:
        r"""Quantum gradient via the parameter-shift rule.

        ``E_i(theta)`` is a finite trigonometric polynomial whose frequencies are
        the unique positive eigenvalue differences of the generator.  Its odd
        part ``[E_i(theta) - E_i(-theta)]/2 = sum_r b_r sin(omega_r theta)`` is
        sampled at symmetric shifts ``+/- theta_j`` and the ``b_r`` recovered by a
        small linear solve; the derivative at zero is ``sum_r omega_r b_r``.  For
        a single-Pauli generator (one frequency) this reduces to the textbook
        two-term shift and is exact; the multi-frequency reconstruction keeps it
        exact for every pool.
        """
        grads = np.empty(len(self._pool_matrices))
        for i in range(len(self._pool_matrices)):
            _, _, freqs = self._pool_eigendecomposition()[i]
            grads[i] = self._psr_one(psi, i, freqs)
        return grads

    def _psr_one(self, psi, i, freqs) -> float:
        R = len(freqs)
        if R == 0:
            return 0.0
        # Symmetric shift points; scaled by 1/omega_max to keep arguments in
        # (0, pi].  2R points over-determine the R sine coefficients (exact).
        base = np.linspace(0.0, np.pi, 2 * R + 1)[1:]
        thetas = base / float(freqs.max())
        odd = np.array([(self._pool_energy_at(psi, i, t)
                         - self._pool_energy_at(psi, i, -t)) / 2.0
                        for t in thetas])
        S = np.sin(np.outer(thetas, freqs))          # (2R, R)
        b, *_ = np.linalg.lstsq(S, odd, rcond=None)
        return float(np.dot(freqs, b))

    def _gradients(self, psi: np.ndarray) -> np.ndarray:
        """Pool screening gradients using the configured :attr:`gradient` method.

        The default ``"analytic"`` is the exact derivative the shift-based
        estimators approximate, and screening is the same quantity on the dense,
        sparse and sector paths.  The sparse and sector paths keep their
        generators as sparse matrices and never form the eigendecompositions the
        shift estimators need, so they screen analytically whatever is asked --
        the estimator choice is a dense-path study of the estimator itself.
        """
        if getattr(self, "_sparse", False) or self._sector is not None:
            return self._analytic_gradients(psi)
        if self.gradient == "parameter_shift":
            return self._parameter_shift_gradients(psi)
        if self.gradient == "finite_difference":
            return self._finite_difference_gradients(psi)
        return self._analytic_gradients(psi)                      # "analytic"

    def _select_operator(self, grads: np.ndarray, iteration: int) -> int:
        """Index of the pool operator to append this iteration.

        ADAPT-VQE's greedy rule: the largest-magnitude gradient.  This is the
        single **selection hook** subclasses override to change *which* operator
        grows the ansatz.  Convergence
        (``max|grad| < tol``) is decided by the caller, independently of the
        selection, so overriding this never changes the stopping criterion.
        """
        return int(np.argmax(np.abs(grads)))

    # -- screening: which operators, and where their gradients come from -- #

    def _reference_gradients(self) -> np.ndarray:
        """Exact pool gradients at the reference determinant, from the Pauli
        algebra alone (no state, no measurement)."""
        return self.pool_gradients.reference_gradients(
            self._new_ansatz().reference_qubits())

    def _gradient_job_limits(self) -> tuple:
        """``(bases, strings)`` a gradient job may hold: the measurement
        budget's ceilings, and the provider's ``max_bases_per_job``."""
        from ..backends.measurement import resolve_measurement_budget

        limits = resolve_measurement_budget(self.measurement_budget)
        provider = self.circuit_provider()
        caps = [c for c in (getattr(provider, "max_bases_per_job", None),
                            limits["bases"]) if c]
        return (min(caps) if caps else None), limits["observables"]

    def _screened_operators(self) -> tuple:
        """``(pool indices screened each step, why it is not every one)``.

        ``"full"`` -- and ``"auto"`` on a local simulation -- is every
        operator.  Otherwise the pool is **ranked** by the exact gradient at
        the reference determinant (largest magnitude first, ties in pool
        order) and ``{"operators": K}`` keeps the first ``K``.  ``"auto"`` on
        a shot-based run keeps the whole pool when its gradient step fits one
        job of the measurement budget, else the longest ranked prefix that
        does (found by doubling, then bisection; each probe is a classical
        count).  A prefix of none means a single operator's commutator does
        not fit a job, which is refused.
        """
        if self._screened is not None:
            return self._screened
        n = len(self._pool_ops)
        spec = self.screening
        if spec == "full" or (spec == "auto" and not self.shots) or n == 0:
            self._screened = (list(range(n)), None)
            return self._screened
        grads = self._reference_gradients()
        ranked = [int(k) for k in np.argsort(-np.abs(grads), kind="stable")]
        if isinstance(spec, dict):
            count = min(spec["operators"], n)
            self._screened = (sorted(ranked[:count]), (
                f"the {count} of {n} operators with the largest gradients at "
                f"the reference determinant (screening operators = {count})"))
            return self._screened
        pool = self.pool_gradients
        provider = self.circuit_provider()
        bases, strings = self._gradient_job_limits()

        def fits(count):
            plan = pool.plan(provider, sorted(ranked[:count]), bases,
                             strings, cache=False)
            return plan.jobs <= 1

        low, high = 0, 1
        while fits(high):
            low = high
            if high == n:
                self._screened = (list(range(n)), None)
                return self._screened
            high = min(2 * high, n)
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (middle, high) if fits(middle) else (low, middle)
        if low == 0:
            from ..backends.measurement import MeasurementBudgetError
            single = pool.plan(provider, [ranked[0]], bases, strings,
                               cache=False)
            raise MeasurementBudgetError(
                f"screening='auto' fits the pool gradients of a shot-based "
                f"step into one job of the measurement budget ({bases} bases, "
                f"{strings} strings), and not even one operator fits: the "
                f"gradient of {self._pool_ops[ranked[0]].label!r} alone is "
                f"{single.strings:,} Pauli strings in {single.bases:,} bases "
                f"({single.jobs} jobs).  Pass screening='full' (several jobs "
                f"per step) or {{'operators': K}}, raise measurement_budget=, "
                f"or shrink the register.", single)
        self._screened = (sorted(ranked[:low]), (
            f"the {low} of {n} operators with the largest gradients at the "
            f"reference determinant, because screening='auto' fits a step "
            f"into one job of the measurement budget ({bases} bases, "
            f"{strings} strings) and the full pool's gradients do not fit; "
            f"pass screening='full' for all of them, in several jobs per "
            f"step"))
        warnings.warn(self._screened[1], RuntimeWarning, stacklevel=4)
        return self._screened

    def _gradient_plan(self):
        """The measurement plan of one screening step (state-independent)."""
        chosen, _why = self._screened_operators()
        bases, strings = self._gradient_job_limits()
        return self.pool_gradients.plan(self.circuit_provider(), chosen,
                                        bases, strings)

    def _screen(self, ansatz, params):
        """``(gradients, gradient_of)`` at the current state.

        ``gradients`` covers the whole pool (an operator outside the screened
        set reads 0, so it is never selected); ``gradient_of(op)`` is the
        gradient of one operator, pooled or not, at the same state -- what a
        coupled-exchange growth asks for its members.

        A local run contracts the state vector.  A shot-based run never has
        one: at the reference determinant (an empty ansatz) every gradient is
        exact and classical; after that the screened gradients are measured
        on the device, after the step's plan has passed the budgets
        (:meth:`~mandacaru.algorithms.base.VariationalDriver._preflight_gradient_step`).
        """
        chosen, _why = self._screened_operators()
        n = len(self._pool_ops)
        if not self.shots:
            psi = (ansatz.state(params) if ansatz.num_parameters
                   else ansatz.reference_state())
            grads = self._gradients(psi)
            if len(chosen) < n:
                kept = np.zeros(n)
                kept[chosen] = grads[chosen]
                grads = kept
            return grads, (lambda op, _psi=psi:
                           self._operator_gradient(op, _psi))
        pool = self.pool_gradients
        grads = np.zeros(n)
        if not ansatz.num_parameters:
            occupied = ansatz.reference_qubits()
            grads[chosen] = pool.reference_gradients(occupied)[chosen]
            return grads, (lambda op, _occ=occupied: float(
                pool.with_generators([op.generator])
                .reference_gradients(_occ)[0]))
        provider = self.circuit_provider()
        bases, strings = self._gradient_job_limits()
        self._preflight_gradient_step(self._gradient_plan())
        problem = self.circuit_problem(ansatz, params)
        values, errors = pool.measure(provider, problem, chosen, bases,
                                      strings)
        grads[chosen] = values
        self.gradient_errors = np.zeros(n)
        self.gradient_errors[chosen] = errors

        def gradient_of(op, _problem=problem):
            single = pool.with_generators([op.generator])
            self._preflight_gradient_step(
                single.plan(provider, [0], bases, strings))
            return float(single.measure(provider, _problem, [0], bases,
                                        strings)[0][0])

        return grads, gradient_of

    def _stop_growth_on_budget(self, exc) -> None:
        """End the growth where a refused gradient measurement left it (the
        optimizer's own budget stop does the same, :meth:`_budget_stop`)."""
        self.run_budget_exhausted = str(exc)
        warnings.warn(f"the run budget stopped the growth before a pool "
                      f"gradient measurement: {exc}", RuntimeWarning,
                      stacklevel=3)

    # -- transfer between geometries -------------------------------------- #

    def inherit_ansatz(self, previous) -> None:
        """Take over the ansatz ``previous`` -- the solver of the preceding
        geometry -- grew, when ``transfer=True``.

        The ASE calculator calls this on each new geometry's solver before
        running it.  Only an ADAPT-VQE run that finished passes its ansatz
        on, with the molecular orbitals it was expressed in.
        """
        from .orbital_tracking import OrbitalSnapshot

        if not self.transfer or type(previous) is not type(self):
            return
        result = getattr(previous, "result", None)
        ansatz = getattr(previous, "ansatz", None)
        if not isinstance(result, ADAPTVQEResult) or ansatz is None:
            return
        context = getattr(previous, "_gradient_context", None) or {}
        pool = getattr(previous, "pool", None)
        self._inherited = {
            "operators": [op.label for op in ansatz.operators],
            "angles": np.asarray(result.optimal_parameters, dtype=float),
            "pool_labels": [op.label for op in previous._pool_ops],
            "symbols": getattr(previous, "_basis_symbols", None),
            "orbitals": OrbitalSnapshot.from_integrals(
                context.get("integrals"),
                getattr(pool, "n_spatial_orbitals", 0)),
        }

    def _transferred_start(self, ansatz: AdaptAnsatz, ref_energy: float,
                           max_iterations: int):
        """The previous geometry's ansatz, relaxed here, as a restored
        start -- or ``None`` and the reason it is not used.

        The operators are followed through the orbital matching
        (:func:`~mandacaru.algorithms.orbital_tracking.carry_ansatz`), every
        angle is re-optimized under this geometry's Hamiltonian, and the
        result is kept only if it lies below the reference energy.  The
        growth loop then screens the pool gradients at that optimum: below
        the convergence threshold the carried ansatz is this geometry's
        answer as it stands, otherwise operators are appended to it.
        """
        from .orbital_tracking import carry_ansatz

        inherited = self._inherited
        labels = [op.label for op in self._pool_ops]
        if inherited["symbols"] != getattr(self, "_basis_symbols", None) \
                or inherited["pool_labels"] != labels:
            return None, ("empty ansatz (transfer refused: another system or "
                          "pool than the previous geometry's)")
        context = getattr(self, "_gradient_context", None) or {}
        operators, angles, how = carry_ansatz(
            inherited["operators"], inherited["angles"],
            inherited["orbitals"], context.get("integrals"),
            self.pool.num_particles, labels, self.transfer_threshold)
        if operators is None:
            return None, f"empty ansatz (transfer refused: {how})"
        if not operators:
            return None, "empty ansatz (the previous geometry's was empty)"
        if len(operators) > max_iterations:
            return None, (f"empty ansatz (transfer refused: "
                          f"{len(operators)} operators > max_iterations "
                          f"{max_iterations})")
        index = self._pool_index
        ops = [self._pool_ops[index[label]] for label in operators]
        trial = self._new_ansatz()
        for op in ops:
            trial.append(op)
        relaxed = self._optimize_all(lambda t: self.ansatz_energy(trial, t),
                                     np.asarray(angles, dtype=float),
                                     gradient=self._energy_gradient(trial))
        energy = float(relaxed.fun)
        if not energy < ref_energy:
            return None, (f"empty ansatz (the relaxed transferred ansatz, "
                          f"{self._to_energy_units(energy):.8f} "
                          f"{self._energy_unit_label()}, is not below the "
                          f"reference)")
        for op in ops:
            ansatz.append(op)
        failures = ([] if relaxed.success
                    else [(0, f"transferred ansatz: {relaxed.message}")])
        return {"parameters": np.asarray(relaxed.x, dtype=float),
                "selected": [op.label for op in ops], "iterations": [],
                "num_evaluations": int(relaxed.nfev),
                "optimizer_steps": int(relaxed.nit or 0),
                "optimizer_failures": failures, "energy": energy,
                "same_problem": False, "transferred": True}, (
            f"previous geometry's ansatz, {len(ops)} operator(s), angles "
            f"re-optimized here; {how}")

    # -- checkpoint / resume helpers ------------------------------------- #

    def _iteration_payload(self, it: AdaptIteration) -> dict:
        """One :class:`AdaptIteration` as checkpoint data (energy in Hartree)."""
        return {"operator_label": it.operator_label,
                "operator_kind": it.operator_kind,
                "max_gradient": float(it.max_gradient),
                "energy_ha": float(to_hartree(it.energy,
                                              self._energy_unit_label())),
                "cnot_count": it.cnot_count, "depth": it.depth,
                "num_parameters": int(it.num_parameters),
                "optimizer_steps": it.optimizer_steps,
                "num_evaluations": it.num_evaluations}

    def _match_pool_operator(self, generator, label: str, kind: str
                             ) -> PoolOperator:
        """The pool operator with this generator, or a stand-alone one.

        A resumed generator is normally one of the pool's own (same encoding,
        same excitation), and using the pool's object keeps the labels and the
        screening consistent.  A generator the pool does not contain -- a
        checkpoint from another pool -- is still valid: it is wrapped as its own
        operator and applied exactly as stored.
        """
        wanted = generator.simplify().terms
        for op in self._pool_ops:
            mine = op.generator.simplify().terms
            if mine.keys() == wanted.keys() and all(
                    abs(mine[k] - wanted[k]) < 1e-10 for k in wanted):
                return op
        return PoolOperator(label=label, generator=generator,
                            support=_support_of(generator), kind=kind or "resumed")

    def _restore_growth(self, ansatz: AdaptAnsatz, record) -> dict:
        """Append a checkpoint's generators to ``ansatz``; return its history.

        The returned ``energy`` is always evaluated here, and ``same_problem``
        says whether the file was written for this run's Hamiltonian.
        """
        from ..core.checkpoint import fingerprint as _hamiltonian_fingerprint

        selected: list[str] = []
        for generator, label, kind in zip(record.generators, record.labels,
                                          record.kinds):
            op = self._match_pool_operator(generator, label, kind)
            ansatz.append(op)
            selected.append(op.label)
        status = record.status or {}
        unit = self._energy_unit_label()
        iterations = [AdaptIteration(
            operator_label=str(it["operator_label"]),
            operator_kind=str(it.get("operator_kind", "")),
            max_gradient=float(it["max_gradient"]),
            energy=self._to_energy_units(float(it["energy_ha"])),
            cnot_count=it.get("cnot_count"), depth=it.get("depth"),
            num_parameters=int(it["num_parameters"]),
            optimizer_steps=it.get("optimizer_steps"),
            num_evaluations=it.get("num_evaluations"))
            for it in status.get("iterations", [])]
        params = np.asarray(record.parameters, dtype=float)

        # The energy is *recomputed* from the restored state under **this**
        # run's Hamiltonian, never read back from the file.  A checkpoint can be
        # resumed into a different problem -- a changed geometry, charge or even
        # a constant shift keeps the register width, the mapping and the
        # reference determinant that `_load_resume` checks -- and with a growth
        # budget of zero the stored number would otherwise be reported as this
        # run's result. (Verified: resuming `ZIII` into `ZIII + 2 IIII` reported
        # -1 Ha for a state whose energy is +1 Ha, and called it converged.)
        energy = self.ansatz_energy(ansatz, params)
        stored = None if record.energy is None else float(record.energy)
        same_problem = (record.hamiltonian_fingerprint()
                        == _hamiltonian_fingerprint(self.hamiltonian))
        drift = None if stored is None else abs(energy - stored)
        if not same_problem:
            # A warm start, not a continuation: the history describes a
            # different operator, so its energies are not comparable to this
            # run's and are dropped rather than plotted alongside them.
            iterations = []
            warnings.warn(
                f"resuming from {self.resume_path!r} into a different "
                f"Hamiltonian: the state is reused as a warm start, its energy "
                f"recomputed here"
                + ("" if drift is None else
                   f" ({self._to_energy_units(energy):+.8f} against "
                   f"{self._to_energy_units(stored):+.8f} {unit} in the file)")
                + ", and the stored iteration history discarded.",
                RuntimeWarning, stacklevel=3)
        elif drift is not None and drift > RESUME_ENERGY_TOLERANCE:
            warnings.warn(
                f"the checkpoint {self.resume_path!r} stores "
                f"{self._to_energy_units(stored):+.10f} {unit} but its state "
                f"evaluates to {self._to_energy_units(energy):+.10f} {unit} "
                f"here (difference {drift:.2e} Ha) for the same Hamiltonian; "
                f"the recomputed value is used.",
                RuntimeWarning, stacklevel=3)
        return {"parameters": params, "selected": selected,
                "iterations": iterations,
                "num_evaluations": int(status.get("num_evaluations", 0)),
                "optimizer_steps": int(status.get("optimizer_steps", 0)),
                "optimizer_failures": [(int(s), str(m)) for s, m
                                       in status.get("optimizer_failures", [])],
                "energy": energy, "unit": unit,
                "same_problem": same_problem, "stored_energy": stored}

    # -- the run log ------------------------------------------------------ #

    def _make_logger(self, targets, geometry, cell, ref_energy,
                     max_iterations, convergence, restored=None):
        """Create a :class:`Logger` and write the header blocks.

        ``targets`` is what
        :meth:`~mandacaru.algorithms.base.VariationalDriver.log_targets`
        returned -- the ``txt=`` file, standard output, or both.  Returns
        ``None`` when it is empty, which is the only way a run reports nothing.
        Resolves the geometry/cell from an ASE ``Atoms`` object or a
        ``(symbols, positions)`` pair, and converts geometry/energy into the
        configured output units (**eV / Angstrom** by default).
        """
        if not targets:
            return None
        logger = self._open_header_logger(targets, geometry, cell)
        # The screening gradient is an expectation value of the Hamiltonian's
        # commutator, so it is in Hartree whatever unit the energy columns use
        # -- worth stating, since the gradient criterion is compared against it.
        gradient_method, gradient_formula = self._gradient_description()
        logger.write_optimizer_setup(
            optimizer_method=self.optimizer.method,
            reference_energy=self._to_energy_units(ref_energy),
            energy_unit=self._energy_unit_label(),
            gradient_method=gradient_method,
            gradient_formula=gradient_formula,
            convergence=convergence, gradient_units="Hartree",
            energy_scale=self._to_energy_units(1.0),
            max_iterations=max_iterations,
            # The first row's dE: from the reference state, or from the energy
            # a resumed run's restored ansatz already had.
            start_energy=(None if restored is None else
                          self._to_energy_units(restored["energy"])),
            # The pool's type and size, before the iteration table; the problem
            # itself is in the [ELECTRONS] block above.
            extra=self._setup_fields(),
            lineage=self._lineage_fields(restored))
        if self.shots:
            logger.write_block("GRADIENT MEASUREMENT",
                               self._gradient_measurement_fields())
        return logger

    def _gradient_measurement_fields(self) -> dict:
        """``[GRADIENT MEASUREMENT]``: what one screening step measures on
        the device and what it costs -- the one block that says it."""
        plan = self._gradient_plan()
        _chosen, why = self._screened_operators()
        fields = plan.fields(energy_bases=self.pool_gradients.energy_bases())
        if why is not None:
            fields["screened_subset"] = why
        return fields

    def _setup_fields(self) -> dict:
        """Driver-specific ``[OPTIMIZATION SETUP]`` lines: the operator pool.

        What the run screens *over*, next to the gradient group that says how it
        screens.  A subclass that adds a setting of its own extends this.
        """
        shots = (f"{self.shots}" if self.shots
                 else "0 (exact expectation values)")
        return {"pool": getattr(self.pool, "name", "?"),
                "pool_class": self.pool.__class__.__name__,
                "pool_size": len(self._pool_ops),
                # How each growth step is optimized and where it executes --
                # until now only the standard-output header said, and that
                # header is off whenever this file is written.
                "growth": ("tetris (all disjoint-support operators per step)"
                           if self.tetris else "single (largest gradient)"),
                "pruning": ("on (drop one irrelevant operator per step)"
                            if self.prune else "off"),
                "screening": str(self.screening),
                "reoptimize_all_parameters": str(self.quenching),
                "parameter_gradient": self._parameter_gradient_description(),
                "state_vector_backend": self._backend_description(),
                "device": str(self.device),
                "backend_provider": str(self.backend_provider),
                "circuit_execution": str(self.execute_circuits),
                "shots": shots,
                # With profiling off the 1q / cnot / depth columns are dashes
                # and the summary has no gate counts: the log has to say why,
                # or an unprofiled run reads like a broken one.
                "circuit_profiling": (
                    "True" if self.profile else
                    "False (no 1q / cnot / depth: pass profile=True)")}

    def _lineage_fields(self, restored=None) -> dict:
        """Closing ``[OPTIMIZATION SETUP]`` lines: what this run was resumed from.

        Empty for a fresh run, which starts from the reference state the
        ``[ELECTRONS]`` block names, with an empty ansatz.  A resumed run starts
        from the grown ansatz the checkpoint carries, and these lines are the
        one record of it: without them the iteration table starts at 1 with no
        sign that anything preceded it.
        """
        if restored is None:
            return ({} if self.start in (None, "empty ansatz")
                    else {"start": self.start})
        if restored.get("transferred"):
            return {
                "start": self.start,
                "transferred_operators": len(restored["selected"]),
                "transferred_energy_" + self._energy_unit_label():
                    f"{self._to_energy_units(restored['energy']):.10f}",
            }
        return {
            "resumed_from": str(self.resume_path),
            "restored_operators": len(restored["selected"]),
            "restored_parameters": len(restored["parameters"]),
            "restored_energy_" + self._energy_unit_label():
                f"{self._to_energy_units(restored['energy']):.10f}",
            "resume_same_hamiltonian": bool(restored.get("same_problem", True)),
        }

    def _expressivity_wanted(self, log_expressivity) -> bool:
        """Validate ``log_expressivity`` and return it.

        It is a plain boolean: ``True`` computes the expressivity every
        iteration, ``False`` (the default) never does and the ``expr`` column
        does not appear at all.  There is no automatic middle setting -- a
        column that is sometimes filled, and a cost that depends on the
        backend, are worse than a decision the caller makes once.
        """
        if log_expressivity is True and self.shots:
            raise ValueError(
                "log_expressivity=True samples state fidelities of the "
                "ansatz, which a shot-based run (shots > 0) does not build")
        if log_expressivity is True or log_expressivity is False:
            return bool(log_expressivity)
        raise ValueError(
            "log_expressivity must be True or False, got "
            f"{log_expressivity!r}")

    def _expressivity(self, ansatz) -> float:
        """Expressivity score ``E`` of the current ansatz (KL from Haar).

        Uses the number-conserving sector dimension as the fixed Haar reference,
        so scores are comparable across iterations (see
        :mod:`mandacaru.algorithms.expressivity`).
        """
        from .expressivity import (active_space_dimension,
                                   calculate_kl_divergence,
                                   sample_pqc_fidelities)
        self._cite("Sim2019")
        # The sector lives on the spin-orbitals: a tapered register has two more.
        n_modes = self.n_qubits + (2 if self.mapping == "parity_reduced" else 0)
        dim = active_space_dimension(n_modes, self.num_particles)
        fidelities = sample_pqc_fidelities(ansatz,
                                           num_samples=EXPRESSIVITY_SAMPLES,
                                           rng=self._expr_rng)
        return calculate_kl_divergence(fidelities, self.n_qubits, num_bins=75,
                                       dim=dim)

    # -- main loop -------------------------------------------------------- #

    def run(self, initial_parameters=None, callback=None,
            geometry=None, cell=None,
            log_expressivity: bool = False) -> ADAPTVQEResult:
        """Grow and optimize the ansatz until convergence.

        Everything that also lives on the constructor -- the stopping controls
        (``max_iterations`` / ``convergence``), the ``output`` log path and
        the ``verbose`` flag -- is taken from the instance, so a configured
        :class:`ADAPTVQE` is driven with a bare ``.run()``.  ``run`` only accepts
        arguments that the constructor does not already carry.

        Parameters
        ----------
        initial_parameters : array_like, optional
            Warm-start parameters for an already-grown ansatz (rarely needed).
        callback : callable, optional
            Invoked once per accepted operator with a dict
            ``{"iteration", "num_operators", "ansatz", "parameters", "energy",
            "max_gradient", "operator_label", "metrics"}`` after the inner
            re-optimization.  Used e.g. by
            :class:`~mandacaru.algorithms.expressivity.ADAPTExpressivityTracker` to
            record how the ansatz's expressibility grows.  The ``ansatz`` passed is
            the live :class:`AdaptAnsatz` at its current size (do not mutate it).
        geometry : ase.Atoms or (symbols, positions), optional
            Initial geometry for the ``output.txt`` metadata block (written only
            when the instance's ``output`` path is set).  An ASE ``Atoms`` object
            supplies symbols, positions and (if periodic) the cell; a
            ``(symbols, positions)`` pair supplies just the geometry.
        cell : (3, 3) array_like, optional
            Explicit unit-cell tensor for the metadata block (overrides any cell
            carried by an ``Atoms`` ``geometry``).
        log_expressivity : bool
            Compute the ansatz's expressivity every iteration (default
            ``False``).  It is a diagnostic rather than a result and it is not
            cheap -- ``2 * EXPRESSIVITY_SAMPLES`` state preparations per
            iteration, each applying every operator in the ansatz, so the cost
            over a run grows quadratically -- so nothing pays for it unless it
            is asked for.  ``True`` adds the ``expr`` column to the run trace
            and to the ``output.txt`` table; ``False`` leaves the column out
            altogether rather than filling it with dashes.
        """
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured:
            raise RuntimeError(
                "ADAPTVQE has no Hamiltonian; construct it with one, or use it "
                "as an ASE calculator with a `hamiltonian_builder`")
        self._check_kpts()

        # Stopping / logging controls come straight from the constructor.
        max_iterations = self.max_iterations
        convergence = self.convergence
        # One report, written to every destination `log_targets` names: the
        # `txt=` file, standard output, or both.  There is no second renderer.
        targets = self.log_targets
        # Resolved once, before the heading: whether this run computes the
        # expressivity at all decides whether the column exists.
        self._expressivity_on = self._expressivity_wanted(log_expressivity)

        # In calculator mode the wall clock is seeded in calculate() so it spans
        # the integration too; in direct mode it starts here.
        timings, run_t0 = self._make_timings()

        ansatz = self._new_ansatz()
        self.ansatz = ansatz        # the grown ansatz, for measured energies
        params = (np.asarray(initial_parameters, dtype=float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        # The determinant's own energy on a shot-based run, which builds no
        # state vector (base: reference_energy_of).
        ref_energy = self.reference_energy_of(ansatz)
        if self.shots:
            # The gradient step's cost is state-independent: planned once,
            # and refused now, before anything is queued, if one step cannot
            # fit the budgets.
            with timings.time("gradient planning"):
                self._preflight_gradient_step(self._gradient_plan())

        # Resume: rebuild the grown ansatz and its history from a checkpoint,
        # then keep growing.  `max_iterations` counts operators in total.
        resumed = self._load_resume(ansatz)
        restored = (self._restore_growth(ansatz, resumed)
                    if resumed is not None else None)
        if restored is not None:
            if initial_parameters is not None:
                raise ValueError("pass either initial_parameters or resume=, "
                                 "not both")
            params = restored["parameters"]
            self.start = f"resumed from {self.resume_path}"
            if not restored["same_problem"] and params.size:
                # A checkpoint written for another Hamiltonian (a previous
                # geometry) holds angles that are optimal there, not here: the
                # pool gradients at those angles would mostly measure that and
                # re-select operators the ansatz already has.  Relax them
                # first, then screen.
                relaxed = self._optimize_all(
                    lambda t: self.ansatz_energy(ansatz, t), params,
                    gradient=self._energy_gradient(ansatz))
                params = np.asarray(relaxed.x, dtype=float)
                restored["parameters"] = params
                restored["energy"] = float(relaxed.fun)
                restored["num_evaluations"] += int(relaxed.nfev)
                restored["optimizer_steps"] += int(relaxed.nit or 0)
                self.start += " (angles re-optimized for this Hamiltonian)"
        elif self._inherited is not None:
            if initial_parameters is not None:
                raise ValueError("pass either initial_parameters or "
                                 "transfer=True, not both")
            restored, self.start = self._transferred_start(
                ansatz, ref_energy, max_iterations)
            if restored is not None:
                params = restored["parameters"]
        else:
            self.start = "empty ansatz"

        # Banner to standard output *before* any data is written to the log.
        self._show_banner()

        logger = self._make_logger(targets, geometry, cell, ref_energy,
                                   max_iterations, convergence,
                                   restored=restored)
        e_unit = self._energy_unit_label()

        iterations: list[AdaptIteration] = []
        selected: list[str] = []
        recorder = self._open_recorder(ansatz, geometry)
        total_evals = 0
        total_steps = 0
        converged = False
        optimizer_failures: list[tuple[int, str]] = []
        #: (iteration, label) of every operator pruning took back out.
        pruned: list[tuple[int, str]] = []
        max_grad = np.inf
        energy = ref_energy
        # The energy change of the last growth step (the log's dE), for the
        # energy criterion; None until a step has been taken.
        delta_energy: float | None = None
        metrics: CircuitMetrics | None = None
        final_expr: float | None = None
        if restored is not None:
            iterations = restored["iterations"]
            selected = restored["selected"]
            total_evals = restored["num_evaluations"]
            total_steps = restored["optimizer_steps"]
            optimizer_failures = restored["optimizer_failures"]
            energy = restored["energy"]

        def checkpoint(complete: bool, converged: bool):
            """Write the state as it stands (every run writes at the end)."""
            self._write_checkpoint(self._checkpoint_record(
                ansatz, params, energy, {
                    "complete": bool(complete), "converged": bool(converged),
                    "iteration": len(iterations),
                    "max_gradient": float(max_grad) if np.isfinite(max_grad)
                    else None,
                    "num_evaluations": int(total_evals),
                    "optimizer_steps": int(total_steps),
                    "reference_energy": float(ref_energy),
                    "optimizer_failures": [[int(s), str(m)]
                                           for s, m in optimizer_failures],
                    "iterations": [self._iteration_payload(it)
                                   for it in iterations]}))

        # Growth steps are bounded as well as operators: pruning can hand an
        # operator back for the next step to re-select, which leaves the
        # operator count unchanged and would otherwise loop forever.
        step_budget = (PRUNE_STEP_BUDGET * max_iterations if self.prune
                       else max_iterations)
        carried = restored is not None and not restored["same_problem"]
        try:
            while len(selected) < max_iterations and len(iterations) < step_budget:
                if getattr(self, "run_budget_exhausted", None):
                    # The processor budget ended the last growth step at its
                    # best point; growing further would submit nothing.
                    break
                try:
                    with timings.time("gradient screening"):
                        grads, gradient_of = self._screen(ansatz, params)
                except RunBudgetExceeded as exc:
                    # Measured on the device and refused before its first
                    # job: the ansatz grown so far is the result.
                    self._stop_growth_on_budget(exc)
                    break
                max_grad = _max_abs(grads)
                if convergence.reached(max_grad, delta_energy) or (
                        carried and delta_energy is None
                        and convergence.gradient is not None
                        and max_grad < convergence.gradient):
                    # An ansatz carried from another Hamiltonian (the previous
                    # geometry) has no growth step whose energy change the
                    # energy criterion could read: at its relaxed optimum the
                    # gradient alone decides whether it stands as it is.
                    converged = True
                    break
                # `op` is what the screening *selected* and what the iteration
                # row names; `n_new` counts everything actually appended, which
                # is one except for an MVP-CEO or a TETRIS layer.
                if self.tetris:
                    chosen = self._select_disjoint(grads)
                else:
                    chosen = [self._select_operator(grads, len(selected))]
                idx = chosen[0]
                op = self._pool_ops[idx]
                index = self._pool_index
                source = [index.get(label) for label in selected]
                n_new = 0
                try:
                    for pick in chosen:
                        n_new += self._grow(ansatz, selected,
                                            self._pool_ops[pick], gradient_of)
                except RunBudgetExceeded as exc:
                    # A coupled-exchange member's gradient was refused before
                    # anything of this step was appended.
                    self._stop_growth_on_budget(exc)
                    break

                # Warm start: reuse previous optimum, new parameters set to 0.
                # `quenching` decides whether the previous angles are re-optimized
                # alongside the new ones or frozen at their prior values.
                with timings.time("parameter optimization"):
                    result = self._optimize_grown(
                        lambda t: self.ansatz_energy(ansatz, t), params,
                        n_new=n_new, gradient=self._energy_gradient(ansatz))
                params = np.asarray(result.x, dtype=float)
                previous_energy = energy
                energy = float(result.fun)
                total_evals += result.nfev
                total_steps += result.nit or 0

                if self.prune:
                    params, dropped = self._prune_ansatz(ansatz, params)
                    if dropped is not None:
                        # The ansatz is one operator shorter; the reported
                        # energy is re-evaluated because it is the energy of a
                        # different state, however slightly.
                        if dropped in selected:
                            selected.remove(dropped)
                        pruned.append((len(iterations) + 1, dropped))
                        energy = float(self.ansatz_energy(ansatz, params))
                delta_energy = energy - previous_energy
                # A row is one appended pool operator after pool operators
                # only; a coupled-exchange step that grows into several
                # excitations is not recorded.
                if recorder is not None and n_new == 1 \
                        and None not in source:
                    recorder.write({
                        "step": len(iterations) + 1, "move": "insert",
                        "position": len(source), "operator": int(idx),
                        "source": source, "proposed": source + [int(idx)],
                        "source_energy": float(previous_energy),
                        "reference_energy": float(ref_energy),
                        "delta_energy": float(delta_energy),
                        "source_gradients": [float(g) for g in grads],
                        "accepted": True, "nfev": int(result.nfev),
                        "optimizer_success": bool(result.success),
                        "screened": False})
                if not result.success:
                    # The inner optimizer did not certify convergence; the
                    # growth continues from its best point, but the run says so.
                    optimizer_failures.append(
                        (len(iterations) + 1, str(result.message)))

                with timings.time("circuit profiling"):
                    metrics = self._profile(ansatz)

                # Expressivity is a column of the trace as well as a logged
                # quantity, so it is computed whenever either will show it --
                # and, by default, only where it is cheap (_expressivity_wanted).
                expr = None
                if self._expressivity_on:
                    with timings.time("expressivity"):
                        expr = self._expressivity(ansatz)
                # Always the *last* iteration's value, so the summary's
                # `final_expressivity_E` is the final ansatz's or is omitted.
                # Keeping the last computed one would report the value of a
                # twelve-operator ansatz as the final one of an eighty-operator
                # run, which is worse than reporting nothing.
                final_expr = expr

                iterations.append(AdaptIteration(
                    operator_label=op.label, operator_kind=op.kind,
                    max_gradient=max_grad,
                    energy=self._to_energy_units(energy),
                    cnot_count=metrics.cnot_count, depth=metrics.depth,
                    num_parameters=ansatz.num_parameters,
                    optimizer_steps=result.nit,
                    num_evaluations=int(result.nfev)))

                if logger is not None:
                    logger.write_iteration(
                        iteration=len(iterations), pool_operators=self._pool_ops,
                        gradients=grads, selected_index=idx, expressivity=expr,
                        energy=self._to_energy_units(energy), energy_unit=e_unit,
                        num_parameters=ansatz.num_parameters, metrics=metrics,
                        optimizer_steps=result.nit)

                if self.checkpoint_path is not None \
                        and len(iterations) % self.checkpoint_every == 0:
                    checkpoint(complete=False, converged=False)

                if callback is not None:
                    callback({
                        "iteration": len(iterations),
                        "num_operators": ansatz.num_parameters,
                        "ansatz": ansatz,
                        "parameters": params,
                        "energy": self._to_energy_units(energy),
                        "energy_unit": e_unit,
                        "max_gradient": max_grad,
                        "operator_label": op.label,
                        "metrics": metrics,
                    })
            if not converged and not getattr(self, "run_budget_exhausted",
                                             None):
                # The loop stopped on max_iterations (or ran none at all).
                # Screen once more so the reported gradient -- and the
                # convergence flag -- describe the *final* state rather than
                # the one before the last operator was appended and optimized.
                try:
                    with timings.time("gradient screening"):
                        max_grad = _max_abs(self._screen(ansatz, params)[0])
                    converged = convergence.reached(max_grad, delta_energy)
                except RunBudgetExceeded as exc:
                    # The last gradient stays the one measured before the
                    # final growth step; the run is not called converged.
                    self._stop_growth_on_budget(exc)

            # The final state, whether or not periodic checkpoints were asked
            # for -- this is what another algorithm (QPE) starts from.
            checkpoint(complete=True, converged=converged)

            if logger is not None:
                logger.write_summary(
                    converged=converged,
                    optimal_energy=self._to_energy_units(energy),
                    reference_energy=self._to_energy_units(ref_energy),
                    correlation_energy=self._to_energy_units(energy - ref_energy),
                    energy_unit=e_unit, num_operators=len(selected),
                    num_parameters=int(params.size),
                    final_max_gradient=max_grad, expressivity=final_expr,
                    num_evaluations=total_evals, metrics=metrics,
                    optimizer=self.optimizer.method,
                    optimizer_steps=total_steps)
        finally:
            if logger is not None:
                logger.close()
            if recorder is not None:
                recorder.close()

        if optimizer_failures:
            steps = ", ".join(str(step) for step, _msg in optimizer_failures)
            warnings.warn(
                f"the inner optimizer reported no convergence in "
                f"{len(optimizer_failures)} of {len(iterations)} growth steps "
                f"(steps {steps}); last message: "
                f"{optimizer_failures[-1][1]!r}", RuntimeWarning, stacklevel=2)

        # Fold the (calculator-mode) integration stage in, then set the wall time.
        self._finalize_timings(timings, run_t0)
        # The performance block closes the step, after the summary.  In
        # calculator mode `Mandacaru` writes it instead, so the nuclear gradient's
        # time -- usually the largest stage of a relaxation step -- is in it.
        self.write_performance(timings)

        # The single Hartree -> output-unit boundary of the run.
        result = ADAPTVQEResult(
            optimal_energy=self._to_energy_units(energy),
            optimal_parameters=params,
            reference_energy=self._to_energy_units(ref_energy),
            converged=converged,
            final_max_gradient=max_grad,
            operators=selected,
            iterations=iterations,
            num_evaluations=total_evals,
            optimizer_steps=total_steps,
            optimizer_failures=optimizer_failures,
            pruned_operators=pruned,
            start=self.start,
            metrics=metrics,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile,
            energy_unit=e_unit)

        return result

    @property
    def _pool_index(self) -> dict:
        """Pool index of each operator label."""
        return {op.label: i for i, op in enumerate(self._pool_ops)}

    def _open_recorder(self, ansatz, geometry) -> EditRecorder | None:
        """The edit store this run's growth steps go into, if ``record``."""
        if self.record is None:
            return None
        import uuid
        from ..version import __version__
        from .mcas_vqe import _formula
        problem = ProblemRecord.from_problem(
            self.hamiltonian, ansatz.reference_qubits(), self._pool_ops)
        return EditRecorder(self.record, problem, run={
            "run": uuid.uuid4().hex[:12], "method": "ADAPT-VQE",
            "mandacaru": __version__, "formula": _formula(geometry),
            "pool": getattr(self.pool, "name", "?"),
            "optimizer": self.optimizer.method,
            "trajectory": None, "previous_problem": None})

    # -- excited states / energy levels (DeflationMixin hook) ------------- #

    def _deflated_ground(self, states, beta, *, state_index: int = 0,
                         max_iterations: int | None = None,
                         convergence: Convergence | Mapping[str, float | None]
                         | None = None):
        r"""Grow a fresh deflated ADAPT ansatz orthogonal to ``states``.

        Both the pool-screening gradient and the inner re-optimization carry the
        overlap penalty, so the adaptive ansatz grows toward the next excited
        state; the reported energy is the bare expectation value.  Called per level
        by :meth:`~mandacaru.algorithms.deflation.DeflationMixin.energy_levels`
        (which also accepts ``max_iterations`` / ``convergence``).
        """
        max_it = (self.max_iterations if max_iterations is None
                  else int(max_iterations))
        criteria = (self.convergence if convergence is None
                    else Convergence.resolve(convergence))
        energy, psi, n_ops, nev = self._grow_deflated(states, beta, max_it,
                                                      criteria)
        return energy, psi, nev, n_ops

    def _deflated_gradients(self, psi: np.ndarray, states, beta: float
                            ) -> np.ndarray:
        r"""Pool gradients of the deflated cost at the current state ``psi``.

        The Hamiltonian part is the exact analytic gradient
        ``2 Re<H psi | A_i psi>``; each deflated state adds
        ``2 beta Re(<psi|psi_j> <psi_j| A_i psi>)``, the derivative of the overlap
        penalty when appending ``exp(theta A_i)`` at ``theta = 0``.
        """
        grads = self._analytic_gradients(psi)
        if not states:
            return grads
        overlaps = [np.vdot(psi, sj) for sj in states]      # <psi|psi_j>
        for i, a in enumerate(self._pool_matrices):
            a_psi = a @ psi
            extra = 0.0
            for ov, sj in zip(overlaps, states):
                extra += 2.0 * beta * float(np.real(ov * np.vdot(sj, a_psi)))
            grads[i] += extra
        return grads

    def _grow_deflated(self, states, beta, max_iterations,
                       convergence: Convergence):
        """Grow one deflated ADAPT ansatz; return ``(energy, psi, n_ops, nfev)``.

        With ``states`` empty this is an ordinary ADAPT ground-state growth.  A
        trimmed sibling of :meth:`run` (no logging / profiling / verbose trace)
        used to build each excited state.
        """
        ansatz = self._new_ansatz()
        params = np.zeros(0)
        total_evals = 0
        # The energy criterion reads the deflated cost -- what this growth
        # minimizes -- from the reference state on.
        reference = ansatz.reference_state()
        cost_now = self.energy(reference) + deflation_penalty(reference, states,
                                                              beta)
        delta_energy: float | None = None
        for _ in range(int(max_iterations)):
            psi = (ansatz.state(params) if ansatz.num_parameters
                   else ansatz.reference_state())
            grads = self._deflated_gradients(psi, states, beta)
            if convergence.reached(_max_abs(grads), delta_energy):
                break
            idx = self._select_operator(grads, ansatz.num_parameters)

            def deflated_gradient(op, _psi=psi, _states=states, _beta=beta):
                """The screening gradient of one operator, penalty included."""
                grad = self._operator_gradient(op, _psi)
                a_psi = self._operator_matrix(op) @ _psi
                for sj in _states:
                    grad += 2.0 * _beta * float(np.real(
                        np.vdot(_psi, sj) * np.vdot(sj, a_psi)))
                return grad

            # This loop tracks the ansatz, not the labels, so the growth's
            # label list is a throwaway.
            n_new = self._grow(ansatz, [], self._pool_ops[idx],
                               deflated_gradient)

            def cost(t, _states=states):
                phi = ansatz.state(t)
                return self.energy(phi) + deflation_penalty(phi, _states, beta)

            result = self._optimize_grown(cost, params, n_new=n_new)
            params = np.asarray(result.x, dtype=float)
            total_evals += result.nfev
            delta_energy = float(result.fun) - cost_now
            cost_now = float(result.fun)
        psi = (ansatz.state(params) if ansatz.num_parameters
               else ansatz.reference_state())
        return self.energy(psi), psi, ansatz.num_parameters, total_evals

    # -- gradient descriptions ------------------------------------------- #

    def _parameter_gradient_description(self) -> str:
        """How the classical optimizer gets the gradient of the energy in the
        ansatz parameters (the ``parameter_gradient`` setup line)."""
        return self._parameter_gradient_text(self._new_ansatz())

    def _gradient_description(self) -> tuple[str, str]:
        """``(how the screening gradient is evaluated, why)``.

        The *requested* estimator is not always the one used: on the sparse and
        sector backends the pool is kept as sparse matrices and the
        eigendecompositions the shift estimators need are never formed, so
        those paths screen analytically whatever was asked for.  The header
        reports what actually runs, and says when it differs from the request.
        """
        if self.shots:
            note = GRADIENT_FORMULAS["measured"]
            if self.gradient != "analytic":
                note += f" (overrides {self.gradient}: shots > 0)"
            return "measured", note
        overridden = (getattr(self, "_sparse", False)
                      or getattr(self, "_sector", None) is not None)
        effective = "analytic" if overridden else self.gradient
        note = GRADIENT_FORMULAS.get(effective, "")
        if overridden and self.gradient != "analytic":
            note += f" (overrides {self.gradient}: sparse pool)"
        return effective, note
