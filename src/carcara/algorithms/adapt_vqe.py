# -*- coding: utf-8 -*-
# file: algorithms/adapt_vqe.py

# This code is part of Carcará.
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

This is an exact state-vector implementation in the same spirit as
:class:`~carcara.algorithms.vqe.VQE`: the Hamiltonian and each generator are
materialized as dense matrices and the ansatz produces the exact :math:`2^N`
state vector, so gradients and energies are noiseless.

Each grown ansatz is also **profiled** for hardware cost: the parameterized
unitary is compiled to a native ``{CNOT, U}`` gate set with Qiskit and its CNOT
count and circuit depth are logged per iteration (see :class:`CircuitMetrics`).
The four operator pools (:mod:`carcara.circuits.pools`) can then be compared on
accuracy-per-CNOT.  Profiling is optional -- if Qiskit is unavailable the run
proceeds and metrics are reported as ``None``.

Beyond the ground state, :meth:`ADAPTVQE.energy_levels` returns the low-lying
**molecular energy levels** (ground + excited states) by variational quantum
deflation, growing a fresh deflated ansatz per level -- see
:mod:`carcara.algorithms.deflation`.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..circuits.adapt_ansatz import AdaptAnsatz
from ..circuits.pools import PoolBase, PoolOperator, _support_of, build_pool
from ..circuits.profiling import CircuitMetrics, profile_ansatz
from ..units import ANGSTROM_TO_BOHR, convert_energy, to_hartree
from .base import VariationalDriver
from .deflation import DeflationMixin, deflation_penalty


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


def _resolve_geometry(geometry):
    """Normalize ``geometry`` to ``(symbols, positions, cell)`` for logging.

    Accepts an ASE ``Atoms`` object (symbols/positions/cell read directly), a
    ``(symbols, positions)`` pair, or ``None``.  ``cell`` is ``None`` for a
    non-periodic input.
    """
    if geometry is None:
        return None, None, None
    # ASE Atoms: duck-typed to avoid a hard dependency here.
    if hasattr(geometry, "get_chemical_symbols") and \
            hasattr(geometry, "get_positions"):
        symbols = list(geometry.get_chemical_symbols())
        positions = np.asarray(geometry.get_positions(), dtype=float)
        cell = np.asarray(geometry.get_cell(), dtype=float)
        cell = cell if np.any(cell) else None
        return symbols, positions, cell
    # (symbols, positions) pair.
    symbols, positions = geometry
    return list(symbols), np.asarray(positions, dtype=float), None


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
    #: ``(growth step, message)`` of every inner optimization that did not
    #: report convergence -- empty when every step converged.
    optimizer_failures: list = field(default_factory=list)
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


#: Random parameter samples per expressivity estimate (two states each).
EXPRESSIVITY_SAMPLES = 400

#: Widest *dense* register whose expressivity ``log_expressivity="auto"`` will
#: compute.  Dense state preparation allocates the whole 2^n vector, so the
#: estimate costs 78 s per iteration at 12 qubits; the sparse and sector
#: backends work on a compressed state and are exempt.
EXPRESSIVITY_DENSE_MAX_QUBITS = 10


class ADAPTVQE(DeflationMixin, VariationalDriver):
    """Adaptive VQE on an exact state-vector backend; also an ASE calculator.

    Two usage modes:

    * **Direct** -- construct with a Hamiltonian and call :meth:`run`.
    * **ASE calculator** -- construct with a ``hamiltonian_builder`` (no
      Hamiltonian), attach to an ``Atoms`` object (``atoms.calc = ADAPTVQE(...)``)
      and let ``atoms.get_total_energy()`` build the Hamiltonian from the current
      geometry and drive :meth:`run`.  ASE energies are returned in **eV**.

    Parameters
    ----------
    hamiltonian : PauliSum or Fermion, optional
        Qubit Hamiltonian, or a fermionic Hamiltonian mapped with ``mapping``.
        Omit it in calculator mode and let ``basis`` (or ``hamiltonian_builder``)
        build it from the geometry instead.
    pool : PoolBase or str
        The operator pool, or a name for :func:`~carcara.circuits.pools.build_pool`
        -- one of ``"ceo"``, ``"fermionic"``, ``"qubit"``, ``"qeb"``.  When a name
        is given ``n_spatial_orbitals`` and ``num_particles`` are required (in
        direct mode; in calculator mode the builder supplies them).
    basis : str or dict
        Basis set used to build the molecular Hamiltonian from an ASE geometry in
        calculator mode.  Either a name -- ``"FAO"`` (default; Full Atomic
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
    optimizer : str or Optimizer
        Classical optimizer for the inner re-optimization.  Either a method name
        -- one of ``"SPSA"``, ``"COBYLA"`` (default), ``"Nelder-Mead"``,
        ``"SLSQP"``, ``"Adam"``, ``"L-BFGS-B"`` -- or a pre-built
        :class:`~carcara.optimizers.optim.Optimizer` instance.
    mapping : str
        Fermion-to-qubit mapping -- one of ``"jordan_wigner"`` (default),
        ``"parity"``, ``"bravyi_kitaev"`` -- used when ``hamiltonian`` is a
        ``Fermion`` and to build a named fermionic pool.
    gradient : str
        How the pool screening gradients are evaluated -- ``"analytic"``
        (default; the exact derivative ``g_i = 2 Re<H psi|A_i psi>``),
        ``"finite_difference"`` (a finite-difference estimate from shifted
        parameters) or ``"parameter-shift"`` (the quantum parameter-shift rule).
        All three converge to the same number; ``"analytic"`` is both the
        cheapest -- one matrix-vector product per pool operator, against
        ``2 x |pool|`` energy evaluations -- and the only one free of a step-size
        truncation error, so the shift-based estimators are opt-in, for studying
        the estimator itself.
    device : str
        Execution device -- ``"AER_simulator"`` (default; ideal simulator) or
        ``"ibm-quantum"`` (reserved for real hardware, not yet runnable).  See
        :mod:`carcara.backends.hardware`.
    max_iterations : int
        Maximum number of operators to append before stopping (default ``50``).
        Used as the default for :meth:`run` / the ASE-calculator evaluation.
    gradient_tolerance : float
        Convergence threshold on the largest pool gradient (default ``1e-3``).
        Used as the default for :meth:`run` / the ASE-calculator evaluation.
    output : str, optional
        Path of the structured ``output.txt`` runtime log (default ``None`` --
        no file).  Used as the default for :meth:`run` / the ASE-calculator
        evaluation.
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
        / ``False`` force it.  In sparse mode screening always uses the analytic
        gradient (the ``gradient`` argument's estimators need the dense
        eigendecompositions and are unavailable).
    sector : bool or str
        Simulate only the ``(n_alpha, n_beta)`` particle-number sector
        (default ``"auto"``).  The Hamiltonian and every pool generator are
        restricted to the sector's basis states
        (:class:`~carcara.core.sector.ParticleSector`) and state vectors carry
        ``C(M, n_alpha) C(M, n_beta)`` amplitudes instead of ``2^n`` -- 100
        instead of 1,048,576 for H2 or LiH in a DZP basis.  Exact, since the
        operators conserve particle number.  ``"auto"`` enables it for
        ``n_qubits >= 16`` (where a full-register sparse Hamiltonian no longer
        fits in memory) whenever states are prepared internally; ``True`` /
        ``False`` force it.  Incompatible with executing the ansatz as a
        circuit (``execute_circuits``), which prepares full-register states.
    atomic_units : bool
        Units used in the ``output.txt`` log.  ``False`` (default) logs energies
        in **eV** and lengths in **Angstrom**; ``True`` logs Hartree and Bohr.
        (ASE's ``get_total_energy`` always returns eV, per the ASE convention.)
    grid : Grid, optional
        Explicit real-space integration grid for the calculator-mode ``basis``
        builder.  When omitted the grid is generated automatically from the ASE
        ``atoms.cell`` at resolution ``h``; a unit cell is then required.
    h : float
        Target grid spacing in **Angstrom** (default ``0.20``) for the automatic
        cell-based grid used when ``grid`` is not given.  Finer ``h`` (e.g.
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
        :meth:`~carcara.core.hamiltonian.MolecularIntegrals.molecular_hamiltonian`.
    initial_state : str, optional
        The ansatz reference state; ``"hartree-fock"`` (default) is the
        Hartree-Fock determinant.  ``None`` is treated as ``"hartree-fock"``.
    charge : int
        Total charge, used to set the electron count in the ``basis`` builder.
    n_electrons : int, optional
        Explicit electron count for the ``basis`` builder (overrides ``charge``).
    frozen_core : bool, str or int
        Frozen-core approximation (default ``False``, no freezing).  ``True`` or
        ``"auto"`` freezes the chemical noble-gas core (``He`` core for Li--Ne,
        ``Ne`` core for Na--Ar, ...); an integer freezes that many lowest molecular
        orbitals.  The frozen (doubly occupied) core orbitals are removed from the
        active space and replaced by their mean-field contribution -- a constant
        core energy plus an effective one-body potential -- so the ansatz, pool and
        qubit count are built for the smaller active space.
    frozen_orbitals : sequence of int, optional
        Explicit list of (doubly occupied) spatial molecular-orbital indices to
        freeze.  Overrides ``frozen_core`` and names exactly which electrons are
        treated as frozen core; the remaining occupied orbitals plus the virtuals
        form the active space.
    hamiltonian_builder : callable, optional
        ``atoms -> (hamiltonian, num_particles, n_spatial_orbitals)``.  An
        explicit override for the built-in ``basis`` builder in calculator mode.
    save_hamiltonian : bool or str
        Write the qubit Hamiltonian (as Pauli strings) to disk once it has been
        built (default ``False``); ``True`` uses ``"hamiltonian"`` plus the
        extension of ``hamiltonian_format``, a string is used as the path.  See
        :mod:`carcara.core.serialization`.
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
        ``execute_circuits``.  See :mod:`carcara.backends.providers`.
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
    run_options : dict, optional
        Extra keyword arguments forwarded to :meth:`run` on each calculator
        evaluation (e.g. ``{"log_expressivity": False}``).  Only arguments that
        :meth:`run` accepts are valid here -- the stopping controls
        (``max_iterations`` / ``gradient_tolerance``), ``output`` and ``verbose``
        are constructor arguments, not ``run`` arguments.
    """

    _GRADIENTS = ("analytic", "finite_difference", "parameter-shift")
    _default_sparse = "auto"

    def __init__(self,
                 hamiltonian=None,
                 pool="fermionic",
                 basis="FAO",
                 num_particles=None,
                 n_spatial_orbitals=None,
                 optimizer: str | Optimizer = "COBYLA",
                 mapping: str = "jordan_wigner",
                 gradient: str = "analytic",
                 device: str = "AER_simulator",
                 max_iterations: int = 50,
                 gradient_tolerance: float = 1e-3,
                 output: str | None = None,
                 profile: bool = True,
                 verbose: bool = True,
                 sparse: bool | str = "auto",
                 sector: bool | str = "auto",
                 atomic_units: bool = False,
                 grid=None,
                 h: float = 0.20,
                 kpts=None,
                 spin: bool = False,
                 initial_state: str | None = "hartree-fock",
                 charge: int = 0,
                 n_electrons=None,
                 frozen_core=False,
                 frozen_orbitals=None,
                 hamiltonian_builder=None,
                 save_hamiltonian: bool | str = False,
                 load_hamiltonian: str | None = None,
                 hamiltonian_format: str = "parquet",
                 backend_provider: str | None = None,
                 execute_circuits: bool | None = None,
                 backend_options: dict | None = None, shots: int = 0,
                 quenching: bool = True, dry_run: bool = False,
                 kinetic: str | None = None,
                 two_qubit_reduction: bool = False,
                 run_options: dict | None = None, **calc_kwargs):
        super().__init__(optimizer=optimizer, mapping=mapping, basis=basis,
                         device=device, grid=grid, h=h, kpts=kpts, spin=spin,
                         initial_state=initial_state, charge=charge,
                         n_electrons=n_electrons, frozen_core=frozen_core,
                         frozen_orbitals=frozen_orbitals,
                         hamiltonian_builder=hamiltonian_builder,
                         save_hamiltonian=save_hamiltonian,
                         load_hamiltonian=load_hamiltonian,
                         hamiltonian_format=hamiltonian_format,
                         backend_provider=backend_provider,
                         execute_circuits=execute_circuits,
                         backend_options=backend_options, shots=shots,
                         quenching=quenching, dry_run=dry_run, kinetic=kinetic,
                         two_qubit_reduction=two_qubit_reduction,
                         run_options=run_options, verbose=verbose,
                         sparse=sparse, atomic_units=atomic_units,
                         **calc_kwargs)

        self.profile = profile
        if not (isinstance(sector, bool)
                or (isinstance(sector, str) and sector.strip().lower() == "auto")):
            raise ValueError(f"unknown sector spec {sector!r}; use True, False "
                             "or 'auto'")
        self.sector = sector
        # Validate the enumerated gradient option up front.
        if gradient not in self._GRADIENTS:
            raise ValueError(
                f"unknown gradient {gradient!r}; use one of {self._GRADIENTS}")
        self.gradient = gradient

        # Run defaults (also the defaults for the ASE-calculator evaluation).
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        self.output = output

        self._pool_spec = pool
        # Seeded RNG for reproducible expressivity logging (output.txt).
        self._expr_rng = np.random.default_rng(0)

        # A cached Hamiltonian is a complete problem specification (operator plus
        # num_particles / n_spatial_orbitals), so loading one puts the driver in
        # direct mode without a geometry -- no integrals, no mapping.
        if hamiltonian is None and self.load_hamiltonian is not None:
            hamiltonian, loaded_particles, loaded_orbitals = \
                self._load_hamiltonian_record()
            num_particles = num_particles or loaded_particles
            n_spatial_orbitals = n_spatial_orbitals or loaded_orbitals

        # Configure eagerly when a Hamiltonian is given (direct mode); otherwise
        # defer to the first calculator evaluation (the ASE hook in the base).
        # A dry run never configures: materializing the Hamiltonian allocates
        # the 2^n matrix whose feasibility is the very thing being asked about.
        if hamiltonian is not None and not self.dry_run:
            self._configure(hamiltonian, num_particles, n_spatial_orbitals)
            self._built_from_hamiltonian = True
        elif hamiltonian is not None:
            self._dry_run_problem = (hamiltonian, num_particles,
                                     n_spatial_orbitals)

    # -- setup helpers ---------------------------------------------------- #

    def _configure(self, hamiltonian, num_particles, n_spatial_orbitals):
        """Resolve the pool, materialize the Hamiltonian and the pool matrices."""
        pool = self._pool_spec
        if isinstance(pool, PoolBase):
            self.pool = pool
        else:
            if n_spatial_orbitals is None or num_particles is None:
                raise ValueError(
                    "building a pool by name requires n_spatial_orbitals and "
                    "num_particles")
            self.pool = build_pool(pool, n_spatial_orbitals, num_particles,
                                   mapping=self.mapping,
                                   two_qubit_reduction=self.two_qubit_reduction)
        self.num_particles = (tuple(num_particles) if num_particles is not None
                              else self.pool.num_particles)

        # Materialize the (dense or sparse) qubit Hamiltonian on the base; a dense
        # pool stores every operator's matrix *and* eigendecomposition (two
        # 2^n x 2^n arrays each), ~46 GB for a 12-qubit water active space, so
        # ``sparse="auto"`` keeps large active spaces as sparse matrices and
        # screens with the exact analytic gradient, densifying only selected
        # operators (in the growable ansatz).
        qubit_h = self._as_pauli_sum(hamiltonian, self.pool.n_qubits,
                                     self.num_particles)
        # The pool is built first: a sector may only be used when every
        # generator keeps the ansatz inside it (see _resolve_sector).
        self._pool_ops = self.pool.operators()
        self._materialize_hamiltonian(
            qubit_h, self.pool.n_qubits,
            sector=self._resolve_sector(self.pool.n_qubits, qubit_h,
                                        self._pool_ops))
        self._maybe_save_hamiltonian(self.num_particles,
                                     self.pool.n_spatial_orbitals)
        self._maybe_dump_hamiltonian(self.num_particles,
                                     self.pool.n_spatial_orbitals)
        self._maybe_dump_pool(self.pool, self._pool_ops)

        if self._sector is not None:
            self._pool_matrices = [self._sector.restrict(op.generator)
                                   for op in self._pool_ops]
            self._pool_eig = None
        elif self._sparse:
            self._pool_matrices = [op.generator.to_sparse_matrix()
                                   for op in self._pool_ops]
            self._pool_eig = None
        else:
            # Dense pool.  The per-operator eigendecomposition is built lazily
            # (``_pool_eigendecomposition``): only the shift-based estimators
            # need it, and it is |pool| dense diagonalizations of the full
            # register that the default analytic screening never looks at.
            self._pool_matrices = [op.matrix() for op in self._pool_ops]
            self._pool_eig = None
        self._configured = True

    def _run_kwargs(self, atoms) -> dict:
        """Forward the geometry to :meth:`run` for the ``output.txt`` metadata."""
        return {"geometry": atoms, **self.run_options}

    # -- energy / gradient ------------------------------------------------ #

    def _analytic_gradients(self, psi: np.ndarray) -> np.ndarray:
        r"""Exact pool gradients ``g_i = 2 Re<H psi | A_i psi>`` (reference)."""
        h_psi = self._h_matrix @ psi
        grads = np.empty(len(self._pool_matrices))
        for i, a in enumerate(self._pool_matrices):
            grads[i] = 2.0 * np.real(np.vdot(h_psi, a @ psi))
        return grads

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
                w, V = np.linalg.eigh(-1j * np.asarray(a))
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

    def _finite_difference_gradients(self, psi: np.ndarray,
                                     eps: float = 1e-4) -> np.ndarray:
        r"""Classical gradient: central finite difference in each pool direction.

        ``g_i ~= [E_i(+eps) - E_i(-eps)] / (2 eps)`` -- a purely classical
        estimate that evaluates the energy at *shifted parameter* values.
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
        if self.gradient == "parameter-shift":
            return self._parameter_shift_gradients(psi)
        if self.gradient == "finite_difference":
            return self._finite_difference_gradients(psi)
        return self._analytic_gradients(psi)                      # "analytic"

    def _select_operator(self, grads: np.ndarray, iteration: int) -> int:
        """Index of the pool operator to append this iteration.

        ADAPT-VQE's greedy rule: the largest-magnitude gradient.  This is the
        single **selection hook** subclasses override to change *which* operator
        grows the ansatz -- a subclass may, for instance, sample
        it stochastically from a softmax of the gradients.  Convergence
        (``max|grad| < tol``) is decided by the caller, independently of the
        selection, so overriding this never changes the stopping criterion.
        """
        return int(np.argmax(np.abs(grads)))

    #: Registers at least this wide use the particle-number sector by default.
    SECTOR_AUTO_QUBITS = 16
    #: Drivers whose states cannot live in a sector override this.
    _supports_sector = True

    #: Largest ``sector.dim * len(terms)`` product worth checking for leakage.
    SECTOR_GUARD_WORK = 20_000_000

    def _resolve_sector(self, n_qubits: int, qubit_h=None, operators=()):
        """The :class:`~carcara.core.sector.ParticleSector` to simulate, or ``None``.

        A sector is only safe when the Hamiltonian *and* every pool generator
        keep the ansatz inside it: :meth:`~carcara.core.sector.ParticleSector.restrict`
        drops whatever leaves, which would quietly replace a generator by its
        projection (``exp(PAP) != P exp(A) P``).  The qubit pool's individual
        Pauli strings do leak by construction.  An explicit ``sector=True``
        raises for such a pool; the automatic choice falls back to the full
        register and says so.
        """
        spec = self.sector
        internal = self.ansatz_provider() is None
        if isinstance(spec, str):
            use = (int(n_qubits) >= self.SECTOR_AUTO_QUBITS
                   and self._supports_sector and internal)
        else:
            use = bool(spec)
        if not use:
            return None
        if not self._supports_sector:
            raise NotImplementedError(
                f"{type(self).__name__} builds full-register reference states "
                "and cannot run in a particle-number sector (sector=True)")
        if not internal:
            raise ValueError(
                "sector=True needs the internal state-vector backend; executing "
                "the ansatz as a circuit prepares full-register states")
        from ..core.sector import ParticleSector
        sector = ParticleSector(n_qubits, self.num_particles, self.mapping,
                                two_qubit_reduction=self.two_qubit_reduction)

        leaking = self._sector_leak(sector, qubit_h, operators)
        if leaking is not None:
            message = (f"{leaking} does not conserve the "
                       f"{self.num_particles} particle-number sector, so "
                       f"restricting it would change the operator "
                       f"(exp(PAP) != P exp(A) P)")
            if not isinstance(spec, str):
                raise ValueError(
                    f"sector=True was requested but {message}; use the "
                    f"'fermionic', 'qeb' or 'ceo' pool, or sector=False")
            warnings.warn(f"simulating the full register: {message}",
                          RuntimeWarning, stacklevel=3)
            return None
        return sector

    def _sector_leak(self, sector, qubit_h, operators) -> str | None:
        """Name of the first operator that leaves ``sector``, or ``None``."""
        def affordable(operator):
            return (operator is not None
                    and sector.dim * max(len(operator.terms), 1)
                    <= self.SECTOR_GUARD_WORK)

        if affordable(qubit_h) and not sector.conserves(qubit_h):
            return "the Hamiltonian"
        for op in operators:
            generator = getattr(op, "generator", None)
            if affordable(generator) and not sector.conserves(generator):
                return f"pool operator {getattr(op, 'label', '?')!r}"
        return None

    # -- checkpoint / resume helpers ------------------------------------- #

    def _iteration_payload(self, it: AdaptIteration) -> dict:
        """One :class:`AdaptIteration` as checkpoint data (energy in Hartree)."""
        return {"operator_label": it.operator_label,
                "operator_kind": it.operator_kind,
                "max_gradient": float(it.max_gradient),
                "energy_ha": float(to_hartree(it.energy,
                                              self._energy_unit_label())),
                "cnot_count": it.cnot_count, "depth": it.depth,
                "num_parameters": int(it.num_parameters)}

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
        """Append a checkpoint's generators to ``ansatz``; return its history."""
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
            num_parameters=int(it["num_parameters"]))
            for it in status.get("iterations", [])]
        params = np.asarray(record.parameters, dtype=float)
        energy = (float(record.energy) if record.energy is not None
                  else self.ansatz_energy(ansatz, params))
        return {"parameters": params, "selected": selected,
                "iterations": iterations,
                "num_evaluations": int(status.get("num_evaluations", 0)),
                "optimizer_failures": [(int(s), str(m)) for s, m
                                       in status.get("optimizer_failures", [])],
                "energy": energy, "unit": unit}

    def _new_ansatz(self) -> AdaptAnsatz:
        """A fresh growable ansatz on the configured evaluation backend.

        Routes through :meth:`~carcara.algorithms.base.VariationalDriver.circuit_provider`,
        so the ansatz evaluates its states either with the internal state-vector
        backend or by executing circuits on Qiskit / Braket / Cirq.
        """
        return AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                           self.mapping, sparse=getattr(self, "_sparse", False),
                           provider=self.ansatz_provider(),
                           two_qubit_reduction=self.two_qubit_reduction,
                           num_particles=self.num_particles,
                           sector=self._sector)

    def _profile(self, ansatz) -> CircuitMetrics:
        """Compiled-circuit metrics for ``ansatz`` on the configured provider."""
        if not self.profile:
            return CircuitMetrics(None, None, ansatz.num_parameters)
        from ..backends.providers import build_provider
        provider = (None if self.backend_provider == "qiskit"
                    else build_provider(self.backend_provider))
        # The reference *qubits* (not the spin-orbital occupations): they
        # differ under the parity / Bravyi-Kitaev maps and the tapered register.
        return profile_ansatz(self.n_qubits, ansatz.reference_qubits(),
                              ansatz.operators, provider=provider)

    def reference_energy(self) -> float:
        return self.energy(self._new_ansatz().reference_state())

    # -- output.txt logging ---------------------------------------------- #

    def _make_logger(self, output_file, geometry, cell, ref_energy,
                     max_iterations, gradient_tol):
        """Create an :class:`AdaptOutputLogger` and write the header blocks.

        Returns ``None`` when ``output_file`` is not given (logging disabled).
        Resolves the geometry/cell from an ASE ``Atoms`` object or a
        ``(symbols, positions)`` pair, and converts geometry/energy into the
        configured output units (**eV / Angstrom** by default).
        """
        if output_file is None:
            return None
        from ..utils.logging import AdaptOutputLogger

        symbols, positions, geom_cell = _resolve_geometry(geometry)
        cell = geom_cell if cell is None else cell

        # Geometry from ASE is in Angstrom; convert to Bohr only if requested.
        if self.atomic_units:
            if positions is not None:
                positions = np.asarray(positions, float) * ANGSTROM_TO_BOHR
            if cell is not None:
                cell = np.asarray(cell, float) * ANGSTROM_TO_BOHR

        logger = AdaptOutputLogger(output_file, n_qubits=self.n_qubits)
        logger.write_metadata(
            symbols=symbols, positions=positions, cell=cell,
            units=self._length_unit_label(),
            title=f"ADAPT-VQE ({self.pool.__class__.__name__}, "
                  f"{self.n_qubits} qubits)")
        logger.write_optimizer_setup(
            optimizer_method=self.optimizer.method,
            reference_energy=self._to_energy_units(ref_energy),
            energy_unit=self._energy_unit_label(),
            gradient_tol=gradient_tol, max_iterations=max_iterations,
            extra={"mapping": self.mapping,
                   "num_particles": self.num_particles,
                   # The pool's type and size, before the iteration table.
                   "pool": getattr(self.pool, "name", "?"),
                   "pool_class": self.pool.__class__.__name__,
                   "pool_size": len(self._pool_ops)})
        return logger

    def _expressivity_wanted(self, log_expressivity) -> bool:
        """Whether to spend the samples on the expressivity score this run.

        ``"auto"`` (the default) asks only where a state preparation is cheap:
        the sparse and sector backends work on a compressed state at any width,
        while the dense backend allocates the full ``2^n`` vector
        ``2 * EXPRESSIVITY_SAMPLES`` times per iteration -- 78 s at 12 qubits
        against 0.07 s sparse.  Without this guard every verbose run of a
        realistic active space would pay that per grown operator.
        """
        if log_expressivity is True:
            return True
        if not log_expressivity:                     # False / None
            return False
        if str(log_expressivity).lower() != "auto":
            raise ValueError("log_expressivity must be 'auto', True or False, "
                             f"got {log_expressivity!r}")
        if self._sector is not None or self._sparse:
            return True
        return self.n_qubits <= EXPRESSIVITY_DENSE_MAX_QUBITS

    def _expressivity(self, ansatz) -> float:
        """Expressivity score ``E`` of the current ansatz (KL from Haar).

        Uses the number-conserving sector dimension as the fixed Haar reference,
        so scores are comparable across iterations (see
        :mod:`carcara.algorithms.expressivity`).
        """
        from .expressivity import (active_space_dimension,
                                   calculate_kl_divergence,
                                   sample_pqc_fidelities)
        # The sector lives on the spin-orbitals: a tapered register has two more.
        n_modes = self.n_qubits + (2 if self.two_qubit_reduction else 0)
        dim = active_space_dimension(n_modes, self.num_particles)
        fidelities = sample_pqc_fidelities(ansatz,
                                           num_samples=EXPRESSIVITY_SAMPLES,
                                           rng=self._expr_rng)
        return calculate_kl_divergence(fidelities, self.n_qubits, num_bins=75,
                                       dim=dim)

    # -- main loop -------------------------------------------------------- #

    def run(self, initial_parameters=None, callback=None,
            geometry=None, cell=None,
            log_expressivity="auto") -> ADAPTVQEResult:
        """Grow and optimize the ansatz until convergence.

        Everything that also lives on the constructor -- the stopping controls
        (``max_iterations`` / ``gradient_tolerance``), the ``output`` log path and
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
            :class:`~carcara.algorithms.expressivity.ADAPTExpressivityTracker` to
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
        log_expressivity : {"auto", True, False}
            Compute the expressivity score each iteration, for the ``expr``
            column of the trace and the ``output.txt`` log.  The default
            ``"auto"`` computes it only where it is cheap -- on the sparse or
            sector backend at any width, and on the dense backend up to
            :data:`EXPRESSIVITY_DENSE_MAX_QUBITS` qubits -- because it is
            ``2 * EXPRESSIVITY_SAMPLES`` state preparations per iteration, which
            on a *dense* 12-qubit register is 78 s against 0.07 s on the sparse
            one.  ``True`` computes it regardless, ``False`` never does (the
            column and the log entry then read ``-`` / ``(not computed)``).
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
        gradient_tol = self.gradient_tolerance
        output_file = self.output
        verbose = self.verbose
        want_expressivity = self._expressivity_wanted(log_expressivity)

        # In calculator mode the wall clock is seeded in calculate() so it spans
        # the integration too; in direct mode it starts here.
        timings, run_t0 = self._make_timings()

        ansatz = self._new_ansatz()
        self.ansatz = ansatz        # the grown ansatz, for measured energies
        params = (np.asarray(initial_parameters, dtype=float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        ref_energy = self.energy(ansatz.reference_state())

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

        # Banner to standard output *before* any data is written to output.txt.
        if verbose:
            self._show_banner()

        logger = self._make_logger(output_file, geometry, cell, ref_energy,
                                   max_iterations, gradient_tol)
        e_unit = self._energy_unit_label()

        if verbose:
            self._print_header(ref_energy, e_unit)
            self._print_iteration_heading(e_unit)

        iterations: list[AdaptIteration] = []
        selected: list[str] = []
        total_evals = 0
        converged = False
        optimizer_failures: list[tuple[int, str]] = []
        max_grad = np.inf
        energy = ref_energy
        metrics: CircuitMetrics | None = None
        final_expr: float | None = None
        if restored is not None:
            iterations = restored["iterations"]
            selected = restored["selected"]
            total_evals = restored["num_evaluations"]
            optimizer_failures = restored["optimizer_failures"]
            energy = restored["energy"]
            if verbose:
                print(f"resumed from {self.resume_path!r}: {len(selected)} "
                      f"operators, E = {self._to_energy_units(energy):+.8f} "
                      f"{e_unit}")

        def checkpoint(complete: bool, converged: bool):
            """Write the state as it stands (every run writes at the end)."""
            self._write_checkpoint(self._checkpoint_record(
                ansatz, params, energy, {
                    "complete": bool(complete), "converged": bool(converged),
                    "iteration": len(iterations),
                    "max_gradient": float(max_grad) if np.isfinite(max_grad)
                    else None,
                    "num_evaluations": int(total_evals),
                    "reference_energy": float(ref_energy),
                    "optimizer_failures": [[int(s), str(m)]
                                           for s, m in optimizer_failures],
                    "iterations": [self._iteration_payload(it)
                                   for it in iterations]}))

        try:
            while len(selected) < max_iterations:
                with timings.time("gradient screening"):
                    psi = ansatz.state(params) if ansatz.num_parameters else \
                        ansatz.reference_state()
                    grads = self._gradients(psi)
                max_grad = _max_abs(grads)
                if max_grad < gradient_tol:
                    converged = True
                    break
                idx = self._select_operator(grads, len(selected))

                op = self._pool_ops[idx]
                ansatz.append(op)
                selected.append(op.label)

                # Warm start: reuse previous optimum, new parameter set to 0.
                # `quenching` decides whether the previous angles are re-optimized
                # alongside the new one or frozen at their prior values.
                previous_energy = energy
                with timings.time("parameter optimization"):
                    result = self._optimize_grown(
                        lambda t: self.ansatz_energy(ansatz, t), params)
                params = np.asarray(result.x, dtype=float)
                energy = float(result.fun)
                total_evals += result.nfev
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
                if want_expressivity and (verbose or logger is not None):
                    with timings.time("expressivity"):
                        expr = self._expressivity(ansatz)
                    final_expr = expr

                if verbose:
                    self._print_iteration(len(iterations) + 1, op, max_grad,
                                          energy, energy - previous_energy,
                                          metrics, e_unit, expressivity=expr)
                iterations.append(AdaptIteration(
                    operator_label=op.label, operator_kind=op.kind,
                    max_gradient=max_grad,
                    energy=self._to_energy_units(energy),
                    cnot_count=metrics.cnot_count, depth=metrics.depth,
                    num_parameters=ansatz.num_parameters))

                if logger is not None:
                    logger.write_iteration(
                        iteration=len(iterations), pool_operators=self._pool_ops,
                        gradients=grads, selected_index=idx, expressivity=expr,
                        energy=self._to_energy_units(energy), energy_unit=e_unit,
                        num_parameters=ansatz.num_parameters, metrics=metrics)

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
            if not converged:
                # The loop stopped on max_iterations (or ran none at all).
                # Screen once more so the reported gradient -- and the
                # convergence flag -- describe the *final* state rather than
                # the one before the last operator was appended and optimized.
                with timings.time("gradient screening"):
                    psi = ansatz.state(params) if ansatz.num_parameters else \
                        ansatz.reference_state()
                    max_grad = _max_abs(self._gradients(psi))
                converged = bool(max_grad < gradient_tol)

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
                    operator_sequence=selected)
        finally:
            if logger is not None:
                logger.close()

        if optimizer_failures:
            steps = ", ".join(str(step) for step, _msg in optimizer_failures)
            warnings.warn(
                f"the inner optimizer reported no convergence in "
                f"{len(optimizer_failures)} of {len(iterations)} growth steps "
                f"(steps {steps}); last message: "
                f"{optimizer_failures[-1][1]!r}", RuntimeWarning, stacklevel=2)

        # Fold the (calculator-mode) integration stage in, then set the wall time.
        self._finalize_timings(timings, run_t0)

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
            optimizer_failures=optimizer_failures,
            metrics=metrics,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile,
            energy_unit=e_unit)

        if verbose:
            self._print_summary(result, e_unit, timings)
        return result

    # -- excited states / energy levels (DeflationMixin hook) ------------- #

    def _deflated_ground(self, states, beta, *, state_index: int = 0,
                         max_iterations: int | None = None,
                         gradient_tolerance: float | None = None):
        r"""Grow a fresh deflated ADAPT ansatz orthogonal to ``states``.

        Both the pool-screening gradient and the inner re-optimization carry the
        overlap penalty, so the adaptive ansatz grows toward the next excited
        state; the reported energy is the bare expectation value.  Called per level
        by :meth:`~carcara.algorithms.deflation.DeflationMixin.energy_levels`
        (which also accepts ``max_iterations`` / ``gradient_tolerance``).
        """
        max_it = (self.max_iterations if max_iterations is None
                  else int(max_iterations))
        gtol = (self.gradient_tolerance if gradient_tolerance is None
                else float(gradient_tolerance))
        energy, psi, n_ops, nev = self._grow_deflated(states, beta, max_it, gtol)
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

    def _grow_deflated(self, states, beta, max_iterations, gradient_tol):
        """Grow one deflated ADAPT ansatz; return ``(energy, psi, n_ops, nfev)``.

        With ``states`` empty this is an ordinary ADAPT ground-state growth.  A
        trimmed sibling of :meth:`run` (no logging / profiling / verbose trace)
        used to build each excited state.
        """
        from .deflation import deflation_penalty

        ansatz = self._new_ansatz()
        params = np.zeros(0)
        total_evals = 0
        for _ in range(int(max_iterations)):
            psi = (ansatz.state(params) if ansatz.num_parameters
                   else ansatz.reference_state())
            grads = self._deflated_gradients(psi, states, beta)
            if _max_abs(grads) < gradient_tol:
                break
            idx = self._select_operator(grads, ansatz.num_parameters)
            ansatz.append(self._pool_ops[idx])

            def cost(t, _states=states):
                phi = ansatz.state(t)
                return self.energy(phi) + deflation_penalty(phi, _states, beta)

            result = self._optimize_grown(cost, params)
            params = np.asarray(result.x, dtype=float)
            total_evals += result.nfev
        psi = (ansatz.state(params) if ansatz.num_parameters
               else ansatz.reference_state())
        return self.energy(psi), psi, ansatz.num_parameters, total_evals

    # -- standard-output trace ------------------------------------------- #

    #: Column layout of the per-iteration table: (heading, width).
    #:
    #: One row per grown operator, one column per property computed at that
    #: step: the screening gradient that drove the selection, the energy and
    #: its change, the ansatz expressivity, the circuit cost (parameters,
    #: CNOTs, single-qubit gates, depth) and the selected operator's kind and
    #: label.  The *pool* is not printed -- its name and size are in the
    #: header and its contents go to ``pool.json`` with
    #: ``verbose_operators=True``.
    #:
    #: ``(key, heading, width, format)``.  Headings are terse because the row
    #: has to fit a terminal: at 80 columns the full table is 79 characters
    #: wide once ``npar`` goes.
    _ITERATION_COLUMNS = (("iter", "iter", 4, "d"),
                          ("grad", "|grad|", 8, ".2e"),
                          ("energy", "E", 11, ".6f"),
                          ("dE", "dE", 8, ".1e"),
                          ("expr", "expr", 6, ".2f"),
                          ("npar", "npar", 4, "s"),
                          ("cnot", "cnot", 5, "s"),
                          ("1q", "1q", 5, "s"),
                          ("depth", "depth", 5, "s"),
                          ("type", "type", 6, "s"),
                          ("operator", "operator", 0, "s"))

    #: Order in which columns are sacrificed when the row will not fit, least
    #: costly first: ``npar`` is always equal to ``iter`` and ``dE`` is the
    #: difference of consecutive energies, so neither carries new information;
    #: only after those does the table give up something it exists to show.
    #: ``iter``, ``grad``, ``energy``, ``type`` and ``operator`` are never
    #: dropped.
    _ITERATION_DROP_ORDER = ("npar", "dE", "1q", "depth", "expr", "cnot")

    #: Terminal width assumed when it cannot be detected (piped output).
    FALLBACK_TERMINAL_WIDTH = 80

    def _extra_header_lines(self) -> list[str]:
        """Extra configuration lines for the verbose header (subclass hook).

        A subclass overrides this to report its own
        stochastic-selection temperature schedule.
        """
        return []

    def _iteration_layout(self, e_unit: str = "eV"):
        """The columns that fit on one line, widest-first, cached per run.

        A wrapped row is not a row: the whole point of the table is that one
        iteration is one line, so when the terminal is too narrow the columns
        that are derivable elsewhere are dropped rather than letting every
        iteration spill onto two lines.  The operator label's width is taken
        from the pool, so the choice is made once and every row matches the
        heading.
        """
        if getattr(self, "_layout_cache", None) is not None:
            return self._layout_cache
        labels = [op.label for op in getattr(self, "_pool_ops", []) or []]
        label_width = max([len(label) for label in labels] + [len("operator")])
        available = shutil.get_terminal_size(
            (self.FALLBACK_TERMINAL_WIDTH, 24)).columns

        def width_of(columns):
            total = label_width + len(columns) - 1          # one space between
            for key, heading, width, _fmt in columns:
                if key != "operator":
                    total += max(width, len(self._heading(heading, e_unit)))
            return total

        columns = list(self._ITERATION_COLUMNS)
        # One column at a time: dropping a whole class to save one character
        # would throw away four of them for nothing.
        for victim in self._ITERATION_DROP_ORDER:
            if width_of(columns) <= available:
                break
            columns = [c for c in columns if c[0] != victim]
        self._layout_cache = tuple(columns)
        return self._layout_cache

    @staticmethod
    def _heading(heading: str, e_unit: str) -> str:
        """The printed heading -- the energy column carries the unit."""
        return f"E ({e_unit})" if heading == "E" else heading

    def _iteration_heading(self, e_unit: str) -> str:
        """The column-heading line of the per-iteration table."""
        cells = []
        for key, heading, width, _fmt in self._iteration_layout(e_unit):
            label = self._heading(heading, e_unit)
            cells.append(label if key == "operator"
                         else f"{label:>{max(width, len(label))}}")
        return " ".join(cells).rstrip()

    def _iteration_rule(self) -> str:
        """Horizontal rule spanning the whole row, labels included."""
        e_unit = self._energy_unit_label()
        labels = [op.label for op in getattr(self, "_pool_ops", []) or []]
        width = max([len(label) for label in labels] + [len("operator")])
        for key, heading, column, _fmt in self._iteration_layout(e_unit):
            if key != "operator":
                width += max(column, len(self._heading(heading, e_unit))) + 1
        return "-" * width

    def _print_header(self, ref_energy: float, e_unit: str) -> None:
        """Print the run configuration banner.

        The qubit Hamiltonian's Pauli-string expansion is deliberately **not**
        printed -- it runs to thousands of lines for a realistic active space and
        drowns the run trace.  Only its size is reported; the operator itself is
        on ``self.hamiltonian`` and can be rendered with
        :func:`~carcara.algorithms.base.format_pauli_sum` or saved with
        ``save_hamiltonian=``.
        """
        rule = "=" * 70
        print(rule)
        print(f"ADAPT-VQE | mapping: {self.mapping} | {self.n_qubits} qubits "
              f"| device: {self.device}")
        screened_analytically = (getattr(self, "_sparse", False)
                                 or getattr(self, "_sector", None) is not None)
        grad_label = ("analytic/sparse" if screened_analytically
                      else self.gradient)
        # The class name is dropped: it is redundant with the pool's own name
        # and pushed this line past 80 columns on a realistic pool.
        print(f"pool: {getattr(self.pool, 'name', '?')} | "
              f"{len(self._pool_ops)} operators | "
              f"optimizer: {self.optimizer.method} | gradient: {grad_label}")
        print(f"k-points: {self._kpts_label()} | spin: {self.spin} "
              f"| reference: {self.initial_state}")
        print(f"backend provider: {self.backend_provider} | circuit execution: "
              f"{self.execute_circuits} | quenching: {self.quenching}")
        for line in self._extra_header_lines():
            print(line)
        print(rule)
        n_terms = len(self.hamiltonian.simplify().terms)
        print(f"Qubit Hamiltonian: {n_terms} Pauli terms")
        print(f"Hartree-Fock reference energy = "
              f"{self._to_energy_units(ref_energy):+.8f} {e_unit}")
        print(rule)

    def _print_iteration_heading(self, e_unit: str) -> None:
        """Print the column heading of the per-iteration table."""
        print(self._iteration_heading(e_unit))
        print(self._iteration_rule())

    def _operator_kind(self, op: PoolOperator) -> str:
        """``op.kind`` without the pool-name prefix (the header already has it).

        ``"fermionic-double"`` -> ``"double"``, ``"qeb-single"`` -> ``"single"``;
        a kind that is not prefixed (``"ceo"``, ``"pauli"``) is left alone.
        """
        prefix = f"{getattr(self.pool, 'name', '')}-"
        kind = str(op.kind)
        return kind[len(prefix):] if prefix != "-" and kind.startswith(prefix) \
            else kind

    def _print_iteration(self, iteration: int, op: PoolOperator,
                         max_grad: float, energy: float, delta: float,
                         metrics: CircuitMetrics | None, e_unit: str,
                         expressivity: float | None = None) -> None:
        """Print one iteration as a single, column-aligned line.

        One column per property of the step: the largest pool gradient that
        drove the selection, the energy and its change, the ansatz's
        expressivity (KL divergence from Haar), the parameter count, the
        compiled CNOT / single-qubit-gate counts and depth, and the selected
        operator's kind and label.  Which of those fit is
        :meth:`_iteration_layout`'s decision -- the line is never wrapped.
        Neither the pool nor the operator's Pauli-string expansion is printed;
        they are on ``result.operators`` and, with ``verbose_operators=True``,
        in ``pool.json``.
        """
        def count(value):
            return "-" if value is None else str(value)

        values = {
            "iter": iteration,
            "grad": float(max_grad),
            "energy": self._to_energy_units(energy),
            "dE": self._to_energy_units(delta),
            "expr": (None if expressivity is None
                     or not np.isfinite(expressivity) else float(expressivity)),
            "npar": count(None if metrics is None else metrics.num_operators),
            "cnot": count(None if metrics is None else metrics.cnot_count),
            "1q": count(None if metrics is None else metrics.num_1q_gates),
            "depth": count(None if metrics is None else metrics.depth),
            "type": self._operator_kind(op),
            "operator": op.label,
        }
        cells = []
        for key, heading, width, fmt in self._iteration_layout(e_unit):
            value = values[key]
            width = max(width, len(self._heading(heading, e_unit)))
            if key == "operator":
                cells.append(str(value))
            elif value is None:
                cells.append(f"{'-':>{width}}")
            elif fmt == "s":
                cells.append(f"{value:>{width}}")
            else:
                cells.append(f"{value:>{width}{fmt}}")
        print(" ".join(cells).rstrip())

    def _print_summary(self, result: ADAPTVQEResult, e_unit: str,
                       timings=None) -> None:
        """Print the closing summary: result line plus timings / resources."""
        rule = "=" * 70
        print(self._iteration_rule())
        status = "converged" if result.converged else "NOT converged"
        print(f"{status}: E = {result.optimal_energy:+.8f} {e_unit}, "
              f"{result.num_operators} operators, "
              f"|grad| = {result.final_max_gradient:.2e}")
        if timings is not None:
            print(timings.format_report())
        print(rule)
