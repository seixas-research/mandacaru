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

from dataclasses import dataclass, field

import numpy as np

from ..circuits.adapt_ansatz import AdaptAnsatz
from ..circuits.pools import PoolBase, PoolOperator, build_pool
from ..circuits.profiling import CircuitMetrics, profile_ansatz
from ..units import ANGSTROM_TO_BOHR, from_hartree
from .base import VariationalDriver, format_pauli_sum
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
    """Record of one ADAPT-VQE macro-iteration."""

    operator_label: str
    operator_kind: str
    max_gradient: float
    energy: float
    cnot_count: int | None
    depth: int | None
    num_parameters: int


@dataclass
class ADAPTVQEResult:
    """Result of an :class:`ADAPTVQE` run."""

    optimal_energy: float
    optimal_parameters: np.ndarray
    reference_energy: float
    converged: bool
    final_max_gradient: float
    operators: list[str]                      # selected operator labels, in order
    iterations: list[AdaptIteration] = field(default_factory=list)
    num_evaluations: int = 0                  # total inner cost evaluations
    metrics: CircuitMetrics | None = None     # final compiled-circuit metrics
    timings: dict | None = None               # per-stage wall time / cores / memory
    integration_profile: dict | None = None   # real-space integration profile

    @property
    def num_operators(self) -> int:
        return len(self.operators)

    @property
    def energy_history(self) -> list[float]:
        return [it.energy for it in self.iterations]

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
        How the pool screening gradients are evaluated -- ``"finite_difference"``
        (default; a finite-difference estimate from shifted parameters) or
        ``"parameter-shift"`` (the quantum parameter-shift rule).
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
        ``{"size": (n1, n2, n3), "gamma": True}`` (``gamma=True`` centres the mesh
        on Gamma).  The real-space engine solves a Gamma-point (molecular)
        problem, so a denser mesh is generated and exposed on :attr:`kpoints` but
        raises ``NotImplementedError`` at run time.
    spin : bool
        Spin polarization (default ``False``).  ``False`` is a closed-shell
        reference (``n_alpha == n_beta``); ``True`` is a spin-polarized (high-spin)
        reference.  The **initial spin state is read primarily from the ASE
        geometry's initial magnetic moments** (``Atoms(..., magmoms=...)``): their
        rounded total is the number of unpaired electrons, so a triplet is set with
        ``magmoms=[1, 1]`` (see
        :func:`~carcara.algorithms._hamiltonian_from_atoms.resolve_num_unpaired`).
        ``spin`` is the fallback when no magnetic moments are set.  Only affects the
        calculator-mode Hamiltonian builder; genuinely open-shell (odd-electron)
        systems raise ``NotImplementedError`` (RHF-only integrals).
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
    run_options : dict, optional
        Extra keyword arguments forwarded to :meth:`run` on each calculator
        evaluation (e.g. ``{"log_expressivity": False}``).  Only arguments that
        :meth:`run` accepts are valid here -- the stopping controls
        (``max_iterations`` / ``gradient_tolerance``), ``output`` and ``verbose``
        are constructor arguments, not ``run`` arguments.
    """

    _GRADIENTS = ("finite_difference", "parameter-shift")
    _result_attr = "adapt_result"
    _default_sparse = "auto"

    def __init__(self,
                 hamiltonian=None,
                 pool="fermionic",
                 basis="FAO",
                 num_particles=None,
                 n_spatial_orbitals=None,
                 optimizer: str | Optimizer = "COBYLA",
                 mapping: str = "jordan_wigner",
                 gradient: str = "finite_difference",
                 device: str = "AER_simulator",
                 max_iterations: int = 50,
                 gradient_tolerance: float = 1e-3,
                 output: str | None = None,
                 profile: bool = True,
                 verbose: bool = True,
                 sparse: bool | str = "auto",
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
                 run_options: dict | None = None, **calc_kwargs):
        super().__init__(optimizer=optimizer, mapping=mapping, basis=basis,
                         device=device, grid=grid, h=h, kpts=kpts, spin=spin,
                         initial_state=initial_state, charge=charge,
                         n_electrons=n_electrons, frozen_core=frozen_core,
                         frozen_orbitals=frozen_orbitals,
                         hamiltonian_builder=hamiltonian_builder,
                         run_options=run_options, verbose=verbose,
                         sparse=sparse, **calc_kwargs)

        self.profile = profile
        # Validate the enumerated gradient option up front.
        if gradient not in self._GRADIENTS:
            raise ValueError(
                f"unknown gradient {gradient!r}; use one of {self._GRADIENTS}")
        self.gradient = gradient

        # Run defaults (also the defaults for the ASE-calculator evaluation).
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        self.output = output

        # Output-unit convention (see class docstring).
        self.atomic_units = bool(atomic_units)
        self.energy_units = "Ha" if atomic_units else "eV"
        self.length_units = "bohr" if atomic_units else "angstrom"

        self._pool_spec = pool
        # Seeded RNG for reproducible expressivity logging (output.txt).
        self._expr_rng = np.random.default_rng(0)

        # Configure eagerly when a Hamiltonian is given (direct mode); otherwise
        # defer to the first calculator evaluation (the ASE hook in the base).
        if hamiltonian is not None:
            self._configure(hamiltonian, num_particles, n_spatial_orbitals)
            self._built_from_hamiltonian = True

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
                                   mapping=self.mapping)
        self.num_particles = (tuple(num_particles) if num_particles is not None
                              else self.pool.num_particles)

        # Materialize the (dense or sparse) qubit Hamiltonian on the base; a dense
        # pool stores every operator's matrix *and* eigendecomposition (two
        # 2^n x 2^n arrays each), ~46 GB for a 12-qubit water active space, so
        # ``sparse="auto"`` keeps large active spaces as sparse matrices and
        # screens with the exact analytic gradient, densifying only selected
        # operators (in the growable ansatz).
        qubit_h = self._as_pauli_sum(hamiltonian, self.pool.n_qubits)
        self._materialize_hamiltonian(qubit_h, self.pool.n_qubits)

        self._pool_ops = self.pool.operators()
        if self._sparse:
            self._pool_matrices = [op.generator.to_sparse_matrix()
                                   for op in self._pool_ops]
            self._pool_eig = None
        else:
            # For A anti-Hermitian, -iA is Hermitian: -iA = V diag(w) V^dag, so
            # exp(theta A) = V diag(exp(i theta w)) V^dag.  The unique positive
            # eigenvalue *differences* are the frequencies of E(theta), used by the
            # parameter-shift gradient.
            self._pool_matrices = [op.matrix() for op in self._pool_ops]
            self._pool_eig = []
            for a in self._pool_matrices:
                w, V = np.linalg.eigh(-1j * a)
                self._pool_eig.append((w, V, _unique_frequencies(w)))
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

    def _pool_energy_at(self, psi: np.ndarray, i: int, theta: float) -> float:
        r"""Energy after appending ``exp(theta A_i)`` to ``psi``.

        ``E_i(theta) = <psi| e^{-theta A_i} H e^{theta A_i} |psi>`` evaluated from
        the cached eigendecomposition of ``A_i`` (no matrix exponential).
        """
        w, V, _ = self._pool_eig[i]
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
            _, _, freqs = self._pool_eig[i]
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

        In the sparse pool path the per-operator eigendecompositions the
        ``"finite_difference"`` / ``"parameter-shift"`` estimators rely on are not
        materialized, so screening uses the exact analytic gradient
        ``g_i = 2 Re<H psi | A_i psi>`` (a sparse matrix-vector product) -- which is
        the very quantity those estimators approximate.
        """
        if getattr(self, "_sparse", False):
            return self._analytic_gradients(psi)
        if self.gradient == "parameter-shift":
            return self._parameter_shift_gradients(psi)
        return self._finite_difference_gradients(psi)      # "finite_difference"

    def _select_operator(self, grads: np.ndarray, iteration: int) -> int:
        """Index of the pool operator to append this iteration.

        ADAPT-VQE's greedy rule: the largest-magnitude gradient.  This is the
        single **selection hook** subclasses override to change *which* operator
        grows the ansatz -- e.g. :class:`~carcara.algorithms.vasqe.VASQE` samples
        it stochastically from a softmax of the gradients.  Convergence
        (``max|grad| < tol``) is decided by the caller, independently of the
        selection, so overriding this never changes the stopping criterion.
        """
        return int(np.argmax(np.abs(grads)))

    def reference_energy(self) -> float:
        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping, sparse=getattr(self, "_sparse", False))
        return self.energy(ansatz.reference_state())

    # -- output.txt logging ---------------------------------------------- #

    def _to_energy_units(self, energy_ha):
        """Convert a Hartree energy to the configured output units (eV default)."""
        return float(from_hartree(energy_ha, self.energy_units))

    def _energy_unit_label(self) -> str:
        return "Ha" if self.energy_units.lower() in ("ha", "hartree", "au") \
            else "eV"

    def _length_unit_label(self) -> str:
        return "Bohr" if self.length_units.lower() in ("bohr", "au", "a0") \
            else "Angstrom"

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

        logger = AdaptOutputLogger(output_file)
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
                   "pool_size": len(self._pool_ops)})
        return logger

    def _expressivity(self, ansatz) -> float:
        """Expressivity score ``E`` of the current ansatz (KL from Haar).

        Uses the number-conserving sector dimension as the fixed Haar reference,
        so scores are comparable across iterations (see
        :mod:`carcara.algorithms.expressivity`).
        """
        from .expressivity import (active_space_dimension,
                                   calculate_kl_divergence,
                                   sample_pqc_fidelities)
        dim = active_space_dimension(self.n_qubits, self.num_particles)
        fidelities = sample_pqc_fidelities(ansatz, num_samples=400,
                                           rng=self._expr_rng)
        return calculate_kl_divergence(fidelities, self.n_qubits, num_bins=75,
                                       dim=dim)

    # -- main loop -------------------------------------------------------- #

    def run(self, initial_parameters=None, callback=None,
            geometry=None, cell=None,
            log_expressivity: bool = True) -> ADAPTVQEResult:
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
        log_expressivity : bool
            Compute and log the expressivity score each iteration when the
            instance's ``output`` path is set (default ``True``).
        """
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

        # In calculator mode the wall clock is seeded in calculate() so it spans
        # the integration too; in direct mode it starts here.
        timings, run_t0 = self._make_timings()

        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping, sparse=self._sparse)
        params = (np.asarray(initial_parameters, dtype=float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        ref_energy = self.energy(ansatz.reference_state())

        # Banner to standard output *before* any data is written to output.txt.
        if verbose:
            self._show_banner()

        logger = self._make_logger(output_file, geometry, cell, ref_energy,
                                   max_iterations, gradient_tol)
        e_unit = self._energy_unit_label()

        if verbose:
            self._print_header(ref_energy, e_unit)

        iterations: list[AdaptIteration] = []
        selected: list[str] = []
        total_evals = 0
        converged = False
        max_grad = np.inf
        energy = ref_energy
        metrics: CircuitMetrics | None = None
        final_expr: float | None = None

        try:
            for _ in range(max_iterations):
                with timings.time("gradient screening"):
                    psi = ansatz.state(params) if ansatz.num_parameters else \
                        ansatz.reference_state()
                    grads = self._gradients(psi)
                max_grad = float(np.max(np.abs(grads)))
                if max_grad < gradient_tol:
                    converged = True
                    break
                idx = self._select_operator(grads, len(selected))

                op = self._pool_ops[idx]
                ansatz.append(op)
                selected.append(op.label)

                # Warm start: reuse previous optimum, new parameter set to 0.
                x0 = np.concatenate([params, [0.0]])
                with timings.time("parameter optimization"):
                    result = self.optimizer.minimize(
                        lambda t: self.energy(ansatz.state(t)), x0)
                params = np.asarray(result.x, dtype=float)
                energy = float(result.fun)
                total_evals += result.nfev

                if verbose:
                    self._print_iteration(len(iterations) + 1, op, max_grad,
                                          energy, e_unit)

                with timings.time("circuit profiling"):
                    metrics = (profile_ansatz(self.n_qubits, ansatz.occupied,
                                              ansatz.operators)
                               if self.profile else
                               CircuitMetrics(None, None, ansatz.num_parameters))
                iterations.append(AdaptIteration(
                    operator_label=op.label, operator_kind=op.kind,
                    max_gradient=max_grad, energy=energy,
                    cnot_count=metrics.cnot_count, depth=metrics.depth,
                    num_parameters=ansatz.num_parameters))

                if logger is not None:
                    with timings.time("expressivity"):
                        expr = (self._expressivity(ansatz)
                                if log_expressivity else None)
                    final_expr = expr
                    logger.write_iteration(
                        iteration=len(iterations), pool_operators=self._pool_ops,
                        gradients=grads, selected_index=idx, expressivity=expr,
                        energy=self._to_energy_units(energy), energy_unit=e_unit,
                        num_parameters=ansatz.num_parameters, metrics=metrics)

                if callback is not None:
                    callback({
                        "iteration": len(iterations),
                        "num_operators": ansatz.num_parameters,
                        "ansatz": ansatz,
                        "parameters": params,
                        "energy": energy,
                        "max_gradient": max_grad,
                        "operator_label": op.label,
                        "metrics": metrics,
                    })
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

        if not converged and len(iterations) == max_iterations:
            # Loop exhausted without meeting the gradient threshold; report the
            # final screening gradient so callers can see how close it got.
            with timings.time("gradient screening"):
                psi = ansatz.state(params) if ansatz.num_parameters else \
                    ansatz.reference_state()
                max_grad = float(np.max(np.abs(self._gradients(psi))))

        # Fold the (calculator-mode) integration stage in, then set the wall time.
        self._finalize_timings(timings, run_t0)

        result = ADAPTVQEResult(
            optimal_energy=energy,
            optimal_parameters=params,
            reference_energy=ref_energy,
            converged=converged,
            final_max_gradient=max_grad,
            operators=selected,
            iterations=iterations,
            num_evaluations=total_evals,
            metrics=metrics,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile)

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

        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping, sparse=self._sparse)
        params = np.zeros(0)
        total_evals = 0
        for _ in range(int(max_iterations)):
            psi = (ansatz.state(params) if ansatz.num_parameters
                   else ansatz.reference_state())
            grads = self._deflated_gradients(psi, states, beta)
            if float(np.max(np.abs(grads))) < gradient_tol:
                break
            idx = self._select_operator(grads, ansatz.num_parameters)
            ansatz.append(self._pool_ops[idx])
            x0 = np.concatenate([params, [0.0]])

            def cost(t, _states=states):
                phi = ansatz.state(t)
                return self.energy(phi) + deflation_penalty(phi, _states, beta)

            result = self.optimizer.minimize(cost, x0)
            params = np.asarray(result.x, dtype=float)
            total_evals += result.nfev
        psi = (ansatz.state(params) if ansatz.num_parameters
               else ansatz.reference_state())
        return self.energy(psi), psi, ansatz.num_parameters, total_evals

    # -- standard-output trace ------------------------------------------- #

    def _print_header(self, ref_energy: float, e_unit: str) -> None:
        """Print the run banner and the qubit Hamiltonian as Pauli strings."""
        rule = "=" * 70
        print(rule)
        print(f"ADAPT-VQE  |  mapping: {self.mapping}  |  {self.n_qubits} qubits "
              f"|  device: {self.device}")
        grad_label = ("analytic (sparse pool)" if getattr(self, "_sparse", False)
                      else self.gradient)
        print(f"pool: {self.pool.__class__.__name__}  |  "
              f"optimizer: {self.optimizer.method}  |  gradient: {grad_label}")
        print(f"k-points: {self._kpts_label()}  |  spin-polarized: {self.spin}  "
              f"|  initial state: {self.initial_state}")
        print(rule)
        n_terms = len(self.hamiltonian.simplify().terms)
        print(f"Qubit Hamiltonian ({n_terms} Pauli terms):")
        print(format_pauli_sum(self.hamiltonian))
        print(f"Hartree-Fock reference energy = "
              f"{self._to_energy_units(ref_energy):+.8f} {e_unit}")
        print(rule)

    def _print_iteration(self, iteration: int, op: PoolOperator,
                         max_grad: float, energy: float, e_unit: str) -> None:
        """Print one iteration: the selected operator as Pauli strings."""
        print(f"\n[iter {iteration}] selected {op.label}  "
              f"(kind={op.kind}, |grad|={max_grad:.6e})")
        print("  ansatz operator (Pauli strings):")
        print(format_pauli_sum(op.generator, indent="    "))
        print(f"  energy = {self._to_energy_units(energy):+.8f} {e_unit}")

    def _print_summary(self, result: ADAPTVQEResult, e_unit: str,
                       timings=None) -> None:
        """Print the closing summary: result line plus timings / resources."""
        rule = "=" * 70
        print(rule)
        status = "converged" if result.converged else "not converged"
        print(f"ADAPT-VQE finished ({status}): "
              f"E = {self._to_energy_units(result.optimal_energy):+.8f} {e_unit}, "
              f"{result.num_operators} operators, "
              f"final |grad| = {result.final_max_gradient:.6e}")
        if timings is not None:
            print(timings.format_report())
        print(rule)
