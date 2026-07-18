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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from ..backends.hardware import normalize_device, require_runnable
from ..circuits.pools import PoolBase, PoolOperator, build_pool
from ..core.mapping import Fermion, PauliSum
from ..optimizers.optim import Optimizer
from ..units import ANGSTROM_TO_BOHR, from_hartree


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


def format_pauli_sum(pauli: PauliSum, indent: str = "    ",
                     max_terms: int | None = None) -> str:
    """Render a :class:`~carcara.core.mapping.PauliSum` as ``coeff * PauliString``.

    Real coefficients (Hermitian operators, e.g. the Hamiltonian) print as plain
    reals; purely imaginary ones (anti-Hermitian generators) print with a ``j``.
    ``max_terms`` truncates long sums with a trailing ``... (k more terms)`` line.
    """
    items = sorted(pauli.simplify().terms.items())
    if not items:
        return f"{indent}0"
    shown = items if max_terms is None else items[:max_terms]
    lines = []
    for label, coeff in shown:
        c = complex(coeff)
        if abs(c.imag) < 1e-12:
            coeff_str = f"{c.real:+.6f}"
        elif abs(c.real) < 1e-12:
            coeff_str = f"{c.imag:+.6f}j"
        else:
            coeff_str = f"({c.real:+.6f}{c.imag:+.6f}j)"
        lines.append(f"{indent}{coeff_str} * {label}")
    if max_terms is not None and len(items) > max_terms:
        lines.append(f"{indent}... ({len(items) - max_terms} more terms)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Circuit profiling.
# --------------------------------------------------------------------------- #

@dataclass
class CircuitMetrics:
    """Structural cost of a compiled ansatz circuit."""

    cnot_count: int | None            # number of CNOT (two-qubit) gates
    depth: int | None                 # circuit depth in the native gate set
    num_operators: int                # generators (parameters) in the ansatz
    num_1q_gates: int | None = None   # single-qubit (``u``) gates
    total_gates: int | None = None    # all gates in the native gate set

    def __repr__(self) -> str:
        return (f"CircuitMetrics(cnots={self.cnot_count}, depth={self.depth}, "
                f"gates={self.total_gates}, n_ops={self.num_operators})")


def _pauli_evolution_gate(generator: PauliSum, time=1.0):
    """Qiskit ``PauliEvolutionGate`` realizing ``exp(theta * A)`` for anti-Herm ``A``.

    With ``A`` anti-Hermitian, ``G = i A`` is Hermitian with real coefficients and
    ``exp(-i * time * G) = exp(time * A)``.  Pauli strings are reversed so Qiskit's
    little-endian qubit order matches Carcará's (qubit 0 = leftmost); this does not
    affect the CNOT/depth counts, which are relabeling-invariant.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = [], []
    for label, coeff in generator.simplify().terms.items():
        herm = 1j * coeff                     # coefficient of G = i A (real)
        labels.append(label[::-1])            # leftmost -> qubit 0 (little-endian)
        coeffs.append(complex(herm).real)
    if not labels:
        return None
    op = SparsePauliOp(labels, np.asarray(coeffs, dtype=float))
    return PauliEvolutionGate(op, time=time)


def profile_ansatz(n_qubits: int, occupied: tuple[int, ...],
                   operators: list[PoolOperator]) -> CircuitMetrics:
    """Compile ``|HF>`` + product of ``exp(A_k)`` to ``{cx, u}`` and count cost.

    Returns a :class:`CircuitMetrics`; ``cnot_count`` / ``depth`` are ``None`` if
    Qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit, transpile
    except Exception:
        return CircuitMetrics(None, None, len(operators))

    qc = QuantumCircuit(n_qubits)
    for q in occupied:                        # Hartree-Fock reference preparation
        qc.x(q)
    for op in operators:
        gate = _pauli_evolution_gate(op.generator)
        if gate is not None:
            qc.append(gate, range(n_qubits))

    compiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    counts = compiled.count_ops()
    return CircuitMetrics(cnot_count=int(counts.get("cx", 0)),
                          depth=int(compiled.depth()),
                          num_operators=len(operators),
                          num_1q_gates=int(counts.get("u", 0)),
                          total_gates=int(sum(counts.values())))


# --------------------------------------------------------------------------- #
# Growable state-vector ansatz.
# --------------------------------------------------------------------------- #

class AdaptAnsatz:
    """A product-of-exponentials ansatz that grows one generator at a time.

    ``|psi(theta)> = prod_k exp(theta_k A_k) |HF>`` applied in append order, with
    each ``exp(theta_k A_k)`` evaluated exactly via the eigendecomposition of the
    anti-Hermitian generator ``A_k`` (cached, so cost evaluations are cheap).

    Exposes the interface expected by :class:`~carcara.algorithms.vqe.VQE`
    (``num_parameters``, ``n_qubits``, ``state``, ``reference_state``).
    """

    def __init__(self, n_qubits: int, occupied: tuple[int, ...],
                 mapping: str = "jordan_wigner"):
        self.n_qubits = int(n_qubits)
        self.mapping = mapping
        self.occupied = tuple(occupied)
        self._ops: list[PoolOperator] = []
        self._eig: list[tuple[np.ndarray, np.ndarray]] = []   # (w, V) per generator
        self._hf = self._reference_vector()

    def _reference_vector(self) -> np.ndarray:
        # The Hartree-Fock determinant is a computational basis state whose bits
        # depend on the fermion-to-qubit map (occupation for JW, parity sums for
        # parity / Bravyi-Kitaev).
        from ..core.mapping import reference_qubit_bits

        bits = reference_qubit_bits(self.mapping, self.n_qubits, self.occupied)
        index = 0
        for i, bit in enumerate(bits):
            if bit:
                index |= 1 << (self.n_qubits - 1 - i)         # qubit 0 = MSB
        vec = np.zeros(2 ** self.n_qubits, dtype=complex)
        vec[index] = 1.0
        return vec

    def append(self, op: PoolOperator) -> None:
        """Add a generator to the end of the ansatz."""
        a = op.matrix()
        # A anti-Hermitian => (-i A) is Hermitian: -i A = V diag(w) V^dag, so
        # A = i V diag(w) V^dag and exp(theta A) = V diag(exp(i theta w)) V^dag.
        w, V = np.linalg.eigh(-1j * a)
        self._ops.append(op)
        self._eig.append((w, V))

    @property
    def num_parameters(self) -> int:
        return len(self._ops)

    @property
    def operators(self) -> list[PoolOperator]:
        return list(self._ops)

    def reference_state(self) -> np.ndarray:
        return self._hf.copy()

    def state(self, theta) -> np.ndarray:
        """Prepared state ``prod_k exp(theta_k A_k) |HF>``."""
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.num_parameters:
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {theta.size}")
        psi = self._hf.copy()
        for angle, (w, V) in zip(theta, self._eig):
            psi = V @ (np.exp(1j * angle * w) * (V.conj().T @ psi))
        return psi


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

class ADAPTVQE(Calculator):
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
    basis : str
        Basis set used to build the molecular Hamiltonian from an ASE geometry in
        calculator mode -- one of ``"FAO"`` (default; Full Atomic Orbitals),
        ``"STO-3G"``, ``"6-31G(d)"``, or any other method understood by
        :meth:`carcara.basis.BasisSet.build`.
    num_particles : (int, int), optional
        ``(n_alpha, n_beta)``; required to build a pool from a name and to set the
        Hartree-Fock reference.  Inferred from the pool object otherwise.
    n_spatial_orbitals : int, optional
        Number of spatial orbitals; required to build a pool from a name.
    optimizer : str or Optimizer
        Classical optimizer for the inner re-optimization.  Either a method name
        -- one of ``"COBYLA"`` (default), ``"Nelder-Mead"``, ``"BFGS"`` -- or a
        pre-built :class:`~carcara.optimizers.optim.Optimizer` instance.
    mapping : str
        Fermion-to-qubit mapping -- one of ``"jordan_wigner"`` (default),
        ``"parity"``, ``"bravyi_kitaev"`` -- used when ``hamiltonian`` is a
        ``Fermion`` and to build a named fermionic pool.
    gradient : str
        How the pool screening gradients are evaluated -- ``"classical"``
        (default; a finite-difference estimate from shifted parameters) or
        ``"parameter-shift_rule"`` (the quantum parameter-shift rule).
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
    atomic_units : bool
        Units used in the ``output.txt`` log.  ``False`` (default) logs energies
        in **eV** and lengths in **Angstrom**; ``True`` logs Hartree and Bohr.
        (ASE's ``get_total_energy`` always returns eV, per the ASE convention.)
    grid : Grid, optional
        Real-space integration grid for the calculator-mode ``basis`` builder.
        Defaults to an automatic cube enclosing the molecule.
    charge : int
        Total charge, used to set the electron count in the ``basis`` builder.
    n_electrons : int, optional
        Explicit electron count for the ``basis`` builder (overrides ``charge``).
    hamiltonian_builder : callable, optional
        ``atoms -> (hamiltonian, num_particles, n_spatial_orbitals)``.  An
        explicit override for the built-in ``basis`` builder in calculator mode.
    run_options : dict, optional
        Extra keyword arguments forwarded to :meth:`run` on each calculator
        evaluation (e.g. ``{"log_expressivity": False}``); overrides the
        ``max_iterations`` / ``gradient_tolerance`` / ``output`` defaults above.
    """

    implemented_properties = ["energy", "free_energy"]

    _GRADIENTS = ("classical", "parameter-shift_rule")
    _OPTIMIZERS = ("COBYLA", "Nelder-Mead", "BFGS")

    def __init__(self,
                 hamiltonian=None,
                 pool="fermionic",
                 basis: str = "FAO",
                 num_particles=None,
                 n_spatial_orbitals=None,
                 optimizer: str | Optimizer = "COBYLA",
                 mapping: str = "jordan_wigner",
                 gradient: str = "classical",
                 device: str = "AER_simulator",
                 max_iterations: int = 50,
                 gradient_tolerance: float = 1e-3,
                 output: str | None = None,
                 profile: bool = True,
                 verbose: bool = True,
                 atomic_units: bool = False,
                 grid=None,
                 charge: int = 0,
                 n_electrons=None,
                 hamiltonian_builder=None,
                 run_options: dict | None = None, **calc_kwargs):
        Calculator.__init__(self, **calc_kwargs)

        self.mapping = mapping
        self.basis = basis
        self.profile = profile
        self.verbose = bool(verbose)
        self.optimizer = self._resolve_optimizer(optimizer)

        # Run defaults (also the defaults for the ASE-calculator evaluation).
        self.max_iterations = int(max_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        self.output = output

        # Validate the enumerated options up front.
        if gradient not in self._GRADIENTS:
            raise ValueError(
                f"unknown gradient {gradient!r}; use one of {self._GRADIENTS}")
        self.gradient = gradient
        self.device = normalize_device(device)     # raises on unknown device

        # Output-unit convention (see class docstring).
        self.atomic_units = bool(atomic_units)
        self.energy_units = "Ha" if atomic_units else "eV"
        self.length_units = "bohr" if atomic_units else "angstrom"

        self._pool_spec = pool
        self.grid = grid
        self.charge = int(charge)
        self.n_electrons = n_electrons
        self.hamiltonian_builder = hamiltonian_builder
        self.run_options = dict(run_options or {})

        # Seeded RNG for reproducible expressivity logging (output.txt).
        self._expr_rng = np.random.default_rng(0)

        # Configure eagerly when a Hamiltonian is given (direct mode); otherwise
        # defer to the first calculator evaluation (:meth:`calculate`).
        self._configured = False
        if hamiltonian is not None:
            self._configure(hamiltonian, num_particles, n_spatial_orbitals)

    # -- setup helpers ---------------------------------------------------- #

    def _resolve_optimizer(self, optimizer: str | Optimizer) -> Optimizer:
        """Normalize the ``optimizer`` argument to an :class:`Optimizer`.

        Accepts a pre-built :class:`Optimizer` (used as-is) or one of the method
        names in :attr:`_OPTIMIZERS` (``"COBYLA"``, ``"Nelder-Mead"``, ``"BFGS"``).
        """
        if isinstance(optimizer, Optimizer):
            return optimizer
        if isinstance(optimizer, str):
            if optimizer not in self._OPTIMIZERS:
                raise ValueError(
                    f"unknown optimizer {optimizer!r}; use one of "
                    f"{self._OPTIMIZERS} or an Optimizer instance")
            return Optimizer(method=optimizer, maxiter=2000)
        raise TypeError(
            "optimizer must be a method name or an Optimizer instance")

    def _configure(self, hamiltonian, num_particles, n_spatial_orbitals):
        """Resolve the pool and materialize the Hamiltonian / pool matrices."""
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
        self.n_qubits = self.pool.n_qubits
        self.num_particles = (tuple(num_particles) if num_particles is not None
                              else self.pool.num_particles)

        # Materialize the qubit Hamiltonian.
        qubit_h = self._as_pauli_sum(hamiltonian)
        if qubit_h.num_qubits != self.n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the pool "
                f"has {self.n_qubits}")
        self.hamiltonian = qubit_h
        h = qubit_h.to_matrix()
        self._h_matrix = 0.5 * (h + h.conj().T)      # Hermitize away rounding

        # Precompute pool-operator matrices and their eigendecompositions once.
        # For A anti-Hermitian, -iA is Hermitian: -iA = V diag(w) V^dag, so
        # exp(theta A) = V diag(exp(i theta w)) V^dag.  The unique positive
        # eigenvalue *differences* are the frequencies of E(theta), used by the
        # parameter-shift gradient.
        self._pool_ops = self.pool.operators()
        self._pool_matrices = [op.matrix() for op in self._pool_ops]
        self._pool_eig = []
        for a in self._pool_matrices:
            w, V = np.linalg.eigh(-1j * a)
            self._pool_eig.append((w, V, _unique_frequencies(w)))
        self._configured = True

    def _as_pauli_sum(self, hamiltonian) -> PauliSum:
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            return hamiltonian.map_to_qubits(self.mapping, n_modes=self.n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    # -- basis-driven Hamiltonian (calculator mode) ----------------------- #

    def _basis_hamiltonian_builder(self, atoms):
        """Build the RHF MO Hamiltonian from ``atoms`` using ``self.basis``.

        Elements and positions come from the ASE object; the basis functions are
        generated by :meth:`carcara.basis.BasisSet.build` for the chosen family
        (``FAO`` / ``STO-3G`` / ``6-31G(d)`` / ...).  Returns
        ``(hamiltonian, num_particles, n_spatial_orbitals)``.
        """
        from ..basis import BasisSet
        from ..core import MolecularIntegrals
        from ..integrals import Grid

        numbers = atoms.get_atomic_numbers()
        symbols = atoms.get_chemical_symbols()
        positions = np.asarray(atoms.get_positions(), dtype=float)

        bset = BasisSet.build(self.basis)
        basis, nuclei = [], []
        for Z, sym, pos in zip(numbers, symbols, positions):
            basis += bset.atom(sym, center=pos, units="angstrom")
            nuclei.append((float(Z), pos))

        grid = self.grid if self.grid is not None else self._auto_grid(positions)
        n_el = (int(self.n_electrons) if self.n_electrons is not None
                else int(sum(int(z) for z in numbers)) - self.charge)
        if n_el % 2 != 0:
            raise ValueError(
                f"the built-in {self.basis!r} builder assumes a closed shell; "
                f"got an odd electron count ({n_el}). Pass a hamiltonian_builder.")

        integrals = MolecularIntegrals(nuclei, basis, grid)
        hamiltonian = integrals.molecular_hamiltonian(mo_basis=True,
                                                      n_electrons=n_el)
        n_orbitals = len(basis)
        return hamiltonian, (n_el // 2, n_el // 2), n_orbitals

    @staticmethod
    def _auto_grid(positions, padding: float = 5.0, spacing: float = 0.2):
        """A cubic grid (Angstrom) enclosing the atoms with ``padding`` around."""
        from ..integrals import Grid

        positions = np.asarray(positions, dtype=float)
        center = positions.mean(axis=0)
        half = 0.5 * float(np.max(positions.max(axis=0) - positions.min(axis=0)))
        return Grid(center=center, box_size=half + padding, h=spacing)

    # -- ASE calculator interface ---------------------------------------- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """ASE hook: build the Hamiltonian from ``atoms`` and run ADAPT-VQE.

        Stores the ground-state energy (in **eV**, per ASE convention) in
        :attr:`results` and the full :class:`ADAPTVQEResult` in
        :attr:`adapt_result`.  The Hamiltonian is built from the current geometry
        with the chosen ``basis`` (or an explicit ``hamiltonian_builder``); a
        fixed Hamiltonian supplied at construction is reused as-is.
        """
        require_runnable(self.device)   # e.g. 'ibm-quantum' is not runnable yet
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms  # the Atoms copy stored by the base class

        builder = self.hamiltonian_builder
        if builder is None and not self._configured:
            builder = self._basis_hamiltonian_builder
        if builder is not None:
            hamiltonian, num_particles, n_spatial_orbitals = builder(atoms)
            self._configure(hamiltonian, num_particles, n_spatial_orbitals)

        result = self.run(geometry=atoms, **self.run_options)
        self.adapt_result = result

        # ASE always works in eV / Angstrom, regardless of the log-unit choice.
        energy_ev = float(from_hartree(result.optimal_energy, "eV"))
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

    # -- energy / gradient ------------------------------------------------ #

    def energy(self, psi: np.ndarray) -> float:
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

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
        """Pool screening gradients using the configured :attr:`gradient` method."""
        if self.gradient == "parameter-shift_rule":
            return self._parameter_shift_gradients(psi)
        return self._finite_difference_gradients(psi)      # "classical"

    def reference_energy(self) -> float:
        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping)
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

    def run(self, max_iterations: int | None = None,
            gradient_tol: float | None = None,
            initial_parameters=None, callback=None,
            output_file: str | None = None, geometry=None, cell=None,
            log_expressivity: bool = True,
            verbose: bool | None = None) -> ADAPTVQEResult:
        """Grow and optimize the ansatz until convergence.

        Every argument is optional: ``max_iterations``, ``gradient_tol``,
        ``output_file`` and ``verbose`` fall back to the instance's
        ``max_iterations`` / ``gradient_tolerance`` / ``output`` / ``verbose``
        constructor arguments when left as ``None``, so a configured
        :class:`ADAPTVQE` can simply be ``.run()``.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of operators to append (default: the instance's
            ``max_iterations``).
        gradient_tol : float, optional
            Stop when the largest pool gradient falls below this threshold
            (default: the instance's ``gradient_tolerance``).
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
        output_file : str, optional
            When given, write a structured runtime trace to this path following
            the ADAPT ``output.txt`` protocol
            (:class:`~carcara.utils.logging.AdaptOutputLogger`): the initial
            geometry and cell, the classical optimizer setup, and -- appended
            live at every iteration -- the pool operators as explicit Pauli
            strings, each operator's gradient magnitude, the selected operator,
            and the ansatz's expressivity score :math:`E`.
        geometry : ase.Atoms or (symbols, positions), optional
            Initial geometry for the ``output.txt`` metadata block.  An ASE
            ``Atoms`` object supplies symbols, positions and (if periodic) the
            cell; a ``(symbols, positions)`` pair supplies just the geometry.
        cell : (3, 3) array_like, optional
            Explicit unit-cell tensor for the metadata block (overrides any cell
            carried by an ``Atoms`` ``geometry``).
        log_expressivity : bool
            Compute and log the expressivity score each iteration when
            ``output_file`` is set (default ``True``).
        verbose : bool, optional
            Print the quantum-simulation trace to standard output -- the qubit
            Hamiltonian as Pauli strings, then each iteration's selected operator
            as Pauli strings.  Defaults to the instance's ``verbose``.
        """
        if not self._configured:
            raise RuntimeError(
                "ADAPTVQE has no Hamiltonian; construct it with one, or use it "
                "as an ASE calculator with a `hamiltonian_builder`")

        # Fall back to the instance-level constructor defaults.
        if max_iterations is None:
            max_iterations = self.max_iterations
        if gradient_tol is None:
            gradient_tol = self.gradient_tolerance
        if output_file is None:
            output_file = self.output
        if verbose is None:
            verbose = self.verbose

        ansatz = AdaptAnsatz(self.n_qubits, self.pool.occupied_orbitals,
                             self.mapping)
        params = (np.asarray(initial_parameters, dtype=float).ravel()
                  if initial_parameters is not None else np.zeros(0))
        ref_energy = self.energy(ansatz.reference_state())

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
                psi = ansatz.state(params) if ansatz.num_parameters else \
                    ansatz.reference_state()
                grads = self._gradients(psi)
                idx = int(np.argmax(np.abs(grads)))
                max_grad = float(abs(grads[idx]))
                if max_grad < gradient_tol:
                    converged = True
                    break

                op = self._pool_ops[idx]
                ansatz.append(op)
                selected.append(op.label)

                # Warm start: reuse previous optimum, new parameter set to 0.
                x0 = np.concatenate([params, [0.0]])
                result = self.optimizer.minimize(
                    lambda t: self.energy(ansatz.state(t)), x0)
                params = np.asarray(result.x, dtype=float)
                energy = float(result.fun)
                total_evals += result.nfev

                if verbose:
                    self._print_iteration(len(iterations) + 1, op, max_grad,
                                          energy, e_unit)

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
            psi = ansatz.state(params) if ansatz.num_parameters else \
                ansatz.reference_state()
            max_grad = float(np.max(np.abs(self._gradients(psi))))

        result = ADAPTVQEResult(
            optimal_energy=energy,
            optimal_parameters=params,
            reference_energy=ref_energy,
            converged=converged,
            final_max_gradient=max_grad,
            operators=selected,
            iterations=iterations,
            num_evaluations=total_evals,
            metrics=metrics)

        if verbose:
            self._print_summary(result, e_unit)
        return result

    # -- standard-output trace ------------------------------------------- #

    def _print_header(self, ref_energy: float, e_unit: str) -> None:
        """Print the run banner and the qubit Hamiltonian as Pauli strings."""
        rule = "=" * 70
        print(rule)
        print(f"ADAPT-VQE  |  mapping: {self.mapping}  |  {self.n_qubits} qubits "
              f"|  device: {self.device}")
        print(f"pool: {self.pool.__class__.__name__}  |  "
              f"optimizer: {self.optimizer.method}  |  gradient: {self.gradient}")
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

    def _print_summary(self, result: ADAPTVQEResult, e_unit: str) -> None:
        """Print the closing summary line."""
        rule = "=" * 70
        print(rule)
        status = "converged" if result.converged else "not converged"
        print(f"ADAPT-VQE finished ({status}): "
              f"E = {self._to_energy_units(result.optimal_energy):+.8f} {e_unit}, "
              f"{result.num_operators} operators, "
              f"final |grad| = {result.final_max_gradient:.6e}")
        print(rule)
