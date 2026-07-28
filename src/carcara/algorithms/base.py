# -*- coding: utf-8 -*-
# file: algorithms/base.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Shared base for the variational state-vector drivers.

:class:`VariationalDriver` factors out everything the concrete algorithms
(:class:`~carcara.algorithms.vqe.VQE`,
:class:`~carcara.algorithms.adapt_vqe.ADAPTVQE`, and their subspace variants)
have in common, so a new method only writes its own optimization loop:

* **problem setup** -- the ASE-calculator surface (``basis`` / ``grid`` / ``h`` /
  ``kpts`` / ``spin`` / ``frozen_core`` / ...) and the geometry-to-Hamiltonian
  builder;
* **the state-vector backend** -- materializing the qubit Hamiltonian as a dense
  or sparse matrix, the canonical expectation value ``energy(psi)``, and the
  Gamma-point guard;
* **run scaffolding** -- the ASE ``calculate`` hook, wall-clock / resource
  timings, and the start-up banner.

Subclasses implement two hooks: :meth:`_configure` (build the ansatz / pool for a
given Hamiltonian and materialize it) and :meth:`run` (the optimization).  The
``energy`` contract is a **state vector in** -- ``energy(psi)`` -- uniformly
across every driver.
"""

from __future__ import annotations

import os
from time import perf_counter as _perf

import numpy as np

from ase.calculators.calculator import Calculator, all_changes

from ..backends.hardware import (device_arn, device_provider, is_aws_device,
                                 normalize_device, require_runnable,
                                 requires_shots)
from ..backends.providers import build_provider, normalize_provider
from ..core.mapping import Fermion, PauliSum
from ..core.serialization import (DEFAULT_FORMAT, load_hamiltonian,
                                  resolve_format, resolve_save_path,
                                  save_hamiltonian)
from ..optimizers.optim import NAMED_OPTIMIZERS, OptimizeResult, resolve_optimizer
from ..units import from_hartree
from ._hamiltonian_from_atoms import monkhorst_pack_kpts, resolve_initial_state


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


class VariationalDriver(Calculator):
    """Base ASE calculator for the variational state-vector eigensolvers.

    Not used directly -- concrete drivers subclass it and implement
    :meth:`_configure` and :meth:`run`.  The constructor accepts the problem-setup
    surface shared by every driver; algorithm-specific arguments (ansatz, pool,
    stopping controls, ...) are added by the subclass and passed through
    ``**shared`` to :meth:`__init__`.

    Parameters
    ----------
    save_hamiltonian : bool or str
        Serialize the qubit Hamiltonian (as Pauli strings) to disk once it has
        been built (default ``False``).  ``True`` writes ``"hamiltonian"`` with
        the extension of ``hamiltonian_format``; a path writes there.  See
        :mod:`carcara.core.serialization`.
    load_hamiltonian : str, optional
        Path of a Hamiltonian file written by ``save_hamiltonian``.  When given,
        the driver loads the qubit Hamiltonian from disk and **skips the molecular
        integrals (one- and two-body) and the fermion-to-qubit transformation
        entirely** -- the file also carries ``num_particles`` /
        ``n_spatial_orbitals``, so the pool / ansatz are rebuilt without a
        geometry.  The format is detected from the file, so this accepts either.
    hamiltonian_format : {"parquet", "json"}
        Format ``save_hamiltonian`` writes (default ``"parquet"``: compact,
        compressed and columnar).  ``"json"`` is plain text and needs no Parquet
        engine, which sidesteps the Qiskit/pyarrow interaction documented in
        :mod:`carcara.core.serialization`.  Loading ignores this and
        auto-detects instead.
    backend_provider : str
        Quantum SDK used to construct (and, when executing, run) the ansatz
        circuits -- ``"qiskit"`` (default), ``"braket"`` (amazon-braket-sdk) or
        ``"cirq"``.  See :mod:`carcara.backends.providers`.
    execute_circuits : bool, optional
        Evaluate the ansatz by *executing* the compiled circuit on the provider's
        local state-vector simulator instead of the internal NumPy state-vector
        backend.  Defaults to ``True`` for ``"braket"`` / ``"cirq"`` (naming them
        is a request to use them) and ``False`` for ``"qiskit"``, which keeps the
        fast default numerics; the results agree to machine precision either way.
    quenching : bool
        Dynamic parametrization (default ``True``).  ``True`` re-optimizes **all**
        variational parameters at every iteration.  ``False`` optimizes only the
        most recently added parameter, freezing all previous ones at their
        already-optimized values.
    pseudopotentials : bool or dict
        Use norm-conserving pseudopotentials (default ``False``).  ``True`` loads
        the bundled Troullier-Martins library from ``pseudos/``; a dict passes
        options (currently ``{"directory": ...}`` to point at another library).

        This replaces the all-electron problem with a **valence-only** one: the
        core electrons are removed, the basis becomes the smooth pseudo-atomic
        orbitals, and the singular :math:`-Z/r` is replaced by a bounded local
        channel plus Kleinman-Bylander projectors.  It is the cure for the
        heavy-atom grid artifacts documented in
        :mod:`carcara.algorithms.forces`, and it shrinks the qubit count as a
        side effect.  Incompatible with ``frozen_core`` (which it subsumes).
    """

    implemented_properties = ["energy", "free_energy"]
    _OPTIMIZERS = NAMED_OPTIMIZERS

    #: Default ``sparse`` policy (``False`` dense; ``"auto"`` for adaptive drivers).
    _default_sparse = False

    def __init__(self, *, optimizer="COBYLA", mapping: str = "jordan_wigner",
                 basis="FAO", device: str = "AER_simulator", grid=None,
                 h: float = 0.20, kpts=None, spin: bool = False,
                 initial_state: str | None = "hartree-fock", charge: int = 0,
                 n_electrons=None, frozen_core=False, frozen_orbitals=None,
                 pseudopotentials=False,
                 hamiltonian_builder=None, run_options: dict | None = None,
                 verbose: bool = True, sparse=None,
                 save_hamiltonian: bool | str = False,
                 load_hamiltonian: str | None = None,
                 hamiltonian_format: str = DEFAULT_FORMAT,
                 backend_provider: str | None = None,
                 execute_circuits: bool | None = None,
                 backend_options: dict | None = None, shots: int = 0,
                 quenching: bool = True, **calc_kwargs):
        Calculator.__init__(self, **calc_kwargs)

        self.verbose = bool(verbose)
        self.optimizer = resolve_optimizer(optimizer, allowed=self._OPTIMIZERS)
        self.mapping = mapping
        self.basis = basis
        self.device = normalize_device(device)      # raises on unknown device
        self.grid = grid
        self.h = float(h)
        self.spin = bool(spin)
        self.frozen_core = frozen_core
        self.frozen_orbitals = frozen_orbitals
        # Norm-conserving pseudopotentials: replace the core + the -Z/r
        # singularity with a smooth valence-only problem.
        self.pseudopotentials = pseudopotentials
        self.initial_state = resolve_initial_state(initial_state)
        # Monkhorst-Pack mesh (ASE); the engine is Gamma-point (molecular), so a
        # denser mesh is stored on ``kpoints`` but rejected at run time.
        self.kpts, self.kpts_gamma, self.kpoints = monkhorst_pack_kpts(kpts)
        self.charge = int(charge)
        self.n_electrons = n_electrons
        self.hamiltonian_builder = hamiltonian_builder
        self.run_options = dict(run_options or {})
        self.sparse = self._default_sparse if sparse is None else sparse

        # Hamiltonian disk cache: `load_hamiltonian` short-circuits the integral
        # engine and the fermion-to-qubit mapping; `save_hamiltonian` dumps the
        # qubit Hamiltonian (Pauli strings) once it has been built.
        self.load_hamiltonian = (None if load_hamiltonian is None
                                 else str(load_hamiltonian))
        self.save_hamiltonian = save_hamiltonian
        # The format applies to *saving*; loading detects it from the file.
        self.hamiltonian_format = resolve_format(hamiltonian_format)
        self._save_path = resolve_save_path(save_hamiltonian,
                                            self.hamiltonian_format)

        # Circuit-construction / execution SDK.  Naming an Amazon Braket device
        # implies the braket provider, so `device="braket-sv1"` alone is enough.
        if backend_provider is None:
            backend_provider = device_provider(self.device) or "qiskit"
        self.backend_provider = normalize_provider(backend_provider)
        # `execute_circuits` defaults to True for the non-Qiskit providers
        # (naming them is a request to use them); the Qiskit default keeps the
        # fast NumPy state-vector numerics unless execution is asked for.
        self.execute_circuits = (self.backend_provider != "qiskit"
                                 if execute_circuits is None
                                 else bool(execute_circuits))
        self.backend_options = dict(backend_options or {})

        # Measurement shots.  Real QPUs never return a state vector, so they
        # require shots > 0 and the energy is estimated from measured
        # qubit-wise-commuting groups (see carcara.backends.measurement).
        self.shots = int(shots)
        if requires_shots(self.device) and self.shots <= 0:
            raise ValueError(
                f"device {self.device!r} is real quantum hardware, which cannot "
                "return a state vector: pass shots > 0 (e.g. shots=8192) so the "
                "energy is estimated from measurements.")
        if self.shots and self.backend_provider != "braket":
            raise NotImplementedError(
                f"shot-based execution is implemented for the 'braket' provider "
                f"only, not {self.backend_provider!r}; use "
                "backend_provider='braket' (optionally with an AWS device).")
        if self.shots:
            self.execute_circuits = True

        # Dynamic parametrization: True re-optimizes every parameter each
        # iteration; False freezes the previous ones (see `_optimize_grown`).
        self.quenching = bool(quenching)

        self._integration_profile = None
        self._gradient_context = None
        self._configured = False
        #: Result of the most recent run (set by :meth:`run` / the ASE hook).
        self.result = None
        # True when a Hamiltonian was supplied at construction (direct mode); the
        # ASE hook then never rebuilds it.  In calculator mode it stays False and
        # the Hamiltonian is (re)built from the geometry on each ``calculate``.
        self._built_from_hamiltonian = False

    # -- k-points --------------------------------------------------------- #

    def _check_kpts(self) -> None:
        """Reject a non-Gamma Monkhorst-Pack mesh (the engine is Gamma-point)."""
        if len(self.kpoints) > 1:
            raise NotImplementedError(
                f"a {self.kpts[0]}x{self.kpts[1]}x{self.kpts[2]} Monkhorst-Pack "
                f"mesh ({len(self.kpoints)} k-points) is not yet supported: the "
                "real-space engine solves a Gamma-point (molecular) problem.  Use "
                "kpts=(1, 1, 1) or kpts=None.")

    def _kpts_label(self) -> str:
        n1, n2, n3 = self.kpts
        centered = ", Gamma-centered" if self.kpts_gamma else ""
        if len(self.kpoints) == 1:
            return f"Gamma ({n1}x{n2}x{n3} Monkhorst-Pack)"
        return (f"{len(self.kpoints)} k-points ({n1}x{n2}x{n3} "
                f"Monkhorst-Pack{centered})")

    # -- Hamiltonian materialization -------------------------------------- #

    @staticmethod
    def _resolve_sparse(sparse, n_qubits: int) -> bool:
        """Resolve the ``sparse`` spec to a bool.

        ``"auto"`` enables the sparse backend for ``n_qubits >= 10`` (where a dense
        pool would need tens of GB); ``True`` / ``False`` force it on / off.
        """
        if isinstance(sparse, str):
            if sparse.strip().lower() == "auto":
                return int(n_qubits) >= 10
            raise ValueError(
                f"unknown sparse spec {sparse!r}; use True, False or 'auto'")
        return bool(sparse)

    def _as_pauli_sum(self, hamiltonian, n_qubits: int) -> PauliSum:
        """Coerce a ``PauliSum`` / ``Fermion`` Hamiltonian to a ``PauliSum``."""
        if isinstance(hamiltonian, PauliSum):
            return hamiltonian
        if isinstance(hamiltonian, Fermion):
            return hamiltonian.map_to_qubits(self.mapping, n_modes=n_qubits)
        raise TypeError("hamiltonian must be a PauliSum or Fermion")

    def _materialize_hamiltonian(self, qubit_h: PauliSum, n_qubits: int) -> bool:
        """Store the (Hermitized) dense or sparse Hamiltonian matrix.

        Sets :attr:`hamiltonian`, :attr:`n_qubits`, :attr:`_sparse` and
        :attr:`_h_matrix`; returns the resolved sparse flag.  A sparse matrix is
        used when :meth:`_resolve_sparse` selects it (large active spaces).
        """
        if qubit_h.num_qubits != n_qubits:
            raise ValueError(
                f"Hamiltonian acts on {qubit_h.num_qubits} qubits but the ansatz "
                f"/ pool has {n_qubits}")
        self.hamiltonian = qubit_h
        self.n_qubits = int(n_qubits)
        self._sparse = self._resolve_sparse(self.sparse, n_qubits)
        if self._sparse:
            hs = qubit_h.to_sparse_matrix()
            self._h_matrix = 0.5 * (hs + hs.conj().T)
        else:
            h = qubit_h.to_matrix()
            self._h_matrix = 0.5 * (h + h.conj().T)     # Hermitize rounding noise
        return self._sparse

    def energy(self, psi: np.ndarray) -> float:
        r"""Expectation value ``<psi| H |psi>`` (real) for a state vector ``psi``."""
        return float(np.real(np.vdot(psi, self._h_matrix @ psi)))

    def ansatz_energy(self, ansatz, theta) -> float:
        r"""Energy of ``ansatz`` at parameters ``theta`` on the configured backend.

        This is the single place the *hardware* path diverges from the simulator
        path.  With ``shots = 0`` the state vector is prepared (internally or by
        executing a circuit) and contracted with the Hamiltonian.  With
        ``shots > 0`` -- mandatory on a real QPU, which never exposes amplitudes
        -- the provider measures ``<H>`` from qubit-wise-commuting groups instead
        (:mod:`carcara.backends.measurement`).
        """
        if not self.shots:
            return self.energy(ansatz.state(theta))
        provider = self.circuit_provider()
        return provider.energy(ansatz.n_qubits, ansatz.reference_qubits(),
                               ansatz.pauli_generators, theta, self.hamiltonian)

    # -- circuit provider ------------------------------------------------- #

    def circuit_provider(self):
        """The :class:`~carcara.backends.providers.CircuitProvider`, or ``None``.

        ``None`` means the driver evaluates the ansatz with the internal
        (NumPy / SciPy-sparse) state-vector backend; a provider means every state
        preparation is compiled to a circuit and executed on that SDK's
        simulator or, for an Amazon Braket device, on the AWS service.  Both
        produce the same unitary -- see :mod:`carcara.backends.providers`.

        The provider is configured from :attr:`device` (Braket devices carry an
        ARN), :attr:`shots` and :attr:`backend_options`.
        """
        if not self.execute_circuits:
            return None
        return build_provider(self.backend_provider, **self._provider_options())

    def _provider_options(self) -> dict:
        """Constructor options for the configured circuit provider."""
        options = dict(self.backend_options)
        if self.backend_provider != "braket":
            return options
        options.setdefault("shots", self.shots)
        if "device" not in options:
            arn = device_arn(self.device)
            if arn is not None:
                options["device"] = arn         # run through the AWS service
            elif is_aws_device(self.device) or self.device == "braket-local":
                options["device"] = "braket_sv"
        return options

    # -- Hamiltonian disk cache ------------------------------------------- #

    def _load_hamiltonian_record(self):
        """Read the cached qubit Hamiltonian named by ``load_hamiltonian``.

        Returns ``(PauliSum, num_particles, n_spatial_orbitals)``.  This is the
        whole point of the cache: neither the one-/two-body integrals nor the
        fermion-to-qubit mapping is touched, and the driver's ``mapping`` is
        adopted from the file so the ansatz / pool stay consistent with it.
        """
        record = load_hamiltonian(self.load_hamiltonian)
        self.mapping = record.mapping
        self._loaded_record = record
        return (record.hamiltonian, record.num_particles,
                record.n_spatial_orbitals)

    def _maybe_save_hamiltonian(self, num_particles=None,
                                n_spatial_orbitals=None) -> str | None:
        """Dump the materialized qubit Hamiltonian when ``save_hamiltonian`` is set.

        A no-op when saving is off, or when the Hamiltonian was just read from the
        very file it would be written to.  Returns the path written, if any.
        """
        if self._save_path is None:
            return None
        if (self.load_hamiltonian is not None
                and os.path.abspath(self.load_hamiltonian)
                == os.path.abspath(self._save_path)):
            return None
        return save_hamiltonian(
            self._save_path, self.hamiltonian, mapping=self.mapping,
            num_particles=num_particles, n_spatial_orbitals=n_spatial_orbitals,
            format=self.hamiltonian_format,
            metadata={"driver": type(self).__name__,
                      "basis": self.basis if isinstance(self.basis, str)
                      else dict(self.basis),
                      "frozen_core": self.frozen_core,
                      "n_qubits": int(self.n_qubits)})

    # -- optimization policy (quenching) ---------------------------------- #

    def _optimize_grown(self, cost, previous_parameters) -> OptimizeResult:
        """Optimize after appending one new parameter, honoring :attr:`quenching`.

        ``quenching=True`` (default) hands **all** parameters to the classical
        optimizer, warm-started from the previous optimum with the new angle at
        zero -- standard ADAPT-VQE.  ``quenching=False`` freezes the previously
        optimized parameters and varies only the newly added one, a cheaper
        one-dimensional line search per growth step that trades variational
        freedom for cost-function evaluations.

        ``cost`` takes the **full** parameter vector in both cases; the returned
        :class:`~carcara.optimizers.optim.OptimizeResult` also carries the full
        vector, so callers need no branching.
        """
        previous = np.asarray(previous_parameters, dtype=float).ravel()
        x0 = np.concatenate([previous, [0.0]])
        if self.quenching:
            return self.optimizer.minimize(cost, x0)

        def last_only(tail):
            return cost(np.concatenate(
                [previous, np.asarray(tail, dtype=float).ravel()]))

        result = self.optimizer.minimize(last_only, np.zeros(1))
        full = np.concatenate([previous,
                               np.asarray(result.x, dtype=float).ravel()])
        return OptimizeResult(x=full, fun=result.fun, nfev=result.nfev,
                              history=result.history, success=result.success,
                              message=result.message)

    def _optimize_all(self, cost, x0) -> OptimizeResult:
        """Optimize a fixed-size parameter vector, honoring :attr:`quenching`.

        ``quenching=True`` (default) is a single joint minimization over every
        parameter.  ``quenching=False`` sweeps the parameters one at a time in
        order -- parameter ``k`` is optimized alone with ``0..k-1`` frozen at their
        already-optimized values and ``k+1..`` held at their starting values --
        the fixed-ansatz analogue of freezing previous growth steps.
        """
        x0 = np.asarray(x0, dtype=float).ravel()
        if self.quenching or x0.size <= 1:
            return self.optimizer.minimize(cost, x0)

        params = x0.copy()
        history: list[float] = []
        nfev = 0
        value = float(cost(params))
        success = True
        for k in range(params.size):
            def single(t, _k=k, _p=params):
                trial = _p.copy()
                trial[_k] = float(np.asarray(t, dtype=float).ravel()[0])
                return cost(trial)

            step = self.optimizer.minimize(single, np.atleast_1d(params[k]))
            params[k] = float(np.asarray(step.x, dtype=float).ravel()[0])
            history.extend(step.history)
            nfev += step.nfev
            value = float(step.fun)
            success = success and step.success
        return OptimizeResult(x=params, fun=value, nfev=nfev, history=history,
                              success=success,
                              message="sequential (quenching=False) sweep")

    # -- geometry -> Hamiltonian (calculator mode) ------------------------ #

    def _build_hamiltonian(self, atoms):
        """Build ``(hamiltonian, num_particles, n_spatial_orbitals)`` from ``atoms``.

        Reads the cached Pauli-string Hamiltonian when ``load_hamiltonian`` is set
        (skipping the integrals *and* the mapping entirely); otherwise uses an
        explicit ``hamiltonian_builder`` if given, or the built-in ``basis`` engine
        (stashing its integration profile for the run summary).
        """
        if self.load_hamiltonian is not None:
            return self._load_hamiltonian_record()
        if self.hamiltonian_builder is not None:
            hamiltonian, num_particles, n_orbitals = \
                self.hamiltonian_builder(atoms)
            return hamiltonian, num_particles, n_orbitals
        from ._hamiltonian_from_atoms import build_basis_hamiltonian
        (hamiltonian, num_particles, n_orbitals, profile,
         context) = build_basis_hamiltonian(
            atoms, self.basis, self.grid, self.h, self.charge, self.n_electrons,
            spin=self.spin, frozen_core=self.frozen_core,
            frozen_orbitals=self.frozen_orbitals,
            pseudopotentials=self.pseudopotentials)
        self._integration_profile = profile
        # Kept for the nuclear gradient: the integral engine that produced this
        # Hamiltonian, and which atom each basis function belongs to.
        self._gradient_context = context
        return hamiltonian, num_particles, n_orbitals

    # -- ASE calculator hook ---------------------------------------------- #

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """ASE hook: build (if needed), run, and store the ground-state energy.

        In calculator mode the Hamiltonian is (re)built from the current geometry
        each call, so energies track the geometry; in direct mode the Hamiltonian
        supplied at construction is reused.  The run result is stored on
        :attr:`result` and the ground-state energy (eV) in ``results``.
        """
        require_runnable(self.device)     # e.g. 'ibm-quantum' is not runnable yet
        self._wall_start = _perf()        # wall clock spans integration + run
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms                # the Atoms copy stored by the base class

        if not self._built_from_hamiltonian:
            hamiltonian, num_particles, n_orbitals = self._build_hamiltonian(atoms)
            self._configure(hamiltonian, num_particles, n_orbitals)

        result = self.run(**self._run_kwargs(atoms))
        self.result = result

        energy_ev = float(from_hartree(result.optimal_energy, "eV"))
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev

    def _run_kwargs(self, atoms) -> dict:
        """Keyword arguments forwarded to :meth:`run` from the ASE hook."""
        return dict(self.run_options)

    # -- run scaffolding -------------------------------------------------- #

    def _make_timings(self):
        """Fresh :class:`~carcara.utils.profiling.Timings` and the run start time.

        Pops the ``_wall_start`` seeded by :meth:`calculate` (so the wall clock
        spans the integration too) or starts the clock now in direct mode.
        """
        from ..integrals import _backend
        from ..utils.profiling import Timings
        timings = Timings(n_cores=_backend.num_threads(),
                          backend="C (OpenMP)" if _backend.HAS_C_BACKEND
                          else "NumPy")
        run_t0 = self.__dict__.pop("_wall_start", None)
        if run_t0 is None:
            run_t0 = _perf()
        return timings, run_t0

    def _finalize_timings(self, timings, run_t0) -> None:
        """Fold the calculator-mode integration stage in and set the wall time."""
        if self._integration_profile is not None:
            for name, secs in self._integration_profile.get("stages_s", {}).items():
                timings.add(f"integration: {name}", secs)
        timings.wall_time = _perf() - run_t0

    def _show_banner(self) -> None:
        """Write the start-up banner to stdout (verbose runs only)."""
        if self.verbose:
            from ..utils import banner
            banner.show()

    # -- subclass hooks --------------------------------------------------- #

    def _configure(self, hamiltonian, num_particles, n_orbitals) -> None:
        """Build the ansatz / pool for ``hamiltonian`` and materialize it."""
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """Run the optimization and return the driver's result dataclass."""
        raise NotImplementedError
