# -*- coding: utf-8 -*-
# file: algorithms/vqe.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Variational Quantum Eigensolver (VQE).

:class:`VQE` minimizes the Rayleigh quotient
:math:`E(\vec\theta) = \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle` over the
parameters of a variational ansatz, returning the (approximate) ground-state
energy and optimal parameters.

This reference implementation is an **exact state-vector simulator**: the qubit
Hamiltonian is materialized as a dense matrix and the ansatz produces the exact
:math:`2^N` state vector, so the energy is the noiseless expectation value.

**This class is the internal layer.**  It is reached only through the
calculator, ``Mandacaru(method="vqe", ...)``, which forwards every option here:

* **direct mode** -- ``Mandacaru(method="vqe", hamiltonian=..., ansatz=...)``
  and :meth:`~mandacaru.algorithms.Mandacaru.run`;
* **ASE calculator mode** -- ``atoms.calc = Mandacaru(method="vqe",
  basis=...)``: ``atoms.get_total_energy()`` builds the Hamiltonian from the
  geometry, builds a default UCCSD ansatz and runs, returning eV.

The ``optimizer`` may be named by string, a ``verbose`` run prints the run
configuration -- the qubit Hamiltonian by its **term count** only; its Pauli
expansion goes to a file with ``verbose_hamiltonian=`` -- and a timing / memory /
cores summary, and the run
returns a :class:`VQEResult` shaped like
:class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQEResult`.

Beyond the ground state, :meth:`VQE.energy_levels` returns the low-lying
**molecular energy levels** (ground + excited states) by variational quantum
deflation -- see :mod:`mandacaru.algorithms.deflation`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core.mapping import Fermion
from ..optimizers.optim import DEFAULT_OPTIMIZER, Optimizer
from ..units import convert_energy
from ..utils.profiling import Timings
from .ansatz_spec import (HVA_REFUSED_OPTIONS, ansatz_name, build_ansatz,
                          reference_problem, resolve_ansatz)
from .base import VariationalDriver
from .deflation import DeflationMixin, deflation_penalty


@dataclass
class VQEResult:
    """Result of a :class:`VQE` run.

    Shaped like :class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQEResult`: it carries
    the optimal energy/parameters, the reference energy, the evaluation count and
    the full cost history, exposes the same convenience views
    (:attr:`num_parameters`, :attr:`energy_history`), and records the timing /
    cores / memory profile of the run.  Every energy (``optimal_energy``,
    ``reference_energy``, ``history``) is in :attr:`energy_unit` -- **eV** by
    default, Hartree when the driver was built with ``atomic_units=True``;
    :meth:`in_units` converts.
    """

    optimal_energy: float                 # minimized energy <psi|H|psi>
    optimal_parameters: np.ndarray        # optimal ansatz parameters
    reference_energy: float               # energy of the ansatz reference state
    num_evaluations: int = 0              # cost-function evaluations
    #: Parameter updates the classical optimizer made -- its *steps*, which are
    #: not its evaluations (a gradient method spends several of the latter on
    #: each of the former).  ``None`` when the method reports neither a count
    #: nor a per-iteration callback.  The same figure
    #: :class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQEResult` reports, so the
    #: two results can be compared on classical cost.
    optimizer_steps: int | None = None
    history: list[float] = field(default_factory=list)   # cost per evaluation
    success: bool = True
    timings: dict | None = None           # per-stage wall time / cores / memory
    integration_profile: dict | None = None   # real-space integration profile
    energy_unit: str = "eV"               # unit of every energy above

    @property
    def num_parameters(self) -> int:
        return len(self.optimal_parameters)

    @property
    def energy_history(self) -> list[float]:
        return list(self.history)

    @property
    def correlation_energy(self) -> float:
        """Energy lowered relative to the reference (``E - E_ref``)."""
        return self.optimal_energy - self.reference_energy

    def in_units(self, units: str = "eV") -> float:
        """The optimal energy converted to ``units`` (``"eV"`` or ``"Ha"``)."""
        return float(convert_energy(self.optimal_energy, self.energy_unit, units))

    def __repr__(self) -> str:
        return (f"VQEResult(energy={self.optimal_energy:.6f} {self.energy_unit}, "
                f"n_params={self.num_parameters}, "
                f"nfev={self.num_evaluations}, success={self.success})")


class VQE(DeflationMixin, VariationalDriver):
    """Variational Quantum Eigensolver on an exact state-vector backend.

    Also an ASE calculator: see the module docstring for the two usage modes.

    Parameters
    ----------
    hamiltonian : PauliSum or Fermion, optional
        The qubit Hamiltonian, or a fermionic Hamiltonian mapped with the
        ansatz's mapping.  Omit in calculator mode.
    ansatz : str, dict or object, optional
        The circuit to optimize (:mod:`mandacaru.algorithms.ansatz_spec`): a
        template name -- ``"uccsd"`` (the default) or ``"hva"``, the
        Hamiltonian variational ansatz -- or ``{"name": ..., <options>}``, e.g.
        ``{"name": "hva", "layers": 3, "evolution": "trotter"}``; the circuit
        is built from the Hamiltonian once it exists.  A pre-built ansatz
        object exposing ``num_parameters``, ``n_qubits``, ``state(theta)`` and
        ``reference_state()`` is used as it is.
    num_particles, n_spatial_orbitals : optional
        ``(n_alpha, n_beta)`` and the spatial-orbital count of a direct
        ``hamiltonian``, from which a named ansatz is built;
        ``n_spatial_orbitals`` defaults to half the modes of a ``Fermion``.
    optimizer : str, dict or Optimizer
        A method name -- one of ``"SPSA"``, ``"COBYLA"``, ``"Nelder-Mead"``,
        ``"SLSQP"`` (default), ``"L-BFGS"``, ``"BFGS"`` -- taking the
        library's budget and tolerance; ``{"method": ..., "maxiter": ...,
        "tol": ...}`` setting them without importing anything; or a pre-built
        :class:`~mandacaru.optimizers.optim.Optimizer`, which is what the dict
        builds.
    verbose : bool
        Print the run configuration (the qubit Hamiltonian as a term count) and
        a timing / resources summary to standard output (default ``True``).
    basis, mapping, device, grid, h, kpts, charge, n_electrons, hamiltonian_builder, ansatz_builder, run_options :
        Calculator-mode options mirroring
        :class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE`: ``basis`` (a name or a
        ``{"name": ..., <options>}`` dict, including the periodic ``"PW"`` plane-wave
        family ``{"name": "PW", "energy_cutoff": 300}``) and grid resolution ``h``
        build the Hamiltonian from the ASE geometry (the grid is generated from
        ``atoms.cell`` unless ``grid`` is given), ``ansatz_builder`` -- a
        callable ``(n_spatial_orbitals, num_particles, mapping) -> ansatz`` --
        replaces the named templates.  ``kpts`` is a Monkhorst-Pack mesh
        resolved with ASE
        (Gamma-point only is runnable; see ``ADAPTVQE``).
    frozen_core : bool, str or int
        Frozen-core approximation (default ``False``, no freezing).  ``True`` /
        ``"auto"`` freezes the chemical noble-gas core; an integer freezes that
        many lowest molecular orbitals.  The frozen orbitals are removed from the
        active space (see :class:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE`).
    frozen_orbitals : sequence of int, optional
        Explicit list of (doubly occupied) spatial MO indices to freeze; overrides
        ``frozen_core`` and names exactly which electrons are core vs active.
    save_hamiltonian : bool or str
        Write the qubit Hamiltonian (Pauli strings) to disk once built (default
        ``False``); ``True`` uses ``"hamiltonian"`` plus the extension of
        ``hamiltonian_format``.
    hamiltonian_format : {"parquet", "json"}
        Format written by ``save_hamiltonian`` (default ``"parquet"``);
        ``"json"`` needs no Parquet engine.  Loading auto-detects the format.
    load_hamiltonian : str, optional
        Read the qubit Hamiltonian from such a file, **skipping the molecular
        integrals and the fermion-to-qubit mapping**.  The file's recorded
        ``num_particles`` / ``n_spatial_orbitals`` are enough to rebuild the
        default UCCSD ansatz, so no geometry is needed.
    backend_provider : str
        SDK used to construct / execute the ansatz circuits -- ``"qiskit"``
        (default), ``"braket"`` or ``"cirq"``; see
        :mod:`mandacaru.backends.providers`.
    execute_circuits : bool, optional
        Prepare states by executing the compiled circuit on the provider's local
        simulator (default: ``True`` for ``"braket"``/``"cirq"``, ``False`` for
        ``"qiskit"``).  A circuit realizes the **Trotter** UCCSD product, so the
        default ansatz is built with ``trotter=True`` when this is on.
    quenching : bool
        Dynamic parametrization (default ``True``).  ``True`` optimizes all
        parameters jointly.  ``False`` sweeps them one at a time -- parameter
        ``k`` alone, with ``0..k-1`` frozen at their optimized values -- the
        fixed-ansatz analog of ADAPT-VQE's frozen-parameter growth.
    """

    _default_sparse = False
    citation_method = "vqe"
    solver_label = "VQE"

    def __init__(self, hamiltonian=None, ansatz=None,
                 optimizer: str | Optimizer = DEFAULT_OPTIMIZER,
                 verbose: bool = True,
                 *, num_particles=None, n_spatial_orbitals=None,
                 ansatz_builder=None, **driver_kwargs):
        spec = resolve_ansatz(ansatz)
        if ansatz_builder is not None and ansatz is not None:
            raise ValueError("pass either ansatz= or ansatz_builder=, not both")
        if spec is not None and spec.name == "hva":
            used = [name for name in HVA_REFUSED_OPTIONS
                    if name != "ansatz_builder" and driver_kwargs.get(name)]
            if used:
                raise ValueError(
                    "the HVA needs its fermionic group decomposition and a "
                    "local optimization; it does not take "
                    + ", ".join(f"{name}=" for name in used))
            # Its exact group exponentials act on sparse matrices; a dense
            # 2^n Hamiltonian is only the default for the UCCSD's sake.
            if driver_kwargs.get("sparse") is None:
                driver_kwargs["sparse"] = "auto"
        # Every other keyword is a VariationalDriver option, forwarded
        # untouched so its name and default live in one place.
        super().__init__(optimizer=optimizer, verbose=verbose, **driver_kwargs)
        # After the base resolved it: Braket and Cirq execute circuits by
        # default, without an explicit execute_circuits=True.
        if spec is not None and spec.evolution == "exact" \
                and self.execute_circuits:
            raise ValueError(
                "exact HVA group evolution has no generic circuit; set "
                "evolution='trotter' for circuit execution")
        self.ansatz_builder = ansatz_builder
        #: The named template, or ``None`` for a pre-built ansatz object.
        self._ansatz_spec = spec
        self._preset_ansatz = ansatz if spec is None else None
        if spec is not None:
            hamiltonian, num_particles, n_spatial_orbitals = reference_problem(
                spec, hamiltonian, num_particles, n_spatial_orbitals)

        # A cached Hamiltonian carries num_particles / n_spatial_orbitals, so it
        # is a complete problem specification: the default UCCSD ansatz can be
        # built from it with no geometry, no integrals and no mapping step.
        if hamiltonian is None and self.load_hamiltonian is not None \
                and self.dry_run:
            # Never even loaded in a dry run: the estimate reads the file's
            # header (`read_hamiltonian_header`), not its Pauli table.
            self._check_cache_header()
        elif hamiltonian is None and self.load_hamiltonian is not None:
            hamiltonian, num_particles, n_orbitals = \
                self._load_hamiltonian_record()
            if self._preset_ansatz is None and (num_particles is None
                                                or n_orbitals is None):
                raise ValueError(
                    f"{self.load_hamiltonian!r} does not record num_particles / "
                    "n_spatial_orbitals, so the default UCCSD ansatz cannot be "
                    "rebuilt from it; pass an explicit `ansatz`")
            self._configure(hamiltonian, num_particles, n_orbitals)
            self._built_from_hamiltonian = True
            self._maybe_write_references()
        # Direct mode: a Hamiltonian with a pre-built ansatz, or with the
        # particle counts a named template is built from.
        elif hamiltonian is not None and (self._preset_ansatz is not None
                                          or num_particles is not None):
            if self._preset_ansatz is not None:
                if num_particles is not None or n_spatial_orbitals is not None:
                    raise ValueError(
                        "a pre-built ansatz carries its own particle and "
                        "orbital counts; omit num_particles= and "
                        "n_spatial_orbitals=")
                num_particles = getattr(self._preset_ansatz,
                                        "num_particles", None)
                n_spatial_orbitals = getattr(self._preset_ansatz,
                                             "n_spatial_orbitals", None)
            elif n_spatial_orbitals is None:
                if not isinstance(hamiltonian, Fermion):
                    raise ValueError(
                        "a qubit Hamiltonian needs n_spatial_orbitals= to "
                        "build a named ansatz")
                n_spatial_orbitals = hamiltonian.n_modes() // 2
            if self.dry_run:
                self._dry_run_problem = (hamiltonian, num_particles,
                                         n_spatial_orbitals)
            else:
                self._configure(hamiltonian, num_particles, n_spatial_orbitals)
                self._maybe_write_references()
                self._built_from_hamiltonian = True
        elif num_particles is not None or n_spatial_orbitals is not None:
            raise ValueError("num_particles= and n_spatial_orbitals= describe "
                             "a direct hamiltonian=; pass one")

    # -- setup ------------------------------------------------------------ #

    def _configure(self, hamiltonian, num_particles, n_orbitals) -> None:
        """Adopt / build the ansatz for ``hamiltonian`` and materialize it.

        A pre-built ansatz is used as it is; otherwise ``ansatz_builder`` or
        the named template
        (:func:`~mandacaru.algorithms.ansatz_spec.build_ansatz`) builds one from ``num_particles`` / ``n_orbitals`` -- and, for the
        HVA, from the fermionic Hamiltonian itself.
        """
        if self._preset_ansatz is not None:
            ansatz = self._preset_ansatz
        elif self.ansatz_builder is not None:
            ansatz = self.ansatz_builder(n_orbitals, num_particles,
                                         self.mapping)
        else:
            ansatz = build_ansatz(
                self._ansatz_spec, hamiltonian=hamiltonian,
                num_particles=num_particles, n_spatial_orbitals=n_orbitals,
                mapping=self.mapping, taper=self.taper,
                provider=self.ansatz_provider(), shots=self.shots)
        self.ansatz = ansatz
        self.mapping = getattr(ansatz, "mapping", self.mapping)
        # Expose the occupation / active-space size like ADAPTVQE does, so the
        # calculator's `num_particles` and the dry run read them uniformly.
        particles = getattr(ansatz, "num_particles", num_particles)
        self.num_particles = (None if particles is None
                              else tuple(int(v) for v in particles))
        self.n_spatial_orbitals = getattr(ansatz, "n_spatial_orbitals",
                                          n_orbitals)
        # An ansatz that tapered its own register (the HVA, whose Z2 sector
        # must keep every fixed group) brings the tapered Hamiltonian with it.
        self._taper_info = getattr(ansatz, "taper_info", None)
        if self._taper_info is not None:
            qubit_h = self._taper_info.hamiltonian
        else:
            if self.taper:
                warnings.warn(
                    f"taper=True found no Z2 symmetry the "
                    f"{type(ansatz).__name__} keeps; the run continues "
                    f"untapered", RuntimeWarning, stacklevel=2)
            qubit_h = self._as_pauli_sum(hamiltonian, ansatz.n_qubits,
                                         particles)
        self._materialize_hamiltonian(qubit_h, ansatz.n_qubits)
        self._maybe_save_hamiltonian(self.num_particles,
                                     self.n_spatial_orbitals)
        self._maybe_dump_hamiltonian(self.num_particles,
                                     self.n_spatial_orbitals)
        self._configured = True

    def _default_parameters(self) -> np.ndarray:
        """Where an optimization starts without ``initial_parameters``.

        The ansatz's own choice when it has one -- the HVA starts every angle
        at its ``seed_angle``, off the stationary all-zero point -- else the
        reference state, all zeros.
        """
        start = getattr(self.ansatz, "initial_parameters", None)
        if start is not None:
            return np.asarray(start(), dtype=float).ravel()
        return np.zeros(self.ansatz.num_parameters)

    # -- energy ----------------------------------------------------------- #

    def energy_at(self, theta) -> float:
        """Expectation value ``<psi(theta)| H |psi(theta)>`` for parameters ``theta``.

        The parameter-space cost function.  (The base class's ``energy(psi)`` is the
        state-vector expectation value; ``energy_at`` prepares the state first --
        or, with ``shots > 0``, measures ``<H>`` on hardware instead.)
        """
        return self.ansatz_energy(self.ansatz, theta)

    def reference_energy(self) -> float:
        """Energy of the ansatz reference state (all parameters zero)."""
        return self.energy(self.ansatz.reference_state())

    def run(self, initial_parameters=None) -> VQEResult:
        """Optimize the parameters and return the ground-state estimate.

        The optimizer and the ``verbose`` flag come from the constructor; the
        only argument is ``initial_parameters`` (which the constructor does not
        carry), defaulting to all-zero (the reference state).
        """
        if self.dry_run:
            return self._dry_run_estimate()
        if not self._configured:
            raise RuntimeError(
                "VQE has no Hamiltonian/ansatz; construct it with both, or use it "
                "as an ASE calculator with a `basis`")
        self._check_kpts()

        n = self.ansatz.num_parameters
        x0 = (self._default_parameters() if initial_parameters is None
              else np.asarray(initial_parameters, dtype=float).ravel())
        if x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {x0.size}")

        # Resume: start from the checkpointed parameters of this same ansatz.
        self._check_checkpointable(self.ansatz)
        resumed = self._load_resume(self.ansatz)
        if resumed is not None:
            if initial_parameters is not None:
                raise ValueError("pass either initial_parameters or resume=, "
                                 "not both")
            self._check_resumed_generators(resumed)
            x0 = np.asarray(resumed.parameters, dtype=float)

        timings, run_t0 = self._make_timings()
        ref_energy = self.reference_energy()
        if self.verbose:
            self._show_banner()
            self._print_header(ref_energy)
            if resumed is not None:
                print(f"resumed from {self.resume_path!r}")

        # Checkpoint the best point seen every `checkpoint_every` evaluations;
        # the optimizer's own iterate may be a trial step, the best is not.
        best = {"x": x0.copy(), "fun": np.inf}

        def track(x, value, nfev):
            if value < best["fun"]:
                best["x"], best["fun"] = np.array(x, dtype=float), float(value)
            if self.checkpoint_path is not None \
                    and nfev % self.checkpoint_every == 0:
                self._record_checkpoint(
                    self.ansatz, best["x"], best["fun"],
                    {"complete": False, "num_evaluations": int(nfev),
                     "reference_energy": float(ref_energy)})

        with timings.time("parameter optimization"):
            result = self._optimize_all(self.energy_at, x0, callback=track)

        self._record_checkpoint(
            self.ansatz, result.x, float(result.fun),
            {"complete": True, "converged": bool(result.success),
             "num_evaluations": int(result.nfev),
             "reference_energy": float(ref_energy)})

        self._finalize_timings(timings, run_t0)

        # The single Hartree -> output-unit boundary of the run.
        vqe_result = VQEResult(
            optimal_energy=self._to_energy_units(result.fun),
            optimal_parameters=result.x,
            reference_energy=self._to_energy_units(ref_energy),
            num_evaluations=result.nfev,
            optimizer_steps=result.nit,
            history=[self._to_energy_units(e) for e in result.history],
            success=result.success,
            timings=timings.as_dict(),
            integration_profile=self._integration_profile,
            energy_unit=self._energy_unit_label())

        if self.verbose:
            self._print_summary(vqe_result, timings)
        return vqe_result

    def _checkpoint_record(self, ansatz, parameters, energy_ha, status,
                           labels=None, kinds=None):
        """The shared record, with an ansatz's own parameter encoding.

        An ansatz whose logical angles are not its generators' angles (the
        HVA: one angle per group, many rotations per group) stores the
        generator angles as ``parameters`` -- what any consumer prepares the
        state from -- and its logical angles and identity beside them.
        """
        encode = getattr(ansatz, "checkpoint_parameters", None)
        if encode is None:
            return super()._checkpoint_record(ansatz, parameters, energy_ha,
                                              status, labels=labels,
                                              kinds=kinds)
        from ..circuits.hva import CHECKPOINT_KEY, LOGICAL_PARAMETERS_KEY

        logical = np.asarray(parameters, dtype=float).ravel()
        expanded = encode(logical)
        if labels is None:
            labels = ansatz.checkpoint_labels()
        if kinds is None:
            kinds = [ansatz_name(ansatz) or "custom"] * expanded.size
        record = super()._checkpoint_record(ansatz, expanded, energy_ha,
                                            status, labels=labels, kinds=kinds)
        record.status[LOGICAL_PARAMETERS_KEY] = logical.tolist()
        record.metadata[CHECKPOINT_KEY] = ansatz.checkpoint_metadata(
            self.hamiltonian)
        return record

    def _load_resume(self, ansatz):
        """The resumed record, decoded to the ansatz's logical angles."""
        record = super()._load_resume(ansatz)
        decode = getattr(ansatz, "parameters_from_checkpoint", None)
        if record is None or decode is None:
            return record
        if record.num_particles != self.num_particles \
                or record.n_spatial_orbitals != self.n_spatial_orbitals:
            raise ValueError(
                f"cannot resume from {self.resume_path!r}: its particle or "
                f"orbital counts differ from this run's")
        record.parameters = decode(record, self.hamiltonian)
        return record

    def _check_resumed_generators(self, record) -> None:
        """A VQE checkpoint resumes only into the ansatz that wrote it."""
        mine = [g.simplify().terms for g in self.ansatz.pauli_generators]
        theirs = [g.simplify().terms for g in record.generators]
        same = len(mine) == len(theirs) and all(
            a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-10 for k in a)
            for a, b in zip(mine, theirs))
        if not same:
            raise ValueError(
                f"cannot resume from {self.resume_path!r}: its "
                f"{len(theirs)} generators are not this ansatz's "
                f"{len(mine)} (a fixed ansatz resumes only from its own "
                "checkpoint; an ADAPT checkpoint is resumed by ADAPTVQE)")

    # -- excited states / energy levels (DeflationMixin hook) ------------- #

    def _deflated_ground(self, states, beta, *, state_index: int = 0,
                         initial_parameters=None, restarts: int = 1,
                         seed: int = 0):
        r"""Lowest state of the fixed ansatz orthogonal to ``states`` (deflation).

        Minimizes ``<psi(theta)|H|psi(theta)> + beta * sum_j |<psi_j|psi>|^2`` over
        the fixed ansatz, from the reference (ground) or seeded random restarts
        (excited states); the reported energy is the bare expectation value.
        Called per level by :meth:`~mandacaru.algorithms.deflation.DeflationMixin.energy_levels`.
        """
        n = self.ansatz.num_parameters
        base_x0 = (self._default_parameters() if initial_parameters is None
                   else np.asarray(initial_parameters, dtype=float).ravel())
        if base_x0.size != n:
            raise ValueError(f"expected {n} initial parameters, got {base_x0.size}")
        rng = np.random.default_rng(seed + int(state_index))

        def cost(theta):
            psi = self.ansatz.state(theta)
            return self.energy(psi) + deflation_penalty(psi, states, beta)

        # First attempt from the warm start (ground) / reference; further restarts
        # (and every excited-state restart) from seeded random points.
        best = None
        total_evals = 0
        for r in range(max(1, int(restarts))):
            x0 = base_x0 if (state_index == 0 and r == 0) else \
                rng.uniform(-np.pi, np.pi, size=n)
            result = self._optimize_all(cost, x0)
            total_evals += result.nfev
            if best is None or result.fun < best.fun:
                best = result
        psi = self.ansatz.state(best.x)
        return self.energy(psi), psi, total_evals, None

    # -- standard-output trace ------------------------------------------- #

    def _print_header(self, ref_energy: float) -> None:
        """Print the run configuration banner.

        The Hamiltonian's Pauli-string expansion is not printed (see
        :meth:`~mandacaru.algorithms.adapt_vqe.ADAPTVQE._print_header`); only its
        term count is.
        """
        rule = "=" * 70
        print(rule)
        print(f"{self.solver_label}  |  mapping: {self.mapping}  |  "
              f"{self.n_qubits} qubits  |  "
              f"optimizer: {self.optimizer.method}  |  device: {self.device}")
        print(f"ansatz: {type(self.ansatz).__name__}  |  "
              f"parameters: {self.ansatz.num_parameters}  |  "
              f"k-points: {self._kpts_label()}")
        print(f"spin-polarized: {self.spin}  |  "
              f"initial state: {self.initial_state}")
        print(f"backend provider: {self.backend_provider}  |  circuit execution: "
              f"{self.execute_circuits}  |  quenching: {self.quenching}")
        for detail in self._ansatz_details():
            print(detail)
        print(rule)
        n_terms = len(self.hamiltonian.simplify().terms)
        print(f"Qubit Hamiltonian: {n_terms} Pauli terms")
        print(f"Reference (all-zero) energy = "
              f"{self._to_energy_units(ref_energy):+.8f} "
              f"{self._energy_unit_label()}")
        print(rule)

    def _ansatz_details(self) -> tuple[str, ...]:
        """The ansatz's own configuration lines inside the shared VQE header."""
        describe = getattr(self.ansatz, "describe", None)
        lines = tuple(describe()) if describe is not None else ()
        if getattr(self, "_taper_info", None) is not None:
            lines += (self._taper_info.summary(),)
        return lines

    def _citation_config(self) -> dict:
        """The shared configuration plus the template that ran."""
        config = super()._citation_config()
        ansatz = getattr(self, "ansatz", None)
        config["ansatz"] = (ansatz_name(ansatz) if ansatz is not None
                            else getattr(self._ansatz_spec, "name", None))
        return config

    def _print_summary(self, result: VQEResult,
                       timings: Timings | None = None) -> None:
        """Print the closing summary: result line plus timings / resources."""
        rule = "=" * 70
        print(rule)
        status = "converged" if result.success else "did not converge"
        print(f"{self.solver_label} finished ({status}): "
              f"E = {result.optimal_energy:+.8f} {result.energy_unit}, "
              f"{result.num_parameters} parameters, {result.num_evaluations} "
              f"evaluations")
        if timings is not None:
            print(timings.format_report())
        print(rule)
