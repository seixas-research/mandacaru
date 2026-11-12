# -*- coding: utf-8 -*-
# file: backends/providers.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Circuit-construction / execution providers: Qiskit, Braket and Cirq.

Mandacaru's ansätze are products of exponentials of **anti-Hermitian generators**,
:math:`|\psi(\vec\theta)\rangle = \prod_k e^{\theta_k A_k}\,|\mathrm{HF}\rangle`.
Each generator is a qubit :class:`~mandacaru.core.mapping.PauliSum` whose terms are
:math:`A = \sum_j i\,c_j P_j` with **real** :math:`c_j` and mutually commuting
Pauli strings :math:`P_j` (a property of the fermionic / qubit excitation
generators), so the exponential factorizes *exactly*:

.. math::

    e^{\theta A} = \prod_j e^{i\,\theta c_j P_j},

and each factor is the textbook Pauli-rotation circuit -- a basis change to the
``Z`` axis, a CNOT ladder accumulating the parity onto one qubit, an
:math:`R_z(-2\theta c_j)`, then the ladder and basis change undone.  Because that
decomposition is exact (no Trotter error), the three providers build *the same*
unitary out of the same gate set (``X``, ``H``, ``S``, ``S†``, ``CNOT``,
``R_z``) and must agree on the resulting state vector to machine precision --
which is what the equivalence tests check.

Providers
---------
``"qiskit"``
    IBM Qiskit (default).  Note Qiskit is **little-endian** (its qubit 0 is the
    least significant bit of the state-vector index) while Mandacaru -- like Braket
    and Cirq -- puts qubit 0 in the *most* significant position, so the Qiskit
    provider lays Mandacaru qubit ``k`` on Qiskit wire ``n-1-k``.  The gate counts
    are unaffected (relabeling is an isomorphism).  With ``shots > 0`` the
    energy comes from the **Estimator** primitive -- locally, on a Qiskit
    Runtime *fake backend* (``device="fake_kingston"``: transpiled to that
    processor, run locally) or on **real IBM Quantum hardware** through the
    Qiskit Runtime ``Estimator`` (``device="ibm_kingston"``, or
    ``"ibm_kingston,ibm_fez,ibm_marrakesh"`` for the least busy of those).
``"braket"``
    Amazon Braket SDK, executed on the local state-vector simulator
    (``LocalSimulator("braket_sv")``).
``"cirq"``
    Google Cirq, executed on ``cirq.Simulator`` with an explicit big-endian
    ``qubit_order``.

Each provider exposes the same three operations -- :meth:`CircuitProvider.build`,
:meth:`CircuitProvider.statevector` and :meth:`CircuitProvider.profile` -- so a
driver only stores a provider name and never branches on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import _native


#: Provider names accepted by :func:`build_provider` / the driver
#: ``backend_provider`` argument.
BACKEND_PROVIDERS = ("qiskit", "braket", "cirq")

_PROVIDER_ALIASES = {
    "qiskit": "qiskit", "ibm": "qiskit", "ibmq": "qiskit",
    "braket": "braket", "amazon-braket": "braket", "amazon_braket": "braket",
    "aws": "braket",
    "cirq": "cirq", "google": "cirq",
}


def normalize_provider(name: str) -> str:
    """Resolve a ``backend_provider`` name/alias to its canonical form."""
    key = str(name).strip().lower()
    canon = _PROVIDER_ALIASES.get(key)
    if canon is None:
        raise ValueError(
            f"unknown backend_provider {name!r}; use one of {BACKEND_PROVIDERS}")
    return canon


# --------------------------------------------------------------------------- #
# Generator -> Pauli-rotation decomposition (provider independent).
# --------------------------------------------------------------------------- #

def run_local_estimator(estimator, pubs, precision: float):
    """Run a *local* Qiskit Estimator synchronously; the one place that
    touches the primitive's private ``_run``.

    The public ``run()`` hands the work to a one-off worker thread
    (``PrimitiveJob``), and Qiskit circuit data allocated on another thread
    can make a later garbage collection spin forever in its Rust allocator
    (reproduced on macOS / Python 3.14 / qiskit 2.5).  A Qiskit release
    without ``_run`` is refused here, with the reason, rather than silently
    routed back through the threaded path.
    """
    from qiskit.primitives.containers.estimator_pub import EstimatorPub

    private = getattr(estimator, "_run", None)
    if not callable(private):
        raise RuntimeError(
            f"{type(estimator).__name__} has no synchronous _run(); this "
            f"Qiskit release needs the local-Estimator adapter in "
            f"mandacaru.backends.providers updated (its threaded run() can "
            f"hang garbage collection)")
    return private([EstimatorPub.coerce(pub, precision) for pub in pubs])


# --------------------------------------------------------------------------- #
# Provider interface.
# --------------------------------------------------------------------------- #

class CircuitProvider(ABC):
    """Builds, executes and profiles a Mandacaru ansatz on one quantum SDK.

    Implementations translate the provider-independent gate stream emitted by
    :meth:`_emit` into their own circuit type; everything above that -- the
    Pauli-rotation decomposition, the CNOT ladders, the endianness convention --
    is shared here, so all three providers realize the same unitary.
    """

    #: Canonical provider name (``"qiskit"`` / ``"braket"`` / ``"cirq"``).
    name: str = ""

    # -- provider-independent gate stream --------------------------------- #

    def _context(self):
        """The native context that compiles this provider's circuits (made on first use)."""
        context = getattr(self, "_native_context", None)
        if context is None:
            from .._native import Context
            context = self._native_context = Context()
        return context

    def _emit(self, n_qubits: int, occupied: list[int], generators, thetas):
        """Yield ``(gate, *qubits[, angle])`` for the full ansatz circuit.

        Gates are ``"x"``, ``"h"``, ``"s"``, ``"sdg"``, ``"cx"`` and ``"rz"`` --
        the common denominator of the three SDKs.  The circuit is the reference
        determinant preparation followed by one Pauli-rotation block per term of
        each generator, in append order. The gate stream is compiled in Rust
        (``execution::compile``), which refuses a generator that is not
        anti-Hermitian or whose terms anticommute (a product of rotations would
        then be a Trotter approximation, a different state), and angles that do
        not match the generators.
        """
        terms = [[(p, complex(c).real, complex(c).imag) for p, c in g.terms.items()]
                 for g in generators]
        for gate, first, second, angle in self._context().compile_circuit(
                n_qubits, list(occupied), terms,
                np.asarray(thetas, dtype=float).ravel().tolist()):
            yield ((gate, first, second) if gate == "cx" else
                   (gate, first, angle) if gate == "rz" else (gate, first))

    def build(self, n_qubits: int, occupied, generators, thetas):
        """Build the SDK-native circuit for the ansatz at parameters ``thetas``."""

    @abstractmethod
    def statevector(self, n_qubits: int, occupied, generators, thetas
                    ) -> np.ndarray:
        """Execute the circuit and return the state vector in Mandacaru's ordering.

        Mandacaru's convention is qubit 0 = most significant bit of the amplitude
        index (the leftmost Kronecker factor), matching
        :meth:`~mandacaru.core.mapping.PauliSum.to_matrix`.
        """

    @abstractmethod
    def profile(self, n_qubits: int, occupied, generators) -> dict:
        """Compile the ansatz and return ``{cnot_count, depth, num_1q, total}``.

        Parameters are irrelevant to the structural cost, so a unit angle is used
        for every generator.
        """

    # -- shot-based energy (the real-hardware path) ----------------------- #

    #: True when the provider can estimate ``<H>`` from measurement shots, i.e.
    #: can target real QPUs rather than only a state-vector simulator.
    supports_shots: bool = False
    #: What this provider can do beyond preparing a state; checked before an
    #: optimization spends anything (:func:`require_capability`).
    #: ``"expectation_values"`` many ``<P>`` of one state (the energy's strings,
    #: what RDMs and forces need); ``"expectation_batch"`` those of several
    #: states in one submission; ``"batch"`` several problems per job.
    capabilities: tuple = ()

    # -- run ledger and budget ------------------------------------------- #

    #: Run-wide budget (see
    #: :func:`~mandacaru.backends.measurement.resolve_run_budget`).
    run_budget = None

    @property
    def is_hardware(self) -> bool:
        """True when a submission reaches a paid, remote processor."""
        return False

    def _ledger(self) -> dict:
        ledger = self.__dict__.get("_run_ledger")
        if ledger is None:
            ledger = self.__dict__["_run_ledger"] = {"jobs": 0, "shots": 0,
                                                     "handles": []}
        return ledger

    def _core_options(self):
        context = getattr(self, "_native_context", None)
        return ({} if context is None else
                {"context": context})

    @property
    def run_jobs(self) -> list:
        """Every job handle this provider has submitted, oldest first (the
        cumulative ledger; :attr:`jobs` is the current submission)."""
        return list(self._ledger()["handles"])

    def run_usage(self) -> dict:
        """``{"jobs": ..., "shots": ...}`` submitted so far in this run."""
        ledger = self._ledger()
        return {"jobs": ledger["jobs"], "shots": ledger["shots"]}

    def _charge(self, shots: int, jobs: int = 1) -> None:
        """Account for a submission about to be queued, or refuse it.

        Called *before* the job is queued, so a refusal costs nothing.
        """
        from .measurement import RunBudgetExceeded, resolve_run_budget

        from .._native import budget_check

        ledger = self._ledger()
        budget = resolve_run_budget(self.run_budget, self.is_hardware) or {}
        (jobs_after, shots_after), over = budget_check(
            (ledger["jobs"], ledger["shots"]), (int(jobs), int(shots)),
            job_limit=budget.get("jobs"), shot_limit=budget.get("total_shots"))
        if over:
            detail = "; ".join(f"{key} {value:,} > {limit:,}"
                               for key, value, limit in over)
            sized = {key: value for key, value, _limit in over}
            raise RunBudgetExceeded(
                f"{self!r}: this submission would exceed the run budget "
                f"({detail}) after {ledger['jobs']:,} jobs and "
                f"{ledger['shots']:,} shots.  Pass "
                f"backend_options={{\"run_budget\": {sized!r}}} sized for "
                f"this run if you mean it, or rehearse on a fake backend "
                f"first.")
        ledger["jobs"] = jobs_after
        ledger["shots"] = shots_after

    def _record_job(self, job) -> None:
        if job is not None:
            self._ledger()["handles"].append(job)

    # -- reconfiguration ------------------------------------------------- #

    def _constructor_options(self) -> dict:
        """The constructor arguments that rebuild this provider."""
        return {}

    def with_options(self, **overrides) -> "CircuitProvider":
        """A new provider with ``overrides`` applied to its constructor
        arguments.

        Setting attributes on a live provider does not reach what it has
        already built -- a resolved backend, a cached Estimator -- so a retry
        with other options gets a fresh object.  An unknown name is refused
        rather than set as an attribute nothing reads.  The run ledger
        carries over: a retry belongs to the same run and budget.
        """
        options = self._constructor_options()
        unknown = sorted(set(overrides) - set(options))
        if unknown:
            raise ValueError(
                f"unknown option(s) {unknown} for {type(self).__name__}; it "
                f"takes {sorted(options)}")
        fresh = type(self)(**{**options, **overrides})
        fresh.__dict__["_run_ledger"] = self._ledger()
        if hasattr(self, "_native_context"):
            fresh._native_context = self._native_context
        return fresh

    @staticmethod
    def _measurement_rotation(basis: str):
        """Gate stream rotating each qubit into the ``basis`` eigenbasis.

        Applied *after* the ansatz and *before* measurement, so a ``Z``-basis
        readout of the rotated register samples the requested Pauli basis:
        ``X`` needs ``H``, ``Y`` needs ``S†`` then ``H``, ``Z``/``I`` need
        nothing.
        """
        for k, letter in enumerate(basis):
            if letter == "X":
                yield ("h", k)
            elif letter == "Y":
                yield ("sdg", k)
                yield ("h", k)


# --------------------------------------------------------------------------- #
# Qiskit.
# --------------------------------------------------------------------------- #

class QiskitProvider(CircuitProvider):
    """Qiskit circuits: exact state vectors, or energies from the ``Estimator``.

    With ``shots > 0`` the energy is the expectation value returned by the
    Estimator primitive -- ``BackendEstimatorV2`` on noiseless Aer (state-vector method)
    locally, which samples the shots and reports their standard errors, the same primitive
    on a Qiskit Runtime *fake backend*, and the Qiskit
    Runtime ``Estimator`` on **IBM Quantum hardware** -- for the Hamiltonian as
    a ``SparsePauliOp`` observable, transpiled to the target (``transpile(qc,
    backend=backend)`` + ``observable.apply_layout``), exactly as an IBM
    notebook does it by hand.

    Parameters
    ----------
    device : str
        ``"statevector"`` (default; ``"AER_simulator"`` accepted) -- the local
        estimator, exact for ``shots = 0`` and sampled to the matching
        precision otherwise.  A ``fake_*`` name (``"fake_kingston"``) -- that
        processor's fake backend, run locally.  ``"ibm-quantum"`` -- the
        least-busy operational QPU of your account; ``"ibm_kingston"`` -- that
        processor; ``"ibm_kingston,ibm_fez,ibm_marrakesh"`` -- the least busy
        of those.
    shots : int
        Shots per circuit; the estimator precision is ``1/sqrt(shots)``.
        ``0`` (default) is the exact local expectation value; a QPU and
        ``"AER_mps"`` require ``shots > 0``.
    ``"AER_mps"`` -- Aer's matrix-product-state simulator, run locally, for circuits too
        wide for a state vector (a few-parameter ansatz on 50 qubits).  It samples, so it needs
        ``shots > 0``, cannot return a state vector, and takes no ``physical_qubits``,
        ``enable_fractional_gates`` or ``estimator_options``.
    max_bond_dimension : int, optional
        ``"AER_mps"`` only: the largest bond dimension the simulator keeps.  Its memory grows
        as the square of it, and a circuit that needs more is truncated -- the approximation.
        ``None`` keeps Aer's default (no cap).  Any other device refuses it.
    instance, token, channel : str, optional
        Passed to :class:`qiskit_ibm_runtime.QiskitRuntimeService`; with none
        given the account saved by ``QiskitRuntimeService.save_account`` is
        used.
    optimization_level : int
        Transpiler level for the target processor (default ``3``).
    estimator_options : dict, optional
        Options of the Runtime ``Estimator`` (resilience, twirling, ...).
    physical_qubits : sequence of int, optional
        Pin the circuit to these physical qubits (Mandacaru qubit ``k`` on
        ``physical_qubits[k]``) instead of letting the transpiler choose.
        Worth doing on hardware: a qubit whose readout has drifted since the
        last calibration ruins an energy while its recorded error still looks
        fine.
    """

    name = "qiskit"
    supports_shots = True

    LOCAL_DEVICES = ("statevector", "aer_simulator", "aer", "simulator", "local")
    LEAST_BUSY = ("ibm-quantum", "ibm", "ibmq", "ibm_quantum")
    FAKE_PREFIX = "fake_"
    MPS_DEVICES = ("aer_mps", "aer-mps", "mps")

    def __init__(self, device: str = "statevector", shots: int = 0,
                 instance: str | None = None, token: str | None = None,
                 channel: str | None = None, optimization_level: int = 3,
                 estimator_options: dict | None = None,
                 physical_qubits=None, seed: int | None = None,
                 max_bases_per_job: int | None = None,
                 enable_fractional_gates: bool = False,
                 run_budget=None, max_bond_dimension: int | None = None):
        self.device_spec = str(device).strip()
        self.physical_qubits = (None if physical_qubits is None
                                else [int(q) for q in physical_qubits])
        self.shots = int(shots)
        self.instance = instance
        self.token = token
        self.channel = channel
        self.optimization_level = int(optimization_level)
        #: Ask Runtime for the processor's **fractional** instruction set, whose
        #: entangler is a continuously parameterized ``RZZ(theta)`` rather than a
        #: fixed CZ.  Every exponentiated Pauli pair an excitation generator
        #: decomposes into *is* such a rotation, so the ansatz needs roughly half
        #: as many entanglers, which is the quantity that sets a run's fidelity.
        #: Requires a processor that offers them (Heron and later); Qiskit spells
        #: it ``use_fractional_gates``, which is what this forwards.
        self.enable_fractional_gates = bool(enable_fractional_gates)
        # A dict is copied; an ``EstimatorOptions`` object is kept as given
        # (Runtime takes either).
        self.estimator_options = (
            dict(estimator_options) if isinstance(estimator_options, dict)
            else estimator_options)
        #: Seed of the local Estimator's sampling, so a shot-based run is
        #: reproducible.  ``None`` samples freshly, which is what a real
        #: measurement does; a seed is for tests and for comparing two runs
        #: whose difference should not be shot noise.  Hardware and the fake
        #: backends ignore it -- their randomness is not ours to fix; the
        #: ``"AER_mps"`` simulator seeds its sampling with it.
        self.seed = None if seed is None else int(seed)
        #: Largest number of qubit-wise-commuting measurement bases in one
        #: submission (``None`` = no limit).  One PUB carrying ~10^5
        #: observables is what exhausted the Runtime program's memory; with a
        #: limit the observables are split into several bounded jobs that
        #: share one transpiled circuit, and a late failure costs only the
        #: chunk it happened in.
        self.max_bases_per_job = (None if max_bases_per_job is None
                                  else int(max_bases_per_job))
        #: Run-wide limit on jobs and shots (on by default for an ``ibm_*``
        #: device; see :func:`~mandacaru.backends.measurement.resolve_run_budget`).
        self.run_budget = run_budget
        #: Bond-dimension cap of the matrix-product-state simulator (``None``: Aer's default).
        self.max_bond_dimension = (None if max_bond_dimension is None
                                   else self._bond_dimension(max_bond_dimension))
        if self.is_mps_device:
            self._refuse_for_mps()
        elif self.max_bond_dimension is not None:
            raise ValueError(
                f"max_bond_dimension is an option of the 'AER_mps' device; "
                f"{self.device_spec!r} has no bond dimension")
        self._backend = None
        self._estimator = None
        #: The Runtime job of the last hardware submission (``None`` locally).
        self.last_job = None
        #: Every job of the current (last) submission, oldest first;
        #: :attr:`run_jobs` is every job of the run.
        self.jobs: list = []
        #: The ``PrimitiveResult`` of the last Estimator submission.
        self.last_result = None
        #: Chunks of an ``expectation_values`` submission that completed,
        #: keyed by what was submitted, so a retry resumes only the rest.
        self._completed_chunks: dict = {}
        #: Compiled parameterized circuits, one per (register, reference,
        #: generator sequence): an optimization transpiles its ansatz once
        #: and binds new angles per evaluation.
        self._templates: dict = {}

    def __repr__(self) -> str:
        return f"QiskitProvider(device={self.device_spec!r}, shots={self.shots})"

    @staticmethod
    def _bond_dimension(value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(
                f"max_bond_dimension must be a positive integer, got {value!r}")
        return value

    def _refuse_for_mps(self) -> None:
        """What the sampled matrix-product-state simulator cannot do, refused by name."""
        if self.shots <= 0:
            raise ValueError(
                "the 'AER_mps' device samples its circuits, so it requires shots > 0 (a "
                "matrix-product state has no state vector to read an exact energy from)")
        for name, given in (("physical_qubits", self.physical_qubits is not None),
                            ("enable_fractional_gates", self.enable_fractional_gates),
                            ("estimator_options", self.estimator_options is not None)):
            if given:
                raise ValueError(
                    f"{name} describes a processor; the 'AER_mps' simulator has none")

    capabilities = ("expectation_values", "expectation_batch", "batch")

    def _constructor_options(self) -> dict:
        return {"device": self.device_spec, "shots": self.shots,
                "instance": self.instance, "token": self.token,
                "channel": self.channel,
                "optimization_level": self.optimization_level,
                "estimator_options": self.estimator_options,
                "physical_qubits": self.physical_qubits, "seed": self.seed,
                "max_bases_per_job": self.max_bases_per_job,
                "enable_fractional_gates": self.enable_fractional_gates,
                "run_budget": self.run_budget,
                "max_bond_dimension": self.max_bond_dimension}

    @property
    def is_hardware(self) -> bool:
        return self.is_ibm_device

    # -- device resolution ------------------------------------------------ #

    @property
    def is_local(self) -> bool:
        return self.device_spec.lower() in self.LOCAL_DEVICES

    @property
    def is_fake_device(self) -> bool:
        return self.device_spec.lower().startswith(self.FAKE_PREFIX)

    @property
    def is_mps_device(self) -> bool:
        """True for the local matrix-product-state simulator (``"AER_mps"``)."""
        return self.device_spec.lower() in self.MPS_DEVICES

    @property
    def is_ibm_device(self) -> bool:
        """True when this provider submits to IBM Quantum hardware."""
        return not (self.is_local or self.is_fake_device or self.is_mps_device)

    @property
    def precision(self) -> float:
        """Estimator target precision: ``1/sqrt(shots)`` (``0`` = exact)."""
        from .._native import estimator_precision
        return estimator_precision(max(int(self.shots), 0))

    def _service(self):
        from qiskit_ibm_runtime import QiskitRuntimeService
        kwargs = {}
        if self.channel is not None:
            kwargs["channel"] = self.channel
        if self.token is not None:
            kwargs["token"] = self.token
        if self.instance is not None:
            kwargs["instance"] = self.instance
        return QiskitRuntimeService(**kwargs)

    def backend(self):
        """The resolved backend (cached); ``None`` for the local estimator."""
        if self.is_local:
            return None
        if self._backend is None:
            if self.is_fake_device:
                self._backend = self._fake_backend(self.device_spec)
            elif self.is_mps_device:
                self._backend = self._mps_backend()
            else:
                self._backend = self._ibm_backend(self._service())
        return self._backend

    def _mps_backend(self):
        """Aer's matrix-product-state simulator, with the bond-dimension cap and the seed."""
        from qiskit_aer import AerSimulator

        options = {"method": "matrix_product_state"}
        if self.max_bond_dimension is not None:
            options["matrix_product_state_max_bond_dimension"] = self.max_bond_dimension
        if self.seed is not None:
            options["seed_simulator"] = self.seed
        return AerSimulator(**options)

    def _ibm_backend(self, service):
        name = self.device_spec.lower()
        fractional = {"use_fractional_gates": self.enable_fractional_gates}
        if name in self.LEAST_BUSY:
            return service.least_busy(operational=True, simulator=False,
                                      **fractional)
        names = [n.strip() for n in name.split(",") if n.strip()]
        if len(names) == 1:
            return service.backend(names[0], **fractional)
        # The least busy of the named processors.
        candidates = [service.backend(n, **fractional) for n in names]
        operational = [b for b in candidates if b.status().operational]
        if not operational:
            raise RuntimeError(f"none of {names} is operational right now")
        return min(operational, key=lambda b: b.status().pending_jobs)

    @staticmethod
    def _fake_backend(name: str):
        from qiskit_ibm_runtime import fake_provider
        stem = name[len("fake_"):].replace("_", " ").title().replace(" ", "")
        for attr in (f"Fake{stem}V2", f"Fake{stem}"):
            cls = getattr(fake_provider, attr, None)
            if cls is not None:
                return cls()
        raise ValueError(
            f"unknown fake backend {name!r}: qiskit_ibm_runtime.fake_provider "
            f"has no Fake{stem}V2")

    def estimator(self):
        """The Estimator primitive for :attr:`device_spec` (cached)."""
        if self._estimator is None:
            if self.is_local and self.precision > 0:
                # Shots are sampled: noiseless Aer (state-vector method, as the exact local
                # estimator also holds the state), each basis measured `shots` times, the
                # standard errors from the counts.
                from qiskit.primitives import BackendEstimatorV2
                from qiskit_aer import AerSimulator

                self._estimator = BackendEstimatorV2(
                    backend=AerSimulator(method="statevector"),
                    options={"default_precision": self.precision,
                             "seed_simulator": self.seed})
            elif self.is_local:
                from qiskit.primitives import StatevectorEstimator
                self._estimator = StatevectorEstimator(seed=self.seed)
            elif self.is_fake_device or self.is_mps_device:
                from qiskit.primitives import BackendEstimatorV2
                from qiskit_aer import AerSimulator

                backend = (self.backend() if self.is_mps_device
                           else AerSimulator.from_backend(self.backend()))
                self._estimator = BackendEstimatorV2(
                    backend=backend,
                    options={"default_precision": self.precision})
            else:
                from qiskit_ibm_runtime import Estimator
                self._estimator = Estimator(mode=self.backend(),
                                            options=self.estimator_options)
        return self._estimator

    # -- circuit construction --------------------------------------------- #

    def build(self, n_qubits: int, occupied, generators, thetas):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)
        # Qiskit is little-endian: Mandacaru qubit k lives on wire n-1-k so the
        # simulated amplitude ordering matches Mandacaru's without a permutation.
        def wire(k):
            return n_qubits - 1 - int(k)

        for op in self._emit(n_qubits, list(occupied), generators, thetas):
            gate = op[0]
            if gate == "cx":
                qc.cx(wire(op[1]), wire(op[2]))
            elif gate == "rz":
                qc.rz(op[2], wire(op[1]))
            else:
                getattr(qc, gate)(wire(op[1]))
        return qc

    def statevector(self, n_qubits: int, occupied, generators, thetas):
        from qiskit.quantum_info import Statevector

        if self.shots or not self.is_local:
            raise ValueError(
                f"{self!r} cannot return a state vector (a processor never "
                "exposes amplitudes); measure Pauli strings with "
                "provider.expectation_values(...) instead")
        qc = self.build(n_qubits, occupied, generators, thetas)
        return np.asarray(Statevector(qc).data, dtype=complex)

    def profile(self, n_qubits: int, occupied, generators) -> dict:
        """Compiled cost of the ansatz, against the device's own ISA when there is one.

        The count used to be taken against a proxy basis, ``["cx", "u"]``, on
        every device.  That is a reasonable structural measure and it is *not*
        the number that decides a hardware run: the processor's own instruction
        set has a different entangler, a coupling map that forces routing swaps,
        and -- with ``enable_fractional_gates`` -- a continuously parameterized
        ``RZZ`` that replaces pairs of fixed entanglers.  A depth quoted against
        the proxy basis therefore understates a real submission, sometimes badly.

        So a real or fake backend is transpiled against, at the same
        ``optimization_level`` a submission would use, and the reported
        ``two_qubit_gates`` is counted by arity
        (:func:`~mandacaru.backends.measurement.two_qubit_gate_count`) rather
        than by gate name.  ``basis`` says which instruction set the numbers
        belong to, because a count without that label is not comparable to
        anything.  The local state-vector path has no ISA and keeps the proxy,
        labelled as such.
        """
        from qiskit import transpile

        from .measurement import two_qubit_gate_count

        qc = self.build(n_qubits, occupied, generators,
                        np.ones(len(generators)))
        backend = self.backend()
        if backend is None:
            compiled = transpile(qc, basis_gates=["cx", "u"],
                                 optimization_level=1)
            basis = "proxy (cx, u)"
        else:
            compiled, _layout = self._transpiled(qc, n_qubits)
            basis = getattr(backend, "name", None) or str(backend)
        counts = compiled.count_ops()
        two_q = two_qubit_gate_count(compiled)
        return {"cnot_count": two_q,
                "two_qubit_gates": two_q,
                "depth": int(compiled.depth()),
                "num_1q_gates": int(sum(
                    v for k, v in counts.items()
                    if k not in ("barrier", "measure", "reset", "delay")
                    and k not in ("cz", "cx", "ecr", "rzz", "swap"))),
                "total_gates": int(sum(counts.values())),
                "isa": basis}

    # -- Estimator energies (the hardware path) ---------------------------- #

    @staticmethod
    def observable(hamiltonian):
        """The Hamiltonian as a ``SparsePauliOp``.

        Mandacaru labels list qubit 0 first and Qiskit labels list the highest
        wire first; with qubit ``k`` on wire ``n-1-k`` the strings coincide.
        """
        return hamiltonian.to_sparse_pauli_op()

    def _transpiled(self, qc, n_qubits: int):
        """``(circuit, layout)`` for this target; ``layout`` is ``None`` locally."""
        backend = self.backend()
        if backend is None:
            return qc, None
        from qiskit import transpile
        initial_layout = None
        if self.physical_qubits is not None:
            if len(self.physical_qubits) != n_qubits:
                raise ValueError(
                    f"physical_qubits has {len(self.physical_qubits)} entries "
                    f"for a {n_qubits}-qubit circuit")
            # Mandacaru qubit k sits on Qiskit wire n-1-k.
            initial_layout = [self.physical_qubits[n_qubits - 1 - w]
                              for w in range(n_qubits)]
        isa = transpile(qc, backend=backend,
                        optimization_level=self.optimization_level,
                        initial_layout=initial_layout,
                        seed_transpiler=0 if self.seed is None else self.seed)
        return isa, isa.layout

    def parametric(self, n_qubits: int, occupied, generators):
        """``(circuit, parameters)``: the ansatz with one symbolic angle per
        generator.

        Every rotation of every generator is kept, zero angle or not, so the
        circuit's structure does not change between two parameter points and
        one compilation serves the whole optimization.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        generators = list(generators)
        params = ParameterVector("theta", len(generators))
        qc = QuantumCircuit(n_qubits)

        def wire(k):
            return n_qubits - 1 - int(k)

        for q in occupied:
            qc.x(wire(q))
        context = self._context()
        for k, generator in enumerate(generators):
            terms = [(label, complex(c).real, complex(c).imag)
                     for label, c in generator.terms.items()]
            for gate, first, second, angle in context.compile_circuit(
                    n_qubits, [], [terms], [1.0]):
                if gate == "cx":
                    qc.cx(wire(first), wire(second))
                elif gate == "rz":
                    qc.rz(angle * params[k], wire(first))
                else:
                    getattr(qc, gate)(wire(first))
        return qc, params

    def compiled(self, n_qubits: int, occupied, generators):
        """``(isa_circuit, layout, parameters)`` for this target, cached.

        The one compiled form of an ansatz: every evaluation binds its
        angles to it, and the measurement plan reads its gate counts off the
        same circuit that is submitted.  Transpilation is seeded (the
        provider's ``seed``, else 0), so the layout is reproducible.
        """
        key = (int(n_qubits), tuple(int(q) for q in occupied),
               tuple(tuple(sorted((str(l), complex(c))
                                  for l, c in getattr(g, "terms", {}).items()))
                     for g in generators))
        cached = self._templates.get(key)
        if cached is None:
            qc, params = self.parametric(n_qubits, occupied, generators)
            isa, layout = self._transpiled(qc, n_qubits)
            cached = self._templates[key] = (isa, layout, params)
        return cached

    def pub(self, n_qubits: int, occupied, generators, thetas, hamiltonian):
        """One Estimator PUB ``(isa_circuit, observable, angles)``."""
        isa, layout, params = self.compiled(n_qubits, occupied, generators)
        observable = self.observable(hamiltonian)
        if layout is not None:
            observable = observable.apply_layout(layout)
        return self._bound_pub(isa, params, observable, thetas)

    @staticmethod
    def _bound_pub(isa, params, observable, thetas):
        values = np.asarray(thetas, dtype=float).ravel()
        if len(values) != len(params):
            raise ValueError(f"{len(values)} angles for {len(params)} "
                             "generators")
        return (isa, observable) if not len(params) else \
            (isa, observable, values)

    def _run_pubs(self, pubs, bases: int | None = None):
        """Run Estimator PUBs as **one** job and return the ``PrimitiveResult``.

        ``bases`` -- the measurement bases the PUBs' observables need -- sizes
        the run-ledger entry; the job is refused before it is queued when it
        would exceed :attr:`run_budget`.
        """
        from .measurement import resilience_multipliers

        if not self.is_local and self.shots <= 0:
            raise ValueError(f"{self!r}: a processor needs shots > 0")
        factors, twirls, per_twirl = resilience_multipliers(self)
        self._charge(self._context().measurement_cost(
            max(int(bases or len(pubs)), 1), shots=max(int(self.shots), 0),
            noise_factors=factors, randomizations=twirls,
            shots_per_randomization=per_twirl)[1])
        estimator = self.estimator()
        if self.is_ibm_device:
            self.last_job = estimator.run(pubs, precision=self.precision)
            self._record_job(self.last_job)
            result = self.last_job.result()
        else:
            result = run_local_estimator(estimator, pubs, self.precision)
        self.last_result = result
        return result

    def expectation_values(self, n_qubits: int, occupied, generators, thetas,
                           labels, max_bases_per_job: int | None = None):
        """``<P>`` of one state for several Pauli strings.

        Returns ``({label: value}, {label: standard error})``.  One circuit and
        an array of observables: what measuring the RDMs of an optimized state
        on a processor costs.  It is :meth:`expectation_batch` of one request.
        """
        return self.expectation_batch(
            n_qubits, occupied, [(generators, thetas, labels)],
            max_bases_per_job)[0]

    @staticmethod
    def _read_chunk(chunk, data):
        """``({label: value}, {label: standard error})`` of a PUB's data."""
        return (dict(zip(chunk, np.asarray(data.evs, dtype=float).reshape(-1))),
                dict(zip(chunk, np.asarray(data.stds, dtype=float).reshape(-1))))

    def expectation_batch(self, n_qubits: int, occupied, requests,
                          max_bases_per_job: int | None = None):
        """``<P>`` of several states, each for its own Pauli strings, in as few
        jobs as the limit allows.

        ``requests`` is a list of ``(generators, thetas, labels)``: the ansatz
        circuit (its generators and angles, on one register and reference) and
        the strings to read off it.  Returns one ``({label: value},
        {label: standard error})`` per request.  Every request is one Estimator
        PUB -- one circuit and an array of observables -- and **all the PUBs ride
        in one job**: a measurement that needs several circuits (the
        double-factorized energy, one basis rotation per circuit) still costs
        one submission.

        With ``max_bases_per_job`` (or the constructor's
        :attr:`max_bases_per_job`) each request's observables are split into
        chunks of at most that many **qubit-wise commuting measurement bases**,
        never cutting a basis in half, and the chunks are packed in order into
        jobs of at most that many bases in all, so the jobs are
        ``ceil(bases / limit)`` as :func:`measurement_cost` counts.  The ansatz
        is transpiled once per circuit and shared by its chunks, so the split
        costs nothing but job overhead, and it bounds what any single Runtime
        program has to hold -- the unsplit form submitted 97,980 observables in
        one PUB and died with "Program runtime ran out of memory".

        Completed chunks are **kept**: each is stored, keyed by the circuit,
        the labels and the options, as soon as its job returns, and a repeat
        of the same request (a rerun after a failure) submits only the chunks
        still missing.  Every job
        of the submission is on :attr:`jobs`.
        """
        from qiskit.quantum_info import SparsePauliOp

        from .measurement import chunk_labels_by_basis, qwc_partition

        limit = (self.max_bases_per_job if max_bases_per_job is None
                 else max_bases_per_job)
        plans, pending = [], []
        for index, (generators, thetas, labels) in enumerate(requests):
            labels = [str(label) for label in labels]
            key = self._submission_key(n_qubits, occupied, generators, thetas,
                                       labels, limit)
            done = self._completed_chunks.setdefault(key, {})
            chunks = chunk_labels_by_basis(labels, limit, **self._core_options())
            plans.append((key, done, len(chunks)))
            for chunk_index, chunk in enumerate(chunks):
                if chunk_index not in done:
                    bases = len(qwc_partition(chunk, **self._core_options()))
                    pending.append((index, chunk_index, chunk, bases))
        jobs, current, load = [], [], 0
        for item in pending:
            if current and limit and load + item[3] > limit:
                jobs.append((current, load))
                current, load = [], 0
            current.append(item)
            load += item[3]
        if current:
            jobs.append((current, load))
        self.jobs = []
        for job, bases in jobs:
            pubs = []
            for index, _chunk_index, chunk, _bases in job:
                generators, thetas, _labels = requests[index]
                qc, layout, params = self.compiled(n_qubits, occupied, generators)
                observables = [SparsePauliOp(label) for label in chunk]
                if layout is not None:
                    observables = [o.apply_layout(layout) for o in observables]
                pubs.append(self._bound_pub(qc, params, observables, thetas))
            result = self._run_pubs(pubs, bases=bases)
            if self.last_job is not None and self.is_ibm_device:
                self.jobs.append(self.last_job)
            for position, (index, chunk_index, chunk, _bases) in enumerate(job):
                plans[index][1][chunk_index] = self._read_chunk(chunk, result[position].data)
        outputs = []
        for key, done, count in plans:
            values: dict[str, float] = {}
            stds: dict[str, float] = {}
            for chunk_index in range(count):
                values.update(done[chunk_index][0])
                stds.update(done[chunk_index][1])
            outputs.append((values, stds))
            # Complete: nothing left to resume.
            self._completed_chunks.pop(key, None)
        return outputs

    def completed_chunks(self) -> int:
        """Chunks of unfinished ``expectation_values`` requests already
        measured -- what a retry of the same request will not resubmit."""
        return sum(len(done) for done in self._completed_chunks.values())

    def _submission_key(self, n_qubits, occupied, generators, thetas, labels,
                        limit) -> str:
        """A fingerprint of what a submission measures and how."""
        import hashlib

        parts = [str(int(n_qubits)), repr(sorted(int(q) for q in occupied)),
                 repr(_native.rounded(np.asarray(thetas, dtype=float).ravel().tolist(), 12)),
                 repr([sorted((str(l), complex(c)) for l, c in
                              getattr(g, "terms", {}).items())
                       for g in generators]),
                 repr(labels), repr(limit), self.device_spec, str(self.shots),
                 repr(self.estimator_options), repr(self.physical_qubits)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

# --------------------------------------------------------------------------- #
# Amazon Braket.
# --------------------------------------------------------------------------- #

class BraketProvider(CircuitProvider):
    """Amazon Braket circuits, on the local simulator or the **AWS service**.

    Braket is big-endian (qubit 0 is the most significant amplitude bit), the
    same convention as Mandacaru, so no wire permutation is needed.

    Two execution modes, chosen by ``shots``:

    * ``shots = 0`` (default) -- ask for the ``StateVector`` result type and get
      the exact amplitudes.  This is **simulator-only**: Braket rejects
      ``StateVector`` whenever ``shots > 0``, and every QPU requires
      ``shots > 0``.
    * ``shots > 0`` -- the **hardware path**.  The Hamiltonian is partitioned
      into qubit-wise-commuting groups (:mod:`mandacaru.backends.measurement`), one
      measurement circuit is submitted per group, and ``<H>`` is assembled from
      the returned bit-string counts.  This is what runs on an IonQ / Rigetti /
      IQM QPU or on the managed SV1 / DM1 / TN1 simulators.

    Parameters
    ----------
    device : str, optional
        A local simulator backend name (``"braket_sv"``, the default,
        ``"braket_dm"``) or an **AWS device ARN**
        (``"arn:aws:braket:::device/quantum-simulator/amazon/sv1"``,
        ``"arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"``, ...).  Anything
        starting with ``"arn:aws:braket"`` is opened with
        :class:`braket.aws.AwsDevice`, which needs configured AWS credentials and
        bills to your account.
    shots : int
        Measurement shots per circuit (default ``0`` = exact state vector).
        Required to be positive for any QPU.
    s3_folder : (str, str), optional
        ``(bucket, prefix)`` for AwsDevice result storage.  Modern Braket regions
        default to a service-managed bucket, so this is usually unnecessary.
    poll_timeout_seconds : float
        How long to wait for an AWS quantum task to finish (default 5 days, the
        Braket default -- QPU queues are long).
    """

    name = "braket"
    supports_shots = True

    #: Prefix identifying an AWS-managed device (vs. a local simulator name).
    AWS_ARN_PREFIX = "arn:aws:braket"

    def __init__(self, device: str = "braket_sv", shots: int = 0,
                 s3_folder=None, poll_timeout_seconds: float = 5 * 24 * 60 * 60,
                 run_budget=None):
        self.device_spec = str(device)
        self.shots = int(shots)
        self.s3_folder = s3_folder
        self.poll_timeout_seconds = float(poll_timeout_seconds)
        #: Run-wide limit on tasks and shots (on by default for an AWS
        #: device; see :func:`~mandacaru.backends.measurement.resolve_run_budget`).
        self.run_budget = run_budget
        self._device = None

    def __repr__(self) -> str:
        return (f"BraketProvider(device={self.device_spec!r}, "
                f"shots={self.shots})")

    def _constructor_options(self) -> dict:
        return {"device": self.device_spec, "shots": self.shots,
                "s3_folder": self.s3_folder,
                "poll_timeout_seconds": self.poll_timeout_seconds,
                "run_budget": self.run_budget}

    @property
    def is_hardware(self) -> bool:
        return self.is_aws_device

    # -- device resolution ------------------------------------------------ #

    @property
    def is_aws_device(self) -> bool:
        """True when this provider targets the AWS Braket service (not local)."""
        return self.device_spec.startswith(self.AWS_ARN_PREFIX)

    def device(self):
        """The resolved Braket device (cached).

        A local simulator name gives a :class:`braket.devices.LocalSimulator`; an
        ARN gives a :class:`braket.aws.AwsDevice`, whose construction contacts
        AWS and therefore requires credentials.
        """
        if self._device is None:
            if self.is_aws_device:
                from braket.aws import AwsDevice
                self._device = AwsDevice(self.device_spec)
            else:
                from braket.devices import LocalSimulator
                self._device = LocalSimulator(self.device_spec)
        return self._device

    def _run(self, circuit, shots: int):
        """Submit ``circuit`` and return its Braket result."""
        return self._submit(circuit, shots).result()

    def _submit(self, circuit, shots: int):
        """Queue ``circuit`` (after the run-budget check) and return its task."""
        self._charge(int(shots))
        device = self.device()
        kwargs = {"shots": int(shots)}
        if self.is_aws_device:
            kwargs["poll_timeout_seconds"] = self.poll_timeout_seconds
            if self.s3_folder is not None:
                kwargs["s3_destination_folder"] = tuple(self.s3_folder)
        task = device.run(circuit, **kwargs)
        if self.is_aws_device:
            self._record_job(task)
        return task

    # -- circuit construction --------------------------------------------- #

    def build(self, n_qubits: int, occupied, generators, thetas,
              measure_basis: str | None = None):
        """Build the Braket circuit, optionally rotated into ``measure_basis``.

        With ``measure_basis`` given, the single-qubit rotations that map that
        Pauli basis onto ``Z`` are appended, so a plain computational-basis
        readout samples the requested basis -- the hardware measurement path.
        """
        from braket.circuits import Circuit

        circuit = Circuit()
        # Touch every qubit so the circuit spans the full register even when a
        # wire carries no gate (an unused qubit would otherwise be dropped and
        # shrink the returned state vector / bit-strings).
        for q in range(n_qubits):
            circuit.i(q)
        stream = list(self._emit(n_qubits, list(occupied), generators, thetas))
        if measure_basis is not None:
            stream += list(self._measurement_rotation(measure_basis))
        for op in stream:
            gate = op[0]
            if gate == "cx":
                circuit.cnot(int(op[1]), int(op[2]))
            elif gate == "rz":
                circuit.rz(int(op[1]), float(op[2]))
            elif gate == "sdg":
                circuit.si(int(op[1]))
            else:
                getattr(circuit, gate)(int(op[1]))
        return circuit

    def statevector(self, n_qubits: int, occupied, generators, thetas):
        if self.shots:
            raise ValueError(
                f"{self!r} is configured for shot-based execution, which cannot "
                "return a state vector (Braket rejects the StateVector result "
                "type whenever shots > 0, and every QPU requires shots > 0). "
                "Use provider.expectation_values(...) for shot-based expectation values, "
                "or set shots=0 for an exact simulator run.")
        circuit = self.build(n_qubits, occupied, generators, thetas)
        circuit.state_vector()
        return np.asarray(self._run(circuit, 0).values[0], dtype=complex)

    def profile(self, n_qubits: int, occupied, generators) -> dict:
        circuit = self.build(n_qubits, occupied, generators,
                             np.ones(len(generators)))
        # Braket has no transpiler here; the emitted stream is already in a native
        # {CNOT, 1q} set, so count it directly.  The identity padding is excluded.
        instructions = [ins for ins in circuit.instructions
                        if ins.operator.name != "I"]
        cnots = sum(1 for ins in instructions if ins.operator.name == "CNot")
        return {"cnot_count": int(cnots),
                "depth": int(circuit.depth),
                "num_1q_gates": int(len(instructions) - cnots),
                "total_gates": int(len(instructions))}

    # -- shot-based energy (QPU compatible) -------------------------------- #

    capabilities = ("expectation_values", "expectation_batch")

    def expectation_values(self, n_qubits: int, occupied, generators, thetas,
                           labels, max_bases_per_job: int | None = None):
        """``<P>`` of one state for several Pauli strings, from shots.

        It is :meth:`expectation_batch` of one request.
        """
        return self.expectation_batch(
            n_qubits, occupied, [(generators, thetas, labels)],
            max_bases_per_job)[0]

    def expectation_batch(self, n_qubits: int, occupied, requests,
                          max_bases_per_job: int | None = None):
        """``<P>`` of several states, each for its own Pauli strings, from shots.

        ``requests`` is a list of ``(generators, thetas, labels)``; the result is
        one ``({label: value}, {label: standard error})`` per request.  Each
        request's strings are partitioned into qubit-wise-commuting bases
        (:func:`~mandacaru.backends.measurement.qwc_partition`), one quantum
        task per basis, and the tasks of **every** request are queued before
        any is awaited; each string is read off its basis's counts.  A Braket
        submission is a task per basis, so the batch costs the sum of the
        requests' bases in tasks, which is what
        ``measurement_cost(..., job_per_basis=True)`` counts.
        ``max_bases_per_job`` is accepted for the provider protocol and has no
        effect.  With ``shots = 0`` the values are the exact state-vector
        expectations (simulator only).
        """
        from .measurement import qwc_partition

        requests = [(generators, thetas, [str(label) for label in labels])
                    for generators, thetas, labels in requests]
        if not self.shots:
            return [self._exact_expectations(
                self.statevector(n_qubits, occupied, generators, thetas), labels)
                for generators, thetas, labels in requests]
        partitions = [qwc_partition(labels, **self._core_options())
                      for _generators, _thetas, labels in requests]
        tasks = [[self._submit(self.build(n_qubits, occupied, generators,
                                          thetas, measure_basis=basis),
                               self.shots)
                  for basis, _members in partition]
                 for (generators, thetas, _labels), partition
                 in zip(requests, partitions)]
        outputs = []
        for (_generators, _thetas, labels), partition, submitted in zip(
                requests, partitions, tasks):
            values, stds = {}, {}
            for (_basis, members), task in zip(partition, submitted):
                self._read_basis(labels, members, self._counts(task.result()),
                                 values, stds)
            outputs.append((values, stds))
        return outputs

    def _exact_expectations(self, psi, labels):
        """The state-vector ``({label: value}, {label: 0})`` of a simulator run."""
        exact = self._context().state_expectations(
            list(labels), np.ascontiguousarray(psi, dtype=complex))
        return dict(zip(labels, exact)), dict.fromkeys(labels, 0.0)

    def _read_basis(self, labels, members, counts, values, stds):
        """Read the strings ``members`` of ``labels`` off one basis's counts: their means and
        standard errors over the counted shots (``Context.count_statistics``)."""
        names = [labels[i] for i in members]
        means, errors = self._context().count_statistics(
            names, [(str(k), float(v)) for k, v in counts.items()])
        values.update(zip(names, means))
        stds.update(zip(names, errors))

    @staticmethod
    def _counts(result) -> dict:
        """Braket measurement counts as ``{bitstring: n}`` (qubit 0 leftmost).

        Braket's ``measurement_counts`` keys are already ordered with qubit 0
        first, matching Mandacaru's convention.
        """
        return {str(bits): int(n)
                for bits, n in result.measurement_counts.items()}

# --------------------------------------------------------------------------- #
# Google Cirq.
# --------------------------------------------------------------------------- #

class CirqProvider(CircuitProvider):
    """Cirq circuits on :class:`cirq.Simulator` with an explicit big-endian order."""

    name = "cirq"

    @staticmethod
    def _qubits(n_qubits: int):
        import cirq
        return [cirq.LineQubit(k) for k in range(n_qubits)]

    def build(self, n_qubits: int, occupied, generators, thetas):
        import cirq

        q = self._qubits(n_qubits)
        moments = []
        for op in self._emit(n_qubits, list(occupied), generators, thetas):
            gate = op[0]
            if gate == "cx":
                moments.append(cirq.CNOT(q[int(op[1])], q[int(op[2])]))
            elif gate == "rz":
                moments.append(cirq.rz(float(op[2])).on(q[int(op[1])]))
            elif gate == "sdg":
                moments.append((cirq.S ** -1).on(q[int(op[1])]))
            elif gate == "x":
                moments.append(cirq.X(q[int(op[1])]))
            elif gate == "h":
                moments.append(cirq.H(q[int(op[1])]))
            else:                                       # "s"
                moments.append(cirq.S(q[int(op[1])]))
        # Identities keep every qubit in the circuit's register (see Braket).
        circuit = cirq.Circuit([cirq.I(qi) for qi in q])
        circuit.append(moments)
        return circuit

    def statevector(self, n_qubits: int, occupied, generators, thetas):
        import cirq

        circuit = self.build(n_qubits, occupied, generators, thetas)
        result = cirq.Simulator(dtype=np.complex128).simulate(
            circuit, qubit_order=self._qubits(n_qubits))
        # cirq's qubit_order is big-endian: the first qubit is the most
        # significant amplitude bit, matching Mandacaru.
        return np.asarray(result.final_state_vector, dtype=complex)

    def profile(self, n_qubits: int, occupied, generators) -> dict:
        import cirq

        circuit = self.build(n_qubits, occupied, generators,
                             np.ones(len(generators)))
        ops = [op for op in circuit.all_operations()
               if not isinstance(op.gate, type(cirq.I))]
        cnots = sum(1 for op in ops if len(op.qubits) == 2)
        # Depth excluding the identity padding moment.
        depth = max(len(cirq.Circuit(ops)), 0)
        return {"cnot_count": int(cnots),
                "depth": int(depth),
                "num_1q_gates": int(len(ops) - cnots),
                "total_gates": int(len(ops))}


# --------------------------------------------------------------------------- #
# Registry.
# --------------------------------------------------------------------------- #

_PROVIDER_CLASSES = {
    "qiskit": QiskitProvider,
    "braket": BraketProvider,
    "cirq": CirqProvider,
}




def require_capability(provider, capability: str, why: str) -> None:
    """Refuse ``provider`` unless it can do ``capability`` (see
    :attr:`CircuitProvider.capabilities`); ``why`` says what needs it.

    A provider object that declares nothing is judged by the method the
    capability names.
    """
    declared = getattr(provider, "capabilities", None)
    able = (capability in declared if declared is not None
            else callable(getattr(provider, capability, None)))
    if not able:
        raise NotImplementedError(
            f"{why}, and {provider!r} cannot measure "
            f"{capability.replace('_', ' ')}; use a QiskitProvider (IBM, fake "
            f"or local), or ask for the energy only")


def qpu_usage(provider, wall_time_s: float | None = None,
              cumulative: bool = False) -> dict:
    """QPU accounting for ``provider``'s most recent submission, or ``{}``.

    With ``cumulative=True`` it covers every job of the run instead
    (:attr:`CircuitProvider.run_jobs`) -- what an optimization on a processor
    costs, one job per objective evaluation.

    Reported for the ``[PERFORMANCE]`` log block.  ``wall_time_s`` -- measured by
    the caller around the submission -- is always included, because it is the
    only figure available for every backend; the *quantum* seconds and the job id
    come from Qiskit Runtime, which reports them for a real device only.  A local
    simulator therefore contributes the wall clock and the device name, and
    nothing is invented.

    Everything is read defensively: a provider or job object that cannot answer
    leaves its key out rather than failing a run that has already finished.
    """
    if provider is None:
        return {}
    usage: dict = {"qpu_device": str(getattr(provider, "device_spec", provider))}
    shots = getattr(provider, "shots", 0)
    if shots:
        usage["qpu_shots"] = int(shots)
    if wall_time_s is not None:
        usage["qpu_wall_time_s"] = round(float(wall_time_s), 4)

    # A chunked submission is several jobs (`max_bases_per_job`); accounting
    # that reported only the last one would under-report the whole cost by the
    # chunk factor, which is exactly the number the chunking makes large.
    if cumulative and hasattr(provider, "run_usage"):
        jobs = list(provider.run_jobs)
        usage["qpu_submissions"] = int(provider.run_usage()["jobs"])
    else:
        jobs = list(getattr(provider, "jobs", None) or ())
        if not jobs:
            job = getattr(provider, "last_job", None)
            jobs = [] if job is None else [job]
    if not jobs:
        return usage
    usage["qpu_jobs"] = len(jobs)
    ids = []
    seconds: dict[str, float] = {}
    for job in jobs:
        try:
            ids.append(str(job.job_id()))
        except Exception:
            pass
        try:
            metrics = job.metrics() or {}
            reported = metrics.get("usage") or {}
            for key, name in (("quantum_seconds", "qpu_seconds"),
                              ("seconds", "qpu_billed_seconds")):
                if reported.get(key) is not None:
                    seconds[name] = seconds.get(name, 0.0) + float(reported[key])
        except Exception:
            # Not a Runtime job, or the service could not be reached: the wall
            # time above stands on its own.
            pass
    if ids:
        usage["qpu_job_ids"] = ", ".join(ids)
    for name, total in seconds.items():
        usage[name] = round(total, 4)
    return usage


def build_provider(name: str = "qiskit", *, context=None, **options) -> CircuitProvider:
    """Return the :class:`CircuitProvider` for ``name``.

    Default-configured providers are cached and shared; passing ``**options``
    (e.g. ``device=``/``shots=`` for :class:`BraketProvider`) builds a fresh,
    uncached instance so two differently-targeted backends never alias.

    Raises
    ------
    ValueError
        If ``name`` is not one of :data:`BACKEND_PROVIDERS`.
    ImportError
        If the provider's SDK is not installed (raised on first use, so naming a
        provider never fails at import time).
    TypeError
        If ``options`` are given for a provider that takes none.
    """
    canon = normalize_provider(name)
    if {"engine","engine_options"} & options.keys():
        raise TypeError("execution engine selection was removed; pass context= to share calculation resources")
    from .._native import Context
    provider = _PROVIDER_CLASSES[canon](**options)
    provider._native_context = Context() if context is None else context
    return provider

def provider_available(name: str) -> bool:
    """True when the SDK backing ``name`` can be imported in this environment."""
    canon = normalize_provider(name)
    module = {"qiskit": "qiskit", "braket": "braket.circuits", "cirq": "cirq"}[canon]
    try:
        __import__(module)
    except Exception:
        return False
    return True
