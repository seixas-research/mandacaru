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

from ..core.mapping import PauliSum

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

def pauli_rotations(generator: PauliSum, atol: float = 1e-12
                    ) -> list[tuple[str, float]]:
    r"""Decompose an anti-Hermitian generator into ``(pauli_string, angle_coeff)``.

    For ``A = sum_j i c_j P_j`` this returns ``[(P_j, c_j), ...]`` with real
    ``c_j``, so ``exp(theta A) = prod_j exp(i theta c_j P_j)`` -- an identity
    that holds **because the terms commute**, which is checked, not assumed.
    Identity strings contribute only a global phase and are dropped.

    Raises
    ------
    ValueError
        If any coefficient has a non-negligible real part (the operator is not
        anti-Hermitian and does not generate a unitary for real ``theta``), or
        if two of its Pauli terms anticommute: the product of their rotations
        would then be a first-order Trotter *approximation* of ``exp(theta A)``
        (fidelity 0.98 for ``0.3i X + 0.2i Z`` at ``theta = 1.7``), silently a
        different state from the one the state-vector backend prepares.  Every
        pool generator Mandacaru builds has commuting terms, in every mapping.
    """
    out: list[tuple[str, float]] = []
    for label, coeff in sorted(generator.simplify().terms.items()):
        c = complex(coeff)
        if abs(c.real) > 1e-8 * max(1.0, abs(c.imag)):
            raise ValueError(
                f"generator term {label!r} has coefficient {c!r}: the generator "
                "must be anti-Hermitian (purely imaginary Pauli coefficients)")
        if abs(c.imag) <= atol or set(label) == {"I"}:
            continue                       # negligible, or a global phase only
        out.append((label, float(c.imag)))
    for i, (first, _c) in enumerate(out):
        for second, _d in out[i + 1:]:
            if not pauli_strings_commute(first, second):
                raise ValueError(
                    f"generator terms {first!r} and {second!r} anticommute, so "
                    "exp(theta A) is not the product of their rotations and a "
                    "circuit built from them would prepare a different state. "
                    "Split the generator into commuting parts (one ansatz "
                    "operator each), or evaluate it on the state-vector "
                    "backend, which exponentiates it exactly.")
    return out


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


def pauli_strings_commute(first: str, second: str) -> bool:
    """Two Pauli strings commute iff they differ, with neither the identity,
    at an **even** number of positions."""
    clashes = sum(1 for a, b in zip(first, second)
                  if a != "I" and b != "I" and a != b)
    return clashes % 2 == 0


def evolve_determinants(provider, n_qubits: int, generators, theta,
                        references: np.ndarray) -> np.ndarray:
    """Run the ansatz circuit on every column of ``references`` (``(2**n, k)``).

    A circuit cannot be initialized in a superposition, so each column must be
    a computational-basis (Slater-determinant) state -- the HF reference and
    the subspace-search references are.  Shared by every ansatz that can
    execute on a :class:`CircuitProvider`.
    """
    columns = []
    for j in range(references.shape[1]):
        index = basis_state_index(references[:, j])
        if index is None:
            raise ValueError(
                f"the {provider.name} circuit backend can only evolve "
                "computational-basis (Slater-determinant) reference states; "
                f"column {j} of `references` is a superposition")
        columns.append(provider.statevector(
            n_qubits, _occupied_qubits(index, n_qubits), generators, theta))
    return np.asarray(columns, dtype=complex).T


def basis_state_index(vector: np.ndarray, atol: float = 1e-9) -> int | None:
    """Index of the single non-zero amplitude of ``vector``, or ``None``.

    Circuits can only be *initialized* in a computational basis state, so the
    provider execution path uses this to check that a reference state is a Slater
    determinant (it always is: Hartree-Fock and the SSVQE references are
    determinants) and falls back to the matrix path otherwise.
    """
    nz = np.flatnonzero(np.abs(np.asarray(vector)) > atol)
    if nz.size != 1:
        return None
    amp = complex(vector[nz[0]])
    if abs(abs(amp) - 1.0) > 1e-6:
        return None
    return int(nz[0])


def _occupied_qubits(index: int, n_qubits: int) -> list[int]:
    """Mandacaru qubit indices set to ``|1>`` in basis state ``index`` (qubit 0 = MSB)."""
    return [k for k in range(n_qubits)
            if (index >> (n_qubits - 1 - k)) & 1]


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

    def _emit(self, n_qubits: int, occupied: list[int], generators, thetas):
        """Yield ``(gate, *qubits[, angle])`` for the full ansatz circuit.

        Gates are ``"x"``, ``"h"``, ``"s"``, ``"sdg"``, ``"cx"`` and ``"rz"`` --
        the common denominator of the three SDKs.  The circuit is the reference
        determinant preparation followed by one Pauli-rotation block per term of
        each generator, in append order.
        """
        thetas = np.asarray(thetas, dtype=float).ravel()
        generators = list(generators)
        if len(thetas) != len(generators):
            raise ValueError(
                f"{len(thetas)} angles for {len(generators)} generators: a "
                "circuit that zips them short prepares a different state")
        for q in occupied:
            yield ("x", q)
        for theta, generator in zip(thetas, generators):
            for label, coeff in pauli_rotations(generator):
                angle = float(theta) * coeff
                if angle == 0.0:
                    continue
                yield from self._pauli_rotation(label, angle)

    @staticmethod
    def _pauli_rotation(label: str, angle: float):
        r"""Gate stream for ``exp(i * angle * P)`` with ``P`` the string ``label``.

        ``exp(i a P) = V exp(i a Z_t) V^dagger`` where ``V`` rotates ``Z`` on the
        last active qubit ``t`` into ``P``: single-qubit basis changes map
        ``X -> Z`` (``H``) and ``Y -> Z`` (``S†`` then ``H``), and a CNOT ladder
        collects the parity of the active qubits onto ``t``.  The rotation itself
        is ``R_z(-2a) = exp(i a Z)``.
        """
        active = [(k, ch) for k, ch in enumerate(label) if ch != "I"]
        if not active:
            return
        # Basis change into the Z eigenbasis (V^dagger).
        for k, ch in active:
            if ch == "X":
                yield ("h", k)
            elif ch == "Y":
                yield ("sdg", k)
                yield ("h", k)
        # Parity ladder: accumulate onto the last active qubit.
        qubits = [k for k, _ in active]
        for a, b in zip(qubits, qubits[1:]):
            yield ("cx", a, b)
        yield ("rz", qubits[-1], -2.0 * angle)
        for a, b in reversed(list(zip(qubits, qubits[1:]))):
            yield ("cx", a, b)
        # Undo the basis change (V).
        for k, ch in active:
            if ch == "X":
                yield ("h", k)
            elif ch == "Y":
                yield ("h", k)
                yield ("s", k)

    # -- provider hooks --------------------------------------------------- #

    @abstractmethod
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
    #: ``"energy"`` a (shot-based) ``<H>``; ``"expectation_values"`` many
    #: ``<P>`` of one state, what RDMs and forces need; ``"batch"`` several
    #: problems in one submission (``energies``).
    capabilities: tuple = ("energy",)

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

        ledger = self._ledger()
        budget = resolve_run_budget(self.run_budget, self.is_hardware)
        after = {"jobs": ledger["jobs"] + int(jobs),
                 "total_shots": ledger["shots"] + int(shots)}
        if budget is not None:
            over = [(key, after[key], budget[key]) for key in after
                    if budget.get(key) is not None
                    and after[key] > budget[key]]
            if over:
                detail = "; ".join(f"{key} {value:,} > {limit:,}"
                                   for key, value, limit in over)
                raise RunBudgetExceeded(
                    f"{self!r}: this submission would exceed the run budget "
                    f"({detail}) after {ledger['jobs']:,} jobs and "
                    f"{ledger['shots']:,} shots.  Raise run_budget= (a dict "
                    f"of jobs / total_shots) if you mean it, or rehearse on "
                    f"a fake backend first.")
        ledger["jobs"] = after["jobs"]
        ledger["shots"] = after["total_shots"]

    def jobs_per_energy(self, hamiltonian) -> int:
        """Submissions one :meth:`energy` of ``hamiltonian`` costs."""
        return 1

    def shots_per_energy(self, hamiltonian) -> int:
        """Shots one :meth:`energy` of ``hamiltonian`` costs (the ledger's
        arithmetic), so a budget can be checked before the first job."""
        return 0

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
        return fresh

    def energy(self, n_qubits: int, occupied, generators, thetas,
               hamiltonian) -> float:
        r"""Estimate ``<psi(thetas)| H |psi(thetas)>`` on this backend.

        The default implementation goes through :meth:`statevector`, which is
        exact but **simulator-only**.  Providers that can measure shots override
        this with the hardware-compatible path (see
        :mod:`mandacaru.backends.measurement`).
        """
        psi = self.statevector(n_qubits, occupied, generators, thetas)
        return float(np.real(np.vdot(psi, hamiltonian.to_matrix() @ psi)))

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
    Estimator primitive -- ``qiskit.primitives.StatevectorEstimator`` locally,
    ``BackendEstimatorV2`` on a Qiskit Runtime *fake backend*, and the Qiskit
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
        ``0`` (default) is the exact local expectation value; a QPU requires
        ``shots > 0``.
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

    def __init__(self, device: str = "statevector", shots: int = 0,
                 instance: str | None = None, token: str | None = None,
                 channel: str | None = None, optimization_level: int = 3,
                 estimator_options: dict | None = None,
                 physical_qubits=None, seed: int | None = None,
                 max_bases_per_job: int | None = None,
                 enable_fractional_gates: bool = False,
                 run_budget=None):
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
        #: backends ignore it -- their randomness is not ours to fix.
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
        self._backend = None
        self._estimator = None
        #: The Runtime job of the last hardware submission (``None`` locally).
        self.last_job = None
        #: Every job of the current (last) submission, oldest first;
        #: :attr:`run_jobs` is every job of the run.
        self.jobs: list = []
        #: The ``PrimitiveResult`` of the last ``energies`` call.
        self.last_result = None
        #: Standard errors of the last ``energies`` call, one per problem.
        self.last_stds: list[float] = []
        #: Chunks of an ``expectation_values`` submission that completed,
        #: keyed by what was submitted, so a retry resumes only the rest.
        self._completed_chunks: dict = {}
        #: Compiled parameterized circuits, one per (register, reference,
        #: generator sequence): an optimization transpiles its ansatz once
        #: and binds new angles per evaluation.
        self._templates: dict = {}

    def __repr__(self) -> str:
        return f"QiskitProvider(device={self.device_spec!r}, shots={self.shots})"

    capabilities = ("energy", "expectation_values", "batch")

    def _constructor_options(self) -> dict:
        return {"device": self.device_spec, "shots": self.shots,
                "instance": self.instance, "token": self.token,
                "channel": self.channel,
                "optimization_level": self.optimization_level,
                "estimator_options": self.estimator_options,
                "physical_qubits": self.physical_qubits, "seed": self.seed,
                "max_bases_per_job": self.max_bases_per_job,
                "enable_fractional_gates": self.enable_fractional_gates,
                "run_budget": self.run_budget}

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
    def is_ibm_device(self) -> bool:
        """True when this provider submits to IBM Quantum hardware."""
        return not (self.is_local or self.is_fake_device)

    @property
    def precision(self) -> float:
        """Estimator target precision: ``1/sqrt(shots)`` (``0`` = exact)."""
        return 0.0 if self.shots <= 0 else 1.0 / np.sqrt(self.shots)

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
            else:
                self._backend = self._ibm_backend(self._service())
        return self._backend

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
            if self.is_local:
                from qiskit.primitives import StatevectorEstimator
                self._estimator = StatevectorEstimator(
                    default_precision=self.precision, seed=self.seed)
            elif self.is_fake_device:
                from qiskit.primitives import BackendEstimatorV2
                self._estimator = BackendEstimatorV2(
                    backend=self.backend(),
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
                "exposes amplitudes); use provider.energy(...) instead")
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
        for k, generator in enumerate(generators):
            for label, coeff in pauli_rotations(generator):
                # `_pauli_rotation(label, 1)` emits rz(-2): scale it by the
                # term's coefficient and the generator's symbolic angle.
                for op in self._pauli_rotation(label, 1.0):
                    gate = op[0]
                    if gate == "cx":
                        qc.cx(wire(op[1]), wire(op[2]))
                    elif gate == "rz":
                        qc.rz(op[2] * float(coeff) * params[k], wire(op[1]))
                    else:
                        getattr(qc, gate)(wire(op[1]))
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
        per_instance = twirls * per_twirl if twirls > 1 else self.shots
        self._charge(max(int(bases or len(pubs)), 1) * factors
                     * max(per_instance, 0))
        estimator = self.estimator()
        if self.is_ibm_device:
            self.last_job = estimator.run(pubs, precision=self.precision)
            self._record_job(self.last_job)
            result = self.last_job.result()
        else:
            result = run_local_estimator(estimator, pubs, self.precision)
        self.last_result = result
        return result

    def _energy_pieces(self, hamiltonian, limit):
        """``hamiltonian`` split into weighted observables of at most
        ``limit`` measurement bases each (the identity rides with the first),
        with the base count of each piece."""
        from .measurement import chunk_labels_by_basis, qwc_partition

        terms = hamiltonian.simplify().terms
        identity = "I" * hamiltonian.num_qubits
        labels = [label for label in terms if label != identity]
        weights = [abs(complex(terms[label])) for label in labels]
        chunks = chunk_labels_by_basis(labels, limit, weights) or [[]]
        pieces = []
        for k, chunk in enumerate(chunks):
            piece = {label: terms[label] for label in chunk}
            if k == 0 and identity in terms:
                piece[identity] = terms[identity]
            if not piece:
                piece = {identity: 0.0}
            op = PauliSum(piece, num_qubits=hamiltonian.num_qubits)
            n_bases = len(qwc_partition(
                chunk, [abs(complex(terms[l])) for l in chunk])) if chunk else 1
            pieces.append((op, n_bases))
        return pieces

    def jobs_per_energy(self, hamiltonian) -> int:
        if not self.max_bases_per_job:
            return 1
        return len(self._energy_pieces(hamiltonian, self.max_bases_per_job))

    def shots_per_energy(self, hamiltonian) -> int:
        from .measurement import resilience_multipliers

        factors, twirls, per_twirl = resilience_multipliers(self)
        per_instance = twirls * per_twirl if twirls > 1 else self.shots
        bases = sum(n for _op, n in self._energy_pieces(
            hamiltonian, self.max_bases_per_job))
        return int(bases * factors * max(per_instance, 0))

    def energies(self, problems) -> list[float]:
        """Expectation values of several ``(n_qubits, occupied, generators,
        thetas, hamiltonian)`` problems.

        One Estimator job carries every problem, as one weighted observable
        each -- unless :attr:`max_bases_per_job` is set, in which case each
        Hamiltonian is split into weighted observables of at most that many
        qubit-wise-commuting bases (the identity counted once) and job ``k``
        carries piece ``k`` of every problem.  The energy is the sum of the
        pieces, its standard error the root sum of squares (:attr:`last_stds`).
        Each circuit is transpiled once and shared by its pieces.
        """
        problems = list(problems)
        limit = self.max_bases_per_job
        prepared = []
        for n_qubits, occupied, generators, thetas, hamiltonian in problems:
            isa, layout, params = self.compiled(n_qubits, occupied,
                                                generators)
            prepared.append(((isa, params, thetas), layout,
                             self._energy_pieces(hamiltonian, limit)))
        energies = [0.0] * len(problems)
        variances = [0.0] * len(problems)
        self.jobs = []
        rounds = max(len(pieces) for _qc, _l, pieces in prepared)
        for k in range(rounds):
            owners, pubs, bases = [], [], 0
            for i, (qc, layout, pieces) in enumerate(prepared):
                if k >= len(pieces):
                    continue
                op, n_bases = pieces[k]
                observable = self.observable(op)
                if layout is not None:
                    observable = observable.apply_layout(layout)
                owners.append(i)
                pubs.append(self._bound_pub(qc[0], qc[1], observable, qc[2]))
                bases += n_bases
            result = self._run_pubs(pubs, bases=bases)
            if self.last_job is not None and self.is_ibm_device:
                self.jobs.append(self.last_job)
            for j, i in enumerate(owners):
                energies[i] += float(np.asarray(result[j].data.evs)
                                     .reshape(-1)[0])
                std = getattr(result[j].data, "stds", None)
                if std is not None:
                    variances[i] += float(np.asarray(std).reshape(-1)[0]) ** 2
        self.last_stds = [float(np.sqrt(v)) for v in variances]
        return energies

    def expectation_values(self, n_qubits: int, occupied, generators, thetas,
                           labels, max_bases_per_job: int | None = None):
        """``<P>`` of one state for several Pauli strings.

        Returns ``({label: value}, {label: standard error})``.  One circuit and
        an array of observables: what measuring the RDMs of an optimized state
        on a processor costs.

        With ``max_bases_per_job`` (or the constructor's
        :attr:`max_bases_per_job`) the observables are split into several
        submissions of at most that many **qubit-wise commuting measurement
        bases**, never cutting a basis in half.  The ansatz is transpiled once
        and shared by every chunk, so the split costs nothing but job
        overhead, and it bounds what any single Runtime program has to hold --
        the unsplit form submitted 97,980 observables in one PUB and died with
        "Program runtime ran out of memory".

        Completed chunks are **kept**: each is stored, keyed by the circuit,
        the labels and the options, as soon as its job returns, and a repeat
        of the same request -- :meth:`~mandacaru.algorithms.Mandacaru.remeasure`
        after a failure -- submits only the chunks still missing.  Every job
        of the submission is on :attr:`jobs`.
        """
        from qiskit.quantum_info import SparsePauliOp

        from .measurement import chunk_labels_by_basis, qwc_partition

        labels = [str(label) for label in labels]
        limit = (self.max_bases_per_job if max_bases_per_job is None
                 else max_bases_per_job)
        key = self._submission_key(n_qubits, occupied, generators, thetas,
                                   labels, limit)
        done = self._completed_chunks.setdefault(key, {})
        chunks = chunk_labels_by_basis(labels, limit)
        values: dict[str, float] = {}
        stds: dict[str, float] = {}
        self.jobs = []
        qc = layout = None
        for index, chunk in enumerate(chunks):
            if index in done:
                values.update(done[index][0])
                stds.update(done[index][1])
                continue
            if qc is None:
                qc, layout, params = self.compiled(n_qubits, occupied,
                                                   generators)
            observables = [SparsePauliOp(label) for label in chunk]
            if layout is not None:
                observables = [o.apply_layout(layout) for o in observables]
            result = self._run_pubs(
                [self._bound_pub(qc, params, observables, thetas)],
                bases=len(qwc_partition(chunk)))
            if self.last_job is not None and self.is_ibm_device:
                self.jobs.append(self.last_job)
            chunk_values = dict(zip(chunk, np.asarray(
                result[0].data.evs, dtype=float).reshape(-1)))
            chunk_stds = dict(zip(chunk, np.asarray(
                result[0].data.stds, dtype=float).reshape(-1)))
            done[index] = (chunk_values, chunk_stds)
            values.update(chunk_values)
            stds.update(chunk_stds)
        # Complete: nothing left to resume.
        self._completed_chunks.pop(key, None)
        return values, stds

    def completed_chunks(self) -> int:
        """Chunks of unfinished ``expectation_values`` requests already
        measured -- what a retry of the same request will not resubmit."""
        return sum(len(done) for done in self._completed_chunks.values())

    def _submission_key(self, n_qubits, occupied, generators, thetas, labels,
                        limit) -> str:
        """A fingerprint of what a submission measures and how."""
        import hashlib

        parts = [str(int(n_qubits)), repr(sorted(int(q) for q in occupied)),
                 repr(np.round(np.asarray(thetas, dtype=float), 12).tolist()),
                 repr([sorted((str(l), complex(c)) for l, c in
                              getattr(g, "terms", {}).items())
                       for g in generators]),
                 repr(labels), repr(limit), self.device_spec, str(self.shots),
                 repr(self.estimator_options), repr(self.physical_qubits)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def energy(self, n_qubits: int, occupied, generators, thetas,
               hamiltonian) -> float:
        """``<psi(thetas)| H |psi(thetas)>`` from the Estimator (see the class)."""
        if not self.shots and self.is_local:
            return super().energy(n_qubits, occupied, generators, thetas,
                                  hamiltonian)
        return self.energies([(n_qubits, occupied, generators, thetas,
                               hamiltonian)])[0]


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
        #: QWC groups per Hamiltonian, so an optimization partitions its
        #: unchanged Hamiltonian once rather than at every evaluation.
        self._groups: dict = {}

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

    def jobs_per_energy(self, hamiltonian) -> int:
        if not self.shots:
            return 1
        from .measurement import qubit_wise_commuting_groups
        key = tuple(sorted((label, complex(c)) for label, c
                           in hamiltonian.terms.items()))
        if key not in self._groups:
            self._groups[key] = qubit_wise_commuting_groups(hamiltonian)
        return max(len(self._groups[key][0]), 1)

    def shots_per_energy(self, hamiltonian) -> int:
        return self.jobs_per_energy(hamiltonian) * max(self.shots, 0)

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
                "Use provider.energy(...) for the shot-based expectation value, "
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

    def energy(self, n_qubits: int, occupied, generators, thetas,
               hamiltonian) -> float:
        r"""Estimate ``<H>``, from shots when ``shots > 0``.

        With ``shots = 0`` this defers to the exact state-vector path.  With
        ``shots > 0`` it runs the QPU-compatible protocol: partition ``H`` into
        qubit-wise-commuting groups, submit one measurement circuit per group,
        and combine the bit-string counts.  The number of submitted tasks is the
        number of groups, not the number of Pauli terms.
        """
        if not self.shots:
            return super().energy(n_qubits, occupied, generators, thetas,
                                  hamiltonian)
        from .measurement import (energy_from_group_counts,
                                  qubit_wise_commuting_groups)

        key = tuple(sorted((label, complex(c)) for label, c
                           in hamiltonian.terms.items()))
        if key not in self._groups:
            self._groups[key] = qubit_wise_commuting_groups(hamiltonian)
        groups, identity = self._groups[key]
        # Every group is queued before any is awaited: the waits overlap
        # instead of adding up, one per group.
        tasks = [self._submit(self.build(n_qubits, occupied, generators,
                                         thetas, measure_basis=basis),
                              self.shots)
                 for basis, _payload in groups]
        counts_per_group = [self._counts(task.result()) for task in tasks]
        return energy_from_group_counts(groups, identity, counts_per_group)

    @staticmethod
    def _counts(result) -> dict:
        """Braket measurement counts as ``{bitstring: n}`` (qubit 0 leftmost).

        Braket's ``measurement_counts`` keys are already ordered with qubit 0
        first, matching Mandacaru's convention.
        """
        return {str(bits): int(n)
                for bits, n in result.measurement_counts.items()}

    def measurement_groups(self, hamiltonian):
        """QWC measurement groups for ``hamiltonian`` (one circuit each).

        Exposed for planning a hardware run: ``len(...)`` is the number of
        quantum tasks a single energy evaluation costs on a QPU.
        """
        from .measurement import qubit_wise_commuting_groups
        return qubit_wise_commuting_groups(hamiltonian)[0]


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

_CACHE: dict[str, CircuitProvider] = {}


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


def build_provider(name: str = "qiskit", **options) -> CircuitProvider:
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
    if options:
        return _PROVIDER_CLASSES[canon](**options)
    provider = _CACHE.get(canon)
    if provider is None:
        provider = _CACHE[canon] = _PROVIDER_CLASSES[canon]()
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
