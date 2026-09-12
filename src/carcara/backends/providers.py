# -*- coding: utf-8 -*-
# file: backends/providers.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Circuit-construction / execution providers: Qiskit, Braket and Cirq.

Carcará's ansätze are products of exponentials of **anti-Hermitian generators**,
:math:`|\psi(\vec\theta)\rangle = \prod_k e^{\theta_k A_k}\,|\mathrm{HF}\rangle`.
Each generator is a qubit :class:`~carcara.core.mapping.PauliSum` whose terms are
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
    least significant bit of the state-vector index) while Carcará -- like Braket
    and Cirq -- puts qubit 0 in the *most* significant position, so the Qiskit
    provider lays Carcará qubit ``k`` on Qiskit wire ``n-1-k``.  The gate counts
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
    ``c_j``, so ``exp(theta A) = prod_j exp(i theta c_j P_j)``.  Identity strings
    contribute only a global phase and are dropped.

    Raises
    ------
    ValueError
        If any coefficient has a non-negligible real part (the operator is not
        anti-Hermitian and does not generate a unitary for real ``theta``).
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
    return out


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
    """Carcará qubit indices set to ``|1>`` in basis state ``index`` (qubit 0 = MSB)."""
    return [k for k in range(n_qubits)
            if (index >> (n_qubits - 1 - k)) & 1]


# --------------------------------------------------------------------------- #
# Provider interface.
# --------------------------------------------------------------------------- #

class CircuitProvider(ABC):
    """Builds, executes and profiles a Carcará ansatz on one quantum SDK.

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
        """Execute the circuit and return the state vector in Carcará's ordering.

        Carcará's convention is qubit 0 = most significant bit of the amplitude
        index (the leftmost Kronecker factor), matching
        :meth:`~carcara.core.mapping.PauliSum.to_matrix`.
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

    def energy(self, n_qubits: int, occupied, generators, thetas,
               hamiltonian) -> float:
        r"""Estimate ``<psi(thetas)| H |psi(thetas)>`` on this backend.

        The default implementation goes through :meth:`statevector`, which is
        exact but **simulator-only**.  Providers that can measure shots override
        this with the hardware-compatible path (see
        :mod:`carcara.backends.measurement`).
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
    """

    name = "qiskit"
    supports_shots = True

    LOCAL_DEVICES = ("statevector", "aer_simulator", "aer", "simulator", "local")
    LEAST_BUSY = ("ibm-quantum", "ibm", "ibmq", "ibm_quantum")
    FAKE_PREFIX = "fake_"

    def __init__(self, device: str = "statevector", shots: int = 0,
                 instance: str | None = None, token: str | None = None,
                 channel: str | None = None, optimization_level: int = 3,
                 estimator_options: dict | None = None):
        self.device_spec = str(device).strip()
        self.shots = int(shots)
        self.instance = instance
        self.token = token
        self.channel = channel
        self.optimization_level = int(optimization_level)
        self.estimator_options = (None if estimator_options is None
                                  else dict(estimator_options))
        self._backend = None
        self._estimator = None
        self.last_job = None

    def __repr__(self) -> str:
        return f"QiskitProvider(device={self.device_spec!r}, shots={self.shots})"

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
        if name in self.LEAST_BUSY:
            return service.least_busy(operational=True, simulator=False)
        names = [n.strip() for n in name.split(",") if n.strip()]
        if len(names) == 1:
            return service.backend(names[0])
        # The least busy of the named processors.
        candidates = [service.backend(n) for n in names]
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
                    default_precision=self.precision)
            elif self.is_fake_device:
                # Synchronous, in the calling thread (the Runtime primitive's
                # local mode runs on a worker thread, and circuits freed
                # across threads can deadlock Qiskit's allocator at GC).
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
        # Qiskit is little-endian: Carcará qubit k lives on wire n-1-k so the
        # simulated amplitude ordering matches Carcará's without a permutation.
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
        from qiskit import transpile

        qc = self.build(n_qubits, occupied, generators,
                        np.ones(len(generators)))
        compiled = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
        counts = compiled.count_ops()
        return {"cnot_count": int(counts.get("cx", 0)),
                "depth": int(compiled.depth()),
                "num_1q_gates": int(counts.get("u", 0)),
                "total_gates": int(sum(counts.values()))}

    # -- Estimator energies (the hardware path) ---------------------------- #

    @staticmethod
    def observable(hamiltonian):
        """The Hamiltonian as a ``SparsePauliOp``.

        Carcará labels list qubit 0 first and Qiskit labels list the highest
        wire first; with qubit ``k`` on wire ``n-1-k`` the strings coincide.
        """
        return hamiltonian.to_sparse_pauli_op()

    def pub(self, n_qubits: int, occupied, generators, thetas, hamiltonian):
        """One Estimator PUB ``(isa_circuit, observable)`` for this target."""
        qc = self.build(n_qubits, occupied, generators, thetas)
        observable = self.observable(hamiltonian)
        backend = self.backend()
        if backend is None:
            return (qc, observable)
        from qiskit import transpile
        isa = transpile(qc, backend=backend,
                        optimization_level=self.optimization_level)
        return (isa, observable.apply_layout(isa.layout))

    def energies(self, problems) -> list[float]:
        """Expectation values of several ``(n_qubits, occupied, generators,
        thetas, hamiltonian)`` problems, submitted as **one** Estimator job."""
        pubs = [self.pub(*problem) for problem in problems]
        kwargs = {} if self.is_local else {"precision": self.precision}
        if not self.is_local and self.shots <= 0:
            raise ValueError(f"{self!r}: a processor needs shots > 0")
        job = self.estimator().run(pubs, **kwargs)
        self.last_job = job
        result = job.result()
        return [float(np.asarray(result[i].data.evs).reshape(-1)[0])
                for i in range(len(pubs))]

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
    same convention as Carcará, so no wire permutation is needed.

    Two execution modes, chosen by ``shots``:

    * ``shots = 0`` (default) -- ask for the ``StateVector`` result type and get
      the exact amplitudes.  This is **simulator-only**: Braket rejects
      ``StateVector`` whenever ``shots > 0``, and every QPU requires
      ``shots > 0``.
    * ``shots > 0`` -- the **hardware path**.  The Hamiltonian is partitioned
      into qubit-wise-commuting groups (:mod:`carcara.backends.measurement`), one
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
                 s3_folder=None, poll_timeout_seconds: float = 5 * 24 * 60 * 60):
        self.device_spec = str(device)
        self.shots = int(shots)
        self.s3_folder = s3_folder
        self.poll_timeout_seconds = float(poll_timeout_seconds)
        self._device = None

    def __repr__(self) -> str:
        return (f"BraketProvider(device={self.device_spec!r}, "
                f"shots={self.shots})")

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
        device = self.device()
        kwargs = {"shots": int(shots)}
        if self.is_aws_device:
            kwargs["poll_timeout_seconds"] = self.poll_timeout_seconds
            if self.s3_folder is not None:
                kwargs["s3_destination_folder"] = tuple(self.s3_folder)
        return device.run(circuit, **kwargs).result()

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

        groups, identity = qubit_wise_commuting_groups(hamiltonian)
        counts_per_group = []
        for basis, _payload in groups:
            circuit = self.build(n_qubits, occupied, generators, thetas,
                                 measure_basis=basis)
            result = self._run(circuit, self.shots)
            counts_per_group.append(self._counts(result))
        return energy_from_group_counts(groups, identity, counts_per_group)

    @staticmethod
    def _counts(result) -> dict:
        """Braket measurement counts as ``{bitstring: n}`` (qubit 0 leftmost).

        Braket's ``measurement_counts`` keys are already ordered with qubit 0
        first, matching Carcará's convention.
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
        # significant amplitude bit, matching Carcará.
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
