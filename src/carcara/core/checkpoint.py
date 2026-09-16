# -*- coding: utf-8 -*-
# file: core/checkpoint.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Wavefunction checkpoints: a variational state on disk, for any algorithm.

Every variational state Carcará prepares has the same shape,

.. math::

    |\Psi\rangle = \prod_k e^{\theta_k A_k}\,|\text{ref}\rangle ,

a computational-basis reference determinant followed by an ordered product of
exponentials of anti-Hermitian qubit generators :math:`A_k`.  ADAPT-VQE grows
that list one generator at a time, UCCSD-VQE fixes it up front, and every
circuit provider already prepares states from exactly this description.  A
:class:`WavefunctionCheckpoint` is that description written down -- the
register, the reference, the generators as Pauli sums, the angles -- plus the
solver's progress, so a run can be **resumed** where it stopped, and the
optional qubit Hamiltonian, so the state can be **handed to another algorithm**
(quantum phase estimation, :mod:`carcara.algorithms.qpe`) with nothing but the
file.

The file is JSON: no dependency, diffable, and a generator is a few dozen Pauli
strings, so the whole thing is kilobytes.  Writes are atomic (a temporary file
renamed into place), because a checkpoint written *during* a run is only useful
if an interruption mid-write cannot corrupt it.

Two views of the same data:

* :meth:`WavefunctionCheckpoint.problem` -- the
  ``(n_qubits, reference_qubits, generators, parameters, hamiltonian)`` tuple
  the circuit providers and the QPE driver consume;
* :meth:`WavefunctionCheckpoint.state_vector` -- the amplitudes, prepared here
  without any driver (sparse closed-form exponentials).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .mapping import PauliSum
from .serialization import MAX_FILE_QUBITS

#: Identifies a Carcará wavefunction checkpoint file.
FORMAT_TAG = "carcara-wavefunction-checkpoint"
#: Current schema version.
FORMAT_VERSION = 1
#: Extension a checkpoint path defaults to.
FILE_EXTENSION = ".json"


# --------------------------------------------------------------------------- #
# Preparing the state from the description.
# --------------------------------------------------------------------------- #

def reference_vector(n_qubits: int, reference_qubits) -> np.ndarray:
    """The computational-basis state with ``reference_qubits`` set to ``|1>``.

    Qubit 0 is the most significant bit, Carcará's convention throughout.
    """
    n = int(n_qubits)
    index = 0
    for k in reference_qubits:
        k = int(k)
        if not 0 <= k < n:
            raise ValueError(f"reference qubit {k} outside a {n}-qubit register")
        index |= 1 << (n - 1 - k)
    vec = np.zeros(2 ** n, dtype=complex)
    vec[index] = 1.0
    return vec


def apply_exponential(generator: PauliSum, angle: float, vector) -> np.ndarray:
    r"""``exp(angle * A) @ vector`` for an anti-Hermitian generator ``A``.

    The excitation generators satisfy :math:`A^3 = -A`, for which the
    exponential has the closed form
    :math:`e^{\theta A} = 1 + \sin\theta\,A + (1-\cos\theta)\,A^2` -- two sparse
    matrix-vector products.  A generator that fails the identity goes through
    :func:`scipy.sparse.linalg.expm_multiply` instead, so the result is exact
    either way.
    """
    from scipy.sparse.linalg import expm_multiply

    vector = np.asarray(vector, dtype=complex)
    if generator.num_qubits and vector.shape[0] != 2 ** generator.num_qubits:
        raise ValueError(
            f"a {generator.num_qubits}-qubit generator cannot act on a vector "
            f"of {vector.shape[0]} amplitudes")
    A = generator.to_sparse_matrix()
    A2 = (A @ A).tocsr()
    residual = A @ A2 + A
    data = residual.tocoo().data
    scale = max(1.0, float(np.max(np.abs(A.tocoo().data))) if A.nnz else 1.0)
    if data.size == 0 or float(np.max(np.abs(data))) < 1e-9 * scale:
        return (vector + np.sin(angle) * (A @ vector)
                + (1.0 - np.cos(angle)) * (A2 @ vector))
    return expm_multiply(float(angle) * A, vector)


def prepare_state(n_qubits: int, reference_qubits, generators, parameters
                  ) -> np.ndarray:
    r"""``prod_k exp(theta_k A_k) |ref>`` as a full-register state vector.

    Independent of every driver and ansatz class: this is the state a
    checkpoint *means*, and what a circuit provider prepares on hardware.
    """
    parameters = np.asarray(parameters, dtype=float).ravel()
    generators = list(generators)
    if parameters.size != len(generators):
        raise ValueError(f"{len(generators)} generators but {parameters.size} "
                         "parameters")
    psi = reference_vector(n_qubits, reference_qubits)
    for generator, angle in zip(generators, parameters):
        psi = apply_exponential(generator, angle, psi)
    return psi


# --------------------------------------------------------------------------- #
# The record.
# --------------------------------------------------------------------------- #

def _pauli_to_payload(operator: PauliSum | None):
    if operator is None:
        return None
    simplified = operator.simplify()
    return {"num_qubits": int(operator.num_qubits),
            "terms": [[label, complex(c).real, complex(c).imag]
                      for label, c in sorted(simplified.terms.items())]}


def _pauli_from_payload(payload) -> PauliSum | None:
    if payload is None:
        return None
    n = int(payload["num_qubits"])
    terms = {}
    for label, real, imag in payload["terms"]:
        if len(label) != n:
            raise ValueError(f"Pauli string {label!r} is not {n} characters")
        terms[label] = terms.get(label, 0j) + complex(float(real), float(imag))
    return PauliSum(terms, num_qubits=n)


@dataclass
class WavefunctionCheckpoint:
    """A variational wavefunction and where its optimization stood.

    Attributes
    ----------
    n_qubits : int
        Register width (after any two-qubit reduction).
    reference_qubits : list of int
        Qubits set to ``|1>`` in the reference determinant, in register terms
        (i.e. *after* the fermion-to-qubit mapping and any tapering).
    generators : list of PauliSum
        Anti-Hermitian generators :math:`A_k`, in the order they are applied.
    parameters : ndarray
        The angles :math:`\\theta_k`.
    labels, kinds : list of str
        Human-readable names and pool categories of the generators (the
        ADAPT-VQE operator sequence).
    mapping, two_qubit_reduction, num_particles, n_spatial_orbitals, occupied_orbitals
        The fermionic problem the register encodes -- what lets a solver
        rebuild its pool / ansatz around the state.
    energy : float or None
        ``<H>`` at these parameters, in Hartree.
    hamiltonian : PauliSum or None
        The qubit Hamiltonian, when the writer chose to include it.  With it a
        file is a complete input for another algorithm.
    method : str
        The algorithm that wrote the state (``"adapt-vqe"``, ``"vqe"``).
    status : dict
        Solver progress needed to resume: ``complete``, ``converged``,
        ``iteration``, ``max_gradient``, ``num_evaluations``, the per-step
        history ... -- all energies in Hartree.
    metadata : dict
        Provenance: version, timestamp, basis, geometry, notes.
    """

    n_qubits: int
    reference_qubits: list[int]
    generators: list[PauliSum]
    parameters: np.ndarray
    labels: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    mapping: str = "jordan_wigner"
    two_qubit_reduction: bool = False
    num_particles: tuple[int, int] | None = None
    n_spatial_orbitals: int | None = None
    occupied_orbitals: list[int] | None = None
    energy: float | None = None
    hamiltonian: PauliSum | None = None
    method: str = ""
    status: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_qubits = int(self.n_qubits)
        self.reference_qubits = [int(k) for k in self.reference_qubits]
        self.parameters = np.asarray(self.parameters, dtype=float).ravel()
        self.generators = list(self.generators)
        if self.parameters.size != len(self.generators):
            raise ValueError(
                f"{len(self.generators)} generators but "
                f"{self.parameters.size} parameters")
        for g in self.generators:
            if g.num_qubits and g.num_qubits != self.n_qubits:
                raise ValueError(
                    f"a {g.num_qubits}-qubit generator in a {self.n_qubits}-"
                    "qubit checkpoint")
        if self.hamiltonian is not None and self.hamiltonian.num_qubits \
                and self.hamiltonian.num_qubits != self.n_qubits:
            raise ValueError(
                f"a {self.hamiltonian.num_qubits}-qubit Hamiltonian in a "
                f"{self.n_qubits}-qubit checkpoint")
        if not self.labels:
            self.labels = [f"A{k}" for k in range(len(self.generators))]
        if not self.kinds:
            self.kinds = [""] * len(self.generators)
        if len(self.labels) != len(self.generators) \
                or len(self.kinds) != len(self.generators):
            raise ValueError("labels / kinds must match the generators")
        if self.num_particles is not None:
            self.num_particles = (int(self.num_particles[0]),
                                  int(self.num_particles[1]))

    # -- the two consumers' views ---------------------------------------- #

    @property
    def num_parameters(self) -> int:
        return len(self.generators)

    def problem(self, hamiltonian: PauliSum | None = None):
        """``(n_qubits, reference_qubits, generators, parameters, hamiltonian)``.

        The tuple every :class:`~carcara.backends.providers.CircuitProvider`
        ``energy`` / ``build`` takes and :class:`~carcara.algorithms.qpe.QuantumPhaseEstimation`
        starts from.  ``hamiltonian`` overrides the stored one.
        """
        h = self.hamiltonian if hamiltonian is None else hamiltonian
        return (self.n_qubits, list(self.reference_qubits),
                list(self.generators), self.parameters.copy(), h)

    def state_vector(self) -> np.ndarray:
        """The prepared state, ``prod_k exp(theta_k A_k) |ref>``."""
        return prepare_state(self.n_qubits, self.reference_qubits,
                             self.generators, self.parameters)

    def circuit(self, provider=None):
        """The state-preparation circuit on an SDK (default: Qiskit).

        Every provider builds circuits from the same description, so this is
        ``provider.build(n_qubits, reference_qubits, generators, parameters)``.
        """
        if provider is None:
            from ..backends.providers import QiskitProvider
            provider = QiskitProvider()
        return provider.build(self.n_qubits, self.reference_qubits,
                              self.generators, self.parameters)

    def expectation(self, hamiltonian: PauliSum | None = None) -> float:
        """``<Psi|H|Psi>`` (Hartree) from the prepared state vector."""
        h = self.hamiltonian if hamiltonian is None else hamiltonian
        if h is None:
            raise ValueError("no Hamiltonian stored; pass one")
        psi = self.state_vector()
        return float(np.real(np.vdot(psi, h.to_sparse_matrix() @ psi)))

    # -- serialization --------------------------------------------------- #

    def to_payload(self) -> dict:
        """The JSON document (plain Python types)."""
        return {
            "format": FORMAT_TAG,
            "version": FORMAT_VERSION,
            "written": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"),
            "energy_unit": "Ha",
            "n_qubits": self.n_qubits,
            "reference_qubits": list(self.reference_qubits),
            "mapping": self.mapping,
            "two_qubit_reduction": bool(self.two_qubit_reduction),
            "num_particles": (None if self.num_particles is None
                              else list(self.num_particles)),
            "n_spatial_orbitals": self.n_spatial_orbitals,
            "occupied_orbitals": (None if self.occupied_orbitals is None
                                  else [int(k) for k in self.occupied_orbitals]),
            "method": self.method,
            "energy": None if self.energy is None else float(self.energy),
            "operators": [{"label": label, "kind": kind,
                           "parameter": float(theta),
                           "generator": _pauli_to_payload(g)}
                          for label, kind, theta, g in zip(
                              self.labels, self.kinds, self.parameters,
                              self.generators)],
            "hamiltonian": _pauli_to_payload(self.hamiltonian),
            "status": _jsonable(self.status),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "WavefunctionCheckpoint":
        if not isinstance(payload, dict) or payload.get("format") != FORMAT_TAG:
            raise ValueError("not a Carcará wavefunction checkpoint")
        version = int(payload.get("version", 0))
        if version > FORMAT_VERSION:
            raise ValueError(f"checkpoint version {version} is newer than this "
                             f"build understands ({FORMAT_VERSION})")
        operators = payload.get("operators", [])
        particles = payload.get("num_particles")
        return cls(
            n_qubits=int(payload["n_qubits"]),
            reference_qubits=[int(k) for k in payload["reference_qubits"]],
            generators=[_pauli_from_payload(op["generator"]) for op in operators],
            parameters=np.array([float(op["parameter"]) for op in operators]),
            labels=[str(op.get("label", f"A{k}"))
                    for k, op in enumerate(operators)],
            kinds=[str(op.get("kind", "")) for op in operators],
            mapping=str(payload.get("mapping", "jordan_wigner")),
            two_qubit_reduction=bool(payload.get("two_qubit_reduction", False)),
            num_particles=None if particles is None else tuple(particles),
            n_spatial_orbitals=payload.get("n_spatial_orbitals"),
            occupied_orbitals=payload.get("occupied_orbitals"),
            energy=payload.get("energy"),
            hamiltonian=_pauli_from_payload(payload.get("hamiltonian")),
            method=str(payload.get("method", "")),
            status=dict(payload.get("status") or {}),
            metadata=dict(payload.get("metadata") or {}))

    def save(self, path) -> str:
        """Write the checkpoint atomically; return the path written.

        The document is written to a temporary file in the same directory and
        renamed over ``path``, so a reader never sees a half-written file --
        the property a checkpoint taken *during* a run depends on.
        """
        path = os.fspath(path)
        if self.n_qubits > MAX_FILE_QUBITS:
            raise ValueError(
                f"refusing to write a {self.n_qubits}-qubit checkpoint: "
                f"Pauli-string files are limited to {MAX_FILE_QUBITS} qubits")
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp",
                                   dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_payload(), fh, indent=1)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path) -> "WavefunctionCheckpoint":
        """Read a checkpoint written by :meth:`save`."""
        path = os.fspath(path)
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        try:
            return cls.from_payload(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path!r}: {error}") from error

    # -- presentation ---------------------------------------------------- #

    def summary(self) -> str:
        status = self.status
        lines = [f"Wavefunction checkpoint ({self.method or 'unknown method'}): "
                 f"{self.n_qubits} qubits, {self.num_parameters} generators",
                 f"  mapping: {self.mapping}"
                 + (" (two-qubit reduction)" if self.two_qubit_reduction else "")
                 + f", reference |1> on qubits {self.reference_qubits}"]
        if self.energy is not None:
            lines.append(f"  energy: {self.energy:+.10f} Ha")
        if status:
            state = ("complete" if status.get("complete")
                     else "in progress")
            lines.append(f"  status: {state}"
                         + (", converged" if status.get("converged") else "")
                         + (f", iteration {status['iteration']}"
                            if "iteration" in status else ""))
        lines.append("  Hamiltonian: "
                     + (f"{len(self.hamiltonian.simplify().terms)} Pauli terms"
                        if self.hamiltonian is not None else "not stored"))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"WavefunctionCheckpoint(n_qubits={self.n_qubits}, "
                f"generators={self.num_parameters}, method={self.method!r}, "
                f"energy={self.energy})")


def _jsonable(value):
    """Recursively coerce NumPy scalars / arrays into JSON-native types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_checkpoint(path) -> WavefunctionCheckpoint:
    """Read a wavefunction checkpoint (module-level spelling of :meth:`load`)."""
    return WavefunctionCheckpoint.load(path)
