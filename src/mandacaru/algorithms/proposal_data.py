# -*- coding: utf-8 -*-
# file: algorithms/proposal_data.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""The recorded architecture edits a learned proposal is trained on.

A Markov-chain ansatz search (MCAS-VQE or VALQA) run with ``record=DIR`` appends
one row per proposal to ``DIR/edits.jsonl`` -- rejected proposals included, as
they are the negative examples -- and stores the problem it searched once, in
``DIR/problems/<key>.npz``:

* the qubit Hamiltonian the chain optimized (after any Z2 taper), as its
  non-identity Pauli strings and real coefficients in Hartree;
* the reference determinant (the qubits set to one);
* the operator pool, each generator as its Pauli strings and the magnitudes
  of their coefficients.

**The shared store.** ``MANDACARU_PROPOSAL_DATA`` names one directory that
every chain records into by default and that training reads by default, so
data accumulates across runs; set it with ``mandacaru --set-proposal-data
DIR``.  ``record=DIR`` sends one run elsewhere and ``record=False`` records
nothing.  The trained model lives beside the edits, as
``<store>/proposal_model.npz`` (:data:`MODEL_FILE`), which is where VALQA
looks for it when no ``proposal_model`` is named.

``key`` is a hash of all three, so runs on the same problem share one file,
and a row names the problem it came from.  Every row also carries what was
known *before* the proposal was evaluated -- the source architecture, its
energy and the pool gradients there -- and what was measured after, the
energy change; the model version and the probability it was drawn with are
kept so the policy that collected the data can be traced.  Energies are in
Hartree whatever the run's unit.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Version of the row layout written to ``edits.jsonl``.
SCHEMA = 1

#: The file holding one JSON row per proposal.
EDITS_FILE = "edits.jsonl"

#: The directory holding one ``<key>.npz`` per searched problem.
PROBLEMS_DIR = "problems"

#: The environment variable naming the shared store.
PROPOSAL_DATA_VARIABLE = "MANDACARU_PROPOSAL_DATA"

#: The command that records :data:`PROPOSAL_DATA_VARIABLE` in the shell.
SET_FLAG = "--set-proposal-data"

#: The trained model's file name inside a store.
MODEL_FILE = "proposal_model.npz"

#: Integer code of each Pauli letter; ``I`` is ``0``.
PAULI_CODES = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def save_npz_atomically(path, **arrays) -> Path:
    """Write a compressed ``.npz`` so that a reader never sees it half written.

    The arrays go to a temporary file in the same directory, which then
    replaces ``path`` in one rename: a concurrent reader -- another run
    loading the store's model or a problem -- gets the old file or the new
    one, never a torn archive.
    """
    path = Path(path)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        # A temporary file is private (0600); the store is not.
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path


def _codes(label: str) -> np.ndarray:
    return np.frombuffer(label.encode("ascii"), dtype=np.uint8).copy()


_LETTER_TO_CODE = np.zeros(256, dtype=np.uint8)
for _letter, _code in PAULI_CODES.items():
    _LETTER_TO_CODE[ord(_letter)] = _code


def _pauli_codes(labels, n_qubits: int) -> np.ndarray:
    """``(len(labels), n_qubits)`` array of Pauli codes."""
    if not labels:
        return np.zeros((0, n_qubits), dtype=np.uint8)
    return _LETTER_TO_CODE[np.stack([_codes(label) for label in labels])]


@dataclass(frozen=True)
class ProblemRecord:
    """The Hamiltonian, reference and pool one chain searched.

    ``terms`` holds the non-identity Pauli strings of the Hamiltonian as codes
    (:data:`PAULI_CODES`), ``coefficients`` their real coefficients in
    Hartree, ``constant`` the identity coefficient.  The pool generator
    ``mu`` is the strings ``pool_strings[pool_offsets[mu]:pool_offsets[mu+1]]``
    with coefficient magnitudes ``pool_weights`` over the same range.
    """

    key: str
    n_qubits: int
    terms: np.ndarray
    coefficients: np.ndarray
    constant: float
    reference: np.ndarray
    pool_labels: tuple
    pool_strings: np.ndarray
    pool_weights: np.ndarray
    pool_offsets: np.ndarray

    @property
    def pool_size(self) -> int:
        return len(self.pool_labels)

    @classmethod
    def from_problem(cls, hamiltonian, reference_qubits, pool_ops
                     ) -> "ProblemRecord":
        """Record a run's qubit Hamiltonian, reference and pool."""
        n = int(hamiltonian.num_qubits)
        identity = "I" * n
        labels = sorted(label for label in hamiltonian.terms
                        if label != identity)
        coefficients = np.array([hamiltonian.terms[label].real
                                 for label in labels], dtype=float)
        constant = float(hamiltonian.terms.get(identity, 0.0).real)
        reference = np.zeros(n, dtype=np.uint8)
        reference[list(reference_qubits)] = 1
        strings, weights, offsets = [], [], [0]
        for op in pool_ops:
            generator = op.generator
            items = sorted((label, abs(coefficient))
                           for label, coefficient in generator.terms.items()
                           if label != identity and abs(coefficient) > 0.0)
            strings += [label for label, _ in items]
            weights += [weight for _, weight in items]
            offsets.append(len(strings))
        record = dict(
            n_qubits=n, terms=_pauli_codes(labels, n),
            coefficients=coefficients, constant=constant,
            reference=reference,
            pool_labels=tuple(op.label for op in pool_ops),
            pool_strings=_pauli_codes(strings, n),
            pool_weights=np.asarray(weights, dtype=float),
            pool_offsets=np.asarray(offsets, dtype=np.int64))
        return cls(key=_problem_key(record), **record)

    def save(self, directory) -> Path:
        """Write ``<directory>/problems/<key>.npz`` unless it exists."""
        folder = Path(directory) / PROBLEMS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.key}.npz"
        if not path.exists():
            save_npz_atomically(
                path, n_qubits=self.n_qubits, terms=self.terms,
                coefficients=self.coefficients, constant=self.constant,
                reference=self.reference,
                pool_labels=np.array(self.pool_labels, dtype=str),
                pool_strings=self.pool_strings,
                pool_weights=self.pool_weights,
                pool_offsets=self.pool_offsets)
        return path

    @classmethod
    def load(cls, path) -> "ProblemRecord":
        with np.load(path, allow_pickle=False) as data:
            return cls(key=Path(path).stem, n_qubits=int(data["n_qubits"]),
                       terms=data["terms"],
                       coefficients=data["coefficients"],
                       constant=float(data["constant"]),
                       reference=data["reference"],
                       pool_labels=tuple(str(s) for s in data["pool_labels"]),
                       pool_strings=data["pool_strings"],
                       pool_weights=data["pool_weights"],
                       pool_offsets=data["pool_offsets"])


def _problem_key(record: dict) -> str:
    """A hash of the Hamiltonian, reference and pool, 16 hex digits.

    Coefficients are rounded to 1e-10 Hartree so that the same problem built
    twice hashes the same despite round-off in the last digits.
    """
    digest = hashlib.sha256()
    digest.update(str(record["n_qubits"]).encode())
    for name in ("terms", "reference", "pool_strings", "pool_offsets"):
        digest.update(np.ascontiguousarray(record[name]).tobytes())
    for name in ("coefficients", "pool_weights"):
        digest.update(np.round(record[name], 10).tobytes())
    digest.update(np.round(record["constant"], 10).tobytes())
    digest.update("\n".join(record["pool_labels"]).encode())
    return digest.hexdigest()[:16]


def shared_store() -> Path | None:
    """The directory :data:`PROPOSAL_DATA_VARIABLE` names, or ``None``."""
    value = os.environ.get(PROPOSAL_DATA_VARIABLE, "").strip()
    return Path(value).expanduser() if value else None


def data_directory(directory=None) -> Path:
    """``directory``, or the shared store when it is ``None``."""
    if directory is not None:
        return Path(directory).expanduser()
    store = shared_store()
    if store is None:
        raise ValueError(
            f"no edit store named and {PROPOSAL_DATA_VARIABLE} is unset; "
            f"pass a directory or run `mandacaru {SET_FLAG} DIR`")
    return store


def resolve_record(record) -> Path | None:
    """The directory a chain records into, from its ``record`` option.

    ``None`` (default): the shared store when :data:`PROPOSAL_DATA_VARIABLE`
    is set, else nothing.  ``True``: the shared store, which must be set.
    ``False``: nothing.  A path: that directory.
    """
    if record is False:
        return None
    if record is None:
        store = shared_store()
        return None if store is None else check_record_directory(store)
    if record is True:
        return check_record_directory(data_directory())
    if isinstance(record, (bool, int, float)):
        raise ValueError(f"record must be a directory, True, False or None, "
                         f"got {record!r}")
    return check_record_directory(record)


def check_record_directory(directory) -> Path:
    """Refuse a ``record=`` path that cannot become the edit store."""
    path = Path(directory).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError(f"record={str(directory)!r} is a file; it must name a "
                         f"directory for {EDITS_FILE} and {PROBLEMS_DIR}/")
    parent = path if path.exists() else path.parent
    if not parent.exists():
        raise ValueError(f"record={str(directory)!r}: the directory "
                         f"{str(parent)!r} does not exist")
    return path


class EditRecorder:
    """Append a chain's proposals to ``<directory>/edits.jsonl``.

    Each row is written as the step completes, so an interrupted run keeps
    what it measured.  A row is one ``write`` on a descriptor opened for
    appending, so runs recording into the same store at the same time do not
    interleave inside a row.  A file whose last row was cut short by an
    interrupted run is continued on a new line; :func:`load_edits` skips the
    cut row.
    """

    def __init__(self, directory, problem: ProblemRecord, run: dict):
        self.directory = check_record_directory(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        problem.save(self.directory)
        self.problem = problem
        self.run = {"schema": SCHEMA, "problem": problem.key, **run}
        path = self.directory / EDITS_FILE
        self._fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                           0o644)
        self._separator = _ends_inside_a_row(path)
        self.rows = 0

    def write(self, row: dict) -> None:
        if self._fd is None:
            raise ValueError("the edit recorder is closed")
        line = json.dumps({**self.run, **row}, separators=(",", ":")) + "\n"
        data = (b"\n" if self._separator else b"") + line.encode("utf-8")
        while data:
            data = data[os.write(self._fd, data):]
        self._separator = False
        self.rows += 1

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _ends_inside_a_row(path: Path) -> bool:
    """Whether ``path`` is non-empty and does not end with a newline."""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return False
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


def load_edits(directory=None) -> tuple[list[dict], dict[str, ProblemRecord]]:
    """Every recorded row and the problems they refer to (``directory``
    defaults to the shared store).

    Rows whose problem file is missing are refused rather than dropped: the
    store would otherwise shrink silently.  A row an interrupted run cut short
    is not valid JSON; it is skipped with a warning naming its line.
    """
    directory = data_directory(directory)
    edits = directory / EDITS_FILE
    if not edits.is_file():
        raise FileNotFoundError(f"no {EDITS_FILE} in {str(directory)!r}: "
                                f"record a chain with record=DIR first")
    rows, cut = [], []
    with open(edits, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                cut.append(number)
                continue
            if row.get("schema") != SCHEMA:
                raise ValueError(f"{edits}:{number}: row schema "
                                 f"{row.get('schema')!r}, expected {SCHEMA}")
            rows.append(row)
    if cut:
        warnings.warn(f"{edits}: skipped {len(cut)} row(s) cut short by an "
                      f"interrupted run, at line(s) "
                      f"{', '.join(map(str, cut))}", RuntimeWarning,
                      stacklevel=2)
    problems = {}
    for key in sorted({row["problem"] for row in rows}):
        path = directory / PROBLEMS_DIR / f"{key}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"rows refer to problem {key}, but "
                                    f"{path} is missing")
        problems[key] = ProblemRecord.load(path)
    return rows, problems
