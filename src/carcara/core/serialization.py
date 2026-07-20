# -*- coding: utf-8 -*-
# file: core/serialization.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""On-disk serialization of the qubit Hamiltonian (Apache Parquet or JSON).

Building a molecular Hamiltonian is by far the most expensive stage of a Carcará
run: the real-space one- and two-body integrals scale as :math:`O(N \log N)` /
:math:`O(M^4)` in the grid and basis size, and the fermion-to-qubit mapping then
composes a Pauli sum per fermionic term.  None of that depends on the *algorithm*
being run, so the result is worth caching: a driver can dump the qubit
Hamiltonian once and every later run (a different pool, optimizer, ansatz,
temperature schedule, ...) reloads it in milliseconds, **bypassing the integral
engine and the mapping entirely**.

Two file formats
----------------
``hamiltonian_format="parquet"`` (default)
    A compact, compressed, columnar table -- the right choice for the
    :math:`10^4`--:math:`10^6`-term Hamiltonians of a realistic active space, and
    directly queryable from pandas / Arrow / Spark.
``hamiltonian_format="json"``
    A plain-text, dependency-free document.  Slower and larger, but readable and
    diffable, and it needs **no Parquet engine at all** -- which matters because
    of the Qiskit/pyarrow interaction described under *Engines* below.

:func:`load_hamiltonian` **detects the format automatically**: first from the
file extension, and failing that from the file's own leading bytes (a Parquet
file begins with the ``PAR1`` magic number, a JSON document with ``{``).  So a
cache written either way is loaded with the same call.

Parquet layout
--------------
The file is a table with one row per Pauli term and three columns:

``pauli`` (string)
    The Pauli string; character ``k`` acts on qubit ``k``, matching
    :class:`~carcara.core.mapping.PauliSum`.
``real`` (double)
    Real part of the coefficient.
``imag`` (double)
    Imaginary part of the coefficient.

The problem metadata a driver needs to rebuild its ansatz or pool *without a
geometry* -- the mapping, the electron count and the orbital count -- travels in
the Parquet **key/value metadata**, so a single self-contained file fully
specifies the problem.

JSON layout
-----------
The same content as a single object: ``format`` / ``version`` / ``num_qubits`` /
``mapping`` / ``num_particles`` / ``n_spatial_orbitals`` / ``metadata``, plus
``terms`` as a list of ``[pauli_string, real, imag]`` triples.

Engines
-------
Two interchangeable Parquet engines are supported, selected by ``engine``:
``"fastparquet"`` (the default when installed) and ``"pyarrow"``.  Both read and
write ordinary Parquet, so files are portable between them and to any other
Parquet reader.  ``engine`` is irrelevant to the JSON format, which uses only the
standard library.

.. warning::

   ``fastparquet`` is preferred **on purpose**.  On some platforms -- reproduced
   here on CPython 3.14 with ``qiskit`` 2.5 and ``pyarrow`` 25 -- calling
   ``pyarrow.parquet.write_table`` in a process that has also run Qiskit's
   ``transpile`` crashes the interpreter (both ship their own native runtimes).
   Since Carcará transpiles circuits for gate-count profiling in the same
   process that saves the Hamiltonian, the default engine avoids that
   combination.  Set ``engine="pyarrow"`` explicitly if you prefer Arrow and your
   environment is unaffected -- or side-step Parquet entirely with
   ``hamiltonian_format="json"``, which has no native dependency and cannot be
   affected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .mapping import PauliSum

#: Value of the ``carcara.format`` metadata key identifying these files.
FORMAT_TAG = "carcara-qubit-hamiltonian"
#: Current schema version.
FORMAT_VERSION = 1
#: File formats understood by ``hamiltonian_format=`` / ``format=``.
HAMILTONIAN_FORMATS = ("parquet", "json")
#: Default format.
DEFAULT_FORMAT = "parquet"
#: Extension appended per format when a path is given without one.
FILE_EXTENSIONS = {"parquet": ".parquet", "json": ".json"}
#: Recognized extensions, for format detection.
_EXTENSION_FORMATS = {".parquet": "parquet", ".pq": "parquet", ".json": "json"}
#: Filename stem used when ``save_hamiltonian=True`` is given without a path.
DEFAULT_STEM = "hamiltonian"
#: Filename used when ``save_hamiltonian=True`` with the default format.
DEFAULT_FILENAME = DEFAULT_STEM + FILE_EXTENSIONS[DEFAULT_FORMAT]
#: Backwards-compatible alias for the default format's extension.
FILE_EXTENSION = FILE_EXTENSIONS[DEFAULT_FORMAT]
#: Magic number every Apache Parquet file starts (and ends) with.
PARQUET_MAGIC = b"PAR1"
#: Parquet engines understood by ``engine=``, in preference order.
PARQUET_ENGINES = ("fastparquet", "pyarrow")
#: Column names of the Pauli-term table.
COLUMNS = ("pauli", "real", "imag")


def resolve_format(fmt: str = DEFAULT_FORMAT) -> str:
    """Validate and normalize a ``hamiltonian_format`` / ``format`` spec."""
    key = str(fmt).strip().lower().lstrip(".")
    if key in ("pq",):
        key = "parquet"
    if key not in HAMILTONIAN_FORMATS:
        raise ValueError(
            f"unknown hamiltonian_format {fmt!r}; use one of "
            f"{HAMILTONIAN_FORMATS}")
    return key


def detect_format(path) -> str:
    """Detect a Hamiltonian file's format from its extension, else its content.

    The extension wins when it is one this build knows (``.parquet``, ``.pq``,
    ``.json``).  Otherwise the file's first bytes decide: Parquet begins with the
    ``PAR1`` magic number, a JSON document with ``{`` (after any whitespace or
    byte-order mark).  This is what lets ``load_hamiltonian`` accept either
    format through the same call.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the format cannot be determined.
    """
    path = os.fspath(path)
    extension = os.path.splitext(path)[1].lower()
    known = _EXTENSION_FORMATS.get(extension)
    if known is not None:
        return known
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such Hamiltonian file: {path!r}")
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head.startswith(PARQUET_MAGIC):
        return "parquet"
    if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{"):
        return "json"
    raise ValueError(
        f"cannot determine the format of {path!r}: it has no recognized "
        f"extension ({', '.join(sorted(_EXTENSION_FORMATS))}) and starts with "
        f"neither the Parquet magic number nor '{{'.  Pass format='parquet' or "
        "format='json' explicitly.")


# --------------------------------------------------------------------------- #
# Engine selection.
# --------------------------------------------------------------------------- #

def available_engines() -> tuple[str, ...]:
    """Parquet engines importable in this environment, in preference order.

    .. note::

       This **imports** every candidate to check it.  Importing ``pyarrow`` is
       itself enough to destabilize Qiskit in the environments described in the
       module warning, so :func:`resolve_engine` deliberately does *not* call
       this -- it stops at the first engine that imports.  Use this function for
       diagnostics, not on the hot path.
    """
    found = []
    for name in PARQUET_ENGINES:
        try:
            __import__(name)
        except Exception:
            continue
        found.append(name)
    return tuple(found)


def resolve_engine(engine: str = "auto") -> str:
    """Resolve an ``engine`` spec to a concrete, importable engine name.

    ``"auto"`` returns the **first** of :data:`PARQUET_ENGINES` that imports and
    stops there -- it never touches the later ones (see the module warning: even
    importing ``pyarrow`` can destabilize Qiskit, so a run that uses
    ``fastparquet`` must not import it as a side effect).
    """
    if engine == "auto":
        for name in PARQUET_ENGINES:
            try:
                __import__(name)
            except Exception:
                continue
            return name
        raise ImportError(
            "the Hamiltonian cache is stored as Apache Parquet and needs a "
            "Parquet engine; install one with `pip install fastparquet` "
            "(preferred) or `pip install pyarrow`")
    if engine not in PARQUET_ENGINES:
        raise ValueError(
            f"unknown Parquet engine {engine!r}; use one of {PARQUET_ENGINES} "
            "or 'auto'")
    try:
        __import__(engine)
    except ImportError as exc:
        raise ImportError(
            f"Parquet engine {engine!r} is not installed; "
            f"`pip install {engine}`") from exc
    return engine


# --------------------------------------------------------------------------- #
# Record.
# --------------------------------------------------------------------------- #

@dataclass
class HamiltonianRecord:
    """A qubit Hamiltonian plus the problem metadata needed to rebuild a driver.

    ``num_particles`` and ``n_spatial_orbitals`` are what let a loaded run skip
    the geometry entirely: they are exactly the two quantities the ADAPT-VQE pool
    and the UCCSD ansatz are built from.
    """

    hamiltonian: PauliSum
    mapping: str = "jordan_wigner"
    num_particles: tuple[int, int] | None = None
    n_spatial_orbitals: int | None = None
    metadata: dict | None = None

    @property
    def num_qubits(self) -> int:
        return self.hamiltonian.num_qubits

    def __repr__(self) -> str:
        return (f"HamiltonianRecord({len(self.hamiltonian.terms)} terms, "
                f"num_qubits={self.num_qubits}, mapping={self.mapping!r}, "
                f"num_particles={self.num_particles})")


def resolve_save_path(spec, fmt: str = DEFAULT_FORMAT,
                      default: str | None = None) -> str | None:
    """Normalize a ``save_hamiltonian`` argument to a path (or ``None``).

    ``False`` / ``None`` disable saving; ``True`` selects
    ``"hamiltonian" + <format extension>``; a string or :class:`os.PathLike` is
    used as the path, with the format's extension appended when it carries none
    of its own.  Accepting a path as well as the documented boolean means a run
    can name its own cache file without a second argument.
    """
    if spec is None or spec is False:
        return None
    fmt = resolve_format(fmt)
    if spec is True:
        return default if default is not None else DEFAULT_STEM + FILE_EXTENSIONS[fmt]
    if isinstance(spec, (str, os.PathLike)):
        path = os.fspath(spec)
        return path if os.path.splitext(path)[1] else path + FILE_EXTENSIONS[fmt]
    raise TypeError(
        f"save_hamiltonian must be a bool or a path, got {type(spec).__name__}")


# --------------------------------------------------------------------------- #
# Write.
# --------------------------------------------------------------------------- #

def save_hamiltonian(path, hamiltonian: PauliSum, *,
                     mapping: str = "jordan_wigner",
                     num_particles=None, n_spatial_orbitals=None,
                     metadata: dict | None = None,
                     format: str | None = None,
                     compression: str = "zstd",
                     engine: str = "auto") -> str:
    """Write ``hamiltonian`` as Pauli terms to ``path``; return the path.

    Terms are sorted by Pauli string so the file is reproducible (an identical
    Hamiltonian gives an identical file) and easy to diff or query.  Negligible
    terms are dropped via :meth:`~carcara.core.mapping.PauliSum.simplify`.

    Parameters
    ----------
    format : {"parquet", "json"}, optional
        Output format.  Defaults to the format implied by ``path``'s extension,
        else ``"parquet"``.  ``"json"`` needs no Parquet engine at all.
    compression : str
        Parquet codec (``"zstd"`` by default, which compresses the highly
        repetitive Pauli strings very effectively).  Ignored for JSON.
    engine : str
        Parquet writer -- ``"auto"``, ``"fastparquet"`` or ``"pyarrow"`` (see the
        module docstring).  Ignored for JSON.
    """
    if format is None:
        extension = os.path.splitext(os.fspath(path))[1].lower()
        format = _EXTENSION_FORMATS.get(extension, DEFAULT_FORMAT)
    format = resolve_format(format)

    simplified = hamiltonian.simplify()
    items = sorted(simplified.terms.items())
    labels = [label for label, _ in items]
    reals = [complex(coeff).real for _, coeff in items]
    imags = [complex(coeff).imag for _, coeff in items]

    key_value = {
        "carcara.format": FORMAT_TAG,
        "carcara.version": str(FORMAT_VERSION),
        "carcara.num_qubits": str(int(simplified.num_qubits)),
        "carcara.mapping": str(mapping),
        "carcara.num_particles": json.dumps(
            None if num_particles is None
            else [int(num_particles[0]), int(num_particles[1])]),
        "carcara.n_spatial_orbitals": json.dumps(
            None if n_spatial_orbitals is None else int(n_spatial_orbitals)),
        "carcara.metadata": json.dumps(dict(metadata or {}), default=str),
    }

    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if format == "json":
        _write_json(path, labels, reals, imags, key_value)
    elif resolve_engine(engine) == "fastparquet":
        _write_fastparquet(path, labels, reals, imags, key_value, compression)
    else:
        _write_pyarrow(path, labels, reals, imags, key_value, compression)
    return path


def _write_json(path, labels, reals, imags, key_value):
    """Write the plain-text document; no third-party dependency."""
    payload = {
        "format": key_value["carcara.format"],
        "version": int(key_value["carcara.version"]),
        "num_qubits": int(key_value["carcara.num_qubits"]),
        "mapping": key_value["carcara.mapping"],
        "num_particles": json.loads(key_value["carcara.num_particles"]),
        "n_spatial_orbitals": json.loads(key_value["carcara.n_spatial_orbitals"]),
        "metadata": json.loads(key_value["carcara.metadata"]),
        "terms": [[label, real, imag]
                  for label, real, imag in zip(labels, reals, imags)],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def _write_fastparquet(path, labels, reals, imags, key_value, compression):
    import fastparquet
    import pandas as pd

    frame = pd.DataFrame({"pauli": pd.Series(labels, dtype="object"),
                          "real": pd.Series(reals, dtype="float64"),
                          "imag": pd.Series(imags, dtype="float64")})
    fastparquet.write(path, frame, compression=str(compression).upper(),
                      custom_metadata=key_value, write_index=False)


def _write_pyarrow(path, labels, reals, imags, key_value, compression):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {"pauli": pa.array(labels, type=pa.string()),
         "real": pa.array(reals, type=pa.float64()),
         "imag": pa.array(imags, type=pa.float64())},
        metadata={k.encode(): v.encode() for k, v in key_value.items()})
    pq.write_table(table, path, compression=compression)


# --------------------------------------------------------------------------- #
# Read.
# --------------------------------------------------------------------------- #

def load_hamiltonian(path, engine: str = "auto",
                     format: str | None = None) -> HamiltonianRecord:
    """Read a Hamiltonian file written by :func:`save_hamiltonian`.

    The format is **detected automatically** (:func:`detect_format`) -- from the
    extension, else from the file's leading bytes -- so Parquet and JSON caches
    load through the same call.  Parquet files are readable by either engine.

    Parameters
    ----------
    engine : str
        Parquet reader (``"auto"`` / ``"fastparquet"`` / ``"pyarrow"``); ignored
        for JSON.
    format : {"parquet", "json"}, optional
        Force the format instead of detecting it.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not a Carcará Hamiltonian, its schema version is newer
        than this build understands, or its Pauli strings are inconsistent.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such Hamiltonian file: {path!r}")
    format = detect_format(path) if format is None else resolve_format(format)

    if format == "json":
        columns, meta = _read_json(path)
    elif resolve_engine(engine) == "fastparquet":
        columns, meta = _read_fastparquet(path)
    else:
        columns, meta = _read_pyarrow(path)

    if meta.get("carcara.format") != FORMAT_TAG:
        raise ValueError(
            f"{path!r} is not a Carcará qubit-Hamiltonian Parquet file "
            f"(expected carcara.format = {FORMAT_TAG!r})")
    version = int(meta.get("carcara.version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path!r} uses Hamiltonian format version {version}, but this build "
            f"understands up to {FORMAT_VERSION}")

    for required in COLUMNS:
        if required not in columns:
            raise ValueError(
                f"{path!r} is missing the {required!r} column "
                f"(found {sorted(columns)})")

    n_qubits = int(meta["carcara.num_qubits"])
    terms: dict[str, complex] = {}
    for label, real, imag in zip(columns["pauli"], columns["real"],
                                 columns["imag"]):
        label = str(label)
        if len(label) != n_qubits:
            raise ValueError(
                f"{path!r}: Pauli string {label!r} has length {len(label)}, "
                f"expected {n_qubits}")
        terms[label] = terms.get(label, 0j) + complex(float(real), float(imag))

    num_particles = json.loads(meta.get("carcara.num_particles", "null"))
    if num_particles is not None:
        num_particles = (int(num_particles[0]), int(num_particles[1]))
    n_orbitals = json.loads(meta.get("carcara.n_spatial_orbitals", "null"))

    return HamiltonianRecord(
        hamiltonian=PauliSum(terms),
        mapping=meta.get("carcara.mapping", "jordan_wigner"),
        num_particles=num_particles,
        n_spatial_orbitals=None if n_orbitals is None else int(n_orbitals),
        metadata=json.loads(meta.get("carcara.metadata", "{}")))


def _read_json(path):
    """Read the plain-text document into the shared ``(columns, meta)`` shape."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path!r} is not a Carcará qubit-Hamiltonian JSON file")

    meta = {
        "carcara.format": str(payload.get("format", "")),
        "carcara.version": str(payload.get("version", 0)),
        "carcara.num_qubits": str(payload.get("num_qubits", 0)),
        "carcara.mapping": str(payload.get("mapping", "jordan_wigner")),
        "carcara.num_particles": json.dumps(payload.get("num_particles")),
        "carcara.n_spatial_orbitals": json.dumps(
            payload.get("n_spatial_orbitals")),
        "carcara.metadata": json.dumps(payload.get("metadata") or {}),
    }
    entries = payload.get("terms", [])
    columns = {"pauli": [e[0] for e in entries],
               "real": [float(e[1]) for e in entries],
               "imag": [float(e[2]) for e in entries]}
    # A term list of the wrong arity would have raised above; an empty file still
    # needs the columns present so the shared validation reports properly.
    return columns, meta


def _read_fastparquet(path):
    import fastparquet

    parquet_file = fastparquet.ParquetFile(path)
    meta = {str(k): str(v)
            for k, v in (parquet_file.key_value_metadata or {}).items()}
    frame = parquet_file.to_pandas()
    columns = {name: frame[name].tolist() for name in frame.columns}
    return columns, meta


def _read_pyarrow(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    raw = table.schema.metadata or {}
    meta = {k.decode(): v.decode() for k, v in raw.items()}
    columns = {name: table.column(name).to_pylist()
               for name in table.column_names}
    return columns, meta
