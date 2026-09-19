# -*- coding: utf-8 -*-
# file: core/serialization.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""On-disk serialization of the qubit Hamiltonian (Apache Parquet or JSON).

Building a molecular Hamiltonian is by far the most expensive stage of a Mandacaru
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
    :class:`~mandacaru.core.mapping.PauliSum`.
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
   Since Mandacaru transpiles circuits for gate-count profiling in the same
   process that saves the Hamiltonian, the default engine avoids that
   combination.  Set ``engine="pyarrow"`` explicitly if you prefer Arrow and your
   environment is unaffected -- or side-step Parquet entirely with
   ``hamiltonian_format="json"``, which has no native dependency and cannot be
   affected.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass

from .mapping import PauliSum

#: Value of the ``mandacaru.format`` metadata key identifying these files.
FORMAT_TAG = "mandacaru-qubit-hamiltonian"
#: Current schema version.
FORMAT_VERSION = 1
#: File formats understood by ``hamiltonian_format=`` / ``format=``.
HAMILTONIAN_FORMATS = ("parquet", "json")
#: Default format.
DEFAULT_FORMAT = "parquet"
#: Extension appended per format when a path is given without one.
FILE_EXTENSIONS = {"parquet": ".parquet", "json": ".json"}
#: Recognized extensions, for format detection.
#: File extensions that name a format, and which format each one names.  An
#: extension from this table is a statement about the file's content: the writer
#: refuses to contradict it and the reader warns when a file does.
EXTENSION_FORMATS = {".parquet": "parquet", ".pq": "parquet", ".json": "json"}
_EXTENSION_FORMATS = EXTENSION_FORMATS          # historical private spelling
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

    The extension decides when it is one this build knows (``.parquet``,
    ``.pq``, ``.json``) **and the file's first bytes agree** with it; otherwise
    the bytes decide: Parquet begins with the ``PAR1`` magic number, a JSON
    document with ``{`` (after any whitespace or byte-order mark).  This is what
    lets ``load_hamiltonian`` accept either format through the same call.

    A file whose content contradicts its name is read as what it *is*, with a
    :class:`RuntimeWarning` -- the alternative is an opaque
    ``UnicodeDecodeError`` from the wrong parser, and the bytes are the truth.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the format cannot be determined.
    """
    path = os.fspath(path)
    extension = os.path.splitext(path)[1].lower()
    named = _EXTENSION_FORMATS.get(extension)
    if not os.path.exists(path):
        if named is not None:
            return named            # a path being written, not read
        raise FileNotFoundError(f"no such Hamiltonian file: {path!r}")
    with open(path, "rb") as fh:
        head = fh.read(16)
    content = None
    if head.startswith(PARQUET_MAGIC):
        content = "parquet"
    elif head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{"):
        content = "json"
    if content is not None:
        if named is not None and named != content:
            warnings.warn(
                f"{path!r} is named like {named!r} but its content is "
                f"{content!r}; reading it as {content!r}",
                RuntimeWarning, stacklevel=2)
        return content
    if named is not None:
        return named
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

    ``two_qubit_reduction`` records whether the stored operator is **already
    tapered**.  Without it the register width and the orbital count disagree
    (``2M - 2`` Pauli characters for ``M`` spatial orbitals) and a loaded driver
    builds a pool two qubits too wide, so a reader must be told.  Files written
    before this field default to ``False``, which is what they were.
    """

    hamiltonian: PauliSum
    mapping: str = "jordan_wigner"
    num_particles: tuple[int, int] | None = None
    n_spatial_orbitals: int | None = None
    two_qubit_reduction: bool = False
    metadata: dict | None = None

    @property
    def num_qubits(self) -> int:
        return self.hamiltonian.num_qubits

    def __repr__(self) -> str:
        return (f"HamiltonianRecord({len(self.hamiltonian.terms)} terms, "
                f"num_qubits={self.num_qubits}, mapping={self.mapping!r}, "
                f"num_particles={self.num_particles})")


#: Widest register whose Hamiltonian cache Mandacaru writes.  The file holds
#: O(M^4) Pauli strings of ``n_qubits`` characters each, which beyond ~50 qubits
#: means gigabytes.  (The ADAPT-VQE ``output.txt`` log is always written; above
#: ``utils.logging.DETAILED_LOG_MAX_QUBITS`` it omits its Pauli strings.)
MAX_FILE_QUBITS = 50


def file_qubits_allowed(n_qubits, what: str) -> bool:
    """``False``, with a ``RuntimeWarning``, above :data:`MAX_FILE_QUBITS` qubits.

    The drivers call it before writing an optional file, so a large simulation
    still runs -- it only skips the file.
    """
    if n_qubits is None or int(n_qubits) <= MAX_FILE_QUBITS:
        return True
    import warnings
    warnings.warn(f"not writing {what}: {int(n_qubits)} qubits exceed the "
                  f"{MAX_FILE_QUBITS}-qubit limit for Hamiltonian files "
                  "(mandacaru.core.serialization.MAX_FILE_QUBITS)",
                  RuntimeWarning, stacklevel=3)
    return False


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
                     two_qubit_reduction: bool = False,
                     metadata: dict | None = None,
                     format: str | None = None,
                     compression: str = "zstd",
                     engine: str = "auto") -> str:
    """Write ``hamiltonian`` as Pauli terms to ``path``; return the path.

    Terms are sorted by Pauli string so the file is reproducible (an identical
    Hamiltonian gives an identical file) and easy to diff or query.  Negligible
    terms are dropped via :meth:`~mandacaru.core.mapping.PauliSum.simplify`.

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
    extension = os.path.splitext(os.fspath(path))[1].lower()
    named = _EXTENSION_FORMATS.get(extension)
    if format is None:
        format = named if named is not None else DEFAULT_FORMAT
    format = resolve_format(format)
    if named is not None and named != format:
        # An explicit format still wins -- the caller said what they meant --
        # but the file's name will lie about its content, so say so once.
        # (`load_hamiltonian` reads it by its bytes, not its name.)
        warnings.warn(
            f"writing {format} content to {os.fspath(path)!r}, whose extension "
            f"{extension!r} says {named}; the file's name will not match what "
            "is in it", RuntimeWarning, stacklevel=2)

    if hamiltonian.num_qubits > MAX_FILE_QUBITS:
        raise ValueError(
            f"refusing to write a {hamiltonian.num_qubits}-qubit Hamiltonian: "
            f"Pauli-string files are limited to MAX_FILE_QUBITS = "
            f"{MAX_FILE_QUBITS} qubits")
    simplified = hamiltonian.simplify()
    items = sorted(simplified.terms.items())
    labels = [label for label, _ in items]
    reals = [complex(coeff).real for _, coeff in items]
    imags = [complex(coeff).imag for _, coeff in items]

    key_value = {
        "mandacaru.format": FORMAT_TAG,
        "mandacaru.version": str(FORMAT_VERSION),
        "mandacaru.num_qubits": str(int(simplified.num_qubits)),
        "mandacaru.mapping": str(mapping),
        "mandacaru.num_particles": json.dumps(
            None if num_particles is None
            else [int(num_particles[0]), int(num_particles[1])]),
        "mandacaru.n_spatial_orbitals": json.dumps(
            None if n_spatial_orbitals is None else int(n_spatial_orbitals)),
        "mandacaru.two_qubit_reduction": json.dumps(bool(two_qubit_reduction)),
        # The Pauli coefficients are the internal Hartree ones; say so in the
        # file so a reader never has to guess.
        "mandacaru.metadata": json.dumps(
            {"energy_unit": "Ha", **dict(metadata or {})}, default=str),
    }

    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    # Written through a temporary file in the same directory and then moved
    # into place: a cache is a *snapshot*, and opening the destination directly
    # means a crash, a serialization error or a full filesystem destroys the
    # previous one and leaves a truncated file where a loadable one was.
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp",
        delete=False)
    handle.close()
    staging = handle.name
    try:
        if format == "json":
            _write_json(staging, labels, reals, imags, key_value)
        elif resolve_engine(engine) == "fastparquet":
            _write_fastparquet(staging, labels, reals, imags, key_value,
                               compression)
        else:
            _write_pyarrow(staging, labels, reals, imags, key_value, compression)
        os.replace(staging, path)
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise
    return path


def _write_json(path, labels, reals, imags, key_value):
    """Write the plain-text document; no third-party dependency."""
    payload = {
        "format": key_value["mandacaru.format"],
        "version": int(key_value["mandacaru.version"]),
        "num_qubits": int(key_value["mandacaru.num_qubits"]),
        "mapping": key_value["mandacaru.mapping"],
        "num_particles": json.loads(key_value["mandacaru.num_particles"]),
        "n_spatial_orbitals": json.loads(key_value["mandacaru.n_spatial_orbitals"]),
        "two_qubit_reduction": json.loads(
            key_value["mandacaru.two_qubit_reduction"]),
        "metadata": json.loads(key_value["mandacaru.metadata"]),
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
        If the file is not a Mandacaru Hamiltonian, its schema version is newer
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

    if meta.get("mandacaru.format") != FORMAT_TAG:
        raise ValueError(
            f"{path!r} is not a Mandacaru qubit-Hamiltonian Parquet file "
            f"(expected mandacaru.format = {FORMAT_TAG!r})")
    version = int(meta.get("mandacaru.version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{path!r} uses Hamiltonian format version {version}, but this build "
            f"understands up to {FORMAT_VERSION}")

    for required in COLUMNS:
        if required not in columns:
            raise ValueError(
                f"{path!r} is missing the {required!r} column "
                f"(found {sorted(columns)})")

    n_qubits = int(meta["mandacaru.num_qubits"])
    terms: dict[str, complex] = {}
    for label, real, imag in zip(columns["pauli"], columns["real"],
                                 columns["imag"]):
        label = str(label)
        if len(label) != n_qubits:
            raise ValueError(
                f"{path!r}: Pauli string {label!r} has length {len(label)}, "
                f"expected {n_qubits}")
        terms[label] = terms.get(label, 0j) + complex(float(real), float(imag))

    num_particles = json.loads(meta.get("mandacaru.num_particles", "null"))
    if num_particles is not None:
        num_particles = (int(num_particles[0]), int(num_particles[1]))
    n_orbitals = json.loads(meta.get("mandacaru.n_spatial_orbitals", "null"))

    return HamiltonianRecord(
        # The width comes from the metadata, not from the labels: a zero
        # operator (or one whose terms cancelled) has no labels to infer it
        # from and would load as a 0-qubit register.
        hamiltonian=PauliSum(terms, num_qubits=n_qubits),
        mapping=meta.get("mandacaru.mapping", "jordan_wigner"),
        num_particles=num_particles,
        n_spatial_orbitals=None if n_orbitals is None else int(n_orbitals),
        two_qubit_reduction=bool(json.loads(
            meta.get("mandacaru.two_qubit_reduction", "false"))),
        metadata=json.loads(meta.get("mandacaru.metadata", "{}")))


def _read_json(path):
    """Read the plain-text document into the shared ``(columns, meta)`` shape."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path!r} is not a Mandacaru qubit-Hamiltonian JSON file")

    tag = str(payload.get("format", ""))
    if tag and tag != FORMAT_TAG:
        # A different Mandacaru JSON document -- most easily the *inspection* dump
        # of `verbose_hamiltonian`, whose term records are objects rather than
        # triples.  Saying so beats failing later with `KeyError: 0`.
        raise ValueError(
            f"{path!r} is a {tag!r} document, not a Mandacaru qubit-Hamiltonian "
            f"cache ({FORMAT_TAG!r}); `verbose_hamiltonian` writes a "
            f"human-readable dump that cannot be loaded back -- use the file "
            f"written by `save_hamiltonian`.")
    if "terms" in payload and payload["terms"] and not isinstance(
            payload["terms"][0], (list, tuple)):
        raise ValueError(
            f"{path!r} stores its terms as records, not [pauli, real, imag] "
            f"triples: this is an inspection dump, not a loadable cache.")

    meta = {
        "mandacaru.format": tag,
        "mandacaru.version": str(payload.get("version", 0)),
        "mandacaru.num_qubits": str(payload.get("num_qubits", 0)),
        "mandacaru.mapping": str(payload.get("mapping", "jordan_wigner")),
        "mandacaru.num_particles": json.dumps(payload.get("num_particles")),
        "mandacaru.n_spatial_orbitals": json.dumps(
            payload.get("n_spatial_orbitals")),
        "mandacaru.two_qubit_reduction": json.dumps(
            bool(payload.get("two_qubit_reduction", False))),
        "mandacaru.metadata": json.dumps(payload.get("metadata") or {}),
    }
    entries = payload.get("terms", [])
    columns = {"pauli": [e[0] for e in entries],
               "real": [float(e[1]) for e in entries],
               "imag": [float(e[2]) for e in entries]}
    # A term list of the wrong arity would have raised above; an empty file still
    # needs the columns present so the shared validation reports properly.
    return columns, meta


def native_pandas_strings():
    """Context manager keeping pandas off its Arrow-backed string dtype.

    pandas 3 sets ``future.infer_string`` by default, so even a *fastparquet*
    read builds its column index as a ``pyarrow`` string array -- which drags
    pyarrow into a process that chose fastparquet precisely to avoid it.  In a
    run that has also called Qiskit's ``transpile`` the result is not a clean
    import error but corruption: ``UnicodeDecodeError`` or ``ArrowException:
    Wrapping <garbage> failed`` while decoding column names.  See the module
    warning above.

    Turning the option off for the duration of the read keeps the whole path in
    NumPy-backed object strings.  It is a no-op on pandas versions without the
    option.
    """
    import pandas as pd

    try:
        return pd.option_context("future.infer_string", False)
    except Exception:                       # pragma: no cover - older pandas
        from contextlib import nullcontext
        return nullcontext()


def _read_fastparquet(path):
    import fastparquet

    parquet_file = fastparquet.ParquetFile(path)
    meta = {str(k): str(v)
            for k, v in (parquet_file.key_value_metadata or {}).items()}
    with native_pandas_strings():
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
