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
    readable by any Parquet reader.
``hamiltonian_format="json"``
    A plain-text document.  Slower and larger, but readable and diffable.

Both are read and written by the native engine (``mandacaru-core``); no Python
Parquet package is involved.  :func:`load_hamiltonian` **detects the format
automatically**: from the file's own leading bytes (a Parquet file begins
with the ``PAR1`` magic number, a JSON document with ``{``), with the file
extension as the fallback; a name that contradicts the content gives a
``RuntimeWarning`` and the content wins.  So a cache written either way is
loaded with the same call.

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
specifies the problem.  The codec is ``compression="zstd"`` (default) or
``"uncompressed"`` (also spelled ``"none"``); the native Parquet reader also
accepts Snappy files, and refuses any other codec.

JSON layout
-----------
The same content as a single object: ``format`` / ``version`` / ``num_qubits`` /
``mapping`` / ``num_particles`` / ``n_spatial_orbitals`` / ``metadata``, plus
``terms`` as a list of ``[pauli_string, real, imag]`` triples.
"""

from __future__ import annotations

import json
import os
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
#: File extensions that name a format, and which format each one names.  An
#: extension from this table is a statement about the file's content: the writer
#: refuses to contradict it and the reader warns when a file does.
EXTENSION_FORMATS = {".parquet": "parquet", ".pq": "parquet", ".json": "json"}
#: Filename stem used when ``save_hamiltonian=True`` is given without a path.
DEFAULT_STEM = "hamiltonian"
#: Filename used when ``save_hamiltonian=True`` with the default format.
DEFAULT_FILENAME = DEFAULT_STEM + FILE_EXTENSIONS[DEFAULT_FORMAT]
#: Magic number every Apache Parquet file starts (and ends) with.
PARQUET_MAGIC = b"PAR1"


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


def _content_format(path) -> str | None:
    """``"parquet"`` or ``"json"`` from the file's first bytes, else ``None``."""
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head.startswith(PARQUET_MAGIC):
        return "parquet"
    if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{"):
        return "json"
    return None


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
    named = EXTENSION_FORMATS.get(extension)
    if not os.path.exists(path):
        if named is not None:
            return named            # a path being written, not read
        raise FileNotFoundError(f"no such Hamiltonian file: {path!r}")
    content = _content_format(path)
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
        f"extension ({', '.join(sorted(EXTENSION_FORMATS))}) and starts with "
        f"neither the Parquet magic number nor '{{'.  Pass format='parquet' or "
        "format='json' explicitly.")


# --------------------------------------------------------------------------- #
# Record.
# --------------------------------------------------------------------------- #

@dataclass
class HamiltonianRecord:
    """A qubit Hamiltonian plus the problem metadata needed to rebuild a driver.

    ``num_particles`` and ``n_spatial_orbitals`` are what let a loaded run skip
    the geometry entirely: they are exactly the two quantities the ADAPT-VQE pool
    and the UCCSD ansatz are built from.

    The mapping name fully describes the register representation, including
    whether the parity register is reduced.
    """

    hamiltonian: PauliSum
    mapping: str = "jordan_wigner"
    num_particles: tuple[int, int] | None = None
    n_spatial_orbitals: int | None = None
    metadata: dict | None = None

    def __post_init__(self):
        from .mapping import _canonical_method
        self.mapping = _canonical_method(self.mapping)

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

def warn_extension_mismatch(path, format, *, stacklevel=2):
    """Warn when ``path``'s extension names a different format than ``format``.

    Every route that writes a Hamiltonian file calls this, so an explicit
    format that contradicts the file name is reported the same way whichever
    route (a loaded Hamiltonian or a geometry run) writes it.
    """
    extension = os.path.splitext(os.fspath(path))[1].lower()
    named = EXTENSION_FORMATS.get(extension)
    if named is not None and named != format:
        warnings.warn(f"writing {format} content to {os.fspath(path)!r}, whose extension {extension!r} says {named}; the file's name will not match what is in it", RuntimeWarning, stacklevel=stacklevel)


def save_hamiltonian(path, hamiltonian: PauliSum, *, mapping: str = "jordan_wigner",
                     num_particles=None, n_spatial_orbitals=None,
                     metadata: dict | None = None, format: str | None = None,
                     compression: str = "zstd", resources=None) -> str:
    """Write ``hamiltonian`` as Pauli terms to ``path``; return the path.

    Terms are sorted by Pauli string so the file is reproducible (an identical
    Hamiltonian gives an identical file) and easy to diff or query.  Negligible
    terms are dropped via :meth:`~mandacaru.core.mapping.PauliSum.simplify`.  The
    document is validated by the native engine before any file is created, and
    the file is replaced atomically.

    Parameters
    ----------
    format : {"parquet", "json"}, optional
        Output format.  Defaults to the format implied by ``path``'s extension,
        else ``"parquet"``.
    compression : {"zstd", "uncompressed", "none"}
        Parquet codec (``"zstd"`` by default, which compresses the highly
        repetitive Pauli strings very effectively).  Ignored for JSON; any
        other name raises ``ValueError`` for Parquet.
    """
    from .mapping import _canonical_method
    mapping = _canonical_method(mapping)
    named = EXTENSION_FORMATS.get(os.path.splitext(os.fspath(path))[1].lower())
    if format is None:
        format = named if named is not None else DEFAULT_FORMAT
    format = resolve_format(format)
    warn_extension_mismatch(path, format, stacklevel=3)
    if hamiltonian.num_qubits > MAX_FILE_QUBITS:
        raise ValueError(f'refusing to write a {hamiltonian.num_qubits}-qubit Hamiltonian: Pauli-string files are limited to MAX_FILE_QUBITS = {MAX_FILE_QUBITS} qubits')
    simplified = hamiltonian.simplify()
    payload = {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "num_qubits": int(simplified.num_qubits),
        "mapping": str(mapping),
        "num_particles": (None if num_particles is None
                          else [int(num_particles[0]), int(num_particles[1])]),
        "n_spatial_orbitals": (None if n_spatial_orbitals is None
                               else int(n_spatial_orbitals)),
        "metadata": {"energy_unit": "Ha", **dict(metadata or {})},
        "terms": [[label, complex(coeff).real, complex(coeff).imag]
                  for label, coeff in sorted(simplified.terms.items())],
    }
    path = os.fspath(path)
    from .._native import Context
    snapshot = Context(**dict(resources or {})).parse_snapshot(
        json.dumps(payload, allow_nan=False, default=str), FORMAT_TAG)
    if format == "json":
        snapshot.save(path)
    else:
        snapshot.save_parquet(path, compression)
    return path


# --------------------------------------------------------------------------- #
# Read.
# --------------------------------------------------------------------------- #

def _checked_format(path, format) -> str:
    """The format of ``path``: detected, or the explicit one if the bytes agree."""
    if format is None:
        return detect_format(path)
    format = resolve_format(format)
    content = _content_format(path)
    if content is not None and content != format:
        raise ValueError(
            f"{path!r} holds {content} content, but format={format!r} was "
            "requested")
    return format


def _particles(value):
    return None if value is None else (int(value[0]), int(value[1]))


def load_hamiltonian(path, format: str | None = None, *,
                     resources=None) -> HamiltonianRecord:
    """Read a Hamiltonian file written by :func:`save_hamiltonian`.

    The format is **detected automatically** (:func:`detect_format`) -- from the
    extension, else from the file's leading bytes -- so Parquet and JSON caches
    load through the same call.  The native engine decodes and validates the
    document (tag, version, register width, Pauli strings, finite coefficients).

    Parameters
    ----------
    format : {"parquet", "json"}, optional
        Assert the format instead of detecting it; content of the other format
        raises ``ValueError``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not a Mandacaru Hamiltonian, its schema version is not
        the one this build understands, or its Pauli strings are inconsistent.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no such Hamiltonian file: {path!r}')
    _checked_format(path, format)
    from .._native import Context
    snapshot = Context(**dict(resources or {})).read_snapshot(path, FORMAT_TAG)
    payload = json.loads(snapshot.json)
    n_qubits = int(payload["num_qubits"])
    terms: dict[str, complex] = {}
    for label, real, imag in payload["terms"]:
        terms[label] = terms.get(label, 0j) + complex(float(real), float(imag))
    return HamiltonianRecord(
        hamiltonian=PauliSum(terms, num_qubits=n_qubits),
        mapping=payload["mapping"],
        num_particles=_particles(payload["num_particles"]),
        n_spatial_orbitals=(None if payload["n_spatial_orbitals"] is None
                            else int(payload["n_spatial_orbitals"])),
        metadata=payload["metadata"])


@dataclass
class HamiltonianHeader:
    """The problem specification of a cache file, **without** its terms.

    What a dry run needs: the register width and the particle numbers decide
    the qubit count, and neither requires the Pauli table.  ``n_terms`` is the
    row count when the container records it (Parquet does), else ``None``.
    """

    num_qubits: int
    mapping: str = "jordan_wigner"
    num_particles: tuple[int, int] | None = None
    n_spatial_orbitals: int | None = None
    metadata: dict | None = None
    n_terms: int | None = None

    def __post_init__(self):
        from .mapping import _canonical_method
        self.mapping = _canonical_method(self.mapping)


def read_hamiltonian_header(path, format: str | None = None, *,
                            resources=None) -> HamiltonianHeader:
    """The :class:`HamiltonianHeader` of a cache file, without loading its terms.

    A **Parquet** file keeps its metadata in the footer, so only the trailer
    and the footer are read -- the Pauli table (the part that grows with the
    problem) is never decoded.  A **JSON** cache is a single document and has to
    be parsed whole to reach its header; nothing is built from the terms, but the
    read is not metadata-only, and for a very large cache Parquet is the format
    to use.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such Hamiltonian file: {path!r}")
    format = _checked_format(path, format)
    from .._native import Context, read_parquet_metadata
    context = Context(**dict(resources or {}))
    if format == "json":
        payload = json.loads(context.read_snapshot(path, FORMAT_TAG).json)
        n_terms = len(payload["terms"])
    else:
        footer = read_parquet_metadata(context, path)
        meta = footer["metadata"]
        if meta.get("mandacaru.format") != FORMAT_TAG:
            raise ValueError(
                f"{path!r} is not a Mandacaru qubit-Hamiltonian Parquet file "
                f"(expected mandacaru.format = {FORMAT_TAG!r})")
        for key in ("mandacaru.version", "mandacaru.num_qubits"):
            if key not in meta:
                raise ValueError(f"{path!r} lacks the {key!r} metadata")
        if int(meta["mandacaru.version"]) != FORMAT_VERSION:
            raise ValueError(
                f"{path!r} uses Hamiltonian format version "
                f"{meta['mandacaru.version']}, but this build understands "
                f"version {FORMAT_VERSION}")
        payload = {
            "num_qubits": int(meta["mandacaru.num_qubits"]),
            "mapping": meta.get("mandacaru.mapping", "jordan_wigner"),
            "num_particles": json.loads(
                meta.get("mandacaru.num_particles", "null")),
            "n_spatial_orbitals": json.loads(
                meta.get("mandacaru.n_spatial_orbitals", "null")),
            "metadata": json.loads(meta.get("mandacaru.metadata", "{}")),
        }
        n_terms = int(footer["num_rows"])
    n_orbitals = payload["n_spatial_orbitals"]
    return HamiltonianHeader(
        num_qubits=int(payload["num_qubits"]),
        mapping=payload["mapping"],
        num_particles=_particles(payload["num_particles"]),
        n_spatial_orbitals=None if n_orbitals is None else int(n_orbitals),
        metadata=payload["metadata"],
        n_terms=n_terms)
