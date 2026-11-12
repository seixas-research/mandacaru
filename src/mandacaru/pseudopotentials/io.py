# -*- coding: utf-8 -*-
# file: pseudopotentials/io.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""On-disk pseudopotential library.

Generating a pseudopotential means running a self-consistent all-electron atom
and solving a nonlinear fit per channel -- seconds per element.  That is far too
slow to repeat inside a geometry optimization, and it is also pure overhead: the
result depends only on the element, never on the molecule.  So each family's
library is generated once and kept in a repository of its own
(``$MANDACARU_PAW_PATH/lda-sr/``; :mod:`.environment`).

File formats
------------
One file per element, in either of two interchangeable formats
(:data:`PSEUDO_FORMATS`).  **Parquet** (``<symbol>.parquet``) is the default and
what the libraries use -- the radial tables are thousands of floats per
element and columnar compression matters across the whole periodic table.
**JSON** (``<symbol>.json``) is the same content as plain text and is what to
write when a dataset has to be inspected, plotted or diffed by hand.  Both are
read and written by the native engine (``mandacaru-core``: ``paw::record``, which
owns the schema, the storage rule of :data:`STRIDE` and :data:`DIGITS`, and every
refusal); no Python Parquet package is involved.  :func:`load_pseudopotential`
detects which it was given (extension first, then magic bytes), so the two are
drop-in equivalents.

Every file carries ``"format": "mandacaru-pseudopotential"``, the schema
``"version"`` and the ``"family"``.  The scalars sit beside the family, and
every radial table under ``"radial_tables"``; in a Parquet file the tables are
the columns and the scalars the key/value metadata.

Both formats are self-describing and readable without Mandacaru: a pseudopotential
is a physical object that people need to inspect, plot and diff, which is why the
text format is kept as a first-class alternative rather than an export.
"""

from __future__ import annotations

from functools import lru_cache
import os

from .dataset import PseudoPotential

#: Identifies a Mandacaru pseudopotential file.
FORMAT_TAG = "mandacaru-pseudopotential"
#: Current schema version.  Version 2 added the ``family`` field, which every
#: readable file carries.
FORMAT_VERSION = 2

#: File formats understood by ``format=``.
PSEUDO_FORMATS = ("parquet", "json")
#: Default: Parquet is ~5x smaller than JSON for these radial tables.
DEFAULT_FORMAT = "parquet"
#: Extension per format.
FILE_EXTENSIONS = {"parquet": ".parquet", "json": ".json"}
_EXTENSION_FORMATS = {".parquet": "parquet", ".pq": "parquet", ".json": "json"}
#: Magic number every Apache Parquet file begins with.
PARQUET_MAGIC = b"PAR1"


def resolve_format(fmt: str = DEFAULT_FORMAT) -> str:
    """Validate and normalize a ``format`` spec."""
    key = str(fmt).strip().lower().lstrip(".")
    if key == "pq":
        key = "parquet"
    if key not in PSEUDO_FORMATS:
        raise ValueError(
            f"unknown pseudopotential format {fmt!r}; use one of "
            f"{PSEUDO_FORMATS}")
    return key


def detect_format(path) -> str:
    """Detect a pseudopotential file's format: extension first, then content.

    Parquet files start with the ``PAR1`` magic number, JSON documents with
    ``{``.  So a file with an unhelpful extension still loads, and the two
    formats are freely interchangeable.
    """
    path = os.fspath(path)
    known = _EXTENSION_FORMATS.get(os.path.splitext(path)[1].lower())
    if known is not None:
        return known
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such pseudopotential file: {path!r}")
    with open(path, "rb") as handle:
        head = handle.read(16)
    if head.startswith(PARQUET_MAGIC):
        return "parquet"
    if head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{"):
        return "json"
    raise ValueError(
        f"cannot determine the format of {path!r}: unrecognized extension and "
        "the content matches neither Parquet nor JSON. Pass format= explicitly.")

#: Highest atomic number of the libraries.
LIBRARY_Z_MAX = 92


def library_elements(z_max: int = LIBRARY_Z_MAX) -> tuple:
    """Chemical symbols of a library -- everything with ``Z <= z_max``."""
    from ase.data import chemical_symbols
    return tuple(chemical_symbols[z] for z in range(1, int(z_max) + 1))


#: Elements of a library.
LIBRARY_ELEMENTS = library_elements()


#: Keep every ``STRIDE``-th radial point when writing **a library**.
#: The generation grid is far finer than a pseudopotential needs (it must resolve
#: the all-electron core during construction); the *result* is smooth by design,
#: so subsampling once costs no accuracy and shrinks the library ~4x.
#:
#: This is deliberately *not* the default of :func:`save_pseudopotential`.
#: Decimating on every write would make a load-then-save cycle lossy and
#: compound with each round trip; it belongs at generation time, where the fine
#: grid actually exists (``mandacaru-build``).  The rule itself
#: (which points, how many digits) is the engine's: ``paw::record::digits``.
STRIDE = 4
#: Significant digits written per number.  Double precision would store 17,
#: which is meaningless here and triples the file size.
DIGITS = 10


def save_pseudopotential(pp: PseudoPotential, path, stride: int = 1,
                         format: str | None = None) -> str:
    """Write ``pp`` to ``path`` (values rounded to :data:`DIGITS` significant
    figures); return the path.

    Parameters
    ----------
    format : {"parquet", "json"}, optional
        Defaults to the format implied by ``path``'s extension, else
        :data:`DEFAULT_FORMAT` (Parquet: ~5x smaller for these tables).
    stride : int
        Keep every ``stride``-th radial point.  The default of 1 writes the
        tables as given, so save-load-save is idempotent; a library is
        generated once at :data:`STRIDE`.
    """
    from .paw import PAWDataset
    from .record import write_dataset

    if not isinstance(pp, PAWDataset):
        raise TypeError(
            f"cannot write a {type(pp).__name__}: only PAW-LCAO and UPAW-LCAO "
            "datasets have a file layout")
    return write_dataset(pp, path, stride=stride, format=format)


@lru_cache(maxsize=128)
def _load_pseudopotential_cached(path: str, mtime_ns: int) -> PseudoPotential:
    from .record import read_dataset

    return read_dataset(path)


def load_pseudopotential(path, format: str | None = None) -> PseudoPotential:
    """Read a pseudopotential written by :func:`save_pseudopotential`.

    The format is **detected automatically** (:func:`detect_format`), so Parquet
    and JSON files load through the same call; ``format`` only asserts it.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such pseudopotential file: {path!r}")
    found = detect_format(path)
    if format is not None and resolve_format(format) != found:
        raise ValueError(f"{path!r} is a {found} file, not {resolve_format(format)}")
    real_path = os.path.realpath(path)
    return _load_pseudopotential_cached(real_path, os.stat(real_path).st_mtime_ns)


# --------------------------------------------------------------------------- #
# Library access.
# --------------------------------------------------------------------------- #

def library_file(symbol: str, paw_dataset,
                 format: str | None = None) -> str:
    """Path of ``symbol``'s file in the dataset folder ``paw_dataset``.

    With no ``format``, an existing file wins (either extension); otherwise the
    default format's extension is used.
    """
    paw_dataset = os.fspath(paw_dataset)
    if format is None:
        for candidate in (DEFAULT_FORMAT, *PSEUDO_FORMATS):
            path = os.path.join(paw_dataset,
                                f"{symbol}{FILE_EXTENSIONS[candidate]}")
            if os.path.exists(path):
                return path
        format = DEFAULT_FORMAT
    return os.path.join(paw_dataset,
                        f"{symbol}{FILE_EXTENSIONS[resolve_format(format)]}")


#: The relativistic treatment of the datasets a library folder holds, by the
#: suffix of its name (``lda-sr/``, ``lda-fr/``;
#: :data:`.environment.LIBRARY_FOLDERS`).
FOLDER_RELATIVITY = {"sr": "scalar", "fr": "dirac"}


def load_library_dataset(symbol: str, folder: str, family: str, cache: dict,
                         *, label: str, noun: str):
    """Load ``symbol`` from the library ``folder`` (cached), or explain.

    ``folder`` comes from :func:`.environment.paw_dataset_folder`, which has
    already refused an unset or wrong variable.  An **empty** folder gets the
    recipe to fill it; one that lacks the element names what it does hold.  A
    file of another family is refused, and so is one whose functional or
    relativistic treatment is not the one its folder names (``lda-sr/``,
    ``lda-fr/``, ``pbe-sr/``, ``pbe-fr/``, ...).

    The cache is keyed by the file and its identity (modification time and
    size), so a dataset replaced on disk -- a library rebuilt under a running
    session -- is read again rather than served from memory.  The dataset
    returned carries ``source`` (the file's real path) and ``source_sha256``
    (the SHA-256 of its bytes), which the run log's ``[BASIS]`` block records.
    """
    path = library_file(symbol, folder)
    if os.path.exists(path):
        # Taken before reading: a file replaced while it is read is then
        # read again on the next call rather than cached as the new one.
        status = os.stat(path)
        stamp = (status.st_mtime_ns, status.st_size)
        cached = cache.get(os.path.realpath(path))
        if cached is not None and cached[0] == stamp:
            return cached[1]
    else:
        available = available_elements(folder)
        if not available:
            raise FileNotFoundError(
                f"the {label} library at {folder!r} holds no datasets.  Update "
                "the repository checkout the library variable names "
                "(`mandacaru-paw --status` shows it), or generate the "
                f"datasets there: mandacaru-build --pp {family} --all "
                "--install")
        raise FileNotFoundError(
            f"no {label} {noun} for {symbol!r} at {path!r}. "
            f"Available: {', '.join(available)}. "
            f"Generate it with mandacaru-build --pp {family} --element {symbol} --install.")
    # The bytes, identified: a path names a checkout that can change under
    # it, so the run log records this digest beside the file's name.
    import hashlib
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    pp = load_pseudopotential(path)
    if str(getattr(pp, "family", "")).lower() != family:
        raise ValueError(f"{path!r} belongs to family {pp.family!r}, not "
                         f"{family!r}")
    # A library folder named for a functional (`pbe/`, or `lda-sr/` whose
    # first part names it) holds only that functional's datasets:
    # `paw_dataset="pbe"` must not silently run an LDA file copied into it.
    # Folders with any other name are the caller's own.
    from .environment import FUNCTIONALS
    name = os.path.basename(os.path.normpath(folder)).lower()
    folder_xc, _, suffix = name.partition("-")
    dataset_xc = str(getattr(pp, "xc", "") or "").lower()
    if folder_xc in FUNCTIONALS and dataset_xc and dataset_xc != folder_xc:
        raise ValueError(f"{path!r} is a {dataset_xc.upper()} dataset in the "
                         f"{folder_xc}/ folder; regenerate it with --xc "
                         f"{folder_xc} or move it to {dataset_xc}/")
    # The same for the treatment its suffix names: a scalar file copied into
    # `lda-fr/` would silently drop the spin-orbit term a calculation
    # selecting that folder asked for, and a Dirac file in `lda-sr/` would
    # add one nobody asked for.
    wanted = FOLDER_RELATIVITY.get(suffix) if folder_xc in FUNCTIONALS else None
    relativity = str(getattr(pp, "relativity", "") or "").lower()
    if wanted is not None and relativity != wanted:
        raise ValueError(f"{path!r} is a dataset with relativity="
                         f"{relativity!r} in the {name}/ folder, which holds "
                         f"relativity={wanted!r} datasets; regenerate it "
                         f"for that folder or move it")
    # Where the dataset came from: the run log's [BASIS] block names it, since
    # a library is a checkout an environment variable points at, anywhere.
    pp.source = os.path.realpath(path)
    pp.source_sha256 = digest
    cache[pp.source] = (stamp, pp)
    return pp


def available_elements(paw_dataset) -> list[str]:
    """Elements present in the dataset folder, in either format."""
    if not os.path.isdir(paw_dataset):
        return []
    found = set()
    for name in os.listdir(paw_dataset):
        stem, extension = os.path.splitext(name)
        if extension.lower() in _EXTENSION_FORMATS:
            found.add(stem)
    return sorted(found)
