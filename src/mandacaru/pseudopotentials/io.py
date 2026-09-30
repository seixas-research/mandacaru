# -*- coding: utf-8 -*-
# file: pseudopotentials/io.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""On-disk pseudopotential library.

Generating a pseudopotential means running a self-consistent all-electron atom
and solving a nonlinear fit per channel -- a second or two per element.  That is
far too slow to repeat inside a geometry optimization, and it is also pure
overhead: the result depends only on the element, never on the molecule.  So each
family's library is generated once and kept in a repository of its own
(``$MANDACARU_PAW_PATH/lda-sr/``, ``$MANDACARU_ONCVPSP_PATH/lda-sr/``;
:mod:`.environment`).

File formats
------------
One file per element, in either of two interchangeable formats
(:data:`PSEUDO_FORMATS`).  **Parquet** (``<symbol>.parquet``) is the default and
what the libraries use -- the radial tables are thousands of floats per
element and columnar compression matters across the whole periodic table.
**JSON** (``<symbol>.json``) is the same content as plain text, needs no Parquet
engine, and is what to write when a dataset has to be inspected, plotted or
diffed by hand.  :func:`load_pseudopotential` auto-detects which it was given
(extension first, then magic bytes), so the two are drop-in equivalents.

Every file carries ``"format": "mandacaru-pseudopotential"``, the schema
``"version"`` and the ``"family"``; the family's own module writes and reads
the rest (``to_payload`` / ``from_payload``, :data:`TABLE_FAMILIES`).  Its
scalars sit beside the family, and every radial table under
``"radial_tables"``; in a Parquet file the tables are the columns and the
scalars the key/value metadata.

Both formats are self-describing and readable without Mandacaru: a pseudopotential
is a physical object that people need to inspect, plot and diff, which is why the
text format is kept as a first-class alternative rather than an export.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .dataset import PseudoPotential

#: Identifies a Mandacaru pseudopotential file.
FORMAT_TAG = "mandacaru-pseudopotential"
#: Current schema version.  Version 2 added the ``family`` field, which every
#: readable file carries.
FORMAT_VERSION = 2
#: Family whose files carry several projectors per channel (see :mod:`.oncv`).
ONCV_FAMILY = "oncvpsp"
#: Family whose files carry partial waves, projectors and one-center matrices
#: (see :mod:`.paw`).
PAW_FAMILY = "paw-lcao"
#: The unitary variant of the same record (``norm_deficit = 0``): a distinct
#: family with the *same* layout, hence the same codec.
UPAW_FAMILY = "upaw-lcao"
#: Families whose payload keeps every radial table under ``"radial_tables"``
#: and whose record is (de)serialized by the family's own module
#: (``to_payload`` / ``from_payload``).  Several families may share one codec
#: (UPAW-LCAO *is* a :class:`~.paw.PAWDataset`), so this maps family -> layout and
#: is not invertible.
TABLE_FAMILIES = {ONCV_FAMILY: ".oncv", PAW_FAMILY: ".paw",
                  UPAW_FAMILY: ".paw"}


def _codec(family: str):
    """The module holding ``to_payload``/``from_payload`` for ``family``."""
    import importlib
    return importlib.import_module(TABLE_FAMILIES[family], __package__)


def _table_record(pp) -> str | None:
    """The table family ``pp`` is an instance of, or ``None``."""
    from .oncv import ONCVPseudoPotential
    from .paw import PAWDataset
    if isinstance(pp, PAWDataset):
        return PAW_FAMILY
    if isinstance(pp, ONCVPseudoPotential):
        return ONCV_FAMILY
    return None

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
#: grid actually exists (each family's ``build_*_library``).
STRIDE = 4
#: Significant digits written per number.  Double precision would store 17,
#: which is meaningless here and triples the file size.
DIGITS = 10


def _table(values, stride=1):
    """Subsample and round a radial table for compact storage.

    The kept points are ``stride - 1, 2 stride - 1, ...``: the generation grid
    starts one step ``h`` from the origin, so these are the radii
    ``stride h, 2 stride h, ...`` -- a grid that again starts one (coarser)
    step from the origin.  Keeping ``0, stride, ...`` instead stored
    ``h, (stride + 1) h, ...``, whose first step differs from every other,
    and every Numerov diagnostic of a loaded dataset (which prepends ``r = 0``
    and takes ``r[1] - r[0]`` as the step) integrated with the wrong step.
    """
    return [float(f"%.{DIGITS}g" % v)
            for v in np.asarray(values, dtype=float)[stride - 1::stride]]


def save_pseudopotential(pp: PseudoPotential, path, stride: int = 1,
                         format: str | None = None, engine: str = "auto") -> str:
    """Write ``pp`` to ``path`` (values rounded to :data:`DIGITS` significant
    figures); return the path.

    Parameters
    ----------
    format : {"parquet", "json"}, optional
        Defaults to the format implied by ``path``'s extension, else
        :data:`DEFAULT_FORMAT` (Parquet: ~5x smaller for these tables).
    engine : str
        Parquet engine (``"auto"`` / ``"fastparquet"`` / ``"pyarrow"``); ignored
        for JSON.  See :mod:`mandacaru.core.serialization` for why fastparquet
        leads.
    stride : int
        Keep every ``stride``-th radial point.  The default of 1 writes the
        tables as given, so save-load-save is idempotent; a library is
        generated once at :data:`STRIDE`.
    """
    if format is None:
        extension = os.path.splitext(os.fspath(path))[1].lower()
        format = _EXTENSION_FORMATS.get(extension, DEFAULT_FORMAT)
    format = resolve_format(format)

    family = str(getattr(pp, "family", ""))
    layout = _table_record(pp)
    if layout is None:
        raise TypeError(
            f"cannot write a {type(pp).__name__}: only ONCVPSP and PAW-LCAO "
            "datasets have a file layout")
    # ONCVPSP / PAW-LCAO records carry several projectors per channel, coupling
    # matrices, partial waves...; their payload is assembled by their own
    # module and every radial table lives under ``radial_tables``.  A
    # record may declare a *variant* of that layout's family (UPAW-LCAO is a
    # PAWDataset), and then it keeps its own name -- the layout only
    # chooses the codec.
    stored = (family if TABLE_FAMILIES.get(family) == TABLE_FAMILIES[layout]
              else layout)
    payload = {"format": FORMAT_TAG, "version": FORMAT_VERSION,
               "family": stored,
               **_codec(layout).to_payload(pp, stride)}
    return _write_payload(path, payload, format, engine)


def _defects_record(defects):
    from .oncv import defects_record
    return defects_record(defects)


def _read_defects(record):
    from .oncv import read_defects
    return read_defects(record)


def _warn_defects(dataset, family):
    from .oncv import warn_defects
    warn_defects(dataset, family)


def _write_payload(path, payload, format, engine) -> str:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if format == "json":
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.write("\n")
    else:
        _write_parquet(path, payload, engine)
    return path


def _radial_columns(payload):
    """The payload's radial tables, one column each."""
    return dict(payload["radial_tables"])


def _write_parquet(path, payload, engine):
    """Radial tables as columns; everything scalar as key/value metadata."""
    from ..core.serialization import native_pandas_strings, resolve_engine

    columns = _radial_columns(payload)
    scalars = {k: v for k, v in payload.items() if k != "radial_tables"}
    metadata = {"mandacaru.pseudopotential": json.dumps(scalars)}

    if resolve_engine(engine) == "fastparquet":
        import fastparquet
        import pandas as pd
        with native_pandas_strings():
            fastparquet.write(path, pd.DataFrame(columns), compression="ZSTD",
                              custom_metadata=metadata, write_index=False)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table(columns,
                         metadata={k.encode(): v.encode()
                                   for k, v in metadata.items()})
        pq.write_table(table, path, compression="zstd")


def _read_parquet(path, engine):
    """Reassemble the JSON-shaped payload from a Parquet file."""
    from ..core.serialization import native_pandas_strings, resolve_engine

    if resolve_engine(engine) == "fastparquet":
        import fastparquet
        parquet_file = fastparquet.ParquetFile(path)
        raw = {str(k): str(v)
               for k, v in (parquet_file.key_value_metadata or {}).items()}
        with native_pandas_strings():
            frame = parquet_file.to_pandas()
            columns = {name: frame[name].tolist() for name in frame.columns}
    else:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        raw = {k.decode(): v.decode()
               for k, v in (table.schema.metadata or {}).items()}
        columns = {name: table.column(name).to_pylist()
                   for name in table.column_names}

    if "mandacaru.pseudopotential" not in raw:
        raise ValueError(
            f"{path!r} is not a Mandacaru pseudopotential Parquet file")
    payload = json.loads(raw["mandacaru.pseudopotential"])
    payload["radial_tables"] = columns
    return payload


def load_pseudopotential(path, format: str | None = None,
                         engine: str = "auto") -> PseudoPotential:
    """Read a pseudopotential written by :func:`save_pseudopotential`.

    The format is **detected automatically** (:func:`detect_format`), so Parquet
    and JSON files load through the same call.

    The all-electron atom is *not* stored (it is large and only needed while
    generating), so :attr:`PseudoPotential.atom` is ``None`` on a loaded object.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such pseudopotential file: {path!r}")
    format = detect_format(path) if format is None else resolve_format(format)

    if format == "json":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = _read_parquet(path, engine)
    if payload.get("format") != FORMAT_TAG:
        raise ValueError(f"{path!r} is not a Mandacaru pseudopotential file")
    if int(payload.get("version", 0)) > FORMAT_VERSION:
        raise ValueError(
            f"{path!r} uses pseudopotential format version "
            f"{payload['version']}, newer than this build ({FORMAT_VERSION})")

    from .families import canonical_family_name
    if "family" not in payload or "radial_tables" not in payload:
        raise ValueError(
            f"{path!r} is not an ONCVPSP or PAW-LCAO dataset (no family, or "
            "no radial tables); this build cannot read it")
    family = canonical_family_name(payload["family"])
    if family not in TABLE_FAMILIES:
        raise ValueError(
            f"{path!r} declares family {family!r}, which this build cannot "
            f"read (it reads {', '.join(sorted(TABLE_FAMILIES))})")
    # The codec reads the family back off the payload, so hand it the
    # canonical spelling rather than the one on disk.
    return _codec(family).from_payload({**payload, "family": family})


# --------------------------------------------------------------------------- #
# Library access.
# --------------------------------------------------------------------------- #

def library_file(symbol: str, directory,
                 format: str | None = None) -> str:
    """Path of ``symbol``'s file in the library directory.

    With no ``format``, an existing file wins (either extension); otherwise the
    default format's extension is used.
    """
    directory = os.fspath(directory)
    if format is None:
        for candidate in (DEFAULT_FORMAT, *PSEUDO_FORMATS):
            path = os.path.join(directory,
                                f"{symbol}{FILE_EXTENSIONS[candidate]}")
            if os.path.exists(path):
                return path
        format = DEFAULT_FORMAT
    return os.path.join(directory,
                        f"{symbol}{FILE_EXTENSIONS[resolve_format(format)]}")


def load_library_dataset(symbol: str, folder: str, family: str, cache: dict,
                         *, label: str, noun: str, builder: str):
    """Load ``symbol`` from the library ``folder`` (cached), or explain.

    ``folder`` comes from :func:`.environment.library_directory`, which has
    already refused an unset or wrong variable.  An **empty** folder gets the
    recipe to fill it; one that lacks the element names what it does hold.  A
    file of another family is refused, and so is one whose functional is not
    the one its folder names (``lda-sr/``, ``pbe/``, ...).
    """
    key = f"{symbol}@{folder}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    path = library_file(symbol, folder)
    if not os.path.exists(path):
        available = available_elements(folder)
        if not available:
            raise FileNotFoundError(
                f"the {label} library at {folder!r} holds no datasets.  Update "
                "the repository checkout the library variable names "
                "(`mandacaru --pseudo-status` shows it), or generate the "
                f"datasets there: mandacaru-build --pp {family} --all "
                "--install")
        raise FileNotFoundError(
            f"no {label} {noun} for {symbol!r} at {path!r}. "
            f"Available: {', '.join(available)}. "
            f"Generate it with {builder}([symbol]).")
    pp = load_pseudopotential(path)
    if str(getattr(pp, "family", "")).lower() != family:
        raise ValueError(f"{path!r} belongs to family {pp.family!r}, not "
                         f"{family!r}")
    # A library folder named for a functional (`pbe/`, or `lda-sr/` whose
    # first part names it) holds only that functional's datasets:
    # `directory="pbe"` must not silently run an LDA file copied into it.
    # Folders with any other name are the caller's own.
    from .environment import FUNCTIONALS
    folder_xc = os.path.basename(os.path.normpath(folder)).lower()
    folder_xc = folder_xc.split("-")[0]
    dataset_xc = str(getattr(pp, "xc", "") or "").lower()
    if folder_xc in FUNCTIONALS and dataset_xc and dataset_xc != folder_xc:
        raise ValueError(f"{path!r} is a {dataset_xc.upper()} dataset in the "
                         f"{folder_xc}/ folder; regenerate it with --xc "
                         f"{folder_xc} or move it to {dataset_xc}/")
    # Where the dataset came from: the run log's [BASIS] block names it, since
    # a library is a checkout an environment variable points at, anywhere.
    pp.source = os.path.realpath(path)
    cache[key] = pp
    return pp


def available_elements(directory) -> list[str]:
    """Elements present in the library directory, in either format."""
    if not os.path.isdir(directory):
        return []
    found = set()
    for name in os.listdir(directory):
        stem, extension = os.path.splitext(name)
        if extension.lower() in _EXTENSION_FORMATS:
            found.add(stem)
    return sorted(found)
