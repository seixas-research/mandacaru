# -*- coding: utf-8 -*-
# file: basis/pseudo_io.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""On-disk pseudopotential library.

Generating a pseudopotential means running a self-consistent all-electron atom
and solving a nonlinear fit per channel -- a second or two per element.  That is
far too slow to repeat inside a geometry optimization, and it is also pure
overhead: the result depends only on the element, never on the molecule.  So the
library is generated once and shipped as plain JSON under ``pseudos/``.

File format
-----------
One file per element, ``<symbol>.json``:

.. code-block:: json

    {
      "format": "carcara-pseudopotential",
      "version": 1,
      "symbol": "O",
      "atomic_number": 8,
      "valence_charge": 6.0,
      "local_l": 1,
      "r": [...],                          // radial grid, Bohr
      "v_local": [...],                    // ionic local channel, Hartree
      "valence_density": [...],
      "channels": {
        "0": {"n": 2, "eigenvalue": ..., "r_cut": ...,
              "pseudo_radial": [...], "v_ionic": [...],
              "occupation": 4.0, "coefficients": [...]},
        ...
      },
      "projectors":  {"0": [...]},         // Kleinman-Bylander chi_l(r)
      "kb_energies": {"0": ...}            // Hartree
    }

Text rather than a binary format on purpose: a pseudopotential is a physical
object that people need to inspect, plot and diff, and the tables are only a few
thousand numbers.  The files are also readable without Carcará installed.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .pseudopotential import Channel, PseudoPotential, generate_pseudopotential

#: Identifies a Carcará pseudopotential file.
FORMAT_TAG = "carcara-pseudopotential"
#: Current schema version.
FORMAT_VERSION = 1

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

#: Highest atomic number in the shipped library (hydrogen through actinium).
LIBRARY_Z_MAX = 89


def library_elements(z_max: int = LIBRARY_Z_MAX) -> tuple:
    """Chemical symbols shipped in ``pseudos/`` -- everything with ``Z <= z_max``."""
    from ase.data import chemical_symbols
    return tuple(chemical_symbols[z] for z in range(1, int(z_max) + 1))


#: Elements shipped in ``pseudos/``.
LIBRARY_ELEMENTS = library_elements()


def generation_points(atomic_number: int, minimum: int = 6000) -> int:
    """Radial grid points for the all-electron solve of element ``Z``.

    The :math:`1s` shell scales as :math:`a_0/Z`, so a heavy atom needs a finer
    grid than a light one.  Valence eigenvalues -- the only thing a
    pseudopotential is built from -- are converged to ~1e-4 Ha at the floor
    already (checked against NIST LSD for Ar, Kr and Xe), so this scaling is
    margin rather than necessity; the deep-core *total* energy converges much
    more slowly and is irrelevant here.
    """
    return max(int(minimum), int(150 * int(atomic_number)))


def default_library_path() -> str:
    """Absolute path of the bundled ``pseudos/`` directory.

    Resolved relative to the repository root so it works from a source checkout;
    the ``CARCARA_PSEUDO_PATH`` environment variable overrides it.
    """
    override = os.environ.get("CARCARA_PSEUDO_PATH")
    if override:
        return os.path.abspath(override)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "pseudos")


#: Keep every ``STRIDE``-th radial point when writing **the bundled library**.
#: The generation grid is far finer than a pseudopotential needs (it must resolve
#: the all-electron core during construction); the *result* is smooth by design,
#: so subsampling once costs no accuracy and shrinks the library ~4x.
#:
#: This is deliberately *not* the default of :func:`save_pseudopotential`.
#: Decimating on every write would make a load-then-save cycle lossy and
#: compound with each round trip; it belongs at generation time, where the fine
#: grid actually exists.  See :func:`build_library`.
STRIDE = 4
#: Significant digits written per number.  Double precision would store 17,
#: which is meaningless here and triples the file size.
DIGITS = 10


def _table(values, stride=1):
    """Subsample and round a radial table for compact storage."""
    return [float(f"%.{DIGITS}g" % v)
            for v in np.asarray(values, dtype=float)[::stride]]


def save_pseudopotential(pp: PseudoPotential, path, stride: int = 1,
                         format: str | None = None, engine: str = "auto") -> str:
    """Write ``pp`` to ``path``; return the path.

    Parameters
    ----------
    format : {"parquet", "json"}, optional
        Defaults to the format implied by ``path``'s extension, else
        :data:`DEFAULT_FORMAT` (Parquet: ~5x smaller for these tables).
    engine : str
        Parquet engine (``"auto"`` / ``"fastparquet"`` / ``"pyarrow"``); ignored
        for JSON.  See :mod:`carcara.core.serialization` for why fastparquet
        leads.

    stride : int
        Keep every ``stride``-th radial point.  The default of 1 writes the
        tables as given, so save-load-save is idempotent; the bundled library is
        generated once at :data:`STRIDE`.

    Values are rounded to :data:`DIGITS` significant figures.
    """
    if format is None:
        extension = os.path.splitext(os.fspath(path))[1].lower()
        format = _EXTENSION_FORMATS.get(extension, DEFAULT_FORMAT)
    format = resolve_format(format)

    payload = {
        "format": FORMAT_TAG,
        "version": FORMAT_VERSION,
        "symbol": pp.symbol,
        "atomic_number": int(pp.atomic_number),
        "valence_charge": float(pp.valence_charge),
        "local_l": int(pp.local_l),
        "r": _table(pp.r, stride),
        "v_local": _table(pp.v_local, stride),
        "valence_density": _table(pp.valence_density, stride),
        "channels": {
            str(l): {
                "n": int(channel.n),
                "eigenvalue": float(channel.eigenvalue),
                "r_cut": float(channel.r_cut),
                "occupation": float(channel.occupation),
                "norm_error": float(channel.norm_error),
                "coefficients": np.asarray(channel.coefficients).tolist(),
                "pseudo_radial": _table(channel.pseudo_radial, stride),
                "v_ionic": _table(channel.v_ionic, stride),
            }
            for l, channel in pp.channels.items()
        },
        "projectors": {str(l): _table(chi, stride)
                       for l, chi in pp.projectors.items()},
        "kb_energies": {str(l): float(e) for l, e in pp.kb_energies.items()},
    }
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
    """Split the payload into the radial table and the scalar metadata."""
    columns = {"r": payload["r"], "v_local": payload["v_local"],
               "valence_density": payload["valence_density"]}
    for key, entry in payload["channels"].items():
        columns[f"pseudo_radial_l{key}"] = entry["pseudo_radial"]
        columns[f"v_ionic_l{key}"] = entry["v_ionic"]
    for key, chi in payload["projectors"].items():
        columns[f"projector_l{key}"] = chi
    return columns


def _write_parquet(path, payload, engine):
    """Radial tables as columns; everything scalar as key/value metadata."""
    from ..core.serialization import native_pandas_strings, resolve_engine

    columns = _radial_columns(payload)
    scalars = {k: v for k, v in payload.items()
               if k not in ("r", "v_local", "valence_density", "projectors")}
    # Channels keep their scalars but drop the (now columnar) radial tables.
    scalars["channels"] = {
        key: {k: v for k, v in entry.items()
              if k not in ("pseudo_radial", "v_ionic")}
        for key, entry in payload["channels"].items()}
    metadata = {"carcara.pseudopotential": json.dumps(scalars)}

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

    if "carcara.pseudopotential" not in raw:
        raise ValueError(
            f"{path!r} is not a Carcará pseudopotential Parquet file")
    payload = json.loads(raw["carcara.pseudopotential"])
    payload["r"] = columns["r"]
    payload["v_local"] = columns["v_local"]
    payload["valence_density"] = columns["valence_density"]
    for key, entry in payload["channels"].items():
        entry["pseudo_radial"] = columns[f"pseudo_radial_l{key}"]
        entry["v_ionic"] = columns[f"v_ionic_l{key}"]
    payload["projectors"] = {
        name.split("_l")[1]: values for name, values in columns.items()
        if name.startswith("projector_l")}
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
        raise ValueError(f"{path!r} is not a Carcará pseudopotential file")
    if int(payload.get("version", 0)) > FORMAT_VERSION:
        raise ValueError(
            f"{path!r} uses pseudopotential format version "
            f"{payload['version']}, newer than this build ({FORMAT_VERSION})")

    r = np.asarray(payload["r"], dtype=float)
    channels = {}
    for key, entry in payload["channels"].items():
        l = int(key)
        channels[l] = Channel(
            l=l, n=int(entry["n"]), eigenvalue=float(entry["eigenvalue"]),
            r_cut=float(entry["r_cut"]),
            coefficients=np.asarray(entry["coefficients"], dtype=float),
            pseudo_radial=np.asarray(entry["pseudo_radial"], dtype=float),
            v_screened=np.zeros_like(r),        # not needed once unscreened
            v_ionic=np.asarray(entry["v_ionic"], dtype=float),
            occupation=float(entry["occupation"]),
            norm_error=float(entry.get("norm_error", 0.0)))

    return PseudoPotential(
        symbol=payload["symbol"],
        atomic_number=int(payload["atomic_number"]),
        valence_charge=float(payload["valence_charge"]),
        r=r,
        channels=channels,
        v_local=np.asarray(payload["v_local"], dtype=float),
        local_l=int(payload["local_l"]),
        projectors={int(k): np.asarray(v, dtype=float)
                    for k, v in payload["projectors"].items()},
        kb_energies={int(k): float(v)
                     for k, v in payload["kb_energies"].items()},
        valence_density=np.asarray(payload["valence_density"], dtype=float),
        atom=None)


# --------------------------------------------------------------------------- #
# Library access.
# --------------------------------------------------------------------------- #

_CACHE: dict[str, PseudoPotential] = {}


def library_file(symbol: str, directory=None,
                 format: str | None = None) -> str:
    """Path of ``symbol``'s file in the library directory.

    With no ``format``, an existing file wins (either extension); otherwise the
    default format's extension is used.
    """
    directory = default_library_path() if directory is None else directory
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


def available_elements(directory=None) -> list[str]:
    """Elements present in the library directory, in either format."""
    directory = default_library_path() if directory is None else directory
    if not os.path.isdir(directory):
        return []
    found = set()
    for name in os.listdir(directory):
        stem, extension = os.path.splitext(name)
        if extension.lower() in _EXTENSION_FORMATS:
            found.add(stem)
    return sorted(found)


def get_pseudopotential(symbol: str, directory=None) -> PseudoPotential:
    """Load ``symbol`` from the library (cached).

    Raises
    ------
    FileNotFoundError
        If the element is not in the library, with a pointer to the generator.
    """
    key = f"{symbol}@{directory}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    path = library_file(symbol, directory)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no pseudopotential for {symbol!r} at {path!r}. Available: "
            f"{', '.join(available_elements(directory)) or '(none)'}. "
            "Regenerate the library with "
            "`python -m carcara.basis.pseudo_io` or call "
            "`build_library()`.")
    pp = load_pseudopotential(path)
    _CACHE[key] = pp
    return pp


def build_library(elements=None, directory=None, *, verbose: bool = True,
                  format: str = DEFAULT_FORMAT, skip_failures: bool = True,
                  stride: int = STRIDE, **generation_options):
    """Generate and save pseudopotentials for ``elements``.

    Returns ``(written_paths, failures)``, where ``failures`` maps a symbol to
    the exception message.  Generation is a nonlinear fit per channel on top of
    a self-consistent atom, and it does not succeed for every element with the
    default cutoff heuristic -- ``skip_failures`` keeps going and reports at the
    end rather than aborting the whole library.

    The tables are written at ``stride`` (:data:`STRIDE` by default): the
    generation grid is far finer than the smooth result needs, and this is the
    one place where decimating is safe because the fine grid is right here.
    """
    from ase.data import atomic_numbers

    elements = LIBRARY_ELEMENTS if elements is None else tuple(elements)
    directory = default_library_path() if directory is None else directory
    written, failures = [], {}
    for symbol in elements:
        options = dict(generation_options)
        options.setdefault("points", generation_points(atomic_numbers[symbol]))
        try:
            pp = generate_pseudopotential(symbol, **options)
            path = save_pseudopotential(
                pp, library_file(symbol, directory, format), format=format,
                stride=stride)
        except Exception as error:                      # noqa: BLE001
            failures[symbol] = f"{type(error).__name__}: {error}"
            if verbose:
                print(f"  {symbol:>2}  FAILED  {failures[symbol][:70]}")
            if not skip_failures:
                raise
            continue
        written.append(path)
        if verbose:
            channels = ",".join(f"l{l}" for l in sorted(pp.channels))
            print(f"  {symbol:>2}  Z_ion={pp.valence_charge:>5.0f}  "
                  f"[{channels:<11}]  local=l{pp.local_l}  -> "
                  f"{os.path.basename(path)}")
    _CACHE.clear()
    return written, failures


if __name__ == "__main__":                                  # pragma: no cover
    import sys

    z_max = int(sys.argv[1]) if len(sys.argv) > 1 else LIBRARY_Z_MAX
    target = default_library_path()
    print(f"Generating the Carcará pseudopotential library (Z <= {z_max}) "
          f"in {target}")
    _written, failed = build_library(library_elements(z_max))
    print(f"\ndone: {len(available_elements(target))} elements written")
    if failed:
        print(f"{len(failed)} failed: {', '.join(sorted(failed))}")
