# -*- coding: utf-8 -*-
# file: utils/dumps.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Optional JSON dumps of the two objects a run is asked about most.

A realistic operator pool runs to hundreds of generators and a qubit
Hamiltonian to thousands of Pauli terms, so neither belongs in a run trace:
standard output reports the *size* of each and the operator actually selected,
and these two files carry the content for anyone who wants to read it.  They
are written only when asked for, by the driver options

``verbose_operators=True``
    the operator pool, to ``pool.json``;
``verbose_hamiltonian=True``
    the qubit Hamiltonian, to ``hamiltonian.inspect.json``.

Either option also accepts a path, so a scan can give every geometry its own
file (and keep them out of the repository root).

The format is plain JSON, for reading: a complex coefficient is an object
``{"real": ..., "imag": ...}``, Pauli labels are the register-ordered strings
Mandacaru uses internally (qubit 0 leftmost).  For the *round-trippable* form of
the Hamiltonian -- the one :class:`~mandacaru.algorithms.base.VariationalDriver`
reloads with ``load_hamiltonian=`` -- use ``save_hamiltonian=`` instead, which
writes a :class:`~mandacaru.core.serialization.HamiltonianRecord` in Parquet or
JSON.  Both are capped at :data:`~mandacaru.core.serialization.MAX_FILE_QUBITS`
qubits; above that the run continues and the file is skipped with a warning.
"""

from __future__ import annotations

import json
import os
import tempfile

from ..core.serialization import file_qubits_allowed
from ..version import __version__

#: Default file names of the two dumps.
POOL_FILE = "pool.json"

#: Default path of the *inspection* dump.  Deliberately not ``hamiltonian.json``:
#: that is what ``save_hamiltonian=True, hamiltonian_format="json"`` writes, and
#: the two documents have incompatible schemas -- writing both left the cache
#: unloadable (``KeyError: 0``) because the inspection dump had overwritten it.
HAMILTONIAN_FILE = "hamiltonian.inspect.json"

#: Tag written into an inspection dump so a reader can tell it from a cache
#: file before indexing its term records.
INSPECTION_TAG = "mandacaru-hamiltonian-inspection"


def resolve_dump_path(spec, default: str) -> str | None:
    """Normalize a ``verbose_operators`` / ``verbose_hamiltonian`` argument.

    ``False`` / ``None`` write nothing, ``True`` selects ``default``, and a
    string or :class:`os.PathLike` is used as the path -- so a scan can name
    one file per geometry instead of overwriting a single one.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return default
    if isinstance(spec, (str, os.PathLike)):
        text = os.fspath(spec)
        if not text:
            raise ValueError("an empty path is not a file name")
        return text
    raise TypeError(f"expected True, False or a path, got {spec!r}")


def _coefficient(value) -> dict:
    """A complex coefficient as ``{"real": ..., "imag": ...}``."""
    value = complex(value)
    return {"real": value.real, "imag": value.imag}


def _terms(pauli) -> list[dict]:
    """``[{"pauli": "XYZI", "real": ..., "imag": ...}, ...]``, largest first."""
    items = sorted(pauli.simplify().terms.items(),
                   key=lambda kv: (-abs(kv[1]), kv[0]))
    return [{"pauli": label, **_coefficient(coeff)} for label, coeff in items]


def _write(path: str, payload: dict) -> str:
    """Write ``payload`` as indented JSON, atomically.

    A dump is a *snapshot*: opening the destination in write mode means a crash,
    a serialization error or a full filesystem destroys the previous one and
    leaves a truncated file in its place.  The payload is written to a temporary
    file in the same directory and then moved onto the destination, which is
    atomic on every platform Mandacaru supports -- the same thing checkpoints do.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=parent, prefix=os.path.basename(path) + ".", suffix=".tmp",
        delete=False)
    try:
        with handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return path


def dump_pool(path: str, pool, operators, *, n_qubits: int | None = None,
              mapping: str | None = None, num_particles=None,
              two_qubit_reduction: bool = False) -> str | None:
    """Write the ADAPT operator pool to ``path`` as JSON.

    Each entry carries the operator's label, kind, support and the Pauli
    expansion of its anti-Hermitian generator -- everything the run trace
    deliberately leaves out.  Returns the path, or ``None`` when the register
    is too wide to write (a warning says so and the run continues).
    """
    operators = list(operators)
    width = n_qubits
    if width is None and operators:
        width = operators[0].n_qubits
    if not file_qubits_allowed(width, f"the operator pool to {path!r}"):
        return None
    payload = {
        "mandacaru_version": __version__,
        "pool": getattr(pool, "name", type(pool).__name__),
        "pool_class": type(pool).__name__,
        "pool_size": len(operators),
        "n_qubits": None if width is None else int(width),
        "mapping": mapping,
        "num_particles": None if num_particles is None
        else [int(n) for n in num_particles],
        "two_qubit_reduction": bool(two_qubit_reduction),
        "operators": [
            {"index": i, "label": op.label, "kind": op.kind,
             "support": [int(q) for q in op.support],
             "num_terms": len(op.generator.simplify().terms),
             "generator": _terms(op.generator)}
            for i, op in enumerate(operators)],
    }
    return _write(path, payload)


def dump_hamiltonian(path: str, hamiltonian, *, n_qubits: int | None = None,
                     mapping: str | None = None, num_particles=None,
                     two_qubit_reduction: bool = False,
                     n_spatial_orbitals: int | None = None) -> str | None:
    """Write the qubit Hamiltonian to ``path`` as JSON.

    Coefficients are in **Hartree** -- the unit the operator itself is in,
    whatever units the driver reports its energies in.  Returns the path, or
    ``None`` when the register is too wide to write.
    """
    simplified = hamiltonian.simplify()
    width = n_qubits if n_qubits is not None else simplified.num_qubits
    if not file_qubits_allowed(width, f"the qubit Hamiltonian to {path!r}"):
        return None
    terms = _terms(simplified)
    payload = {
        # A reader can tell this document from the round-trippable cache file
        # before it indexes `terms`, instead of failing on a missing key.
        "format": INSPECTION_TAG,
        "mandacaru_version": __version__,
        "n_qubits": int(width),
        "num_terms": len(terms),
        "energy_unit": "Ha",
        "mapping": mapping,
        "num_particles": None if num_particles is None
        else [int(n) for n in num_particles],
        "n_spatial_orbitals": None if n_spatial_orbitals is None
        else int(n_spatial_orbitals),
        "two_qubit_reduction": bool(two_qubit_reduction),
        "terms": terms,
    }
    return _write(path, payload)
