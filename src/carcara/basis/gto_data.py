# -*- coding: utf-8 -*-
# file: basis/gto_data.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Gaussian basis-set data and a parser for the NWChem/Gaussian94 format.

Built-in families span the main design philosophies:

* **Pople** split-valence -- ``STO-3G``, ``6-31G(d)``, ``6-311G(d,p)``;
* **Dunning** correlation-consistent -- ``cc-pVDZ``, ``cc-pVTZ``;
* **Karlsruhe** DFT-optimized -- ``def2-SVP``, ``def2-TZVP``.

The data live as verbatim NWChem/Gaussian94 text files in ``gto_sets/`` (fetched
from the `Basis Set Exchange <https://www.basissetexchange.org>`_, spherical
harmonics), covering H--Ar.  :func:`parse_nwchem` reads that same format, so any
element/family can be added with :func:`register` from a fresh download without
touching this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.data import atomic_numbers

# Angular-momentum letter -> l.  "SP" is a Pople shared-exponent shell that
# expands into one S and one P shell.
_L_LETTER = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}

_DATA_DIR = Path(__file__).resolve().parent / "gto_sets"


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "")


def parse_nwchem(text: str) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Parse NWChem/Gaussian94 basis text into ``{Z: [(l, exps, coeffs), ...]}``.

    Handles segmented and **general** contractions (a shell block with several
    coefficient columns yields one contracted function per column) and ``SP``
    shells (expanded into a shared-exponent ``S`` and ``P``).  Comment (``#``)
    lines and the ``BASIS``/``END`` delimiters are ignored.
    """
    data: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    element: str | None = None
    ltype: str | None = None
    prims: list[list[float]] = []

    def flush():
        nonlocal prims, element, ltype
        if element is None or ltype is None or not prims:
            prims = []
            return
        arr = np.array(prims, dtype=float)
        exps = arr[:, 0]
        shells = data.setdefault(atomic_numbers[element], [])
        if ltype == "SP":
            shells.append((0, exps, arr[:, 1]))
            shells.append((1, exps, arr[:, 2]))
        else:
            l = _L_LETTER[ltype]
            for j in range(1, arr.shape[1]):          # one function per column
                col = arr[:, j]
                if np.any(col != 0.0):                # skip padding columns
                    shells.append((l, exps, col))
        prims = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("BASIS") or upper == "END":
            flush()
            continue
        tok = line.split()
        is_header = (len(tok) >= 2 and tok[0][0].isalpha()
                     and (tok[1].upper() in _L_LETTER or tok[1].upper() == "SP"))
        if is_header:
            flush()
            element = tok[0].capitalize()
            ltype = tok[1].upper()
        else:
            prims.append([float(t.replace("D", "E").replace("d", "e"))
                          for t in tok])
    flush()
    return data


# --------------------------------------------------------------------------- #
# Registry: file-backed families in gto_sets/ plus any registered at runtime.
# --------------------------------------------------------------------------- #

_RAW_BASES: dict[str, str] = {}                 # name -> NWChem text (runtime)
_PARSED_CACHE: dict[str, dict[int, list]] = {}


def register(name: str, text: str) -> None:
    """Register a basis-set family from NWChem/Gaussian94 ``text`` under ``name``."""
    key = _normalize(name)
    _RAW_BASES[key] = text
    _PARSED_CACHE.pop(key, None)


def available_bases() -> list[str]:
    """Names of all registered basis-set families (built-in files + runtime)."""
    names = set(_RAW_BASES)
    if _DATA_DIR.is_dir():
        names |= {p.stem for p in _DATA_DIR.glob("*.nwchem")}
    return sorted(names)


def get_basis_data(name: str) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Parsed ``{Z: [(l, exps, coeffs), ...]}`` for the named family."""
    key = _normalize(name)
    if key not in _PARSED_CACHE:
        text = _RAW_BASES.get(key)
        if text is None:
            path = _DATA_DIR / f"{key}.nwchem"
            if not path.is_file():
                raise ValueError(
                    f"unknown basis set {name!r}; available: {available_bases()} "
                    f"(add one with carcara.basis.register)")
            text = path.read_text()
        _PARSED_CACHE[key] = parse_nwchem(text)
    return _PARSED_CACHE[key]
