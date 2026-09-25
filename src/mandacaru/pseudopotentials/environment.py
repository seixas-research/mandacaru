# -*- coding: utf-8 -*-
# file: pseudopotentials/environment.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Where the pseudopotential libraries live: one environment variable each.

The NCPP, ONCVPSP and PAW-LCAO datasets are not part of the package.  Each
family lives in a repository of its own (``mandacaru-ncpp``,
``mandacaru-oncvpsp``, ``mandacaru-paw``), and an environment variable names
the checkout:

====================  ==========================  ==========================
family                variable                    set it with
====================  ==========================  ==========================
``ncpp``              ``MANDACARU_NCPP_PATH``     ``mandacaru --set-ncpp DIR``
``oncvpsp``           ``MANDACARU_ONCVPSP_PATH``  ``mandacaru --set-oncvpsp DIR``
``paw-lcao``          ``MANDACARU_PAW_PATH``      ``mandacaru --set-paw DIR``
====================  ==========================  ==========================

Inside a checkout the datasets sit one directory per exchange-correlation
functional -- ``<checkout>/lda/<Symbol>.parquet`` today, ``<checkout>/pbe/``
when there are PBE datasets -- so one repository serves every functional.
UPAW-LCAO has no repository: it is generated on demand, and a UPAW-LCAO library
built with ``mandacaru-build --pp UPAW --install`` goes to
``$MANDACARU_PAW_PATH/upaw-lcao/<xc>/``.

The variables are read at the moment a dataset is needed, never cached, so a
calculation that needs a library and cannot find it fails in the
:class:`~mandacaru.algorithms.Mandacaru` constructor (whose dry-run build
loads every dataset) with a :class:`LibraryPathError` that names the command
to run.  A basis option ``directory=...`` bypasses the variable for one run:
it names the folder that holds the ``<Symbol>.parquet`` files itself.
"""

from __future__ import annotations

import os

#: Family -> the environment variable naming its repository checkout.
FAMILY_VARIABLES = {"ncpp": "MANDACARU_NCPP_PATH",
                    "oncvpsp": "MANDACARU_ONCVPSP_PATH",
                    "paw-lcao": "MANDACARU_PAW_PATH"}
#: Family -> the ``mandacaru`` flag that sets its variable.
SET_FLAGS = {"ncpp": "--set-ncpp", "oncvpsp": "--set-oncvpsp",
             "paw-lcao": "--set-paw"}
#: Family -> the repository that holds its datasets.
REPOSITORIES = {"ncpp": "mandacaru-ncpp", "oncvpsp": "mandacaru-oncvpsp",
                "paw-lcao": "mandacaru-paw"}
#: Exchange-correlation functionals a checkout may hold a directory for.
FUNCTIONALS = ("lda", "pbe")
#: The functional of every dataset loaded without saying which.
DEFAULT_XC = "lda"
#: Where a UPAW-LCAO library goes inside the PAW-LCAO checkout.
UPAW_SUBDIRECTORY = "upaw-lcao"

_DATA_EXTENSIONS = (".parquet", ".pq", ".json")


class LibraryPathError(FileNotFoundError):
    """A pseudopotential library is not configured, or not where it is said
    to be.  The message names the command that fixes it."""


def _family(name: str) -> str:
    """Canonical family key of ``name`` (the registry resolves aliases)."""
    from .families import canonical_family_name

    key = canonical_family_name(name)
    if key not in FAMILY_VARIABLES:
        raise ValueError(f"the {name!r} family has no library variable; the "
                         f"families with one are {sorted(FAMILY_VARIABLES)}")
    return key


def _functional(xc: str) -> str:
    xc = str(xc).strip().lower()
    if xc not in FUNCTIONALS:
        raise ValueError(f"xc must be one of {FUNCTIONALS}, not {xc!r}")
    return xc


def _how_to_set(key: str) -> str:
    return (f"Clone {REPOSITORIES[key]} and point Mandacaru at it:\n"
            f"    git clone https://github.com/seixas-research/"
            f"{REPOSITORIES[key]}\n"
            f"    mandacaru {SET_FLAGS[key]} /path/to/{REPOSITORIES[key]}\n"
            "then open a new terminal (or `source` your shell configuration) "
            "so the variable is defined.")


def repository_path(family: str) -> str:
    """The checkout that ``family``'s variable names, validated.

    Raises
    ------
    LibraryPathError
        When the variable is unset or empty, or names something that is not
        a directory.
    """
    key = _family(family)
    variable = FAMILY_VARIABLES[key]
    value = os.environ.get(variable, "").strip()
    if not value:
        raise LibraryPathError(
            f"{variable} is not set, so Mandacaru cannot find the {key} "
            f"pseudopotential library.  {_how_to_set(key)}")
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isdir(path):
        raise LibraryPathError(
            f"{variable}={value!r} is not a directory, so Mandacaru cannot "
            f"find the {key} pseudopotential library.  Point it at a checkout "
            f"of {REPOSITORIES[key]}:\n"
            f"    mandacaru {SET_FLAGS[key]} /path/to/{REPOSITORIES[key]}")
    return path


def library_directory(family: str, xc: str = DEFAULT_XC, directory=None, *,
                      must_exist: bool = True) -> str:
    """The folder holding ``family``'s ``<Symbol>.parquet`` files for ``xc``.

    ``directory`` is the caller's own folder, returned as given (no variable,
    no functional subdirectory).  Otherwise it is ``<checkout>/<xc>`` of
    :func:`repository_path`; with ``must_exist`` (the default, for loading) a
    missing functional directory is an error, and without it (for writing) it
    is simply returned for the caller to create.
    """
    if directory is not None:
        return os.fspath(directory)
    key = _family(family)
    xc = _functional(xc)
    folder = os.path.join(repository_path(key), xc)
    if must_exist and not os.path.isdir(folder):
        present = [f for f in FUNCTIONALS
                   if os.path.isdir(os.path.join(repository_path(key), f))]
        raise LibraryPathError(
            f"the {key} library at {repository_path(key)!r} has no {xc}/ "
            f"directory (it has: {', '.join(present) or 'none'}).  Update the "
            f"{REPOSITORIES[key]} checkout, or generate the datasets into it: "
            f"mandacaru-build --pp {key} --xc {xc} --all --install")
    return folder


def upaw_directory(xc: str = DEFAULT_XC, directory=None) -> str | None:
    """The UPAW-LCAO library folder, or ``None`` when there is none to use.

    UPAW-LCAO is generated on demand, so an unset ``MANDACARU_PAW_PATH`` is
    not an error here: it only means there is no library to read from.
    """
    if directory is not None:
        return os.fspath(directory)
    if not os.environ.get(FAMILY_VARIABLES["paw-lcao"], "").strip():
        return None
    return os.path.join(repository_path("paw-lcao"), UPAW_SUBDIRECTORY,
                        _functional(xc))


def dataset_count(folder: str) -> int:
    """How many dataset files ``folder`` holds (0 when it does not exist)."""
    if not os.path.isdir(folder):
        return 0
    return sum(1 for name in os.listdir(folder)
               if os.path.splitext(name)[1].lower() in _DATA_EXTENSIONS)


def status_lines() -> list[str]:
    """One line per family: its variable, where it points, what it holds."""
    lines = []
    for key, variable in FAMILY_VARIABLES.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            lines.append(f"{key:<9} {variable} is not set  "
                         f"(mandacaru {SET_FLAGS[key]} DIR)")
            continue
        try:
            root = repository_path(key)
        except LibraryPathError:
            lines.append(f"{key:<9} {variable}={value}  NOT A DIRECTORY  "
                         f"(mandacaru {SET_FLAGS[key]} DIR)")
            continue
        held = [f"{xc}/: {dataset_count(os.path.join(root, xc))} datasets"
                for xc in FUNCTIONALS if os.path.isdir(os.path.join(root, xc))]
        lines.append(f"{key:<9} {variable}={root}  "
                     + ("; ".join(held) if held else "no lda/ or pbe/ directory"))
    return lines
