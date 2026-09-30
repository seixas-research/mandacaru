# -*- coding: utf-8 -*-
# file: pseudopotentials/environment.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Where the pseudopotential libraries live: one environment variable each.

The ONCVPSP, PAW-LCAO and UPAW-LCAO datasets are not part of the
package.  Each family lives in a repository of its own
(``mandacaru-oncvpsp``, ``mandacaru-paw``, ``mandacaru-upaw``), and an
environment variable names the checkout:

====================  ==========================  ==========================
family                variable                    set it with
====================  ==========================  ==========================
``oncvpsp``           ``MANDACARU_ONCVPSP_PATH``  ``mandacaru --set-oncvpsp DIR``
``paw-lcao``          ``MANDACARU_PAW_PATH``      ``mandacaru --set-paw DIR``
``upaw-lcao``         ``MANDACARU_UPAW_PATH``     ``mandacaru --set-upaw DIR``
====================  ==========================  ==========================

Inside a checkout the datasets sit one directory per set, named for its
functional and relativistic treatment (:data:`LIBRARY_FOLDERS`): every family
ships ``lda-sr/`` (scalar-relativistic LDA, the default); PAW-LCAO adds
``lda-dirac/`` (with the spin-orbit term).  A set without an entry, such as ONCVPSP's PBE, sits in
the folder named for the functional (``pbe/``).
UPAW-LCAO is the one optional library: without ``MANDACARU_UPAW_PATH`` its
datasets are generated on demand (a few seconds per element), so an unset
variable is not an error for it (:data:`OPTIONAL_FAMILIES`).

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
FAMILY_VARIABLES = {"oncvpsp": "MANDACARU_ONCVPSP_PATH",
                    "paw-lcao": "MANDACARU_PAW_PATH",
                    "upaw-lcao": "MANDACARU_UPAW_PATH"}
#: Family -> the ``mandacaru`` flag that sets its variable.
SET_FLAGS = {"oncvpsp": "--set-oncvpsp",
             "paw-lcao": "--set-paw", "upaw-lcao": "--set-upaw"}
#: Family -> the repository that holds its datasets.
REPOSITORIES = {"oncvpsp": "mandacaru-oncvpsp",
                "paw-lcao": "mandacaru-paw", "upaw-lcao": "mandacaru-upaw"}
#: Families that work without their variable: UPAW-LCAO is generated on
#: demand when there is no library to read.
OPTIONAL_FAMILIES = ("upaw-lcao",)
#: Exchange-correlation functionals a checkout may hold a directory for.
FUNCTIONALS = ("lda", "pbe")
#: The functional of every dataset loaded without saying which.
DEFAULT_XC = "lda"
#: The folder of a set inside a family's checkout, where it is not simply
#: named for its functional.
LIBRARY_FOLDERS = {("oncvpsp", "lda", "scalar"): "lda-sr",
                   ("paw-lcao", "lda", "scalar"): "lda-sr",
                   ("paw-lcao", "lda", "dirac"): "lda-dirac"}
#: The folder ``Mandacaru(directory=...)`` reads by default, every family's
#: scalar-relativistic LDA set.
DEFAULT_LIBRARY_FOLDER = "lda-sr"

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


def library_folder_name(family: str, xc: str = DEFAULT_XC,
                        relativity: str = "scalar") -> str:
    """The folder name of ``family``'s ``xc`` set with ``relativity``
    (:data:`LIBRARY_FOLDERS`, else the functional itself)."""
    key, xc = _family(family), _functional(xc)
    return LIBRARY_FOLDERS.get((key, xc, str(relativity).lower()), xc)


def _folders(root: str) -> list[str]:
    """The set folders of a checkout: its non-hidden subdirectories."""
    return sorted(entry for entry in os.listdir(root)
                  if not entry.startswith(".")
                  and os.path.isdir(os.path.join(root, entry)))


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
                      must_exist: bool = True,
                      relativity: str = "scalar") -> str:
    """The folder holding ``family``'s ``<Symbol>.parquet`` files for ``xc``.

    ``directory`` is the caller's own folder, returned as given (no variable,
    no set subdirectory).  Otherwise it is ``<checkout>/<folder>`` of
    :func:`repository_path`, the folder :func:`library_folder_name` names; with
    ``must_exist`` (the default, for loading) a missing folder is an error,
    and without it (for writing) it is simply returned for the caller to
    create.
    """
    if directory is not None:
        return os.fspath(directory)
    key = _family(family)
    name = library_folder_name(key, xc, relativity)
    root = repository_path(key)
    folder = os.path.join(root, name)
    if must_exist and not os.path.isdir(folder):
        raise LibraryPathError(
            f"the {key} library at {root!r} has no {name}/ folder (it has: "
            f"{', '.join(_folders(root)) or 'none'}).  Update the "
            f"{REPOSITORIES[key]} checkout, or generate the datasets into it: "
            f"mandacaru-build --pp {key} --xc {_functional(xc)} --all --install")
    return folder


#: Families whose library is laid out as ``<checkout>/<folder>/``, the folder
#: a calculation selects with ``Mandacaru(directory=...)``.  UPAW-LCAO is
#: generated on demand into its own library and is not one of them.
FOLDER_FAMILIES = ("oncvpsp", "paw-lcao")


def library_folder(family: str, folder: str = DEFAULT_XC) -> str:
    """``<checkout>/<folder>`` of ``family``'s library, which must exist.

    ``folder`` is one directory name inside the checkout -- ``"lda-sr"``,
    ``"lda-dirac"`` for PAW-LCAO, ``"pbe"`` for ONCVPSP, or any other set placed beside them -- not a path: a
    separator, ``"."`` or ``".."`` is refused, so a calculation cannot reach
    outside the library its variable names.  A caller's own folder anywhere
    on disk is the basis option ``directory=`` instead.

    Raises
    ------
    ValueError
        When ``folder`` is not a plain directory name.
    LibraryPathError
        When the variable is unset, or the folder is not in the checkout; the
        message lists the folders that are.
    """
    name = str(folder).strip()
    if (not name or name in (".", "..") or os.sep in name
            or (os.altsep and os.altsep in name) or os.path.isabs(name)):
        raise ValueError(
            f"directory must be the name of one folder inside the "
            f"pseudopotential library, such as 'lda-sr' or 'lda-dirac', not "
            f"{folder!r}; to load datasets from a folder elsewhere, pass it "
            f"as the basis option, basis={{'name': ..., 'directory': path}}")
    key = _family(family)
    root = repository_path(key)
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        present = _folders(root)
        raise LibraryPathError(
            f"the {key} library at {root!r} has no {name}/ folder (it has: "
            f"{', '.join(present) or 'none'}).  Choose one of those with "
            f"directory=, or generate the datasets into it with "
            f"mandacaru-build --pp {key} ... --install")
    return path


def upaw_directory(xc: str = DEFAULT_XC, directory=None) -> str | None:
    """The UPAW-LCAO library folder, ``$MANDACARU_UPAW_PATH/<xc>``, or
    ``None`` when there is none to use.

    UPAW-LCAO is generated on demand, so an unset ``MANDACARU_UPAW_PATH`` is
    not an error here: it only means there is no library to read from.  A
    set variable that names no directory still is.
    """
    if directory is not None:
        return os.fspath(directory)
    if not os.environ.get(FAMILY_VARIABLES["upaw-lcao"], "").strip():
        return None
    return os.path.join(repository_path("upaw-lcao"), _functional(xc))


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
            optional = ("; optional, generated on demand"
                        if key in OPTIONAL_FAMILIES else "")
            lines.append(f"{key:<9} {variable} is not set  "
                         f"(mandacaru {SET_FLAGS[key]} DIR{optional})")
            continue
        try:
            root = repository_path(key)
        except LibraryPathError:
            lines.append(f"{key:<9} {variable}={value}  NOT A DIRECTORY  "
                         f"(mandacaru {SET_FLAGS[key]} DIR)")
            continue
        held = [f"{name}/: {dataset_count(os.path.join(root, name))} datasets"
                for name in _folders(root)
                if dataset_count(os.path.join(root, name))]
        lines.append(f"{key:<9} {variable}={root}  "
                     + ("; ".join(held) if held else "no dataset folder"))
    return lines
