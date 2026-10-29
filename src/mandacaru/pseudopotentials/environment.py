# -*- coding: utf-8 -*-
# file: pseudopotentials/environment.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Where the pseudopotential libraries live: one environment variable each.

The PAW-LCAO and UPAW-LCAO datasets are not part of the package.  Each
family lives in a repository of its own (``mandacaru-paw``,
``mandacaru-upaw``), and an environment variable names the checkout:

====================  ==========================  ==========================
family                variable                    set it with
====================  ==========================  ==========================
``paw-lcao``          ``MANDACARU_PAW_PATH``      ``mandacaru-paw --set DIR``
``upaw-lcao``         ``MANDACARU_UPAW_PATH``     ``mandacaru-paw --set-upaw DIR``
====================  ==========================  ==========================

``mandacaru-paw --status`` reports where each variable points
(:mod:`mandacaru.pseudopotentials.library_cli`).

Inside a checkout the datasets sit one folder per set, named for its
functional and relativistic treatment (:data:`LIBRARY_FOLDERS`): PAW-LCAO
ships ``lda-sr/`` (scalar-relativistic LDA), ``lda-fr/`` (with the
spin-orbit term), ``pbe-sr/`` (scalar-relativistic PBE) and ``pbe-fr/``
(PBE with the spin-orbit term).  A calculation
reads the set of its functional (:func:`default_library_folder`): ``pbe-sr/``
for PBE, HSE06, r2SCAN and r2SCAN+rVV10, ``lda-sr/`` for LDA and every
method without a functional.  A set without an entry sits in the folder
named for the functional.
UPAW-LCAO is the one optional library: without ``MANDACARU_UPAW_PATH`` its
datasets are generated on demand (a few seconds per element), so an unset
variable is not an error for it (:data:`OPTIONAL_FAMILIES`).

The variables are read at the moment a dataset is needed, never cached, so a
calculation that needs a library and cannot find it fails in the
:class:`~mandacaru.algorithms.Mandacaru` constructor (whose dry-run build
loads every dataset) with a :class:`LibraryPathError` that names the command
to run.  A basis option ``paw_dataset=...`` bypasses the variable for one run:
it names the folder that holds the ``<Symbol>.parquet`` files itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

#: Family -> the environment variable naming its repository checkout.
FAMILY_VARIABLES = {"paw-lcao": "MANDACARU_PAW_PATH",
                    "upaw-lcao": "MANDACARU_UPAW_PATH"}
#: Family -> the command that sets its variable (``DIR`` is the checkout).
SET_COMMANDS = {"paw-lcao": "mandacaru-paw --set DIR",
                "upaw-lcao": "mandacaru-paw --set-upaw DIR"}
#: Family -> the repository that holds its datasets.
REPOSITORIES = {"paw-lcao": "mandacaru-paw", "upaw-lcao": "mandacaru-upaw"}
#: Families that work without their variable: UPAW-LCAO is generated on
#: demand when there is no library to read.
OPTIONAL_FAMILIES = ("upaw-lcao",)
#: Exchange-correlation functionals a checkout may hold a folder for.
FUNCTIONALS = ("lda", "pbe")
#: The functional of every dataset loaded without saying which.
DEFAULT_XC = "lda"
#: The folder of a set inside a family's checkout, where it is not simply
#: named for its functional.
LIBRARY_FOLDERS = {("paw-lcao", "lda", "scalar"): "lda-sr",
                   ("paw-lcao", "lda", "dirac"): "lda-fr",
                   ("paw-lcao", "pbe", "scalar"): "pbe-sr",
                   ("paw-lcao", "pbe", "dirac"): "pbe-fr",
                   ("upaw-lcao", "lda", "scalar"): "lda-sr",
                   ("upaw-lcao", "lda", "dirac"): "lda-fr",
                   ("upaw-lcao", "pbe", "scalar"): "pbe-sr",
                   ("upaw-lcao", "pbe", "dirac"): "pbe-fr"}
#: The folder a calculation without a functional (Hartree-Fock, the
#: quantum solvers) or with LDA reads, the scalar-relativistic LDA set.
DEFAULT_LIBRARY_FOLDER = "lda-sr"
#: The functional whose datasets each Kohn-Sham functional reads, by the
#: canonical names of :data:`~mandacaru.integrals.exchange_correlation.GRID_FUNCTIONALS`.
#: PBE has its own set; the screened hybrid built on it and the meta-GGAs,
#: for which no datasets are generated, read the PBE set too.
DATASET_FUNCTIONALS = {"lda": "lda", "pbe": "pbe", "hse06": "pbe",
                       "r2scan": "pbe", "pbe+rvv10": "pbe",
                       "r2scan+rvv10": "pbe"}

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


def default_library_folder(functional: str | None = None) -> str:
    """The PAW-LCAO folder a calculation with the Kohn-Sham ``functional``
    (a canonical name) reads unless ``Mandacaru(paw_dataset=...)`` names
    another: the scalar-relativistic set of :data:`DATASET_FUNCTIONALS`.
    ``None``, a method without a functional, reads
    :data:`DEFAULT_LIBRARY_FOLDER`."""
    if functional is None:
        return DEFAULT_LIBRARY_FOLDER
    return library_folder_name("paw-lcao", DATASET_FUNCTIONALS[functional])


def _folders(root: str) -> list[str]:
    """The set folders of a checkout: its non-hidden subdirectories."""
    return sorted(entry for entry in os.listdir(root)
                  if not entry.startswith(".")
                  and os.path.isdir(os.path.join(root, entry)))


def _set_command(key: str, directory: str = "DIR") -> str:
    """The shell command that sets ``key``'s variable to ``directory``."""
    return SET_COMMANDS[key].replace("DIR", directory)


def _how_to_set(key: str) -> str:
    return (f"Clone {REPOSITORIES[key]} and point Mandacaru at it:\n"
            f"    git clone https://github.com/seixas-research/"
            f"{REPOSITORIES[key]}\n"
            f"    {_set_command(key, f'/path/to/{REPOSITORIES[key]}')}\n"
            "then open a new terminal (or `source` your shell configuration) "
            "so the variable is defined.")


def repository_path(family: str) -> str:
    """The checkout that ``family``'s variable names, validated.

    Raises
    ------
    LibraryPathError
        When the variable is unset or empty, or names something that is not
        a folder.
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
            f"{variable}={value!r} is not a folder, so Mandacaru cannot "
            f"find the {key} pseudopotential library.  Point it at a checkout "
            f"of {REPOSITORIES[key]}:\n"
            f"    {_set_command(key, f'/path/to/{REPOSITORIES[key]}')}")
    return path


def paw_dataset_folder(family: str, xc: str = DEFAULT_XC, paw_dataset=None, *,
                      must_exist: bool = True,
                      relativity: str = "scalar") -> str:
    """The folder holding ``family``'s ``<Symbol>.parquet`` files for ``xc``.

    ``paw_dataset`` is the caller's own folder, returned as given (no variable,
    no set subfolder).  Otherwise it is ``<checkout>/<folder>`` of
    :func:`repository_path`, the folder :func:`library_folder_name` names; with
    ``must_exist`` (the default, for loading) a missing folder is an error,
    and without it (for writing) it is simply returned for the caller to
    create.
    """
    return _paw_dataset_folder(family, xc, paw_dataset, must_exist, relativity,
                               _variable_value(family))


def _variable_value(family: str) -> str:
    """The family's variable as set now: part of every cache key below, so a
    changed variable is never answered from an earlier value."""
    return os.environ.get(FAMILY_VARIABLES.get(_family(family), ""), "")


@lru_cache(maxsize=64)
def _paw_dataset_folder(family, xc, paw_dataset, must_exist, relativity,
                        variable_value) -> str:
    if paw_dataset is not None:
        p = os.fspath(paw_dataset)
        if not os.path.isdir(p):
            try:
                return library_folder(family, p)
            except Exception:
                pass
        return p
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
#: a calculation selects with ``Mandacaru(paw_dataset=...)``.  UPAW-LCAO is
#: generated on demand into its own library and is not one of them.
FOLDER_FAMILIES = ("paw-lcao",)


def library_folder(family: str, folder: str = DEFAULT_XC) -> str:
    """``<checkout>/<folder>`` of ``family``'s library, which must exist.

    ``folder`` is one folder name inside the checkout -- ``"lda-sr"``,
    ``"lda-fr"``, or any other set placed beside them -- not a path: a
    separator, ``"."`` or ``".."`` is refused, so a calculation cannot reach
    outside the library its variable names.  A caller's own folder anywhere
    on disk is the basis option ``paw_dataset=`` instead.

    Raises
    ------
    ValueError
        When ``folder`` is not a plain folder name.
    LibraryPathError
        When the variable is unset, or the folder is not in the checkout; the
        message lists the folders that are.
    """
    return _library_folder(family, folder, _variable_value(family))


@lru_cache(maxsize=64)
def _library_folder(family, folder, variable_value) -> str:
    name = str(folder).strip()
    if (not name or name in (".", "..") or os.sep in name
            or (os.altsep and os.altsep in name) or os.path.isabs(name)):
        raise ValueError(
            f"paw_dataset must be the name of one folder inside the "
            f"pseudopotential library, such as 'lda-sr', 'lda-fr' or 'pbe-sr', "
            f"not "
            f"{folder!r}; to load datasets from a folder elsewhere, pass it "
            f"as the basis option, basis={{'name': ..., 'paw_dataset': path}}")
    key = _family(family)
    root = repository_path(key)
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        present = _folders(root)
        raise LibraryPathError(
            f"the {key} library at {root!r} has no {name}/ folder (it has: "
            f"{', '.join(present) or 'none'}).  Choose one of those with "
            f"paw_dataset=, or generate the datasets into it with "
            f"mandacaru-build --pp {key} ... --install")
    return path


def upaw_dataset_folder(xc: str = DEFAULT_XC, paw_dataset=None,
                        relativity: str = "scalar") -> str | None:
    """The UPAW-LCAO library folder, ``$MANDACARU_UPAW_PATH/lda-sr`` (the
    folder :func:`library_folder_name` names for ``xc`` and ``relativity``), or
    ``None`` when there is none to use.

    UPAW-LCAO is generated on demand, so an unset ``MANDACARU_UPAW_PATH`` is
    not an error here: it only means there is no library to read from.  A
    set variable that names no folder still is.
    """
    if paw_dataset is not None:
        p = os.fspath(paw_dataset)
        if not os.path.isdir(p):
            try:
                return library_folder("upaw-lcao", p)
            except Exception:
                pass
        return p
    if not os.environ.get(FAMILY_VARIABLES["upaw-lcao"], "").strip():
        return None
    return os.path.join(repository_path("upaw-lcao"),
                        library_folder_name("upaw-lcao", xc, relativity))


def dataset_count(folder: str) -> int:
    """How many dataset files ``folder`` holds (0 when it does not exist)."""
    if not os.path.isdir(folder):
        return 0
    return sum(1 for name in os.listdir(folder)
               if os.path.splitext(name)[1].lower() in _DATA_EXTENSIONS)


@dataclass(frozen=True)
class LibrarySet:
    """One dataset folder of a checkout: ``lda-sr/``, ``pbe-fr/``, ..."""

    name: str
    functional: str
    relativity: str
    count: int


@dataclass(frozen=True)
class LibraryStatus:
    """Where one family's variable points and what the checkout holds.

    ``state`` is ``"ok"`` (a folder), ``"unset"`` (no value) or ``"invalid"``
    (a value that names no folder); ``path`` is the checkout when ``ok``.
    """

    family: str
    variable: str
    value: str
    state: str
    path: str | None
    sets: tuple[LibrarySet, ...]

    @property
    def optional(self) -> bool:
        return self.family in OPTIONAL_FAMILIES

    @property
    def ok(self) -> bool:
        """Whether nothing needs fixing: a folder that holds datasets, or an
        optional family left unset (it is generated on demand)."""
        return (self.state == "ok" and bool(self.sets)) or (
            self.state == "unset" and self.optional)


def _set_of(family: str, name: str, count: int) -> LibrarySet:
    for (key, functional, relativity), folder in LIBRARY_FOLDERS.items():
        if key == family and folder == name:
            return LibrarySet(name, functional, relativity, count)
    return LibrarySet(name, name, "", count)


def library_status() -> list[LibraryStatus]:
    """One :class:`LibraryStatus` per family, read from the environment now."""
    report = []
    for key, variable in FAMILY_VARIABLES.items():
        value = os.environ.get(variable, "").strip()
        if not value:
            report.append(LibraryStatus(key, variable, "", "unset", None, ()))
            continue
        try:
            root = repository_path(key)
        except LibraryPathError:
            report.append(LibraryStatus(key, variable, value, "invalid", None,
                                        ()))
            continue
        sets = tuple(_set_of(key, name, dataset_count(os.path.join(root, name)))
                     for name in _folders(root)
                     if dataset_count(os.path.join(root, name)))
        report.append(LibraryStatus(key, variable, value, "ok", root, sets))
    return report
