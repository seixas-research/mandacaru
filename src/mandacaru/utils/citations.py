# -*- coding: utf-8 -*-
# file: src/mandacaru/utils/citations.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Which references a run actually used, and the ``references.bib`` it writes.

:func:`citation_keys` reads a run's configuration -- the method, the operator
pool and its growth strategy, the fermion-to-qubit mapping, the basis family
and the options that changed how it was built, the classical optimizer, the
SDK the circuits went through -- and returns the keys of the entries in
:mod:`mandacaru.utils.bibliography` that apply.  It cites what *ran*: a PAW
basis with an ``energy_shift`` pulls in the confinement papers, a plain one
does not; ``tetris=True`` pulls in TETRIS-ADAPT-VQE, the default does not.

The selector is deliberately separate from the data: adding a paper is an edit
to the table, teaching a new option to cite it is an edit here.
"""

from __future__ import annotations

import os

from ..core.atomic import atomic_path
from .bibliography import REFERENCES, bibtex

#: Written when ``references=True`` names no path.
DEFAULT_REFERENCES_FILE = "references.bib"

#: Accepted values of the ``references=`` option (besides a path).
REFERENCES_CHOICES = (True, False, None, "auto")

#: Pool name -> the papers that define it.  ``"ceo"`` couples the qubit
#: excitations of :cite:`Yordanov2021`, so it cites both.
_POOL_KEYS = {
    "fermionic": ("Grimsley2019",),
    "qubit": ("Tang2021",),
    "qeb": ("Yordanov2021",),
    "ceo": ("Yordanov2021", "Ramoa2025"),
    "ceo-ovp": ("Yordanov2021", "Ramoa2025"),
}

_MAPPING_KEYS = {
    "jordan_wigner": ("Jordan1928",),
    "parity": ("Seeley2012",),
    "parity_reduced": ("Seeley2012", "Bravyi2017"),
    "bravyi_kitaev": ("BravyiKitaev2002", "Seeley2012"),
}

_METHOD_KEYS = {
    "vqe": ("Peruzzo2014", "Romero2019"),
    "adapt-vqe": ("Peruzzo2014", "Grimsley2019"),
    "subspace-vqe": ("Peruzzo2014", "Romero2019", "Nakanishi2019"),
    "subspace-adapt-vqe": ("Peruzzo2014", "Grimsley2019", "Nakanishi2019"),
    "qpe": ("Kitaev1995", "AspuruGuzik2005"),
}

_OPTIMIZER_KEYS = {
    "spsa": ("Spall1992",),
    "cobyla": ("Powell1994",),
    "nelder-mead": ("Nelder1965",),
    "slsqp": ("Kraft1988",),
    "adam": ("Kingma2015",),
    "l-bfgs-b": ("Byrd1995",),
    # L-BFGS is Liu & Nocedal; the bound-constrained extension Mandacaru
    # actually calls into (with no bounds) is Byrd et al.
    "l-bfgs": ("Liu1989", "Byrd1995"),
    "bfgs": ("Fletcher1970",),
    "nlcg-pr": ("Polak1969",),
}

#: Pseudopotential family -> the papers its datasets are generated from.
_FAMILY_KEYS = {
    "ncpp": ("Troullier1991", "Kleinman1982"),
    "oncvpsp": ("Hamann2013", "Kleinman1982"),
    "paw": ("Bloechl1994",),
    "upaw": ("Bloechl1994", "Ivanov2024"),
}


def _basis_keys(name, options) -> list:
    """References for an all-electron basis name and its options."""
    upper = str(name).upper()
    keys = []
    if upper in ("NAO", "NAO-AE"):
        keys += ["Sankey1989", "Junquera2001"]
    elif upper == "PW":
        keys.append("Ewald1921")
    elif upper.startswith("STO-") or upper == "GTO":
        keys += ["Slater1930", "Hehre1969"]
    elif upper.startswith("CC-P"):
        keys.append("Dunning1989")
    elif upper.startswith("DEF2"):
        keys.append("Weigend2005")
    elif upper[:1].isdigit() and "G" in upper:
        # A Pople name: 6-31G(d), 6-311+G(2df,2p), 3-21G, ...
        keys += ["Slater1930", "Hehre1969", "Hehre1972"]
    if upper in ("NAO", "NAO-AE") and options.get("size") not in (None, "SZ"):
        keys.append("Artacho1999")
    return keys


def _pseudo_keys(family, options) -> list:
    """References for a pseudopotential family and the options that built it."""
    keys = list(_FAMILY_KEYS.get(family, ()))
    size = options.get("size")
    if size not in (None, "SZ"):
        keys.append("Artacho1999")
    # A confined first zeta is the Sankey/Junquera construction, whatever the
    # dataset it is applied to; an unconfined one is the free-atom orbital.
    if options.get("energy_shift"):
        keys += ["Sankey1989", "Junquera2001"]
    if options.get("filter"):
        keys.append("Anglada2006")
    return keys


def citation_keys(*, method=None, pool=None, mapping=None, basis=None,
                  family=None, basis_options=None, optimizer=None,
                  backend_provider=None, shots=0, execute_circuits=False,
                  profile=False, tetris=False, prune=False,
                  has_geometry=True, built_basis=True, extras=()) -> list:
    """The bibliography keys a run with this configuration should cite.

    Every argument is optional: what is not known is not cited.  ``family`` and
    ``basis_options`` describe the basis that was actually *built* (the
    Hamiltonian builder's context), so they override what ``basis`` was
    spelled as; pass them when the run got that far.  ``extras`` adds keys for
    anything the caller knows about that this signature does not cover, such
    as an expressibility analysis.

    The returned list is in the bibliography's own order, so the same run
    always produces the same file.
    """
    keys: list = ["Mandacaru", "Harris2020", "Virtanen2020"]
    if has_geometry:
        keys.append("Larsen2017")

    keys += list(_METHOD_KEYS.get(_dashed(method), ()))
    keys += list(_POOL_KEYS.get(_dashed(pool), ()))
    if tetris:
        keys.append("Anastasiou2024")
    if prune:
        keys.append("VaqueroSabater2025")
    keys += list(_MAPPING_KEYS.get(_key(mapping), ()))
    keys += list(_OPTIMIZER_KEYS.get(_key(optimizer), ()))

    options = dict(basis_options or {})
    if family:
        keys += _pseudo_keys(_key(family), options)
    elif basis is not None:
        keys += _basis_keys(basis, options)

    if built_basis:
        # Every molecular Hamiltonian is built in the symmetrically
        # orthonormalized basis, from orbitals a DIIS-accelerated SCF produced.
        keys += ["Loewdin1950", "Pulay1980"]

    if _key(backend_provider) == "qiskit" and (shots or execute_circuits
                                               or profile):
        keys.append("Qiskit2024")

    keys += [str(key) for key in extras]
    # Unique, in the database's order.
    wanted = set(keys)
    return [key for key in REFERENCES if key in wanted]


def _key(value):
    """A registry name lowered to the spelling the tables above use."""
    if value is None:
        return None
    return str(value).strip().lower()


def _dashed(value):
    """As :func:`_key`, with underscores as hyphens.

    Method and pool names are spelling-insensitive across the codebase
    (``"adapt_vqe"`` is ``"adapt-vqe"``); mapping names are not -- they keep
    their underscores -- so only these two go through it.
    """
    key = _key(value)
    return None if key is None else key.replace("_", "-")


def resolve_references_path(references, txt=None) -> str | None:
    """The path ``references=`` asks for, or ``None`` when nothing is written.

    ``"auto"`` (the default) writes ``references.bib`` **beside the ``txt=``
    run log** when one is written and nothing otherwise, so a run that produces no files
    still produces none.  ``True`` writes ``references.bib`` in the working
    directory, a string names the file, and ``False`` / ``None`` switch it off.
    """
    if references is None or references is False:
        return None
    if references is True:
        return DEFAULT_REFERENCES_FILE
    if isinstance(references, str) and references != "auto":
        return references
    if references != "auto":
        raise ValueError(
            f"references={references!r} is not a path or one of "
            f"{REFERENCES_CHOICES}")
    if not txt:
        return None
    return os.path.join(os.path.dirname(os.fspath(txt)),
                        DEFAULT_REFERENCES_FILE)


def write_references(path, keys, header=None) -> str:
    """Write ``keys`` to ``path`` as BibTeX and return the path.

    Written through a temporary file (:func:`~mandacaru.core.atomic.atomic_path`)
    like every other file the run produces, so a reader never sees half of it.
    """
    text = bibtex(keys, header=header)
    with atomic_path(path) as temporary:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
    return str(path)
