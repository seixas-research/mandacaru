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
:mod:`mandacaru.utils.bibliography` that apply.  It cites what *ran*: a PAW-LCAO
basis with an ``energy_shift`` pulls in the confinement papers, a plain one
does not; ``tetris=True`` pulls in TETRIS-ADAPT-VQE, the default does not.

The selector is deliberately separate from the data: adding a paper is an edit
to the table, teaching a new option to cite it is an edit here.
"""

from __future__ import annotations

from .bibliography import REFERENCES


#: Pool name -> the papers that define it.  ``"ceo"`` couples the qubit
#: excitations of :cite:`Yordanov2021`, so it cites both.
_POOL_KEYS = {
    "fermionic": ("Grimsley2019",),
    # The fermionic pool with the S_z filter and the reality of the
    # generators lifted; no separate source introduces it.
    "spin-orbit": ("Grimsley2019",),
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
    "rhf": ("Roothaan1951",),
    "uhf": ("PopleNesbet1954",),
    "ghf": ("JimenezHoyos2011",),
    "dft": ("HohenbergKohn1964", "KohnSham1965"),
    # The circuit a fixed-ansatz VQE optimizes -- here and in subspace-vqe --
    # is cited through _ANSATZ_KEYS.
    "vqe": ("Peruzzo2014",),
    "adapt-vqe": ("Peruzzo2014", "Grimsley2019"),
    "subspace-vqe": ("Peruzzo2014", "Nakanishi2019"),
    "subspace-adapt-vqe": ("Peruzzo2014", "Grimsley2019", "Nakanishi2019"),
    # The chain searches ansatz structures; each is relaxed by VQE.  The
    # annealing paper is cited with the rest because the default temperature
    # is annealed.
    # Grimsley2019 for the pool gradient the default proposal draws from.
    "vasqa": ("Peruzzo2014", "Grimsley2019", "Metropolis1953",
              "Hastings1970", "Kirkpatrick1983"),
    # VALSQA is the same chain; its learned proposal is either a pairwise
    # Bradley-Terry ranker on the insertion descriptors or a message-passing
    # network over the Hamiltonian (the Gaussian-process proposal was retired,
    # so it is not cited).
    "valsqa": ("Peruzzo2014", "Grimsley2019", "Metropolis1953",
               "Hastings1970", "Kirkpatrick1983", "Gilmer2017",
               "BradleyTerry1952"),
    "qpe": ("Kitaev1995", "AspuruGuzik2005"),
}

#: A quantum method on a crystal also cites the machinery that makes it one: the
#: Born-von Karman supercell its mesh realizes, the Monkhorst-Pack sampling, the
#: Ewald sum of its electrostatics and the localized-orbital Bloch sum that
#: unfolds the supercell mean field onto the primitive cell
#: (``citation_keys(periodic=True)``).
_PERIODIC_KEYS = ("BornKarman1912", "MonkhorstPack1976", "Ewald1921",
                  "SlaterKoster1954")

#: The methods whose energy a circuit can measure (``shots`` or
#: ``execute_circuits``); their gradients are then parameter-shift differences.
_CIRCUIT_METHODS = ("vqe", "adapt-vqe", "subspace-vqe", "subspace-adapt-vqe",
                    "vasqa", "valsqa")
_PARAMETER_SHIFT_KEYS = ("Mitarai2018", "Schuld2019")

#: ``trotter=True`` applies the ansatz as an ordered product of exponentials.
_TROTTER_KEYS = ("Trotter1959",)

#: Exchange-correlation functional (``Mandacaru(xc=...)``, with ``vdw="rvv10"``
#: already folded in as ``"pbe+rvv10"`` / ``"r2scan+rvv10"``) -> the papers its
#: energy expression is built from.  HSE06 is PBE with the short-range exchange
#: hole screened and partly replaced by exact exchange, which a PAW sphere
#: corrects at its one-center terms.
_XC_KEYS = {
    "lda": ("PerdewZunger1981",),
    "pbe": ("PBE1996", "PerdewWang1992"),
    "r2scan": ("Furness2020",),
    "hse06": ("PBE1996", "PerdewWang1992", "Heyd2003", "Krukau2006",
              "Paier2005"),
    "pbe+rvv10": ("PBE1996", "PerdewWang1992", "Vydrov2010", "Sabatini2013",
                  "RomanPerez2009"),
    "r2scan+rvv10": ("Furness2020", "Ning2022", "Vydrov2010", "Sabatini2013",
                     "RomanPerez2009"),
}

#: ``vdw=`` correction that is not a functional of its own (``"rvv10"`` is the
#: ``+rvv10`` of ``xc``).
_VDW_KEYS = {
    "ts": ("Tkatchenko2009", "ChuDalgarno2004", "Ruiz2012", "Grimme2006",
           "Hirshfeld1977"),
}

#: Occupation smearing of a crystal (``smearing=``).  Every method is a free
#: energy at a finite electronic temperature (Mermin); the other two add their
#: own broadening function.
_SMEARING_KEYS = {
    "fermi-dirac": ("Mermin1965",),
    "gaussian": ("Mermin1965", "FuHo1983"),
    "methfessel-paxton": ("Mermin1965", "MethfesselPaxton1989"),
}

#: Hubbard correction (``hubbard_u=``): its ``form`` and the linear-response
#: determination of U (``hubbard_response``).
_HUBBARD_KEYS = {
    "dudarev-fll": ("Dudarev1998",),
    "liechtenstein-fll": ("Liechtenstein1995",),
    "linear-response": ("Cococcioni2005",),
}

#: Crystal eigensolver (``eigensolver=``); ``"direct"`` is a dense
#: diagonalization and cites nothing.
_EIGENSOLVER_KEYS = {
    "direct": (),
    "davidson": ("Davidson1975",),
    "rmm-diis": ("WoodZunger1985", "Pulay1980"),
    "mixed": ("Davidson1975", "WoodZunger1985", "Pulay1980"),
}

#: Divergence treatment of the periodic exact exchange (``exxdiv_treatment=``).
_EXXDIV_KEYS = {
    "none": (),
    "gygi-baldereschi": ("GygiBaldereschi1986",),
}

#: Simulated devices with a method of their own (``device=``, canonical name).
_DEVICE_KEYS = {
    "AER_mps": ("Vidal2003", "Schollwoeck2011"),
}

#: Analyses and workflows a script calls after the run (``analysis=``).
_TIME_EVOLUTION_KEYS = ("Trotter1959", "Strang1968", "Suzuki1991")
_ANALYSIS_KEYS = {
    "tetrahedron-dos": ("JepsenAndersen1971", "LehmannTaut1972",
                        "BloechlJepsenAndersen1994"),
    "spectral-function": ("Lehmann1954", "Lanczos1950",
                          "GaglianoBalseiro1987"),
    "band-path": ("SetyawanCurtarolo2010",),
    "irreducible-zone": ("MonkhorstPack1976",),
    "time-evolution": _TIME_EVOLUTION_KEYS,
    "quantum-echoes": _TIME_EVOLUTION_KEYS + ("GoogleQuantumAI2025",),
    "nested-otoc": _TIME_EVOLUTION_KEYS + ("GoogleQuantumAI2025",),
    "wannier": ("Marzari1997", "Marzari2012"),
    # The downfolded many-body problem of Wannier functions: the truncated Coulomb interaction of
    # the fragment, and, with a screening, the constrained random-phase approximation in its
    # projector-weighted form.
    "downfold": ("Spencer2008",),
    "downfold-screening": ("Aryasetiawan2004", "Sasioglu2011", "Spencer2008"),
    "polarization": ("KingSmith1993", "Resta1994"),
    "isosurface": ("Kuhn1960", "Doi1991"),
}

#: Circuit template of ``method="vqe"`` (``ansatz=``) -> the papers that
#: define it.
_ANSATZ_KEYS = {
    "uccsd": ("Romero2019",),
    "hva": ("Wecker2015",),
}

_OPTIMIZER_KEYS = {
    "spsa": ("Spall1992",),
    "cobyla": ("Powell1994",),
    "nelder-mead": ("Nelder1965",),
    "slsqp": ("Kraft1988",),
    # L-BFGS is Liu & Nocedal; the bound-constrained extension Mandacaru
    # actually calls into (with no bounds) is Byrd et al.
    "l-bfgs": ("Liu1989", "Byrd1995"),
    "bfgs": ("Fletcher1970",),
    "cg": ("Polak1969",),
}

#: Pseudopotential family -> the papers its datasets are generated from.
_FAMILY_KEYS = {
    "paw-lcao": ("Bloechl1994", "Hamann2013"),
    "upaw-lcao": ("Bloechl1994", "Hamann2013", "Ivanov2025"),
}


#: What an ``active_space`` method cites.  The energy ordering is the canonical
#: orbital order the mean field already produced, so it adds nothing of its
#: own; the others are constructions with papers behind them.
#: The theory behind a kind of force (``citation_keys(forces=...)``).
_FORCE_KEYS = {
    # The z-vector and the Lagrangian correction of a parameter residual; the
    # coupled-perturbed Hartree-Fock orbital response; the assignment that follows
    # orbitals between geometries when an ansatz is transferred.
    "variational": ("HandySchaefer1984", "Helgaker1989", "Mitarai2020",
                    "GerrattMills1968", "Pople1979", "Kuhn1955", "Munkres1957"),
    # The analytic gradient of a mean-field energy: the Hellmann-Feynman term
    # and the basis-function (Pulay) term.
    "analytic": ("Feynman1939", "Pulay1969"),
}

_ACTIVE_METHOD_KEYS = {
    "energy": (),
    "mp2": ("Moller1934", "Sosa1989", "TaubeBartlett2005"),
    "dlpno-mp2": ("Moller1934", "Pinski2015", "FosterBoys1960", "Pulay1983",
                  "Sosa1989"),
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


#: What a dataset's recorded provenance cites.  These describe how the
#: *reference atom* was solved, not how the molecule is, so they are keyed off
#: the dataset rather than off the basis options -- a run that loads a
#: scalar-relativistic library cites Koelling and Harmon whether or not it
#: asked for one.
_PROVENANCE_KEYS = {
    ("xc", "pbe"): ["PBE1996", "PerdewWang1992"],
    ("relativity", "scalar"): ["KoellingHarmon1977"],
    ("relativity", "dirac"): ["KoellingHarmon1977", "Kleinman1980"],
    ("nlcc", True): ["Louie1982"],
    ("relativistic_exchange", True): ["MacDonaldVosko1979"],
}


def provenance_keys(datasets) -> list:
    """Keys for how the pseudopotential datasets of a run were generated.

    ``datasets`` is whatever the integrals carry -- each needs only ``xc``,
    ``relativity``, ``nlcc`` and ``relativistic_exchange``.  A dataset that records none of them predates
    the options and is the non-relativistic LDA construction, which the family
    key already covers.
    """
    keys: list = []
    for dataset in datasets or ():
        for field in ("xc", "relativity"):
            value = str(getattr(dataset, field, "") or "").lower()
            keys += _PROVENANCE_KEYS.get((field, value), [])
        if (getattr(dataset, "nlcc", None) or {}).get("applied"):
            keys += _PROVENANCE_KEYS[("nlcc", True)]
        if getattr(dataset, "relativistic_exchange", False):
            keys += _PROVENANCE_KEYS[("relativistic_exchange", True)]
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
    if options.get("filter_cutoff"):
        keys.append("Anglada2006")
    # Diffuse shells: the augmentation recipe, sized and cut like the
    # numerical orbitals of the confinement papers.
    if options.get("diffuse"):
        from ..pseudopotentials.diffuse import validate_diffuse
        if validate_diffuse(options["diffuse"]) is not None:
            keys += ["Kendall1992", "Artacho1999", "Junquera2001"]
    return keys


#: What a ``two_body`` representation cites (the dense default, ``None``, is the
#: integrals themselves and cites nothing); the Cholesky factors have their papers.
_TWO_BODY_KEYS = {
    "cholesky": ("BeebeLinderberg1977", "Koch2003", "Folkestad2019",
                 "Aquilante2008"),
}


#: What a ``two_body`` representation of the active-space tensor cites (the dense
#: default is the tensor itself).
_TWO_BODY_REPRESENTATION_KEYS = {
    "dense": (),
    "double-factorized": ("Peng2017", "Motta2021", "vonBurg2021"),
}


#: What a ``measurement_scheme`` cites (the commuting groups are the Pauli terms themselves).
_MEASUREMENT_SCHEME_KEYS = {
    "qwc": (),
    "double-factorized": ("Motta2021", "Huggins2021", "Kivlichan2018"),
}


def citation_keys(*, method=None, ansatz=None, pool=None, mapping=None,
                  basis=None,
                  family=None, basis_options=None, optimizer=None,
                  backend_provider=None, shots=0, execute_circuits=False,
                  profile=False, tetris=False, prune=False,
                  has_geometry=True, built_basis=True, extras=(),
                  datasets=(), active_method=None, two_body=None,
                  two_body_representation=None,
                  taper=False, measurement_scheme=None, periodic=False,
                  forces=None, xc=None, vdw=None, smearing=None, hubbard=(),
                  eigensolver=None, exxdiv_treatment=None, device=None,
                  trotter=False, analysis=()) -> list:
    """The bibliography keys a run with this configuration should cite.

    Every argument is optional: what is not known is not cited.  ``family`` and
    ``basis_options`` describe the basis that was actually *built* (the
    Hamiltonian builder's context), so they override what ``basis`` was
    spelled as; pass them when the run got that far.  ``extras`` adds keys for
    anything the caller knows about that this signature does not cover, such
    as an expressibility analysis. ``periodic`` marks a run on a crystal.
    ``forces="variational"`` marks a run that differentiates a variational quantum
    state: the z-vector and the Lagrangian formulation of its residual-gradient
    correction, and the derivative theory of variational quantum states;
    ``forces="analytic"`` one that differentiates a mean-field energy.
    ``xc`` and ``vdw`` name the functional and the dispersion correction (``xc``
    as the calculator resolves it, ``"pbe+rvv10"`` included), ``smearing`` the
    occupation broadening of a crystal, ``hubbard`` the Hubbard form or forms
    (``"dudarev-fll"``, ``"liechtenstein-fll"``) and ``"linear-response"`` when
    U was determined, ``eigensolver`` and ``exxdiv_treatment`` the crystal
    eigensolver and the exact-exchange divergence treatment, ``device`` the
    simulated device (``"AER_mps"`` is a matrix-product-state simulation) and
    ``trotter`` the product-of-exponentials form of the ansatz.  ``analysis``
    names what a script computed afterwards: ``"tetrahedron-dos"``,
    ``"spectral-function"``, ``"band-path"``, ``"irreducible-zone"``, ``"time-evolution"``, ``"quantum-echoes"``,
    ``"nested-otoc"``, ``"wannier"``, ``"downfold"`` (a Wannier fragment's many-body
    problem), ``"downfold-screening"`` (the same with ``screening=``), ``"polarization"`` or
    ``"isosurface"``.
    Circuit-measured energies (``shots`` or ``execute_circuits`` on a quantum
    method) also cite the parameter-shift gradient.

    The returned list is in the bibliography's own order, so the same run
    always produces the same file.
    """
    keys: list = ["Mandacaru", "Harris2020", "Virtanen2020"]
    if has_geometry:
        keys.append("Larsen2017")

    keys += list(_METHOD_KEYS.get(_dashed(method), ()))
    if periodic:
        keys += list(_PERIODIC_KEYS)
    keys += list(_ANSATZ_KEYS.get(_key(ansatz), ()))
    keys += list(_POOL_KEYS.get(_dashed(pool), ()))
    if tetris:
        keys.append("Anastasiou2024")
    if prune:
        keys.append("VaqueroSabater2025")
    keys += list(_MAPPING_KEYS.get(_key(mapping), ()))
    if taper:
        keys.append("Bravyi2017")
    keys += list(_OPTIMIZER_KEYS.get(_key(optimizer), ()))
    keys += list(_ACTIVE_METHOD_KEYS.get(_key(active_method), ()))
    keys += list(_TWO_BODY_KEYS.get(_key(two_body), ()))
    keys += list(_TWO_BODY_REPRESENTATION_KEYS.get(_key(two_body_representation), ()))
    keys += list(_MEASUREMENT_SCHEME_KEYS.get(_key(measurement_scheme), ()))
    keys += list(_FORCE_KEYS.get(_key(forces), ()))
    keys += list(_XC_KEYS.get(_key(xc), ()))
    keys += list(_VDW_KEYS.get(_key(vdw), ()))
    keys += list(_SMEARING_KEYS.get(_dashed(smearing), ()))
    for name in _names(hubbard):
        keys += list(_HUBBARD_KEYS.get(_dashed(name), ()))
    keys += list(_EIGENSOLVER_KEYS.get(_dashed(eigensolver), ()))
    keys += list(_EXXDIV_KEYS.get(_dashed(exxdiv_treatment), ()))
    keys += list(_DEVICE_KEYS.get(_device(device), ()))
    if trotter:
        keys += list(_TROTTER_KEYS)
    if (shots or execute_circuits) and _dashed(method) in _CIRCUIT_METHODS:
        keys += list(_PARAMETER_SHIFT_KEYS)
    for name in _names(analysis):
        keys += list(_ANALYSIS_KEYS.get(_dashed(name), ()))

    options = dict(basis_options or {})
    if family:
        keys += _pseudo_keys(_key(family), options)
        keys += provenance_keys(datasets)
    elif basis is not None:
        keys += _basis_keys(basis, options)

    if built_basis:
        # Every molecular Hamiltonian is built in the symmetrically
        # orthonormalized basis, from orbitals a DIIS-accelerated SCF produced,
        # whose level shift is armed against an energy that turns uphill.
        keys += ["Loewdin1950", "Pulay1980", "SaundersHillier1973"]

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


def _names(value) -> tuple:
    """``value`` as a tuple of names: a string is one name, ``None`` none."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _device(value):
    """The canonical device name of ``value``, or ``None`` for one this build
    does not know (an unknown device cites nothing; the calculator refuses it)."""
    if value is None:
        return None
    from ..backends.hardware import normalize_device
    try:
        return normalize_device(value)
    except ValueError:
        return None
