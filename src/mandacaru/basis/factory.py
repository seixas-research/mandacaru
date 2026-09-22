# -*- coding: utf-8 -*-
# file: basis/factory.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Unified factory for building localized basis sets.

``BasisSet.build`` selects a basis-set family and returns an object that turns
elements/geometries into lists of :class:`~mandacaru.basis.base.BasisFunction`
ready for :class:`~mandacaru.integrals.IntegralEngine`:

.. code-block:: python

    nao = BasisSet.build(method="NAO", energy_shift=0.03)
    gto = BasisSet.build(method="GTO", n_gaussians=3)          # STO-3G

    orbitals = gto.atom("O", center=[0.0, 0.0, 0.0])          # one atom
    basis = gto.molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])  # a geometry

The GTO family is generated from scratch (see :mod:`mandacaru.basis.sto_ng`): a
least-squares STO-nG fit of Gaussians to Slater-type orbitals, so it needs no
tabulated basis-set data.
"""

from __future__ import annotations

from collections.abc import Sequence

from ase.data import atomic_numbers

from .base import BasisFunction
from .gaussian import GaussianOrbital
from .gaussian_families import (GaussianRecipe, gaussian_shells,
                                parse_basis_name, shell_notation)
from .hao import HydrogenicAtomicOrbital
from .multizeta import (DEFAULT_NAO_SIZE, DEFAULT_SPLIT_NORM, build_shells,
                        orbitals_from_tables, resolve_split_scheme,
                        resolve_zeta)
from .nao import (DEFAULT_ENERGY_SHIFT, NumericalAtomicOrbital,
                  energy_shift_to_rc, solve_confined_radial)
from . import nao_ae
from .pople import pople_631g_shells
from .sto_ng import sto_ng_shells
from ._config import (ground_state_config, unoccupied_subshells,
                      valence_subshells)


def _to_atomic_number(element) -> int:
    """Accept an element symbol (``"O"``) or atomic number (``8``)."""
    if isinstance(element, str):
        try:
            return atomic_numbers[element.capitalize()]
        except KeyError:
            raise ValueError(f"unknown element symbol {element!r}") from None
    return int(element)


class BasisSet:
    """Base factory. Use :meth:`build` to obtain a concrete basis set."""

    method: str = ""

    @staticmethod
    def build(method, **kwargs) -> "BasisSet":
        """Construct a basis set of the requested ``method``.

        ``method`` may also be a **per-element mapping** -- a dict of chemical
        symbol (or ``"*"`` for the default) to a basis spec, each a name or a
        ``{"name": ..., <options>}`` dict -- giving a
        :class:`PerElementBasisSet` that uses a different family on different
        elements: ``BasisSet.build({"O": {"name": "NAO", "size": "DZP"}, "H":
        "6-31G", "*": "HAO"})``.

        Supported methods: ``"HAO"`` (Hydrogenic Atomic Orbitals -- the minimal
        analytic single-zeta atomic family),
        ``"NAO"`` (confined numerical atomic orbitals), ``"NAO-AE"``
        (all-electron numerical atomic orbitals with hydrogen-like tiers, see
        :mod:`mandacaru.basis.nao_ae`), ``"GTO"`` / ``"STO-nG"`` (native
        minimal Gaussian), and every **named Gaussian family** understood by
        :func:`~mandacaru.basis.gaussian_families.parse_basis_name` -- the Pople
        split-valence sets (``"6-31G"``, ``"6-31+G*"``, ``"6-311+G(2df,2p)"``,
        ``"3-21G"``, ...), the Dunning correlation-consistent sets
        (``"cc-pVDZ"`` ... ``"cc-pV5Z"``, ``"aug-cc-pVDZ"``, ``"cc-pCVDZ"``)
        and the Karlsruhe ``"def2-..."`` sets -- all generated natively with the
        published shell structure (see :mod:`mandacaru.basis.gaussian_families`).
        """
        if isinstance(method, dict):
            if kwargs:
                raise TypeError("a per-element mapping takes no extra options")
            return PerElementBasisSet(method)
        key = method.upper().replace(" ", "")
        if key in ("HAO", "FULLATOMICORBITALS"):
            return HAOBasisSet(**kwargs)
        if key == "NAO":
            return NAOBasisSet(**kwargs)
        if key in ("NAO-AE", "NAO_AE", "AE-NAO"):
            return NAOAEBasisSet(**kwargs)
        if key == "GTO":
            return GTOBasisSet(**kwargs)
        if key in ("STO-3G", "STO-4G", "STO4G", "STO-5G", "STO5G",
                   "STO-6G", "STO6G"):
            return GTOBasisSet(n_gaussians=int(key[3:].lstrip("-")[0]),
                               **kwargs)
        if key == "6-31G":
            return Pople631GBasisSet(polarization=False, **kwargs)
        if key in ("6-31G(D)", "631G(D)", "6-31G*", "631G*", "6-31GD"):
            return Pople631GBasisSet(polarization=True, **kwargs)
        # Every other named Gaussian family: Pople, Dunning, Karlsruhe.
        try:
            recipe = parse_basis_name(method)
        except ValueError as error:
            if ":" in str(method):
                # A namespaced name ('published:cc-pVTZ'); the grammar's own
                # message explains the namespace and must not be replaced.
                raise
            raise ValueError(
                f"unknown basis method {method!r}; use 'HAO', 'NAO', 'NAO-AE', "
                f"'GTO', an STO-nG name, or a named Gaussian family such as "
                f"'6-31+G*', '6-311+G(2df,2p)', 'cc-pVTZ', 'aug-cc-pVDZ' or "
                f"'def2-TZVP'") from error
        return GaussianBasisSet(recipe, **kwargs)

    # -- interface --------------------------------------------------------- #

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        """Basis functions for one atom (``element`` symbol or Z) at ``center``."""
        raise NotImplementedError

    def molecule(self, symbols: Sequence, positions: Sequence,
                 units: str = "angstrom") -> list[BasisFunction]:
        """Concatenated basis functions for a whole geometry."""
        basis: list[BasisFunction] = []
        for sym, pos in zip(symbols, positions):
            basis.extend(self.atom(sym, center=pos, units=units))
        return basis


class PerElementBasisSet(BasisSet):
    """A different basis family on different elements.

    Built from a mapping of chemical symbol to basis spec (name or
    ``{"name": ..., <options>}`` dict), with ``"*"`` as the default for
    elements not listed.  Each element's functions come from that family's own
    :class:`BasisSet`, so a polarized double-zeta water next to a minimal
    sodium ion is ``{"O": {"name": "NAO", "size": "DZP"}, "H": {"name": "NAO",
    "size": "DZP"}, "Na": "HAO"}``.  Plane waves, which are not atom-centered,
    are refused.
    """

    method = "per-element"

    def __init__(self, mapping: dict):
        from ..algorithms._hamiltonian_from_atoms import (
            DEFAULT_ELEMENT_KEY, is_per_element_basis, resolve_basis)
        if not is_per_element_basis(mapping):
            raise ValueError(
                "expected a mapping of chemical symbols (or '*') to basis "
                "specs, e.g. {'O': 'HAO', 'H': '6-31G'}")
        self.mapping = {(k if k == DEFAULT_ELEMENT_KEY else k.capitalize()): v
                        for k, v in mapping.items()}
        self._default_key = DEFAULT_ELEMENT_KEY
        self._resolve = resolve_basis
        self._sets: dict[str, BasisSet] = {}
        labels = []
        for symbol, spec in self.mapping.items():
            name, options = resolve_basis(spec)
            labels.append(f"{symbol}: {name}"
                          + (f" {options}" if options else ""))
        self.name = "per-element {" + ", ".join(labels) + "}"

    def family_for(self, element) -> BasisSet:
        """The :class:`BasisSet` that serves ``element``."""
        from ase.data import chemical_symbols
        symbol = (chemical_symbols[element] if isinstance(element, int)
                  else str(element).capitalize())
        if symbol not in self._sets:
            spec = self.mapping.get(symbol, self.mapping.get(self._default_key))
            if spec is None:
                raise ValueError(
                    f"no basis given for element {symbol!r}; add it to the "
                    f"per-element mapping or a '{self._default_key}' default")
            name, options = self._resolve(spec)
            if isinstance(name, str) and name.upper().replace("-", "") in (
                    "PW", "PLANEWAVE"):
                raise ValueError("the plane-wave basis is not atom-centered "
                                 "and cannot be assigned to one element")
            self._sets[symbol] = BasisSet.build(name, **options)
        return self._sets[symbol]

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        return self.family_for(element).atom(element, center=center,
                                             units=units)

    def __repr__(self) -> str:
        return f"PerElementBasisSet({self.mapping!r})"


class NAOBasisSet(BasisSet):
    """Confined Numerical Atomic Orbitals (SIESTA/Sankey-type).

    Generates one radial function per valence subshell (all ``m``), each solved
    in a hard-wall sphere of radius ``r_c``.  The radial potential uses the
    Slater effective charge of the subshell, so valence orbitals of
    many-electron atoms have realistic extents.

    Multiple zetas and polarization shells are available through ``size``,
    giving the usual SZ / DZ / DZP / TZP / ... hierarchy.  The extra zetas come
    from the split-valence construction and the polarization shells from an
    ``l+1`` solve in the same sphere -- see :mod:`mandacaru.basis.multizeta`.

    Parameters
    ----------
    energy_shift : float
        Confinement energy shift in eV (default ``0.03``); sets ``r_c``.
    r_c : float, optional
        Confinement radius in Bohr; overrides ``energy_shift`` when given.
    n_grid : int
        Radial finite-difference resolution.
    size : str or (int, int)
        Basis size: ``"DZP"`` (default), ``"SZ"``, ``"DZ"``, ``"TZP"``,
        ``"TZ2P"``, ``"QZP"``, ... or an explicit
        ``(n_zeta, n_polarization)`` pair.  The default is double-zeta plus
        polarization because single zeta has no radial or angular freedom at
        all; pass ``size="SZ"`` for the older, much cheaper minimal basis.
    tail_norm : float or sequence
        GPAW's split-valence scheme, **the default** (``(0.16, 0.3, 0.6)``):
        the *norm* of the tail each extra zeta leaves outside its split
        radius, every zeta split from the first one.
    split_norm : float
        Selects the SIESTA-style scheme instead: the fraction of the orbital's
        *squared* norm left outside the split radius (SIESTA's default is
        ``0.15``), each zeta split from the previous one with the fraction
        halved.  Mutually exclusive with ``tail_norm``.
    """

    method = "NAO"

    def __init__(self, energy_shift: float = DEFAULT_ENERGY_SHIFT,
                 r_c: float | None = None, n_grid: int = 2000,
                 size=DEFAULT_NAO_SIZE, split_norm: float | None = None,
                 tail_norm=None):
        self.energy_shift = energy_shift
        self.r_c = float(r_c) if r_c is not None \
            else energy_shift_to_rc(energy_shift)
        self.n_grid = n_grid
        self.size = size
        self.n_zeta, self.n_polarization = resolve_zeta(size)
        self.split_norm, self.tail_norms = resolve_split_scheme(split_norm,
                                                                tail_norm)

    def _solver(self, Z):
        """``(n, l) -> (r, R)`` for a confined orbital of this atom.

        Slater's rules are defined only for *occupied* subshells, but a
        polarization shell is unoccupied by construction.  For those the
        screening of the outermost occupied subshell is reused: the polarizing
        function lives in the same region of the atom and sees essentially the
        same effective charge, which is what makes it a useful partner to the
        orbital it polarizes.
        """
        valence = valence_subshells(Z)
        fallback_state = max(valence)               # outermost occupied (n, l)
        fallback = HydrogenicAtomicOrbital.slater_effective_charge(Z, *fallback_state)

        def solve(n, l):
            try:
                z_eff = HydrogenicAtomicOrbital.slater_effective_charge(Z, n, l)
            except ValueError:
                z_eff = fallback                    # unoccupied: polarization
            r, radial, _energy = solve_confined_radial(n, l, z_eff, self.r_c,
                                                       self.n_grid)
            return r, radial
        return solve

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        Z = _to_atomic_number(element)
        if self.n_zeta == 1 and self.n_polarization == 0:
            # Single zeta, no polarization: the original path, unchanged.
            orbitals: list[BasisFunction] = []
            for (n, l) in valence_subshells(Z):
                z_eff = HydrogenicAtomicOrbital.slater_effective_charge(Z, n, l)
                for m in range(-l, l + 1):
                    orbitals.append(NumericalAtomicOrbital(
                        n, l, m, Z=z_eff, r_c=self.r_c, center=center,
                        units=units, n_grid=self.n_grid))
            return orbitals

        tables = build_shells(valence_subshells(Z), self._solver(Z),
                              n_zeta=self.n_zeta,
                              n_polarization=self.n_polarization,
                              split_norm=(DEFAULT_SPLIT_NORM
                                          if self.split_norm is None
                                          else self.split_norm),
                              tail_norms=self.tail_norms)
        return orbitals_from_tables(tables, center=center, units=units)

    def __repr__(self) -> str:
        return (f"NAOBasisSet(size={self.size!r}, "
                f"energy_shift={self.energy_shift}, r_c={self.r_c:.3f})")


class NAOAEBasisSet(BasisSet):
    r"""All-electron numerical atomic orbitals with hydrogen-like tiers.

    The minimal basis is every occupied shell of the self-consistent LDA atom
    (core included), re-solved under a smooth confining wall; ``tier`` adds
    hydrogen-like polarization / diffuse / contracted functions sized from the
    atom's own valence radius; each ``l`` channel is Gram-Schmidt
    orthonormalized.  See :mod:`mandacaru.basis.nao_ae` for the construction.

    Parameters
    ----------
    tier : int
        ``0`` -- minimal (atomic orbitals only); ``1`` (default) -- plus one
        polarization shell and one diffuse function per valence channel;
        ``2`` -- plus a second polarization channel, a noded polarization
        function and one contracted function per valence channel.
    onset, width : float
        Confinement onset and ramp width in Ångström (defaults ``3.0`` /
        ``1.0``); every function is exactly zero beyond ``onset + width``.
        The real-space box must extend at least that far around every atom.
    scale : float
        Wall strength (Hartree * Bohr^2, default ``1.0``).
    extra : sequence of tuple, optional
        Additional hydrogen-like functions with explicit effective charges.
    linear_dependence_tol : float
        Gram-Schmidt rejection threshold on the residual norm (``1e-4``).
    tail_norm : float or None
        Tier functions are shortened until at most this fraction of their norm
        lies beyond the onset (``1e-4``; ``None`` disables).
    points, r_max : int, float
        Radial grid of the atomic solver (see
        :func:`~mandacaru.basis.atomic_solver.solve_atom`).
    """

    method = "NAO-AE"
    name = "NAO-AE"

    def __init__(self, tier: int = nao_ae.DEFAULT_TIER,
                 onset: float = nao_ae.DEFAULT_ONSET,
                 width: float = nao_ae.DEFAULT_WIDTH,
                 scale: float = nao_ae.DEFAULT_SCALE, extra=None,
                 linear_dependence_tol: float = nao_ae.DEFAULT_LINEAR_DEPENDENCE_TOL,
                 tail_norm: float | None = nao_ae.DEFAULT_TAIL_NORM,
                 points: int = nao_ae.DEFAULT_POINTS,
                 r_max: float = nao_ae.DEFAULT_R_MAX):
        self.tier = int(tier)
        if not (0 <= self.tier <= nao_ae.MAX_TIER):
            raise ValueError(
                f"tier must be in [0, {nao_ae.MAX_TIER}], got {tier}")
        self.onset = float(onset)
        self.width = float(width)
        self.scale = float(scale)
        self.extra = (None if extra is None
                      else tuple((int(n), int(l), float(z)) for n, l, z in extra))
        self.linear_dependence_tol = float(linear_dependence_tol)
        self.tail_norm = None if tail_norm is None else float(tail_norm)
        self.points = int(points)
        self.r_max = float(r_max)
        self._species: dict[int, list] = {}

    @property
    def wall_radius(self) -> float:
        """Radius (Bohr) beyond which every function is exactly zero."""
        from ..units import to_bohr
        return float(to_bohr(self.onset + self.width, "angstrom"))

    def species(self, element) -> list:
        """The orthonormalized :class:`~mandacaru.basis.nao_ae.RadialFunction` list."""
        Z = _to_atomic_number(element)
        if Z not in self._species:
            self._species[Z] = nao_ae.build_species(
                Z, tier=self.tier, onset=self.onset, width=self.width,
                scale=self.scale, extra=self.extra,
                linear_dependence_tol=self.linear_dependence_tol,
                tail_norm=self.tail_norm, points=self.points,
                r_max=self.r_max, units="angstrom")
        return self._species[Z]

    def tables(self, element) -> list:
        """Radial tables of ``element`` (see :func:`~mandacaru.basis.nao_ae.radial_tables`)."""
        return nao_ae.radial_tables(self.species(element), self.wall_radius)

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        return orbitals_from_tables(self.tables(element), center=center,
                                    units=units)

    def describe(self, element) -> str:
        """Human-readable listing of the radial functions of ``element``."""
        return nao_ae.describe_species(self.species(element))

    def __repr__(self) -> str:
        return (f"NAOAEBasisSet(tier={self.tier}, onset={self.onset}, "
                f"width={self.width})")


class GTOBasisSet(BasisSet):
    """Minimal STO-nG Gaussian-Type Orbitals, generated natively from scratch.

    One contracted Gaussian per occupied atomic subshell (core and valence),
    each a least-squares STO-nG fit of ``n_gaussians`` primitives to the Slater
    orbital of the subshell (:mod:`mandacaru.basis.sto_ng`).  No tabulated
    basis-set data is used.

    Parameters
    ----------
    n_gaussians : int
        Number of primitive Gaussians per contraction (the ``n`` of STO-nG,
        default ``3`` -> an STO-3G-like minimal basis).
    """

    method = "GTO"

    def __init__(self, n_gaussians: int = 3):
        self.n_gaussians = int(n_gaussians)
        self.name = f"STO-{self.n_gaussians}G"
        #: ``"native:<name>-recipe"`` -- generated here, not published data.
        self.provenance = f"native:{self.name}-recipe"

    def shells(self, element) -> list:
        """``(l, exponents, coefficients)`` contracted shells of ``element``."""
        return [(l, exps, coeffs) for (_n, l, exps, coeffs)
                in sto_ng_shells(_to_atomic_number(element), self.n_gaussians)]

    def to_dict(self, elements) -> dict:
        """Every generated exponent and coefficient, with its conventions."""
        from .gaussian_families import shells_record
        return shells_record(elements, self.shells, self.name, "sto",
                             f"minimal STO-{self.n_gaussians}G, one "
                             f"{self.n_gaussians}-primitive contraction per "
                             "occupied subshell")

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        Z = _to_atomic_number(element)
        orbitals: list[BasisFunction] = []
        for (_n, l, exps, coeffs) in sto_ng_shells(Z, self.n_gaussians):
            for m in range(-l, l + 1):
                orbitals.append(GaussianOrbital(
                    l, m, exps, coeffs, center=center, units=units))
        return orbitals

    def __repr__(self) -> str:
        return (f"GTOBasisSet(name={self.name!r}, "
                f"provenance={self.provenance!r})")


class HAOBasisSet(BasisSet):
    r"""Analytic Hydrogenic Atomic Orbitals: one orbital per occupied subshell.

    For each occupied ``(n, l)`` subshell of the atom, builds the ``2l + 1``
    :class:`~mandacaru.basis.HydrogenicAtomicOrbital` functions with the **actual atomic
    number** ``Z`` as the orbital's nuclear charge (the bare hydrogenic orbital of
    the element -- no Slater screening).  A cheap, fully analytic reference basis
    (e.g. H -> 1s; Li -> 1s, 2s; C -> 1s, 2s, 2p).

    **Virtual levels.** ``virtual_orbitals=k`` appends the ``k`` lowest
    *unoccupied* subshells of each atom, in aufbau order
    (:func:`~mandacaru.basis._config.unoccupied_subshells`) and with the same bare
    ``Z``: hydrogen gains ``2s``, carbon ``3s``, iron ``4p``.  The occupied set
    is untouched, so ``virtual_orbitals=0`` (the default) is the historical
    minimal basis exactly.

    A **level is a whole subshell**, so one virtual ``p`` level adds three
    functions, not one -- the count is of levels, matching "one orbital per
    subshell" above.  Half a shell would break the atom's spherical symmetry and
    make the energy depend on how the molecule is oriented in the box, so the
    shells are always complete.  :meth:`function_count` reports what a given
    ``k`` actually costs.

    Why it matters: the minimal basis has no room above the occupied orbitals,
    so a correlated method has almost nothing to correlate *into* -- on H2 the
    occupied-only HAO basis gives 2 spatial orbitals (4 qubits) and a single
    double excitation.  Each virtual level widens that active space, lowering
    the variational energy at the cost of more qubits.

    .. note::

       A virtual hydrogenic orbital is diffuse -- H ``2s`` has
       :math:`\langle r \rangle = 6\,a_0 \approx 3.2` Angstrom -- so the cell
       must be large enough to contain it, or the grid clips its tail.  The
       engine's resolution check warns when a function is not represented.

    Parameters
    ----------
    virtual_orbitals : int
        Number of unoccupied subshells to append per atom (default ``0``).
    """

    method = "HAO"
    name = "HAO"

    def __init__(self, virtual_orbitals: int = 0):
        if isinstance(virtual_orbitals, bool):
            raise TypeError("virtual_orbitals counts subshells; pass an int, "
                            f"not {virtual_orbitals!r}")
        try:
            count = int(virtual_orbitals)
        except (TypeError, ValueError):
            raise TypeError(
                f"virtual_orbitals must be an integer, got "
                f"{type(virtual_orbitals).__name__}") from None
        if count != virtual_orbitals:
            raise ValueError(
                f"virtual_orbitals must be a whole number of subshells, got "
                f"{virtual_orbitals!r}")
        if count < 0:
            raise ValueError(
                f"virtual_orbitals must be >= 0, got {count}")
        #: Unoccupied subshells appended to every atom.
        self.virtual_orbitals = count

    def subshells(self, element) -> list[tuple[int, int]]:
        """The ``(n, l)`` levels of ``element``: occupied, then the virtual ones.

        The occupied part keeps its historical ``(n, l)``-sorted order; the
        virtual levels follow in aufbau order.
        """
        Z = _to_atomic_number(element)
        return (sorted(ground_state_config(Z))
                + unoccupied_subshells(Z, self.virtual_orbitals))

    def function_count(self, element) -> int:
        """Basis functions for one atom -- ``sum(2l + 1)`` over its levels."""
        return sum(2 * l + 1 for (_n, l) in self.subshells(element))

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        Z = _to_atomic_number(element)
        orbitals: list[BasisFunction] = []
        for (n, l) in self.subshells(element):
            for m in range(-l, l + 1):
                orbitals.append(HydrogenicAtomicOrbital(n, l, m, Z=float(Z),
                                                  center=center, units=units))
        return orbitals

    def __repr__(self) -> str:
        return f"HAOBasisSet(virtual_orbitals={self.virtual_orbitals})"



class GaussianBasisSet(BasisSet):
    """A named Gaussian basis set (Pople / Dunning / Karlsruhe / STO-nG).

    Built from a :class:`~mandacaru.basis.gaussian_families.GaussianRecipe` --
    the structure the name encodes -- with exponents and contraction
    coefficients generated natively per atom (see
    :mod:`mandacaru.basis.gaussian_families`).  ``BasisSet.build("cc-pVTZ")``,
    ``BasisSet.build("def2-SVP")`` and ``BasisSet.build("6-311+G(2df,2p)")``
    all land here.

    The name buys the published basis set's **shell structure**, not its
    published numbers, and :attr:`provenance` says which -- ``cc-pVTZ`` here is
    ``native:cc-pVTZ-recipe``.  :meth:`to_dict` writes out every exponent and
    coefficient together with the angular convention they are meant in, which is
    what a comparison against another package needs.  ``published:`` is a
    reserved namespace that resolves to nothing: Mandacaru ships no basis tables.

    Parameters
    ----------
    recipe : GaussianRecipe or str
        The recipe, or a basis-set name to parse.
    """

    method = "gaussian"

    def __init__(self, recipe):
        if isinstance(recipe, str):
            recipe = parse_basis_name(recipe)
        if not isinstance(recipe, GaussianRecipe):
            raise TypeError("recipe must be a GaussianRecipe or a basis name")
        self.recipe = recipe
        self.name = recipe.name
        self.family = recipe.family
        #: ``"native:<name>-recipe"`` -- generated here, not published data.
        self.provenance = recipe.provenance

    def shells(self, element) -> list:
        """``(l, exponents, coefficients)`` contracted shells of ``element``."""
        return gaussian_shells(_to_atomic_number(element), self.recipe)

    def notation(self, element) -> str:
        """Contracted-shell notation, e.g. ``"[4s3p2d1f]"`` for cc-pVTZ carbon."""
        return shell_notation(_to_atomic_number(element), self.recipe)

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        orbitals: list[BasisFunction] = []
        for (l, exps, coeffs) in self.shells(element):
            for m in range(-l, l + 1):
                orbitals.append(GaussianOrbital(l, m, exps, coeffs,
                                                center=center, units=units))
        return orbitals

    def to_dict(self, elements) -> dict:
        """Every generated exponent and coefficient, with its conventions.

        See :func:`~mandacaru.basis.gaussian_families.basis_set_data`; ``elements``
        are chemical symbols or atomic numbers.  JSON-serializable.
        """
        from .gaussian_families import basis_set_data
        return basis_set_data(elements, self.recipe)

    def __repr__(self) -> str:
        return (f"GaussianBasisSet(name={self.name!r}, "
                f"provenance={self.provenance!r})")


class Pople631GBasisSet(BasisSet):
    """Native Pople split-valence basis 6-31G / 6-31G(d), generated from scratch.

    Core subshells are single 6-primitive contractions; valence subshells are
    split into a contracted inner + uncontracted outer function; with
    ``polarization=True`` a ``d`` shell is added on non-hydrogen atoms (the
    ``(d)`` of 6-31G(d)).  See :mod:`mandacaru.basis.pople`; no tabulated basis-set
    data is used.

    Parameters
    ----------
    polarization : bool
        Add ``d`` polarization functions on non-hydrogen atoms (``6-31G(d)`` if
        ``True``, plain ``6-31G`` if ``False``).
    """

    method = "6-31G"

    def __init__(self, polarization: bool = True):
        self.polarization = bool(polarization)
        self.name = "6-31G(d)" if polarization else "6-31G"
        #: ``"native:<name>-recipe"`` -- generated here, not published data.
        self.provenance = f"native:{self.name}-recipe"

    def shells(self, element) -> list:
        """``(l, exponents, coefficients)`` contracted shells of ``element``."""
        return pople_631g_shells(_to_atomic_number(element), self.polarization)

    def to_dict(self, elements) -> dict:
        """Every generated exponent and coefficient, with its conventions."""
        from .gaussian_families import shells_record
        return shells_record(elements, self.shells, self.name, "pople",
                             "6-primitive cores, 3+1 split valence"
                             + (", d polarization on Z > 2"
                                if self.polarization else ""))

    def atom(self, element, center=(0.0, 0.0, 0.0),
             units: str = "angstrom") -> list[BasisFunction]:
        Z = _to_atomic_number(element)
        orbitals: list[BasisFunction] = []
        for (l, exps, coeffs) in pople_631g_shells(Z, self.polarization):
            for m in range(-l, l + 1):
                orbitals.append(GaussianOrbital(l, m, exps, coeffs,
                                                center=center, units=units))
        return orbitals

    def __repr__(self) -> str:
        return (f"Pople631GBasisSet(name={self.name!r}, "
                f"provenance={self.provenance!r})")
