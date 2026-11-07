# -*- coding: utf-8 -*-
# file: pseudopotentials/orbitals.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Pseudo-atomic orbitals and Kleinman-Bylander projector functions.

A pseudopotential calculation needs two families of radial functions sampled on
the grid:

* the **pseudo-atomic orbitals** :math:`R^{\text{ps}}_{l}(r)\,Y_{lm}`, which form
  the valence basis.  They are the natural choice -- they are exactly the states
  the pseudopotential was built to reproduce, they are nodeless and smooth, and
  they contain no core, so the basis size drops with the electron count;
* the **separable projectors** :math:`\chi_{l}(r)\,Y_{lm}`, which carry
  the nonlocal part of the potential.

Both are radial tables of a family's dataset (:mod:`.dataset`), splined and
multiplied by a spherical harmonic -- the same construction
:class:`~mandacaru.basis.nao.NumericalAtomicOrbital` uses, so they drop into the
integral engine unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _native
from ..units import to_bohr
from ..basis._angular import sample_dtype as _dtype
from ..basis._angular import (radial_angular_gradient, spherical_coords,
                              spherical_harmonic)
from ..basis.base import BasisFunction
from .native_context import context as _context


@dataclass(frozen=True)
class _Coefficients:
    """Breakpoints ``x`` and cubic coefficients ``c`` ``(4, n - 1)`` of a
    radial table, as the native sampling kernels read them."""

    x: np.ndarray
    c: np.ndarray


class _RadialTabulated(BasisFunction):
    """Shared machinery: spline a radial table and attach an angular factor."""

    def __init__(self, r, radial, l: int, m: int, center=None,
                 units: str = "angstrom", *, real: bool):
        if abs(int(m)) > int(l):
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.l, self.m = int(l), int(m)
        #: Real (molecule) or complex (crystal) spherical harmonics.
        self.real = bool(real)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))
        self._r = np.asarray(r, dtype=float)
        self._values = np.asarray(radial, dtype=float)
        self._r_max = float(self._r[-1])
        #: The table's piecewise-cubic coefficients, ``.x`` breakpoints and
        #: ``.c`` of shape ``(4, n - 1)`` (highest power first), for the native
        #: sampling kernels; the native routine also validates the table.
        self._spline = _Coefficients(
            self._r, _native.radial_table_coefficients(
                _context(), self._r, self._values))

    def _table(self, r, slope: bool) -> np.ndarray:
        nodes = np.asarray(r, dtype=float)
        out = _native.radial_table_evaluate(
            _context(), self._r, self._values, self.l,
            np.ascontiguousarray(nodes).ravel(), slope)
        return out.reshape(nodes.shape)

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function; zero beyond the tabulated range.

        Below the first table point the function continues as
        ``R(r_0) (r / r_0)^l``, as :meth:`~mandacaru.basis.multizeta.
        TabulatedOrbital.radial` does.  A constant continuation would leave an
        ``l > 0`` function finite at its own nucleus, where the angular factor
        is undefined: a grid node on the nucleus then sampled ``R(r_0)``
        times a direction set by round-off, and the same Bloch sum came out
        with opposite signs in two equivalent cells.
        """
        return self._table(r, False)

    def radial_slope(self, r) -> np.ndarray:
        """``dR/dr`` of :meth:`radial`: the spline's derivative, and the
        derivative of the power-law continuation below the first table
        point."""
        return self._table(r, True)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample the function on Cartesian coordinates (Bohr)."""
        r, theta, phi = spherical_coords(x, y, z, self.center)
        return np.asarray(self.radial(r)
                          * spherical_harmonic(self.l, self.m, theta, phi,
                                               real=self.real),
                          dtype=_dtype(self.real))

    def gradient(self, x, y, z, delta: float = 1e-3) -> np.ndarray:
        """Analytic gradient (``delta`` is unused)."""
        return radial_angular_gradient(
            self.radial, self.radial_slope, self.l, self.m,
            np.asarray(x, dtype=float) - self.center[0],
            np.asarray(y, dtype=float) - self.center[1],
            np.asarray(z, dtype=float) - self.center[2], real=self.real)


class PseudoAtomicOrbital(_RadialTabulated):
    r"""A valence pseudo-atomic orbital :math:`R^{\text{ps}}_{l}(r)Y_{lm}`.

    Parameters
    ----------
    pseudopotential : PseudoPotential
        The element's pseudopotential; supplies the radial table.
    l, m : int
        Angular momentum and magnetic quantum number.
    center : array_like
        Cartesian center in ``units``.
    radial : array_like, optional
        Replaces the channel's tabulated radial function (on the same radial
        grid).  The one caller is the Fourier filter
        (:mod:`mandacaru.basis.filtering`), which hands over the band-limited
        version of exactly this function; everything else -- the quantum
        numbers, the reference eigenvalue, the class -- is unchanged, so the
        filter is a change of *basis function*, never of pseudopotential.
    index : int
        The channel's partial wave the orbital is: 0, or 1 for a semicore
        channel's valence state (sodium's 3s next to its 2s).
    eigenvalue : float, optional
        The orbital's level when it is not the channel's reference energy:
        an empty channel's bound state (calcium's 3d).
    """

    def __init__(self, pseudopotential, l: int, m: int, center=None,
                 units: str = "angstrom", radial=None, index: int = 0,
                 eigenvalue: float | None = None, *, real: bool = True):
        channel = pseudopotential.channels[int(l)]
        index = int(index)
        super().__init__(pseudopotential.r,
                         channel.pseudo_waves[index] if radial is None
                         else radial, l, m, center, units, real=real)
        self.symbol = pseudopotential.symbol
        self.n = int(channel.n) + index
        self.eigenvalue = float(channel.reference_energies[index]
                                if eigenvalue is None else eigenvalue)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    def __repr__(self) -> str:
        return (f"PseudoAtomicOrbital({self.symbol}, n={self.n}, l={self.l}, "
                f"m={self.m}, eps={self.eigenvalue:.4f})")


class KBProjector(_RadialTabulated):
    r"""A separable projector :math:`\chi_{l}(r)Y_{lm}`.

    Carries the diagonal coupling element :attr:`kb_energy` and the index of
    the atom it belongs to, which is what the nonlocal matrix and its nuclear
    derivative need.

    Every projector also carries the three labels the **general separable
    form** of :meth:`mandacaru.core.hamiltonian.MolecularIntegrals.kb_nonlocal`
    keys on: :attr:`atom_index`, the :attr:`channel` ``(l, m)`` and, within
    that channel, its radial :attr:`index`.  The nonlocal coupling matrix
    :math:`D` is block-diagonal over ``(atom, l, m)``; a family with several
    radial projectors per channel (PAW-LCAO) numbers them with
    ``index`` and supplies the block.
    """

    def __init__(self, pseudopotential, l: int, m: int, center=None,
                 units: str = "angstrom", atom_index: int = 0,
                 index: int = 0, *, radial, kb_energy: float,
                 real: bool = True):
        chi = np.asarray(radial, dtype=float)
        super().__init__(pseudopotential.r, chi, l, m, center, units,
                         real=real)
        self.symbol = pseudopotential.symbol
        self.kb_energy = float(kb_energy)
        self.atom_index = int(atom_index)
        #: Radial projector index within the ``(atom, l, m)`` channel.
        self.index = int(index)
        channel = pseudopotential.channels.get(int(l))
        self.channel_r_cut = (float("nan") if channel is None
                              else float(channel.r_cut))
        # The projector's support: a PAW-LCAO projector reaches out to the
        # local potential's radius when that exceeds the channel's cutoff.
        radius = getattr(pseudopotential, "projector_radius", None)
        self.r_cut = (float("nan") if channel is None
                      else float(radius(l)) if radius is not None
                      else float(channel.r_cut))

    @property
    def channel(self) -> tuple[int, int]:
        """The ``(l, m)`` channel this projector belongs to."""
        return (self.l, self.m)

    @property
    def block_key(self) -> tuple[int, int, int]:
        """``(atom_index, l, m)`` -- the key of its block in the coupling matrix."""
        return (self.atom_index, self.l, self.m)

    def __repr__(self) -> str:
        return (f"KBProjector({self.symbol}, l={self.l}, m={self.m}, "
                f"E_KB={self.kb_energy:+.4f}, atom={self.atom_index}, "
                f"index={self.index})")


# --------------------------------------------------------------------------- #
# Building the valence basis and the projector set for a molecule.
# --------------------------------------------------------------------------- #

class PseudoShell(_RadialTabulated):
    """One ``m`` component of a zeta, polarization or diffuse shell of a
    PAW-LCAO basis: the shell's :class:`~mandacaru.basis.multizeta.RadialTable`
    (kept as :attr:`table`) times a harmonic, evaluated by the native table
    routines like every other function of the pseudopotential basis."""

    def __init__(self, table, m: int, center=None, units: str = "angstrom",
                 *, real: bool):
        super().__init__(table.r, table.values, table.l, m, center, units,
                         real=real)
        self.table = table
        self.n, self.zeta = int(table.n), int(table.zeta)
        self.polarization = bool(table.polarization)
        self.diffuse = bool(table.diffuse)
        #: Where the table ends and the function is zero.
        self.r_c = float(table.r[-1])

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    @property
    def kind(self) -> str:
        """``"diffuse"``, ``"polarization"`` or ``"zeta"``."""
        return ("diffuse" if self.diffuse
                else "polarization" if self.polarization else "zeta")

    def __repr__(self) -> str:
        kind = ("diffuse" if self.diffuse
                else "pol" if self.polarization else f"zeta{self.zeta}")
        return (f"PseudoShell(n={self.n}, l={self.l}, m={self.m}, {kind}, "
                f"r_c={self.r_c:.2f})")


def _shells(tables, center, units: str, real: bool) -> list[PseudoShell]:
    """Every ``m`` component of every radial table, in table order."""
    return [PseudoShell(table, m, center=center, units=units, real=real)
            for table in tables for m in range(-table.l, table.l + 1)]


def _orbitals(pp, orbitals, center, units: str, real: bool) -> list:
    """Every ``m`` component of the ``(state, radial)`` pairs of an element's
    minimal set."""
    return [PseudoAtomicOrbital(pp, state.l, m, center=center, units=units,
                                radial=radial, index=state.index,
                                eigenvalue=state.level, real=real)
            for state, radial in orbitals for m in range(-state.l, state.l + 1)]


def pseudo_basis(symbols, positions, potentials, units: str = "angstrom",
                 size="SZ", split_norm=None, filter_wavevector=None,
                 first_zeta=None, tail_norms=None, polarization_shape=None,
                 diffuse_shells=None, *, real: bool = True):
    """Valence pseudo-atomic basis for a molecule (real spherical
    harmonics) or, ``real=False``, a crystal (complex ones).

    Returns ``(functions, atom_of_orbital)``: the orbitals and which atom each
    belongs to.

    ``size`` selects the multiple-zeta / polarized hierarchy exactly as it does
    for the all-electron NAO family (``"SZ"``, ``"DZ"``, ``"DZP"``, ``"TZP"``,
    ...); a ``{symbol: size}`` mapping (with an optional ``"*"`` default)
    gives each element its own size.  The **first zeta always comes from the pseudopotential** and cannot be
    replaced: the dataset's construction pseudizes each valence orbital
    inside its cutoff radius and the projectors are built from
    those specific pseudo-orbitals, so pairing the potential with an unrelated
    all-electron radial function would be inconsistent.  The extra zetas are
    split-valence refinements *of* that pseudo-orbital, and the polarization
    shell is split from the highest occupied channel, so everything stays
    matched to the potential it came from.

    ``first_zeta`` (optional) is a callable ``(symbol, potential, l) -> R(r)``
    on the potential's radial table, or ``None`` to keep the stored orbital.
    It is how a family supplies a **confined** first zeta -- the
    ``energy_shift`` of :mod:`~.confinement` -- and it is the one sanctioned
    replacement of the pseudopotential's own orbital: the function it returns
    solves the *same* pseudo-Hamiltonian (same projectors, same couplings) in a
    confining potential, so it stays matched to the operator in the sense the
    paragraph above requires.  The extra zetas and the polarization shell are
    then split from the confined orbital and inherit its cutoff radius.

    ``tail_norms`` (a sequence) selects ``zeta_split="first_zeta"``: every
    extra zeta split from the *first* one, at the radius leaving a tail of
    that **norm**; ``None`` is ``"last_zeta"``, each zeta split from the
    previous one by ``split_norm`` (see
    :func:`~mandacaru.basis.multizeta.zeta_tables`, which runs the native
    construction).

    ``polarization_shape`` (optional) is a callable ``(symbol, potential) ->
    (l, R(r))`` -- or ``None`` to decline for that element -- that replaces the polarization shell built here --
    :func:`~.confinement.polarization_factory` supplies the quasi-Gaussian.
    Further polarization functions are then split-valence refinements of that
    shape with the *same* ``l``, instead of shells of higher ``l``.

    ``diffuse_shells`` (optional) is a callable ``(symbol, potential,
    existing, filter_wavevector) -> [RadialTable, ...]`` that **appends**
    diffuse shells to each atom after its ordinary functions (``existing``:
    the atom's radial functions as ``(n, l, r, values)``, filtered) --
    :func:`~.diffuse.diffuse_factory` supplies it.  The ordinary basis is
    built exactly as without it, and the atom-major order is kept.

    ``filter_wavevector`` (the cutoff wave-vector :math:`k_c` in
    Bohr\\ :sup:`-1` that the ``filter_cutoff`` option resolves to, ``None``
    = off) runs every radial table through the Fourier filter of
    :mod:`mandacaru.basis.filtering` before it becomes a basis function, so the
    grid sampling, the PAW-LCAO atom-centered projection quadrature and the displaced
    sampling the forces are built from all see one band-limited function.  The
    filter is applied **after** the split-valence construction: the split is
    defined on the pseudopotential's own orbital, and it is each *final* basis
    function that has to be representable on the grid.

    This does soften the first zeta -- the very function the Kleinman-Bylander
    projectors were built from -- and that is legitimate but not free.  The
    operator is untouched (the projectors, their energies and the local channel
    all come from the dataset); only the trial space changes, and the filter
    tends to the identity as the cutoff grows.  What is lost is that the
    *exact* pseudo-orbital is an eigenstate of the pseudo-Hamiltonian at the
    reference energy, so the atomic reference is no longer reproduced exactly.
    That is a variational price, measured in
    ``docs/source/guide/pseudopotentials.md``; PAW-LCAO and UPAW-LCAO turn the
    filter on by default because a PAW-LCAO partial wave is already built
    band-limited, so the filter barely moves it.
    """
    from ..basis.filtering import filter_radial, filter_table
    from ..basis.multizeta import zeta_tables
    from ..models.radial import DEFAULT_SPLIT_NORM, DEFAULT_TAIL_NORMS, resolve_zeta

    k_c = None if filter_wavevector is None else float(filter_wavevector)
    split = DEFAULT_SPLIT_NORM if split_norm is None else float(split_norm)

    # The filter memoizes by content (`filtering.filter_radial`): the same
    # radial function of the same element is filtered once, across atoms and
    # across steps.
    def filtered(values, r, l):
        return values if k_c is None else filter_radial(
            r, values, l, k_c, context=_context())[0]

    def filtered_table(table):
        return table if k_c is None else filter_table(
            table, k_c, context=_context())[0]

    confined: dict = {}

    def first_radial(symbol, pp, l, index=0, empty=False):
        """The first zeta's ``R(r)`` of the channel's ``index``-th occupied
        state: the family's confined orbital when it supplies one, else the
        pseudopotential's stored partial wave -- or, for an empty channel's
        bound level, its eigenstate (the stored waves there scatter)."""
        key = (symbol, int(l), int(index))
        if key not in confined:
            radial = (None if first_zeta is None
                      else first_zeta(symbol, pp, l, index))
            if radial is None and empty:
                radial = bound_radial(pp, l)
            confined[key] = (np.asarray(pp.channels[l].pseudo_waves[index],
                                        dtype=float)
                             if radial is None
                             else np.asarray(radial, dtype=float))
        return confined[key]

    def size_of(symbol):
        """``size`` may be one spec for all atoms or ``{symbol: spec}`` with an
        optional ``"*"`` default -- a polarized water next to a minimal ion."""
        if isinstance(size, dict):
            table = {(k if k == "*" else k.capitalize()): v
                     for k, v in size.items()}
            spec = table.get(symbol.capitalize(), table.get("*"))
            if spec is None:
                raise ValueError(
                    f"no pseudo-basis size for element {symbol!r}; add it or "
                    "a '*' default to the size mapping")
            return spec
        return size

    def element(symbol, pp):
        """``(orbitals, tables)`` of one element, built once for all its
        atoms: the ``(state, radial)`` pairs of the minimal set (``radial``
        ``None`` is the stored partial wave) and the radial tables of
        everything else."""
        # Empty scattering or bound-reference channels improve the nonlocal
        # operator without adding electrons. They supply projectors, not
        # minimal valence basis functions (or a new polarization parent).
        # Every occupied state gets a function -- a semicore channel two --
        # and so does an empty channel's bound level (calcium's 3d); only the
        # valence ones are split into zetas and polarized.
        states = basis_states(pp)
        n_zeta, n_polarization = resolve_zeta(size_of(symbol))
        if n_zeta == 1 and n_polarization == 0:
            # Minimal valence set: the original path, unchanged.
            orbitals, existing = [], []
            for state in states:
                l = state.l
                radial = first_radial(symbol, pp, l, state.index, state.empty)
                if k_c is None and first_zeta is None and not state.empty:
                    radial = None          # the stored table, byte for byte
                else:
                    radial = filtered(radial, pp.r, l)
                orbitals.append((state, radial))
                existing.append((state.n, l, pp.r,
                                 pp.channels[l].pseudo_waves[state.index]
                                 if radial is None else radial))
            extra = ([] if diffuse_shells is None
                     else diffuse_shells(symbol, pp, existing, k_c))
            return orbitals, extra

        tables = []
        valence = [state for state in states if not state.semicore]
        outer = max(valence, key=lambda state: state.l)
        l_max = outer.l
        for state in states:
            for table in zeta_tables(pp.r,
                                     first_radial(symbol, pp, state.l,
                                                  state.index, state.empty),
                                     state.n, state.l,
                                     1 if state.semicore else n_zeta,
                                     split, tail_norms=tail_norms,
                                     context=_context()):
                tables.append(filtered_table(table))
        # Polarization: raise the outermost channel to l+1.  Solving a new
        # confined orbital is not an option here -- there is no pseudopotential
        # channel for an unoccupied l -- so the shape is r^k R_outer(r),
        # normalized: it keeps the channel's range and vanishes as r^l at the
        # nucleus.  (Reusing R_outer itself gave R_p(0) != 0, a function
        # discontinuous at its own center whose derivative blew up whenever a
        # nucleus sat on a grid node.)
        outermost = first_radial(symbol, pp, l_max, outer.index, outer.empty)
        r_table = np.asarray(pp.r, dtype=float)
        custom = (polarization_shape(symbol, pp)
                  if polarization_shape is not None and n_polarization > 0
                  else None)
        if custom is not None:
            l, shape = custom
            # For the polarization zetas the tail norms are squared-norm
            # fractions (the valence zetas square them); `last_zeta` keeps its
            # convention for them too, rather than mixing the two.
            tails = (None if tail_norms is None
                     else tail_norms or DEFAULT_TAIL_NORMS)
            for table in zeta_tables(pp.r, shape, l + 1, l, n_polarization,
                                     split, tail_norms=tails, fractions=True,
                                     context=_context()):
                table.polarization = True
                tables.append(filtered_table(table))
            n_polarization = 0                 # the loop below has nothing left
        for offset in range(n_polarization):
            l = l_max + 1 + offset
            shape = _native.pao_power_shape(r_table, outermost, offset + 1)
            for table in zeta_tables(pp.r, shape, l + 1, l, 1, split,
                                     context=_context()):
                table.polarization = True
                # `filter_table` copies the record, `polarization` included.
                tables.append(filtered_table(table))

        if diffuse_shells is not None:
            # After the polarization shells, still on this atom: the
            # atom-major order the populations and `atom_of_orbital` rely on.
            tables.extend(diffuse_shells(
                symbol, pp, [(t.n, t.l, t.r, t.values) for t in tables], k_c))
        return [], tables

    built: dict = {}
    functions, owners = [], []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = potentials[symbol]
        if symbol not in built:
            built[symbol] = element(symbol, pp)
        orbitals, tables = built[symbol]
        atom_functions = _orbitals(pp, orbitals, position, units, real)
        atom_functions.extend(_shells(tables, position, units, real))
        functions.extend(atom_functions)
        owners.extend(index for _ in atom_functions)
    return functions, owners


@dataclass(frozen=True)
class OccupiedState:
    """One state a dataset's basis is built on: channel ``l``, its partial
    wave ``index``, principal number ``n``, whether it is semicore, and --
    for an empty channel's bound level (calcium's 3d, :func:`basis_states`)
    -- ``empty`` and that ``level`` (Hartree)."""

    l: int
    index: int
    n: int
    semicore: bool
    empty: bool = False
    level: float | None = None


#: An empty channel's lowest level is a bound state below this (Hartree);
#: above it the level belongs to the box (potassium's d: +0.004).
BOUND_LEVEL = -1e-3

#: ``(id(dataset), l) -> (dataset, lowest level)``.
_LEVEL_CACHE: dict = {}


def _spectrum_box(pp, l: int) -> float:
    """The radius of the box channel ``l``'s spectrum is solved in (Bohr): the engine's rule
    (``paw::diagnose::spectrum::extent``), asked of the dataset's record."""
    from .record import record_of
    return float(_native.paw_spectrum_box(_context(), *record_of(pp), int(l)))


def _lowest_level(pp, l: int) -> float:
    entry = _LEVEL_CACHE.get((id(pp), int(l)))
    if entry is None or entry[0] is not pp:
        from .confinement import channel_arrays
        level = _native.pao_bound_level(
            _context(), *channel_arrays(pp, int(l)), _spectrum_box(pp, int(l)))
        entry = (pp, float(level))
        _LEVEL_CACHE[(id(pp), int(l))] = entry
    return entry[1]


def bound_radial(pp, l: int) -> np.ndarray:
    """``R(r)`` on ``pp.r`` of channel ``l``'s lowest eigenstate in the
    dataset's own operator, zero past the box it was solved in."""
    from .confinement import channel_arrays
    radial, _level = _native.pao_bound_radial(
        _context(), *channel_arrays(pp, int(l)), _spectrum_box(pp, int(l)))
    return radial


#: Empty channels that carry projectors for their scattering only: their
#: bound level stays out of the basis.  Lithium's p channel binds a 2p at
#: -0.042 Ha; as a basis state it would make the minimal LiH basis five
#: orbitals instead of two, and at DZP it moved bcc Li, Li2, LiH and LiF by
#: -0.5, +1.4, -1.6 and +1.3 % against the polarization shell it replaces
#: (HISTORY.md, 2026-10-07).  Beryllium's p channel (2p bound at about
#: -0.08 Ha) likewise, so a minimal Be basis stays one orbital.
SCATTERING_ONLY_CHANNELS = {"Li": (1,), "Be": (1,)}


def basis_states(pp) -> list[OccupiedState]:
    """:func:`occupied_states` plus the bound level of every empty channel
    that has one (but :data:`SCATTERING_ONLY_CHANNELS`).  Calcium's d
    channel binds a 3d at -0.078 Ha that the crystal fills; without it in
    the basis fcc Ca came out 7-9 % too large with every zeta and
    polarization count (HISTORY.md, 2026-10-06)."""
    states = occupied_states(pp)
    held = {s.l for s in states} | set(SCATTERING_ONLY_CHANNELS.get(str(pp.symbol), ()))
    items = sorted(pp.channels.items())
    for l, channel in items:
        lvl = _lowest_level(pp, l) if int(l) not in held else 0.0
        if int(l) not in held and lvl < BOUND_LEVEL:
            states.append(OccupiedState(int(l), 0, int(channel.n), False,
                                        empty=True, level=lvl))
    return states


def occupied_states(pp) -> list[OccupiedState]:
    """Every occupied reference state of a dataset, channel by channel: one
    per occupied channel, two in a semicore one (sodium's 2s and 3s; the
    second is one shell up)."""
    semicore = {tuple(o) for o in getattr(pp, "semicore_subshells", ())}
    out, items = [], sorted(pp.channels.items())
    for l, channel in items:
        for i, occ in enumerate(getattr(channel, "occupations", None)
                               or [channel.occupation]):
            if occ > 0.0:
                n = int(channel.n) + i
                out.append(OccupiedState(int(l), i, n,
                                         (n, int(l)) in semicore))
    return out


def valence_electrons(symbols, potentials) -> float:
    """Total valence charge -- the electron count a PP calculation carries."""
    return float(sum(potentials[symbol].valence_charge for symbol in symbols))
