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
multiplied by a spherical harmonic -- the same construction as the numerical
atomic orbitals of the Rust engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _native
from ..units import to_bohr
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
        ``R(r_0) (r / r_0)^l``, as the engine's orbital evaluator
        (``_native.radial_orbital_evaluate``) does.  A constant continuation would leave an
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

    def _sample(self, x, y, z, gradient: bool) -> np.ndarray:
        """``R(r) Y_lm`` (or its gradient) at the points, sampled by the native orbital routine."""
        flat = [np.ascontiguousarray(a, dtype=np.float64).ravel() for a in (x, y, z)]
        return _native.radial_orbital_sample(
            _context(), self._r, self._values, self.l, self.m, *flat,
            [float(c) for c in self.center], self.real, gradient)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample the function on Cartesian coordinates (Bohr)."""
        return self._sample(x, y, z, False).reshape(np.shape(x))

    def gradient(self, x, y, z, delta: float = 1e-3) -> np.ndarray:
        """Analytic gradient, ``(3, n)`` (``delta`` is unused)."""
        return self._sample(x, y, z, True)


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


def pseudo_basis(symbols, positions, potentials, units: str = "angstrom", options=None,
                 filter_wavevector=None, *, family: str = "paw-lcao", real: bool = True,
                 report: dict | None = None):
    """Valence pseudo-atomic basis for a molecule (real spherical harmonics) or, ``real=False``,
    a crystal (complex ones): ``(functions, atom_of_orbital)``.

    ``options`` is a basis dict of ``family`` (``size``, ``energy_shift``,
    ``confinement_potential``, ``polarization_method``, ``zeta_split``, ``diffuse``), read with
    the calculator's own rules; ``filter_wavevector`` is the Fourier-filter cutoff (Bohr^-1,
    ``None`` = off). The radial shells are built by the Rust recipe the calculator uses
    (``paw::basis_recipe``); this returns them as :class:`PseudoShell` functions, atom-major.
    ``report``, when given, receives what the recipe resolved per element: ``confinement``,
    ``polarization`` and ``diffuse`` (every diffuse shell built, accepted or not).
    """
    from ..basis.multizeta import RadialTable
    from ..integrations.basis_recipe import basis_tables
    from .families import resolve_family

    spec = resolve_family(family)
    resolved = spec.resolved_options(dict(options or {}))
    unique = {symbol: potentials[symbol] for symbol in dict.fromkeys(symbols)}
    report = {} if report is None else report
    for key in ("confinement", "polarization", "diffuse"):
        report.setdefault(key, {})
    built = basis_tables(_context(), spec, resolved, unique, filter_wavevector, report)
    functions, owners = [], []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        tables = [RadialTable(r=t.r, values=t.values, n=int(t.n), l=int(t.l), zeta=int(t.zeta),
                              polarization=bool(t.polarization), diffuse=bool(t.diffuse))
                  for t in built[symbol]]
        atom_functions = _shells(tables, position, units, real)
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


def basis_states(pp) -> list[OccupiedState]:
    """The states the basis of dataset ``pp`` is built on, from the engine's recipe
    (``paw::basis_recipe``): every occupied reference state (two in a semicore channel) and the
    bound level of every empty channel that has one, below -1e-3 Ha (calcium's 3d binds at
    -0.078 Ha and the crystal fills it; lithium's and beryllium's p channels carry projectors for
    scattering only and stay out)."""
    from .record import record_of
    return [OccupiedState(int(l), int(index), int(n), bool(semicore), bool(empty), level)
            for l, index, n, semicore, empty, level
            in _native.paw_basis_states(_context(), *record_of(pp))]


def valence_electrons(symbols, potentials) -> float:
    """Total valence charge -- the electron count a PP calculation carries."""
    return float(_native.total([potentials[symbol].valence_charge for symbol in symbols]))
