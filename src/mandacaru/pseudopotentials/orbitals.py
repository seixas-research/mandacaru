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
* the **Kleinman-Bylander projectors** :math:`\chi_{l}(r)\,Y_{lm}`, which carry
  the nonlocal part of the potential.

Both are radial tables from :mod:`mandacaru.pseudopotentials.generation`, splined and
multiplied by a spherical harmonic -- the same construction
:class:`~mandacaru.basis.nao.NumericalAtomicOrbital` uses, so they drop into the
integral engine unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from ..units import to_bohr
from ..basis._angular import spherical_coords, spherical_harmonic
from ..basis.base import BasisFunction


class _RadialTabulated(BasisFunction):
    """Shared machinery: spline a radial table and attach an angular factor."""

    def __init__(self, r, radial, l: int, m: int, center=None,
                 units: str = "angstrom"):
        if abs(int(m)) > int(l):
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.l, self.m = int(l), int(m)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))
        self._r = np.asarray(r, dtype=float)
        self._values = np.asarray(radial, dtype=float)
        self._r_max = float(self._r[-1])
        self._spline = CubicSpline(self._r, self._values, extrapolate=False)

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function; zero beyond the tabulated range."""
        r = np.asarray(r, dtype=float)
        values = self._spline(np.clip(r, self._r[0], self._r_max))
        return np.where((r > self._r_max) | np.isnan(values), 0.0, values)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample the function on Cartesian coordinates (Bohr)."""
        r, theta, phi = spherical_coords(x, y, z, self.center)
        return np.asarray(self.radial(r)
                          * spherical_harmonic(self.l, self.m, theta, phi),
                          dtype=np.complex128)


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
    """

    def __init__(self, pseudopotential, l: int, m: int, center=None,
                 units: str = "angstrom", radial=None):
        channel = pseudopotential.channels[int(l)]
        super().__init__(pseudopotential.r,
                         channel.pseudo_radial if radial is None else radial,
                         l, m, center, units)
        self.symbol = pseudopotential.symbol
        self.n = int(channel.n)
        self.eigenvalue = float(channel.eigenvalue)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    def __repr__(self) -> str:
        return (f"PseudoAtomicOrbital({self.symbol}, n={self.n}, l={self.l}, "
                f"m={self.m}, eps={self.eigenvalue:.4f})")


class KBProjector(_RadialTabulated):
    r"""A Kleinman-Bylander projector :math:`\chi_{l}(r)Y_{lm}`.

    Carries the Kleinman-Bylander energy :attr:`kb_energy` and the index of the
    atom it belongs to, which is what the nonlocal matrix and its nuclear
    derivative need.

    Every projector also carries the three labels the **general separable
    form** of :meth:`mandacaru.core.hamiltonian.MolecularIntegrals.kb_nonlocal`
    keys on: :attr:`atom_index`, the :attr:`channel` ``(l, m)`` and, within
    that channel, its radial :attr:`index`.  The nonlocal coupling matrix
    :math:`D` is block-diagonal over ``(atom, l, m)``; a family with several
    radial projectors per channel (ONCVPSP) numbers them with ``index`` and
    supplies the block, whereas the Kleinman-Bylander form has a single
    projector per channel (``index = 0``) and the :math:`1\times1` block
    :math:`[E^{KB}_l]`.
    """

    def __init__(self, pseudopotential, l: int, m: int, center=None,
                 units: str = "angstrom", atom_index: int = 0,
                 index: int = 0, radial=None, kb_energy=None):
        chi = (pseudopotential.projectors[int(l)] if radial is None
               else np.asarray(radial, dtype=float))
        super().__init__(pseudopotential.r, chi, l, m, center, units)
        self.symbol = pseudopotential.symbol
        self.kb_energy = float(pseudopotential.kb_energies[int(l)]
                               if kb_energy is None else kb_energy)
        self.atom_index = int(atom_index)
        #: Radial projector index within the ``(atom, l, m)`` channel.
        self.index = int(index)
        channel = pseudopotential.channels.get(int(l))
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

def pseudo_basis(symbols, positions, potentials, units: str = "angstrom",
                 size="SZ", split_norm=None, filter_cutoff=None,
                 first_zeta=None, tail_norms=None, polarization_shape=None):
    """Valence pseudo-atomic basis for a molecule.

    Returns ``(functions, atom_of_orbital)``: the orbitals and which atom each
    belongs to.

    ``size`` selects the multiple-zeta / polarized hierarchy exactly as it does
    for the all-electron NAO family (``"SZ"``, ``"DZ"``, ``"DZP"``, ``"TZP"``,
    ...); a ``{symbol: size}`` mapping (with an optional ``"*"`` default)
    gives each element its own size.  The **first zeta always comes from the pseudopotential** and cannot be
    replaced: the Troullier-Martins construction pseudizes each valence orbital
    inside its cutoff radius and the Kleinman-Bylander projectors are built from
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

    ``tail_norms`` (a sequence, GPAW's ``tailnorm``) switches the extra zetas
    from the SIESTA-style scheme ``split_norm`` controls to GPAW's: every zeta
    split from the *first* one, at the radius leaving a tail of that **norm**
    (see :func:`~mandacaru.basis.multizeta.zeta_tables`).

    ``polarization_shape`` (optional) is a callable ``(symbol, potential) ->
    (l, R(r))`` -- or ``None`` to decline for that element -- that replaces the polarization shell built here --
    :func:`~.confinement.polarization_factory` supplies GPAW's quasi-Gaussian.
    Further polarization functions are then split-valence refinements of that
    shape with the *same* ``l``, as GPAW makes them, instead of shells of
    higher ``l``.

    ``filter_cutoff`` (a wave-vector in Bohr\\ :sup:`-1`, ``None`` = off) runs
    every radial table through the Fourier filter of
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
    ``docs/source/guide/pseudopotentials.md``, and it is why the norm-conserving
    families (NCPP, ONCVPSP) leave the filter **off** by default while PAW-LCAO and
    UPAW-LCAO turn it on -- a PAW-LCAO partial wave is already built band-limited, so
    the filter barely moves it.
    """
    from ..basis.filtering import filter_radial, filter_table
    from ..basis.multizeta import (DEFAULT_SPLIT_NORM, GPAW_TAIL_NORMS,
                                   orbitals_from_tables, resolve_zeta,
                                   zeta_tables)

    k_c = None if filter_cutoff is None else float(filter_cutoff)
    # One filter evaluation per (element, radial function), not per atom and
    # not per m: water's two hydrogens are the same element, and a p shell is
    # three functions off one table.
    cache: dict = {}

    def filtered(symbol, key, r, values, l):
        if k_c is None:
            return values
        memo = cache.get((symbol, key))
        if memo is None:
            memo, _info = filter_radial(r, values, l, k_c)
            cache[(symbol, key)] = memo
        return memo

    def filtered_table(symbol, key, table):
        if k_c is None:
            return table
        memo = cache.get((symbol, key))
        if memo is None:
            memo, _info = filter_table(table, k_c)
            cache[(symbol, key)] = memo
        return memo

    confined: dict = {}

    def first_radial(symbol, pp, l):
        """The first zeta's ``R(r)``: the family's confined orbital when it
        supplies one, else the pseudopotential's stored orbital."""
        key = (symbol, int(l))
        if key not in confined:
            radial = None if first_zeta is None else first_zeta(symbol, pp, l)
            confined[key] = (np.asarray(pp.channels[l].pseudo_radial,
                                        dtype=float)
                             if radial is None
                             else np.asarray(radial, dtype=float))
        return confined[key]

    def split_norm_of(symbol):
        """``split_norm`` may be one value or ``{symbol: value}`` (with ``*``)."""
        spec = split_norm
        if isinstance(spec, dict):
            table = {(k if k == "*" else k.capitalize()): v
                     for k, v in spec.items()}
            spec = table.get(symbol.capitalize(), table.get("*"))
        return DEFAULT_SPLIT_NORM if spec is None else float(spec)

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

    functions, owners = [], []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = potentials[symbol]
        n_zeta, n_polarization = resolve_zeta(size_of(symbol))
        if n_zeta == 1 and n_polarization == 0:
            # Minimal valence set: the original path, unchanged.
            for l in sorted(pp.channels):
                radial = first_radial(symbol, pp, l)
                if k_c is None and first_zeta is None:
                    radial = None          # the stored table, byte for byte
                else:
                    radial = filtered(symbol, ("sz", l), pp.r, radial, l)
                for m in range(-l, l + 1):
                    functions.append(PseudoAtomicOrbital(
                        pp, l, m, center=position, units=units, radial=radial))
                    owners.append(index)
            continue

        tables = []
        l_max = max(pp.channels)
        for l in sorted(pp.channels):
            channel = pp.channels[l]
            for table in zeta_tables(pp.r, first_radial(symbol, pp, l),
                                     int(channel.n), l, n_zeta,
                                     split_norm_of(symbol),
                                     tail_norms=tail_norms):
                tables.append(filtered_table(symbol, ("z", l, table.zeta),
                                             table))
        # Polarization: raise the outermost channel to l+1.  Solving a new
        # confined orbital is not an option here -- there is no pseudopotential
        # channel for an unoccupied l -- so the shape is r^k R_outer(r),
        # normalized: it keeps the channel's range and vanishes as r^l at the
        # nucleus.  (Reusing R_outer itself gave R_p(0) != 0, a function
        # discontinuous at its own center whose derivative blew up whenever a
        # nucleus sat on a grid node.)
        outermost = first_radial(symbol, pp, l_max)
        r_table = np.asarray(pp.r, dtype=float)
        custom = (polarization_shape(symbol, pp)
                  if polarization_shape is not None and n_polarization > 0
                  else None)
        if custom is not None:
            l, shape = custom
            # GPAW hands `rsplit_by_norm` the tail norm *unsquared* for these
            # (it squares it for the valence zetas), so here the numbers are
            # squared-norm fractions; `zeta_tables` squares what it is given.
            norms = [np.sqrt(t) for t in (tail_norms or GPAW_TAIL_NORMS)]
            if tail_norms is None:
                # The SIESTA-style scheme was asked for: keep it for the
                # polarization zetas too, rather than mixing conventions.
                norms = None
            for table in zeta_tables(pp.r, shape, l + 1, l, n_polarization,
                                     split_norm_of(symbol),
                                     tail_norms=norms):
                table.polarization = True
                tables.append(filtered_table(symbol, ("pol", l, table.zeta),
                                             table))
            n_polarization = 0                 # the loop below has nothing left
        for offset in range(n_polarization):
            l = l_max + 1 + offset
            shape = r_table ** (offset + 1) * outermost
            shape = shape / np.sqrt(np.trapezoid(shape * shape * r_table ** 2,
                                                 r_table))
            for table in zeta_tables(pp.r, shape, l + 1, l, 1,
                                     split_norm_of(symbol)):
                table.polarization = True
                # `filter_table` copies the record, `polarization` included.
                tables.append(filtered_table(symbol, ("pol", l, table.zeta),
                                             table))

        atom_functions = orbitals_from_tables(tables, center=position,
                                              units=units)
        functions.extend(atom_functions)
        owners.extend([index] * len(atom_functions))
    return functions, owners


def kb_projectors(symbols, positions, potentials, units: str = "angstrom"):
    """Kleinman-Bylander projectors for a molecule (one per ``(atom, l, m)``)."""
    projectors = []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = potentials[symbol]
        for l in sorted(pp.projectors):
            for m in range(-l, l + 1):
                projectors.append(KBProjector(pp, l, m, center=position,
                                              units=units, atom_index=index))
    return projectors


def kb_coupling_blocks(projectors) -> dict:
    r"""Nonlocal coupling blocks of a Kleinman-Bylander projector set.

    The general separable nonlocal term is :math:`H^{NL} = C\,D\,C^\dagger`
    with :math:`D` block-diagonal over ``(atom, l, m)``.  For the KB form each
    block is the :math:`1\times1` matrix :math:`[E^{KB}_l]`; this returns
    ``{(atom_index, l, m): [[E_KB]]}``, the layout
    :class:`~mandacaru.core.hamiltonian.MolecularIntegrals` takes as
    ``nonlocal_coupling``.
    """
    blocks = {}
    for projector in projectors:
        key = projector.block_key
        if key in blocks:
            raise ValueError(
                f"two Kleinman-Bylander projectors share the channel {key}; "
                "the KB form has exactly one projector per (atom, l, m)")
        blocks[key] = np.array([[projector.kb_energy]], dtype=complex)
    return blocks


def valence_electrons(symbols, potentials) -> float:
    """Total valence charge -- the electron count a PP calculation carries."""
    return float(sum(potentials[symbol].valence_charge for symbol in symbols))
