# -*- coding: utf-8 -*-
# file: basis/nao.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Numerical Atomic Orbitals (Sankey-type, radially confined).

An NAO is a numerical radial function times a spherical harmonic,

.. math::

    \psi_{nlm}(\mathbf r) = R_{nl}(r)\, Y_l^m(\theta, \phi),

where :math:`R_{nl}(r)` is obtained by solving the radial Schrödinger equation
inside a hard-wall sphere of radius :math:`r_c` (so the orbital is *strictly*
zero for :math:`r \ge r_c`).  The confinement radius is fixed by the *energy
shift* :math:`\delta E` -- the amount the confinement raises the free-atom level:

.. math::

    \delta E = \frac{\pi^2 \hbar^2}{2 m r_c^2}
    \;\;\Longrightarrow\;\;
    r_c = \frac{\pi \hbar}{\sqrt{2 m\, \delta E}}
        \;\xrightarrow{\text{a.u.}}\; \frac{\pi}{\sqrt{2\, \delta E}} .

The radial equation is solved on a uniform grid by finite differences for
``u(r) = r R(r)`` with Dirichlet boundary conditions ``u(0) = u(r_c) = 0``:

.. math::

    -\tfrac12 u'' + \Big(-\frac{Z}{r} + \frac{l(l+1)}{2 r^2}\Big) u = E\, u,

using a *screened* charge ``Z`` (e.g. the Slater effective charge) so that
valence orbitals of many-electron atoms have physically sensible extents.  The
resulting samples are interpolated by a cubic spline and taper smoothly to zero
at ``r_c``.
"""

from __future__ import annotations

import numpy as np

from .. import _native
from ..units import EV_TO_HARTREE, to_bohr
from ._angular import sample_dtype as _dtype
from ._angular import spherical_coords, spherical_harmonic
from .base import BasisFunction

#: Default energy shift (eV) setting the confinement radius.
DEFAULT_ENERGY_SHIFT = 0.03


def energy_shift_to_rc(energy_shift: float = DEFAULT_ENERGY_SHIFT) -> float:
    r"""Confinement radius ``r_c`` (Bohr) from an ``energy_shift`` (eV).

    Inverts :math:`\delta E = \pi^2/(2 r_c^2)` (atomic units) after converting
    the shift from eV to Hartree.
    """
    dE = float(energy_shift) * EV_TO_HARTREE
    if dE <= 0.0:
        raise ValueError(f"energy_shift must be > 0, got {energy_shift}")
    return float(np.pi / np.sqrt(2.0 * dE))


def solve_confined_radial(n: int, l: int, Z: float, r_c: float,
                          n_grid: int = 2000):
    """Solve the confined radial Schrödinger equation for ``R_{nl}(r)``.

    Finite-difference eigensolve of ``u = r R`` on a uniform grid in
    ``(0, r_c)`` with a ``-Z/r`` potential, selecting the eigenstate with
    ``n - l - 1`` radial nodes.

    Returns
    -------
    (r, R, energy) : (ndarray, ndarray, float)
        The radial grid (including the endpoints ``0`` and ``r_c``), the
        normalized radial function ``R`` on it (``R(r_c) = 0``, ``int |R|^2 r^2
        dr = 1``), and the confined eigenvalue in Hartree.
    """
    if not (0 <= l < n):
        raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
    k = n - l - 1  # number of radial nodes -> eigenstate index (0-based)
    if n_grid <= k + 2:
        raise ValueError("n_grid too small for the requested state")

    ctx = _native.Context(threads=1)
    res = ctx.confined_radial(int(n), int(l), float(Z), float(r_c), intervals=int(n_grid))
    return res["r"], res["radial"], float(res["energy"])


class NumericalAtomicOrbital(BasisFunction):
    r"""A radially confined numerical atomic orbital ``R_{nl}(r) Y_l^m``.

    Parameters
    ----------
    n, l, m : int
        Principal, azimuthal and magnetic quantum numbers.
    Z : float
        (Screened) nuclear charge in the radial potential ``-Z/r``.
    r_c : float, optional
        Confinement radius in Bohr.  If ``None`` it is derived from
        ``energy_shift`` via :func:`energy_shift_to_rc`.
    energy_shift : float
        Energy shift in eV (default ``0.03``) used when ``r_c`` is not given.
    center : array_like, shape (3,)
        Cartesian center, in ``units`` (default Angstrom); stored in Bohr.
    units : {"angstrom", "bohr"}
        Unit of ``center``.
    n_grid : int
        Number of radial finite-difference intervals.

    Attributes
    ----------
    r_c : float
        Confinement radius (Bohr); the orbital is exactly zero for ``r >= r_c``.
    energy : float
        Confined radial eigenvalue (Hartree).
    """

    def __init__(self, n: int, l: int, m: int, Z: float = 1.0,
                 r_c: float | None = None,
                 energy_shift: float = DEFAULT_ENERGY_SHIFT,
                 center=None, units: str = "angstrom", n_grid: int = 2000,
                 *, real: bool = True):
        if not (0 <= l < n):
            raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
        if abs(m) > l:
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.n, self.l, self.m = int(n), int(l), int(m)
        #: Real (molecule) or complex (crystal) spherical harmonics.
        self.real = bool(real)
        self.Z = float(Z)
        self.r_c = float(r_c) if r_c is not None \
            else energy_shift_to_rc(energy_shift)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))

        self._r_grid, self._R_grid, self.energy = solve_confined_radial(
            self.n, self.l, self.Z, self.r_c, n_grid)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function ``R_{nl}(r)`` (0 for ``r >= r_c``)."""
        shape = np.shape(r)
        res = _native.radial_orbital_evaluate(
            self._r_grid, self._R_grid,
            np.ascontiguousarray(np.asarray(r, dtype=float).ravel(), dtype=float),
            int(self.l), float(self.r_c),
        )
        return res.reshape(shape)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample ``psi_{nlm}`` on Cartesian coordinates (Bohr)."""
        r, theta, phi = spherical_coords(x, y, z, self.center)
        R = self.radial(r)
        Y = spherical_harmonic(self.l, self.m, theta, phi, real=self.real)
        return np.asarray(R * Y, dtype=_dtype(self.real))

    def __repr__(self) -> str:
        return (f"NumericalAtomicOrbital(n={self.n}, l={self.l}, m={self.m}, "
                f"Z={self.Z}, r_c={self.r_c:.3f}, energy={self.energy:.4f})")
