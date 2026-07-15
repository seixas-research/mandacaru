# -*- coding: utf-8 -*-
# file: basis/nao.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Numerical Atomic Orbitals (Sankey-type, radially confined).

Following the SIESTA methodology, an NAO is a numerical radial function times a
spherical harmonic,

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
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal

from ..units import EV_TO_HARTREE, to_bohr
from ._angular import spherical_coords, spherical_harmonic
from .base import BasisFunction

#: Default energy shift (eV) setting the confinement radius, as in SIESTA.
DEFAULT_ENERGY_SHIFT = 0.03

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


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

    h = r_c / n_grid
    r = np.arange(1, n_grid) * h  # interior nodes; the wall at r_c is excluded
    inv_h2 = 1.0 / (h * h)
    diag = inv_h2 - Z / r + l * (l + 1) / (2.0 * r * r)
    offdiag = -0.5 * inv_h2 * np.ones(r.size - 1)

    evals, evecs = eigh_tridiagonal(diag, offdiag, select="i",
                                    select_range=(k, k))
    u = evecs[:, 0]
    R = u / r
    norm = np.sqrt(_trapezoid(R * R * r * r, r))
    R = R / norm
    if R[0] < 0.0:  # fix the global sign so R > 0 near the origin
        R = -R

    # Close the grid with r = 0 (extrapolated) and the hard wall r_c (R = 0).
    slope = (R[1] - R[0]) / (r[1] - r[0])
    R0 = R[0] - slope * r[0]
    r_full = np.concatenate(([0.0], r, [r_c]))
    R_full = np.concatenate(([R0], R, [0.0]))
    return r_full, R_full, float(evals[0])


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
                 center=None, units: str = "angstrom", n_grid: int = 2000):
        if not (0 <= l < n):
            raise ValueError(f"require 0 <= l < n, got n={n}, l={l}")
        if abs(m) > l:
            raise ValueError(f"require |m| <= l, got l={l}, m={m}")
        self.n, self.l, self.m = int(n), int(l), int(m)
        self.Z = float(Z)
        self.r_c = float(r_c) if r_c is not None \
            else energy_shift_to_rc(energy_shift)
        self.center = (np.zeros(3) if center is None
                       else to_bohr(center, units))

        self._r_grid, self._R_grid, self.energy = solve_confined_radial(
            self.n, self.l, self.Z, self.r_c, n_grid)
        # Cubic spline of the sampled radial function; zero outside [0, r_c].
        self._spline = CubicSpline(self._r_grid, self._R_grid,
                                   extrapolate=False)

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.n, self.l, self.m)

    def radial(self, r) -> np.ndarray:
        """Interpolated radial function ``R_{nl}(r)`` (0 for ``r >= r_c``)."""
        r = np.asarray(r, dtype=float)
        R = self._spline(r)
        # Outside [0, r_c] the spline returns NaN -> the confined orbital is 0.
        return np.where(np.isnan(R) | (r >= self.r_c), 0.0, R)

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample ``psi_{nlm}`` on Cartesian coordinates (Bohr)."""
        r, theta, phi = spherical_coords(x, y, z, self.center)
        R = self.radial(r)
        Y = spherical_harmonic(self.l, self.m, theta, phi)
        return np.asarray(R * Y, dtype=np.complex128)

    def __repr__(self) -> str:
        return (f"NumericalAtomicOrbital(n={self.n}, l={self.l}, m={self.m}, "
                f"Z={self.Z}, r_c={self.r_c:.3f}, energy={self.energy:.4f})")
