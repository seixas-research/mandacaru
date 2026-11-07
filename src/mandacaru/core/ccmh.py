# -*- coding: utf-8 -*-
# file: core/ccmh.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Finite-size correction from the structure factor (Chiesa *et al.*).

A periodic calculation evaluates the electron-electron energy as a **discrete
sum** over the reciprocal lattice of the supercell,

.. math::

    V_N = \frac{1}{2\Omega} \sum_{\mathbf k \neq 0}
          v(\mathbf k)\,\bigl[S(\mathbf k) - 1\bigr] ,
    \qquad v(\mathbf k) = \frac{4\pi}{k^2},

where the thermodynamic limit wants the **integral**
:math:`\tfrac12 \int \frac{d^3k}{(2\pi)^3} v(k)[S(k)-1]`.  For a smooth
integrand the two agree to exponential accuracy -- except that the sum has no
:math:`\mathbf k = 0` term.  Each k-point of the supercell carries reciprocal
volume :math:`(2\pi)^3/\Omega`, so the piece the sum omits is

.. math::

    \Delta V = \frac{1}{2\Omega}\,
      \lim_{\mathbf k \to 0} \bigl[v(\mathbf k) S(\mathbf k)\bigr] .
    \label{eq-ccmh}
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .. import _native
from .mpc import spatial_rdms

__all__ = ["StructureFactor", "CCMHCorrection", "structure_factor",
           "ccmh_correction", "spatial_rdms"]

#: Reciprocal-lattice shells used to extrapolate ``S(k)/k^2`` to ``k = 0``.
DEFAULT_SHELLS = 3

#: ``|m|`` of the integer reciprocal-lattice triples enumerated.
GVECTOR_RANGE = 3

#: Largest relative spread of ``S(k)/k^2`` across the fitted shells that still
#: counts as "inside the quadratic regime".
QUADRATIC_REGIME_SPREAD = 0.5


@dataclass
class StructureFactor:
    """``S(k)`` of the correlated state at the supercell's reciprocal lattice.

    Attributes
    ----------
    kvectors : ndarray, shape (nk, 3)
        Cartesian reciprocal vectors in Bohr^-1, excluding ``k = 0``.
    values : ndarray, shape (nk,)
        ``S(k)``, real and non-negative for a physical state.
    magnitudes : ndarray, shape (nk,)
        ``|k|`` in Bohr^-1.
    n_electrons : float
        ``tr D``, the normalization used.
    overlap_error : float
        ``max |M(0) - I|``.
    """

    kvectors: np.ndarray
    values: np.ndarray
    magnitudes: np.ndarray
    n_electrons: float
    overlap_error: float

    def shells(self, count: int = DEFAULT_SHELLS):
        """``(|k|, S)`` averaged over the ``count`` smallest shells."""
        if len(self.magnitudes) == 0:
            return np.empty(0), np.empty(0)
        return _native.ccmh_shells(
            np.ascontiguousarray(self.magnitudes, dtype=float),
            np.ascontiguousarray(self.values, dtype=float),
            int(count),
        )


@dataclass
class CCMHCorrection:
    r"""The leading finite-size error of the electron-electron energy."""

    correction: float
    limit: float
    rpa_limit: float
    plasma_frequency: float
    density: float
    volume: float
    n_cells: int
    n_electrons: float
    quadratic_spread: float = 0.0
    energy_per_cell: float | None = None
    structure: StructureFactor | None = field(default=None, repr=False)

    @property
    def reliable(self) -> bool:
        """Whether ``S(k)/k^2`` was flat enough for the limit to mean anything."""
        return self.quadratic_spread <= QUADRATIC_REGIME_SPREAD

    @property
    def correction_per_cell(self) -> float:
        """Hartree to add to a per-primitive-cell energy."""
        return self.correction / self.n_cells

    @property
    def corrected_energy_per_cell(self) -> float | None:
        """The finite-size-corrected energy per primitive cell, in Hartree."""
        if self.energy_per_cell is None:
            return None
        return self.energy_per_cell + self.correction_per_cell

    @property
    def rpa_correction(self) -> float:
        r""":math:`\pi / (\Omega \omega_p)` -- what the RPA limit would give."""
        return 6.283185307179586 * self.rpa_limit / self.volume

    def summary(self) -> str:
        """One line, in eV."""
        from ..units import from_hartree

        return (f"CCMH finite-size correction: "
                f"{from_hartree(self.correction_per_cell, 'eV'):+.6f} eV/cell "
                f"(lim S/k^2 = {self.limit:.5f} Bohr^2 against RPA "
                f"{self.rpa_limit:.5f}; omega_p = "
                f"{from_hartree(self.plasma_frequency, 'eV'):.3f} eV, "
                f"n = {self.density:.6f} Bohr^-3)")


def _reciprocal_vectors(cell, limit: int = GVECTOR_RANGE):
    """Integer triples and their Cartesian reciprocal vectors, without ``0``."""
    return _native.ccmh_reciprocal_vectors(
        np.ascontiguousarray(cell, dtype=float),
        int(limit),
    )


def structure_factor(integrals, D, Gamma, *, limit: int = GVECTOR_RANGE):
    r"""``S(k)`` of the state described by ``(D, Gamma)``."""
    D_arr = np.ascontiguousarray(D.real, dtype=float)
    Gamma_arr = np.ascontiguousarray(Gamma.real, dtype=float)
    grid = integrals.grid
    shape = tuple(int(s) for s in grid.shape)
    psi = np.ascontiguousarray(integrals._engine._psi, dtype=float)

    A = integrals._lowdin_x() if integrals.orthogonalize else np.eye(len(psi), dtype=complex)
    V = integrals.mo_coefficients
    if V is not None:
        A = A @ V
    A = np.ascontiguousarray(A, dtype=complex)

    res = _native.compute_structure_factor(
        psi,
        shape,
        float(grid.dV),
        np.ascontiguousarray(integrals.cell, dtype=float),
        int(limit),
        A,
        D_arr,
        Gamma_arr,
    )

    return StructureFactor(
        kvectors=res["kvectors"],
        values=res["values"],
        magnitudes=res["magnitudes"],
        n_electrons=float(res["n_electrons"]),
        overlap_error=float(res["overlap_error"]))


def ccmh_correction(structure: StructureFactor, *, volume, n_cells: int = 1,
                    shells: int = DEFAULT_SHELLS, energy_per_cell=None):
    r"""Take :math:`\lim_{k\to0} S(k)/k^2` and turn it into :math:`\Delta V`."""
    k, s = structure.shells(shells)
    if len(k) == 0:
        raise ValueError("no reciprocal-lattice shells to extrapolate from")
    raw = _native.ccmh_extrapolate(
        np.ascontiguousarray(k, dtype=float),
        np.ascontiguousarray(s, dtype=float),
        float(volume),
        int(n_cells),
        structure.n_electrons,
    )
    spread = raw["quadratic_spread"]
    if spread > QUADRATIC_REGIME_SPREAD:
        import warnings

        warnings.warn(
            f"the CCMH extrapolation is outside the quadratic regime: "
            f"S(k)/k^2 varies by {spread:.0%} across the {len(k)} fitted "
            f"shells, so S(k) has not reached its small-k form where the "
            f"smallest reciprocal-lattice vector sits.  Use a larger supercell, "
            f"or treat this number as a diagnostic only.",
            RuntimeWarning, stacklevel=2)

    return CCMHCorrection(
        correction=raw["correction_per_electron"],
        limit=raw["limit"],
        quadratic_spread=spread,
        rpa_limit=raw["rpa_limit"],
        plasma_frequency=raw["plasma_frequency"],
        density=raw["density"],
        volume=raw["volume"],
        n_cells=raw["n_cells"],
        n_electrons=structure.n_electrons,
        energy_per_cell=(None if energy_per_cell is None
                         else float(energy_per_cell)),
        structure=structure)
