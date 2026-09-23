# -*- coding: utf-8 -*-
# file: core/kzk.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Finite-size correction through a size-dependent LDA (Kwee-Zhang-Krakauer).

Kwee, Zhang and Krakauer, *Finite-Size Correction in Many-Body Electronic
Structure Calculations*, Phys. Rev. Lett. **100**, 126404 (2008).

The idea is to do the finite-size bookkeeping in **density-functional theory**,
where it is cheap, and hand the answer to the many-body calculation.  They
build an exchange-correlation functional that knows the supercell size,
:math:`\epsilon_{xc}^{\text{FS}}(r_s, L)`, by fitting unpolarized jellium in
the *same* supercell; the correction to a many-body energy is then the
difference between the infinite-size and finite-size LDA energies evaluated on
the **same** density,

.. math::

    \Delta E = \int n(\mathbf r)\,
      \bigl[\epsilon_{xc}^{\infty}(n) - \epsilon_{xc}^{\text{FS}}(n, L)\bigr]\,
      d\mathbf r ,

and the corrected energy is :math:`E_{\text{MB}} + \Delta E`.  Everything else
in the density functional -- kinetic, Hartree, ionic -- cancels between the two,
because only the exchange-correlation functional differs.

Why this one suits a force calculation: the correction is a **functional of the
density alone**.  It does not touch the interaction kernel, so unlike the model
periodic Coulomb route of :mod:`mandacaru.core.mpc` it leaves every two-electron
integral -- and therefore every Pulay term -- exactly as it was.

Units
-----

**The parametrization is in Rydberg atomic units**, as the paper states, while
Mandacaru works in Hartree.  So is the sign trap: Mandacaru's own
:func:`~mandacaru.basis.atomic_solver.lda_correlation` is the Perdew-Zunger
form in *Hartree* (its ``A = 0.0311`` is half the Rydberg ``0.0622``), and it
has to be doubled before it enters anything here.  Every public function below
returns **Hartree**; the Rydberg convention is confined to the private
parametrization.

The parametrization
-------------------

With :math:`4\pi r_s^3/3 = 1/n` and :math:`L` the supercell's linear size, the
jellium electron count is :math:`N = (3/4\pi)(L/r_s)^3`, which sets three
density boundaries: :math:`\gamma = r_s(N{=}2)`,
:math:`\gamma_h = r_s(N{=}12)` and :math:`\gamma_l = r_s(N{=}1/2)`.

Exchange, their Eq. (3):

.. math::

    \epsilon_x(r_s, L) = \begin{cases}
      \dfrac{a_0}{r_s} + \dfrac{a_1}{L^2} r_s + \dfrac{a_2}{L^3} r_s^2,
        & r_s \le \gamma \\[2mm]
      \dfrac{a_3 L^5}{r_s^6}, & \text{otherwise}
    \end{cases}

Correlation, their Eq. (6):

.. math::

    \epsilon_c(r_s, L) = \begin{cases}
      \epsilon_c^{\infty}(r_s) - \dfrac{a_1}{L^2} r_s + \dfrac{g(r_s)}{L^3},
        & r_s \le \gamma_h \\[1mm]
      f(r_s), & \gamma_h < r_s \le \gamma_l \\[1mm]
      0, & \text{otherwise}
    \end{cases}

with :math:`g(r_s) = g_1 r_s \ln r_s + g_2 r_s + g_3 r_s^{3/2} + g_4 r_s^2`.
The :math:`a_1/L^2` terms enter exchange and correlation with **opposite
signs** so that they cancel in :math:`\epsilon_{xc}`, leaving the physical
:math:`\mathcal{O}(1/L^3)` -- which is both the paper's stated design and a
free check on the implementation (:func:`leading_term_cancellation`).

:math:`f` is a cubic on :math:`[\gamma_h, \gamma_l]` carrying **no fitted
parameters**: it is fixed by requiring :math:`\epsilon_c` and its derivative to
be continuous at both ends, four conditions for four coefficients.  It is built
here as :math:`f(r) = (r - \gamma_l)^2[\alpha + \beta(r - \gamma_l)]`, which
satisfies the two conditions at :math:`\gamma_l` identically, leaving
:math:`\alpha` and :math:`\beta` to the two at :math:`\gamma_h`.

Limits worth knowing
--------------------

The functional was fitted for **cubic** supercells.  The paper handles
non-cubic ones with "an effective :math:`L` equal to the size of a cubic
supercell of the same volume" and reports the shape dependence as weak;
:func:`effective_length` does exactly that, and
:class:`KZKCorrection` records how far from cubic the cell was so the
assumption is visible rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KZKCorrection", "effective_length", "fs_exchange",
           "fs_correlation", "fs_exchange_correlation",
           "infinite_exchange_correlation", "kzk_correction",
           "leading_term_cancellation"]

#: Hartree per Rydberg.  The parametrization below is Rydberg throughout.
HARTREE_PER_RYDBERG = 0.5

#: ``a_1, a_2, a_3`` of Eq. (3), Table I, in Rydberg atomic units.
A1, A2, A3 = -2.2037, 0.4710, -0.0150

#: ``g_1 .. g_4`` of Eq. (6), Table I, in Rydberg atomic units.
G1, G2, G3, G4 = 0.1182, 1.1656, -5.2884, -1.1233

#: Densities below this are treated as vacuum and contribute nothing.  The
#: parametrization runs on ``r_s``, which diverges as ``n -> 0``, and a cell
#: with vacuum in it has a great many such points.
DENSITY_FLOOR = 1e-12


def _a0() -> float:
    r"""The infinite-size Slater exchange constant, in Rydberg.

    :math:`\epsilon_x^{\infty} = -\tfrac34 (3/\pi)^{1/3} n^{1/3}` in Hartree,
    which with :math:`n = 3/(4\pi r_s^3)` is :math:`a_0 / r_s`.  Computed
    rather than taken as the paper's rounded ``-0.916``.
    """
    hartree = -0.75 * (9.0 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0)
    return hartree / HARTREE_PER_RYDBERG


#: ``a_0`` of Eq. (3): -0.9163305866 Ry, the paper's "a_0 ~= -0.916 Ry".
A0 = _a0()


def effective_length(volume) -> float:
    """``L`` for a supercell of this volume, in Bohr.

    The functional was fitted on cubic cells, so a non-cubic one is handed the
    side of the cube with the same volume -- the paper's own prescription.
    """
    return float(np.cbrt(float(volume)))


def _rs_at(n_electrons, length) -> float:
    r"""``r_s`` at which the jellium supercell holds ``n_electrons``.

    Inverts :math:`N = (3/4\pi)(L/r_s)^3`.
    """
    return float(length) * (3.0 / (4.0 * np.pi * float(n_electrons))) ** (1 / 3)


def _pz_correlation_rydberg(rs):
    """``(e_c, de_c/dr_s)`` of Perdew-Zunger, in **Rydberg**.

    Mandacaru's own routine is in Hartree and returns ``(e_c, v_c)``; the
    derivative comes from the exact relation
    ``v_c = e_c - (r_s/3) de_c/dr_s`` rather than a finite difference.
    """
    from ..basis.atomic_solver import lda_correlation

    rs = np.asarray(rs, dtype=float)
    density = 3.0 / (4.0 * np.pi * rs ** 3)
    ec, vc = lda_correlation(density)
    derivative = 3.0 * (ec - vc) / rs
    return ec / HARTREE_PER_RYDBERG, derivative / HARTREE_PER_RYDBERG


def _g(rs):
    """``g(r_s)`` of Eq. (6), Rydberg."""
    rs = np.asarray(rs, dtype=float)
    return (G1 * rs * np.log(rs) + G2 * rs + G3 * rs ** 1.5 + G4 * rs ** 2)


def _dg(rs):
    """``dg/dr_s``, Rydberg."""
    rs = np.asarray(rs, dtype=float)
    return (G1 * (np.log(rs) + 1.0) + G2 + 1.5 * G3 * np.sqrt(rs)
            + 2.0 * G4 * rs)


def fs_exchange(rs, length):
    """Finite-size exchange energy per electron, Eq. (3), in **Hartree**."""
    rs = np.asarray(rs, dtype=float)
    L = float(length)
    gamma = _rs_at(2, L)
    low = A0 / rs + (A1 / L ** 2) * rs + (A2 / L ** 3) * rs ** 2
    high = A3 * L ** 5 / rs ** 6
    return np.where(rs <= gamma, low, high) * HARTREE_PER_RYDBERG


def _correlation_high(rs, length):
    """The ``r_s <= gamma_h`` branch and its derivative, Rydberg."""
    L = float(length)
    ec, dec = _pz_correlation_rydberg(rs)
    value = ec - (A1 / L ** 2) * rs + _g(rs) / L ** 3
    slope = dec - A1 / L ** 2 + _dg(rs) / L ** 3
    return value, slope


def _intermediate_cubic(length):
    r"""``(alpha, beta, gamma_h, gamma_l)`` of the intermediate-density cubic.

    :math:`f(r) = (r - \gamma_l)^2 [\alpha + \beta (r - \gamma_l)]` vanishes
    with its derivative at :math:`\gamma_l` by construction, which is the
    continuity with the zero branch.  The two remaining coefficients are fixed
    by matching value and slope at :math:`\gamma_h`.
    """
    L = float(length)
    gamma_h = _rs_at(12, L)
    gamma_l = _rs_at(0.5, L)
    value, slope = _correlation_high(np.array(gamma_h), L)
    value, slope = float(value), float(slope)
    u = gamma_h - gamma_l                       # negative
    beta = (slope - 2.0 * value / u) / u ** 2
    alpha = value / u ** 2 - beta * u
    return alpha, beta, gamma_h, gamma_l


def fs_correlation(rs, length):
    """Finite-size correlation energy per electron, Eq. (6), in **Hartree**."""
    rs = np.asarray(rs, dtype=float)
    L = float(length)
    alpha, beta, gamma_h, gamma_l = _intermediate_cubic(L)

    high, _slope = _correlation_high(rs, L)
    shifted = rs - gamma_l
    middle = shifted ** 2 * (alpha + beta * shifted)

    value = np.where(rs <= gamma_h, high,
                     np.where(rs <= gamma_l, middle, 0.0))
    return value * HARTREE_PER_RYDBERG


def fs_exchange_correlation(rs, length):
    """``e_x + e_c`` at finite supercell size, in **Hartree**."""
    return fs_exchange(rs, length) + fs_correlation(rs, length)


def infinite_exchange_correlation(rs):
    """``e_xc`` in the infinite-size limit, in **Hartree**.

    Slater exchange plus Perdew-Zunger correlation -- the same functional
    Mandacaru's atomic solver uses, so the two halves of the correction are
    built from one parametrization.
    """
    rs = np.asarray(rs, dtype=float)
    ec, _slope = _pz_correlation_rydberg(rs)
    return (A0 / rs + ec) * HARTREE_PER_RYDBERG


def leading_term_cancellation(rs, length) -> float:
    r"""How completely the ``a_1 / L^2`` terms cancel in ``e_xc``.

    The paper builds exchange with :math:`+a_1 r_s/L^2` and correlation with
    :math:`-a_1 r_s/L^2` precisely so the pair cancels and
    :math:`\epsilon_{xc}` scales as :math:`1/L^3`.  In the high-density branch
    the cancellation is exact, so this returns the residual as a fraction of
    the term itself: it must be at round-off wherever
    ``r_s <= min(gamma, gamma_h)``.
    """
    rs = np.asarray(rs, dtype=float)
    L = float(length)
    term = abs(A1 / L ** 2) * rs * HARTREE_PER_RYDBERG
    total = fs_exchange_correlation(rs, L)
    without = (A0 / rs + _pz_correlation_rydberg(rs)[0]
               + (A2 / L ** 3) * rs ** 2 + _g(rs) / L ** 3)
    residual = np.abs(total - without * HARTREE_PER_RYDBERG)
    scale = np.maximum(term, 1e-300)
    return float(np.max(residual / scale))


@dataclass
class KZKCorrection:
    r"""The size-dependent-LDA finite-size correction.

    Attributes
    ----------
    correction : float
        :math:`\Delta E = \int n (\epsilon_{xc}^{\infty} -
        \epsilon_{xc}^{\text{FS}})`, Hartree **per supercell**.  Add it to the
        supercell energy.
    xc_infinite, xc_finite : float
        The two integrals separately, Hartree per supercell.
    length : float
        The effective cubic ``L`` in Bohr.
    volume : float
        Supercell volume in Bohr^3.
    n_cells : int
        Primitive cells in the supercell.
    n_electrons : float
        :math:`\int n`, from the grid -- the check that the density used is the
        one the run produced.
    cubic_deviation : float
        How far the cell is from cubic: ``max|l_i - L| / L`` over the three
        lattice-vector lengths.  The functional was fitted on cubic cells, so a
        large value means the effective-``L`` prescription is working hard.
    rs_range : tuple of float
        Smallest and largest ``r_s`` that contributed, in Bohr.  The
        parametrization is built on jellium; a range reaching far past
        ``gamma_l`` means much of the cell is vacuum the fit never saw.
    energy_per_cell : float or None
        The uncorrected per-cell energy in Hartree, when supplied.
    """

    correction: float
    xc_infinite: float
    xc_finite: float
    length: float
    volume: float
    n_cells: int
    n_electrons: float
    cubic_deviation: float
    rs_range: tuple
    energy_per_cell: float | None = None

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

    def summary(self) -> str:
        """One line, in eV."""
        from ..units import from_hartree

        return (f"KZK finite-size correction: "
                f"{from_hartree(self.correction_per_cell, 'eV'):+.6f} eV/cell "
                f"(L = {self.length:.3f} Bohr, {self.n_electrons:.4f} "
                f"electrons on the grid, r_s in "
                f"[{self.rs_range[0]:.3f}, {self.rs_range[1]:.3f}] Bohr, "
                f"cubic deviation {self.cubic_deviation:.1%})")


def kzk_correction(density, volume_element, cell, *, n_cells: int = 1,
                   energy_per_cell=None, floor: float = DENSITY_FLOOR):
    r"""Integrate :math:`n (\epsilon_{xc}^{\infty} - \epsilon_{xc}^{\text{FS}})`.

    Parameters
    ----------
    density : array_like
        The electron density on the grid, in Bohr^-3.
    volume_element : float
        ``dV`` of the grid, in Bohr^3.
    cell : (3, 3) array_like
        Supercell lattice vectors as rows, in Bohr.
    n_cells : int
        Primitive cells in the supercell.
    energy_per_cell : float, optional
        The uncorrected per-cell energy in Hartree, to carry through.
    floor : float
        Densities at or below this are skipped as vacuum.

    Returns
    -------
    KZKCorrection
    """
    density = np.asarray(density, dtype=float).ravel()
    cell = np.asarray(cell, dtype=float)
    volume = abs(float(np.linalg.det(cell)))
    length = effective_length(volume)
    lengths = np.linalg.norm(cell, axis=1)
    cubic_deviation = float(np.max(np.abs(lengths - length)) / length)

    dV = float(volume_element)
    n_electrons = float(density.sum() * dV)

    keep = density > floor
    if not np.any(keep):
        raise ValueError(
            "the density is everywhere below the vacuum floor, so there is "
            "nothing to integrate; check that the state and grid are the "
            "ones the run produced.")
    live = density[keep]
    rs = (3.0 / (4.0 * np.pi * live)) ** (1.0 / 3.0)

    infinite = infinite_exchange_correlation(rs)
    finite = fs_exchange_correlation(rs, length)

    xc_infinite = float((live * infinite).sum() * dV)
    xc_finite = float((live * finite).sum() * dV)
    return KZKCorrection(
        correction=xc_infinite - xc_finite,
        xc_infinite=xc_infinite,
        xc_finite=xc_finite,
        length=length,
        volume=volume,
        n_cells=int(n_cells),
        n_electrons=n_electrons,
        cubic_deviation=cubic_deviation,
        rs_range=(float(rs.min()), float(rs.max())),
        energy_per_cell=(None if energy_per_cell is None
                         else float(energy_per_cell)))
