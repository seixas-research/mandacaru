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
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .. import _native

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

#: Densities below this are treated as vacuum and contribute nothing.
DENSITY_FLOOR = 1e-12


def _a0() -> float:
    return float(_native.kzk_a0())


#: ``a_0`` of Eq. (3): -0.9163305866 Ry, the paper's "a_0 ~= -0.916 Ry".
A0 = _a0()


def effective_length(volume) -> float:
    """``L`` for a supercell of this volume, in Bohr."""
    return float(_native.kzk_effective_length(float(volume)))


def _rs_at(n_electrons, length) -> float:
    """``r_s`` at which the jellium supercell holds ``n_electrons``."""
    return float(_native.kzk_rs_at(float(n_electrons), float(length)))


def _pz_correlation_rydberg(rs):
    """``(e_c, de_c/dr_s)`` of Perdew-Zunger, in **Rydberg**."""
    return _native.kzk_pz_correlation_rydberg(rs)


def _g(rs):
    """``g(r_s)`` of Eq. (6), Rydberg."""
    return _native.kzk_g(rs)


def _dg(rs):
    """``dg/dr_s``, Rydberg."""
    return _native.kzk_dg(rs)


def fs_exchange(rs, length):
    """Finite-size exchange energy per electron, Eq. (3), in **Hartree**."""
    return _native.kzk_fs_exchange(rs, float(length))


def _intermediate_cubic(length):
    """``(alpha, beta, gamma_h, gamma_l)`` of the intermediate-density cubic."""
    return _native.kzk_intermediate_cubic(float(length))


def fs_correlation(rs, length):
    """Finite-size correlation energy per electron, Eq. (6), in **Hartree**."""
    return _native.kzk_fs_correlation(rs, float(length))


def fs_exchange_correlation(rs, length):
    """``e_x + e_c`` at finite supercell size, in **Hartree**."""
    return _native.kzk_fs_exchange_correlation(rs, float(length))


def infinite_exchange_correlation(rs):
    """``e_xc`` in the infinite-size limit, in **Hartree**."""
    return _native.kzk_infinite_exchange_correlation(rs)


def leading_term_cancellation(rs, length) -> float:
    """How completely the ``a_1 / L^2`` terms cancel in ``e_xc``."""
    return float(_native.kzk_leading_term_cancellation(
        np.ascontiguousarray(rs, dtype=float), float(length)
    ))


@dataclass
class KZKCorrection:
    r"""The size-dependent-LDA finite-size correction."""

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
    res = _native.kzk_correction(
        np.ascontiguousarray(density, dtype=float),
        float(volume_element),
        np.ascontiguousarray(cell, dtype=float),
        int(n_cells),
        float(floor),
        None,
    )
    return KZKCorrection(
        correction=float(res["correction"]),
        xc_infinite=float(res["xc_infinite"]),
        xc_finite=float(res["xc_finite"]),
        length=float(res["length"]),
        volume=float(res["volume"]),
        n_cells=int(res["n_cells"]),
        n_electrons=float(res["n_electrons"]),
        cubic_deviation=float(res["cubic_deviation"]),
        rs_range=(float(res["rs_range"][0]), float(res["rs_range"][1])),
        energy_per_cell=(None if energy_per_cell is None
                         else float(energy_per_cell)))
