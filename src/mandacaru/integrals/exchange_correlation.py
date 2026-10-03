# -*- coding: utf-8 -*-
# file: integrals/exchange_correlation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Exchange-correlation energy and potential on the three-dimensional grid.

The functionals are written once, as energies per unit volume: LDA and PBE in
:mod:`mandacaru.basis.xc` (:func:`~mandacaru.basis.xc.xc_partials`, shared
with the radial atom), the r\ :sup:`2`\ SCAN meta-GGA in
:mod:`mandacaru.basis.r2scan`.  This module evaluates them on the real-space
grid the integrals already use, for the Kohn-Sham solver behind
``Mandacaru(method="dft")``:

.. math::

    E_{xc} = \int f(\rho, \sigma, \tau)\,d^3r, \qquad
    v_{xc} = \frac{\partial f}{\partial\rho}
        - 2\nabla\cdot\Big(\frac{\partial f}{\partial\sigma}\nabla\rho\Big),
    \qquad \sigma = |\nabla\rho|^2 .

A meta-GGA also depends on :math:`\tau = \tfrac12\sum_i f_i|\nabla\psi_i|^2`,
which is not a function of the density, so its derivative
:math:`\partial f/\partial\tau` is returned separately
(:attr:`XCTerms.tau_potential`) and enters the Kohn-Sham matrix as
:math:`\tfrac12\int \partial_\tau f\,\nabla\phi_p^*\cdot\nabla\phi_q`.

Gradients and divergences are spectral (FFT) derivatives over the grid, with
the reciprocal vectors of the cell it spans, so skewed cells are handled
exactly.  A molecular grid is a zero-padded box rather than a period, which is
harmless because the density and every flux vanish at its faces long before
the box ends; a crystal's grid *is* the period.

Partial core densities
----------------------
A dataset unscreened with a core density inside :math:`v_{xc}` (the nonlinear
core correction) needs that same density added wherever the functional is
evaluated: :func:`xc_core_density` picks it per dataset and
:func:`core_density_on_grid` sums it on the grid.  For a meta-GGA each atom's
core also contributes its von Weizsaecker kinetic-energy density,
:math:`\tau_c = \tilde\rho_c'^2/8\tilde\rho_c`, the exact one for a
single-orbital density and the one that leaves the core iso-orbital
(:math:`\bar\alpha = 0`) on its own.  It is evaluated **radially**
(:func:`core_tau_function`) and placed on the grid like the core density
(:func:`core_tau_on_grid`), never as :math:`|\nabla\rho|^2/8\rho` of the
sampled density: that divides by the density wherever it is small, and a
sampled or Fourier-filtered core is small with a finite gradient exactly
where its tail rings -- the spikes moved with the atoms and made the r2SCAN
energy of bulk silicon rough by 170 uHa.

Adding a core density changes what the energy of the isolated reference atom
evaluates to.  :func:`core_correction_offset` is the per-atom constant that
restores the valence-only convention every dataset's energies are quoted in,
so a Kohn-Sham total energy and a many-body total energy built on the same
datasets share their zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..basis.xc import DENSITY_FLOOR, xc_partials, xc_potential

#: Functionals the grid evaluator accepts, as ``Mandacaru(method="dft", xc=...)``.
GRID_FUNCTIONALS = ("lda", "pbe", "r2scan")

#: Spellings accepted for each functional.
_ALIASES = {"lda": "lda", "pz": "lda", "pz81": "lda", "ldapz": "lda",
            "pbe": "pbe", "gga": "pbe", "ggapbe": "pbe",
            "r2scan": "r2scan", "r²scan": "r2scan", "mggar2scan": "r2scan"}

#: Below this density the gradient terms are dropped on the grid.  The reduced
#: gradient diverges in the exponential tail, where the density carries no
#: weight but the spectral derivative carries round-off.
GRADIENT_DENSITY_FLOOR = 1e-10


def resolve_functional(name: str) -> str:
    """Canonical name (``"lda"``, ``"pbe"`` or ``"r2scan"``) of a spec."""
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in _ALIASES:
        raise ValueError(
            f"unknown exchange-correlation functional {name!r}; "
            f"available: {', '.join(GRID_FUNCTIONALS)}")
    return _ALIASES[key]


def is_meta_gga(name: str) -> bool:
    """Whether the functional depends on the kinetic-energy density."""
    return resolve_functional(name) == "r2scan"


@dataclass(frozen=True)
class XCTerms:
    """The functional evaluated on one density.

    ``energy`` is :math:`E_{xc}` (Hartree), ``potential`` the flat local
    :math:`v_{xc}` and ``tau_potential`` :math:`\\partial f/\\partial\\tau`
    (``None`` unless the functional is a meta-GGA).
    """

    energy: float
    potential: np.ndarray
    tau_potential: np.ndarray | None = None


def _wavevectors(grid):
    """Cartesian ``(Gx, Gy, Gz)`` of the grid's FFT, each ``(n1, n2, n3)``.

    From the reciprocal vectors of the cell the grid spans
    (:func:`~mandacaru.integrals.reciprocal.wavevectors`), so a skewed cell --
    hexagonal, monoclinic, triclinic -- is differentiated as exactly as an
    orthogonal one.
    """
    from .reciprocal import wavevectors
    return tuple(wavevectors(grid))


def gradient(grid, field: np.ndarray) -> np.ndarray:
    """Spectral gradient of flat fields: ``(..., ngrid) -> (3, ..., ngrid)``.

    Real input gives a real gradient; complex input (orbitals) a complex one.
    """
    field = np.asarray(field)
    lead = field.shape[:-1]
    values = np.fft.fftn(field.reshape(*lead, *grid.shape), axes=(-3, -2, -1))
    out = []
    for k in _wavevectors(grid):
        component = np.fft.ifftn(1j * k * values, axes=(-3, -2, -1))
        out.append(component.reshape(*lead, -1))
    out = np.stack(out)
    return np.real(out) if np.isrealobj(field) else out


def divergence(grid, vector: np.ndarray) -> np.ndarray:
    """Spectral divergence of a flat ``(3, ngrid)`` real vector field."""
    total = np.zeros(grid.shape, dtype=complex)
    for component, k in zip(vector, _wavevectors(grid)):
        total += 1j * k * np.fft.fftn(component.reshape(grid.shape))
    return np.real(np.fft.ifftn(total)).reshape(-1)


def evaluate(grid, density: np.ndarray, functional: str = "lda", *,
             relativistic: bool = False, tau: np.ndarray | None = None
             ) -> XCTerms:
    r"""``E_xc``, ``v_xc`` (and ``df/dtau``) of a flat density on ``grid``.

    Parameters
    ----------
    grid : Grid
        The integration grid (Bohr).
    density : ndarray
        Electron density on the flat grid (electrons per Bohr^3), any partial
        core already added.
    functional : str
        ``"lda"``, ``"pbe"`` or ``"r2scan"`` (:func:`resolve_functional`).
    relativistic : bool
        The relativistic exchange factor of a relativistic reference atom
        (LDA and PBE; see :func:`~mandacaru.basis.xc.relativistic_exchange_factors`).
    tau : ndarray, optional
        Kinetic-energy density on the grid; required by a meta-GGA.
    """
    key = resolve_functional(functional)
    rho = np.maximum(np.asarray(density, dtype=float), 0.0)
    dV = grid.dV
    if key == "lda":
        f, df_drho, _ = xc_partials(rho, None, "lda", relativistic)
        return XCTerms(energy=float(np.sum(f) * dV), potential=df_drho)

    grad = gradient(grid, rho)
    weighted = rho > GRADIENT_DENSITY_FLOOR
    sigma = np.where(weighted, np.sum(grad * grad, axis=0), 0.0)
    if key == "pbe":
        g = np.sqrt(sigma)
        f, df_drho, df_dg = xc_partials(rho, g, "pbe", relativistic)
        # df/dsigma = (df/dg) / 2g, zero where the gradient term is dropped.
        with np.errstate(divide="ignore", invalid="ignore"):
            df_dsigma = np.where(weighted & (g > 0.0) & (rho > DENSITY_FLOOR),
                                 0.5 * df_dg / g, 0.0)
        df_dtau = None
    else:
        if tau is None:
            raise ValueError("a meta-GGA needs the kinetic-energy density tau")
        from ..basis import r2scan
        f, df_drho, df_dsigma, df_dtau = r2scan.partials(
            np.where(weighted, rho, 0.0), sigma, np.asarray(tau, dtype=float))
    potential = df_drho - divergence(grid, 2.0 * df_dsigma * grad)
    return XCTerms(energy=float(np.sum(f) * dV),
                   potential=np.asarray(potential, dtype=float),
                   tau_potential=df_dtau)


# --------------------------------------------------------------------------- #
# Partial core densities.
# --------------------------------------------------------------------------- #

def xc_core_density(dataset) -> np.ndarray | None:
    """The radial core density ``dataset`` was unscreened with, or ``None``.

    A norm-conserving dataset stores exactly that density as
    ``core_density``.  A PAW-LCAO dataset stores the **true** core there and
    unscreens with its smooth core ``smooth_core_density``.
    """
    record = getattr(dataset, "nlcc", None) or {}
    if not record.get("applied"):
        return None
    if hasattr(dataset, "smooth_core_density"):
        if record.get("source") != "smooth_core":
            raise NotImplementedError(
                f"the {dataset.symbol} dataset was unscreened with a core "
                f"density re-pseudized at r_nlcc = {record.get('r_nlcc')}, "
                "which the dataset does not store")
        core = dataset.smooth_core_density
    else:
        core = dataset.core_density
    if core is None or not np.any(core):
        return None
    return np.asarray(core, dtype=float)


def core_density_function(dataset):
    r"""``r -> core density`` of ``dataset``'s core correction, or ``None``.

    A cubic spline of the radial table, zero beyond it.  Linear interpolation
    would put a kink at every table node, and a grid point crossing one as
    its atom moves makes the energy rough at the micro-Hartree level -- too
    rough for a finite-difference check of the force.
    """
    core = xc_core_density(dataset)
    if core is None:
        return None
    return _radial_spline(np.asarray(dataset.r, dtype=float), core)


#: Below this fraction of its peak a core density's kinetic-energy density
#: is taken as zero (the ratio is then all round-off).
CORE_TAU_FLOOR = 1e-12


def core_tau_function(dataset):
    r"""``r -> tau_c`` of ``dataset``'s core correction, or ``None``.

    :math:`\tau_c = \tilde\rho_c'(r)^2 / 8\tilde\rho_c(r)` from the cubic
    spline of the radial table and its derivative, zero below
    :data:`CORE_TAU_FLOOR` of the core's peak; itself splined, so it is as
    smooth as the core.  A smooth core decays smoothly, and so does this
    ratio.
    """
    core = xc_core_density(dataset)
    if core is None:
        return None
    from scipy.interpolate import CubicSpline

    r = np.asarray(dataset.r, dtype=float)
    spline = CubicSpline(r, core)
    slope = spline.derivative()(r)
    floor = CORE_TAU_FLOOR * float(np.max(np.abs(core)))
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(core > floor, slope * slope / (8.0 * core), 0.0)
    return _radial_spline(r, tau)


def _radial_spline(r, values):
    """``radius -> values`` by a cubic spline, zero beyond the table."""
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(r, values)
    edge = float(r[-1])

    def function(radius):
        radius = np.asarray(radius, dtype=float)
        return np.where(radius <= edge, spline(np.clip(radius, r[0], edge)),
                        0.0)
    return function


def core_tau_on_grid(grid, datasets, centers) -> np.ndarray | None:
    """Sum of the atoms' core kinetic-energy densities on the flat grid.

    The counterpart of :func:`core_density_on_grid`, same frame and same
    ``None`` when no dataset carries a core correction.
    """
    return _place(grid, [core_tau_function(d) for d in datasets or ()],
                  centers)


def _place(grid, functions, centers) -> np.ndarray | None:
    total = None
    for function, center in zip(functions, centers):
        if function is None:
            continue
        distance = np.sqrt((grid.X - center[0]) ** 2
                           + (grid.Y - center[1]) ** 2
                           + (grid.Z - center[2]) ** 2).reshape(-1)
        values = function(distance)
        total = values if total is None else total + values
    return total


def core_density_on_grid(grid, datasets, centers) -> np.ndarray | None:
    r"""Sum of the datasets' partial core densities on the flat grid.

    ``centers`` are the atomic positions **in Bohr** (the frame of
    :class:`~mandacaru.integrals.potentials.Potentials`), one per dataset.
    ``None`` when no dataset carries a core correction.
    """
    return _place(grid, [core_density_function(d) for d in datasets or ()],
                  centers)


def _valence_xc_terms(r, valence, core, functional, relativistic):
    r""":math:`E_{xc}[\rho_v+\rho_c] - \int\rho_v v_{xc}[\rho_v+\rho_c]` radially."""
    shell = 4.0 * np.pi * r * r
    e_xc, v_xc = xc_potential(r, valence + core, functional,
                              relativistic=relativistic)
    return float(np.trapezoid((valence + core) * e_xc * shell, r)
                 - np.trapezoid(valence * v_xc * shell, r))


def core_correction_offset(dataset) -> float:
    r"""Per-atom constant (Hartree) restoring the valence-only energy zero.

    The reference atom's Kohn-Sham energy with the core inside the functional
    is :math:`\sum f\varepsilon - E_H + E_{xc}[\tilde\rho_v+\tilde\rho_c] -
    \int\tilde\rho_v v_{xc}[\tilde\rho_v+\tilde\rho_c]`; a dataset's energies
    are quoted with :math:`E_{xc}[\tilde\rho_v] - \int\tilde\rho_v
    v_{xc}[\tilde\rho_v]` instead.  The difference, evaluated in the
    dataset's own functional, is a constant of the species.  Zero without a
    core correction.
    """
    core = xc_core_density(dataset)
    valence = getattr(dataset, "valence_density", None)
    if core is None or valence is None:
        return 0.0
    r = np.asarray(dataset.r, dtype=float)
    valence = np.asarray(valence, dtype=float)
    functional = getattr(dataset, "xc", "lda") or "lda"
    relativistic = bool(getattr(dataset, "relativistic_exchange", False))
    bare = _valence_xc_terms(r, valence, np.zeros_like(r), functional,
                             relativistic)
    dressed = _valence_xc_terms(r, valence, core, functional, relativistic)
    return bare - dressed
