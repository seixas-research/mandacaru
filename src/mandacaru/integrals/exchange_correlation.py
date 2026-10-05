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

A screened hybrid (:data:`HYBRIDS`, HSE06 from :mod:`mandacaru.basis.hse`)
is evaluated here only in its semilocal part, and only when the caller says
it adds the short-range exact exchange (``screening=``): a hybrid without it
would be a different functional.

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

from ..basis import hse
from ..basis.xc import DENSITY_FLOOR, xc_partials, xc_potential

#: Functionals the grid evaluator accepts, as ``Mandacaru(method="dft", xc=...)``.
GRID_FUNCTIONALS = ("lda", "pbe", "r2scan", "hse06")

#: Spellings accepted for each functional.
_ALIASES = {"lda": "lda", "pz": "lda", "pz81": "lda", "ldapz": "lda",
            "pbe": "pbe", "gga": "pbe", "ggapbe": "pbe",
            "r2scan": "r2scan", "r²scan": "r2scan", "mggar2scan": "r2scan",
            "hse06": "hse06", "hse": "hse06"}

#: Screened hybrids: ``(omega, fraction)`` -- the range separation (1/Bohr)
#: and the fraction of short-range exchange that is exact.  The grid
#: evaluates their semilocal part only; the Kohn-Sham solver adds
#: ``fraction`` of the short-range exact exchange.
HYBRIDS = {"hse06": (hse.OMEGA, hse.EXACT_FRACTION)}

#: Below this density the gradient terms are dropped on the grid.  The reduced
#: gradient diverges in the exponential tail, where the density carries no
#: weight but the spectral derivative carries round-off.
GRADIENT_DENSITY_FLOOR = 1e-10


def resolve_functional(name: str) -> str:
    """Canonical name (``"lda"``, ``"pbe"``, ``"r2scan"`` or ``"hse06"``)
    of a spec."""
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in _ALIASES:
        raise ValueError(
            f"unknown exchange-correlation functional {name!r}; "
            f"available: {', '.join(GRID_FUNCTIONALS)}")
    return _ALIASES[key]


def is_meta_gga(name: str) -> bool:
    """Whether the functional depends on the kinetic-energy density."""
    return resolve_functional(name) == "r2scan"


def is_hybrid(name: str) -> bool:
    """Whether the functional mixes in exact exchange (:data:`HYBRIDS`)."""
    return resolve_functional(name) in HYBRIDS


def takes_relativistic_exchange(name: str) -> bool:
    """Whether the relativistic exchange factor applies (LDA and PBE).

    r\\ :sup:`2`\\ SCAN has no relativistic form, and a hybrid's exact
    exchange has none either, so neither takes the factor.
    """
    return resolve_functional(name) in ("lda", "pbe")


def _screening(key: str, screening):
    """The ``(omega, fraction)`` a hybrid's semilocal part is evaluated with.

    A hybrid without its exact-exchange part is a different functional, so
    the caller has to say it adds that part by passing ``screening``
    (:data:`HYBRIDS` holds the defaults).
    """
    if key not in HYBRIDS:
        return None
    if screening is None:
        raise ValueError(
            f"xc={key!r} is a hybrid: the grid evaluates only its semilocal "
            "part, and the solver must add the short-range exact exchange "
            "and say so by passing screening=(omega, fraction).")
    return float(screening[0]), float(screening[1])


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


def _clipped(raw, potential) -> np.ndarray:
    """The potential of a density the functional clipped at zero: the
    functional sees ``max(raw, 0)``, so its derivative in ``raw`` vanishes
    where ``raw`` is negative (half a filtered partial core can make a
    spin channel's density slightly negative where the channel is empty)."""
    return np.where(np.asarray(raw, dtype=float) < 0.0, 0.0, potential)


def _short_range_exchange(grid, density, omega: float, scale: float = 1.0):
    r""":math:`E_x^{\rm SR}` and :math:`v_x^{\rm SR}` of one density, for the
    hole-model exchange of :func:`~mandacaru.basis.hse.short_range_partials`.

    ``scale`` is the spin scaling: a channel's exchange is
    :math:`\tfrac12 E_x[2\rho_\sigma]`, which is ``scale = 2`` applied to
    :math:`\rho_\sigma` (energy halved, potential at the doubled density).
    """
    rho = scale * np.maximum(np.asarray(density, dtype=float), 0.0)
    grad = gradient(grid, rho)
    weighted = rho > GRADIENT_DENSITY_FLOOR
    sigma = np.where(weighted, np.sum(grad * grad, axis=0), 0.0)
    f, df_drho, df_dsigma = hse.short_range_partials(rho, sigma, omega)
    df_dsigma = np.where(weighted, df_dsigma, 0.0)
    # d/d rho_s of (1/scale) E[scale rho_s] is the potential of E evaluated
    # at the scaled density.
    potential = _clipped(density, df_drho
                         - divergence(grid, 2.0 * df_dsigma * grad))
    return float(np.sum(f) * grid.dV) / scale, potential


def evaluate(grid, density: np.ndarray, functional: str = "lda", *,
             relativistic: bool = False, tau: np.ndarray | None = None,
             screening=None, exchange_density=None) -> XCTerms:
    r"""``E_xc``, ``v_xc`` (and ``df/dtau``) of a flat density on ``grid``.

    Parameters
    ----------
    grid : Grid
        The integration grid (Bohr).
    density : ndarray
        Electron density on the flat grid (electrons per Bohr^3), any partial
        core already added.
    functional : str
        ``"lda"``, ``"pbe"``, ``"r2scan"`` or ``"hse06"``
        (:func:`resolve_functional`).
    relativistic : bool
        The relativistic exchange factor of a relativistic reference atom
        (LDA and PBE; see :func:`~mandacaru.basis.xc.relativistic_exchange_factors`).
    tau : ndarray, optional
        Kinetic-energy density on the grid; required by a meta-GGA.
    screening : (float, float), optional
        ``(omega, fraction)`` of a hybrid (:data:`HYBRIDS`), required by one.
        Only the semilocal part is evaluated here -- the full-range
        exchange and the correlation of ``density``, minus ``fraction``
        times the short-range exchange of ``exchange_density`` -- and
        passing this states that the caller adds ``fraction`` of the
        short-range exact exchange.
    exchange_density : ndarray, optional
        A hybrid's: the density whose short-range semilocal exchange the
        exact exchange replaces -- the valence density, the one the orbitals
        carry (default ``density``).  A partial core stays with its full
        semilocal exchange, core-valence included: it has no orbitals to
        take exact exchange from.  The potential is the derivative with
        respect to the valence density, which both parts depend on alike.
    """
    key = resolve_functional(functional)
    screening = _screening(key, screening)
    rho = np.maximum(np.asarray(density, dtype=float), 0.0)
    dV = grid.dV
    if key == "lda":
        f, df_drho, _ = xc_partials(rho, None, "lda", relativistic)
        return XCTerms(energy=float(np.sum(f) * dV),
                       potential=_clipped(density, df_drho))

    grad = gradient(grid, rho)
    weighted = rho > GRADIENT_DENSITY_FLOOR
    sigma = np.where(weighted, np.sum(grad * grad, axis=0), 0.0)
    df_dtau = None
    if key == "pbe":
        g = np.sqrt(sigma)
        f, df_drho, df_dg = xc_partials(rho, g, "pbe", relativistic)
        # df/dsigma = (df/dg) / 2g, zero where the gradient term is dropped.
        with np.errstate(divide="ignore", invalid="ignore"):
            df_dsigma = np.where(weighted & (g > 0.0) & (rho > DENSITY_FLOOR),
                                 0.5 * df_dg / g, 0.0)
    elif key in HYBRIDS:
        # The full-range part through the spin-resolved kernel at zeta = 0:
        # f(rho, sigma) = F(rho/2, rho/2, sigma/4, sigma/4, sigma/4).
        from ..basis.xc_spin import spin_partials
        quarter = 0.25 * sigma
        f, v_up, v_dn, d_uu, d_ud, d_dd, _t, _u = spin_partials(
            key, 0.5 * rho, 0.5 * rho, quarter, quarter, quarter)
        df_drho = 0.5 * (v_up + v_dn)
        df_dsigma = np.where(weighted, 0.25 * (d_uu + d_ud + d_dd), 0.0)
    else:
        if tau is None:
            raise ValueError("a meta-GGA needs the kinetic-energy density tau")
        from ..basis import r2scan
        f, df_drho, df_dsigma, df_dtau = r2scan.partials(
            np.where(weighted, rho, 0.0), sigma, np.asarray(tau, dtype=float))
    potential = _clipped(density, df_drho
                         - divergence(grid, 2.0 * df_dsigma * grad))
    energy = float(np.sum(f) * dV)
    if screening is not None and screening[1] != 0.0:
        omega, fraction = screening
        exchange = rho if exchange_density is None else exchange_density
        e_sr, v_sr = _short_range_exchange(grid, exchange, omega)
        energy -= fraction * e_sr
        potential = potential - fraction * v_sr
    return XCTerms(energy=energy,
                   potential=np.asarray(potential, dtype=float),
                   tau_potential=df_dtau)


@dataclass(frozen=True)
class SpinXCTerms:
    """The spin-polarized functional on one pair of densities: ``energy``
    (Hartree), the flat potentials of each channel and, for a meta-GGA, the
    two :math:`\\partial f/\\partial\\tau_\\sigma`."""

    energy: float
    potential_up: np.ndarray
    potential_dn: np.ndarray
    tau_potential_up: np.ndarray | None = None
    tau_potential_dn: np.ndarray | None = None


def evaluate_spin(grid, density_up, density_dn, functional: str = "lda", *,
                  relativistic: bool = False, tau_up=None, tau_dn=None,
                  screening=None, exchange_densities=None) -> SpinXCTerms:
    r"""The spin-polarized :func:`evaluate`: ``E_xc`` and each channel's
    :math:`v_\sigma = \partial f/\partial\rho_\sigma -
    \nabla\cdot(2\,\partial f/\partial\sigma_{\sigma\sigma}\nabla\rho_\sigma
    + \partial f/\partial\sigma_{\uparrow\downarrow}\nabla\rho_{\bar\sigma})`
    (:mod:`mandacaru.basis.xc_spin`).  A spin-unpolarized partial core is the
    caller's to split: half of it in each channel.  ``screening`` as for
    :func:`evaluate`; ``exchange_densities`` is the ``(up, down)`` pair of
    its ``exchange_density`` (the short-range exchange of each channel
    follows the exact spin scaling).
    """
    from ..basis.xc_spin import spin_partials

    key = resolve_functional(functional)
    screening = _screening(key, screening)
    up = np.maximum(np.asarray(density_up, dtype=float), 0.0)
    dn = np.maximum(np.asarray(density_dn, dtype=float), 0.0)
    dV = grid.dV
    if key == "lda":
        f, v_up, v_dn, *_ = spin_partials("lda", up, dn,
                                          relativistic=relativistic)
        return SpinXCTerms(energy=float(np.sum(f) * dV),
                           potential_up=_clipped(density_up, v_up),
                           potential_dn=_clipped(density_dn, v_dn))
    # As `evaluate`: the gradients of the full densities, and the gradient
    # terms (not the densities) dropped below the floor -- masking the
    # densities themselves would ring through the spectral derivative.
    weighted = (up + dn) > GRADIENT_DENSITY_FLOOR
    grad_up, grad_dn = gradient(grid, up), gradient(grid, dn)
    s_uu = np.where(weighted, np.sum(grad_up * grad_up, axis=0), 0.0)
    s_ud = np.where(weighted, np.sum(grad_up * grad_dn, axis=0), 0.0)
    s_dd = np.where(weighted, np.sum(grad_dn * grad_dn, axis=0), 0.0)
    if key == "r2scan":
        if tau_up is None or tau_dn is None:
            raise ValueError("a meta-GGA needs the kinetic-energy densities")
        tau_up = np.asarray(tau_up, dtype=float)
        tau_dn = np.asarray(tau_dn, dtype=float)
        # The meta-GGA is not evaluated below the floor at all, as in
        # `evaluate` (its iso-orbital indicator is noise there).
        up, dn = np.where(weighted, up, 0.0), np.where(weighted, dn, 0.0)
    f, v_up, v_dn, d_uu, d_ud, d_dd, t_up, t_dn = spin_partials(
        key, up, dn, s_uu, s_ud, s_dd, tau_up, tau_dn,
        relativistic=relativistic)
    d_uu, d_ud, d_dd = (np.where(weighted, d, 0.0)
                        for d in (d_uu, d_ud, d_dd))
    potential_up = _clipped(density_up, v_up - divergence(
        grid, 2.0 * d_uu * grad_up + d_ud * grad_dn))
    potential_dn = _clipped(density_dn, v_dn - divergence(
        grid, 2.0 * d_dd * grad_dn + d_ud * grad_up))
    energy = float(np.sum(f) * dV)
    if screening is not None and screening[1] != 0.0:
        omega, fraction = screening
        pair = ((up, dn) if exchange_densities is None
                else exchange_densities)
        e_up, w_up = _short_range_exchange(grid, pair[0], omega, scale=2.0)
        e_dn, w_dn = _short_range_exchange(grid, pair[1], omega, scale=2.0)
        energy -= fraction * (e_up + e_dn)
        potential_up = potential_up - fraction * w_up
        potential_dn = potential_dn - fraction * w_dn
    meta = key == "r2scan"
    return SpinXCTerms(energy=energy,
                       potential_up=np.asarray(potential_up, dtype=float),
                       potential_dn=np.asarray(potential_dn, dtype=float),
                       tau_potential_up=t_up if meta else None,
                       tau_potential_dn=t_dn if meta else None)


# --------------------------------------------------------------------------- #
# Partial core densities.
# --------------------------------------------------------------------------- #

def xc_core_density(dataset) -> np.ndarray | None:
    """The radial core density ``dataset`` was unscreened with, or ``None``.

    A PAW-LCAO dataset stores the **true** core as ``core_density`` and
    unscreens with its smooth core ``smooth_core_density``.
    """
    record = getattr(dataset, "nlcc", None) or {}
    if not record.get("applied"):
        return None
    if record.get("source") != "smooth_core":
        raise NotImplementedError(
            f"the {dataset.symbol} dataset was unscreened with a core "
            f"density re-pseudized at r_nlcc = {record.get('r_nlcc')}, "
            "which the dataset does not store")
    core = dataset.smooth_core_density
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
