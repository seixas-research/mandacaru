# -*- coding: utf-8 -*-
# file: basis/filtering.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Fourier filtering of radial basis functions -- the cure for the egg-box.

Why
---
Every integral in Mandacaru is a sum over a uniform real-space grid of spacing
:math:`h`, so the grid can only represent wave-vectors up to its Nyquist value
:math:`k_N = \pi/h`.  Whatever a basis function carries above :math:`k_N` is
**aliased**: it does not disappear, it folds back onto the represented
wave-vectors with a phase that depends on where the function's center sits
between two nodes.  Rigidly translating a molecule by a fraction of :math:`h`
therefore changes the kinetic energy, the local-potential integral and the
electron repulsion -- the *egg-box* -- and :func:`atoms.get_forces` faithfully
differentiates that artifact.  Measured on water / PAW-SZ, the net force
(which a free molecule's exact force must make vanish) is 1.85 / 0.86 / 0.41 /
0.099 / 0.025 eV/Angstrom at ``h`` = 0.30 / 0.25 / 0.20 / 0.16 / 0.13.

The standard cure in real-space codes is to remove the unrepresentable
components from the *radial functions*, once, **before** they are ever sampled
(SIESTA's ``FilterCutoff``, Anglada & Soler, PRB 73, 115122 (2006); the
mask-function filter of Wang, and of Tafipolsky & Schmid, JCP 124, 174102
(2006), which GPAW uses for its projectors).  A band-limited function is
sampled *exactly* by the grid, so its integrals stop depending on where the
nucleus falls between nodes.

How
---
Each radial function belongs to a definite angular momentum, so the right
transform is the spherical Bessel (Hankel) pair

.. math::

    F_l(k) = \int_0^\infty R_l(r)\, j_l(kr)\, r^2\,dr , \qquad
    R_l(r) = \frac{2}{\pi}\int_0^\infty F_l(k)\, j_l(kr)\, k^2\,dk ,

whose closure relation gives a Parseval identity,
:math:`\int R^2 r^2 dr = \tfrac{2}{\pi}\int F^2 k^2 dk`, so "the weight of the
function above :math:`k_c`" is a well-defined number and is what
:func:`residual_weight` reports.  Filtering multiplies :math:`F_l` by a smooth
window that is 1 below :math:`(1-\texttt{rolloff})k_c` and rolls to 0 at
:math:`k_c`: a hard step would ring with a :math:`\sin(k_c r)/r` tail that
reaches far past the orbital.

Two properties of doing this **per angular-momentum channel** matter here:

* the reconstruction automatically behaves as :math:`r^l` at the origin,
  because :math:`j_l(kr)\sim (kr)^l/(2l+1)!!`.  That is not a detail -- a
  radial function that does not vanish as :math:`r^l` at its own nucleus broke
  the forces badly once already (the polarization-shell bug recorded in
  ``CLAUDE.md``), so it is asserted by the test suite;
* the filter commutes with the angular part, so a filtered table drops into
  :class:`~mandacaru.basis.multizeta.TabulatedOrbital` unchanged and everything
  downstream -- grid sampling, the PAW atom-centered projection quadrature, the
  displaced sampling the forces are built from -- sees one consistent function.

Confinement: what we actually do, and why
-----------------------------------------
A band-limited function cannot also have compact support, so a filter and a
confinement radius are in conflict and something has to give.  Three schemes
are implemented (:data:`FILTER_METHODS`) and the default was chosen by
measuring all three on the functions Mandacaru really filters:

``"plain"``
    Filter and keep the tail.  Exactly band-limited by construction, but the
    ringing spreads every function over the whole 30-Bohr table -- a compact
    split zeta stops being compact, which is the one thing it exists for.
``"switch"`` (**the default**)
    Filter, then multiply by a smooth window that is 1 out to the function's
    own support radius and 0 at :data:`SWITCH_MARGIN` times it.
``"mask"``
    Divide by a window that starts decaying *inside* the support, filter,
    multiply back -- the classic mask trick, which pre-compensates the
    multiplication so the result stays accurate near a hard confinement edge.

**Measured, at** :math:`k_c = 7.5\,a_0^{-1}` (765 eV), as the residual weight
above :math:`k_c` before -> after, and the range the result occupies
(smaller residual is better)::

    function              before     plain    switch      mask   range(switch)
    PAW H 1s   zeta 1    1.2e-06   4.9e-07   4.9e-07   5.3e-05     17.3 Bohr
    PAW H 1s   zeta 3    1.0e-04   3.7e-06   1.6e-05   2.0e-05      2.2
    H polarization (p)   3.4e-07   ~1e-12    ~1e-12    4.5e-08     21.8
    PAW O 2s   zeta 1    1.3e-05   7.4e-08   7.4e-08   6.5e-07      9.5
    PAW O 2s   zeta 3    9.3e-03   7.5e-07   2.8e-04   3.2e-04      1.7
    PAW O 2p   zeta 1    8.4e-05   ~2e-12    2.1e-11   9.8e-07     13.7
    PAW O 2p   zeta 3    4.7e-03   ~1e-11    2.8e-04   3.6e-04      1.9
    O polarization (d)   8.4e-05   ~1e-13    4.0e-12   1.1e-04     17.5

``"mask"`` loses on **every** function, and on the long-ranged ones it is
worse than doing nothing (H 1s 1.2e-06 -> 5.3e-05; the O polarization shell
8.4e-05 -> 1.1e-04).  That is exactly the failure the method's own caveat
predicts: these orbitals are bound states whose tail is still decaying where
the mask starts, so dividing by a window heading to zero amplifies the tail
and the multiplication afterwards cannot undo it.  The mask trick is for a
function **chopped** at a confinement radius with a finite value there; none
of ours is.  ``"plain"`` wins on residual but spreads every function over the
whole 30-Bohr table.  So the default is ``"switch"``: identical to ``"plain"``
on the first zetas (their tails are already 1e-7 of the peak where it acts),
and on the compact split zetas it trades a residual of a few times 1e-4 for
keeping the function short-ranged -- which is the whole point of a split
zeta.  Anglada-Soler's variational "optimal confined functions" were not
implemented: they re-solve a constrained minimization per orbital and buy
accuracy in the strictly *confined* limit, which is not where these tables
live.

What it costs, and who gets it by default
-----------------------------------------
Filtering removes variational freedom, so the *reported* energy goes **up**:
water / PAW-SZ by 832 meV at ``h`` = 0.25 and 422 meV at ``h`` = 0.20.  That
number looks alarming and mostly is not a loss.  Holding the cutoff **fixed**
(602 eV) and refining the grid, the filtered-to-unfiltered gap shrinks with
``h`` -- 530 / 422 / 294 / 237 meV at ``h`` = 0.25 / 0.20 / 0.16 / 0.13, close
to linear in ``h`` -- so most of it is the *unfiltered* basis's own grid
error, not a smaller function space.  The finite-difference Laplacian
under-estimates the kinetic energy of a function the grid does not resolve, so
an unfiltered orbital is reported too low; filtering removes exactly the
components that were being mis-integrated.  On H2, where both bases are
resolved by ``h`` = 0.12, the gap is 2 meV and has the *opposite* sign (the
filtered function is slightly better).

So the default is per family (:attr:`~mandacaru.pseudopotentials.families.FamilySpec.default_options`):
**on for PAW and UPAW**, whose smooth partial waves are built band-limited
(:func:`~mandacaru.pseudopotentials.oncv.optimize_pseudo_waves` minimizes the
kinetic energy beyond ``q_cut``) so the filter has little to take, and **off
for NCPP and ONCVPSP**, whose orbitals are not optimized that way.
``basis={"name": "PAW", "filter": False}`` restores the unfiltered basis
exactly.  The measured prices and benefits are in
``docs/source/guide/pseudopotentials.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
from scipy.special import spherical_jn

from ..units import EV_TO_HARTREE

#: ``filter=True`` / ``"auto"`` puts the cutoff at this fraction of the grid's
#: Nyquist wave-vector :math:`\pi/h` -- i.e. **exactly at** it.  The principle
#: is the whole point of the filter: remove what the grid cannot represent and
#: keep everything it can.  It is also what the measurement says.  Water /
#: PAW-SZ, peak-to-peak energy over one grid period under a rigid shift along
#: (1,1,1), against the variational cost (the rise in :math:`E_\text{RHF}`):
#:
#: ::
#:
#:     k_c/k_N   h = 0.25 Angstrom       h = 0.20 Angstrom
#:               ripple(meV) cost(meV)   ripple(meV) cost(meV)
#:     off         268.7        0          132.8        0
#:     1.3         142.8      489           17.2      273
#:     1.2          94.4      584            7.2      326
#:     1.1          39.0      700          [ 4.6]     365
#:     1.0        [ 27.1]     832            6.0      422
#:     0.95         43.0     1020            6.6      477
#:     0.9          75.9     1477            6.7      541
#:     0.8         162.1     4240            6.8      679
#:
#: The ripple has a clear minimum at 1.0-1.1 and **rises on both sides**: above
#: it because the unrepresentable components are still there, below it because
#: the cutoff starts eating the band the grid *can* carry, which distorts the
#: orbital near the core where the local potential varies fastest (the
#: ``V_loc`` column of the same scan goes 21.0 -> 36.4 -> 70.5 -> 160.1 meV
#: from 1.0 down to 0.8 while the kinetic and electron-repulsion ripples stay
#: at their floor).  Every fraction below 1.0 is therefore worse on *both*
#: axes -- more ripple and more cost -- so the choice is only between 1.0 and
#: something above it, and 1.0 is the one that leaves nothing aliased.
FILTER_NYQUIST_FRACTION = 1.0

#: Width of the :math:`k`-space roll-off, as a fraction of :math:`k_c`: the
#: window is 1 below :math:`(1-\text{rolloff})k_c` and reaches 0 at
#: :math:`k_c`.  A hard step (rolloff 0) rings with a ``sin(k_c r)/r`` tail
#: that only decays as ``1/r``; a raised cosine over a quarter of the band
#: decays as ``1/r^3`` and is invisible past the orbital.  Measured: with the
#: switch-off in place a hard step leaves 5x more residual weight above
#: :math:`k_c` on the compact split zetas (1.4e-3 against 2.8e-4 for O 2s
#: zeta 3) -- the ringing the switch chops off comes straight back as
#: high-:math:`k` content.
ROLLOFF_FRACTION = 0.25

#: Points per oscillation of :math:`j_l(k r_\text{max})` in the :math:`k`
#: quadrature.  The transform is exact enough at 4 (verified: the round trip
#: does not move between 4 and 16); 8 is margin, and the cost is one
#: ``(n_k, n_r)`` matrix per function -- milliseconds.
TRANSFORM_OVERSAMPLE = 8

#: A function's **support radius** is where its remaining tail norm drops
#: below this fraction of the total -- where the switch-off (or the mask) is
#: placed relative to.
SUPPORT_TAIL_NORM = 1e-6

#: The switch-off (or mask) reaches zero at this multiple of the support
#: radius, clipped to the end of the table.  The filtered function needs room
#: for its ringing tail before it is switched off: measured on the compact
#: split zetas, 1.25 / 1.5 / 2.0 / 3.0 leave 5.6e-4 / 2.8e-4 / 9.9e-5 /
#: 1.9e-5 of the norm above :math:`k_c` (O 2s zeta 3) at 1.44 / 1.73 / 2.30 /
#: 3.46 Bohr of range.  1.5 is the knee: it halves the residual for 20 % more
#: range, and a split zeta that stretches to three times its support is no
#: longer the short-ranged function the split-valence construction built.
SWITCH_MARGIN = 1.5

#: The ``"mask"`` method's window is exactly 1 below this fraction of its
#: outer radius -- i.e. it starts decaying *inside* the support, which is what
#: makes it a mask rather than a switch-off (and, here, what makes it lose).
MASK_INNER_FRACTION = 0.6

#: Available real-space treatments; see the module docstring.
FILTER_METHODS = ("plain", "switch", "mask")

#: The default, chosen on the measurements tabulated in the module docstring.
DEFAULT_FILTER_METHOD = "switch"

#: Warn when the filter throws away more than this fraction of a function's
#: norm -- at that point the cutoff is not smoothing the basis, it is
#: replacing it.
FILTER_WEIGHT_WARN = 0.05


# --------------------------------------------------------------------------- #
# The option: what ``basis={"name": "PAW", "filter": ...}`` may say.
# --------------------------------------------------------------------------- #

def validate_filter(spec):
    """Normalize a ``filter`` basis option; raise ``ValueError`` on anything else.

    Accepted spellings, and only these:

    ``None`` / ``False``
        off -- the default, and byte-identical to a basis built without the
        option at all;
    ``True`` / ``"auto"``
        tie the cutoff to the grid, :math:`k_c =`
        :data:`FILTER_NYQUIST_FRACTION` :math:`\\times\\,\\pi/h`;
    a positive number
        a **kinetic-energy cutoff in eV** (user-facing energies are eV
        everywhere in Mandacaru), i.e. :math:`k_c = \\sqrt{2E}` in atomic units.

    Returns ``False``, ``True`` or a ``float``.
    """
    if spec is None or spec is False:
        return False
    if spec is True:
        return True
    if isinstance(spec, str):
        if spec.strip().lower() == "auto":
            return True
        raise ValueError(
            f"unknown basis filter {spec!r}; use True or 'auto' to tie the "
            "cutoff to the grid spacing, a positive number for an explicit "
            "kinetic-energy cutoff in eV, or False to switch filtering off")
    if isinstance(spec, (int, float, np.integer, np.floating)):
        value = float(spec)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"a basis filter cutoff must be a positive, finite energy in "
                f"eV, got {spec!r}")
        return value
    raise ValueError(
        f"unknown basis filter {spec!r} (type {type(spec).__name__}); use "
        "True / 'auto', a positive kinetic-energy cutoff in eV, or False")


def filter_cutoff(spec, spacing_bohr=None) -> float | None:
    """The cutoff wave-vector :math:`k_c` in Bohr\\ :sup:`-1`, or ``None`` when off.

    ``spacing_bohr`` is the **coarsest** grid spacing (the worst axis of an
    anisotropic grid is what limits the representable band), needed only for
    the ``True`` / ``"auto"`` form.
    """
    spec = validate_filter(spec)
    if spec is False:
        return None
    if spec is True:
        if spacing_bohr is None:
            raise ValueError(
                "filter='auto' needs the grid spacing; pass spacing_bohr (or "
                "give an explicit cutoff in eV)")
        spacing = float(spacing_bohr)
        if spacing <= 0.0:
            raise ValueError(f"the grid spacing must be positive, got {spacing!r}")
        return FILTER_NYQUIST_FRACTION * np.pi / spacing
    return float(np.sqrt(2.0 * float(spec) * EV_TO_HARTREE))


def filter_label(spec) -> str:
    """One short phrase describing the option, for the dry run and the log."""
    spec = validate_filter(spec)
    if spec is False:
        return "unfiltered"
    if spec is True:
        return (f"filtered (auto: {FILTER_NYQUIST_FRACTION:g} x Nyquist)")
    return f"filtered ({spec:g} eV)"


def cutoff_energy_ev(k_c: float) -> float:
    """The kinetic-energy cutoff in eV a wave-vector corresponds to."""
    return float(0.5 * k_c * k_c / EV_TO_HARTREE)


# --------------------------------------------------------------------------- #
# The transform pair.
# --------------------------------------------------------------------------- #

def bessel_matrix(l: int, k, r) -> np.ndarray:
    """The ``(n_k, n_r)`` kernel :math:`j_l(k_i r_j)`, shared by both directions."""
    return spherical_jn(int(l), np.outer(np.asarray(k, dtype=float),
                                         np.asarray(r, dtype=float)))


def _augment_origin(r, values, l: int):
    """Prepend ``r = 0`` using the :math:`r^l` continuation.

    The radial tables start at ``r_0 > 0`` (0.0025-0.005 Bohr), so a plain
    quadrature silently drops :math:`[0, r_0]`.  The omission is tiny
    (:math:`\\sim r_0^3/3` of the norm) but it is free to avoid, and it keeps
    the round trip clean at the origin, which is exactly where the
    :math:`r^l` behavior is checked.
    """
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    if r[0] <= 0.0:
        return r, values
    edge = values[0] if int(l) == 0 else 0.0
    return np.concatenate(([0.0], r)), np.concatenate(([edge], values))


def spherical_bessel_transform(r, values, l: int, k, matrix=None) -> np.ndarray:
    r""":math:`F_l(k) = \int R_l(r) j_l(kr) r^2 dr` on the tabulated grid."""
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    kernel = bessel_matrix(l, k, r) if matrix is None else matrix
    return np.trapezoid(kernel * (values * r * r), r, axis=1)


def inverse_spherical_bessel_transform(k, transform, l: int, r,
                                       matrix=None) -> np.ndarray:
    r""":math:`R_l(r) = \frac{2}{\pi}\int F_l(k) j_l(kr) k^2 dk`."""
    k = np.asarray(k, dtype=float)
    transform = np.asarray(transform, dtype=float)
    kernel = bessel_matrix(l, k, r) if matrix is None else matrix
    weight = (transform * k * k)[:, None]
    return (2.0 / np.pi) * np.trapezoid(kernel * weight, k, axis=0)


def transform_grid(r_max: float, k_max: float,
                   oversample: int = TRANSFORM_OVERSAMPLE) -> np.ndarray:
    """A :math:`k` grid dense enough to resolve :math:`j_l(k\\,r_\\text{max})`."""
    step = np.pi / (float(oversample) * float(r_max))
    count = int(np.ceil(float(k_max) / step)) + 1
    return np.linspace(0.0, float(k_max), max(count, 8))


def cutoff_window(k, k_c: float, rolloff: float = ROLLOFF_FRACTION
                  ) -> np.ndarray:
    """A raised-cosine window: 1 below ``(1-rolloff) k_c``, 0 at ``k_c``."""
    k = np.asarray(k, dtype=float)
    k_c = float(k_c)
    inner = (1.0 - float(rolloff)) * k_c
    window = np.ones_like(k)
    if k_c > inner:
        s = np.clip((k - inner) / (k_c - inner), 0.0, 1.0)
        window = 0.5 * (1.0 + np.cos(np.pi * s))
    return np.where(k >= k_c, 0.0, window)


def radial_norm(r, values) -> float:
    r""":math:`\int R^2 r^2 dr` -- the norm the Parseval identity balances."""
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    return float(np.trapezoid(values * values * r * r, r))


def band_weight(r, values, l: int, k_c: float,
                oversample: int = TRANSFORM_OVERSAMPLE) -> float:
    """Fraction of :math:`\\int R^2r^2dr` carried by wave-vectors below ``k_c``.

    Computed from *below*, never by integrating a truncated high-:math:`k`
    tail: the quadrature of :math:`j_l(kr)` on the tabulated radial grid
    degrades once ``k`` approaches :math:`\\pi/\\Delta r` (only ~6 points per
    oscillation at ``k = 50`` on the hydrogen table), and integrating out
    there produces spurious weight rather than measuring it.
    """
    r_aug, values_aug = _augment_origin(r, values, l)
    total = radial_norm(r_aug, values_aug)
    if total <= 0.0:
        return 1.0
    k = transform_grid(r_aug[-1], k_c, oversample)
    F = spherical_bessel_transform(r_aug, values_aug, l, k)
    inside = (2.0 / np.pi) * float(np.trapezoid(F * F * k * k, k))
    return inside / total


def residual_weight(r, values, l: int, k_c: float,
                    oversample: int = TRANSFORM_OVERSAMPLE) -> float:
    """``1 - band_weight`` -- the fraction of the norm the grid cannot represent."""
    return 1.0 - band_weight(r, values, l, k_c, oversample)


# --------------------------------------------------------------------------- #
# Confinement: support radius and mask.
# --------------------------------------------------------------------------- #

def support_radius(r, values, tail: float = SUPPORT_TAIL_NORM) -> float:
    """Radius beyond which the function carries less than ``tail`` of its norm.

    This is the range the *unfiltered* function occupies; the mask is placed
    beyond it so the division never meets a tail that has not decayed.
    """
    from scipy.integrate import cumulative_trapezoid

    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    density = values * values * r * r
    total = float(np.trapezoid(density, r))
    if total <= 0.0:
        return float(r[-1])
    outside = (total - cumulative_trapezoid(density, r, initial=0.0)) / total
    index = int(np.argmax(outside <= float(tail)))
    if outside[index] > float(tail):        # never drops: use the whole table
        return float(r[-1])
    return float(r[max(index, 1)])


def mask_function(r, r_inner: float, r_outer: float) -> np.ndarray:
    """A raised-cosine mask: 1 below ``r_inner``, 0 at and beyond ``r_outer``.

    Value *and* slope vanish at ``r_outer`` and the slope vanishes at
    ``r_inner``, so multiplying by it adds as little high-:math:`k` content as
    a compactly supported window can.
    """
    r = np.asarray(r, dtype=float)
    r_inner, r_outer = float(r_inner), float(r_outer)
    if not r_outer > r_inner:
        return np.where(r < r_outer, 1.0, 0.0)
    s = np.clip((r - r_inner) / (r_outer - r_inner), 0.0, 1.0)
    return np.where(r >= r_outer, 0.0, 0.5 * (1.0 + np.cos(np.pi * s)))


# --------------------------------------------------------------------------- #
# The filter itself.
# --------------------------------------------------------------------------- #

def filter_radial(r, values, l: int, k_c: float, *,
                  method: str = DEFAULT_FILTER_METHOD,
                  rolloff: float = ROLLOFF_FRACTION,
                  oversample: int = TRANSFORM_OVERSAMPLE,
                  renormalize: bool = True,
                  switch_margin: float = SWITCH_MARGIN,
                  warn: bool = True):
    """Remove the content above ``k_c`` from one radial table.

    Returns ``(filtered_values, info)`` where ``info`` records the residual
    weight above ``k_c`` before and after, the support radius, the mask radius
    and the largest real-space change -- the numbers the guide quotes and the
    tests assert.

    The result is rescaled to the **original** norm rather than to 1: the
    tables do not all arrive normalized (the PAW hydrogen 1s carries 0.979),
    and forcing them to 1 would make the filter change the function even in
    the ``k_c -> infinity`` limit where it must be the identity.  The scale is
    in any case invisible downstream -- the basis is Loewdin-orthogonalized
    before the Hamiltonian is built.
    """
    if method not in FILTER_METHODS:
        raise ValueError(f"unknown filter method {method!r}; "
                         f"use one of {list(FILTER_METHODS)}")
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    l = int(l)
    k_c = float(k_c)
    if k_c <= 0.0:
        raise ValueError(f"the filter cutoff must be positive, got {k_c!r}")

    r_aug, values_aug = _augment_origin(r, values, l)
    original_norm = radial_norm(r_aug, values_aug)
    before = 1.0 - band_weight(r, values, l, k_c, oversample)

    r_support = support_radius(r_aug, values_aug)
    r_outer = min(float(switch_margin) * r_support, float(r_aug[-1]))
    r_inner = (MASK_INNER_FRACTION * r_outer if method == "mask"
               else min(r_support, r_outer))
    use_window = method != "plain" and r_outer < float(r_aug[-1])

    work = values_aug
    mask = None
    if use_window:
        mask = mask_function(r_aug, r_inner, r_outer)
        if method == "mask":
            # Pre-compensate the multiplication that follows.  Safe because
            # the mask reaches zero beyond the support (MASK_MARGIN), so the
            # divisor is never small where the function is not.
            inside = r_aug < r_outer
            work = np.zeros_like(values_aug)
            np.divide(values_aug, mask, out=work, where=inside & (mask > 0.0))

    k = transform_grid(r_aug[-1], k_c, oversample)
    matrix = bessel_matrix(l, k, r_aug)
    transform = spherical_bessel_transform(r_aug, work, l, k, matrix)
    transform = transform * cutoff_window(k, k_c, rolloff)
    filtered = inverse_spherical_bessel_transform(k, transform, l, r_aug,
                                                  matrix)
    if use_window:
        filtered = filtered * mask

    new_norm = radial_norm(r_aug, filtered)
    kept = new_norm / original_norm if original_norm > 0.0 else 1.0
    if renormalize and new_norm > 0.0 and original_norm > 0.0:
        filtered = filtered * np.sqrt(original_norm / new_norm)

    if r_aug.size != r.size:                 # drop the prepended r = 0 node
        filtered = filtered[1:]
    after = 1.0 - band_weight(r, filtered, l, k_c, oversample)

    scale = float(np.max(np.abs(values))) or 1.0
    info = {
        "l": l, "cutoff": k_c, "method": method,
        "residual_before": float(before), "residual_after": float(after),
        "support_radius": float(r_support), "mask_radius": float(r_outer),
        "masked": bool(use_window),
        "norm_kept": float(kept),
        "max_change": float(np.max(np.abs(filtered - values)) / scale),
    }
    if warn and 1.0 - kept > FILTER_WEIGHT_WARN:
        warnings.warn(
            f"the basis filter at k_c = {k_c:.3g} Bohr^-1 "
            f"({cutoff_energy_ev(k_c):.0f} eV) removed "
            f"{100 * (1.0 - kept):.1f}% of the norm of an l = {l} radial "
            "function: at that cutoff the filter is replacing the basis, not "
            "smoothing it.  Raise the cutoff (or the grid resolution).",
            RuntimeWarning, stacklevel=2)
    return filtered, info


def filter_table(table, k_c: float, **kwargs):
    """Filter a :class:`~mandacaru.basis.multizeta.RadialTable` in place of itself.

    Returns a **new** table (the input is a shared, cached object -- the
    pseudopotential datasets are memoized per element, so mutating one would
    leak the filter into every later calculation in the process).
    """
    values, info = filter_radial(table.r, table.values, table.l, k_c, **kwargs)
    return replace(table, values=values), info
