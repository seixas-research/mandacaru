# -*- coding: utf-8 -*-
# file: experimental/pseudopotentials/oncv.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Optimized norm-conserving Vanderbilt pseudopotentials (ONCVPSP) -- experimental.

The family ``"oncvpsp"`` (alias ``"oncv"``) implements D. R. Hamann's
construction, Phys. Rev. B **88**, 085117 (2013): **two projectors per
angular-momentum channel** built from two reference energies, a Bessel-function
pseudo wave function whose residual kinetic energy beyond a wave-vector cutoff
is minimized, and a local potential that is *not* one of the channels.  Written
from scratch on top of the same LDA radial atom the Troullier-Martins family
uses (:mod:`carcara.basis.atomic_solver`); nothing is read from tables.

Construction
------------
1. **All-electron reference.**  The self-consistent spherical LDA atom gives
   the screened potential :math:`V^{AE}(r)`.  For every valence :math:`l` two
   partial waves are integrated in it with the Numerov method on the atom's
   uniform radial grid (:func:`generation_points`: 1500 points per unit of
   :math:`Z`, at least 6000, out to 30 Bohr; the regular solution is seeded
   at the origin with a six-term power series, :func:`_origin_seed`):
   :math:`\varphi_1` is the valence bound state (found by outward/inward
   shooting, :func:`bound_state`, so it satisfies the radial equation to
   fourth order), :math:`\varphi_2` is a second bound state of that
   :math:`l` when the atom has one, else the **scattering state** at
   :math:`\varepsilon_2 = \varepsilon_1 + \Delta` (:data:`DEFAULT_ENERGY_OFFSET`
   = 1 Ha), integrated outward and normalized to one inside :math:`r_c`.
   The two reference energies bracket the valence window; a larger
   :math:`\Delta` lowers the norm of the coupling matrix but degrades the
   logarithmic derivatives between the references (O: midpoint error
   8e-7 at 1 Ha, 3e-4 at 2 Ha), while the molecular energies of H2/LiH move
   by only 4e-5 Ha.

2. **Pseudo partial waves** (Sec. II of Hamann).  Inside :math:`r_c`,

   .. math::

       \tilde\varphi_i(r) = \sum_{n=1}^{N} c_{in}\, j_l(q_n r),

   the RRKJ form (Rappe, Rabe, Kaxiras and Joannopoulos, PRB **41**, 1227
   (1990)) with the wave vectors :math:`q_n` taken as the interleaved zeros
   of :math:`j_l` and :math:`j_l'` at :math:`r_c` (:func:`bessel_wavevectors`;
   :data:`DEFAULT_N_BESSEL` = 8 functions, Hamann's ``nbas``).  The
   constraints are continuity of the value and of the first three derivatives
   at :math:`r_c` (Hamann's ``ncon = 4``; the targets come from the
   all-electron radial equation, :func:`matching_targets`) and the
   **generalized norm conservation**

   .. math::

       \langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}
       = \langle\varphi_i|\varphi_j\rangle_{r<r_c}
       \quad\text{for all } i, j ,

   the Vanderbilt condition that makes the two-projector potential
   transferable (it guarantees the correct energy derivative of the
   logarithmic derivative at both reference energies).  Within the remaining
   freedom the **residual kinetic energy**

   .. math::

       E^{r}_i(q_c) = \tfrac12 \int_{q_c}^{\infty} q^4\,|\tilde\varphi_i(q)|^2\,dq

   is minimized -- Hamann's optimization -- where :math:`\tilde\varphi_i(q)`
   is the spherical Bessel transform of the whole pseudo wave (Bessel
   expansion inside, all-electron tail outside).  The objective is a quadratic
   form in the coefficients, the matching conditions are linear and the norm
   condition of :math:`\tilde\varphi_i` itself is the only quadratic
   constraint, so the minimum is found *exactly*: the linear constraints are
   eliminated through their null space and the Lagrange multiplier of the
   norm is a one-dimensional root (:func:`constrained_minimum`).

3. **Local potential.**  A smooth even polynomial continuation of the screened
   all-electron potential inside :math:`r_{cl}` (value and four derivatives
   matched; Hamann's polynomial ``lpopt``), unscreened with the Hartree and
   LDA exchange-correlation potentials of the pseudo valence density.  It is
   never one of the channels, so **every valence channel -- the s channel of H
   and Li included -- carries two projectors**.

4. **Projectors and coupling matrix.**  With :math:`T_l j_l(q r) = \tfrac12
   q^2 j_l(q r)` the projectors are analytic,

   .. math::

       \chi_i(r) = (\varepsilon_i - T_l - V^{scr}_{loc})\,\tilde\varphi_i
                 = \sum_n c_{in}\,(\varepsilon_i - \tfrac12 q_n^2
                   - V^{scr}_{loc}(r))\, j_l(q_n r), \qquad r < r_c ,

   and vanish beyond :math:`r_c`.  :math:`B_{ij} = \langle\tilde\varphi_i|
   \chi_j\rangle` is symmetric by generalized norm conservation (checked, the
   asymmetry must stay below :data:`B_ASYMMETRY_TOLERANCE`), and the
   nonlocal potential is

   .. math::

       V_{NL} = \sum_{ij} |\chi_i\rangle D_{ij} \langle\chi_j| ,
       \qquad D = B^{-1} ,

   one :math:`2\times2` block per ``(atom, l, m)`` -- exactly the
   ``nonlocal_coupling`` blocks of
   :meth:`carcara.core.hamiltonian.MolecularIntegrals.kb_nonlocal`.  The raw
   Vanderbilt form is what is stored and used (its off-diagonal coupling is
   real); :func:`diagonalized_projectors` gives Hamann's equivalent
   orthogonalized pair with a diagonal coupling.

Not implemented: the nonlinear core correction, scalar-relativistic or
spin-orbit terms, projectors for angular momenta above the valence (those
channels see the local potential alone), and GGA reference atoms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import spherical_jn

from ...basis.atomic_solver import (AtomicResult, hartree_potential, lda_xc,
                                    solve_atom)
from .generation import (Channel, PseudoPotential, _local_derivatives,
                         _valence_configuration)

#: Registry name of the family and its alias.
FAMILY = "oncvpsp"
FAMILY_ALIASES = ("oncv",)
#: Subdirectory of the pseudopotential library holding the ONCVPSP files.
LIBRARY_SUBDIR = "oncvpsp"

#: Spherical Bessel functions per pseudo partial wave (Hamann's ``nbas``).
DEFAULT_N_BESSEL = 8
#: Wave-vector cutoff of the residual kinetic energy (Bohr^-1, Hamann's ``qcut``).
DEFAULT_Q_CUT = 5.0
#: Second reference energy above the bound state when the channel has no second
#: bound state (Hartree, Hamann's ``debl``).
DEFAULT_ENERGY_OFFSET = 1.0
#: Default cutoff radius as a multiple of the outermost maximum of ``r R(r)``.
DEFAULT_RC_FACTOR = 1.3
#: Local-potential radius as a multiple of the smallest channel cutoff.
DEFAULT_LOCAL_FACTOR = 0.9
#: Per-element cutoff radii (Bohr) overriding the factor heuristic -- close to
#: Hamann's choices for the first row.  Lithium is *not* pushed further out:
#: at 3.0 Bohr the LiH energy jumps by 0.15 Ha and at 3.3 Bohr the s channel
#: grows a ghost state at -0.83 Ha, because the polynomial local potential over
#: so wide a core no longer resembles the atom (checked 2026-09-14).
DEFAULT_CUTOFFS = {
    "H": {0: 1.30},
    "Li": {0: 2.60},
    "C": {0: 1.50, 1: 1.50},
    "N": {0: 1.45, 1: 1.45},
    "O": {0: 1.45, 1: 1.45},
    "F": {0: 1.40, 1: 1.40},
}
#: Largest tolerated asymmetry of the Vanderbilt matrix ``B`` (Hartree).
B_ASYMMETRY_TOLERANCE = 1e-5   # Ha; heavy atoms (U) reach ~2e-6 from quadrature alone
#: Upper wave vector and spacing of the Fourier grid of the residual energy.
Q_MAX, Q_STEP = 60.0, 0.1
#: Points of the fine quadrature grid inside ``r_c``.
INNER_POINTS = 801
#: Finest spacing (Bohr) of the tail quadrature of the residual energy.
TAIL_SPACING = 0.00125


# --------------------------------------------------------------------------- #
# Spherical Bessel functions and their derivatives.
# --------------------------------------------------------------------------- #

def bessel_derivatives(l: int, x):
    """``j_l(x)`` and its first three derivatives with respect to ``x``.

    The second and third derivatives follow from the spherical Bessel
    equation :math:`x^2 j'' + 2x j' + (x^2 - l(l+1)) j = 0`, so no numerical
    differentiation is involved.
    """
    x = np.asarray(x, dtype=float)
    j = spherical_jn(l, x)
    dj = spherical_jn(l, x, derivative=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = l * (l + 1)
        d2j = -2.0 * dj / x - (1.0 - ll / x ** 2) * j
        d3j = (2.0 / x ** 2) * dj - (2.0 / x) * d2j \
            - (2.0 * ll / x ** 3) * j - (1.0 - ll / x ** 2) * dj
    return j, dj, d2j, d3j


def bessel_wavevectors(l: int, r_cut: float, n: int = DEFAULT_N_BESSEL,
                       x_max: float = 400.0) -> np.ndarray:
    r"""Wave vectors of the Bessel basis: the interleaved zeros of
    :math:`j_l(x)` and :math:`j_l'(x)` at :math:`x = q r_c`, in ascending order.

    The zeros of a spherical Bessel function and of its derivative interlace,
    so the resulting set is evenly graded in :math:`q` and the basis functions
    alternate between contributing the value and the slope at :math:`r_c` --
    the matrix of matching conditions (value and any number of derivatives)
    has full rank.  The RRKJ choice (all functions sharing the all-electron
    logarithmic derivative at :math:`r_c`) is *not* usable beyond the second
    derivative: with :math:`q j_l'(q r_c) = L j_l(q r_c)` the free-particle
    Bessel equation turns every derivative row into :math:`j_l(q_n r_c)`
    times a polynomial in :math:`q_n^2`, and the third-derivative row becomes
    a combination of the value and second-derivative rows.
    """
    def zeros(function):
        x = np.linspace(0.1, x_max, 80001)
        values = function(x)
        found = []
        for i in np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]:
            found.append(brentq(function, x[i], x[i + 1], xtol=1e-14,
                                rtol=1e-14))
            if len(found) >= n:
                break
        return found

    roots = sorted(zeros(lambda x: spherical_jn(l, x))
                   + zeros(lambda x: spherical_jn(l, x, derivative=True)))
    if len(roots) < n:
        raise RuntimeError(f"found only {len(roots)} Bessel wave vectors for "
                           f"l={l}, rc={r_cut:.3f}")
    return np.array(roots[:n]) / float(r_cut)


# --------------------------------------------------------------------------- #
# Numerov integration on the uniform atomic grid.
# --------------------------------------------------------------------------- #

def _with_origin(r: np.ndarray):
    """The atomic grid (which starts at one step) with ``r = 0`` prepended."""
    return np.concatenate([[0.0], r])


def numerov_outward(r0: np.ndarray, f: np.ndarray, s: np.ndarray,
                    u_start: tuple, start: int = 1) -> np.ndarray:
    r"""Integrate :math:`u'' = f(r)\,u + s(r)` outward on a uniform grid.

    ``u_start = (u[start], u[start+1])`` seeds the recursion; nothing at
    indices below ``start`` is used (so a ``-Z/r`` singularity at ``r0[0]``
    is harmless as long as ``start >= 1``).
    """
    h2 = (r0[1] - r0[0]) ** 2
    u = np.zeros_like(r0)
    u[start], u[start + 1] = u_start
    a = 1.0 - h2 * f / 12.0
    b = 2.0 * (1.0 + 5.0 * h2 * f / 12.0)
    c = h2 / 12.0
    for i in range(start + 1, r0.size - 1):
        u[i + 1] = (b[i] * u[i] - a[i - 1] * u[i - 1]
                    + c * (s[i + 1] + 10.0 * s[i] + s[i - 1])) / a[i + 1]
    return u


def _numerov_inward(r0: np.ndarray, f: np.ndarray, stop: int,
                    kappa: float) -> np.ndarray:
    """Homogeneous inward integration from a decaying start down to ``stop``."""
    h = r0[1] - r0[0]
    h2 = h * h
    u = np.zeros_like(r0)
    n = r0.size
    u[n - 1] = np.exp(-kappa * r0[n - 1])
    u[n - 2] = np.exp(-kappa * r0[n - 2])
    a = 1.0 - h2 * f / 12.0
    b = 2.0 * (1.0 + 5.0 * h2 * f / 12.0)
    for i in range(n - 2, stop, -1):
        u[i - 1] = (b[i] * u[i] - a[i + 1] * u[i + 1]) / a[i - 1]
    return u


def _radial_f(r0: np.ndarray, potential0: np.ndarray, l: int,
              energy: float) -> np.ndarray:
    """``f = 2[V + l(l+1)/2r^2 - E]`` with a finite (unused) value at r = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * (potential0 + l * (l + 1) / (2.0 * r0 ** 2) - energy)
    f[0] = 0.0
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def _origin_seed(r0: np.ndarray, l: int, z_eff: float, potential0=None,
                 energy: float = 0.0, order: int = 6):
    r"""Power-series start of the regular solution at the first two points.

    For :math:`V = -Z/r + c_0` near the origin, :math:`u = r^{l+1}\sum_k a_k
    r^k` with :math:`a_k\,k(k+2l+1) = -2Z a_{k-1} + 2c_0 a_{k-2}`
    (:math:`c_0 = V(r_1) + Z/r_1 - E`).  The two-term start
    :math:`1 - Zr/(l+1)` is not enough: its :math:`O((Zr)^2)` error admixes
    the irregular solution and shifts the oxygen 2s eigenvalue by 2e-2 Ha on
    a 0.005 Bohr grid; six terms leave ~1e-8.
    """
    c0 = 0.0 if potential0 is None else float(potential0[1] + z_eff / r0[1]
                                              - energy)
    a = [1.0, -z_eff / (l + 1)]
    for k in range(2, order + 1):
        a.append((-2.0 * z_eff * a[k - 1] + 2.0 * c0 * a[k - 2])
                 / (k * (k + 2 * l + 1)))
    def series(x):
        return x ** (l + 1) * sum(ak * x ** k for k, ak in enumerate(a))
    return (series(r0[1]), series(r0[2]))


def _derivative(u: np.ndarray, h: float, index: int) -> float:
    """Fourth-order centered first derivative at one grid index."""
    return (u[index - 2] - 8 * u[index - 1] + 8 * u[index + 1]
            - u[index + 2]) / (12.0 * h)


def scattering_wave(r: np.ndarray, potential: np.ndarray, l: int,
                    energy: float, z_eff: float) -> np.ndarray:
    """Outward Numerov solution ``u(r)`` at ``energy`` (arbitrary scale)."""
    r0 = _with_origin(r)
    v0 = np.concatenate([[0.0], potential])
    f = _radial_f(r0, v0, l, energy)
    u = numerov_outward(r0, f, np.zeros_like(r0),
                        _origin_seed(r0, l, z_eff, v0, energy), start=1)
    return u[1:]


def bound_state(r: np.ndarray, potential: np.ndarray, l: int,
                energy_guess: float, z_eff: float, window: float = 5e-3):
    """Numerov bound state ``(u, energy)`` near ``energy_guess``.

    Outward and inward integrations are matched at the outermost classical
    turning point; the mismatch of logarithmic derivatives is brought to zero
    with Brent's method on the energy, starting from a bracket of
    ``+/- window`` around the guess (widened if necessary).  ``u`` is
    normalized to ``int u^2 dr = 1`` and positive near the origin.
    """
    r0 = _with_origin(r)
    v0 = np.concatenate([[0.0], potential])
    h = r0[1] - r0[0]

    def mismatch(energy):
        f = _radial_f(r0, v0, l, energy)
        # Outermost classical turning point of the effective potential.
        turning = np.nonzero(f[10:] < 0)[0]
        match = int(turning[-1]) + 10 if turning.size else r0.size // 2
        match = min(max(match, 20), r0.size - 20)
        out = numerov_outward(r0, f, np.zeros_like(r0),
                              _origin_seed(r0, l, z_eff, v0, energy), start=1)
        kappa = np.sqrt(max(-2.0 * energy, 1e-6))
        inn = _numerov_inward(r0, f, match - 3, kappa)
        scale = out[match] / inn[match]
        inn = inn * scale
        return ((_derivative(out, h, match) - _derivative(inn, h, match))
                / out[match]), out, inn, match

    lo, hi = energy_guess - window, energy_guess + window
    for _ in range(12):
        if mismatch(lo)[0] * mismatch(hi)[0] < 0:
            break
        lo, hi = lo - 2 * window, hi + 2 * window
    else:
        raise RuntimeError(f"no Numerov bound state near {energy_guess:.6f} "
                           f"Ha for l={l}")
    energy = brentq(lambda e: mismatch(e)[0], lo, hi, xtol=1e-13, rtol=1e-13)
    _m, out, inn, match = mismatch(energy)
    u = np.concatenate([out[:match], inn[match:]])
    u = u[1:]
    u = u / np.sqrt(np.trapezoid(u * u, r))
    if u[:20].sum() < 0:
        u = -u
    return u, float(energy)


def log_derivative_ae(r: np.ndarray, potential: np.ndarray, l: int,
                      energy: float, r_cut: float, z_eff: float) -> float:
    r"""All-electron logarithmic derivative :math:`R'/R` at ``r_cut``."""
    u = scattering_wave(r, potential, l, energy, z_eff)
    return _log_derivative_of_u(r, u, r_cut)


def _log_derivative_of_u(r, u, r_cut) -> float:
    """``R'/R = u'/u - 1/r`` at ``r_cut`` from a fourth-order stencil."""
    h = r[1] - r[0]
    index = int(np.argmin(np.abs(r - r_cut)))
    return float(_derivative(u, h, index) / u[index] - 1.0 / r[index])


# --------------------------------------------------------------------------- #
# Constrained minimization of the residual kinetic energy.
# --------------------------------------------------------------------------- #

def constrained_minimum(K: np.ndarray, k: np.ndarray, A: np.ndarray,
                        b: np.ndarray, G: np.ndarray, norm: float
                        ) -> np.ndarray:
    r"""Minimize :math:`c^TKc + 2k^Tc` subject to :math:`Ac = b` and
    :math:`c^TGc = \text{norm}`.

    The linear constraints are eliminated exactly, :math:`c = c_0 + Z y` with
    :math:`Z` an orthonormal basis of the null space of :math:`A`.  In that
    space the problem is a quadratic objective on an ellipsoid, whose global
    minimizer is the stationary point :math:`(K_z - \lambda G_z) y =
    \lambda g_z - k_z` with the multiplier below the smallest generalized
    eigenvalue of :math:`(K_z, G_z)` (More and Sorensen); the norm is then a
    monotonic function of :math:`\lambda` on that branch and the multiplier is
    a one-dimensional root.
    """
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    n = K.shape[0]
    c0, *_ = np.linalg.lstsq(A, b, rcond=None)
    _u, sigma, vt = np.linalg.svd(A)
    rank = int(np.sum(sigma > 1e-10 * sigma.max()))
    Z = vt[rank:].T                                   # (n, n - rank)
    if Z.shape[1] == 0:
        return c0
    Kz, Gz = Z.T @ K @ Z, Z.T @ G @ Z
    kz, gz = Z.T @ (K @ c0 + k), Z.T @ (G @ c0)

    def solution(lam):
        return np.linalg.solve(Kz - lam * Gz, lam * gz - kz)

    def residual(lam):
        c = c0 + Z @ solution(lam)
        return c @ G @ c - norm

    from scipy.linalg import eigh
    mu = eigh(Kz, Gz, eigvals_only=True)[0]
    hi = mu - 1e-9 * max(abs(mu), 1.0)
    # On this branch the norm grows monotonically towards +inf as lam -> mu.
    # Walk down until the norm falls below the target, then bracket.
    lo = hi - 1.0
    while residual(lo) > 0:
        lo = hi - 2.0 * (hi - lo)
        if hi - lo > 1e12:
            raise RuntimeError("the linear constraints already require more "
                               "norm than norm conservation allows")
    if residual(hi) < 0:
        # Extremely close to the pole: shrink towards mu until positive.
        for _ in range(60):
            hi = mu - 0.5 * (mu - hi)
            if residual(hi) > 0:
                break
    lam = brentq(residual, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    return c0 + Z @ solution(lam)


# --------------------------------------------------------------------------- #
# One channel: two pseudo partial waves, projectors, coupling.
# --------------------------------------------------------------------------- #

@dataclass
class ONCVChannel(Channel):
    """One ONCVPSP channel: the TM :class:`Channel` plus the second wave.

    ``pseudo_radial``/``eigenvalue``/``coefficients`` describe the first
    (bound) partial wave, which is also the first-zeta basis function.
    """

    reference_energies: list = field(default_factory=list)   # Hartree
    wavevectors: list = field(default_factory=list)          # per wave: q_n
    wave_coefficients: list = field(default_factory=list)    # per wave: c_n
    pseudo_waves: list = field(default_factory=list)         # per wave: R(r)
    projectors: list = field(default_factory=list)           # per wave: chi(r)
    coupling: np.ndarray = None                              # D (2x2)
    vanderbilt: np.ndarray = None                            # B (2x2)
    vanderbilt_asymmetry: float = 0.0                        # max |B - B^T|
    residual_kinetic: list = field(default_factory=list)     # Hartree per wave
    norm_matrix_error: float = 0.0
    q_cut: float = DEFAULT_Q_CUT

    def __repr__(self) -> str:
        energies = ", ".join(f"{e:+.4f}" for e in self.reference_energies)
        return (f"ONCVChannel(l={self.l}, n={self.n}, eps=[{energies}] Ha, "
                f"rc={self.r_cut:.3f} Bohr, {len(self.projectors)} projectors)")


def _inner_grid(r_cut: float) -> np.ndarray:
    return np.linspace(0.0, r_cut, INNER_POINTS)


def _resample(r: np.ndarray, values: np.ndarray, r_new: np.ndarray):
    """Cubic-spline resampling (linear interpolation would leave an O(h^2)
    error that shows up as an asymmetry of the Vanderbilt matrix)."""
    from scipy.interpolate import CubicSpline
    return CubicSpline(r, values)(r_new)


def _bessel_table(l: int, qs: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``(len(qs), len(r))`` table of ``j_l(q_n r)``."""
    return spherical_jn(l, np.outer(qs, r))


def _bessel_transform_table(l: int, qs, r_in, q_grid) -> np.ndarray:
    r"""``B[n, k] = sqrt(2/pi) int_0^rc j_l(q_n r) j_l(q_k r) r^2 dr``."""
    basis = _bessel_table(l, qs, r_in) * r_in * r_in          # (N, R)
    kernel = _bessel_table(l, q_grid, r_in)                    # (Q, R)
    return np.sqrt(2.0 / np.pi) * simpson(basis[:, None, :] * kernel[None, :, :],
                                          x=r_in, axis=-1)


def _tail_transform(l: int, r, wave, r_cut, bound: bool, q_grid) -> np.ndarray:
    r"""Bessel transform of the all-electron tail, from exactly ``r_cut``.

    The tail is resampled with a cubic spline onto a grid four times finer
    than the atomic one and integrated with Simpson's rule: its transform has
    a :math:`q^{-2}` step contribution that must cancel against the inside
    expansion's to the accuracy of the residual energy, so the two pieces
    need the same (high) quadrature accuracy.  A scattering tail, which does
    not decay, is tapered smoothly between :math:`3r_c` and :math:`5r_c`.
    """
    if bound:
        significant = np.nonzero(np.abs(wave * r) > 1e-10)[0]
        r_end = float(r[min(significant[-1] + 1, r.size - 1)])
    else:
        r_end = min(5.0 * r_cut, float(r[-1]))
    step = min(0.25 * (r[1] - r[0]), TAIL_SPACING)
    n = int(np.ceil((r_end - r_cut) / step)) | 1          # odd count for Simpson
    r_tail = np.linspace(r_cut, r_end, n + 1)
    values = _resample(r, wave, r_tail)
    if not bound:
        x = np.clip((r_tail - 3.0 * r_cut) / (2.0 * r_cut), 0.0, 1.0)
        values = values * np.cos(0.5 * np.pi * x) ** 2
    weighted = values * r_tail * r_tail
    # In blocks of q: the full (Q, R) kernel would be hundreds of MB for a
    # slowly decaying tail on the fine oxygen grid.
    out = np.empty(q_grid.size)
    block = max(1, int(2_000_000 // r_tail.size))
    for start in range(0, q_grid.size, block):
        kernel = _bessel_table(l, q_grid[start:start + block], r_tail)
        out[start:start + block] = simpson(kernel * weighted[None, :],
                                           x=r_tail, axis=-1)
    return np.sqrt(2.0 / np.pi) * out


@dataclass
class PseudoWaves:
    """Optimized pseudo partial waves of one channel (independent of V_loc)."""

    l: int
    r_cut: float
    energies: list
    waves: list                    # all-electron R_i(r) on the atomic grid
    wavevectors: list
    coefficients: list
    residual_kinetic: list
    norms: np.ndarray              # all-electron inner norm matrix
    achieved: np.ndarray           # pseudo inner norm matrix
    q_cut: float

    @property
    def norm_matrix_error(self) -> float:
        return float(np.max(np.abs(self.achieved - self.norms)))


def matching_targets(r: np.ndarray, u: np.ndarray, potential: np.ndarray,
                     l: int, energy: float, r_cut: float) -> np.ndarray:
    r"""``[R, R', R'', R''']`` of a partial wave at ``r_cut`` (a grid point).

    ``R'`` comes from a fourth-order stencil on ``u = rR``; ``R''`` and
    ``R'''`` follow from the radial equation
    :math:`u'' = 2[V + l(l+1)/2r^2 - E]\,u` and its derivative, so the
    targets are consistent with the all-electron equation to the accuracy of
    the Numerov solution itself (a polynomial fit of the tabulated wave would
    leave ~1e-6 inconsistencies that surface as an asymmetric :math:`B`).
    """
    h = r[1] - r[0]
    k = int(np.argmin(np.abs(r - r_cut)))
    rk, uk = r[k], u[k]
    up = _derivative(u, h, k)
    vp = _derivative(potential, h, k)
    f = 2.0 * (potential[k] + l * (l + 1) / (2.0 * rk * rk) - energy)
    fp = 2.0 * (vp - l * (l + 1) / rk ** 3)
    upp = f * uk
    uppp = fp * uk + f * up
    return np.array([
        uk / rk,
        up / rk - uk / rk ** 2,
        upp / rk - 2.0 * up / rk ** 2 + 2.0 * uk / rk ** 3,
        uppp / rk - 3.0 * upp / rk ** 2 + 6.0 * up / rk ** 3 - 6.0 * uk / rk ** 4,
    ])


def optimize_pseudo_waves(r: np.ndarray, v_ae: np.ndarray, l: int,
                          waves: list, energies: list, r_cut: float,
                          q_cut: float = DEFAULT_Q_CUT,
                          n_bessel: int = DEFAULT_N_BESSEL,
                          norm_factor: float = 1.0) -> PseudoWaves:
    r"""The Bessel expansions of the pseudo partial waves of one channel.

    Each wave in turn: match value and first three derivatives at ``r_cut``
    (:func:`matching_targets`), conserve the inner norm and every cross norm
    with the earlier waves, and minimize the residual kinetic energy beyond
    ``q_cut`` in what freedom is left (:func:`constrained_minimum`).
    ``r_cut`` is snapped to the nearest grid node, so the matching point and
    the boundary of the inner quadrature coincide.

    ``norm_factor`` scales the target inner-norm matrix,
    :math:`\langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c} =
    f\,\langle\varphi_i|\varphi_j\rangle_{r<r_c}`: 1 (the default) is
    norm conservation; the PAW family uses :math:`f < 1` so its overlap
    correction :math:`(1-f)\langle\varphi_i|\varphi_j\rangle` is positive
    definite by construction.
    """
    r_cut = _snap(r, r_cut)
    r_in = _inner_grid(r_cut)
    q_grid = np.arange(0.0, Q_MAX + 0.5 * Q_STEP, Q_STEP)
    weight = 0.5 * q_grid ** 4 * (q_grid >= q_cut)

    waves_in = [_resample(r, w, r_in) for w in waves]
    norms = np.array([[simpson(a * b * r_in * r_in, x=r_in) for b in waves_in]
                      for a in waves_in])

    coefficients, wavevectors, pseudo_in, residuals = [], [], [], []
    for i, (wave, energy) in enumerate(zip(waves, energies)):
        # Value and first three derivatives of R at r_c (Hamann's ncon = 4).
        target = matching_targets(r, wave * r, v_ae, l, energy, r_cut)
        qs = bessel_wavevectors(l, r_cut, n_bessel)
        j, dj, d2j, d3j = bessel_derivatives(l, qs * r_cut)
        A = [j, qs * dj, qs ** 2 * d2j, qs ** 3 * d3j]
        b = list(target[:4])
        basis_in = _bessel_table(l, qs, r_in)
        G = simpson(basis_in[:, None, :] * basis_in[None, :, :]
                    * (r_in * r_in)[None, None, :], x=r_in, axis=-1)
        for k in range(i):                       # cross norms are linear
            A.append(G @ coefficients[k])
            b.append(float(norm_factor) * norms[i, k])

        transform = _bessel_transform_table(l, qs, r_in, q_grid)   # (N, Q)
        bound = energy < 0 and abs(wave[-1] * r[-1]) < 1e-6
        tail = _tail_transform(l, r, wave, r_cut, bound, q_grid)
        K = (transform * weight) @ transform.T * Q_STEP
        kvec = (transform * weight) @ tail * Q_STEP
        k0 = float(np.sum(weight * tail * tail) * Q_STEP)

        c = constrained_minimum(K, kvec, np.array(A), np.array(b), G,
                                float(norm_factor) * norms[i, i])
        coefficients.append(c)
        wavevectors.append(qs)
        pseudo_in.append(c @ basis_in)
        residuals.append(float(c @ K @ c + 2.0 * kvec @ c + k0))

    achieved = np.array([[simpson(a * b * r_in * r_in, x=r_in) for b in pseudo_in]
                         for a in pseudo_in])
    return PseudoWaves(l=l, r_cut=float(r_cut), energies=list(energies),
                       waves=list(waves), wavevectors=wavevectors,
                       coefficients=coefficients, residual_kinetic=residuals,
                       norms=norms, achieved=achieved, q_cut=float(q_cut))


def assemble_channel(r: np.ndarray, pw: PseudoWaves, v_loc: np.ndarray,
                     n: int, occupation: float = 0.0,
                     strict: bool = True) -> ONCVChannel:
    r"""Projectors and coupling of a channel for a given screened ``v_loc``.

    :math:`\chi_i = \sum_n c_{in}(\varepsilon_i - q_n^2/2 - V_{loc}) j_l(q_n r)`
    inside ``r_cut`` (zero beyond), :math:`B_{ij} = \langle\tilde\varphi_i|
    \chi_j\rangle`, :math:`D = B^{-1}`.  With ``strict`` an asymmetry of
    :math:`B` above :data:`B_ASYMMETRY_TOLERANCE` raises.
    """
    l, r_cut = pw.l, pw.r_cut
    r_in = _inner_grid(r_cut)
    inside = r <= r_cut
    v_loc_in = _resample(r, v_loc, r_in)

    pseudo_in, chi_in = [], []
    for c, qs, energy in zip(pw.coefficients, pw.wavevectors, pw.energies):
        basis_in = _bessel_table(l, qs, r_in)
        pseudo_in.append(c @ basis_in)
        chi_in.append(((energy - 0.5 * qs ** 2)[:, None] * basis_in
                       - v_loc_in[None, :] * basis_in).T @ c)
    B = np.array([[simpson(p * x * r_in * r_in, x=r_in) for x in chi_in]
                  for p in pseudo_in])
    asymmetry = float(np.max(np.abs(B - B.T)))
    if strict and asymmetry > B_ASYMMETRY_TOLERANCE:
        raise RuntimeError(
            f"the Vanderbilt matrix of l={l} is asymmetric by {asymmetry:.2e} "
            "Ha (generalized norm conservation or the matching failed)")
    B = 0.5 * (B + B.T)
    D = np.linalg.inv(B)
    D = 0.5 * (D + D.T)

    pseudo_waves, projectors = [], []
    for c, qs, wave, energy in zip(pw.coefficients, pw.wavevectors, pw.waves,
                                   pw.energies):
        basis = _bessel_table(l, qs, r)
        pseudo_waves.append(np.where(inside, c @ basis, wave))
        chi = ((energy - 0.5 * qs ** 2)[:, None] * basis
               - v_loc[None, :] * basis).T @ c
        projectors.append(np.where(inside, chi, 0.0))

    return ONCVChannel(
        l=l, n=n, eigenvalue=float(pw.energies[0]), r_cut=float(r_cut),
        coefficients=pw.coefficients[0], pseudo_radial=pseudo_waves[0],
        v_screened=v_loc, v_ionic=None, occupation=float(occupation),
        norm_error=pw.norm_matrix_error,
        reference_energies=[float(e) for e in pw.energies],
        wavevectors=list(pw.wavevectors), wave_coefficients=list(pw.coefficients),
        pseudo_waves=pseudo_waves, projectors=projectors, coupling=D,
        vanderbilt=B, vanderbilt_asymmetry=asymmetry,
        residual_kinetic=list(pw.residual_kinetic),
        norm_matrix_error=pw.norm_matrix_error, q_cut=pw.q_cut)


# --------------------------------------------------------------------------- #
# The local potential.
# --------------------------------------------------------------------------- #

def polynomial_local_potential(r: np.ndarray, v_ae: np.ndarray,
                               r_local: float, shift: float = 0.0
                               ) -> np.ndarray:
    r"""Even polynomial continuation of ``v_ae`` inside ``r_local``.

    :math:`V(r) = \sum_{k} a_k r^{2k}` for :math:`r < r_{cl}`, with the
    coefficients fixed by continuity of the value and the first four
    derivatives at :math:`r_{cl}` (five even powers) and, when ``shift`` is
    nonzero, a sixth power raising the value at the origin by ``shift``
    Hartree above the five-coefficient continuation -- Hamann's ``dvloc0``,
    the knob that sets how strongly the projectors have to act.
    :math:`V'(0) = 0` by parity, so the potential is smooth at the origin.
    """
    target = _local_derivatives(r, v_ae, r_local, order=4)
    powers = np.arange(0, 12, 2)

    def rows(n_powers):
        out = []
        for order in range(5):
            row = []
            for power in powers[:n_powers]:
                if power < order:
                    row.append(0.0)
                    continue
                factor = (np.prod([power - j for j in range(order)])
                          if order else 1.0)
                row.append(factor * r_local ** (power - order))
            out.append(row)
        return out

    a5 = np.linalg.solve(np.array(rows(5)), target)
    origin = a5[0] + float(shift)
    system = np.array(rows(6) + [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    a = np.linalg.solve(system, np.concatenate([target, [origin]]))
    inside = r <= r_local
    v_in = sum(coefficient * r ** power for coefficient, power in zip(a, powers))
    return np.where(inside, v_in, v_ae)


# --------------------------------------------------------------------------- #
# The pseudopotential record.
# --------------------------------------------------------------------------- #

@dataclass
class ONCVPseudoPotential(PseudoPotential):
    r"""An ONCVPSP pseudopotential: local potential + two projectors per channel.

    Inherits the :class:`~.generation.PseudoPotential` layout so the valence
    basis (:func:`~.orbitals.pseudo_basis`), the local potential sampler
    (:meth:`local_potential`) and the multiple-zeta hierarchy work unchanged;
    every ``channels[l]`` is an :class:`ONCVChannel`.  ``projectors`` maps
    ``l -> [chi_1(r), chi_2(r)]`` and ``coupling`` maps ``l -> D`` (the
    :math:`2\times2` block).  ``kb_energies`` is empty: the coupling is a
    matrix, not a number per channel.
    """

    coupling: dict = field(default_factory=dict)           # l -> (2, 2)
    v_local_screened: np.ndarray = None
    r_cut_local: float = 0.0
    local_shift: float = 0.0
    q_cut: float = DEFAULT_Q_CUT
    energy_offset: float = DEFAULT_ENERGY_OFFSET

    @property
    def nonlocal_channels(self) -> list:
        return sorted(self.projectors)

    def projector(self, l: int, radius, index: int = 0) -> np.ndarray:
        """Interpolate projector ``index`` of channel ``l`` onto ``radius``."""
        radius = np.asarray(radius, dtype=float)
        chi = self.projectors[int(l)][int(index)]
        return np.where(radius <= self.r[-1],
                        np.interp(np.clip(radius, self.r[0], self.r[-1]),
                                  self.r, chi), 0.0)

    def reference_energies(self, l: int) -> list:
        return list(self.channels[int(l)].reference_energies)

    def residual_kinetic(self, l: int) -> list:
        return list(self.channels[int(l)].residual_kinetic)

    def __repr__(self) -> str:
        channels = ", ".join(f"l={l}x{len(self.projectors.get(l, []))}"
                             for l in sorted(self.channels))
        return (f"ONCVPseudoPotential({self.symbol}, Z_ion="
                f"{self.valence_charge:g}, [{channels}], rcl="
                f"{self.r_cut_local:.2f}, shift={self.local_shift:+.2f}, "
                f"qc={self.q_cut:g})")


def generation_points(atomic_number: int, minimum: int = 6000,
                      per_z: int = 1500) -> int:
    """Uniform radial grid points for the reference atom of element ``Z``.

    The Numerov partial waves must satisfy the all-electron radial equation
    to ~1e-7 (measured through the Wronskian identity that makes the
    Vanderbilt matrix symmetric); with the core scale :math:`a_0/Z` that
    needs ``1500 * Z`` points out to 30 Bohr -- 12000 for oxygen, where 6000
    leaves a 6e-6 asymmetry.
    """
    return max(int(minimum), int(per_z) * int(atomic_number))


def _snap(r: np.ndarray, radius: float) -> float:
    """The grid node nearest to ``radius``."""
    return float(r[int(np.argmin(np.abs(r - radius)))])


def _cutoff_for(symbol, l, r, radial, r_cut, rc_factor):
    if isinstance(r_cut, dict):
        return float(r_cut[l])
    if r_cut is not None:
        return float(r_cut)
    table = DEFAULT_CUTOFFS.get(symbol)
    if table is not None and l in table:
        return float(table[l])
    peak = r[int(np.argmax(np.abs(radial * r)))]
    return float(rc_factor * peak)


def generate_oncv(symbol: str, *, r_cut=None, rc_factor: float = DEFAULT_RC_FACTOR,
                  r_cut_local: float | None = None,
                  local_factor: float = DEFAULT_LOCAL_FACTOR,
                  local_shift: float = 0.0,
                  q_cut: float = DEFAULT_Q_CUT,
                  energy_offset: float = DEFAULT_ENERGY_OFFSET,
                  n_bessel: int = DEFAULT_N_BESSEL,
                  points: int | None = None, r_max: float = 30.0,
                  atom: AtomicResult | None = None) -> ONCVPseudoPotential:
    r"""Generate an ONCVPSP pseudopotential for ``symbol``.

    Parameters
    ----------
    r_cut : float or dict, optional
        Channel cutoff radii (Bohr), one value or ``{l: r_c}``; defaults to
        :data:`DEFAULT_CUTOFFS` for the element, else ``rc_factor`` times the
        outermost maximum of the bound partial wave.
    r_cut_local : float, optional
        Radius of the polynomial local potential (default ``local_factor``
        times the smallest channel cutoff).
    local_shift : float
        Raise of the local potential at the origin (Hartree, Hamann's
        ``dvloc0``), a knob against ghost states; zero by default.
    q_cut : float
        Wave-vector cutoff of the residual kinetic energy (Bohr^-1).
    energy_offset : float
        Second reference energy above the bound state for channels without
        a second bound state (Hartree).
    n_bessel : int
        Spherical Bessel functions per pseudo wave.
    points : int, optional
        Radial grid points of the all-electron atom; default
        :func:`generation_points` (finer for heavier atoms, so the Numerov
        partial waves satisfy the radial equation to the ~1e-7 needed for a
        symmetric Vanderbilt matrix).
    """
    from ase.data import atomic_numbers

    atomic_number = int(atomic_numbers[symbol])
    if atom is None:
        atom = solve_atom(atomic_number,
                          points=(generation_points(atomic_number)
                                  if points is None else int(points)),
                          r_max=r_max, tolerance=1e-7, mixing=0.25)
    valence_config, _core = _valence_configuration(atomic_number)
    if not valence_config:
        raise ValueError(f"{symbol} has no valence subshells to pseudize")
    valence_charge = float(sum(valence_config.values()))
    r, v_ae = atom.r, atom.v_effective
    z_eff = float(atomic_number)

    # Partial waves per l: every occupied valence state of that l (bound,
    # Numerov-refined) plus, when there is only one, the scattering state.
    per_l: dict = {}
    for (n, l), occupancy in sorted(valence_config.items()):
        u, energy = bound_state(r, v_ae, l, atom.eigenvalues[(n, l)], z_eff)
        per_l.setdefault(l, []).append((n, energy, u / r, occupancy))

    # Cutoff radii snapped to the atomic grid, so "inside r_c" and the
    # matching point are the same grid node.
    cutoffs = {l: _snap(r, _cutoff_for(symbol, l, r, states[0][2], r_cut,
                                        rc_factor))
               for l, states in per_l.items()}
    r_local = _snap(r, float(r_cut_local) if r_cut_local is not None
                    else float(local_factor * min(cutoffs.values())))

    pseudo_waves: dict = {}
    for l, states in per_l.items():
        waves = [w for _n, _e, w, _o in states]
        energies = [e for _n, e, _w, _o in states]
        if len(states) == 1:
            energy_2 = energies[0] + float(energy_offset)
            u2 = scattering_wave(r, v_ae, l, energy_2, z_eff)
            inside = r <= cutoffs[l]
            u2 = u2 / np.sqrt(np.trapezoid(u2[inside] ** 2, r[inside]))
            waves.append(u2 / r)
            energies.append(energy_2)
        pseudo_waves[l] = optimize_pseudo_waves(
            r, v_ae, l, waves[:2], energies[:2], cutoffs[l], q_cut=q_cut,
            n_bessel=n_bessel)

    shift = float(local_shift)
    v_loc = polynomial_local_potential(r, v_ae, r_local, shift)

    channels: dict = {}
    for l, states in per_l.items():
        channels[l] = assemble_channel(
            r, pseudo_waves[l], v_loc, n=states[0][0],
            occupation=float(sum(o for _n, _e, _w, o in states)))

    # Unscreen with the pseudo valence density.
    valence_density = np.zeros_like(r)
    for channel in channels.values():
        valence_density += channel.occupation * channel.pseudo_radial ** 2 \
            / (4.0 * np.pi)
    v_hartree = hartree_potential(r, valence_density)
    _e_xc, v_xc = lda_xc(valence_density)
    v_local_ionic = v_loc - v_hartree - v_xc
    for channel in channels.values():
        channel.v_ionic = v_local_ionic

    return ONCVPseudoPotential(
        symbol=symbol, atomic_number=atomic_number,
        valence_charge=valence_charge, r=r, channels=channels,
        v_local=v_local_ionic, local_l=-1,
        projectors={l: list(c.projectors) for l, c in channels.items()},
        kb_energies={}, valence_density=valence_density, atom=atom,
        family=FAMILY,
        coupling={l: np.array(c.coupling) for l, c in channels.items()},
        v_local_screened=v_loc, r_cut_local=r_local, local_shift=float(shift),
        q_cut=float(q_cut), energy_offset=float(energy_offset))


# --------------------------------------------------------------------------- #
# Diagnostics: spectrum, logarithmic derivatives, report.
# --------------------------------------------------------------------------- #

#: Spacing (Bohr) of the uniform grid the radial spectrum is solved on.
SPECTRUM_SPACING = 0.005


def _channel_operator(pp: ONCVPseudoPotential, l: int, r_max: float,
                      stride: int):
    """Grid, potential and ``u``-form projectors of the pseudo Hamiltonian.

    Everything is resampled (cubic spline) onto a uniform grid of spacing
    ``stride * SPECTRUM_SPACING`` starting one step from the origin, so the
    grid's implicit Dirichlet node sits exactly at ``r = 0`` and the
    finite-difference Laplacian is in its asymptotic regime even for a
    library file decimated to 0.02 Bohr (where the stored grid alone would
    leave a 1e-2 Ha eigenvalue error).
    """
    from scipy.interpolate import CubicSpline

    h = stride * SPECTRUM_SPACING
    r = np.arange(1, int(r_max / h) + 1) * h
    v = CubicSpline(pp.r, pp.v_local_screened)(r) + l * (l + 1) / (2.0 * r * r)
    chi_u = [CubicSpline(pp.r, np.asarray(chi))(r) * r
             for chi in pp.projectors[l]]
    return r, v, chi_u, np.asarray(pp.coupling[l], dtype=float)


def _spectrum_on_grid(r, v, chi_u, D, n_states):
    """Lowest eigenvalues with the 3-point Laplacian (Dirichlet at r = 0)."""
    from scipy.linalg import eigh

    h = r[1] - r[0]
    n = r.size
    H = np.diag(1.0 / h ** 2 + v)
    idx = np.arange(n - 1)
    H[idx, idx + 1] = H[idx + 1, idx] = -0.5 / h ** 2
    if chi_u:
        X = np.array(chi_u)                                    # (P, n)
        H += h * X.T @ D @ X
    return eigh(H, eigvals_only=True, subset_by_index=[0, n_states - 1])


def _spectrum_extent(pp: ONCVPseudoPotential, l: int, floor: float = 1e-5,
                     bounds=(12.0, 24.0)) -> float:
    """Box radius for :func:`radial_spectrum`: where the bound pseudo wave has
    decayed to ``floor`` of its maximum (a diffuse Li 2s reaches 18 Bohr)."""
    r = pp.r
    u = np.abs(pp.channels[int(l)].pseudo_radial * r)
    peak = int(np.argmax(u))
    tail = np.nonzero(u[peak:] > floor * u[peak])[0]
    extent = float(r[peak + tail[-1]]) if tail.size else bounds[1]
    return float(np.clip(extent, *bounds))


def radial_spectrum(pp: ONCVPseudoPotential, l: int, n_states: int = 3,
                    r_max: float | None = None, stride: int = 2) -> np.ndarray:
    r"""Lowest eigenvalues of :math:`T_l + V^{scr}_{loc} + \sum|\chi_i\rangle
    D_{ij}\langle\chi_j|` on the radial grid.

    The 3-point Laplacian (whose Dirichlet condition at the origin is exact
    for every :math:`l`, unlike a wider stencil) is solved at ``stride`` and
    ``2 * stride`` and Richardson-extrapolated, :math:`(4E_h - E_{2h})/3`,
    which removes its :math:`O(h^2)` error (``stride=2``, i.e. 0.01/0.02
    Bohr, reproduces the reference energies to ~1e-5 Ha in a fraction of a
    second; ``stride=1`` to ~1e-6).  A ghost state shows up as an eigenvalue
    below the bound reference energy.
    """
    r_max = _spectrum_extent(pp, l) if r_max is None else float(r_max)
    fine = _spectrum_on_grid(*_channel_operator(pp, l, r_max, stride), n_states)
    coarse = _spectrum_on_grid(*_channel_operator(pp, l, r_max, 2 * stride),
                               n_states)
    return (4.0 * fine - coarse) / 3.0


def log_derivative_ps(pp: ONCVPseudoPotential, l: int, energy: float,
                      r_cut: float | None = None) -> float:
    r"""Pseudo logarithmic derivative :math:`R'/R` at ``r_cut``.

    The nonlocal term is handled exactly: ``u = u_0 + sum_j a_j u_j`` with
    ``u_0`` the homogeneous outward Numerov solution and ``u_j`` the solutions
    of ``(T + V - E) u_j = -chi_j``; the amplitudes ``a = D <chi|u>`` follow
    from a 2x2 linear system.
    """
    l = int(l)
    r_cut = pp.channels[l].r_cut if r_cut is None else float(r_cut)
    r = pp.r
    r0 = _with_origin(r)
    v0 = np.concatenate([[pp.v_local_screened[0]], pp.v_local_screened])
    f = _radial_f(r0, v0, l, energy)
    chi_u = [np.concatenate([[0.0], np.asarray(chi) * r])
             for chi in pp.projectors[l]]
    D = np.asarray(pp.coupling[l], dtype=float)
    seed = (r0[1] ** (l + 1), r0[2] ** (l + 1))
    u0 = numerov_outward(r0, f, np.zeros_like(r0), seed, start=1)
    uj = [numerov_outward(r0, f, 2.0 * chi, (0.0, 0.0), start=1)
          for chi in chi_u]
    dr = r0[1] - r0[0]
    m0 = np.array([np.trapezoid(chi * u0, dx=dr) for chi in chi_u])
    M = np.array([[np.trapezoid(chi * u, dx=dr) for u in uj] for chi in chi_u])
    a = np.linalg.solve(np.eye(len(chi_u)) - D @ M, D @ m0)
    u = u0 + sum(ai * ui for ai, ui in zip(a, uj))
    return _log_derivative_of_u(r0, u, r_cut)


def check_oncv_channel(pp: ONCVPseudoPotential, l: int,
                       midpoint: bool = True) -> dict:
    """Validation numbers of one channel.

    ``eigenvalue_error``
        Lowest eigenvalue of the pseudo atomic Hamiltonian of that ``l`` minus
        the bound reference energy (a ghost state makes it negative).
    ``norm_matrix_error``, ``vanderbilt_asymmetry``, ``residual_kinetic``,
    ``coupling_norm``, ``nodes``
        Stored construction diagnostics.
    ``tail_error``
        Largest deviation of the bound pseudo wave from the all-electron one
        beyond ``r_c``.
    ``log_derivative_errors``
        ``{energy: (|L_ps - L_ae|, L_ae)}`` at the two reference energies and
        (when ``midpoint``) halfway between them.

    The last two need the generating all-electron atom (``pp.atom``); on a
    potential loaded from the library they are ``None``.
    """
    channel = pp.channels[int(l)]
    r = pp.r
    energies = list(channel.reference_energies)
    spectrum = radial_spectrum(pp, l)
    out = {
        "eigenvalue_error": float(spectrum[0] - energies[0]),
        "spectrum": spectrum,
        "tail_error": None,
        "norm_matrix_error": float(channel.norm_matrix_error),
        "vanderbilt_asymmetry": float(channel.vanderbilt_asymmetry),
        "coupling_norm": float(np.linalg.norm(channel.coupling, 2)),
        "log_derivative_errors": None,
        "residual_kinetic": list(channel.residual_kinetic),
        "nodes": int(np.sum(np.diff(np.sign(
            channel.pseudo_radial[(r > 0.05) & (r < 5.0)])) != 0)),
    }
    ae = pp.atom
    if ae is None:
        return out
    z = float(pp.atomic_number)
    probes = list(energies)
    if midpoint:
        probes.append(0.5 * (energies[0] + energies[1]))
    errors = {}
    for energy in probes:
        l_ae = log_derivative_ae(r, ae.v_effective, l, energy, channel.r_cut, z)
        l_ps = log_derivative_ps(pp, l, energy)
        errors[float(energy)] = (float(abs(l_ps - l_ae)), float(l_ae))
    bound_u, _e = bound_state(r, ae.v_effective, l, energies[0], z)
    outside = r > channel.r_cut
    out["tail_error"] = float(np.max(np.abs(
        channel.pseudo_radial[outside] - (bound_u / r)[outside])))
    out["log_derivative_errors"] = errors
    return out


def report_oncv(pp: ONCVPseudoPotential) -> str:
    """Human-readable validation summary for every channel."""
    lines = [f"{pp!r}",
             f"  valence charge  : {pp.valence_charge:g}",
             f"  local potential : polynomial inside rcl = {pp.r_cut_local:.3f} "
             f"Bohr, shift = {pp.local_shift:+.2f} Ha, V_loc(0) = "
             f"{pp.v_local_screened[0]:+.4f} Ha (screened)"]
    for l in sorted(pp.channels):
        channel = pp.channels[l]
        checks = check_oncv_channel(pp, l)
        lines.append(f"  l={l}: rc={channel.r_cut:.3f}  eps="
                     + ", ".join(f"{e:+.4f}" for e in channel.reference_energies)
                     + f"  E_res=" + ", ".join(f"{e:.2e}" for e in
                                               channel.residual_kinetic)
                     + f" Ha  norm-matrix err={checks['norm_matrix_error']:.1e}"
                     f"  eps err={checks['eigenvalue_error']:+.1e}  "
                     f"nodes={checks['nodes']}")
        for energy, (error, l_ae) in (checks["log_derivative_errors"]
                                      or {}).items():
            lines.append(f"        L(E={energy:+.4f}) = {l_ae:+.5f}  "
                         f"|dL| = {error:.1e}")
        lines.append("        D = " + np.array2string(
            channel.coupling, precision=5, suppress_small=True).replace("\n", ""))
    return "\n".join(lines)


def diagonalized_projectors(pp: ONCVPseudoPotential, l: int):
    r"""Hamann's orthogonalized form: ``(projectors, energies)`` with
    :math:`V_{NL} = \sum_k |\chi'_k\rangle d_k \langle\chi'_k|`, from the
    eigendecomposition :math:`D = \sum_k d_k v_k v_k^T`,
    :math:`\chi'_k = \sum_i v_{ki}\chi_i`.  Identical operator, diagonal
    coupling.
    """
    D = np.asarray(pp.coupling[int(l)], dtype=float)
    values, vectors = np.linalg.eigh(D)
    chis = np.array(pp.projectors[int(l)])
    return [vectors[:, k] @ chis for k in range(values.size)], list(values)


# --------------------------------------------------------------------------- #
# Molecule: projectors, coupling blocks, the family builder.
# --------------------------------------------------------------------------- #

def oncv_projectors(symbols, positions, potentials, units: str = "angstrom"):
    """Two :class:`~.orbitals.KBProjector` per ``(atom, l, m)``."""
    from .orbitals import KBProjector

    projectors = []
    for index, (symbol, position) in enumerate(zip(symbols, positions)):
        pp = potentials[symbol]
        for l in sorted(pp.projectors):
            D = np.asarray(pp.coupling[l], dtype=float)
            for m in range(-l, l + 1):
                for i, chi in enumerate(pp.projectors[l]):
                    projectors.append(KBProjector(
                        pp, l, m, center=position, units=units,
                        atom_index=index, index=i, radial=chi,
                        kb_energy=float(D[i, i])))
    return projectors


def oncv_coupling_blocks(projectors, symbols, potentials) -> dict:
    """``{(atom, l, m): D_l}`` -- the Vanderbilt blocks for the projectors."""
    blocks = {}
    for projector in projectors:
        key = projector.block_key
        if key not in blocks:
            pp = potentials[symbols[projector.atom_index]]
            blocks[key] = np.asarray(pp.coupling[projector.l], dtype=complex)
    return blocks


def oncv_library_path(directory=None) -> str:
    """The ONCVPSP library directory (``library/oncvpsp`` by default)."""
    from .io import library_root
    if directory is not None:
        return os.fspath(directory)
    return os.path.join(library_root(), LIBRARY_SUBDIR)


_CACHE: dict = {}


def get_oncv(symbol: str, directory=None) -> ONCVPseudoPotential:
    """Load ``symbol`` from the ONCVPSP library (cached)."""
    from .io import available_elements, library_file, load_pseudopotential

    folder = oncv_library_path(directory)
    key = f"{symbol}@{folder}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    path = library_file(symbol, folder)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no ONCVPSP pseudopotential for {symbol!r} at {path!r}. "
            f"Available: {', '.join(available_elements(folder)) or '(none)'}. "
            "Generate it with build_oncv_library([symbol]).")
    pp = load_pseudopotential(path)
    if str(getattr(pp, "family", "")).lower() != FAMILY:
        raise ValueError(f"{path!r} belongs to family {pp.family!r}, not "
                         f"{FAMILY!r}")
    _CACHE[key] = pp
    return pp


def build_oncv_library(elements=("H", "Li", "C", "N", "O", "F"),
                       directory=None, *, verbose: bool = True,
                       format: str | None = None, stride: int | None = None,
                       **generation_options):
    """Generate and save ONCVPSP potentials for ``elements``; returns paths."""
    from .io import DEFAULT_FORMAT, STRIDE, library_file, save_pseudopotential

    folder = oncv_library_path(directory)
    format = DEFAULT_FORMAT if format is None else format
    stride = STRIDE if stride is None else int(stride)
    written = []
    for symbol in elements:
        pp = generate_oncv(symbol, **generation_options)
        path = save_pseudopotential(pp, library_file(symbol, folder, format),
                                    format=format, stride=stride)
        written.append(path)
        if verbose:
            print(f"  {symbol:>2}  Z_ion={pp.valence_charge:>4.0f}  "
                  + "  ".join(f"l{l}: rc={c.r_cut:.2f} E_res="
                              + "/".join(f"{e:.1e}" for e in c.residual_kinetic)
                              for l, c in sorted(pp.channels.items()))
                  + f"  -> {os.path.basename(path)}")
    _CACHE.clear()
    return written


def build_oncv(atoms, grid, h, charge, spin, options, kinetic=None):
    r"""Valence-only Hamiltonian from ONCVPSP pseudopotentials.

    Same 5-tuple as the Troullier-Martins builder
    (:func:`~.families._build_tm`): the basis is the bound pseudo partial
    waves (with the ``size`` hierarchy), the external potential the local
    channel, and the nonlocal term the two-projector Vanderbilt form with one
    :math:`2\times2` coupling block per ``(atom, l, m)``; no overlap
    correction (norm-conserving).
    """
    from ...algorithms._hamiltonian_from_atoms import (
        DEFAULT_KINETIC, _num_particles, _warn_unresolved, coherent_positions,
        grid_from_cell, resolve_num_unpaired)
    from ...core import MolecularIntegrals
    from .orbitals import pseudo_basis, valence_electrons

    directory = options.get("directory")
    symbols = atoms.get_chemical_symbols()
    positions = coherent_positions(atoms)
    potentials = {symbol: get_oncv(symbol, directory) for symbol in set(symbols)}

    basis_fns, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, size=options.get("size", "SZ"),
        split_norm=options.get("split_norm"))
    projectors = oncv_projectors(symbols, positions, potentials)
    blocks = oncv_coupling_blocks(projectors, symbols, potentials)
    nuclei = [(potentials[symbol].valence_charge, position)
              for symbol, position in zip(symbols, positions)]

    n_el = int(round(valence_electrons(symbols, potentials))) - int(charge)
    g = (grid if grid is not None
         else grid_from_cell(atoms, h, center=positions.mean(axis=0)))
    n_unpaired = resolve_num_unpaired(atoms, spin, n_el)
    num_particles = _num_particles(n_el, n_unpaired, "PP")
    integrals = MolecularIntegrals(
        nuclei, basis_fns, g, softening=0.0,
        pseudopotentials=[potentials[s] for s in symbols],
        kb_projectors=projectors, nonlocal_coupling=blocks,
        nonlocal_overlap=None,
        kinetic=kinetic or DEFAULT_KINETIC["pseudopotentials"])
    hamiltonian = integrals.molecular_hamiltonian(mo_basis=True,
                                                  n_electrons=n_el,
                                                  num_particles=num_particles)
    _warn_unresolved(integrals, basis_fns, h)

    context = {"integrals": integrals, "atom_of_orbital": atom_of_orbital,
               "frozen": (), "n_electrons": n_el,
               "pseudopotentials": potentials, "kb_projectors": projectors,
               "nonlocal_coupling": blocks, "family": FAMILY}
    return (hamiltonian, num_particles, len(basis_fns),
            integrals.integration_profile(), context)


# --------------------------------------------------------------------------- #
# On-disk payload (used by io.py for family "oncvpsp").
# --------------------------------------------------------------------------- #

def to_payload(pp: ONCVPseudoPotential, stride: int = 1) -> dict:
    """JSON-shaped payload; radial tables live under ``"radial_tables"``."""
    from .io import _table

    tables = {"r": _table(pp.r, stride),
              "v_local": _table(pp.v_local, stride),
              "v_local_screened": _table(pp.v_local_screened, stride),
              "valence_density": _table(pp.valence_density, stride)}
    channels = {}
    for l, channel in pp.channels.items():
        key = str(l)
        for i, (wave, chi) in enumerate(zip(channel.pseudo_waves,
                                            channel.projectors)):
            tables[f"pseudo_wave_l{key}_{i}"] = _table(wave, stride)
            tables[f"projector_l{key}_{i}"] = _table(chi, stride)
        channels[key] = {
            "n": int(channel.n),
            "r_cut": float(channel.r_cut),
            "occupation": float(channel.occupation),
            "reference_energies": [float(e) for e in channel.reference_energies],
            "wavevectors": [np.asarray(q).tolist() for q in channel.wavevectors],
            "wave_coefficients": [np.asarray(c).tolist()
                                  for c in channel.wave_coefficients],
            "coupling": np.asarray(channel.coupling, dtype=float).tolist(),
            "vanderbilt": np.asarray(channel.vanderbilt, dtype=float).tolist(),
            "vanderbilt_asymmetry": float(channel.vanderbilt_asymmetry),
            "residual_kinetic": [float(e) for e in channel.residual_kinetic],
            "norm_matrix_error": float(channel.norm_matrix_error),
            "q_cut": float(channel.q_cut),
        }
    return {"symbol": pp.symbol, "atomic_number": int(pp.atomic_number),
            "valence_charge": float(pp.valence_charge),
            "r_cut_local": float(pp.r_cut_local),
            "local_shift": float(pp.local_shift), "q_cut": float(pp.q_cut),
            "energy_offset": float(pp.energy_offset),
            "channels": channels, "radial_tables": tables}


def from_payload(payload: dict) -> ONCVPseudoPotential:
    """Rebuild the record written by :func:`to_payload`."""
    tables = payload["radial_tables"]
    r = np.asarray(tables["r"], dtype=float)
    v_local = np.asarray(tables["v_local"], dtype=float)
    v_screened = np.asarray(tables["v_local_screened"], dtype=float)
    channels, projectors, coupling = {}, {}, {}
    for key, entry in payload["channels"].items():
        l = int(key)
        waves, chis = [], []
        i = 0
        while f"pseudo_wave_l{key}_{i}" in tables:
            waves.append(np.asarray(tables[f"pseudo_wave_l{key}_{i}"], float))
            chis.append(np.asarray(tables[f"projector_l{key}_{i}"], float))
            i += 1
        D = np.asarray(entry["coupling"], dtype=float)
        channels[l] = ONCVChannel(
            l=l, n=int(entry["n"]),
            eigenvalue=float(entry["reference_energies"][0]),
            r_cut=float(entry["r_cut"]),
            coefficients=np.asarray(entry["wave_coefficients"][0], float),
            pseudo_radial=waves[0], v_screened=v_screened, v_ionic=v_local,
            occupation=float(entry["occupation"]),
            norm_error=float(entry.get("norm_matrix_error", 0.0)),
            reference_energies=[float(e) for e in entry["reference_energies"]],
            wavevectors=[np.asarray(q, float) for q in entry["wavevectors"]],
            wave_coefficients=[np.asarray(c, float)
                               for c in entry["wave_coefficients"]],
            pseudo_waves=waves, projectors=chis, coupling=D,
            vanderbilt=np.asarray(entry["vanderbilt"], dtype=float),
            vanderbilt_asymmetry=float(entry.get("vanderbilt_asymmetry", 0.0)),
            residual_kinetic=[float(e) for e in entry["residual_kinetic"]],
            norm_matrix_error=float(entry.get("norm_matrix_error", 0.0)),
            q_cut=float(entry.get("q_cut", payload.get("q_cut", DEFAULT_Q_CUT))))
        projectors[l] = chis
        coupling[l] = D
    return ONCVPseudoPotential(
        symbol=payload["symbol"], atomic_number=int(payload["atomic_number"]),
        valence_charge=float(payload["valence_charge"]), r=r,
        channels=channels, v_local=v_local, local_l=-1, projectors=projectors,
        kb_energies={},
        valence_density=np.asarray(tables["valence_density"], dtype=float),
        atom=None, family=FAMILY, coupling=coupling,
        v_local_screened=v_screened, r_cut_local=float(payload["r_cut_local"]),
        local_shift=float(payload.get("local_shift", 0.0)),
        q_cut=float(payload.get("q_cut", DEFAULT_Q_CUT)),
        energy_offset=float(payload.get("energy_offset",
                                        DEFAULT_ENERGY_OFFSET)))


# --------------------------------------------------------------------------- #
# Registration.
# --------------------------------------------------------------------------- #

def _register():
    from .families import FamilySpec, PSEUDO_FAMILIES, register_family
    if FAMILY in PSEUDO_FAMILIES:
        return PSEUDO_FAMILIES[FAMILY]
    return register_family(FamilySpec(
        name=FAMILY,
        description="optimized norm-conserving Vanderbilt (Hamann 2013): "
                    "two projectors per channel, polynomial local potential",
        generate=lambda symbol, **options: generate_oncv(symbol, **options),
        get=get_oncv,
        build=build_oncv,
        norm_conserving=True,
        aliases=FAMILY_ALIASES,
    ))


ONCV_FAMILY = _register()
