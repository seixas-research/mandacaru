# -*- coding: utf-8 -*-
# file: pseudopotentials/partial_waves.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Reference waves, optimized smooth partial waves and the checks of a
generated dataset -- the radial machinery of PAW-LCAO and UPAW-LCAO
generation (:mod:`.paw`).

* **All-electron reference waves.**  Bound states and scattering states of the
  reference atom's potential by Numerov integration (:func:`numerov_outward`,
  :func:`bound_state`, :func:`scattering_wave`), scalar-relativistic or with a
  Dirac :math:`\kappa`, and the reference set of a channel
  (:func:`reference_waves`).
* **Optimized smooth partial waves** (D. R. Hamann, Phys. Rev. B **88**,
  085117 (2013)).  Inside :math:`r_c` a smooth wave is a sum of spherical
  Bessel functions matched in value and first three derivatives to the
  all-electron wave (:func:`matching_targets`), with its residual kinetic
  energy beyond a wave-vector cutoff minimized under the norm condition
  (:func:`optimize_pseudo_waves`, :func:`constrained_minimum`).
* **The local potential**, a polynomial continuation of the all-electron
  potential inside :math:`r_{cl}` (:func:`polynomial_local_potential`).
* **Checks**: ghost states below a reference level (:func:`ghost_errors`,
  :func:`local_potential_ghosts`), scattering phases against the all-electron
  atom (:func:`scattering_errors`, :func:`log_derivative_errors`), the repair
  search that removes a ghost (:func:`ghost_free`), and the defect record a
  dataset carries and warns about on every load (:func:`defects_record`,
  :func:`warn_defects`).

Written from scratch on top of Mandacaru's own radial atom
(:mod:`mandacaru.basis.atomic_solver`); nothing is read from tables.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import spherical_jn

from .dataset import _local_derivatives, _valence_configuration


#: Spherical Bessel functions per pseudo partial wave (Hamann's ``nbas``).
DEFAULT_N_BESSEL = 8


#: Wave-vector cutoff of the residual kinetic energy (Bohr^-1, Hamann's ``qcut``).
DEFAULT_Q_CUT = 5.0


#: Exchange-correlation functional of the reference atom and the unscreening.
DEFAULT_XC = "lda"


#: Radial equation of the reference atom.  Scalar-relativistic by default:
#: mass-velocity and Darwin are a first-row effect already (the 1s of neon
#: moves by 0.06 Ha) and grow fast with Z, while the cost over the
#: non-relativistic solve is one extra tridiagonal solve per state.
DEFAULT_RELATIVITY = "scalar"


#: Whether to build a partial core density and unscreen with it.  On by
#: default: the unscreening is where the nonlinearity of v_xc is committed,
#: and leaving it uncorrected is an error of the generator, not a choice
#: about the calculation that follows.
DEFAULT_NLCC = True


#: Channels added above the highest valence l unless an element-specific
#: bound-state deficit requires one. La's empty 4f is bound by the neutral
#: all-electron atom but absent from its projector-free local channel.
DEFAULT_EXTRA_L = 0


#: Fraction of the radial grid the fallback cutoff of :func:`_cutoff_for` may
#: reach before it refuses.  Half leaves the Wronskian stencil and the tail
#: comparison room to work in.
CUTOFF_GRID_FRACTION = 0.5


#: Fallback cutoff (Bohr) above which :func:`_cutoff_for` warns.  A
#: deliberate Li cutoff of 2.60 Bohr was the largest tabulated one, and Li
#: grows a ghost state at 3.3.
CUTOFF_WARN_RADIUS = 3.3


#: How far (Hartree) a channel's lowest eigenvalue must lie below its bound
#: reference, *with the reference level itself displaced to second place*, for
#: the channel to count as holding a ghost state.  The displacement is what
#: tells a ghost from an inaccurate level: a d channel whose one level sits
#: 6e-4 Hartree low (the shipped Ga) has no extra state, a ghosted s channel
#: has an extra state 0.1 to 100 Hartree down and its true level above it.
GHOST_TOLERANCE = 1e-4


#: Local-potential raises (Hartree) :func:`ghost_free` tries, in order, once
#: the channel cutoffs are balanced.
GHOST_REMEDY_SHIFTS = (0.0, 10.0, 20.0, 40.0, 80.0)


#: Raises tried first with every channel at its *own* cutoff.  Once a
#: PAW-LCAO local potential follows the largest cutoff, a raise acts on the
#: whole extended channel, and thorium (10 Ha) and uranium (5 Ha) come out
#: clean without stretching their compact 5f channel -- which balancing did,
#: at 0.09-0.13 rad of phase error.
OWN_CUTOFF_SHIFTS = (5.0, 10.0, 20.0, 40.0)


#: Fractions of the local radius tried, at no raise, before any raise.  A
#: raise acts on every angular momentum the projectors do not cover, and on
#: whatever of a neighbor's orbital they do not span: aluminum's first
#: repair (5 Ha over 3.08 Bohr) scattered d 0.52 rad off and put fcc Al at
#: 4.48 Angstrom; a 0.6x radius at no raise passes every check and gives
#: 4.11 (HISTORY.md, 2026-10-06, "K21 diagnosed").
SHORTER_LOCAL_FACTORS = (0.75, 0.6, 0.45)


#: What a generator does about a ghost state: build around it, refuse,
#: return the dataset as it came out (for studying one), or repair and,
#: where no remedy works, return the least-defective attempt
#: with its defects recorded on it (``flag``), so a library can hold every
#: element and a calculation that loads a flagged one is warned.
GHOST_MODES = ("repair", "refuse", "keep", "flag")


#: A repaired channel must scatter like the all-electron atom: the phase
#: :math:`\arctan L(E)` of its logarithmic derivative at :math:`r_c` within
#: this many radians over :math:`\varepsilon_{ref} \pm` :data:`PHASE_WINDOW`
#: Hartree.  Removing a ghost with a raised local potential can move a
#: scattering resonance into the valence window instead -- iron's s channel
#: at a 10 Hartree raise is ghost-free and 0.74 rad wrong at +0.25 Hartree.
PHASE_TOLERANCE = 0.05
#: Half-width (Hartree) and spacing of the energy window of that test.
PHASE_WINDOW, PHASE_STEP = 0.5, 0.05
#: A resonance just outside that window is caught by a looser bound over a
#: wider one: gallium's s channel at a 20 Hartree raise is 0.027 rad inside
#: :math:`\pm 0.5` Hartree and 0.94 rad at +0.55.  The bound is loose on
#: purpose -- a d channel balanced out to 3.3 Bohr drifts to 0.07 rad at the
#: far edge without anything being wrong with it.
RESONANCE_WINDOW, RESONANCE_TOLERANCE = 1.0, 0.3
#: A channel without projectors scatters off the local potential alone and
#: is held to this many radians over the highest valence reference
#: :math:`\pm` :data:`PHASE_WINDOW` (:func:`unprojected_scattering_errors`).
#: Measured against crystals (HISTORY.md, 2026-10-06, "K21 diagnosed"):
#: B-F and H 0.02-0.07 and aluminum without a raise 0.12-0.15 give their
#: lattices and bonds; sodium at 0.30-0.38 is 10 % off either way, the
#: shipped aluminum (0.52) 12 % and the shipped sodium (1.22) 40 %.
UNPROJECTED_TOLERANCE = 0.2
#: Upper wave vector and spacing of the Fourier grid of the residual energy.
Q_MAX, Q_STEP = 60.0, 0.1


#: Points of the fine quadrature grid inside ``r_c``.
INNER_POINTS = 801


#: Finest spacing (Bohr) of the tail quadrature of the residual energy.
TAIL_SPACING = 0.00125


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
    from ..basis.radial_backend import numerov_outward_kernel

    h2 = float((r0[1] - r0[0]) ** 2)
    u = np.zeros_like(r0, dtype=float)
    u[start], u[start + 1] = u_start
    return numerov_outward_kernel(np.asarray(f, dtype=float),
                                  np.asarray(s, dtype=float), h2, int(start),
                                  u)


def _numerov_inward(r0: np.ndarray, f: np.ndarray, stop: int,
                    kappa: float) -> np.ndarray:
    """Homogeneous inward integration from a decaying start down to ``stop``."""
    from ..basis.radial_backend import numerov_inward_kernel

    h = r0[1] - r0[0]
    h2 = float(h * h)
    u = np.zeros_like(r0, dtype=float)
    n = r0.size
    u[n - 1] = np.exp(-kappa * r0[n - 1])
    u[n - 2] = np.exp(-kappa * r0[n - 2])
    return numerov_inward_kernel(np.asarray(f, dtype=float), h2, int(stop), u)


def _radial_f(r0: np.ndarray, potential0: np.ndarray, l: int,
              energy: float) -> np.ndarray:
    """``f = 2[V + l(l+1)/2r^2 - E]`` with a finite (unused) value at r = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 2.0 * (potential0 + l * (l + 1) / (2.0 * r0 ** 2) - energy)
    f[0] = 0.0
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def _origin_seed(r0: np.ndarray, l: int, z_eff: float, potential0=None,
                 energy: float = 0.0, order: int = 6, exponent=None):
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
    power = float(l + 1) if exponent is None else float(exponent)

    def series(x):
        return x ** power * sum(ak * x ** k for k, ak in enumerate(a))
    return (series(r0[1]), series(r0[2]))


def _derivative(u: np.ndarray, h: float, index: int) -> float:
    """Fourth-order centered first derivative at one grid index.

    The stencil spans ``index - 2 .. index + 2``, so it needs
    ``2 <= index <= len(u) - 3``.  Outside that it used to raise numpy's
    ``IndexError: index N is out of bounds for axis 0 with size N``, which says
    nothing about *why* an index reached the end of a radial grid -- the real
    cause is always a channel cutoff that landed there, and finding that out
    from the numpy message cost hours once.  So the range is checked here and
    the error names the cause.
    """
    n = len(u)
    if not 2 <= index <= n - 3:
        raise ValueError(
            f"the fourth-order derivative stencil needs a grid index in "
            f"[2, {n - 3}] and was asked for {index} on a {n}-point grid.  An "
            f"index at the edge of a radial grid means the channel cutoff was "
            f"placed there: see `_cutoff_for`, which falls back to "
            f"`rc_factor * (outermost peak of |r R|)` and returns a radius at "
            f"the end of the grid when the reference wave is not localized.")
    return (u[index - 2] - 8 * u[index - 1] + 8 * u[index + 1]
            - u[index + 2]) / (12.0 * h)


def _relativistic_arrays(r0, v0, l, energy, treatment, kappa, z_eff):
    r"""``(f, sqrt(M), seed exponent)`` for the Numerov recursion.

    The integrated variable is :math:`W`, and :math:`P = M^{1/2}W` is the
    large component every caller wants -- see
    :mod:`mandacaru.basis.relativity`.  ``treatment="none"`` returns the
    non-relativistic ``f``, no factor and no exponent, so the relativistic
    path is a strict generalization of the original one.

    The ``r = 0`` node is prepended for the recursion and never used
    (``start >= 1``), so the :math:`1/r^3` in :math:`M''` is harmless there.

    .. note::

       **The seed exponent has to be the relativistic one.**  :math:`M \to
       Z/2c^2r` as :math:`r\to0`, so :math:`M^{-1/2}\sim r^{1/2}` and the
       integrated variable behaves as

       .. math::

           W \sim r^{\gamma + 1/2},
           \qquad \gamma = \sqrt{\kappa^2 - (Z\alpha)^2},

       not as :math:`r^{l+1}`.  For the 2s channel of oxygen that is
       :math:`r^{1.498}` against :math:`r^{1}`, and seeding the wrong power
       put the shooting eigenvalue **9.35 mHa** away from the one the
       self-consistent atom had -- against 1.1e-4 non-relativistically.  It
       surfaced three steps downstream, as a PAW-LCAO ionic potential missing
       :math:`-Z_{ion}/r` by 8e-4 Hartree, because the pseudization and the
       SCF had ended up using different 2s waves.

       The sub-leading terms of :func:`_origin_seed` stay the
       non-relativistic ones.  They carry the screening, a correction of
       relative order :math:`Zr`, which is not where the failure was.
    """
    from ..basis.relativity import _resolve, mass_factor, relativistic_f

    if _resolve(treatment) == "none":
        return _radial_f(r0, v0, l, energy), None, None
    with np.errstate(divide="ignore", invalid="ignore"):
        f = relativistic_f(r0, v0, l, kappa, energy, atomic_number=z_eff,
                           treatment=treatment)
        M, _dM, _d2M = mass_factor(r0, v0, energy, z_eff)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f[0] = 0.0
    M = np.nan_to_num(M, nan=1.0, posinf=1.0, neginf=1.0)
    M[0] = M[1]
    return f, np.sqrt(M), _relativistic_seed(r0, v0, l, kappa, treatment,
                                            z_eff)


def _relativistic_seed(r0, v0, l, kappa, treatment, z_eff):
    r"""The first two nodes of :math:`W`, taken from the tridiagonal solver.

    A power-series seed is the wrong tool here.  The leading exponent is
    :math:`\gamma + 1/2` rather than :math:`l+1`, and the recursion of
    :func:`_origin_seed` -- which carries the screening and is what makes the
    non-relativistic seed good to 1e-8 -- is derived *for* the integer power,
    so substituting the relativistic one into it is worse than leaving it
    alone: oxygen's 2s shooting eigenvalue misses the self-consistent atom by
    9.4 mHa with the integer power and 57 mHa with the fractional one.

    :func:`~mandacaru.basis.relativity.solve_radial_relativistic` needs no
    seed at all -- it is a tridiagonal eigenproblem with :math:`P(0)=0` built
    into the discretization -- so its solution near the origin already has the
    right power *and* the right screening.  Two nodes of it start the Numerov
    recursion, which then refines the whole wave to fourth order.  The
    energy dependence of those two nodes is :math:`O(arepsilon r^2)`, i.e.
    1e-6 of the value at the first grid point, so one seed serves every trial
    energy of the shoot.
    """
    from ..basis.relativity import (_resolve, mass_factor,
                                    solve_radial_relativistic)

    r = r0[1:]
    P, _eps = solve_radial_relativistic(
        r, v0[1:], l, 0, kappa=kappa,
        treatment="dirac" if _resolve(treatment) == "dirac" else "scalar",
        atomic_number=z_eff)
    with np.errstate(divide="ignore", invalid="ignore"):
        M, _dM, _d2M = mass_factor(r, v0[1:], float(_eps), z_eff)
    W = P / np.sqrt(np.maximum(M, 1e-30))
    scale = W[1] if abs(W[1]) > 0 else 1.0
    return (float(W[0] / scale), float(W[1] / scale))


def scattering_wave(r: np.ndarray, potential: np.ndarray, l: int,
                    energy: float, z_eff: float, treatment: str = "none",
                    kappa: int | None = None) -> np.ndarray:
    """Outward Numerov solution ``u(r)`` at ``energy`` (arbitrary scale)."""
    r0 = _with_origin(r)
    v0 = np.concatenate([[0.0], potential])
    f, sqrt_M, power = _relativistic_arrays(r0, v0, l, energy, treatment,
                                            kappa, z_eff)
    seed = (power if power is not None
            else _origin_seed(r0, l, z_eff, v0, energy))
    u = numerov_outward(r0, f, np.zeros_like(r0), seed, start=1)
    if sqrt_M is not None:
        u = u * sqrt_M
    return u[1:]


def bound_state(r: np.ndarray, potential: np.ndarray, l: int,
                energy_guess: float, z_eff: float, window: float = 5e-3,
                treatment: str = "none", kappa: int | None = None):
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
        f, sqrt_M, power = _relativistic_arrays(r0, v0, l, energy,
                                                treatment, kappa, z_eff)
        # Outermost classical turning point of the effective potential, read
        # off the non-relativistic f.  The relativistic one carries V'' in
        # its Darwin term, and a gradient-corrected potential puts grid-scale
        # spikes there: PBE holmium's 4f saw isolated "allowed" points out to
        # 4.2 Bohr, well past its real turning point at 0.84, and matching
        # out there -- where the compact 4f has tunnelled to nothing --
        # never bracketed.  The relativistic shift of a turning point is
        # negligible, and the eigenvalue does not depend on where it matches.
        turning = np.nonzero(_radial_f(r0, v0, l, energy)[10:] < 0)[0]
        match = int(turning[-1]) + 10 if turning.size else r0.size // 2
        match = min(max(match, 20), r0.size - 20)
        seed = (power if power is not None
                else _origin_seed(r0, l, z_eff, v0, energy))
        out = numerov_outward(r0, f, np.zeros_like(r0), seed, start=1)
        decay = np.sqrt(max(-2.0 * energy, 1e-6))
        inn = _numerov_inward(r0, f, match - 3, decay)
        scale = out[match] / inn[match]
        inn = inn * scale
        # The matching condition is on W; multiplying both branches by the
        # same sqrt(M) afterwards leaves the logarithmic-derivative mismatch
        # unchanged, so the eigenvalue is the relativistic one either way.
        if sqrt_M is not None:
            out, inn = out * sqrt_M, inn * sqrt_M
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
                      energy: float, r_cut: float, z_eff: float,
                      treatment: str = "none",
                      kappa: int | None = None) -> float:
    r"""All-electron logarithmic derivative :math:`R'/R` at ``r_cut``."""
    u = scattering_wave(r, potential, l, energy, z_eff, treatment, kappa)
    return _log_derivative_of_u(r, u, r_cut)


def _log_derivative_of_u(r, u, r_cut) -> float:
    """``R'/R = u'/u - 1/r`` at ``r_cut`` from a fourth-order stencil."""
    h = r[1] - r[0]
    index = int(np.argmin(np.abs(r - r_cut)))
    return float(_derivative(u, h, index) / u[index] - 1.0 / r[index])


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
    a one-dimensional root.  When the norm stays finite at the pole and
    below its target (the *hard case*), :func:`_hard_case` gives the
    minimizer.
    """
    K = np.asarray(K, dtype=float)
    G = np.asarray(G, dtype=float)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    c0, *_ = np.linalg.lstsq(A, b, rcond=None)
    _u, sigma, vt = np.linalg.svd(A)
    rank = int(np.sum(sigma > 1e-10 * sigma.max()))
    Z = vt[rank:].T                                   # (n, n - rank)
    if Z.shape[1] == 0:
        return c0
    Kz, Gz = Z.T @ K @ Z, Z.T @ G @ Z
    kz, gz = Z.T @ (K @ c0 + k), Z.T @ (G @ c0)

    # The pencil diagonalized once (V^T Gz V = 1, V^T Kz V = diag(mu)):
    # y(lam) = V (V^T r) / (mu - lam) needs no solve, so it stays exact
    # arbitrarily close to the pole.  A linear solve there did not: the Bessel
    # basis can be nearly redundant on the null space (vanadium's PAW-LCAO d
    # channel: Gz down to 1e-11, Kz - lam Gz at condition 3e16 and an exact
    # zero pivot).
    from scipy.linalg import eigh
    values, vectors = eigh(Kz, Gz)
    mu = float(values[0])

    def solution(lam):
        return vectors @ ((vectors.T @ (lam * gz - kz)) / (values - lam))

    def residual(lam):
        c = c0 + Z @ solution(lam)
        return c @ G @ c - norm

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
        # Extremely close to the pole: shrink towards mu until positive --
        # unless the norm stays finite there (the hard case).
        for _ in range(60):
            nearer = mu - 0.5 * (mu - hi)
            if not hi < nearer < mu:
                break
            hi = nearer
            if residual(hi) > 0:
                break
        if residual(hi) < 0:
            return _hard_case(K, k, G, norm, c0, Z, values, vectors, gz, kz)
    lam = brentq(residual, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500)
    return c0 + Z @ solution(lam)


def _hard_case(K, k, G, norm, c0, Z, values, vectors, gz, kz) -> np.ndarray:
    r""":func:`constrained_minimum` when the norm stays below its target all
    the way to the pole (the right-hand side has no component along the
    lowest eigenvector :math:`v`): :math:`\lambda = \mu`, the other
    components as usual, plus the multiple of :math:`v` that meets the norm
    -- of its two signs, the one with the lower objective."""
    mu = float(values[0])
    coefficients = vectors.T @ (mu * gz - kz)
    coefficients[0] = 0.0
    coefficients[1:] /= values[1:] - mu
    base, direction = c0 + Z @ (vectors @ coefficients), Z @ vectors[:, 0]
    # (base + t direction)^T G (base + t direction) = norm.
    a = float(direction @ G @ direction)
    b = 2.0 * float(direction @ G @ base)
    c = float(base @ G @ base) - norm
    root = np.sqrt(max(b * b - 4.0 * a * c, 0.0))
    candidates = [base + t * direction
                  for t in ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a))]
    return min(candidates, key=lambda x: float(x @ K @ x + 2.0 * k @ x))


def check_reference_atom(atom, xc: str, relativity: str) -> None:
    """Refuse an all-electron ``atom`` that did not converge, or that was
    solved differently from the dataset being asked for.

    An unconverged atom is not a reference: PBE holmium stopped at
    ``converged=False`` with its 4f at -0.075 Hartree spread out to 5.7 Bohr
    (converged: -0.103), and the generator failed much later, on a bound
    state that the wrong potential did not hold.

    The generator records ``xc`` and ``relativity`` from its arguments, and it
    unscreens with ``xc``; an atom solved otherwise would give a dataset whose
    record is false and, for a different functional, whose ionic potential is
    wrong (a PBE-screened potential unscreened with LDA).
    """
    if not getattr(atom, "converged", True):
        raise RuntimeError(
            f"the all-electron reference atom Z={atom.atomic_number} "
            f"(xc={atom.xc!r}, relativity={atom.relativity!r}) did not "
            f"converge in {atom.iterations} iterations; a dataset built on "
            "it would pseudize the wrong orbitals")
    solved = (str(atom.xc).lower(), str(atom.relativity).lower())
    wanted = (str(xc).lower(), str(relativity).lower())
    if solved != wanted:
        raise ValueError(
            f"the atom passed in was solved with xc={solved[0]!r}, "
            f"relativity={solved[1]!r}, but the dataset asks for "
            f"xc={wanted[0]!r}, relativity={wanted[1]!r}; pass matching "
            "options or let the generator solve the atom")


class GhostStateError(RuntimeError):
    """A generated channel binds a state below its reference energy."""


class GhostStateWarning(UserWarning):
    """A dataset in use carries a ghost state (or wrong scattering) that its
    generator could not remove."""


def defects_record(defects) -> dict:
    """``defects`` as a file stores it (string keys, lists)."""
    defects = defects or {}
    record = {"ghosts": {str(l): float(e)
                         for l, e in defects.get("ghosts", {}).items()},
              "phases": {str(l): [float(a), float(b)]
                         for l, (a, b) in defects.get("phases", {}).items()}}
    if defects.get("residuals"):
        record["residuals"] = {str(l): float(e)
                               for l, e in defects["residuals"].items()}
    if defects.get("s_miss") is not None:
        record["s_miss"] = float(defects["s_miss"])
    if defects.get("unprojected"):
        record["unprojected"] = {str(l): float(e)
                                 for l, e in defects["unprojected"].items()}
    return record


def read_defects(record) -> dict:
    """The ``defects`` of a stored record; ``{}`` when there are none."""
    record = record or {}
    ghosts = {int(l): float(e) for l, e in (record.get("ghosts") or {}).items()}
    phases = {int(l): (float(a), float(b))
              for l, (a, b) in (record.get("phases") or {}).items()}
    residuals = {int(l): float(e) for l, e in
                 (record.get("residuals") or {}).items()}
    s_miss = record.get("s_miss")
    unprojected = {int(l): float(e) for l, e in
                   (record.get("unprojected") or {}).items()}
    if (not ghosts and not phases and not residuals and s_miss is None
            and not unprojected):
        return {}
    result = {"ghosts": ghosts, "phases": phases}
    if residuals:
        result["residuals"] = residuals
    if s_miss is not None:
        result["s_miss"] = float(s_miss)
    if unprojected:
        result["unprojected"] = unprojected
    return result


def warn_defects(dataset, family: str) -> None:
    """Raise the :class:`GhostStateWarning` of a dataset that has defects,
    on every load -- the library, a path or a user's own directory."""
    import warnings

    if getattr(dataset, "defects", None):
        warnings.warn(defect_message(dataset.symbol, family, dataset.defects),
                      GhostStateWarning, stacklevel=3)


def defect_message(symbol: str, family: str, defects: dict) -> str:
    """The warning text for a dataset's recorded ``defects``."""
    parts = [f"l={l} ghost {float(e):+.3g} Ha"
             for l, e in sorted(defects.get("ghosts", {}).items())]
    parts += [f"l={l} phase error {float(near):.2f}/{float(far):.2f} rad"
              for l, (near, far) in sorted(defects.get("phases", {}).items())]
    parts += [f"l={l} residual kinetic energy {float(e):.3g} Ha"
              for l, e in sorted(defects.get("residuals", {}).items())]
    if defects.get("s_miss") is not None:
        parts.append(f"s projectors miss an intruding hydrogen 1s by "
                     f"{float(defects['s_miss']):.2f} of its norm")
    parts += [f"l={l} (no projectors) phase error {float(e):.2f} rad"
              for l, e in sorted(defects.get("unprojected", {}).items())]
    if defects.get("ghosts"):
        what = "ghost states"
        consequence = (f"A variational calculation containing {symbol} can "
                       f"collapse into a spurious state, so its energies and "
                       f"forces are not reliable.")
    elif defects.get("s_miss") is not None and not defects.get("phases"):
        what = "an incomplete s channel"
        consequence = (f"A neighbor's orbital entering the {symbol} sphere is "
                       f"not represented, so bonds to {symbol} can be too "
                       f"long or collapse.")
    elif defects.get("unprojected") and not defects.get("phases"):
        what = "a local potential that scatters wrongly"
        consequence = (f"Angular momenta without projectors see the local "
                       f"potential alone, so bonds and lattice constants of "
                       f"{symbol} compounds can be several percent off.")
    elif defects.get("residuals"):
        what = "divergent projector residuals"
        consequence = (f"Its projectors are numerically unreliable, so "
                       f"energies and forces of systems containing "
                       f"{symbol} carry an error of unknown size.")
    else:
        what = "scattering errors"
        consequence = (f"Its channels do not scatter like the all-electron "
                       f"atom, so energies and forces of systems containing "
                       f"{symbol} carry an error of unknown size.")
    return (f"the {family} dataset for {symbol} has {what} "
            f"({'; '.join(parts)}).  {consequence}")


def local_potential_ghosts(pp) -> dict:
    """``{l: eps_ps - eps_ae}`` for every channel *without* projectors
    (``l`` up to one above the highest channel) in which the screened local
    potential alone binds more states than the all-electron atom has valence
    states of that ``l``.

    Such a channel is governed by ``v_local_screened`` only, so the spectral
    test of :func:`ghost_errors` on the constructed channels never sees it
    -- iron's PAW-LCAO ``p`` channel held a level at -4.63 Ha against a 4p at
    -0.05 Ha while every constructed channel was clean.  Both spectra are
    counted below zero in the same box, on the reference atom's uniform grid;
    the all-electron count less that ``l``'s core shells is the number of
    states the local potential may bind.  Needs the all-electron atom (a
    dataset read from a library has none, and gives ``{}``).
    """
    from scipy.linalg import eigvalsh_tridiagonal

    if getattr(pp, "atom", None) is None:
        return {}
    r = np.asarray(pp.atom.r, dtype=float)
    h = float(r[1] - r[0])
    _valence, core = _valence_configuration(
        int(pp.atomic_number), configuration=pp.atom.occupations)
    core = dict(core)
    for orbital in getattr(pp, "frozen_subshells", ()):
        core[orbital] = pp.atom.occupations[orbital]
    for orbital in getattr(pp, "semicore_subshells", ()):
        core.pop(orbital, None)
    off = np.full(r.size - 1, -0.5 / h ** 2)

    def bound(v, l):
        diag = 1.0 / h ** 2 + v + l * (l + 1) / (2.0 * r * r)
        return eigvalsh_tridiagonal(diag, off, select="v",
                                    select_range=(float(v.min()) - 1.0, 0.0))

    out = {}
    for l in range(max(pp.channels) + 2):
        if l in pp.channels:
            continue
        smooth = bound(np.interp(r, pp.r, pp.v_local_screened), l)
        ae = bound(np.asarray(pp.atom.v_effective, dtype=float), l)
        allowed = ae[sum(1 for (_n, lc) in core if lc == l):]
        if smooth.size > allowed.size:
            reference = float(allowed[0]) if allowed.size else 0.0
            out[int(l)] = float(smooth[0]) - reference
    return out


def ghost_errors(pp, levels) -> dict:
    """``{l: eps_0 - eps_ref}`` for every channel holding a ghost state.

    ``levels(pp, l)`` returns the lowest eigenvalues of the channel's
    pseudo Hamiltonian (two, or three for a semicore channel).  A ghost is an *extra* state: the lowest level lies
    more than :data:`GHOST_TOLERANCE` below the reference and the second one
    is closer to the reference than the first. For frozen-core scattering-only
    channels, any bound pseudo level is an extra state and therefore a ghost.
    """
    out = {}
    for l, channel in pp.channels.items():
        reference = float(channel.reference_energies[0])
        if reference >= 0.0:
            if getattr(pp, "frozen_subshells", ()):
                first = float(levels(pp, l)[0])
                if first < -GHOST_TOLERANCE:
                    out[int(l)] = first
            continue
        spectrum = [float(e) for e in levels(pp, l)]
        first, second = spectrum[:2]
        if (first - reference < -GHOST_TOLERANCE
                and abs(second - reference) < abs(first - reference)):
            out[int(l)] = first - reference
        # A semicore channel's second bound reference (sodium's 3s) must be
        # its second level: an extra state between the two is a ghost too.
        occupied = _occupied_references(channel)
        if len(occupied) > 1 and len(spectrum) > 2:
            upper, third = spectrum[1], spectrum[2]
            if (upper - occupied[1] < -GHOST_TOLERANCE
                    and abs(third - occupied[1]) < abs(upper - occupied[1])):
                out[int(l)] = min(out.get(int(l), 0.0), upper - occupied[1])
    # Channels without projectors (PAW-LCAO): the local potential alone.
    unconstructed = getattr(pp, "unconstructed_ghosts", None)
    if unconstructed is not None:
        out.update(unconstructed())
    return out


def scattering_errors(pp, log_derivative,
                      ae_cache: dict[tuple, float] | None = None) -> dict:
    r"""``{l: (near, far)}``: the largest phase error
    :math:`|\arctan L_{ps} - \arctan L_{ae}|` at each channel's
    :math:`r_c`, within :math:`\varepsilon_{ref} \pm` :data:`PHASE_WINDOW`
    (``near``) and :data:`RESONANCE_WINDOW` (``far``) Hartree, wrapped
    modulo :math:`\pi` so a pole of :math:`L` is not an error.  Needs the
    all-electron atom on ``pp``. Scattering-only channels representing a
    frozen subshell are evaluated at positive energies in the same windows.
    ``ae_cache`` reuses all-electron
    logarithmic derivatives at the same channel, energy and matching radius across
    ghost-repair attempts; it is local to one reference atom.
    """
    from ..basis.relativity import _resolve as _resolve_relativity

    treatment = _resolve_relativity(getattr(pp, "relativity", "none"))
    # A Dirac dataset's ordinary channels are the j average -- PAW-LCAO's
    # partial waves solve the scalar (kappa = -1) equation outright -- and
    # its spin-orbit part is a separate term this check does not see.  So
    # its phases are compared with the scalar all-electron wave; a Dirac
    # log-derivative would need a kappa this per-l check does not have.
    if treatment == "dirac":
        treatment = "scalar"
    window = PHASE_WINDOW
    # Compare where the smooth wave obeys the all-electron equation again:
    # past the projectors, which for PAW-LCAO can reach beyond r_cut.
    radius = getattr(pp, "projector_radius", None)
    out = {}
    for l, channel in pp.channels.items():
        reference = float(channel.reference_energies[0])
        if reference >= 0.0 and not getattr(pp, "frozen_subshells", ()):
            continue
        r_match = float(radius(l)) if radius is not None else channel.r_cut
        near = far = 0.0
        # Around every occupied reference: a semicore channel's valence
        # state (sodium's 3s) sits 2 Ha above its first reference.
        centers = [reference] + _occupied_references(channel)[1:]
        for center, offset in ((c, o) for c in centers for o in np.arange(
                -RESONANCE_WINDOW, RESONANCE_WINDOW + 0.5 * PHASE_STEP,
                PHASE_STEP)):
            energy = center + offset
            if reference >= 0.0 and energy <= 0.0:
                continue
            key = (id(pp.atom), l, float(energy), r_match, treatment)
            if ae_cache is not None and key in ae_cache:
                l_ae = ae_cache[key]
            else:
                l_ae = log_derivative_ae(pp.r, pp.atom.v_effective, l, energy,
                                         r_match, float(pp.atomic_number),
                                         treatment)
                if ae_cache is not None:
                    ae_cache[key] = l_ae
            l_ps = log_derivative(pp, l, energy, r_match)
            d = np.arctan(l_ps) - np.arctan(l_ae)
            error = abs((d + 0.5 * np.pi) % np.pi - 0.5 * np.pi)
            far = max(far, error)
            if abs(offset) <= window + 1e-9:
                near = max(near, error)
        out[int(l)] = (float(near), float(far))
    return out


def _occupied_references(channel) -> list[float]:
    """The reference energies of a channel's occupied partial waves, first
    one always included (a library channel without per-wave occupations
    has one bound reference)."""
    energies = [float(e) for e in channel.reference_energies]
    occupations = getattr(channel, "occupations", None) or [1.0]
    return [energies[0]] + [e for e, o in zip(energies[1:], occupations[1:])
                            if o > 0.0]


def unprojected_scattering_errors(pp, log_derivative,
                                  ae_cache: dict[tuple, float] | None = None
                                  ) -> dict:
    r"""``{l: error}``: the largest phase error of every channel *without*
    projectors (``l`` up to one above the highest constructed channel),
    which scatters off the screened local potential alone, within
    :data:`PHASE_WINDOW` of the highest valence reference energy.

    The phases are compared past the local potential's radius and every
    channel's cutoff, where both waves obey the same equation again.  Needs
    the all-electron atom on ``pp`` (a dataset read from a library has none,
    and gives ``{}``).  This is the test the constructed channels' own
    (:func:`scattering_errors`) never made: aluminum repaired with a 5 Ha
    local shift scattered d 0.84 rad off, and that alone put fcc Al at
    4.48 Angstrom (HISTORY.md, 2026-10-06, "K21 diagnosed").
    """
    from ..basis.relativity import _resolve as _resolve_relativity

    if getattr(pp, "atom", None) is None:
        return {}
    treatment = _resolve_relativity(getattr(pp, "relativity", "none"))
    if treatment == "dirac":
        treatment = "scalar"
    bound = [e for channel in pp.channels.values()
             for e in _occupied_references(channel) if e < 0.0]
    if not bound:
        return {}
    reference = max(bound)
    r_match = max([float(getattr(pp, "r_cut_local", 0.0))]
                  + [float(channel.r_cut) for channel in pp.channels.values()])
    out = {}
    for l in range(max(pp.channels) + 2):
        if l in pp.channels:
            continue
        worst = 0.0
        for offset in np.arange(-PHASE_WINDOW, PHASE_WINDOW + 0.5 * PHASE_STEP,
                                PHASE_STEP):
            energy = reference + offset
            key = (id(pp.atom), l, float(energy), r_match, treatment)
            if ae_cache is not None and key in ae_cache:
                l_ae = ae_cache[key]
            else:
                l_ae = log_derivative_ae(pp.r, pp.atom.v_effective, l, energy,
                                         r_match, float(pp.atomic_number),
                                         treatment)
                if ae_cache is not None:
                    ae_cache[key] = l_ae
            d = np.arctan(log_derivative(pp, l, energy, r_match)) - np.arctan(l_ae)
            worst = max(worst, abs((d + 0.5 * np.pi) % np.pi - 0.5 * np.pi))
        out[int(l)] = float(worst)
    return out


def _defect_badness(candidate: tuple) -> tuple[int, float, int]:
    """Rank one failed construction, preferring one without a ghost.

    Among ghosted candidates the shallowest deepest ghost wins, then the one
    with fewer affected channels.  Without a ghost, what counts is how far the
    worst defect exceeds its own tolerance: a phase error against
    :data:`PHASE_TOLERANCE` near the reference (:data:`RESONANCE_TOLERANCE`
    farther out), an unprojected channel's against
    :data:`UNPROJECTED_TOLERANCE`, an acceptance defect (the PAW-LCAO
    intruding-1s miss) against 1.  A construction whose acceptance failed
    ranks below every one that only scatters wrongly; among those that
    failed it, the size still counts (Gd-LDA was once kept at a miss of 23.8
    over one of 1.4 with a 0.09 rad phase).  The same ranking is used while
    searching and when returning a flagged dataset.
    """
    _pp, ghosts, wrong = candidate[:3]
    extra = candidate[3] if len(candidate) > 3 else {}
    if ghosts:
        return (2, -min(ghosts.values()), len(ghosts))
    phase = max((max(near / PHASE_TOLERANCE, far / RESONANCE_TOLERANCE)
                 for near, far in wrong.values()), default=0.0)
    unprojected = max(extra.get("unprojected", {}).values(), default=0.0)
    # A failed intruding-1s miss is its own tier, above every scattering
    # defect: it can collapse a bond (PbO, CoH, SnH), where a wrongly
    # scattering channel costs percent-level accuracy.  Ranked together,
    # the first lda-sr rebuild with the unprojected check traded shipped,
    # clean U, Ta, Re and Hf for misses of 1.05-2.14 (HISTORY.md,
    # 2026-10-07).
    tier = 1 if "s_miss" in extra else 0
    return (tier, max(phase, float(extra.get("s_miss", 0.0)),
                      unprojected / UNPROJECTED_TOLERANCE), 0)


def _describe_defects(extra: dict) -> str:
    """The family and unprojected-channel defects, for messages."""
    parts = [f"l={l} unprojected phase error {e:.2f} rad"
             for l, e in sorted(extra.get("unprojected", {}).items())]
    if extra.get("s_miss") is not None:
        parts.append(f"intruding-1s miss {extra['s_miss']:.2f}")
    return ", ".join(parts)


def _least_defective(candidates: list[tuple]):
    """Return the least defective attempt with its diagnostic record set."""
    best = min(candidates, key=_defect_badness)
    pp, ghosts, wrong = best[:3]
    extra = best[3] if len(best) > 3 else {}
    pp.defects = {"ghosts": {int(l): float(e) for l, e in ghosts.items()},
                  "phases": {int(l): (float(a), float(b))
                             for l, (a, b) in wrong.items()}}
    pp.defects.update(extra)
    return pp


def _wrong_phases(phases: dict) -> dict:
    """The channels of :func:`scattering_errors` outside tolerance."""
    return {l: (near, far) for l, (near, far) in phases.items()
            if near > PHASE_TOLERANCE or far > RESONANCE_TOLERANCE}


def _describe(errors: dict) -> str:
    return ", ".join(f"l={l} {e:+.3g} Ha" for l, e in sorted(errors.items()))


def ghost_free(generate, levels, log_derivative, symbol: str, options: dict,
               mode: str, overrides=None, acceptance=None):
    r"""``generate(symbol, **options)``, rebuilt until no channel holds a ghost.

    A deep local well can bind an extra level that the nonlocal projectors do
    not lift. The local radius now follows the *largest* channel cutoff, and
    each projector continues through that radius; the older minimum-cutoff
    construction produced widespread ghosts. A phase error can remain even
    when the bound spectrum is clean.

    **What is done about it.**  ``overrides`` go into every attempt, and when
    there are any they are first tried alone with the construction otherwise
    unchanged -- PAW-LCAO passes ``norm_deficit=0``, which is all iron needs.
    Next the local radius is shortened by each of
    :data:`SHORTER_LOCAL_FACTORS` with no raise, then the local potential is
    raised by each of :data:`OWN_CUTOFF_SHIFTS` with the cutoffs untouched.  Finally every channel is given the largest
    cutoff and the local potential is raised by each of
    :data:`GHOST_REMEDY_SHIFTS` in turn (Hamann's ``dvloc0``).  A construction is
    accepted when it has no ghost **and** every bound channel scatters like the
    atom -- to :data:`PHASE_TOLERANCE` near its reference and
    :data:`RESONANCE_TOLERANCE` farther out (:func:`scattering_errors`), and
    every angular momentum *without* projectors scatters like it to
    :data:`UNPROJECTED_TOLERANCE` (:func:`unprojected_scattering_errors`); a
    raise that trades the ghost for a misplaced resonance is not a repair.  The
    self-consistent atom is solved once and shared by every attempt, and a
    dataset that was clean to begin with is returned exactly as before.

    ``acceptance(pp)`` -- a family's own check of an otherwise clean
    construction -- returns ``{}`` or the defects it found (PAW-LCAO: the
    intruding-1s miss of its s channel, ``{"s_miss": value}``).  A failed
    acceptance is repaired like a phase error, every repair has to pass it
    too, and the search then also tries 9 and 10 Bessel functions with each
    local shift, and the s cutoff at 0.95-0.85 of its value.

    A caller that fixed ``r_cut``, ``r_cut_local`` or ``local_shift`` has
    made the choice this search would make, so a ghost there is refused
    rather than overridden; so is one with ``mode="refuse"``, and one no
    remedy removes.  ``mode="keep"`` returns the first construction as is.
    """
    if mode not in GHOST_MODES:
        raise ValueError(f"ghosts must be one of {GHOST_MODES}, not {mode!r}")
    first = generate(symbol, **options, ghosts="keep")
    if mode == "keep":
        return first
    phase_cache: dict[tuple, float] = {}

    def defects(pp) -> dict:
        """The family's acceptance defects and the unprojected channels
        that scatter wrongly."""
        found = {} if acceptance is None else dict(acceptance(pp))
        loose = {l: e for l, e in unprojected_scattering_errors(
            pp, log_derivative, phase_cache).items()
            if e > UNPROJECTED_TOLERANCE}
        if loose:
            found["unprojected"] = loose
        return found

    errors = ghost_errors(first, levels)
    if not errors:
        # No ghost is not enough: the first construction has to scatter like
        # the atom too, or it is repaired like a ghosted one.  (Aluminum's p
        # channel was once returned 0.95 rad off, untested.)
        wrong = _wrong_phases(scattering_errors(first, log_derivative,
                                                phase_cache))
        extra = defects(first)
        if not wrong and not extra:
            return first
    else:
        wrong, extra = {}, {}
    if errors:
        problem = f"ghost state below the reference ({_describe(errors)})"
    elif wrong:
        problem = "wrong scattering (" + ", ".join(
            f"l={l} {near:.3f}/{far:.3f} rad"
            for l, (near, far) in sorted(wrong.items())) + ")"
    else:
        problem = f"an incomplete construction ({_describe_defects(extra)})"
    pinned = [name for name in ("r_cut", "r_cut_local", "local_shift")
              if options.get(name) is not None]
    if pinned and not errors and not wrong and mode != "refuse":
        # Only the family's acceptance failed, and the caller chose the
        # parameters a repair would change: keep the construction and
        # record what it misses, as a flagged dataset would.
        return _least_defective([(first, {}, {}, extra)])
    if mode == "refuse" or pinned:
        why = (f"with {', '.join(pinned)} fixed by the caller" if pinned
               else "and ghosts='refuse'")
        raise GhostStateError(
            f"{symbol}: {problem} "
            f"{why}.  Leave the cutoffs and the local potential to the "
            f"generator, or pass ghosts='keep' to study it.")

    radius = max(float(channel.r_cut) for channel in first.channels.values())
    balanced = {"r_cut": {int(l): radius for l in first.channels},
                "r_cut_local": float(options["local_factor"]) * radius}
    overrides = dict(overrides or {})
    attempts = ([(", ".join(f"{k}={v:g}" for k, v in overrides.items()),
                  overrides)] if overrides else [])
    attempts += [(f"local radius x {factor:g}, no shift",
                  dict(overrides, local_shift=0.0,
                       local_factor=float(options["local_factor"]) * factor))
                 for factor in SHORTER_LOCAL_FACTORS]
    attempts += [(f"own cutoffs, shift {shift:g}",
                  dict(overrides, local_shift=float(shift)))
                 for shift in OWN_CUTOFF_SHIFTS]
    if acceptance is not None and "n_bessel" in options:
        # Whatever failed first, a repair must also pass the acceptance, and
        # the s channel's completeness grows with its Bessel functions; the
        # scans that repaired the d block used 9 and 10 (HISTORY.md,
        # 2026-09-28).  Mg-LDA misses by 1.5 at 8 and by 0.07 at 10.
        attempts += [(f"{count} Bessel functions, shift {shift:g}",
                      dict(overrides, n_bessel=count,
                           local_shift=float(shift)))
                     for count in (9, 10) for shift in (0.0,) + OWN_CUTOFF_SHIFTS]
        # Then a shorter s sphere, the other channels as built (Tl, Po, Bi,
        # Os, Re, Ta and V were repaired at 0.85-0.95 of their s cutoff).
        for factor in (0.95, 0.9, 0.85):
            shorter = {int(l): float(channel.r_cut) * (factor if l == 0
                                                         else 1.0)
                       for l, channel in first.channels.items()}
            attempts += [(f"s cutoff x {factor:g}, 10 Bessel functions, "
                          f"shift {shift:g}",
                          dict(overrides, r_cut=shorter, n_bessel=10,
                               local_shift=float(shift)))
                         for shift in (0.0, 10.0, 20.0)]
    attempts += [(f"balanced at {radius:.3f} Bohr, shift {shift:g}",
                  dict(overrides, **balanced, local_shift=float(shift)))
                 for shift in GHOST_REMEDY_SHIFTS]
    tried = []
    # Keep only the best failed construction. A heavy dataset contains
    # many full-grid arrays; retaining every trial until the search ends can
    # exhaust memory when several elements are built in parallel.
    best = ((first, errors, {} if errors else wrong, extra)
            if mode == "flag" else None)
    for label, remedy in attempts:
        trial = dict(options, atom=first.atom, **remedy)
        try:
            pp = generate(symbol, **trial, ghosts="keep")
        except (ValueError, RuntimeError) as error:
            tried.append(f"{label}: {type(error).__name__}")
            continue
        remaining = ghost_errors(pp, levels)
        if remaining:
            tried.append(f"{label}: ghost {_describe(remaining)}")
            candidate = (pp, remaining, {})
            if best is not None and _defect_badness(candidate) < \
                    _defect_badness(best):
                best = candidate
            continue
        wrong = _wrong_phases(scattering_errors(pp, log_derivative,
                                                phase_cache))
        if wrong:
            candidate = (pp, {}, wrong, defects(pp))
            if best is not None and _defect_badness(candidate) < \
                    _defect_badness(best):
                best = candidate
            tried.append(f"{label}: phase " + ", ".join(
                f"l={l} {near:.3f}/{far:.3f} rad"
                for l, (near, far) in sorted(wrong.items())))
            continue
        found = defects(pp)
        if found:
            candidate = (pp, {}, {}, found)
            if best is not None and _defect_badness(candidate) < \
                    _defect_badness(best):
                best = candidate
            tried.append(f"{label}: {_describe_defects(found)}")
            continue
        return pp
    if mode == "flag":
        return _least_defective([best])
    raise GhostStateError(
        f"{symbol}: {problem} and "
        f"no remedy removed it while keeping the scattering "
        f"({'; '.join(tried)}).")


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
    norms: np.ndarray              # all-electron inner *overlap* matrix
    achieved: np.ndarray           # pseudo inner norm matrix
    q_cut: float
    #: What the norm condition actually imposed.  Equal to :attr:`norms`
    #: non-relativistically; the Wronskian form of :func:`norm_targets`
    #: otherwise.  The two are kept apart because they play different roles:
    #: :attr:`norms` is the true all-electron overlap, which is what PAW-LCAO's
    #: overlap correction ``q = norms - achieved`` has to reconstruct, while
    #: :attr:`targets` is the constraint that makes the Vanderbilt matrix
    #: symmetric.  Conflating them leaves ``D^scr - (dT + dV)`` nonzero by
    #: their difference -- 2.2e-3 Ha for oxygen, against a 1e-10 tolerance.
    targets: np.ndarray = None

    @property
    def norm_matrix_error(self) -> float:
        """How far the smooth waves fell from the condition imposed on them."""
        reference = self.norms if self.targets is None else self.targets
        return float(np.max(np.abs(self.achieved - reference)))

    @property
    def relativistic_norm_shift(self) -> float:
        """``max |targets - norms|`` -- the O(c^-2) Wronskian correction."""
        if self.targets is None:
            return 0.0
        return float(np.max(np.abs(self.targets - self.norms)))


def matching_targets(r: np.ndarray, u: np.ndarray, potential: np.ndarray,
                     l: int, energy: float, r_cut: float,
                     treatment: str = "none", kappa: int | None = None,
                     z_eff: float = 0.0) -> np.ndarray:
    r"""``[R, R', R'', R''']`` of a partial wave at ``r_cut`` (a grid point).

    ``R'`` comes from a fourth-order stencil on ``u = rR``; the second and
    third derivatives follow from the radial equation the wave actually
    satisfies, so the targets are consistent with it to the accuracy of the
    Numerov solution itself (a polynomial fit of the tabulated wave would
    leave ~1e-6 inconsistencies that surface as an asymmetric :math:`B`).

    In general that equation is :math:`u'' = a\,u + b\,u'` with

    .. math::

        a = \frac{l(l+1)}{r^2} + \frac{\kappa M'}{Mr} + 2M(V-\varepsilon),
        \qquad b = \frac{M'}{M} ,

    and the third derivative is :math:`a'u + au' + b'u' + bu''`.
    Non-relativistically :math:`M\equiv1` kills :math:`b` and leaves
    :math:`a = 2[V + l(l+1)/2r^2 - \varepsilon]`, which is the form this
    function had before relativity was an option -- so the ``"none"`` path is
    the same arithmetic it always was.
    """
    h = r[1] - r[0]
    k = int(np.argmin(np.abs(r - r_cut)))
    rk, uk = r[k], u[k]
    up = _derivative(u, h, k)
    centrifugal = l * (l + 1) / (r * r)

    from ..basis.relativity import SCALAR_KAPPA, _resolve, mass_factor

    if _resolve(treatment) == "none":
        a = centrifugal + 2.0 * (potential - energy)
        b = np.zeros_like(r)
    else:
        kk = SCALAR_KAPPA if _resolve(treatment) == "scalar" else int(kappa)
        M, dM, _d2M = mass_factor(r, potential, energy, z_eff)
        # The spin-orbit half of the closed-form combination; the Darwin half
        # belongs to W, not to P, so only kappa M'/(Mr) appears here.
        a = centrifugal + kk * dM / (M * r) + 2.0 * M * (potential - energy)
        b = dM / M
    ap = _derivative(a, h, k)
    bp = _derivative(b, h, k)
    upp = a[k] * uk + b[k] * up
    uppp = ap * uk + a[k] * up + bp * up + b[k] * upp
    return np.array([
        uk / rk,
        up / rk - uk / rk ** 2,
        upp / rk - 2.0 * up / rk ** 2 + 2.0 * uk / rk ** 3,
        uppp / rk - 3.0 * upp / rk ** 2 + 6.0 * up / rk ** 3 - 6.0 * uk / rk ** 4,
    ])


def _pseudo_waves_record(l, r_cut, energies, waves, wavevectors, coefficients,
                         residuals, norms, pseudo_in, r_in, q_cut,
                         targets=None) -> PseudoWaves:
    """The :class:`PseudoWaves` of a finished optimization, with the inner
    norms the pseudo waves actually achieved (shared with the PAW-LCAO family)."""
    achieved = np.array([[simpson(a * b * r_in * r_in, x=r_in) for b in pseudo_in]
                         for a in pseudo_in])
    return PseudoWaves(l=l, r_cut=float(r_cut), energies=list(energies),
                       waves=list(waves), wavevectors=wavevectors,
                       coefficients=coefficients, residual_kinetic=residuals,
                       norms=norms, achieved=achieved, q_cut=float(q_cut),
                       targets=(norms if targets is None else targets))


def inner_overlaps(r, waves, r_cut):
    """``<phi_i|phi_j>`` inside ``r_cut``, on the refined inner grid.

    The *true* all-electron overlap, whatever equation the waves solve.  PAW-LCAO's
    overlap correction is built from this; the norm **condition** imposed on
    the smooth waves is :func:`norm_targets`, which differs from it at
    ``O(c^-2)`` once the waves are relativistic.
    """
    r_in = _inner_grid(r_cut)
    inside = [_resample(np.asarray(r, dtype=float), w, r_in) for w in waves]
    return np.array([[simpson(a * b * r_in * r_in, x=r_in) for b in inside]
                     for a in inside])


def norm_targets(r, waves, energies, r_cut, v_ae=None, treatment="none",
                 kappa=None, z_eff=0.0):
    r"""The generalized-norm matrix the pseudo waves must reproduce.

    Non-relativistically this is just
    :math:`\langle\varphi_i|\varphi_j\rangle_{r<r_c}`.  It is **not** that
    when the all-electron waves solve a relativistic equation and the pseudo
    waves -- Bessel expansions, built to be used in a Schrodinger calculation
    -- solve a non-relativistic one.  What the condition has to enforce is
    that the *pseudo* system reproduce the all-electron logarithmic derivative
    **and its energy derivative** at :math:`r_c`, and the quantity that does
    that is a Wronskian, not an overlap:

    .. math::

        \langle\tilde\varphi_i|\tilde\varphi_j\rangle_{r<r_c}
            = -\frac{W_{ij}(r_c)}{2(\varepsilon_j - \varepsilon_i)},
        \qquad W_{ij} = u_i u_j' - u_j u_i' .

    Integrating the relativistic radial equation gives
    :math:`W_{ij}(r_c)/M(r_c) = -2(\varepsilon_j-\varepsilon_i)
    \int_0^{r_c} u_iu_j\,dr` in the non-relativistic limit :math:`M\to1`, so
    the two agree exactly there -- and differ by :math:`O(c^{-2})` otherwise.
    That difference is the whole of the asymmetry the Vanderbilt matrix shows
    when a scalar-relativistic atom is pseudized with the non-relativistic
    condition: 1.4e-4 Ha for oxygen, against a 1e-5 tolerance.

    The **diagonal** carries the energy derivative rather than a difference of
    two energies, and its relativistic form is
    :math:`M(r_c)\int(2M-1)u^2/M\,dr`; it does not enter the symmetry of
    :math:`B`, only the transferability the norm was conserved for.
    """
    from ..basis.relativity import _resolve, mass_factor

    r = np.asarray(r, dtype=float)
    r_in = _inner_grid(r_cut)
    waves_in = [_resample(r, w, r_in) for w in waves]
    # The base matrix keeps the refined-grid Simpson quadrature the
    # non-relativistic construction has always used, so `treatment="none"`
    # returns the same numbers to the last bit.
    targets = np.array([[simpson(a * b * r_in * r_in, x=r_in)
                         for b in waves_in] for a in waves_in])
    if _resolve(treatment) == "none":
        return targets

    h = float(r[1] - r[0])
    k = int(np.argmin(np.abs(r - r_cut)))
    us = [np.asarray(w, dtype=float) * r for w in waves]
    M_in = [_resample(r, mass_factor(r, v_ae, e, z_eff)[0], r_in)
            for e in energies]
    for i in range(len(us)):
        M_at_cut = float(mass_factor(r, v_ae, energies[i], z_eff)[0][k])
        # Diagonal: the relativistic weight of the energy-derivative identity.
        targets[i, i] = M_at_cut * simpson(
            (2.0 * M_in[i] - 1.0) / M_in[i] * waves_in[i] ** 2
            * r_in * r_in, x=r_in)
        for j in range(len(us)):
            gap = energies[j] - energies[i]
            if i == j or abs(gap) < 1e-12:
                continue
            wronskian = (us[i][k] * _derivative(us[j], h, k)
                         - us[j][k] * _derivative(us[i], h, k))
            targets[i, j] = -wronskian / (2.0 * gap)
    return targets


def optimize_pseudo_waves(r: np.ndarray, v_ae: np.ndarray, l: int,
                          waves: list, energies: list, r_cut: float,
                          q_cut: float = DEFAULT_Q_CUT,
                          n_bessel: int = DEFAULT_N_BESSEL,
                          norm_factor: float = 1.0,
                          treatment: str = "none",
                          kappa: int | None = None,
                          z_eff: float = 0.0) -> PseudoWaves:
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
    norm conservation; the PAW-LCAO family uses :math:`f < 1` so its overlap
    correction :math:`(1-f)\langle\varphi_i|\varphi_j\rangle` is positive
    definite by construction.
    """
    r_cut = _snap(r, r_cut)
    r_in = _inner_grid(r_cut)
    q_grid = np.arange(0.0, Q_MAX + 0.5 * Q_STEP, Q_STEP)
    weight = 0.5 * q_grid ** 4 * (q_grid >= q_cut)

    targets = norm_targets(r, waves, energies, r_cut, v_ae, treatment, kappa,
                           z_eff)
    norms = inner_overlaps(r, waves, r_cut)

    coefficients, wavevectors, pseudo_in, residuals = [], [], [], []
    for i, (wave, energy) in enumerate(zip(waves, energies)):
        # Value and first three derivatives of R at r_c (Hamann's ncon = 4).
        target = matching_targets(r, wave * r, v_ae, l, energy, r_cut,
                                  treatment, kappa, z_eff)
        qs = bessel_wavevectors(l, r_cut, n_bessel)
        j, dj, d2j, d3j = bessel_derivatives(l, qs * r_cut)
        A = [j, qs * dj, qs ** 2 * d2j, qs ** 3 * d3j]
        b = list(target[:4])
        basis_in = _bessel_table(l, qs, r_in)
        G = simpson(basis_in[:, None, :] * basis_in[None, :, :]
                    * (r_in * r_in)[None, None, :], x=r_in, axis=-1)
        for k in range(i):                       # cross norms are linear
            A.append(G @ coefficients[k])
            b.append(float(norm_factor) * targets[i, k])

        transform = _bessel_transform_table(l, qs, r_in, q_grid)   # (N, Q)
        bound = energy < 0 and abs(wave[-1] * r[-1]) < 1e-6
        tail = _tail_transform(l, r, wave, r_cut, bound, q_grid)
        K = (transform * weight) @ transform.T * Q_STEP
        kvec = (transform * weight) @ tail * Q_STEP
        k0 = float(np.sum(weight * tail * tail) * Q_STEP)

        c = constrained_minimum(K, kvec, np.array(A), np.array(b), G,
                                float(norm_factor) * targets[i, i])
        coefficients.append(c)
        wavevectors.append(qs)
        pseudo_in.append(c @ basis_in)
        residuals.append(float(c @ K @ c + 2.0 * kvec @ c + k0))

    return _pseudo_waves_record(l, r_cut, energies, waves, wavevectors,
                                coefficients, residuals, norms, pseudo_in,
                                r_in, q_cut, targets)


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


def _cutoff_for(symbol, l, r, radial, r_cut, rc_factor, defaults=None,
                energy=None):
    """Cutoff radius of channel ``l``: explicit, tabulated (``defaults``, the
    family's own table), else ``rc_factor`` times the outermost peak.

    The fallback is **bounded**, and the bound raises rather than clamping.
    Clamping would turn a crash into a wrong number: the cutoff sets where the
    pseudo wave stops matching the all-electron one, so a value silently moved
    to fit the grid produces a dataset that looks fine and is not.

    The bound exists because ``rc_factor * peak`` is unbounded and a diffuse
    reference state sends it past the end of the grid.  Generating the library
    scalar-relativistically, La's :math:`4f` came out at
    :math:`\varepsilon = -0.0076` Hartree peaking at 28.6 Bohr in a 30 Bohr
    box -- a box state, not an atomic one -- so the cutoff landed at 37.1 Bohr
    and :func:`norm_targets` indexed one point past the array.  Three elements
    (La, Ac, Th) died that way, 254 to 735 minutes into a 12.8 hour run.

    A peak beyond half the grid means the reference state is not bound by the
    atom, and no cutoff can rescue it: the configuration or the box is what
    needs fixing.  ``rc_factor * peak`` is also a poor estimator even when it
    is in range -- against six tabulated cutoffs it ran from 0.65x (F) to
    1.56x (Li) -- so the warning below fires well before the refusal does.
    """
    if isinstance(r_cut, dict):
        return float(r_cut[l])
    if r_cut is not None:
        return float(r_cut)
    table = ({} if defaults is None else defaults).get(symbol)
    if table is not None and l in table:
        return float(table[l])
    peak = float(r[int(np.argmax(np.abs(radial * r)))])
    candidate = float(rc_factor * peak)
    detail = (f"its reference state peaks at {peak:.3f} Bohr"
              + ("" if energy is None else f" with eps = {energy:.5f} Ha")
              + f" on a grid reaching {r[-1]:.3f} Bohr")
    if candidate > CUTOFF_GRID_FRACTION * float(r[-1]):
        raise ValueError(
            f"{symbol} l={l}: the fallback cutoff {candidate:.3f} Bohr exceeds "
            f"{CUTOFF_GRID_FRACTION:g} of the radial grid, because {detail}.  A "
            f"reference state peaking that far out is bound by the box rather "
            f"than by the atom, and no cutoff fixes that -- change the "
            f"reference configuration, or enlarge r_max.  Pass an explicit "
            f"r_cut= to override.")
    if candidate > CUTOFF_WARN_RADIUS:
        warnings.warn(
            f"{symbol} l={l}: the fallback cutoff is {candidate:.3f} Bohr, "
            f"beyond the {CUTOFF_WARN_RADIUS:g} Bohr where a smooth partial-wave "
            f"channel is normally trustworthy ({detail}).  rc_factor times the "
            f"outermost peak overestimates a diffuse channel; check the "
            f"dataset with check_paw_channel, or pass an explicit r_cut=.",
            RuntimeWarning, stacklevel=2)
    return candidate


def reference_bound_state(r, potential, l, n_nodes, energy_guess, z_eff,
                          treatment, kappa):
    r"""The valence bound state, from the solver the reference atom used.

    Always the Numerov shoot (:func:`bound_state`), at whatever level of
    theory the atom was solved with.  The pseudization needs its waves to be
    eigenstates of the potential it was handed, and Numerov satisfies the
    radial equation to fourth order, which the optimized pseudization needs: a
    tridiagonal wave of the same potential left an s channel with a ghost 56
    Hartree below the reference.

    What makes this *consistent* is that the reference atom now finishes on
    Numerov orbitals too (:func:`~mandacaru.basis.atomic_solver.solve_atom`,
    ``polish``).  Before it did, the self-consistent field and the
    pseudization disagreed about oxygen's relativistic 2s by 9.5 mHa -- they
    were using two discretizations of a state whose :math:`r^{\gamma}` cusp
    neither resolves well -- and that surfaced as a PAW-LCAO ionic potential
    missing :math:`-Z_{ion}/r` by 8e-4 Hartree out to 11 Bohr.
    """
    del n_nodes                    # the energy guess selects the state
    return bound_state(r, potential, l, energy_guess, z_eff,
                       treatment=treatment, kappa=kappa)


def reference_waves(symbol, atom, valence_config, z_eff, r_cut, rc_factor,
                    energy_offset, defaults=None, treatment: str = "none",
                    kappa: int | None = None, extra_l: int = 0,
                    extra_energy: float | None = None):
    """``(per_l, cutoffs, references)`` -- the all-electron input of a channel.

    ``per_l[l]`` lists every occupied valence state ``(n, energy, R, occupancy)``
    (bound, Numerov-refined); ``cutoffs[l]`` is the cutoff radius snapped to the
    atomic grid, so "inside r_c" and the matching point are the same node; and
    ``references[l] = (waves, energies)`` are the two partial waves each family
    pseudizes -- the bound state(s) plus, when there is only one, the scattering
    state ``energy_offset`` above it, normalized inside the sphere.

    ``treatment`` and ``kappa`` select the radial equation
    (:mod:`mandacaru.basis.relativity`); ``kappa`` is required for
    ``treatment="dirac"`` and then names which :math:`j` this set of channels
    belongs to.

    ``extra_l`` adds that many unoccupied channels above the highest valence
    :math:`l`.  Without them those angular momenta see the local potential
    alone -- which is the right answer only if the local potential happens to
    scatter them correctly, and it does not, because it was built to be smooth
    rather than to reproduce any channel.  An unbound channel has no bound
    state to anchor it, so *both* its references are scattering states, at the
    highest occupied valence eigenvalue and ``energy_offset`` above it; that
    places the pair in the energy window where an atom in a molecule actually
    samples these channels. ``extra_energy`` overrides the first reference
    with a positive scattering energy when a deep occupied shell was frozen.
    """
    r, v_ae = atom.r, atom.v_effective

    def kappa_of_l(l):
        """``kappa`` may be one value or a ``{l: kappa}`` map (one j branch)."""
        return kappa.get(l) if isinstance(kappa, dict) else kappa

    per_l: dict = {}
    for (n, l), occupancy in sorted(valence_config.items()):
        k = kappa_of_l(l)
        guess = atom.eigenvalues[(n, l)]
        if treatment == "dirac" and atom.eigenvalues_j:
            guess = atom.eigenvalues_j.get((n, l, k), guess)
        u, energy = reference_bound_state(r, v_ae, l, n - l - 1, guess, z_eff,
                                          treatment, k)
        # A channel's first reference is a statement about a *bound* state, so a
        # non-negative reference energy is not a hard case to handle -- it is
        # the wrong input.  Under strict aufbau filling Ac's 5f came out at
        # +0.0065 Hartree and Pa's at +0.0672: box states with atomic labels.
        # Pseudizing one produces a dataset that cannot be diagnosed later,
        # which is worse than the IndexError it used to cause 254 minutes into
        # a library build.  `relaxed_configuration` is what stops this
        # happening; the check is here so that a hand-passed configuration
        # cannot walk around it.
        if not energy < 0.0:
            raise ValueError(
                f"{symbol}: the {n}{'spdf'[l]} reference state is not bound "
                f"(eps = {energy:+.5f} Ha).  A channel cannot be built on an "
                f"unbound reference state; the reference configuration "
                f"is what needs changing, not the cutoff or the grid.")
        per_l.setdefault(l, []).append((n, energy, u / r, occupancy))

    cutoffs = {l: _snap(r, _cutoff_for(symbol, l, r, states[0][2], r_cut,
                                        rc_factor, defaults,
                                        energy=states[0][1]))
               for l, states in per_l.items()}

    references: dict = {}
    for l, states in per_l.items():
        waves = [w for _n, _e, w, _o in states]
        energies = [e for _n, e, _w, _o in states]
        if len(states) == 1:
            energy_2 = energies[0] + float(energy_offset)
            u2 = scattering_wave(r, v_ae, l, energy_2, z_eff, treatment,
                                 kappa_of_l(l))
            inside = r <= cutoffs[l]
            u2 = u2 / np.sqrt(np.trapezoid(u2[inside] ** 2, r[inside]))
            waves.append(u2 / r)
            energies.append(energy_2)
        references[l] = (waves[:2], energies[:2])

    if int(extra_l) > 0:
        _add_extra_channels(per_l, cutoffs, references, int(extra_l),
                            float(energy_offset), atom, z_eff, treatment,
                            extra_energy=extra_energy, kappa=kappa)
    return per_l, cutoffs, references


def unoccupied_level(atom, l: int, n_core: int) -> float | None:
    """The all-electron atom's lowest level of the empty angular momentum
    ``l`` above its ``n_core`` core shells of that ``l`` (lithium's 2p at
    -0.042 Ha), or ``None`` when ``l`` binds nothing."""
    from ..basis.atomic_solver import solve_radial
    _u, level = solve_radial(atom.r, atom.v_effective, int(l), int(n_core))
    return float(level) if level < 0.0 else None


#: First references an empty channel falls back to, above its anchor, when
#: the anchor's smooth waves gain a node (:func:`spurious_nodes`): the
#: channel's own bound level, then these scattering energies (Hartree).
#: Potassium's d at its 4s (-0.089) gains one, at 0.0 it does not;
#: aluminum's d (``extra_l=1``) needs 0.1 (HISTORY.md, 2026-10-07).
EXTRA_ANCHOR_ENERGIES = (0.0, 0.1, 0.25)


def _add_extra_channels(per_l, cutoffs, references, extra_l, energy_offset,
                        atom, z_eff, treatment, extra_energy: float | None = None,
                        kappa=None):
    """Append ``extra_l`` unoccupied channels above the highest valence
    :math:`l`, both references scattering states at the widest cutoff."""
    r, v_ae = atom.r, atom.v_effective
    highest = max(per_l)
    # The highest occupied level: a semicore channel's first reference is
    # its semicore state (potassium's 3p at -0.69 Ha, under a 4s at -0.07).
    anchor = (max(energy for states in per_l.values()
                  for _n, energy, _w, _o in states)
              if extra_energy is None else float(extra_energy))
    widest = max(cutoffs.values())
    for l in range(highest + 1, highest + extra_l + 1):
        k = kappa.get(l) if isinstance(kappa, dict) else kappa
        energies = [anchor, anchor + energy_offset]
        cutoffs[l] = widest
        inside = r <= widest
        waves = []
        for energy in energies:
            u = scattering_wave(r, v_ae, l, energy, z_eff, treatment, k)
            u = u / np.sqrt(np.trapezoid(u[inside] ** 2, r[inside]))
            waves.append(u / r)
        references[l] = (waves, energies)
        # A scattering reference contributes no valence charge.
        per_l[l] = [(l + 1, energies[0], waves[0], 0.0)]


def _sign_changes(values: np.ndarray) -> int:
    """Nodes of a radial function: sign changes past its round-off floor."""
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > 1e-6 * np.max(np.abs(values))]
    return int(np.count_nonzero(np.diff(np.sign(values))))


def spurious_nodes(r: np.ndarray, pw: PseudoWaves) -> list[int]:
    """Indices of the smooth partial waves of ``pw`` with more nodes inside
    ``r_c`` than their all-electron waves.

    Pseudization removes nodes; it never has to add one.  A smooth wave
    that gains one belongs to the wrong branch of the norm-conserving fit:
    lithium's p channel anchored at the 2s energy (-0.106 Ha) came out
    with a node inside 2.6 Bohr that its all-electron wave does not have,
    couplings of -1.8 to -2.7 Ha, an intruding-function miss of 0.79, and
    bcc Li 3 % too short in plane waves (HISTORY.md, 2026-10-07).
    """
    radius = np.linspace(0.02, 1.0, 400) * float(pw.r_cut)
    out = []
    for i, (wave, qs, c) in enumerate(zip(pw.waves, pw.wavevectors,
                                          pw.coefficients)):
        smooth = c @ _bessel_table(pw.l, qs, radius)
        all_electron = np.interp(radius, r, np.asarray(wave))
        if _sign_changes(smooth) > _sign_changes(all_electron):
            out.append(i)
    return out


def _spectrum_extent(pp, l: int, floor: float = 1e-5,
                     bounds=(12.0, 24.0)) -> float:
    """Box radius for :func:`radial_spectrum`: where the outermost occupied
    pseudo wave has decayed to ``floor`` of its maximum (a diffuse Li 2s
    reaches 18 Bohr; a semicore channel's valence wave, not its compact
    semicore one)."""
    r = pp.r
    channel = pp.channels[int(l)]
    occupied = len(_occupied_references(channel))
    u = np.abs(channel.pseudo_waves[occupied - 1] * r)
    peak = int(np.argmax(u))
    tail = np.nonzero(u[peak:] > floor * u[peak])[0]
    extent = float(r[peak + tail[-1]]) if tail.size else bounds[1]
    return float(np.clip(extent, *bounds))


def log_derivative_errors(pp, l, energies, r_cut, pseudo_log_derivative,
                          midpoint: bool = True, kappa=None) -> dict:
    """``{energy: (|L_ps - L_ae|, L_ae)}`` at the reference energies (and their
    midpoint), against the all-electron atom stored on ``pp``.

    ``pseudo_log_derivative(pp, l, energy)`` is the family's own pseudo-side
    evaluation -- the only part that differs between families.

    The all-electron side is integrated with **the equation the reference
    atom solved**, taken from ``pp.relativity``.  This is the whole content of
    a relativistic pseudopotential: the pseudo side is a Schrodinger problem
    and the all-electron side is not, and transferability means the smooth
    non-relativistic system reproduces the relativistic scattering.  Comparing
    against a non-relativistic all-electron logarithmic derivative instead
    measures the relativistic shift and calls it an error -- 9.6e-3 for the
    oxygen s channel, against a 1e-3 tolerance.
    """
    ae = pp.atom
    treatment = getattr(pp, "relativity", "none")
    probes = list(energies)
    if midpoint:
        probes.append(0.5 * (energies[0] + energies[1]))
    errors = {}
    for energy in probes:
        l_ae = log_derivative_ae(pp.r, ae.v_effective, l, energy, r_cut,
                                 float(pp.atomic_number), treatment, kappa)
        l_ps = pseudo_log_derivative(pp, l, energy)
        errors[float(energy)] = (float(abs(l_ps - l_ae)), float(l_ae))
    return errors

