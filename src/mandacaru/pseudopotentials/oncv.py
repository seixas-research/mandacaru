# -*- coding: utf-8 -*-
# file: pseudopotentials/oncv.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Optimized norm-conserving Vanderbilt pseudopotentials (ONCVPSP).

The family ``"oncvpsp"`` (alias ``"oncv"``) implements D. R. Hamann's
construction, Phys. Rev. B **88**, 085117 (2013): **two projectors per
angular-momentum channel** built from two reference energies, a Bessel-function
pseudo wave function whose residual kinetic energy beyond a wave-vector cutoff
is minimized, and a local potential that is *not* one of the channels.  Written
from scratch on top of the same LDA radial atom the Troullier-Martins family
uses (:mod:`mandacaru.basis.atomic_solver`); nothing is read from tables.

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
   :meth:`mandacaru.core.hamiltonian.MolecularIntegrals.kb_nonlocal`.  The raw
   Vanderbilt form is what is stored and used (its off-diagonal coupling is
   real); :func:`diagonalized_projectors` gives Hamann's equivalent
   orthogonalized pair with a diagonal coupling.

Beyond the plain construction
-----------------------------

Five options change what the reference atom is and what the channels cover.
All of them are generation-time: they change the pseudopotential, not the
calculation that later uses it.

``relativity``
    ``"scalar"`` by default.  The reference atom solves the Koelling-Harmon
    equation rather than the Schrodinger one, and ``"dirac"`` solves each
    :math:`j` separately and stores both the :math:`(2j+1)` average and the
    spin-orbit difference (:mod:`mandacaru.basis.relativity`).  The pseudo
    partial waves stay non-relativistic -- they are Bessel expansions, meant
    for a Schrodinger calculation -- so the **generalized norm condition
    changes**: what has to be conserved is
    :math:`-W_{ij}(r_c)/2(\varepsilon_j-\varepsilon_i)`, a Wronskian, which
    equals the inner overlap only when :math:`M\to1`
    (:func:`norm_targets`).  Conserving the overlap instead leaves the
    Vanderbilt matrix asymmetric by 1.4e-4 Ha for oxygen, against a 1e-5
    tolerance.

``xc``
    ``"lda"`` by default, or ``"pbe"`` for a GGA reference atom
    (:mod:`mandacaru.basis.xc`).  The same functional screens the atom and
    unscreens the local potential; they are one argument because they must
    agree.

``nlcc``
    On by default.  A partial core density is built and the unscreening uses
    :math:`v_{xc}[\tilde\rho_c + \tilde\rho_v]`
    (:mod:`mandacaru.pseudopotentials.core_correction`).

``extra_l``
    Channels above the highest valence :math:`l`, each with two scattering
    references.  Zero by default, which leaves those angular momenta to the
    local potential.

Defaults note
-------------

``relativity="scalar"`` and ``nlcc=True`` are **on by default**, and both
change every generated pseudopotential.  ``relativity="none"`` with
``nlcc=False`` reproduces the pre-relativistic construction bit for bit.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import spherical_jn

from ..basis.atomic_solver import AtomicResult, hartree_potential, solve_atom
from ..basis.xc import xc_potential
from .generation import (Channel, PseudoPotential, _local_derivatives,
                         _valence_configuration)

#: Registry name of the family and its alias.
FAMILY = "oncvpsp"
FAMILY_ALIASES = ("oncv",)

#: Spherical Bessel functions per pseudo partial wave (Hamann's ``nbas``).
DEFAULT_N_BESSEL = 8
#: Wave-vector cutoff of the residual kinetic energy (Bohr^-1, Hamann's ``qcut``).
DEFAULT_Q_CUT = 5.0
#: Second reference energy above the bound state when the channel has no second
#: bound state (Hartree, Hamann's ``debl``).
DEFAULT_ENERGY_OFFSET = 1.0
#: Default cutoff radius as a multiple of the outermost maximum of ``r R(r)``.
DEFAULT_RC_FACTOR = 1.3

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

#: Channels added above the highest valence l.  Zero: an extra channel is a
#: real improvement for an atom whose unoccupied l matters chemically, and
#: dead weight otherwise, so it is asked for rather than assumed.
DEFAULT_EXTRA_L = 0
#: Local-potential radius as a multiple of the largest channel cutoff.
DEFAULT_LOCAL_FACTOR = 0.9
#: Per-element cutoff radii (Bohr) overriding the factor heuristic -- close to
#: Hamann's choices for the first row.  Lithium is *not* pushed further out:
#: at 3.0 Bohr the LiH energy jumps by 0.15 Ha and at 3.3 Bohr the s channel
#: grows a ghost state at -0.83 Ha, because the polynomial local potential over
#: so wide a core no longer resembles the atom.
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
#: Fraction of the radial grid the fallback cutoff of :func:`_cutoff_for` may
#: reach before it refuses.  Half leaves the Wronskian stencil and the tail
#: comparison room to work in.
CUTOFF_GRID_FRACTION = 0.5
#: Fallback cutoff (Bohr) above which :func:`_cutoff_for` warns.  Li's
#: deliberate cutoff, 2.60, is the largest in :data:`DEFAULT_CUTOFFS`, and
#: Li grows a ghost state at 3.3.
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
#: Upper wave vector and spacing of the Fourier grid of the residual energy.
Q_MAX, Q_STEP = 60.0, 0.1
#: A residual above this value signals an unstable reference wave, not a
#: usable plane-wave hardness estimate. Deep negative-energy second
#: references in the old W-Hg 4f datasets reach 1e4-1e18 Ha.
MAX_RESIDUAL_KINETIC = 1e4  # Hartree
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
    return record


def read_defects(record) -> dict:
    """The ``defects`` of a stored record; ``{}`` when there are none."""
    record = record or {}
    ghosts = {int(l): float(e) for l, e in (record.get("ghosts") or {}).items()}
    phases = {int(l): (float(a), float(b))
              for l, (a, b) in (record.get("phases") or {}).items()}
    residuals = {int(l): float(e) for l, e in
                 (record.get("residuals") or {}).items()}
    if not ghosts and not phases and not residuals:
        return {}
    result = {"ghosts": ghosts, "phases": phases}
    if residuals:
        result["residuals"] = residuals
    return result


def warn_defects(dataset, family: str) -> None:
    """Raise the :class:`GhostStateWarning` of a dataset that has defects,
    on every load -- the library, a path or a user's own directory."""
    import warnings

    if family == FAMILY:
        residuals = _oncv_residual_defects(dataset.channels)
        if residuals:
            dataset.defects = dict(getattr(dataset, "defects", None) or {})
            dataset.defects["residuals"] = residuals
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
    if defects.get("ghosts"):
        what = "ghost states"
        consequence = (f"A variational calculation containing {symbol} can "
                       f"collapse into a spurious state, so its energies and "
                       f"forces are not reliable.")
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

    ``levels(pp, l)`` returns the two lowest eigenvalues of the channel's
    pseudo Hamiltonian.  A ghost is an *extra* state: the lowest level lies
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
        first, second = (float(e) for e in levels(pp, l)[:2])
        if (first - reference < -GHOST_TOLERANCE
                and abs(second - reference) < abs(first - reference)):
            out[int(l)] = first - reference
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
    treatment = getattr(pp, "relativity", "none")
    # A family may judge "near" over a narrower window (the single-projector
    # NCPP: first-order transferability, KBView.phase_window).
    window = float(getattr(pp, "phase_window", PHASE_WINDOW))
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
        for offset in np.arange(-RESONANCE_WINDOW,
                                RESONANCE_WINDOW + 0.5 * PHASE_STEP, PHASE_STEP):
            energy = reference + offset
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


def _defect_badness(candidate: tuple) -> tuple[int, float, int]:
    """Rank one failed construction, preferring a phase-only defect.

    Among phase-only defects the smaller maximum error wins. Among ghosted
    candidates the shallowest deepest ghost wins, then the one with fewer
    affected channels. The same ranking is used while searching and when
    returning a flagged dataset.
    """
    _pp, ghosts, wrong = candidate
    if not ghosts:
        return (0, max(max(near, far) for near, far in wrong.values()), 0)
    return (1, -min(ghosts.values()), len(ghosts))


def _least_defective(candidates: list[tuple]):
    """Return the least defective attempt with its diagnostic record set."""
    pp, ghosts, wrong = min(candidates, key=_defect_badness)
    pp.defects = {"ghosts": {int(l): float(e) for l, e in ghosts.items()},
                  "phases": {int(l): (float(a), float(b))
                             for l, (a, b) in wrong.items()}}
    return pp


def _wrong_phases(phases: dict) -> dict:
    """The channels of :func:`scattering_errors` outside tolerance."""
    return {l: (near, far) for l, (near, far) in phases.items()
            if near > PHASE_TOLERANCE or far > RESONANCE_TOLERANCE}


def _describe(errors: dict) -> str:
    return ", ".join(f"l={l} {e:+.3g} Ha" for l, e in sorted(errors.items()))


def ghost_free(generate, levels, log_derivative, symbol: str, options: dict,
               mode: str, overrides=None):
    r"""``generate(symbol, **options)``, rebuilt until no channel holds a ghost.

    A deep local well can bind an extra level that the nonlocal projectors do
    not lift. The local radius now follows the *largest* channel cutoff, and
    each projector continues through that radius; the older minimum-cutoff
    construction produced widespread ghosts. A phase error can remain even
    when the bound spectrum is clean.

    **What is done about it.**  ``overrides`` go into every attempt, and when
    there are any they are first tried alone with the construction otherwise
    unchanged -- PAW-LCAO passes ``norm_deficit=0``, which is all iron needs.
    Next the local potential is raised by each of :data:`OWN_CUTOFF_SHIFTS`
    with the cutoffs untouched. ONCVPSP also tries modest contractions of
    only the widest channel, then targeted expansions of a compact highest-l
    channel. Finally every
    channel is given the largest cutoff and the local potential is raised by
    each of
    :data:`GHOST_REMEDY_SHIFTS` in turn (Hamann's ``dvloc0``).  A construction is
    accepted when it has no ghost **and** every bound channel scatters like the
    atom -- to :data:`PHASE_TOLERANCE` near its reference and
    :data:`RESONANCE_TOLERANCE` farther out (:func:`scattering_errors`); a
    raise that trades the ghost for a misplaced resonance is not a repair.  The
    self-consistent atom is solved once and shared by every attempt, and a
    dataset that was clean to begin with is returned exactly as before.

    A caller that fixed ``r_cut``, ``r_cut_local`` or ``local_shift`` has
    made the choice this search would make, so a ghost there is refused
    rather than overridden; so is one with ``mode="refuse"``, and one no
    remedy removes.  ``mode="keep"`` returns the first construction as is.
    """
    if mode not in GHOST_MODES:
        raise ValueError(f"ghosts must be one of {GHOST_MODES}, not {mode!r}")
    if "_channel_cache" in options and options["_channel_cache"] is None:
        # ONCV channel optimization depends on the atom and cutoffs, but not
        # on the local-potential shift. Reuse it across repair attempts.
        options = dict(options, _channel_cache={})
    first = generate(symbol, **options, ghosts="keep")
    if mode == "keep":
        return first
    phase_cache: dict[tuple, float] = {}
    errors = ghost_errors(first, levels)
    if not errors:
        # No ghost is not enough: the first construction has to scatter like
        # the atom too, or it is repaired like a ghosted one.  (Aluminum's p
        # channel was once returned 0.95 rad off, untested.)
        wrong = _wrong_phases(scattering_errors(first, log_derivative,
                                                phase_cache))
        if not wrong:
            return first
    problem = (f"ghost state below the reference ({_describe(errors)})"
               if errors else "wrong scattering (" + ", ".join(
                   f"l={l} {near:.3f}/{far:.3f} rad"
                   for l, (near, far) in sorted(wrong.items())) + ")")
    pinned = [name for name in ("r_cut", "r_cut_local", "local_shift")
              if options.get(name) is not None]
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
    attempts += [(f"own cutoffs, shift {shift:g}",
                  dict(overrides, local_shift=float(shift)))
                 for shift in OWN_CUTOFF_SHIFTS]
    if getattr(first, "family", None) == "oncvpsp":
        # A diffuse outer channel can make r_cl much larger than a compact
        # d/f cutoff.  Raising V_loc alone leaves its phase error, while
        # balancing every channel destroys the compact one's transferability.
        # Contract only the widest cutoff; Ce's 6s/4f/5d reference has a
        # clean interval around 0.9 times its original 6s cutoff.
        for factor in (0.9, 0.85):
            contracted = {
                int(l): float(channel.r_cut) *
                (factor if abs(channel.r_cut - radius) < 1e-8 else 1.0)
                for l, channel in first.channels.items()}
            for shift in (0.0,) + OWN_CUTOFF_SHIFTS:
                attempts.append((f"widest cutoff x {factor:g}, shift {shift:g}",
                                 dict(overrides, r_cut=contracted,
                                      local_shift=float(shift))))
        highest_l = max(first.channels)
        compact_radius = float(first.channels[highest_l].r_cut)
        if compact_radius < 0.6 * radius:
            # A deeply bound d/f reference can be too compact for its own
            # scattering window. Gd 4f at 0.74 Bohr remains phase-wrong at
            # every own-cutoff shift; 1.2 Bohr is clean without stretching
            # that channel all the way to its 4.8 Bohr s cutoff.
            for factor in (1.5, 1.75, 2.0):
                expanded = {
                    int(l): float(channel.r_cut) *
                    (factor if l == highest_l else 1.0)
                    for l, channel in first.channels.items()}
                for shift in (0.0,) + OWN_CUTOFF_SHIFTS:
                    attempts.append((f"l={highest_l} cutoff x {factor:g}, "
                                     f"shift {shift:g}",
                                     dict(overrides, r_cut=expanded,
                                          local_shift=float(shift))))
            # A compact highest-l shell can start below 0.5 Bohr, making
            # multiplicative changes too small. These larger radii still
            # have to pass the residual-kinetic guard: Bi's deep 4f second
            # reference can phase-match but diverge by 1e15 Ha, in which
            # case freezing 4f and using positive scattering is required.
            for target in (1.5, 1.75, 2.0):
                if target <= 2.0 * compact_radius or target >= radius:
                    continue
                expanded = {
                    int(l): (target if l == highest_l else float(channel.r_cut))
                    for l, channel in first.channels.items()}
                for shift in (0.0,) + OWN_CUTOFF_SHIFTS:
                    attempts.append((f"l={highest_l} cutoff {target:g} Bohr, "
                                     f"shift {shift:g}",
                                     dict(overrides, r_cut=expanded,
                                          local_shift=float(shift))))
    attempts += [(f"balanced at {radius:.3f} Bohr, shift {shift:g}",
                  dict(overrides, **balanced, local_shift=float(shift)))
                 for shift in GHOST_REMEDY_SHIFTS]
    tried = []
    # Keep only the best failed construction. A heavy ONCV dataset contains
    # many full-grid arrays; retaining every trial until the search ends can
    # exhaust memory when several elements are built in parallel.
    best = (first, errors, {} if errors else wrong) if mode == "flag" else None
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
            candidate = (pp, {}, wrong)
            if best is not None and _defect_badness(candidate) < \
                    _defect_badness(best):
                best = candidate
            tried.append(f"{label}: phase " + ", ".join(
                f"l={l} {near:.3f}/{far:.3f} rad"
                for l, (near, far) in sorted(wrong.items())))
            continue
        return pp
    if mode == "flag":
        return _least_defective([best])
    raise GhostStateError(
        f"{symbol}: {problem} and "
        f"no remedy removed it while keeping the scattering "
        f"({'; '.join(tried)}).")


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


def assemble_channel(r: np.ndarray, pw: PseudoWaves, v_loc: np.ndarray,
                     n: int, occupation: float = 0.0,
                     strict: bool = True, v_ae: np.ndarray | None = None,
                     r_local: float = 0.0) -> ONCVChannel:
    r"""Projectors and coupling of a channel for a given screened ``v_loc``.

    :math:`\chi_i = \sum_n c_{in}(\varepsilon_i - q_n^2/2 - V_{loc}) j_l(q_n r)`
    inside ``r_cut``; between ``r_cut`` and the local radius ``r_local``,
    where the pseudo wave *is* the all-electron one, :math:`\chi_i =
    (V_{AE} - V_{loc})\varphi_i` (needs ``v_ae``); zero beyond both.
    :math:`B_{ij} = \langle\tilde\varphi_i|\chi_j\rangle` then carries the
    shell term :math:`\langle\varphi_i|V_{AE} - V_{loc}|\varphi_j\rangle`,
    and :math:`D = B^{-1}`.  This is the PAW-LCAO construction of
    :func:`~.paw.assemble_paw_channel` at zero overlap correction: it lets
    :math:`r_{cl}` follow the *largest* cutoff instead of the smallest,
    which is what removed the ghost states of the deep all-electron well.
    With ``strict`` an asymmetry of :math:`B`
    above :data:`B_ASYMMETRY_TOLERANCE` raises.
    """
    l, r_cut = pw.l, pw.r_cut
    r_in = _inner_grid(r_cut)
    inside = r <= r_cut
    v_loc_in = _resample(r, v_loc, r_in)
    extended = v_ae is not None and float(r_local) > r_cut
    shell = (r > r_cut) & (r <= float(r_local))

    pseudo_in, chi_in = [], []
    for c, qs, energy in zip(pw.coefficients, pw.wavevectors, pw.energies):
        basis_in = _bessel_table(l, qs, r_in)
        pseudo_in.append(c @ basis_in)
        chi_in.append(((energy - 0.5 * qs ** 2)[:, None] * basis_in
                       - v_loc_in[None, :] * basis_in).T @ c)
    B = np.array([[simpson(p * x * r_in * r_in, x=r_in) for x in chi_in]
                  for p in pseudo_in])
    if extended:
        # The shell (r_cut, r_local]: T phi = (eps - V_AE) phi there, so
        # chi = (V_AE - V_loc) phi.  Non-relativistic identity; for a
        # scalar-relativistic wave it is O(c^-2) off, as before.
        r_sh = np.linspace(r_cut, float(r_local), INNER_POINTS)
        dv = _resample(r, v_ae - v_loc, r_sh)
        waves_sh = [_resample(r, wave, r_sh) for wave in pw.waves]
        B = B + np.array([[simpson(a * dv * b * r_sh * r_sh, x=r_sh)
                           for b in waves_sh] for a in waves_sh])
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
        chi = np.where(inside, chi, 0.0)
        if extended:
            chi = np.where(shell, (v_ae - v_loc) * wave, chi)
        projectors.append(chi)

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
    #: Full neutral reference-atom occupations, retained so a stored dataset
    #: can be audited for missing occupied angular-momentum channels.
    reference_configuration: dict[tuple[int, int], float] = field(default_factory=dict)
    #: Occupied reference subshells moved into the frozen pseudopotential core.
    frozen_subshells: tuple[tuple[int, int], ...] = ()
    #: First reference energy of added scattering-only channels, when chosen
    #: explicitly to represent a frozen highest-l shell.
    scattering_energy: float | None = None
    #: How the reference atom was solved.  These default to the *pre-
    #: relativistic* construction rather than to :data:`DEFAULT_RELATIVITY`
    #: on purpose: a record that does not say how it was made was made the
    #: old way, and claiming otherwise would put false provenance on every
    #: pseudopotential the library already holds.  :func:`generate_oncv`
    #: always passes the real values.
    xc: str = "lda"
    relativity: str = "none"
    #: Partial core density (zero when there is no core correction), and the
    #: record of how it was built -- see
    #: :mod:`mandacaru.pseudopotentials.core_correction`.
    core_density: np.ndarray = None
    nlcc: dict = field(default_factory=dict)
    #: Channels added above the highest valence l.
    extra_l: int = 0
    #: ``(l, kappa) -> ONCVChannel``, filled only by ``relativity="dirac"``.
    channels_j: dict = field(default_factory=dict)
    #: ``l -> {"projectors": [...], "coupling": D}`` of the ``L . S`` term.
    #: Empty unless the pseudopotential was generated with ``"dirac"``.
    spin_orbit: dict = field(default_factory=dict)

    #: What its generator could not remove (``ghosts="flag"``), as for
    #: PAW-LCAO: ``{"ghosts": {l: depth}, "phases": {l: (near, far)}}``.
    defects: dict = field(default_factory=dict)

    @property
    def has_spin_orbit(self) -> bool:
        """Whether this pseudopotential carries a spin-orbit term."""
        return bool(self.spin_orbit)

    def projector_radius(self, l: int) -> float:
        """Radius beyond which channel ``l``'s projectors vanish: its own
        ``r_cut``, or ``r_cut_local`` when that is larger."""
        return max(float(self.channels[int(l)].r_cut), float(self.r_cut_local))

    def unconstructed_ghosts(self) -> dict:
        """Ghost states the local potential alone binds in a channel without
        projectors (:func:`local_potential_ghosts`)."""
        return local_potential_ghosts(self)

    @property
    def has_core_correction(self) -> bool:
        """Whether a partial core density was built and unscreened with."""
        return bool(self.nlcc.get("applied"))

    def spin_orbit_projector(self, l: int, radius, index: int = 0):
        """Interpolate spin-orbit projector ``index`` of channel ``l``."""
        radius = np.asarray(radius, dtype=float)
        chi = self.spin_orbit[int(l)]["projectors"][int(index)]
        return np.where(radius <= self.r[-1],
                        np.interp(np.clip(radius, self.r[0], self.r[-1]),
                                  self.r, chi), 0.0)

    def core_charge(self) -> float:
        """Electrons in the partial core density (0 without a correction)."""
        if self.core_density is None:
            return 0.0
        return float(np.trapezoid(
            self.core_density * 4.0 * np.pi * self.r * self.r, self.r))

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
                f"qc={self.q_cut:g}"
                f"{', GHOSTED' if (self.defects or {}).get('ghosts') else ''}"
                f"{', SCATTERING OFF' if (self.defects or {}).get('phases') else ''})")


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
    is in range -- against the six cutoffs in :data:`DEFAULT_CUTOFFS` it runs
    from 0.65x (F) to 1.56x (Li) -- so the warning below fires well before the
    refusal does.
    """
    if isinstance(r_cut, dict):
        return float(r_cut[l])
    if r_cut is not None:
        return float(r_cut)
    table = (DEFAULT_CUTOFFS if defaults is None else defaults).get(symbol)
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
            f"beyond the {CUTOFF_WARN_RADIUS:g} Bohr where a norm-conserving "
            f"channel is normally trustworthy ({detail}).  rc_factor times the "
            f"outermost peak overestimates a diffuse channel; check the "
            f"dataset with check_oncv_channel, or pass an explicit r_cut=.",
            RuntimeWarning, stacklevel=2)
    return candidate


def reference_bound_state(r, potential, l, n_nodes, energy_guess, z_eff,
                          treatment, kappa):
    r"""The valence bound state, from the solver the reference atom used.

    Always the Numerov shoot (:func:`bound_state`), at whatever level of
    theory the atom was solved with.  The pseudization needs its waves to be
    eigenstates of the potential it was handed, and Numerov satisfies the
    radial equation to fourth order, which the ONCVPSP construction needs: a
    tridiagonal wave of the same potential leaves its s channel with a ghost
    56 Hartree below the reference.

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
    state ``energy_offset`` above it, normalized inside the sphere.  Shared by
    the ONCVPSP and PAW-LCAO generators.

    ``treatment`` and ``kappa`` select the radial equation
    (:mod:`mandacaru.basis.relativity`); ``kappa`` is required for
    ``treatment="dirac"`` and then names which :math:`j` this set of channels
    belongs to.

    ``extra_l`` adds that many **unbound** channels above the highest valence
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
        # A norm-conserving channel is a statement about a *bound* state, so a
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
                f"(eps = {energy:+.5f} Ha).  A norm-conserving channel cannot "
                f"be built from an unbound state; the reference configuration "
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
        _add_unbound_channels(r, v_ae, z_eff, per_l, cutoffs, references,
                              int(extra_l), float(energy_offset), treatment,
                              extra_energy=extra_energy)
    return per_l, cutoffs, references


def _add_unbound_channels(r, v_ae, z_eff, per_l, cutoffs, references,
                          extra_l, energy_offset, treatment,
                          extra_energy: float | None = None):
    """Append ``extra_l`` scattering-only channels above the valence l."""
    highest = max(per_l)
    anchor = (max(energies[0] for energies in
                  (e for _w, e in references.values()))
              if extra_energy is None else float(extra_energy))
    widest = max(cutoffs.values())
    for l in range(highest + 1, highest + extra_l + 1):
        energies = [anchor, anchor + energy_offset]
        cutoffs[l] = widest
        inside = r <= widest
        waves = []
        for energy in energies:
            k = -(l + 1) if treatment == "dirac" else None
            u = scattering_wave(r, v_ae, l, energy, z_eff, treatment, k)
            u = u / np.sqrt(np.trapezoid(u[inside] ** 2, r[inside]))
            waves.append(u / r)
        references[l] = (waves, energies)
        # n = l + 1 is the lowest principal quantum number this l could have,
        # and the occupancy is zero: the channel exists to scatter, not to
        # hold charge, so it contributes nothing to the valence density.
        per_l[l] = [(l + 1, energies[0], waves[0], 0.0)]


def _validate_reference_configuration(
        atomic_number: int,
        configuration: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int]:
    """Validate a neutral reference atom without importing a configuration table.

    Orbital labels must be physically possible and occupations integral and
    within their spin-degenerate capacity. Zero-occupation entries are dropped
    so the result compares directly with ``AtomicResult.occupations``.
    """
    from numbers import Integral

    if not isinstance(configuration, dict) or not configuration:
        raise TypeError("reference_configuration must be a nonempty dict")
    result: dict[tuple[int, int], int] = {}
    for orbital, occupation in configuration.items():
        if (not isinstance(orbital, tuple) or len(orbital) != 2
                or any(isinstance(x, bool) or not isinstance(x, Integral)
                       for x in orbital)):
            raise ValueError("reference_configuration keys must be (n, l) "
                             "integer pairs")
        n, l = (int(x) for x in orbital)
        if n < 1 or l < 0 or l > 3 or l >= n:
            raise ValueError(f"invalid reference subshell {(n, l)}")
        if (isinstance(occupation, bool)
                or not isinstance(occupation, Integral)
                or not 0 <= occupation <= 2*(2*l+1)):
            raise ValueError(f"invalid occupation of subshell {(n, l)}")
        if occupation:
            result[(n, l)] = int(occupation)
    if sum(result.values()) != int(atomic_number):
        raise ValueError("reference_configuration must contain exactly "
                         f"{atomic_number} electrons")
    return result


def _validate_frozen_subshells(
        valence: dict[tuple[int, int], float],
        frozen_subshells: Sequence[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...]:
    """Validate occupied valence subshells selected for the frozen core.

    The full neutral atom is still solved. Only the pseudopotential's
    valence/core partition changes, so a deep filled shell such as Bi 4f14
    can remain in the nonlinear core correction while its f channel is
    represented by positive-energy scattering projectors.
    """
    from numbers import Integral

    if frozen_subshells is None:
        return ()
    if not isinstance(frozen_subshells, (tuple, list)):
        raise TypeError("frozen_subshells must be a list of (n, l) pairs")
    result: set[tuple[int, int]] = set()
    for orbital in frozen_subshells:
        if (not isinstance(orbital, tuple) or len(orbital) != 2
                or any(isinstance(x, bool) or not isinstance(x, Integral)
                       for x in orbital)):
            raise ValueError("frozen_subshells must contain (n, l) "
                             "integer pairs")
        key = (int(orbital[0]), int(orbital[1]))
        if key not in valence:
            raise ValueError(f"{key} is not an occupied valence subshell")
        if key in result:
            raise ValueError(f"duplicate frozen subshell {key}")
        result.add(key)
    if len(result) == len(valence):
        raise ValueError("at least one occupied valence subshell must remain")
    return tuple(sorted(result))


def _oncv_residual_defects(channels: dict[int, object]) -> dict[int, float]:
    """Return channels with a nonfinite or divergent projector residual.

    The values are stored in Hartree. This also audits older serialized
    datasets whose generators did not reject exponentially growing second
    reference waves.
    """
    result = {}
    for l, channel in channels.items():
        values = np.asarray(channel.residual_kinetic, dtype=float)
        if values.size and (not np.isfinite(values).all()
                            or np.max(values) > MAX_RESIDUAL_KINETIC):
            result[int(l)] = float(np.max(values))
    return result


def _check_oncv_residual_kinetic(waves: dict[int, PseudoWaves]) -> None:
    """Reject an ONCV partial wave whose high-q residual has diverged.

    Deep negative-energy second references can grow exponentially outside
    the core, giving a deceptively clean phase match but an unusable
    projector. The cutoff is deliberately far above ordinary residuals; it
    only detects this numerical pathology.
    """
    for l, record in waves.items():
        for index, residual in enumerate(record.residual_kinetic, start=1):
            if not np.isfinite(residual) or residual > MAX_RESIDUAL_KINETIC:
                raise ValueError(
                    f"l={l} reference {index}: residual kinetic energy "
                    f"{residual:.3g} Ha exceeds {MAX_RESIDUAL_KINETIC:g} Ha; "
                    "choose a positive-energy scattering reference or "
                    "freeze a deep semicore shell")


def generate_oncv(symbol: str, *, r_cut=None, rc_factor: float = DEFAULT_RC_FACTOR,
                  r_cut_local: float | None = None,
                  local_factor: float = DEFAULT_LOCAL_FACTOR,
                  local_shift: float | None = None,
                  q_cut: float = DEFAULT_Q_CUT,
                  energy_offset: float = DEFAULT_ENERGY_OFFSET,
                  n_bessel: int = DEFAULT_N_BESSEL,
                  points: int | None = None, r_max: float = 30.0,
                  atom: AtomicResult | None = None,
                  reference_configuration: dict[tuple[int, int], int] | None = None,
                  frozen_subshells: Sequence[tuple[int, int]] | None = None,
                  scattering_energy: float | None = None,
                  xc: str = DEFAULT_XC,
                  relativity: str = DEFAULT_RELATIVITY,
                  nlcc: bool | float = DEFAULT_NLCC,
                  extra_l: int = DEFAULT_EXTRA_L,
                  _channel_cache: dict | None = None,
                  ghosts: str = "repair") -> ONCVPseudoPotential:
    r"""Generate an ONCVPSP pseudopotential for ``symbol``.

    Parameters
    ----------
    r_cut : float or dict, optional
        Channel cutoff radii (Bohr), one value or ``{l: r_c}``; defaults to
        :data:`DEFAULT_CUTOFFS` for the element, else ``rc_factor`` times the
        outermost maximum of the bound partial wave.
    r_cut_local : float, optional
        Radius of the polynomial local potential (default ``local_factor``
        times the largest channel cutoff).
    local_shift : float, optional
        Raise of the local potential at the origin (Hartree, Hamann's
        ``dvloc0``), a knob against ghost states; zero by default.
    q_cut : float
        Wave-vector cutoff of the residual kinetic energy (Bohr^-1).
    energy_offset : float
        Second reference energy above the bound state for channels without
        a second bound state (Hartree).
    n_bessel : int
        Spherical Bessel functions per pseudo wave.
    xc : str
        Exchange-correlation functional of the reference atom and of the
        unscreening: ``"lda"`` or ``"pbe"`` (:mod:`mandacaru.basis.xc`).  The
        two must be the same functional, and they are, because both read this
        one argument.
    relativity : str
        ``"none"``, ``"scalar"`` (the default) or ``"dirac"``
        (:mod:`mandacaru.basis.relativity`).  ``"dirac"`` builds a separate
        channel for each :math:`j`, then stores their :math:`(2j+1)` average
        as the ordinary channel and their difference as the spin-orbit term --
        so a Dirac pseudopotential is a drop-in replacement for a
        scalar-relativistic one that additionally *carries* spin-orbit
        coupling.  It costs twice the channels and about twice the generation
        time.
    nlcc : bool or float
        Nonlinear core correction
        (:mod:`mandacaru.pseudopotentials.core_correction`).  ``True`` (the
        default) puts the matching radius where the core density falls to the
        valence density; a float sets that radius in Bohr directly; ``False``
        unscreens with the valence density alone, as before.
    extra_l : int
        Channels to add above the highest valence :math:`l`, each with two
        scattering references.  Zero by default, which leaves those angular
        momenta to the local potential.
    points : int, optional
        Radial grid points of the all-electron atom; default
        :func:`generation_points` (finer for heavier atoms, so the Numerov
        partial waves satisfy the radial equation to the ~1e-7 needed for a
        symmetric Vanderbilt matrix).
    reference_configuration : dict, optional
        Complete neutral-atom occupation map ``{(n, l): electrons}`` used for
        the all-electron SCF. This makes a bound but chemically incomplete
        Aufbau reference explicit; for example, Ce may be generated with an
        occupied 5d channel. The occupations must sum to the atomic number.
        When ``atom`` is supplied, its occupations must agree.
    frozen_subshells : sequence of tuple, optional
        Occupied reference subshells, as ``(n, l)`` pairs, moved from valence
        to the frozen core.
        A scattering-only channel is automatically added when this removes
        the highest angular momentum, for example Bi 4f14.
    scattering_energy : float, optional
        First energy (Hartree) of an added scattering-only channel. With a
        frozen subshell the default is +0.25 Ha, keeping the references
        out of a deep, exponentially growing negative-energy region.
    _channel_cache : dict, optional
        Internal cache shared by the ghost-repair attempts. Partial-wave
        optimization is reused when only ``local_shift`` changes.
    ghosts : str
        ``"repair"`` (the default) rebuilds a channel set that binds a ghost
        state with balanced cutoffs and a raised local potential;
        ``"refuse"`` raises :class:`GhostStateError` instead; ``"keep"``
        returns it unexamined.  See :func:`ghost_free`.
    """
    if ghosts != "keep":
        options = {k: v for k, v in locals().items()
                   if k not in ("symbol", "ghosts")}
        return ghost_free(generate_oncv, _oncv_levels, log_derivative_ps,
                          symbol, options, ghosts)

    from ase.data import atomic_numbers

    from ..basis.relativity import _resolve as _resolve_relativity
    from ..basis.relativity import kappa_values
    from .core_correction import partial_core_density

    atomic_number = int(atomic_numbers[symbol])
    relativity = _resolve_relativity(relativity)
    if reference_configuration is not None:
        reference_configuration = _validate_reference_configuration(
            atomic_number, reference_configuration)
    supplied = atom is not None
    if not supplied:
        atom = solve_atom(atomic_number,
                          points=(generation_points(atomic_number)
                                  if points is None else int(points)),
                          r_max=r_max, tolerance=1e-7, mixing=0.25,
                          xc=xc, relativity=relativity,
                          configuration=reference_configuration)
    check_reference_atom(atom, xc, relativity)
    if supplied:
        if reference_configuration is not None and dict(atom.occupations) != \
                reference_configuration:
            raise ValueError("the ONCV reference_configuration disagrees "
                             "with the supplied atom")
    valence_config, core_config = _valence_configuration(
        atomic_number, configuration=atom.occupations)
    frozen = _validate_frozen_subshells(valence_config, frozen_subshells)
    for orbital in frozen:
        core_config[orbital] = valence_config.pop(orbital)
    if not valence_config:
        raise ValueError(f"{symbol} has no valence subshells to pseudize")
    highest_valence_l = max(l for _n, l in valence_config)
    effective_extra_l = max(int(extra_l),
                            max((l for _n, l in frozen), default=highest_valence_l)
                            - highest_valence_l)
    if scattering_energy is None and frozen and effective_extra_l:
        scattering_energy = 0.25
    if scattering_energy is not None and not scattering_energy > 0:
        raise ValueError("scattering_energy must be positive")
    valence_charge = float(sum(valence_config.values()))
    r, v_ae = atom.r, atom.v_effective
    z_eff = float(atomic_number)
    shift = 0.0 if local_shift is None else float(local_shift)

    def channel_set(kappa_map):
        """Every optimized channel of one j branch, cached across shifts."""
        cutoff_key = (tuple(sorted((int(l), float(rc))
                                   for l, rc in r_cut.items()))
                      if isinstance(r_cut, dict)
                      else None if r_cut is None else float(r_cut))
        branch_key = (tuple(sorted(kappa_map.items()))
                      if kappa_map is not None else None)
        cache_key = (id(atom), branch_key, cutoff_key, float(rc_factor),
                     float(energy_offset), effective_extra_l,
                     scattering_energy, float(q_cut), int(n_bessel),
                     relativity)
        if _channel_cache is not None and cache_key in _channel_cache:
            return _channel_cache[cache_key]
        per_l, cutoffs, references = reference_waves(
            symbol, atom, valence_config, z_eff, r_cut, rc_factor,
            energy_offset, treatment=relativity, kappa=kappa_map,
            extra_l=effective_extra_l,
            extra_energy=scattering_energy)
        waves_of_l = {
            l: optimize_pseudo_waves(r, v_ae, l, waves, energies,
                                     cutoffs[l], q_cut=q_cut,
                                     n_bessel=n_bessel, treatment=relativity,
                                     kappa=(kappa_map or {}).get(l),
                                     z_eff=z_eff)
            for l, (waves, energies) in references.items()}
        _check_oncv_residual_kinetic(waves_of_l)
        result = (per_l, cutoffs, waves_of_l)
        if _channel_cache is not None:
            _channel_cache[cache_key] = result
        return result

    if relativity == "dirac":
        # One full construction per j.  `kappa_values(l)` is ordered
        # [l, -(l+1)] = [j = l-1/2, j = l+1/2]; an s channel has only the
        # second, and both branches then ask for the same kappa = -1.
        branches = [{l: kappa_values(l)[0] for l in range(5)},
                    {l: kappa_values(l)[-1] for l in range(5)}]
    else:
        branches = [None]

    built = [channel_set(branch) for branch in branches]
    per_l, cutoffs, _waves = built[0]
    # The local potential follows the *largest* cutoff; a compact channel's
    # projectors reach out to r_cl instead (assemble_channel).
    r_local = _snap(r, float(r_cut_local) if r_cut_local is not None
                    else float(local_factor * max(cutoffs.values())))
    v_loc = polynomial_local_potential(r, v_ae, r_local, shift)

    branch_channels = []
    for branch_per_l, _branch_cutoffs, waves_of_l in built:
        branch_channels.append({
            l: assemble_channel(
                r, waves_of_l[l], v_loc, n=states[0][0],
                occupation=float(sum(o for _n, _e, _w, o in states)),
                v_ae=v_ae, r_local=r_local)
            for l, states in branch_per_l.items()})

    if relativity == "dirac":
        channels, channels_j, spin_orbit = _combine_j_channels(
            branch_channels, r)
    else:
        channels, channels_j, spin_orbit = branch_channels[0], {}, {}

    # Unscreen with the pseudo valence density, plus a partial core density
    # when the nonlinear core correction is on: the all-electron potential was
    # screened by v_xc[rho_core + rho_valence], and v_xc is not linear.
    valence_density = np.zeros_like(r)
    for channel in channels.values():
        valence_density += channel.occupation * channel.pseudo_radial ** 2 \
            / (4.0 * np.pi)
    core_density = np.zeros_like(r)
    nlcc_details = {"applied": False, "r_nlcc": None,
                    "reason": "not requested"}
    if nlcc is not False and core_config:
        true_core, _true_valence = atom.partition_density(valence_config)
        core_density, nlcc_details = partial_core_density(
            r, true_core, valence_density,
            r_nlcc=None if nlcc is True else float(nlcc))
    v_hartree = hartree_potential(r, valence_density)
    _e_xc, v_xc = xc_potential(r, valence_density + core_density, xc)
    v_local_ionic = v_loc - v_hartree - v_xc
    for channel in list(channels.values()) + list(channels_j.values()):
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
        q_cut=float(q_cut), energy_offset=float(energy_offset),
        reference_configuration=dict(atom.occupations),
        frozen_subshells=frozen, scattering_energy=scattering_energy,
        xc=str(xc), relativity=relativity, core_density=core_density,
        nlcc=dict(nlcc_details), extra_l=effective_extra_l,
        channels_j=channels_j, spin_orbit=spin_orbit)


def _combine_j_channels(branch_channels, r):
    r"""``(scalar channels, j-resolved channels, spin-orbit blocks)``.

    Any :math:`j`-dependent separable operator is exactly two terms,

    .. math::

        V_{l,j} = V^{\text{avg}}_l + V^{\text{SO}}_l\,
                  \mathbf{L}\cdot\mathbf{S} ,

    because :math:`\mathbf{L}\cdot\mathbf{S}` takes the single value
    :math:`l/2` on :math:`j = l+\tfrac12` and :math:`-(l+1)/2` on
    :math:`j = l-\tfrac12`.  Solving that 2x2 system gives the
    :math:`(2j+1)`-weighted average and
    :math:`\frac{2}{2l+1}(V_{l+1/2} - V_{l-1/2})` -- the combinations
    :func:`~mandacaru.basis.relativity.j_average` and
    :func:`~mandacaru.basis.relativity.spin_orbit_difference` compute.

    Both terms are built from the **union** of the two branches' projectors,
    with only the coupling matrices reweighted: a projector set is not a
    number, so averaging the potentials means keeping both sets and scaling
    what multiplies them.  The scalar channel therefore carries four
    projectors per :math:`l` where a ``relativity="scalar"`` run carries two,
    and reproduces each :math:`j` exactly rather than approximately.
    """
    from scipy.linalg import block_diag

    lower, upper = branch_channels           # j = l - 1/2, j = l + 1/2
    channels: dict = {}
    channels_j: dict = {}
    spin_orbit: dict = {}
    for l in sorted(upper):
        if l == 0:
            # One j only: the two branches solved the same equation.
            channels[0] = upper[0]
            channels_j[(0, -1)] = upper[0]
            continue
        down, up = lower[l], upper[l]
        channels_j[(l, l)] = down
        channels_j[(l, -(l + 1))] = up
        w_down, w_up = 2.0 * l, 2.0 * l + 2.0
        total = w_down + w_up
        averaged = replace(
            up,
            projectors=list(down.projectors) + list(up.projectors),
            coupling=block_diag(np.asarray(down.coupling) * (w_down / total),
                                np.asarray(up.coupling) * (w_up / total)))
        channels[l] = averaged
        factor = 2.0 / (2 * l + 1)
        spin_orbit[l] = {
            "projectors": list(down.projectors) + list(up.projectors),
            "coupling": block_diag(
                -factor * np.asarray(down.coupling),
                +factor * np.asarray(up.coupling)),
        }
    return channels, channels_j, spin_orbit


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


def _oncv_levels(pp, l: int) -> np.ndarray:
    """The two lowest eigenvalues of one ONCVPSP channel (:func:`ghost_free`)."""
    return radial_spectrum(pp, l, n_states=2)


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
    radius = getattr(pp, "projector_radius", None)
    if r_cut is None:
        r_cut = (radius(l) if radius is not None else pp.channels[l].r_cut)
    r_cut = float(r_cut)
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
    if not chi_u:
        # A channel without projectors (the local channel of a
        # Kleinman-Bylander dataset): the local potential alone.
        return _log_derivative_of_u(r0, u0, r_cut)
    m0 = np.array([np.trapezoid(chi * u0, dx=dr) for chi in chi_u])
    M = np.array([[np.trapezoid(chi * u, dx=dr) for u in uj] for chi in chi_u])
    a = np.linalg.solve(np.eye(len(chi_u)) - D @ M, D @ m0)
    u = u0 + sum(ai * ui for ai, ui in zip(a, uj))
    return _log_derivative_of_u(r0, u, r_cut)


def log_derivative_errors(pp, l, energies, r_cut, pseudo_log_derivative,
                          midpoint: bool = True, kappa=None) -> dict:
    """``{energy: (|L_ps - L_ae|, L_ae)}`` at the reference energies (and their
    midpoint), against the all-electron atom stored on ``pp``.

    ``pseudo_log_derivative(pp, l, energy)`` is the family's own pseudo-side
    evaluation -- the only part that differs between ONCVPSP and PAW-LCAO.

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


def check_oncv_channel(pp: ONCVPseudoPotential, l: int,
                       midpoint: bool = True) -> dict:
    """Validation numbers of one channel.

    ``eigenvalue_error``
        Lowest eigenvalue of the pseudo atomic Hamiltonian of that ``l`` minus
        the bound reference energy (a ghost state makes it negative).

    ``norm_matrix_error``, ``vanderbilt_asymmetry``, ``residual_kinetic``, ``coupling_norm``, ``nodes``
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
    errors = log_derivative_errors(pp, l, energies, pp.projector_radius(l),
                                   log_derivative_ps, midpoint)
    # The same equation the reference atom solved: outside r_c the pseudo
    # wave *is* the all-electron wave, and re-deriving that wave
    # non-relativistically would measure the relativistic shift as a tail
    # error (2.9e-4 for lithium, against a 1e-6 tolerance).
    bound_u, _e = bound_state(r, ae.v_effective, l, energies[0], z,
                              treatment=getattr(pp, "relativity", "none"))
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
                     + "  E_res=" + ", ".join(f"{e:.2e}" for e in
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


def oncv_spin_orbit_blocks(projectors, symbols, potentials) -> dict:
    """``{(atom, l): D_SO}`` -- the spin-orbit blocks, empty without them.

    Keyed by ``(atom, l)`` rather than ``(atom, l, m)``: the spin-orbit term
    is the one part of the nonlocal potential that is **not** diagonal in
    ``m``, so it cannot be a block of the same block-diagonal matrix
    (:func:`mandacaru.core.spin_orbit.spin_orbit_one_body` consumes it).
    """
    blocks: dict = {}
    for projector in projectors:
        pp = potentials[symbols[projector.atom_index]]
        table = getattr(pp, "spin_orbit", None)
        if not table or projector.l not in table:
            continue
        key = (projector.atom_index, projector.l)
        if key not in blocks:
            blocks[key] = np.asarray(table[projector.l]["coupling"],
                                     dtype=complex)
    return blocks


def oncv_coupling_blocks(projectors, symbols, potentials) -> dict:
    """``{(atom, l, m): D_l}`` -- the Vanderbilt blocks for the projectors."""
    blocks = {}
    for projector in projectors:
        key = projector.block_key
        if key not in blocks:
            pp = potentials[symbols[projector.atom_index]]
            blocks[key] = np.asarray(pp.coupling[projector.l], dtype=complex)
    return blocks


def oncv_library_path(directory=None, xc: str = DEFAULT_XC, *,
                      must_exist: bool = True) -> str:
    """The ONCVPSP library folder: ``directory`` when given, else
    ``$MANDACARU_ONCVPSP_PATH/<xc>`` (:func:`.environment.library_directory`)."""
    from .environment import library_directory
    return library_directory(FAMILY, xc, directory, must_exist=must_exist)


_CACHE: dict = {}


def get_oncv(symbol: str, directory=None,
             xc: str = DEFAULT_XC) -> ONCVPseudoPotential:
    """Load ``symbol`` from the ONCVPSP library (cached): ``directory``, or
    ``$MANDACARU_ONCVPSP_PATH/<xc>``."""
    from .io import load_library_dataset

    return load_library_dataset(
        symbol, oncv_library_path(directory, xc), FAMILY, _CACHE,
        label="ONCVPSP", noun="pseudopotential", builder="build_oncv_library")


def build_oncv_library(elements=("H", "Li", "C", "N", "O", "F"),
                       directory=None, *, verbose: bool = True,
                       format: str | None = None, stride: int | None = None,
                       **generation_options):
    """Generate and save ONCVPSP potentials for ``elements``; returns paths."""
    from .io import DEFAULT_FORMAT, STRIDE, library_file, save_pseudopotential

    folder = oncv_library_path(directory,
                               generation_options.get("xc", DEFAULT_XC),
                               must_exist=False)
    os.makedirs(folder, exist_ok=True)
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


def build_oncv(atoms, grid, h, charge, spin, options, kinetic=None, **active):
    r"""Valence-only Hamiltonian from ONCVPSP pseudopotentials.

    Same 5-tuple as the Troullier-Martins builder
    (:func:`~.families._build_tm`): the basis is the bound pseudo partial
    waves (with the ``size`` hierarchy), the external potential the local
    channel, and the nonlocal term the two-projector Vanderbilt form with one
    :math:`2\times2` coupling block per ``(atom, l, m)``; no overlap
    correction (norm-conserving).
    """
    from .families import build_valence_hamiltonian

    return build_valence_hamiltonian(
        atoms, grid, h, charge, spin, options, kinetic, family=FAMILY,
        load=get_oncv, **active,
        projectors=lambda symbols, positions, potentials, _options:
            oncv_projectors(symbols, positions, potentials),
        coupling=oncv_coupling_blocks,
        spin_orbit=oncv_spin_orbit_blocks)


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
    if pp.core_density is not None and np.any(pp.core_density):
        tables["core_density"] = _table(pp.core_density, stride)
    spin_orbit = {str(l): np.asarray(block["coupling"]).real.tolist()
                  for l, block in (pp.spin_orbit or {}).items()}
    for l, block in (pp.spin_orbit or {}).items():
        for i, chi in enumerate(block["projectors"]):
            tables[f"spin_orbit_projector_l{l}_{i}"] = _table(chi, stride)
    return {"symbol": pp.symbol, "atomic_number": int(pp.atomic_number),
            "valence_charge": float(pp.valence_charge),
            "r_cut_local": float(pp.r_cut_local),
            "local_shift": float(pp.local_shift), "q_cut": float(pp.q_cut),
            "energy_offset": float(pp.energy_offset),
            "reference_configuration": [
                [int(n), int(l), float(occupation)]
                for (n, l), occupation in sorted(pp.reference_configuration.items())
                if occupation],
            "frozen_subshells": [[int(n), int(l)]
                                  for n, l in pp.frozen_subshells],
            "scattering_energy": (None if pp.scattering_energy is None
                                  else float(pp.scattering_energy)),
            "xc": str(pp.xc), "relativity": str(pp.relativity),
            "extra_l": int(pp.extra_l), "nlcc": dict(pp.nlcc or {}),
            "spin_orbit": spin_orbit, "defects": defects_record(pp.defects),
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
    dataset = ONCVPseudoPotential(
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
                                        DEFAULT_ENERGY_OFFSET)),
        reference_configuration={
            (int(n), int(l)): float(occupation)
            for n, l, occupation in payload.get("reference_configuration", [])},
        frozen_subshells=tuple((int(n), int(l)) for n, l in
                                payload.get("frozen_subshells", [])),
        scattering_energy=(None if payload.get("scattering_energy") is None
                           else float(payload["scattering_energy"])),
        # A payload without these keys predates them, and a record written
        # before relativity was an option is non-relativistic with no core
        # correction.  Defaulting to the *current* defaults here would label
        # every file already in the library as something it is not.
        xc=str(payload.get("xc", "lda")),
        relativity=str(payload.get("relativity", "none")),
        extra_l=int(payload.get("extra_l", 0)),
        nlcc=dict(payload.get("nlcc") or {"applied": False, "r_nlcc": None,
                                          "reason": "written before the "
                                                    "core correction"}),
        core_density=(np.asarray(tables["core_density"], dtype=float)
                      if "core_density" in tables else np.zeros_like(r)),
        spin_orbit={
            int(l): {
                "coupling": np.asarray(block, dtype=complex),
                "projectors": [
                    np.asarray(tables[f"spin_orbit_projector_l{l}_{i}"],
                               dtype=float)
                    for i in range(len(block))],
            }
            for l, block in (payload.get("spin_orbit") or {}).items()},
        defects=read_defects(payload.get("defects")))
    warn_defects(dataset, FAMILY)
    return dataset


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
