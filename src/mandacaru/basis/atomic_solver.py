# -*- coding: utf-8 -*-
# file: basis/atomic_solver.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Self-consistent all-electron radial atom (spherical LDA).

This is the reference calculation a norm-conserving pseudopotential is built
*from*: it supplies the all-electron valence orbitals :math:`R_{nl}(r)`, their
eigenvalues :math:`\varepsilon_{nl}`, and the screening (Hartree + exchange
-correlation) potential that has to be removed again when the pseudopotential is
"unscreened".

The atom is treated as spherically symmetric, so the Kohn-Sham problem collapses
to a set of one-dimensional radial equations for :math:`u_{nl} = r R_{nl}`,

.. math::

    -\tfrac12 u'' + \Big[\frac{l(l+1)}{2r^2} + V_{\text{eff}}(r)\Big] u
        = \varepsilon\, u ,
    \qquad
    V_{\text{eff}} = -\frac{Z}{r} + V_H[\rho] + V_{xc}[\rho],

solved on a uniform radial grid by a tridiagonal eigensolve and iterated to
self-consistency with Pulay density mixing (:class:`_PulayMixer`).

Exchange-correlation is the local density approximation: Slater exchange plus
the Perdew-Zunger (1981) parameterization of the Ceperley-Alder correlation
energy, with its high-density coefficients :math:`C, D` fixed so that the
energy and its slope are continuous at :math:`r_s = 1` (:data:`PZ_PARA`).
That is the standard choice for generating pseudopotentials, and it keeps this
module free of any external data -- consistent with Mandacaru generating every
basis from scratch.

.. note::

   A uniform grid (rather than the logarithmic grid atomic codes usually use) is
   deliberate: it keeps the eigenproblem a plain symmetric tridiagonal matrix,
   whose k-th eigenpair :func:`~mandacaru.basis.radial_backend.
   tridiagonal_eigenpair` finds directly (in C, with a SciPy fallback) with no
   shooting or node counting by hand.  The cost is more points -- resolving a :math:`1s`
   orbital of scale :math:`a_0/Z` needs a fine spacing -- but the solve is
   one-dimensional and takes milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from .radial_backend import tridiagonal_eigenpair


#: Default radial grid: points and outer radius (Bohr).
DEFAULT_POINTS = 4000
DEFAULT_R_MAX = 25.0
#: Densities kept by the Pulay mixer.  Eight is the usual plateau; more history
#: buys nothing once the residuals span the slow subspace.
MIXING_HISTORY = 8
#: Negative charge, as a fraction of the electron count, that an extrapolated
#: density may carry before the mixer falls back to a linear step.
NEGATIVE_CHARGE_TOLERANCE = 1e-9
#: A valence eigenvalue at or above this (Hartree) is not a bound state, and a
#: configuration containing one is rejected whatever its total energy.
BOUND_STATE_CEILING = -1e-3
#: Fraction of the radial box beyond which a valence orbital's maximum means the
#: state is held by the box rather than by the atom.
BOX_STATE_FRACTION = 0.5


class _PulayMixer:
    r"""Pulay (DIIS) density mixing, in the radial metric.

    The metric matters as much as the algorithm.  :math:`n(r)` spans ten orders
    of magnitude between the nucleus and the valence tail, so a least squares in
    the plain Euclidean norm optimizes the core and leaves the valence to bare
    linear mixing.  The inner product here is

    .. math:: \langle A, B\rangle = \int A\,B\;4\pi r^2\,\mathrm{d}r ,

    the one in which a charge is a charge.

    Given past inputs :math:`n_i` and residuals :math:`R_i = n^{\text{out}}_i -
    n_i`, the coefficients minimize :math:`\|\sum_i c_i R_i\|` subject to
    :math:`\sum_i c_i = 1`, and the next input is :math:`\sum_i c_i (n_i +
    \beta R_i)`.  With one density in hand that is exactly linear mixing, so the
    first iteration is unchanged.
    """

    def __init__(self, weight, mixing: float, electrons: float,
                 history: int = MIXING_HISTORY):
        self.weight = np.asarray(weight, dtype=float)
        self.beta = float(mixing)
        self.electrons = float(electrons)
        self.history = int(history)
        self.inputs: list = []
        self.residuals: list = []

    def _dot(self, a, b) -> float:
        return float(np.sum(self.weight * a * b))

    def _accept(self, mixed, linear):
        r"""``mixed``, or ``linear`` when the extrapolation is not a density.

        Charge needs no repair: every stored input holds :math:`N`, every
        residual holds zero, and :math:`\sum_i c_i = 1`, so the combination
        holds :math:`N` identically.  Non-negativity is the one property Pulay
        can violate, and the honest response is to fall back to the linear
        step.  Clipping and rescaling instead -- perturbing every point of the
        density to repair a few -- is what an earlier version of this class did,
        and it stalled the SCF outright: oxygen sat at a residual of 1e-2 for
        200 iterations and never converged, where the linear mixing it replaced
        converged in 86.  With the fallback it converges in 13.
        """
        deficit = float(np.sum(self.weight * np.where(mixed < 0.0, -mixed, 0.0)))
        if deficit > NEGATIVE_CHARGE_TOLERANCE * self.electrons:
            return linear
        return np.where(mixed > 0.0, mixed, 0.0)

    def __call__(self, density_in, density_out):
        residual = density_out - density_in
        linear = density_in + self.beta * residual
        self.inputs.append(np.asarray(density_in, dtype=float))
        self.residuals.append(residual)
        if len(self.inputs) > self.history:
            del self.inputs[0]
            del self.residuals[0]
        m = len(self.inputs)
        if m < 2:
            return linear
        matrix = np.empty((m + 1, m + 1))
        for i in range(m):
            for j in range(i, m):
                matrix[i, j] = matrix[j, i] = self._dot(self.residuals[i],
                                                        self.residuals[j])
        matrix[:m, m] = 1.0
        matrix[m, :m] = 1.0
        matrix[m, m] = 0.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0
        try:
            coefficients = np.linalg.solve(matrix, rhs)[:m]
        except np.linalg.LinAlgError:
            # Linearly dependent residuals: the history has nothing left to
            # extrapolate along.  Restart it and take the linear step.
            self.inputs, self.residuals = [self.inputs[-1]], [self.residuals[-1]]
            return linear
        if not np.all(np.isfinite(coefficients)):
            self.inputs, self.residuals = [self.inputs[-1]], [self.residuals[-1]]
            return linear
        mixed = np.zeros_like(linear)
        for c, past_in, past_residual in zip(coefficients, self.inputs,
                                             self.residuals):
            mixed += c * (past_in + self.beta * past_residual)
        return self._accept(mixed, linear)


def density_change(previous, current, shell) -> float:
    r"""How far two radial densities are apart, as the SCF loop measures it.

    The maximum of the deviation in :math:`n(r)` **and** in the radial density
    :math:`4\pi r^2 n(r)`, so the test is never blind to one of them.

    The second term is the one that matters and the reason this function
    exists.  :math:`n(r)` runs from :math:`10^4` at the nucleus of a heavy atom
    to :math:`10^{-7}` in the valence tail, so ``max|dn|`` alone is a statement
    about the core: La's aufbau atom reported ``converged=True`` with its
    :math:`6s` at :math:`-0.012` Hartree peaking at 28 Bohr, while the same
    configuration run harder puts it at :math:`-0.192` Hartree peaking at 4.8 --
    a different atom, reached because the branch flip is invisible in
    :math:`n(r)`.  Taking the larger of the two can only ever tighten the
    criterion, never loosen it.
    """
    difference = np.asarray(current, dtype=float) - np.asarray(previous,
                                                              dtype=float)
    return max(float(np.max(np.abs(difference))),
               float(np.max(np.abs(shell * difference))))


# --------------------------------------------------------------------------- #
# Reference configuration.
# --------------------------------------------------------------------------- #

#: ``(Z, xc, relativity, r_max) -> configuration`` decided by
#: :func:`relaxed_configuration`.  One SCF sweep per element per process.
_CONFIGURATION_CACHE: dict = {}


def selection_points(atomic_number: int) -> int:
    """Uniform radial points :func:`relaxed_configuration` judges an atom on.

    Not a free parameter.  At 6000 points cerium's aufbau ``4f`` and ``6s`` come
    out at :math:`+0.036` and :math:`+0.029` Hartree peaking at 21.7 and 25.9
    Bohr, where 12000 points gives :math:`-0.048` at 0.73 Bohr and
    :math:`-0.121` at 4.19 -- a qualitatively different atom.  A bound-state
    test on the coarse grid is reading discretization error, and it did: it
    declared cerium's aufbau filling unbound and "repaired" a configuration that
    was never broken.

    So the floor scales with :math:`Z` like everything else that has to resolve
    a :math:`a_0/Z` cusp on a uniform mesh.  It stays well below
    :func:`mandacaru.pseudopotentials.partial_waves.generation_points` (``1500 Z``),
    because ranking configurations needs the valence bound and ordered, not the
    total energy converged.
    """
    return max(12000, 300 * int(atomic_number))


def relaxed_configuration(atomic_number: int, *, xc: str = "lda",
                          relativity: str = "none", r_max: float = DEFAULT_R_MAX,
                          points: int | None = None):
    r"""The reference configuration to solve this atom in: ``{(n, l): q}``.

    **The aufbau filling is used unless it produces an unbound valence state.**
    That is the whole rule, and it is deliberately narrower than "whichever
    configuration has the lowest energy".

    The defect it exists to repair is sharp.  Under strict aufbau filling the
    ``f`` shell of La, Ac, Th and Pa is not bound at all:

        La  4f^1  eps = -0.00425 Ha, peaking at 27.6 Bohr in a 30 Bohr box
        Ac  5f^1  eps = +0.00647 Ha
        Th  5f^2  eps = -0.01226 Ha, peaking at 26.3 Bohr
        Pa  5f^3  eps = +0.06716 Ha, peaking at 29.99 Bohr -- the last node

    A positive eigenvalue is not a valence orbital, it is a box state, and a
    norm-conserving channel built from one is meaningless.  That is what killed
    three elements 254 to 735 minutes into a 12.8 hour library build.  Moving one
    electron to the ``d`` shell binds it -- La's ``5d`` sits at -0.111 Hartree
    peaking at 3.34 Bohr -- and LDA prefers that by 30.87 Hartree, a margin
    stable at -30.71, -30.87 and -31.12 across 6000, 12000 and 24000 points.

    **Why it does not simply minimize the energy.**  Chromium's ``3d^5 4s^1``
    margin reads -0.355, -0.125 and -0.029 Hartree over that same grid range: a
    number shrinking toward zero, which could still change sign.  Acting on it
    would be acting on noise, so this function **declines to reproduce the Cr
    and Cu anomalies** rather than claim a variational result it cannot support.
    No table of experimental configurations enters this function either way;
    the only overrides are the two PAW-LCAO reference configurations of Pa and
    U in ``pseudopotentials.paw.DEFAULT_REFERENCE_CONFIGURATIONS``.

    **Cost.**  A configuration with no occupied ``f`` shell in its valence
    returns immediately, with no solve at all: the aufbau ``d`` shell of a
    transition metal is bound, and the energy margins there are the
    unresolvable ones above.  So only the ``f`` block pays, and only La, Ac, Th
    and Pa pay more than a single solve.  Cached per
    ``(Z, xc, relativity, r_max)``.
    """
    from ._config import rearrangements, valence_subshells

    Z = int(atomic_number)
    candidates = rearrangements(Z)
    aufbau = dict(candidates[0])
    if len(candidates) == 1:
        return aufbau
    # Only an occupied f shell can be the unbound one; see "Cost" above.
    if not any(l == 3 and aufbau.get((n, l), 0) > 0
               for (n, l) in valence_subshells(Z, configuration=aufbau)):
        return aufbau

    # The grid is part of the key: at 6000 points cerium's aufbau 4f and 6s
    # come out unbound, at 12000 they are bound -- so which configuration wins
    # depends on it (see `selection_points`).
    grid = selection_points(Z) if points is None else int(points)
    key = (Z, str(xc).strip().lower(), str(relativity).strip().lower(),
           float(r_max), grid)
    cached = _CONFIGURATION_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    def solved(configuration):
        try:
            return solve_atom(Z, points=grid, r_max=float(r_max),
                              configuration=configuration, xc=xc,
                              relativity=relativity)
        except (ValueError, RuntimeError):
            return None           # a candidate the solver cannot follow

    atom = solved(aufbau)
    if atom is not None and atom.converged and bound_valence(atom):
        _CONFIGURATION_CACHE[key] = aufbau
        return dict(aufbau)

    best, best_energy = None, np.inf
    for candidate in candidates[1:]:
        trial = solved(candidate)
        if (trial is not None and trial.converged and bound_valence(trial)
                and trial.total_energy < best_energy):
            best, best_energy = candidate, trial.total_energy
    # Nothing bound: keep aufbau and let the generator refuse it by name, which
    # says more than a silently substituted configuration would.
    chosen = dict(aufbau if best is None else best)
    _CONFIGURATION_CACHE[key] = chosen
    return dict(chosen)


def bound_valence(atom) -> bool:
    r"""Is every valence orbital of ``atom`` bound by the atom?

    Two ways it can fail to be.  The eigenvalue can be at or above zero, which
    is not a bound state at all -- Ac's aufbau ``5f`` comes out at
    :math:`+0.0065` Hartree.  Or it can be
    negative but tiny with its maximum against the wall of the box, which is a
    box state wearing an atomic label: La's aufbau ``4f`` sits at
    :math:`-0.0043` Hartree peaking at 27.6 Bohr in a 30 Bohr box.

    Either way a norm-conserving pseudopotential channel built from it is
    meaningless, which is how three elements died partway through a 12.8 hour
    library build.  This is a **property of the configuration**, not a matter
    of which configuration has the lowest energy, so it is tested separately
    and it overrides the energy.
    """
    from ._config import valence_subshells

    valence = valence_subshells(atom.atomic_number,
                               configuration=atom.occupations)
    edge = BOX_STATE_FRACTION * float(atom.r[-1])
    for state in valence:
        eps = atom.eigenvalues.get(state)
        if eps is None or eps > BOUND_STATE_CEILING:
            return False
        u = atom.orbitals.get(state)
        if u is not None and atom.r[int(np.argmax(np.abs(u)))] > edge:
            return False
    # A Dirac atom has one level per j.  The (2j+1) average can be bound
    # while one branch is not (Pa's 5f: +0.001 / +0.005 Ha), and a branch
    # cannot be built on an unbound state.
    for (n, l, kappa), eps in atom.eigenvalues_j.items():
        if (n, l) not in valence:
            continue
        if eps > BOUND_STATE_CEILING:
            return False
        u = atom.orbitals_j.get((n, l, kappa))
        if u is not None and atom.r[int(np.argmax(np.abs(u)))] > edge:
            return False
    return True


# --------------------------------------------------------------------------- #
# Exchange-correlation (LDA).
# --------------------------------------------------------------------------- #

def lda_exchange(rho: np.ndarray):
    r"""Slater exchange: returns ``(energy density e_x, potential v_x)``.

    :math:`e_x = -\tfrac34 (3/\pi)^{1/3}\rho^{1/3}` per electron and
    :math:`v_x = \tfrac43 e_x`.
    """
    rho = np.maximum(rho, 1e-30)
    ex = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    return ex, (4.0 / 3.0) * ex


def _continuous_pz(gamma, beta1, beta2, a, b):
    r"""Perdew-Zunger parameters ``(gamma, beta1, beta2, A, B, C, D)`` whose
    high-density :math:`C, D` make :math:`\varepsilon_c` and
    :math:`d\varepsilon_c/dr_s` continuous at :math:`r_s = 1`.

    The published four-digit :math:`C, D` leave the two forms 3.2e-5 Ha
    apart there (paramagnetic) and the slope 1.3e-5 Ha apart; that jump in
    the potential is a jump in the energy surface wherever the density
    crosses :math:`r_s = 1`, which finite-difference forces see (0.02
    eV/Angstrom in dense regions).  Fixing :math:`C, D` from the
    low-density Pade form -- the same correction as libxc's
    ``LDA_C_PZ_MOD`` -- moves them by 0.03-1 % of themselves:

    .. math::

        D = \frac{\gamma}{1 + \beta_1 + \beta_2} - B, \qquad
        C = -\frac{\gamma(\beta_1/2 + \beta_2)}{(1 + \beta_1 + \beta_2)^2}
            - A - D.
    """
    denominator = 1.0 + beta1 + beta2
    d = gamma / denominator - b
    c = -gamma * (beta1 / 2.0 + beta2) / denominator ** 2 - a - d
    return (gamma, beta1, beta2, a, b, c, d)


#: Perdew-Zunger (1981) paramagnetic and ferromagnetic fits, ``(gamma,
#: beta1, beta2, A, B, C, D)`` in Hartree, continuous at :math:`r_s = 1`.
PZ_PARA = _continuous_pz(-0.1423, 1.0529, 0.3334, 0.0311, -0.048)
PZ_FERRO = _continuous_pz(-0.0843, 1.3981, 0.2611, 0.01555, -0.0269)


def lda_correlation(rho: np.ndarray):
    r"""Perdew-Zunger (1981) correlation: returns ``(e_c, v_c)``.

    The standard parameterization of the Ceperley-Alder uniform-electron-gas
    correlation energy, in its unpolarized form, split at :math:`r_s = 1`
    and continuous there (:data:`PZ_PARA`).
    """
    rho = np.maximum(rho, 1e-30)
    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    gamma, beta1, beta2, a, b, c, d = PZ_PARA

    # High-density (rs < 1) logarithmic form.
    log_rs = np.log(rs)
    ec_high = a * log_rs + b + c * rs * log_rs + d * rs
    vc_high = (a * log_rs + (b - a / 3.0)
               + (2.0 / 3.0) * c * rs * log_rs
               + (2.0 * d - c) * rs / 3.0)

    # Low-density (rs >= 1) Pade form.
    sqrt_rs = np.sqrt(rs)
    denom = 1.0 + beta1 * sqrt_rs + beta2 * rs
    ec_low = gamma / denom
    vc_low = ec_low * (1.0 + (7.0 / 6.0) * beta1 * sqrt_rs
                       + (4.0 / 3.0) * beta2 * rs) / denom

    high = rs < 1.0
    return (np.where(high, ec_high, ec_low),
            np.where(high, vc_high, vc_low))


def lda_xc(rho: np.ndarray):
    """Total LDA exchange-correlation ``(e_xc, v_xc)`` (Hartree)."""
    ex, vx = lda_exchange(rho)
    ec, vc = lda_correlation(rho)
    return ex + ec, vx + vc


# --------------------------------------------------------------------------- #
# Radial solves.
# --------------------------------------------------------------------------- #

def solve_radial(r: np.ndarray, potential: np.ndarray, l: int, n_nodes: int):
    r"""Bound state of ``-1/2 u'' + [l(l+1)/2r^2 + V] u = eps u``.

    Parameters
    ----------
    r : ndarray
        Uniform radial grid, strictly positive and equally spaced.
    potential : ndarray
        :math:`V_{\text{eff}}(r)` on that grid (Hartree).
    l : int
        Angular momentum.
    n_nodes : int
        Radial nodes wanted, ``n - l - 1``; selects which eigenvalue to return.

    Returns
    -------
    (u, eps) : (ndarray, float)
        :math:`u = rR` normalized so ``int u^2 dr = 1``, and the eigenvalue.
    """
    step = float(r[1] - r[0])
    diag = 1.0 / step ** 2 + potential + l * (l + 1) / (2.0 * r * r)
    offdiag = -0.5 / step ** 2 * np.ones(r.size - 1)
    value, u = tridiagonal_eigenpair(diag, offdiag, n_nodes)
    u = u / np.sqrt(np.trapezoid(u * u, r))
    if u[0] < 0:                                # fix the global sign
        u = -u
    return u, value


def hartree_potential(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r"""Radial Hartree potential of a spherical density.

    .. math::

        V_H(r) = \frac{4\pi}{r}\int_0^r \rho(r')r'^2\,dr'
                 + 4\pi\int_r^\infty \rho(r')r'\,dr' .
    """
    from scipy.integrate import cumulative_trapezoid

    inner = cumulative_trapezoid(rho * r * r, r, initial=0.0)
    outer_total = np.trapezoid(rho * r, r)
    outer = outer_total - cumulative_trapezoid(rho * r, r, initial=0.0)
    return 4.0 * np.pi * (inner / r + outer)


# --------------------------------------------------------------------------- #
# Result container.
# --------------------------------------------------------------------------- #

@dataclass
class AtomicResult:
    """Converged all-electron atom."""

    atomic_number: int
    r: np.ndarray                       # radial grid (Bohr)
    orbitals: dict                      # (n, l) -> u_nl = r * R_nl
    eigenvalues: dict                   # (n, l) -> Hartree
    occupations: dict                   # (n, l) -> electrons
    density: np.ndarray                 # rho(r), spherical
    v_effective: np.ndarray             # -Z/r + V_H + V_xc
    v_hartree: np.ndarray
    v_xc: np.ndarray
    total_energy: float = 0.0
    converged: bool = False
    iterations: int = 0
    details: dict = field(default_factory=dict)
    #: ``(n, l, kappa) -> u``, filled only by a ``relativity="dirac"`` run.
    #: The spin-orbit-resolved counterparts of ``orbitals``/``eigenvalues``/
    #: ``occupations``, whose ``(n, l)`` entries then hold the ``(2j+1)``
    #: -weighted average -- so every consumer that predates spin-orbit
    #: coupling keeps reading the quantity it always read.
    orbitals_j: dict = field(default_factory=dict)
    eigenvalues_j: dict = field(default_factory=dict)
    occupations_j: dict = field(default_factory=dict)

    def radial(self, n: int, l: int) -> np.ndarray:
        """``R_nl(r) = u_nl / r``."""
        return self.orbitals[(n, l)] / self.r

    @property
    def relativity(self) -> str:
        """Which radial equation produced this atom."""
        return str(self.details.get("relativity", "none"))

    @property
    def xc(self) -> str:
        """Which exchange-correlation functional produced it."""
        return str(self.details.get("xc", "lda"))

    def spin_orbit_splitting(self, n: int, l: int) -> float:
        r""":math:`\varepsilon_{l+1/2} - \varepsilon_{l-1/2}` (Hartree).

        Zero for an s shell, and zero for any atom not solved with
        ``relativity="dirac"`` -- in which case it is zero because the
        equation has no spin-orbit term, not because the atom has no
        splitting.
        """
        if int(l) == 0 or not self.eigenvalues_j:
            return 0.0
        return float(self.eigenvalues_j[(n, l, -(l + 1))]
                     - self.eigenvalues_j[(n, l, l)])

    def partition_density(self, valence):
        """``(core, valence)`` spherical densities for a valence selection.

        ``valence`` is an iterable of ``(n, l)`` subshells treated as valence;
        everything else in the configuration is core.  This is the split a
        nonlinear core correction is built from
        (:func:`mandacaru.pseudopotentials.core_correction.partial_core_density`).
        """
        wanted = {(int(n), int(l)) for n, l in valence}
        core = np.zeros_like(self.r)
        outer = np.zeros_like(self.r)
        weight = 4.0 * np.pi * self.r * self.r
        if self.occupations_j:
            items = [((n, l), q, self.orbitals_j[(n, l, k)])
                     for (n, l, k), q in self.occupations_j.items() if q > 0]
        else:
            items = [((n, l), q, self.orbitals[(n, l)])
                     for (n, l), q in self.occupations.items() if q > 0]
        for key, occupancy, u in items:
            share = occupancy * u * u / weight
            if key in wanted:
                outer += share
            else:
                core += share
        return core, outer

    @property
    def n_electrons(self) -> int:
        return int(round(sum(self.occupations.values())))

    def __repr__(self) -> str:
        return (f"AtomicResult(Z={self.atomic_number}, "
                f"{len(self.orbitals)} subshells, "
                f"E={self.total_energy:.6f} Ha, converged={self.converged})")


# --------------------------------------------------------------------------- #
# Self-consistent field.
# --------------------------------------------------------------------------- #

def solve_atom(atomic_number: int, *, points: int=DEFAULT_POINTS, r_max: float=DEFAULT_R_MAX, max_iterations: int=200, tolerance: float=1e-06, mixing: float=0.3, configuration=None, confinement=None, xc: str='lda', relativity: str='none', polish: int=12, grid: str='uniform', resources=None) -> AtomicResult:
    """Self-consistent spherical LDA atom.

    Parameters
    ----------
    atomic_number : int
        Nuclear charge :math:`Z`.
    points, r_max : int, float
        Uniform radial grid: ``points`` nodes out to ``r_max`` Bohr.  The default
        resolves the :math:`1s` shell of the first two rows; heavier atoms want
        more points.
    mixing : float
        Fraction of the linear step taken by the Pulay mixer (and the whole
        step whenever the Pulay extrapolation is not a density).
    configuration : dict, optional
        ``{(n, l): occupancy}``.  Defaults to :func:`relaxed_configuration`:
        the aufbau ground state unless it leaves the valence unbound.
    confinement : callable or ndarray, optional
        An extra external potential (Hartree) added **only when integrating
        the orbitals** -- a confining wall for a localized basis (see
        :mod:`mandacaru.basis.nao_ae`).  A callable is evaluated on the radial
        grid; an array must already be on it.  It is *not* part of the
        returned ``v_effective`` / ``v_hartree`` / ``v_xc``, which remain the
        genuine self-consistent potentials of the (confined) density.
    xc : str
        ``"lda"`` (Slater + Perdew-Zunger, the default and the historical
        behavior) or ``"pbe"`` (:mod:`mandacaru.basis.xc`).
    relativity : str
        ``"none"`` (the default), ``"scalar"`` (Koelling-Harmon: mass-velocity
        and Darwin, no spin-orbit) or ``"dirac"`` (each ``j`` solved
        separately).  See :mod:`mandacaru.basis.relativity`.  A ``"dirac"``
        atom additionally fills ``orbitals_j`` / ``eigenvalues_j`` /
        ``occupations_j``; its ``(n, l)`` entries carry the ``(2j+1)``-weighted
        average, so a consumer that does not know about spin-orbit coupling
        sees the scalar-relativistic atom.  Both relativistic treatments also
        use the relativistic correction to LDA exchange
        (:func:`mandacaru.basis.xc.relativistic_exchange_factors`).
    polish : int
        Extra self-consistency steps a **relativistic** atom takes with the
        Numerov shoot the pseudopotential generators use, after the
        tridiagonal field has converged (0 disables it; ignored when
        ``relativity="none"``).  Only the valence is re-solved.  Without it
        the atom and the generators discretize a relativistic s state two
        different ways -- its :math:`r^\\gamma` cusp is resolved well by
        neither -- and disagree about where the level is: 9.5 mHa for
        oxygen's 2s, which reaches the ionic potential as an 8e-4 Hartree
        error in its tail.
    grid : str
        ``"uniform"`` (the default) is the grid every consumer of this atom
        expects.  ``"log"`` solves each state on a logarithmic grid instead
        (:mod:`~mandacaru.basis.loggrid`), which is the only way to converge a
        relativistic :math:`l=0` level, and is meant for **measuring** the
        uniform atom rather than for feeding a generator: the pseudopotential
        constructions need waves that satisfy the uniform-grid equation, which
        a splined log-grid wave does not.  ``polish`` is forced off with it --
        polishing re-solves the valence with the uniform Numerov shoot, so
        running both makes them fight over the valence levels and oxygen's
        relativistic total-energy shift comes out with the wrong sign.

    Returns
    -------
    AtomicResult
        Converged orbitals, eigenvalues, density and potentials.
    """
    if str(xc).lower() != 'lda' or str(relativity).lower() not in ('none', 'nonrelativistic', 'non-relativistic', 'nr') or grid != 'uniform':
        raise NotImplementedError('native spherical atoms currently support nonrelativistic uniform-grid LDA')
    if callable(confinement):
        confinement=confinement(np.arange(1,points+1,dtype=float)*r_max/(points+1))
    from .. import _native
    context = _native.Context(**dict(resources or {}))
    spec = None if configuration is None else [(n, l, float(q)) for (n, l), q in configuration.items() if q > 0]
    raw = context.solve_atom(atomic_number, points=points, r_max=r_max, max_iterations=max_iterations, tolerance=tolerance, mixing=mixing, configuration=spec, confinement=None if confinement is None else np.ascontiguousarray(confinement, dtype=float))
    return AtomicResult(**raw, details={'engine': 'rust', 'points': points, 'r_max': r_max, 'mixing': mixing, 'confined': confinement is not None, 'xc': 'lda', 'relativity': 'none', 'relativistic_exchange': False, 'memory_peak_bytes': context.memory_peak_bytes})
