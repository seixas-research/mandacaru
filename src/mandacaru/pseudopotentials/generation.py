# -*- coding: utf-8 -*-
# file: pseudopotentials/generation.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Norm-conserving pseudopotentials: Troullier-Martins + Kleinman-Bylander.

Why Mandacaru needs these
-----------------------
The real-space integral engine cannot resolve a heavy-atom core.  The :math:`1s`
shell has length scale :math:`a_0/Z` -- 0.066 Angstrom for oxygen -- while a
practical grid spacing is 0.15--0.30 Angstrom.  The consequences were measured in
:mod:`mandacaru.algorithms.forces`: a spurious force of :math:`\sim\!10^3`
eV/Angstrom on an isolated oxygen atom (whose exact force is zero by symmetry),
growing rather than shrinking under grid refinement, and an egg-box energy error
of ~178 eV for water.  Neither is a bug in the gradient; both come from
integrating :math:`-Z/r` and a cusped core density on a grid too coarse for them.

A pseudopotential removes the cause.  The core electrons are taken out of the
calculation entirely and the singular :math:`-Z/r` is replaced by a *smooth*
potential that reproduces the valence scattering properties.  The length scale
the grid must resolve then becomes the valence one (~1 Bohr), which existing
grids already handle comfortably.

Construction
------------
1. **All-electron reference.** A spherical LDA atom
   (:mod:`mandacaru.basis.atomic_solver`) provides the valence orbitals
   :math:`R_{nl}`, eigenvalues :math:`\varepsilon_{nl}` and screening potential.

2. **Troullier-Martins pseudization** (Phys. Rev. B **43**, 1993 (1991)).  Inside
   a cutoff :math:`r_c` the valence orbital is replaced by

   .. math::

       R^{\text{ps}}_l(r) = r^{\,l}\exp\big[p(r)\big], \qquad
       p(r) = c_0 + c_2 r^2 + c_4 r^4 + \dots + c_{12} r^{12},

   a nodeless, smooth function.  The seven coefficients are fixed by seven
   conditions: **norm conservation** inside :math:`r_c` (what makes the
   pseudopotential transferable -- it preserves the scattering phase shift to
   first order in energy), continuity of :math:`p` and its first four
   derivatives at :math:`r_c`, and :math:`V''_{\text{ps}}(0) = 0`, which for this
   ansatz reads :math:`c_2^2 + (2l+5)c_4 = 0` and is what makes the potential
   flat rather than merely finite at the origin.

3. **Inversion.** The radial Schrödinger equation is inverted for the potential
   that has :math:`R^{\text{ps}}_l` as its ground state at the *same* eigenvalue,

   .. math::

       V^{\text{scr}}_l(r) = \varepsilon_l
         + \frac{(l+1)\,p'(r)}{r} + \frac{p''(r) + p'(r)^2}{2},

   which is manifestly finite at :math:`r = 0` because :math:`p' \sim 2c_2 r`.

4. **Unscreening.** The Hartree and exchange-correlation potentials of the
   *pseudo* valence density are subtracted, leaving the ionic potential that can
   be used in any chemical environment.

5. **Kleinman-Bylander separation** (Phys. Rev. Lett. **48**, 1425 (1982)).  The
   semilocal form :math:`\sum_l V_l(r)\hat P_l` needs one radial integral per
   angular-momentum channel *per pair of basis functions*.  The separable form

   .. math::

       V_{NL} = \sum_{lm} |\chi_{lm}\rangle\, E^{KB}_l\, \langle\chi_{lm}| ,
       \qquad
       \chi_l = \delta V_l\, R^{\text{ps}}_l , \qquad
       E^{KB}_l = \Big[\langle R^{\text{ps}}_l|\delta V_l|R^{\text{ps}}_l\rangle\Big]^{-1},

   replaces that with a **rank-one update per channel**: each basis function is
   projected onto :math:`\chi_{lm}` once, and the nonlocal matrix is an outer
   product.  For :math:`M` basis functions the cost drops from :math:`O(M^2)`
   radial integrals to :math:`O(M)` projections.

.. note::

   This module *generates* pseudopotentials from scratch, like every other basis
   family in Mandacaru -- no tabulated tables are read.  The reference atom is
   spin-restricted LDA, so open-shell atoms use a spherically averaged valence
   configuration, which is the standard practice for pseudopotential generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from ..basis._config import ground_state_config, valence_subshells
from ..basis.atomic_solver import (AtomicResult, hartree_potential,
                            solve_atom, solve_radial)

#: Powers of ``r`` in the Troullier-Martins polynomial ``p(r)``.
TM_POWERS = np.array([0, 2, 4, 6, 8, 10, 12])

#: Default cutoff radius as a multiple of the outermost density maximum.
DEFAULT_RC_FACTOR = 1.15
#: Defaults of the reference atom: LDA, scalar-relativistic, with the
#: nonlinear core correction -- the construction of the other families.
DEFAULT_XC = "lda"
DEFAULT_RELATIVITY = "scalar"
DEFAULT_NLCC = True


# --------------------------------------------------------------------------- #
# Local polynomial derivatives.
# --------------------------------------------------------------------------- #

def _local_derivatives(r: np.ndarray, values: np.ndarray, r0: float,
                       order: int = 4, window: int = 25, degree: int = 8):
    """Derivatives of ``values(r)`` at ``r0``, up to ``order``.

    Fits a polynomial in ``(r - r0)`` over a window of grid points centered on
    ``r0`` and reads the derivatives off its coefficients.  A local polynomial is
    used rather than repeated finite differencing because the Troullier-Martins
    matching needs the *fourth* derivative, which naive differencing renders
    useless.
    """
    index = int(np.argmin(np.abs(r - r0)))
    low = max(index - window, 0)
    high = min(index + window + 1, r.size)
    shifted = r[low:high] - r0
    coefficients = np.polyfit(shifted, values[low:high], degree)
    # np.polyfit returns highest power first; derivative k at 0 is k! * a_k.
    ascending = coefficients[::-1]
    factorial = 1.0
    out = []
    for k in range(order + 1):
        out.append(ascending[k] * factorial if k < ascending.size else 0.0)
        factorial *= (k + 1)
    return np.array(out)


def _polynomial_and_derivatives(coefficients: np.ndarray, r: np.ndarray):
    """``p(r)``, ``p'(r)`` and ``p''(r)`` for the TM even polynomial."""
    r = np.asarray(r, dtype=float)
    p = np.zeros_like(r)
    dp = np.zeros_like(r)
    d2p = np.zeros_like(r)
    for coefficient, power in zip(coefficients, TM_POWERS):
        p += coefficient * r ** power
        if power >= 1:
            dp += coefficient * power * r ** (power - 1)
        if power >= 2:
            d2p += coefficient * power * (power - 1) * r ** (power - 2)
    return p, dp, d2p


# --------------------------------------------------------------------------- #
# Troullier-Martins pseudization of one channel.
# --------------------------------------------------------------------------- #

@dataclass
class Channel:
    """One angular-momentum channel of a pseudopotential."""

    l: int
    n: int                              # principal quantum number of the valence shell
    eigenvalue: float                   # Hartree, reproduced by construction
    r_cut: float                        # Bohr
    coefficients: np.ndarray            # TM polynomial coefficients
    pseudo_radial: np.ndarray           # R_ps(r) on the atom's grid
    v_screened: np.ndarray              # screened channel potential
    v_ionic: np.ndarray = None          # after unscreening
    occupation: float = 0.0
    norm_error: float = 0.0

    def __repr__(self) -> str:
        return (f"Channel(l={self.l}, n={self.n}, "
                f"eps={self.eigenvalue:+.6f} Ha, rc={self.r_cut:.3f} Bohr)")


def _tm_conditions(coefficients, l, r_cut, target, target_norm, r_inner):
    """Residuals of the seven Troullier-Martins conditions."""
    powers = TM_POWERS.astype(float)

    # Matching of p and its first four derivatives at r_c.
    residuals = []
    for order in range(5):
        value = 0.0
        for coefficient, power in zip(coefficients, powers):
            if power < order:
                continue
            factor = np.prod([power - j for j in range(order)]) if order else 1.0
            value += coefficient * factor * r_cut ** (power - order)
        residuals.append(value - target[order])

    # V''(0) = 0  ->  c2^2 + (2l+5) c4 = 0.
    residuals.append(coefficients[1] ** 2 + (2 * l + 5) * coefficients[2])

    # Norm conservation inside r_c.
    p, _dp, _d2p = _polynomial_and_derivatives(coefficients, r_inner)
    norm = np.trapezoid(r_inner ** (2 * l + 2) * np.exp(2.0 * p), r_inner)
    residuals.append(norm - target_norm)
    return residuals


def pseudize_channel(atom: AtomicResult, n: int, l: int, r_cut: float,
                     inner_points: int = 400) -> Channel:
    r"""Troullier-Martins pseudo-orbital and screened potential for one channel.

    Solves the seven nonlinear conditions for the polynomial coefficients, then
    inverts the radial equation for :math:`V^{\text{scr}}_l`.
    """
    r = atom.r
    radial = atom.radial(n, l)
    eigenvalue = atom.eigenvalues[(n, l)]

    # psi = ln(R / r^l) is smooth and finite at the origin; its value and first
    # four derivatives at r_c are what the polynomial must reproduce.
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.log(np.abs(radial) / r ** l)
    target = _local_derivatives(r, psi, r_cut, order=4)

    inside = r <= r_cut
    target_norm = float(np.trapezoid((radial[inside] * r[inside]) ** 2,
                                     r[inside]))

    # A valence orbital with radial nodes (2s, orthogonal to the 1s core) is
    # *negative* beyond its outermost node, hence at r_c.  The pseudo-orbital
    # r^l exp(p) is strictly positive, so it must carry that sign explicitly --
    # otherwise it would introduce a spurious node exactly at the matching point.
    cut_index = int(np.argmin(np.abs(r - r_cut)))
    sign = 1.0 if radial[cut_index] >= 0 else -1.0

    # Least-squares fit of the polynomial to psi as the starting point.
    r_inner = np.linspace(1e-6, r_cut, inner_points)
    fit_mask = (r > 0.30 * r_cut) & inside
    design = np.stack([r[fit_mask] ** power for power in TM_POWERS], axis=1)
    guess, *_ = np.linalg.lstsq(design, psi[fit_mask], rcond=None)

    # The seven conditions are mildly nonlinear (through the norm integral and
    # the quadratic curvature condition), and the plain Powell solve can stall
    # for unlucky r_c.  Retry from perturbed starting points and with the
    # Levenberg-Marquardt variant before giving up.
    arguments = (l, r_cut, target, target_norm, r_inner)
    rng = np.random.default_rng(0)
    solution = None
    for attempt in range(8):
        start = guess if attempt == 0 else guess * (
            1.0 + 0.05 * attempt * rng.normal(size=guess.size))
        method = "hybr" if attempt % 2 == 0 else "lm"
        trial = root(_tm_conditions, start, method=method, args=arguments,
                     options=({"xtol": 1e-13, "maxfev": 40000} if method == "hybr"
                              else {"xtol": 1e-13, "maxiter": 40000}))
        if trial.success and np.max(np.abs(trial.fun)) < 1e-8:
            solution = trial
            break
    if solution is None:
        raise RuntimeError(
            f"Troullier-Martins fit failed for n={n}, l={l}, rc={r_cut:.3f}. "
            "Try a different r_cut -- it must lie beyond the outermost node of "
            "the all-electron valence orbital, and far enough out that the "
            "pseudo-orbital stays smooth.")
    coefficients = solution.x

    # Assemble R_ps: the polynomial form inside r_c, the all-electron tail outside.
    p, dp, d2p = _polynomial_and_derivatives(coefficients, np.maximum(r, 1e-12))
    pseudo = np.where(inside, sign * r ** l * np.exp(p), radial)

    # Invert the radial equation for the potential that supports it.  The sign of
    # R_ps drops out: the potential depends only on p and its derivatives.
    v_inside = eigenvalue + (l + 1) * dp / np.maximum(r, 1e-12) \
        + 0.5 * (d2p + dp * dp)
    v_screened = np.where(inside, v_inside, atom.v_effective)

    # Measure norm conservation on the *fine* inner grid the constraint was
    # imposed on.  Evaluating it on the atom's coarse uniform grid instead would
    # report a quadrature error near the origin (~1e-4) rather than the actual
    # constraint residual (~1e-7).
    p_fine, _dp_fine, _d2p_fine = _polynomial_and_derivatives(coefficients,
                                                              r_inner)
    achieved = float(np.trapezoid(r_inner ** (2 * l + 2) * np.exp(2.0 * p_fine),
                                  r_inner))
    return Channel(
        l=l, n=n, eigenvalue=float(eigenvalue), r_cut=float(r_cut),
        coefficients=coefficients, pseudo_radial=pseudo,
        v_screened=v_screened,
        occupation=float(atom.occupations.get((n, l), 0.0)),
        norm_error=abs(achieved - target_norm) / max(target_norm, 1e-30))


# --------------------------------------------------------------------------- #
# The pseudopotential.
# --------------------------------------------------------------------------- #

@dataclass
class PseudoPotential:
    r"""A norm-conserving pseudopotential in Kleinman-Bylander separable form.

    The potential acting on the valence electrons is

    .. math::

        \hat V_{\text{ps}} = V_{\text{loc}}(r)
          + \sum_{lm} |\chi_{lm}\rangle E^{KB}_l \langle\chi_{lm}| ,

    with :attr:`v_local` smooth everywhere (no :math:`-Z/r` singularity) and one
    rank-one projector per angular-momentum channel.
    """

    symbol: str
    atomic_number: int
    valence_charge: float               # Z_ion = Z - n_core electrons
    r: np.ndarray
    channels: dict                      # l -> Channel
    v_local: np.ndarray                 # local channel, ionic
    local_l: int                        # which l was taken as local
    projectors: dict = field(default_factory=dict)   # l -> chi_l(r)
    kb_energies: dict = field(default_factory=dict)  # l -> E_KB (Hartree)
    valence_density: np.ndarray = None
    atom: AtomicResult = None
    #: Pseudopotential **family** this object belongs to -- the key of
    #: :data:`mandacaru.pseudopotentials.families.PSEUDO_FAMILIES`
    #: (``"ncpp"``: norm-conserving Troullier-Martins with one Kleinman-Bylander
    #: projector per channel).  Written to and read from the library files; a
    #: file without the field is the historical TM family.
    family: str = "ncpp"
    #: How the reference atom was solved.  The defaults describe the
    #: *original* construction on purpose: a file that does not say how it
    #: was made was made that way.  :func:`generate_pseudopotential` always
    #: passes the real values.
    xc: str = "lda"
    relativity: str = "none"
    #: Partial core density the local potential was unscreened with (zero
    #: without a core correction), and the record of how it was built
    #: (:mod:`.core_correction`).
    core_density: np.ndarray = None
    nlcc: dict = field(default_factory=dict)
    #: What its generator could not remove (``ghosts="flag"``), as for the
    #: other families: ``{"ghosts": {l: depth}, "phases": {l: (near, far)}}``.
    defects: dict = field(default_factory=dict)

    def local_potential(self, radius) -> np.ndarray:
        """Interpolate ``V_loc`` onto arbitrary radii (Bohr).

        Beyond the grid the potential is the bare ionic tail
        :math:`-Z_{\\text{ion}}/r`, which is what it decays to by construction.
        """
        radius = np.asarray(radius, dtype=float)
        inside = radius <= self.r[-1]
        out = np.where(inside,
                       np.interp(np.clip(radius, self.r[0], self.r[-1]),
                                 self.r, self.v_local),
                       -self.valence_charge / np.maximum(radius, 1e-12))
        # At r below the first grid point the potential is flat (finite).
        return np.where(radius < self.r[0], self.v_local[0], out)

    def projector_radius(self, l: int) -> float:
        r"""Radius beyond which the KB projector of channel ``l`` vanishes:
        :math:`\chi_l = (V_l - V_{loc}) R^{ps}_l` is nonzero out to the larger
        of the channel's and the local channel's cutoffs."""
        own = float(self.channels[int(l)].r_cut)
        if int(l) not in self.projectors or self.local_l not in self.channels:
            return own
        return max(own, float(self.channels[self.local_l].r_cut))

    def projector(self, l: int, radius) -> np.ndarray:
        """Interpolate the KB projector ``chi_l`` onto arbitrary radii."""
        radius = np.asarray(radius, dtype=float)
        chi = self.projectors[l]
        return np.where(radius <= self.r[-1],
                        np.interp(np.clip(radius, self.r[0], self.r[-1]),
                                  self.r, chi), 0.0)

    def __repr__(self) -> str:
        channels = ", ".join(f"l={l}" for l in sorted(self.channels))
        extras = "".join(
            label for label, on in (
                (", scalar-relativistic", self.relativity == "scalar"),
                (", NLCC", bool((self.nlcc or {}).get("applied"))),
                (", GHOSTED", bool((self.defects or {}).get("ghosts"))),
                (", SCATTERING OFF", bool((self.defects or {}).get("phases"))))
            if on)
        return (f"PseudoPotential({self.symbol}, Z_ion={self.valence_charge:g}, "
                f"[{channels}], local=l{self.local_l}, family={self.family!r}"
                f"{extras})")


def _valence_configuration(atomic_number: int, configuration=None):
    """``(valence subshells, core subshells)`` of the reference atom.

    ``configuration`` defaults to the aufbau filling; a generator that has
    already solved its reference atom passes that atom's ``occupations``, so
    the split describes the atom whose orbitals are about to be pseudized
    rather than a different one.
    """
    configuration = (ground_state_config(atomic_number)
                     if configuration is None
                     else {k: v for k, v in configuration.items() if v > 0})
    valence = set(valence_subshells(atomic_number,
                                    configuration=configuration))
    valence_config = {k: v for k, v in configuration.items() if k in valence}
    core_config = {k: v for k, v in configuration.items() if k not in valence}
    return valence_config, core_config


def generate_pseudopotential(symbol: str, *, r_cut=None,
                             rc_factor: float = DEFAULT_RC_FACTOR,
                             local_l: int | None = None,
                             points: int | None = None, r_max: float = 30.0,
                             atom: AtomicResult | None = None,
                             xc: str = DEFAULT_XC,
                             relativity: str = DEFAULT_RELATIVITY,
                             nlcc: bool | float = DEFAULT_NLCC,
                             ghosts: str = "flag") -> PseudoPotential:
    r"""Generate a Troullier-Martins / Kleinman-Bylander pseudopotential.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    r_cut : float or dict, optional
        Cutoff radius in Bohr, either one value for every channel or
        ``{l: r_cut}``.  The default places it at ``rc_factor`` times the
        outermost maximum of each valence orbital -- the standard heuristic: far
        enough out that the pseudo-orbital is smooth, close enough in that the
        chemically active region is untouched.
    local_l : int, optional
        Channel used as the local potential (default: the highest available
        :math:`l`).  The others become Kleinman-Bylander projectors.
    atom : AtomicResult, optional
        A pre-computed all-electron atom, to avoid re-running the SCF.
    xc : str
        Exchange-correlation functional of the reference atom and of the
        unscreening (``"lda"``, the default, or ``"pbe"``).
    relativity : str
        ``"scalar"`` (the default) solves the reference atom with the
        scalar-relativistic (Koelling-Harmon) equation and pseudizes its
        large component; ``"none"`` is the original non-relativistic atom.
        The Troullier-Martins inversion stays non-relativistic, as in every
        norm-conserving generator: it builds the potential whose *Schrodinger*
        ground state is the pseudo wave at the relativistic eigenvalue, so
        the relativistic shift of the valence levels lives in the potential.
        Norm conservation uses the large-component norm, which differs from
        the relativistic Wronskian identity by :math:`O(c^{-2})`.
    nlcc : bool or float
        Nonlinear core correction (Louie-Froyen-Cohen, :mod:`.core_correction`):
        unscreen with :math:`V_{xc}[\tilde\rho_c + \tilde\rho_v]` and keep the
        partial core density :math:`\tilde\rho_c`.  ``True`` (the default)
        matches it where the core density falls to the valence one; a float
        sets that radius in Bohr; ``False`` unscreens with the valence alone.
    ghosts : str
        ``"flag"`` (the default) checks every channel for a ghost state and
        for wrong scattering (:func:`kb_ghost_search`), tries the other
        channels as the local one, and when none is clean returns the
        least-defective construction with its defects recorded, so loading it
        warns.  The default differs from the other families on purpose: a
        single-channel atom (Li, Na, Ba) has no other local channel to try, and
        one Kleinman-Bylander projector drifts off the all-electron phase far
        from the reference (lithium 0.52 rad at +1 Ha) -- an NCPP property, not
        a construction to refuse.  ``"repair"`` raises
        :class:`~.oncv.GhostStateError` when nothing is clean, ``"refuse"`` at
        the first defect, and ``"keep"`` returns the construction unexamined.

    Returns
    -------
    PseudoPotential
        With :attr:`v_local` smooth at the origin and one projector per remaining
        channel.
    """
    if ghosts != "keep":
        options = {k: v for k, v in locals().items()
                   if k not in ("symbol", "ghosts")}
        return kb_ghost_search(symbol, options, ghosts)

    from ase.data import atomic_numbers

    from ..basis.relativity import _resolve as _resolve_relativity
    from ..basis.xc import xc_potential
    from .core_correction import partial_core_density

    atomic_number = int(atomic_numbers[symbol])
    relativity = _resolve_relativity(relativity)
    if relativity == "dirac":
        raise ValueError("the NCPP family has one projector per l and no "
                         "spin-orbit term; use relativity='scalar', or the "
                         "ONCVPSP or PAW-LCAO family for 'dirac'")
    if atom is None:
        # The ONCVPSP grid (1500 Z points): on a coarser uniform grid a
        # scalar-relativistic s level has not converged -- aluminum's 3s
        # comes out 6 mHa *above* the non-relativistic one at 6000 points,
        # and 1 mHa below it, the textbook sign, at 19500.
        from .oncv import generation_points
        atom = solve_atom(atomic_number,
                          points=(generation_points(atomic_number)
                                  if points is None else int(points)),
                          r_max=r_max, tolerance=1e-7, mixing=0.25, xc=xc,
                          relativity=relativity)
    from .oncv import check_reference_atom
    check_reference_atom(atom, xc, relativity)

    valence_config, core_config = _valence_configuration(
        atomic_number, configuration=atom.occupations)
    if not valence_config:
        raise ValueError(f"{symbol} has no valence subshells to pseudize")
    valence_charge = float(sum(valence_config.values()))

    r = atom.r
    channels: dict = {}
    for (n, l), occupancy in sorted(valence_config.items(), key=lambda kv: kv[0][1]):
        if isinstance(r_cut, dict):
            cut = float(r_cut[l])
        elif r_cut is not None:
            cut = float(r_cut)
        else:
            radial = atom.radial(n, l)
            peak = r[int(np.argmax(np.abs(radial * r)))]    # outermost max of u
            cut = float(rc_factor * peak)
        channels[l] = pseudize_channel(atom, n, l, cut)

    # Valence pseudo-density, used to unscreen -- with a partial core density
    # when the nonlinear core correction is on: the all-electron potential
    # was screened by v_xc[rho_core + rho_valence], and v_xc is not linear.
    valence_density = np.zeros_like(r)
    for l, channel in channels.items():
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
    elif nlcc is not False:
        nlcc_details["reason"] = "the atom has no core to correct for"
    v_hartree = hartree_potential(r, valence_density)
    _e_xc, v_xc = xc_potential(r, valence_density + core_density, xc)
    for channel in channels.values():
        channel.v_ionic = channel.v_screened - v_hartree - v_xc

    local = max(channels) if local_l is None else int(local_l)
    if local not in channels:
        raise ValueError(f"local_l={local} is not among the channels "
                         f"{sorted(channels)}")
    v_local = channels[local].v_ionic

    # Kleinman-Bylander projectors for the remaining channels.
    projectors, kb_energies = {}, {}
    for l, channel in channels.items():
        if l == local:
            continue
        delta_v = channel.v_ionic - v_local
        chi = delta_v * channel.pseudo_radial
        denominator = float(np.trapezoid(
            channel.pseudo_radial * delta_v * channel.pseudo_radial * r * r, r))
        if abs(denominator) < 1e-12:
            continue                       # channel indistinguishable from local
        projectors[l] = chi
        kb_energies[l] = 1.0 / denominator

    return PseudoPotential(
        symbol=symbol, atomic_number=atomic_number,
        valence_charge=valence_charge, r=r, channels=channels,
        v_local=v_local, local_l=local, projectors=projectors,
        kb_energies=kb_energies, valence_density=valence_density, atom=atom,
        family="ncpp", xc=str(xc), relativity=relativity,
        core_density=core_density, nlcc=dict(nlcc_details))


# --------------------------------------------------------------------------- #
# Ghost states of the Kleinman-Bylander form.
# --------------------------------------------------------------------------- #

class _KBChannelView:
    """What the shared ghost and phase tests read off one channel."""

    def __init__(self, channel):
        self.l = channel.l
        self.r_cut = float(channel.r_cut)
        self.pseudo_radial = channel.pseudo_radial
        self.reference_energies = [float(channel.eigenvalue)]


#: Half-width (Hartree) of the "near" window NCPP scattering is judged over.
#: One Kleinman-Bylander projector matches the all-electron logarithmic
#: derivative at the reference energy and, through norm conservation, its
#: first energy derivative -- nothing more -- so over the +-0.5 Ha of the
#: two-projector families a sound single-channel potential drifts off
#: (lithium 0.15, sodium 0.10, barium 0.31 rad).  Over +-0.2 Ha the same
#: 0.05 rad tolerance measures that first-order transferability; the far
#: bound (0.3 rad over +-1 Ha) is the one every family meets.
NCPP_PHASE_WINDOW = 0.2


class KBView:
    r"""A Troullier-Martins / Kleinman-Bylander dataset in the form the
    ONCVPSP spectral and scattering tests read (:func:`~.oncv.radial_spectrum`,
    :func:`~.oncv.log_derivative_ps`, :func:`~.oncv.scattering_errors`,
    :func:`~.oncv.ghost_errors`).

    The screened local potential is the local channel's own
    :math:`V^{scr}_{loc}`, and channel :math:`l` carries one projector
    :math:`\chi_l = \delta V_l\,R^{ps}_l` with the coupling
    :math:`E^{KB}_l`.  :math:`\delta V_l = V_l - V_{loc}` is nonzero out to
    the larger of the two cutoffs, so that is where the projector ends and
    where scattering is compared (:meth:`projector_radius`).  Needs a freshly
    generated dataset: a library file keeps neither the screened potentials
    nor the all-electron atom.
    """

    def __init__(self, pp: "PseudoPotential"):
        if pp.atom is None:
            raise ValueError("the ghost test needs the all-electron atom of a "
                             "freshly generated dataset")
        self.pp = pp
        self.symbol, self.atomic_number = pp.symbol, pp.atomic_number
        self.r, self.atom = pp.r, pp.atom
        self.relativity = getattr(pp, "relativity", "none")
        self.phase_window = NCPP_PHASE_WINDOW
        self.channels = {l: _KBChannelView(c) for l, c in pp.channels.items()}
        local = pp.channels[pp.local_l]
        self.v_local_screened = local.v_screened
        self.projectors = {l: ([pp.projectors[l]] if l in pp.projectors else [])
                           for l in pp.channels}
        self.coupling = {l: (np.array([[pp.kb_energies[l]]])
                             if l in pp.kb_energies else np.zeros((0, 0)))
                         for l in pp.channels}

    def projector_radius(self, l: int) -> float:
        return self.pp.projector_radius(l)

    def unconstructed_ghosts(self) -> dict:
        from .oncv import local_potential_ghosts
        return local_potential_ghosts(self)


def kb_levels(pp, l: int) -> np.ndarray:
    """The two lowest levels of channel ``l`` of a KB dataset (or view)."""
    from .oncv import radial_spectrum
    view = pp if isinstance(pp, KBView) else KBView(pp)
    return radial_spectrum(view, l, n_states=2)


def kb_defects(pp) -> tuple[dict, dict]:
    """``(ghosts, wrong phases)`` of a freshly generated KB dataset:
    :func:`~.oncv.ghost_errors` (every channel, and those without a
    projector) and the channels whose scattering phase is off tolerance
    (:func:`~.oncv.scattering_errors`)."""
    from .oncv import (_wrong_phases, ghost_errors, log_derivative_ps,
                       scattering_errors)
    view = KBView(pp)
    ghosts = ghost_errors(view, kb_levels)
    wrong = {} if ghosts else _wrong_phases(
        scattering_errors(view, log_derivative_ps))
    return ghosts, wrong


def kb_ghost_search(symbol: str, options: dict, mode: str) -> PseudoPotential:
    r"""``generate_pseudopotential(symbol, **options)``, checked for ghosts.

    A Kleinman-Bylander projector can bind a state below the reference (Gonze,
    Stumpf and Scheffler, Phys. Rev. B **44**, 8503 (1991)), and which channel
    serves as the local potential decides whether it does -- the standard
    remedy.  The construction with the default local channel (the highest
    :math:`l`) is checked first; if it holds a ghost, in a channel with a
    projector or without one, or scatters off tolerance, every other channel
    is tried as the local one, the self-consistent atom solved once and
    shared.  A caller who fixed ``local_l`` gets the error rather than an
    override.  ``mode`` is as for :func:`~.oncv.ghost_free`: ``"refuse"``
    raises at the first defect, ``"flag"`` keeps the least-defective
    construction with its defects recorded, and ``"repair"`` raises
    :class:`~.oncv.GhostStateError` when nothing is clean.
    """
    from .oncv import GHOST_MODES, GhostStateError, _describe, _least_defective

    if mode not in GHOST_MODES:
        raise ValueError(f"ghosts must be one of {GHOST_MODES}, not {mode!r}")
    first = generate_pseudopotential(symbol, **options, ghosts="keep")
    ghosts, wrong = kb_defects(first)
    if not ghosts and not wrong:
        return first

    def problem(g, w):
        if g:
            return f"ghost state ({_describe(g)})"
        return "wrong scattering (" + ", ".join(
            f"l={l} {a:.3f}/{b:.3f} rad" for l, (a, b) in sorted(w.items())) + ")"

    candidates = [(first, ghosts, wrong)]
    tried = [f"local l={first.local_l}: {problem(ghosts, wrong)}"]
    if mode != "refuse" and options.get("local_l") is None:
        for local in sorted(first.channels, reverse=True):
            if local == first.local_l:
                continue
            trial = dict(options, local_l=local, atom=first.atom)
            pp = generate_pseudopotential(symbol, **trial, ghosts="keep")
            g, w = kb_defects(pp)
            if not g and not w:
                return pp
            candidates.append((pp, g, w))
            tried.append(f"local l={local}: {problem(g, w)}")
    if mode == "flag":
        return _least_defective(candidates)
    raise GhostStateError(f"{symbol}: no Kleinman-Bylander construction is "
                          f"free of defects ({'; '.join(tried)}).")


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #

def check_channel(pseudopotential: PseudoPotential, l: int) -> dict:
    r"""Verify one channel: norm conservation, tail matching, eigenvalue.

    Returns a dict with

    ``norm_error``
        Relative difference of :math:`\int_0^{r_c}|R|^2r^2dr` between the
        pseudo- and all-electron orbitals.  Norm conservation is the property
        that makes the pseudopotential transferable, so this is the headline
        number.
    ``tail_error``
        Largest deviation of :math:`R^{\text{ps}}` from :math:`R^{\text{AE}}`
        beyond :math:`r_c`, where they must agree exactly.
    ``eigenvalue_error``
        Difference between the all-electron eigenvalue and the one obtained by
        re-solving the radial equation in the *screened* pseudopotential.  It
        tests the inversion step.
    ``v_local_finite``
        Whether the potential is finite at the first grid point (it must be --
        removing the :math:`-Z/r` singularity is the whole point).
    """
    channel = pseudopotential.channels[l]
    atom = pseudopotential.atom
    r = pseudopotential.r
    all_electron = atom.radial(channel.n, l)

    outside = r > channel.r_cut
    tail_error = float(np.max(np.abs(channel.pseudo_radial[outside]
                                     - all_electron[outside])))

    # Re-solve in the screened potential: the eigenvalue must come back.
    u, eps = solve_radial(r, channel.v_screened, l, 0)
    return {
        "norm_error": float(channel.norm_error),
        "tail_error": tail_error,
        "eigenvalue_error": float(eps - channel.eigenvalue),
        "v_local_finite": bool(np.isfinite(channel.v_screened[0])),
        "v_at_origin": float(channel.v_screened[0]),
        "nodes": int(np.sum(np.diff(np.sign(channel.pseudo_radial[r < 5.0])) != 0)),
    }


def report(pseudopotential: PseudoPotential) -> str:
    """Human-readable validation summary for every channel."""
    lines = [f"{pseudopotential!r}",
             f"  valence charge  : {pseudopotential.valence_charge:g}",
             f"  local channel   : l = {pseudopotential.local_l}",
             f"  V_loc(r->0)     : {pseudopotential.v_local[0]:+.4f} Ha "
             f"(all-electron would diverge)"]
    for l in sorted(pseudopotential.channels):
        checks = check_channel(pseudopotential, l)
        channel = pseudopotential.channels[l]
        lines.append(
            f"  l={l}: rc={channel.r_cut:.3f} Bohr  "
            f"norm err={checks['norm_error']:.2e}  "
            f"tail err={checks['tail_error']:.2e}  "
            f"eps err={checks['eigenvalue_error']:+.2e} Ha  "
            f"nodes={checks['nodes']}")
        if l in pseudopotential.kb_energies:
            lines.append(f"        E_KB = {pseudopotential.kb_energies[l]:+.6f} Ha")
    return "\n".join(lines)
