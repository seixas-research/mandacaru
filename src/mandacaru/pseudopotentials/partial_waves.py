# -*- coding: utf-8 -*-
# file: pseudopotentials/partial_waves.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Reference waves, optimized smooth partial waves and the checks of a
generated dataset -- the radial machinery of PAW-LCAO and UPAW-LCAO
generation (:mod:`.paw`).

Every number is computed in Rust (``mandacaru-core``, ``paw`` and ``models``;
the extension's ``paw_*`` functions).  This module adapts datasets and arrays
to those routines and carries the generator's control flow and wording:

* **All-electron reference waves.**  Bound states and scattering states of the
  reference atom's potential by Numerov integration (:func:`bound_state`,
  :func:`scattering_wave`), scalar-relativistic or with a Dirac
  :math:`\kappa`, and the reference set of a channel (:func:`reference_waves`).
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

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .. import _native
from .dataset import _valence_configuration
from .native_context import context

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
#: Bound levels deeper than this (Hartree) are counted against the atom by
#: :func:`ghost_errors` (excited ghosts).  Shallower ones depend on the box (a Rydberg 7s at
#: -0.002 Ha is in a 30 Bohr box and not in a 20 Bohr one); every excited
#: ghost found in the libraries lies below -0.14 Ha.
EXCITED_GHOST_MARGIN = 0.1
#: Self-consistency tolerance and density mixing of the generator's
#: all-electron reference atom (:func:`reference_atom`).
REFERENCE_TOLERANCE, REFERENCE_MIXING = 1e-7, 0.25


#: Points of the fine quadrature grid inside ``r_c`` (the Rust grid's:
#: ``len(_native.paw_inner_grid(r_c))``; ``test_partial_waves`` pins them equal).
INNER_POINTS = 801


def _array(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=float)


def _flat(values) -> tuple:
    """``(contiguous 1-D float array, original shape)``."""
    values = np.asarray(values, dtype=float)
    return np.ascontiguousarray(values.ravel()), values.shape


def bessel_derivatives(l: int, x):
    """``j_l(x)`` and its first three derivatives with respect to ``x``.

    The second and third derivatives follow from the spherical Bessel
    equation :math:`x^2 j'' + 2x j' + (x^2 - l(l+1)) j = 0`, so no numerical
    differentiation is involved.
    """
    flat, shape = _flat(x)
    out = _native.paw_bessel_derivatives(context(), int(l), flat)
    return tuple(row.reshape(shape) for row in out)


def bessel_wavevectors(l: int, r_cut: float, n: int = DEFAULT_N_BESSEL
                       ) -> np.ndarray:
    r"""Wave vectors of the Bessel basis: the interleaved zeros of
    :math:`j_l(x)` and :math:`j_l'(x)` at :math:`x = q r_c`, in ascending order.

    The zeros of a spherical Bessel function and of its derivative interlace,
    so the resulting set is evenly graded in :math:`q` and the basis functions
    alternate between contributing the value and the slope at :math:`r_c` --
    the matrix of matching conditions (value and any number of derivatives)
    has full rank.  The RRKJ choice (all functions sharing the all-electron
    logarithmic derivative at :math:`r_c`) is *not* usable beyond the second
    derivative: the third-derivative row becomes a combination of the value
    and second-derivative rows.
    """
    return np.array(_native.paw_bessel_wavevectors(int(l), float(r_cut),
                                                   int(n)))


def _with_origin(r: np.ndarray):
    """The atomic grid (which starts at one step) with ``r = 0`` prepended."""
    return np.concatenate([[0.0], r])


def numerov_outward(r0: np.ndarray, f: np.ndarray, s: np.ndarray,
                    u_start: tuple, start: int = 1) -> np.ndarray:
    r"""Integrate :math:`u'' = f(r)\,u + s(r)` outward on a uniform grid.

    ``u_start = (u[start], u[start+1])`` seeds the recursion; nothing at
    indices below ``start`` is used.
    """
    from ..basis.radial_backend import numerov_outward_kernel

    h2 = float((r0[1] - r0[0]) ** 2)
    u = np.zeros_like(r0, dtype=float)
    u[start], u[start + 1] = u_start
    return numerov_outward_kernel(np.asarray(f, dtype=float),
                                  np.asarray(s, dtype=float), h2, int(start),
                                  u)


def _radial_f(r0: np.ndarray, potential0: np.ndarray, l: int,
              energy: float) -> np.ndarray:
    """``f = 2[V + l(l+1)/2r^2 - E]`` on the grid with the origin prepended."""
    return _native.paw_radial_f(_array(r0), _array(potential0), int(l),
                                float(energy))


def scattering_wave(r: np.ndarray, potential: np.ndarray, l: int,
                    energy: float, z_eff: float, treatment: str = "none",
                    kappa: int | None = None) -> np.ndarray:
    """Outward Numerov solution ``u(r)`` at ``energy`` (arbitrary scale)."""
    return _native.paw_scattering_wave(context(), _array(r), _array(potential),
                                       int(l), float(energy), float(z_eff),
                                       str(treatment), kappa)


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
    u, energy = _native.paw_bound_state(
        context(), _array(r), _array(potential), int(l), float(energy_guess),
        float(z_eff), float(window), str(treatment), kappa)
    return u, float(energy)


def log_derivative_ae(r: np.ndarray, potential: np.ndarray, l: int,
                      energy: float, r_cut: float, z_eff: float,
                      treatment: str = "none",
                      kappa: int | None = None) -> float:
    r"""All-electron logarithmic derivative :math:`R'/R` at ``r_cut``."""
    return float(_native.paw_log_derivatives_ae(
        context(), _array(r), _array(potential), int(l), [float(energy)],
        float(r_cut), float(z_eff), str(treatment), kappa)[0])


def _log_derivative_of_u(r, u, r_cut) -> float:
    """``R'/R = u'/u - 1/r`` at ``r_cut`` from a fourth-order stencil."""
    return _native.paw_log_derivative_of_u(_array(r), _array(u), float(r_cut))


def constrained_minimum(K: np.ndarray, k: np.ndarray, A: np.ndarray,
                        b: np.ndarray, G: np.ndarray, norm: float
                        ) -> np.ndarray:
    r"""Minimize :math:`c^TKc + 2k^Tc` subject to :math:`Ac = b` and
    :math:`c^TGc = \text{norm}`.

    The linear constraints are eliminated exactly, :math:`c = c_0 + Z y`; in
    the null space the problem is a quadratic objective on an ellipsoid, whose
    global minimizer is the stationary point with the multiplier below the
    smallest generalized eigenvalue (More and Sorensen), a one-dimensional
    root; when the norm stays finite at the pole the *hard case* applies.
    """
    return _native.paw_constrained_minimum(
        _array(K), _array(k), np.atleast_2d(_array(A)), _array(b), _array(G),
        float(norm))


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


class DiagnosticError(RuntimeError):
    """A check of a dataset could not be evaluated: a level, a logarithmic
    derivative or a potential is not finite, or there is no all-electron
    reference atom to compare with.  Never a pass: ``max(0.0, nan)`` is
    ``0.0`` and every comparison with NaN is false, so an unevaluated check
    would otherwise read as a perfect one."""


def _reference(pp, check: str):
    """The all-electron reference atom on ``pp``, or :class:`DiagnosticError`
    naming the ``check`` that cannot run without it."""
    atom = getattr(pp, "atom", None)
    if atom is None:
        raise DiagnosticError(
            f"{getattr(pp, 'symbol', '?')}: {check} not evaluated: the "
            "dataset has no all-electron reference atom (one read from a file "
            "has none).  Attach one, pp.atom = reference_atom(pp), or skip "
            "the check explicitly (ghost_errors(..., with_atom=False)).")
    return atom


def _finite(value, pp, what: str) -> float:
    """``value`` as a float, or :class:`DiagnosticError` naming the element
    and ``what`` was evaluated (channel, energy)."""
    value = float(value)
    if not math.isfinite(value):
        raise DiagnosticError(
            f"{getattr(pp, 'symbol', '?')}: {what} is {value}; a check that "
            "cannot be evaluated is a failure, not a pass")
    return value


def _snapped(pp, radius: float) -> float:
    """``radius`` on the dataset's own grid.  A loaded dataset's grid is the
    generation grid at the library stride, while the reference atom keeps
    the full one: both logarithmic derivatives must be taken at the same
    node."""
    return _native.paw_snap(_array(pp.r), float(radius))


def reference_atom(pp, points: int | None = None, r_max: float = 30.0):
    """The all-electron atom :func:`~.paw.generate_paw` builds ``pp`` on,
    solved again from the dataset's own record (element, functional,
    relativity) -- what a dataset read from a file needs before the checks
    against the atom (:func:`scattering_errors`,
    :func:`unprojected_scattering_errors`, :func:`local_potential_ghosts`)
    can run on it."""
    from ..basis.atomic_solver import solve_atom

    z = int(pp.atomic_number)
    return solve_atom(z, points=(generation_points(z) if points is None
                                 else int(points)),
                      r_max=float(r_max), tolerance=REFERENCE_TOLERANCE,
                      mixing=REFERENCE_MIXING, xc=pp.xc,
                      relativity=pp.relativity)


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
    on every load -- the library, a path or a user's own folder."""
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


def _core_shells(pp, atom) -> dict:
    """The core subshells of the all-electron atom on ``pp`` that the
    dataset leaves out of its valence."""
    _valence, core = _valence_configuration(
        int(pp.atomic_number), configuration=atom.occupations)
    core = dict(core)
    for orbital in getattr(pp, "frozen_subshells", ()):
        core[orbital] = atom.occupations[orbital]
    for orbital in getattr(pp, "semicore_subshells", ()):
        core.pop(orbital, None)
    return core


def _ae_treatment(pp) -> str:
    """The all-electron equation a check of ``pp`` is made with.  A Dirac
    dataset's ordinary channels are the j average -- PAW-LCAO's partial waves
    solve the scalar (kappa = -1) equation outright -- and its spin-orbit part
    is a separate term these checks do not see, so their phases are compared
    with the scalar all-electron wave."""
    treatment = str(getattr(pp, "relativity", "none")).strip().lower()
    return ("scalar" if treatment in ("dirac", "full", "relativistic")
            else treatment)


def local_potential_ghosts(pp) -> dict:
    """``{l: eps_ps - eps_ae}`` for every channel *without* projectors
    (``l`` up to one above the highest channel) in which the screened local
    potential alone binds more states than the all-electron atom has valence
    states of that ``l``.

    Such a channel is governed by ``v_local_screened`` only, so the spectral
    test of :func:`ghost_errors` on the constructed channels never sees it --
    iron's PAW-LCAO ``p`` channel held a level at -4.63 Ha against a 4p at
    -0.05 Ha while every constructed channel was clean.  Both spectra are
    counted below zero in the same box, on the reference atom's uniform grid;
    the all-electron count less that ``l``'s core shells is the number of
    states the local potential may bind.  Needs the all-electron atom on
    ``pp`` (:func:`reference_atom` for a dataset read from a file), and
    raises :class:`DiagnosticError` without one: an empty result means
    checked and clean, never "not evaluated".
    """
    atom = _reference(pp, "the local-potential ghost check")
    r = _array(atom.r)
    core = _core_shells(pp, atom)
    smooth_v = _native.paw_interpolate_linear(r, _array(pp.r),
                                              _array(pp.v_local_screened))
    ae_v = _array(atom.v_effective)

    def bound(v, what, l):
        try:
            return _native.paw_bound_levels(context(), r, v, l, None, 0.0)
        except ValueError:
            _finite(float("nan"), pp, what)

    out = {}
    for l in range(max(pp.channels) + 2):
        if l in pp.channels:
            continue
        smooth = bound(smooth_v, "the screened local potential", l)
        ae = bound(ae_v, "the all-electron potential", l)
        allowed = ae[sum(1 for (_n, lc) in core if lc == l):]
        if len(smooth) > len(allowed):
            reference = float(allowed[0]) if allowed else 0.0
            out[int(l)] = float(smooth[0]) - reference
    return out


def all_electron_count(pp, l: int, energy: float) -> int:
    """Bound states of the all-electron atom on ``pp`` below ``energy`` in
    channel ``l``, its core (and frozen) shells not counted: the nodes of
    the regular solution at ``energy`` (the oscillation theorem), solved with
    the treatment the dataset was built from."""
    atom = _reference(pp, "the all-electron state count")
    nodes, total = _native.paw_node_count(
        context(), _array(atom.r), _array(atom.v_effective), int(l),
        float(energy), float(pp.atomic_number), _ae_treatment(pp))
    _finite(total, pp, f"the l={l} all-electron wave at {energy:+.3f} Ha")
    core = _core_shells(pp, atom)
    return nodes - sum(1 for (_n, lc) in core if lc == int(l))


def ghost_errors(pp, levels, *, with_atom: bool = True) -> dict:
    """``{l: eps_0 - eps_ref}`` for every channel holding a ghost state.

    ``levels(pp, l)`` returns the lowest eigenvalues of the channel's
    pseudo Hamiltonian (two, or three for a semicore channel).  A ghost is an *extra* state: the lowest level lies
    more than :data:`GHOST_TOLERANCE` below the reference and the second one
    is closer to the reference than the first. For frozen-core scattering-only
    channels, any bound pseudo level is an extra state and therefore a ghost.

    Two checks need the all-electron atom on ``pp``.  *Excited* ghosts:
    a channel holding more levels below -:data:`EXCITED_GHOST_MARGIN` than
    the atom has valence states there (:func:`all_electron_count`) holds a
    spurious one above its reference, reported as ``{l: that level}``.  A
    channel whose second reference energy is still bound (a compact semicore
    shell 1 Ha below zero) can put one just above it, outside every
    scattering window: the 4f channels of W-Hg (W -0.15, Au -2.07 Ha), the
    4p of Tc (-0.60 Ha) and the LDA 5p of Ta (-0.46 Ha) in the libraries of
    2026-10-11 (HISTORY.md, "The PAW audit report checked").  And the
    channels without projectors (:func:`local_potential_ghosts`).
    ``with_atom=False`` leaves both out on purpose, for a dataset read from
    a file that has none.  A level that is not finite raises
    :class:`DiagnosticError`.
    """
    count = getattr(pp, "all_electron_count", None) if with_atom else None
    out = {}
    for l, channel in pp.channels.items():
        reference = float(channel.reference_energies[0])
        if reference >= 0.0:
            if getattr(pp, "frozen_subshells", ()):
                first = _finite(levels(pp, l)[0], pp, f"the l={l} lowest level")
                if first < -GHOST_TOLERANCE:
                    out[int(l)] = first
            continue
        spectrum = [_finite(e, pp, f"an l={l} level") for e in levels(pp, l)]
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
        if count is not None:
            deep = sorted(e for e in spectrum if e < -EXCITED_GHOST_MARGIN)
            allowed = max(int(count(l, -EXCITED_GHOST_MARGIN)), 0)
            if len(deep) > allowed:
                out[int(l)] = min(out.get(int(l), 0.0), deep[allowed])
    # Channels without projectors (PAW-LCAO): the local potential alone.
    unconstructed = getattr(pp, "unconstructed_ghosts", None)
    if with_atom and unconstructed is not None:
        out.update(unconstructed())
    return out


def _occupied_references(channel) -> list[float]:
    """The reference energies of a channel's occupied partial waves, first
    one always included (a library channel without per-wave occupations
    has one bound reference)."""
    energies = [float(e) for e in channel.reference_energies]
    occupations = getattr(channel, "occupations", None) or [1.0]
    return [energies[0]] + [e for e, o in zip(energies[1:], occupations[1:])
                            if o > 0.0]


def _ae_log_derivatives(atom, pp, l, energies, r_match, treatment, cache):
    """All-electron logarithmic derivatives at ``energies``: from ``cache``
    (keyed by reference atom, channel, energy, matching radius and treatment)
    where they were computed before, natively otherwise."""
    keys = [(id(atom), l, float(e), r_match, treatment) for e in energies]
    missing = [(e, key) for e, key in zip(energies, keys)
               if cache is None or key not in cache]
    fresh = {}
    if missing:
        values = _native.paw_log_derivatives_ae(
            context(), _array(atom.r), _array(atom.v_effective), int(l),
            [float(e) for e, _k in missing], float(r_match),
            float(pp.atomic_number), treatment)
        fresh = {key: float(v) for (_e, key), v in zip(missing, values)}
        if cache is not None:
            cache.update(fresh)
    return [fresh[key] if key in fresh else cache[key] for key in keys]


def _phase_summary(pp, label, l, energies, offsets, atom, log_derivative,
                   r_match, treatment, cache, window):
    """``(near, far)`` phase errors over ``energies``; a phase that is not
    finite raises :class:`DiagnosticError`."""
    ae = _ae_log_derivatives(atom, pp, l, energies, r_match, treatment, cache)
    ps = [log_derivative(pp, l, energy, r_match) for energy in energies]
    near, far, bad = _native.paw_phase_summary(
        _array(ps), _array(ae), _array(offsets), float(window))
    if bad is not None:
        _finite(float("nan"), pp,
                f"the l={l} {label} at {energies[bad]:+.3f} Ha")
    return near, far


def scattering_errors(pp, log_derivative,
                      ae_cache: dict[tuple, float] | None = None) -> dict:
    r"""``{l: (near, far)}``: the largest phase error
    :math:`|\arctan L_{ps} - \arctan L_{ae}|` at each channel's
    :math:`r_c`, within :math:`\varepsilon_{ref} \pm` :data:`PHASE_WINDOW`
    (``near``) and :data:`RESONANCE_WINDOW` (``far``) Hartree, wrapped
    modulo :math:`\pi` so a pole of :math:`L` is not an error.  Needs the
    all-electron atom on ``pp`` (:class:`DiagnosticError` without one, and
    for a phase that is not finite). A channel whose first reference is a
    scattering energy (the empty f above a frozen 4f, potassium's d
    anchored at 0 Ha) is evaluated at the positive energies of the same
    windows; such a channel without a frozen subshell was once skipped.
    ``ae_cache`` reuses all-electron logarithmic derivatives at the same
    channel, energy and matching radius across ghost-repair attempts; it is
    local to one reference atom.
    """
    atom = _reference(pp, "the scattering check")
    treatment = _ae_treatment(pp)
    # Compare where the smooth wave obeys the all-electron equation again:
    # past the projectors, which for PAW-LCAO can reach beyond r_cut.
    radius = getattr(pp, "projector_radius", None)
    offsets = [float(o) for o in
               _native.paw_window_offsets(RESONANCE_WINDOW, PHASE_STEP)]
    out = {}
    for l, channel in pp.channels.items():
        reference = float(channel.reference_energies[0])
        r_match = _snapped(pp, radius(l) if radius is not None
                           else channel.r_cut)
        near = far = 0.0
        # Around every occupied reference: a semicore channel's valence
        # state (sodium's 3s) sits 2 Ha above its first reference.
        for center in [reference] + _occupied_references(channel)[1:]:
            kept = [(center + offset, offset) for offset in offsets
                    if not (reference >= 0.0 and center + offset <= 0.0)]
            if not kept:
                continue
            n, f = _phase_summary(
                pp, "phase error", l, [e for e, _o in kept],
                [o for _e, o in kept], atom, log_derivative, r_match,
                treatment, ae_cache, PHASE_WINDOW)
            near, far = max(near, n), max(far, f)
        out[int(l)] = (float(near), float(far))
    return out


def unprojected_scattering_errors(pp, log_derivative,
                                  ae_cache: dict[tuple, float] | None = None
                                  ) -> dict:
    r"""``{l: error}``: the largest phase error of every channel *without*
    projectors (``l`` up to one above the highest constructed channel),
    which scatters off the screened local potential alone, within
    :data:`PHASE_WINDOW` of the highest valence reference energy.

    The phases are compared past the local potential's radius and every
    channel's cutoff, where both waves obey the same equation again.  Needs
    the all-electron atom on ``pp`` (:func:`reference_atom` for a dataset
    read from a file); without one, or for a phase that is not finite, it
    raises :class:`DiagnosticError`.  This is the test the constructed
    channels' own (:func:`scattering_errors`) never made: aluminum repaired
    with a 5 Ha local shift scattered d 0.84 rad off, and that alone put fcc
    Al at 4.48 Angstrom (HISTORY.md, 2026-10-06, "K21 diagnosed").
    """
    atom = _reference(pp, "the unprojected-channel scattering check")
    treatment = _ae_treatment(pp)
    bound = [e for channel in pp.channels.values()
             for e in _occupied_references(channel) if e < 0.0]
    if not bound:
        raise DiagnosticError(
            f"{getattr(pp, 'symbol', '?')}: the unprojected-channel "
            "scattering check not evaluated: no bound occupied reference to "
            "center its energy window on")
    reference = max(bound)
    r_match = _snapped(pp, max(
        [float(getattr(pp, "r_cut_local", 0.0))]
        + [float(channel.r_cut) for channel in pp.channels.values()]))
    offsets = [float(o) for o in
               _native.paw_window_offsets(PHASE_WINDOW, PHASE_STEP)]
    out = {}
    for l in range(max(pp.channels) + 2):
        if l in pp.channels:
            continue
        _near, far = _phase_summary(
            pp, "(no projectors) phase error", l,
            [reference + offset for offset in offsets], offsets, atom,
            log_derivative, r_match, treatment, ae_cache, PHASE_WINDOW)
        out[int(l)] = float(far)
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
    A check that cannot be evaluated (:class:`DiagnosticError`: a level or
    phase that is not finite) fails a construction like a defect, but such a
    construction is never returned, not even flagged.
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

    # A check that cannot be evaluated (a level or a phase that is not
    # finite) fails the construction like any other defect, but it cannot be
    # ranked: such a construction is never returned, flagged or not.
    unevaluated = None
    try:
        errors = ghost_errors(first, levels)
        if not errors:
            # No ghost is not enough: the first construction has to scatter
            # like the atom too, or it is repaired like a ghosted one.
            # (Aluminum's p channel was once returned 0.95 rad off,
            # untested.)
            wrong = _wrong_phases(scattering_errors(first, log_derivative,
                                                    phase_cache))
            extra = defects(first)
            if not wrong and not extra:
                return first
        else:
            wrong, extra = {}, {}
    except DiagnosticError as error:
        unevaluated = str(error)
        errors, wrong, extra = {}, {}, {}
    if unevaluated is not None:
        problem = f"a check that could not be evaluated ({unevaluated})"
    elif errors:
        problem = f"ghost state below the reference ({_describe(errors)})"
    elif wrong:
        problem = "wrong scattering (" + ", ".join(
            f"l={l} {near:.3f}/{far:.3f} rad"
            for l, (near, far) in sorted(wrong.items())) + ")"
    else:
        problem = f"an incomplete construction ({_describe_defects(extra)})"
    pinned = [name for name in ("r_cut", "r_cut_local", "local_shift")
              if options.get(name) is not None]
    if (pinned and unevaluated is None and not errors and not wrong
            and mode != "refuse"):
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
            if mode == "flag" and unevaluated is None else None)

    def keep(candidate):
        nonlocal best
        if mode == "flag" and (best is None or _defect_badness(candidate)
                               < _defect_badness(best)):
            best = candidate

    for label, remedy in attempts:
        trial = dict(options, atom=first.atom, **remedy)
        try:
            pp = generate(symbol, **trial, ghosts="keep")
        except (ValueError, RuntimeError) as error:
            tried.append(f"{label}: {type(error).__name__}")
            continue
        try:
            remaining = ghost_errors(pp, levels)
            wrong = found = {}
            if not remaining:
                wrong = _wrong_phases(scattering_errors(pp, log_derivative,
                                                        phase_cache))
                found = defects(pp)
        except DiagnosticError as error:
            tried.append(f"{label}: {error}")
            continue
        if remaining:
            tried.append(f"{label}: ghost {_describe(remaining)}")
            keep((pp, remaining, {}))
            continue
        if wrong:
            keep((pp, {}, wrong, found))
            tried.append(f"{label}: phase " + ", ".join(
                f"l={l} {near:.3f}/{far:.3f} rad"
                for l, (near, far) in sorted(wrong.items())))
            continue
        if found:
            keep((pp, {}, {}, found))
            tried.append(f"{label}: {_describe_defects(found)}")
            continue
        return pp
    if mode == "flag" and best is not None:
        return _least_defective([best])
    raise GhostStateError(
        f"{symbol}: {problem} and "
        f"no remedy removed it while keeping the scattering "
        f"({'; '.join(tried)}).")


def _resample(r: np.ndarray, values: np.ndarray, r_new: np.ndarray):
    """Cubic-spline resampling (linear interpolation would leave an O(h^2)
    error that shows up as an asymmetry of the Vanderbilt matrix)."""
    return _native.paw_resample(context(), _array(r), _array(values),
                                _array(r_new))


def _inner_grid(r_cut: float) -> np.ndarray:
    """The fine quadrature grid inside ``r_cut`` (``INNER_POINTS`` nodes)."""
    return _native.paw_inner_grid(float(r_cut))


def _bessel_table(l: int, qs: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``(len(qs), len(r))`` table of ``j_l(q_n r)``."""
    return _native.paw_bessel_table(context(), int(l), _array(qs), _array(r))


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


def matching_targets(r: np.ndarray, u: np.ndarray, potential: np.ndarray,
                     l: int, energy: float, r_cut: float,
                     treatment: str = "none", kappa: int | None = None,
                     z_eff: float = 0.0) -> np.ndarray:
    r"""``R`` and its first three derivatives at ``r_cut`` (a grid point).

    ``R'`` comes from a fourth-order stencil on ``u = rR``; the second and
    third derivatives follow from the radial equation the wave actually
    satisfies, :math:`u'' = a\,u + b\,u'` with
    :math:`a = l(l+1)/r^2 + \kappa M'/(Mr) + 2M(V-\varepsilon)` and
    :math:`b = M'/M` (:math:`M\equiv1` non-relativistically), so the targets
    are consistent with it to the accuracy of the Numerov solution itself.
    """
    return _native.paw_matching_targets(
        context(), _array(r), _array(u), _array(potential), int(l),
        float(energy), float(r_cut), str(treatment), kappa, float(z_eff))


def inner_overlaps(r, waves, r_cut):
    """``<phi_i|phi_j>`` inside ``r_cut``, on the refined inner grid: the
    *true* all-electron overlap, whatever equation the waves solve."""
    return _native.paw_inner_overlaps(context(), _array(r),
                                      _array(np.atleast_2d(waves)),
                                      float(r_cut))


def norm_targets(r, waves, energies, r_cut, v_ae=None, treatment="none",
                 kappa=None, z_eff=0.0):
    r"""The generalized-norm matrix the pseudo waves must reproduce.

    Non-relativistically this is just the inner overlap.  When the
    all-electron waves solve a relativistic equation and the pseudo waves
    solve a non-relativistic one, what has to be enforced is that the *pseudo*
    system reproduce the all-electron logarithmic derivative **and its energy
    derivative** at :math:`r_c`, which is a Wronskian,
    :math:`-W_{ij}(r_c)/2(\varepsilon_j-\varepsilon_i)`, off the diagonal,
    and :math:`M(r_c)\int(2M-1)u^2/M\,dr` on it.
    """
    return _native.paw_norm_targets(
        context(), _array(r), _array(np.atleast_2d(waves)),
        [float(e) for e in energies], float(r_cut),
        _array(np.zeros_like(r) if v_ae is None else v_ae), str(treatment),
        kappa, float(z_eff))


def optimize_pseudo_waves(r: np.ndarray, v_ae: np.ndarray, l: int,
                          waves: list, energies: list, r_cut: float,
                          q_cut: float = DEFAULT_Q_CUT,
                          n_bessel: int = DEFAULT_N_BESSEL,
                          norm_factor: float | None = 1.0,
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
    f\,\langle\varphi_i|\varphi_j\rangle_{r<r_c}`: 1 is norm conservation;
    the PAW-LCAO family uses :math:`f < 1` so its overlap correction
    :math:`(1-f)\langle\varphi_i|\varphi_j\rangle` is positive definite by
    construction.  ``norm_factor=None`` imposes no norm condition at all: the
    minimum of the residual energy in the null space of the matching
    conditions, regularized by a tiny multiple of the metric.
    """
    out = _native.paw_optimize_pseudo_waves(
        context(), _array(r), _array(v_ae), int(l),
        _array(np.atleast_2d(waves)), [float(e) for e in energies],
        float(r_cut), float(q_cut), int(n_bessel),
        None if norm_factor is None else float(norm_factor), str(treatment),
        kappa, float(z_eff))
    coefficients = list(out["coefficients"])
    return PseudoWaves(
        l=int(l), r_cut=float(out["r_cut"]), energies=list(energies),
        waves=list(waves),
        wavevectors=[out["wavevectors"]] * len(coefficients),
        coefficients=coefficients,
        residual_kinetic=list(out["residual_kinetic"]), norms=out["norms"],
        achieved=out["achieved"], q_cut=float(q_cut), targets=out["targets"])


#: Half-width of the fit that reads a gradient-corrected potential's
#: derivatives at :math:`r_{cl}` (:func:`polynomial_local_potential`), as a
#: fraction of :math:`r_{cl}`: +-15 of the GGA's derivative nodes
#: (:data:`~mandacaru.basis.xc.GGA_DERIVATIVE_STEP` apart).  Its potential
#: is a cubic spline through those nodes, so a fit over the default +-25
#: grid points spans two or three of them and reads their kinks as a fourth
#: derivative: PBE aluminum's at r_cl = 3.09 Bohr came out -51 (LDA -0.31)
#: and its local potential -13.4 Ha deep at the origin (LDA -1.5); at 1.39
#: Bohr -14800, -152 Ha and a ghost 112 Ha deep.  Read over 0.1-0.2 r_cl the
#: PBE derivatives agree with one another and the potential with LDA's
#: (HISTORY.md, 2026-10-07, "The PBE set").  An LDA potential reads the same
#: over any of these windows and keeps the grid-point fit.
GGA_LOCAL_FIT_FRACTION = 0.15


def polynomial_local_potential(r: np.ndarray, v_ae: np.ndarray,
                               r_local: float, shift: float = 0.0,
                               fit_window: float | None = None
                               ) -> np.ndarray:
    r"""Even polynomial continuation of ``v_ae`` inside ``r_local``.

    :math:`V(r) = \sum_{k} a_k r^{2k}` for :math:`r < r_{cl}`, with the
    coefficients fixed by continuity of the value and the first four
    derivatives at :math:`r_{cl}` (five even powers) and, when ``shift`` is
    nonzero, a sixth power raising the value at the origin by ``shift``
    Hartree above the five-coefficient continuation -- Hamann's ``dvloc0``,
    the knob that sets how strongly the projectors have to act.
    :math:`V'(0) = 0` by parity, so the potential is smooth at the origin.

    The derivatives are read off a polynomial fit over +-25 grid points
    around :math:`r_{cl}`, or over +-``fit_window`` Bohr when that is wider
    (a gradient-corrected potential, :data:`GGA_LOCAL_FIT_FRACTION`).
    """
    return context().paw_local_potential(
        _array(r), _array(v_ae), float(r_local), shift=float(shift),
        fit_window=None if fit_window is None else float(fit_window))


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
    return _native.paw_snap(_array(r), float(radius))


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
    peak = _native.paw_peak_radius(_array(r), _array(radial))
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


def reference_waves(symbol, atom, valence_config, z_eff, r_cut, rc_factor,
                    energy_offset, defaults=None, treatment: str = "none",
                    kappa: int | None = None, extra_l: int = 0,
                    extra_energy: float | None = None,
                    second_references: dict | None = None):
    """``(per_l, cutoffs, references)`` -- the all-electron input of a channel.

    ``per_l[l]`` lists every occupied valence state ``(n, energy, R, occupancy)``
    (bound, Numerov-refined); ``cutoffs[l]`` is the cutoff radius snapped to the
    atomic grid, so "inside r_c" and the matching point are the same node; and
    ``references[l] = (waves, energies)`` are the two partial waves each family
    pseudizes -- the bound state(s) plus, when there is only one, the scattering
    state ``energy_offset`` above it, normalized inside the sphere.
    ``second_references`` (``{l: energy}``, Hartree) places that second
    reference at an absolute energy instead, for the listed channels with one
    bound state.

    ``treatment`` and ``kappa`` select the radial equation
    (:mod:`mandacaru.basis.relativity`); ``kappa`` is required for
    ``treatment="dirac"`` and then names which :math:`j` this set of channels
    belongs to.  The valence state is always the Numerov shoot, at the level
    of theory the reference atom was solved with: the pseudization needs
    waves that are eigenstates of the potential it was handed to fourth
    order, which a tridiagonal wave of the same potential is not (it left an
    s channel with a ghost 56 Hartree below the reference).

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
    r, v_ae = _array(atom.r), _array(atom.v_effective)

    def kappa_of_l(l):
        """``kappa`` may be one value or a ``{l: kappa}`` map (one j branch)."""
        return kappa.get(l) if isinstance(kappa, dict) else kappa

    per_l: dict = {}
    for (n, l), occupancy in sorted(valence_config.items()):
        k = kappa_of_l(l)
        guess = atom.eigenvalues[(n, l)]
        if treatment == "dirac" and atom.eigenvalues_j:
            guess = atom.eigenvalues_j.get((n, l, k), guess)
        wave, energy = _native.paw_bound_state(
            context(), r, v_ae, int(l), float(guess), float(z_eff), 5e-3,
            str(treatment), k, True)
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
        per_l.setdefault(l, []).append((n, energy, wave, occupancy))

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
            if second_references and l in second_references:
                energy_2 = float(second_references[l])
                if not energy_2 > energies[0]:
                    raise ValueError(
                        f"{symbol} l={l}: the second reference energy "
                        f"{energy_2:+.4f} Ha is not above the bound state "
                        f"({energies[0]:+.4f} Ha)")
            waves.append(_scattering_reference(
                r, v_ae, l, energy_2, z_eff, cutoffs[l], treatment,
                kappa_of_l(l)))
            energies.append(energy_2)
        references[l] = (waves[:2], energies[:2])

    if int(extra_l) > 0:
        _add_extra_channels(per_l, cutoffs, references, int(extra_l),
                            float(energy_offset), atom, z_eff, treatment,
                            extra_energy=extra_energy, kappa=kappa)
    return per_l, cutoffs, references


def _scattering_reference(r, v_ae, l, energy, z_eff, radius, treatment,
                          kappa=None):
    """The scattering wave at ``energy`` as a radial function ``R``,
    normalized inside ``radius``."""
    return _native.paw_reference_waves(
        context(), r, v_ae, int(l), float(energy), float(z_eff),
        float(radius), str(treatment), kappa)


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
                        atom, z_eff, treatment,
                        extra_energy: float | None = None, kappa=None):
    """Append ``extra_l`` unoccupied channels above the highest valence
    :math:`l`, both references scattering states at the widest cutoff."""
    r, v_ae = _array(atom.r), _array(atom.v_effective)
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
        waves = [_scattering_reference(r, v_ae, l, energy, z_eff, widest,
                                       treatment, k)
                 for energy in energies]
        references[l] = (waves, energies)
        # A scattering reference contributes no valence charge.
        per_l[l] = [(l + 1, energies[0], waves[0], 0.0)]


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
    return list(_native.paw_spurious_nodes(
        int(pw.l), float(pw.r_cut), _array(r), _array(pw.waves),
        _array(pw.wavevectors), _array(pw.coefficients)))


def _spectrum_extent(pp, l: int, floor: float = 1e-5,
                     bounds=(12.0, 24.0)) -> float:
    """Box radius for :func:`radial_spectrum`: where the outermost occupied
    pseudo wave has decayed to ``floor`` of its maximum (a diffuse Li 2s
    reaches 18 Bohr; a semicore channel's valence wave, not its compact
    semicore one)."""
    channel = pp.channels[int(l)]
    occupied = len(_occupied_references(channel))
    return _native.paw_spectrum_extent(
        _array(pp.r), _array(channel.pseudo_waves[occupied - 1]),
        float(floor), float(bounds[0]), float(bounds[1]))


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
    treatment = str(getattr(pp, "relativity", "none"))
    probes = list(energies)
    if midpoint:
        probes.append(0.5 * (energies[0] + energies[1]))
    r_cut = _snapped(pp, r_cut)
    l_ae = _native.paw_log_derivatives_ae(
        context(), _array(ae.r), _array(ae.v_effective), int(l),
        [float(e) for e in probes], float(r_cut), float(pp.atomic_number),
        treatment, kappa)
    return {float(energy): (float(abs(pseudo_log_derivative(pp, l, energy)
                                      - ae_value)), float(ae_value))
            for energy, ae_value in zip(probes, l_ae)}
