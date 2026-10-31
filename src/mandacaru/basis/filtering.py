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
differentiates that artifact.  Measured on water / PAW-LCAO-SZ, the net force
(which a free molecule's exact force must make vanish) is 1.85 / 0.86 / 0.41 /
0.099 / 0.025 eV/Angstrom at ``grid_spacing`` = 0.30 / 0.25 / 0.20 / 0.16 / 0.13.

The standard cure in real-space codes is to remove the unrepresentable
components from the *radial functions*, once, **before** they are ever sampled
(Anglada & Soler, PRB 73, 115122 (2006); the mask-function filter of Wang,
and of Tafipolsky & Schmid, JCP 124, 174102 (2006), also used for
projectors).  A band-limited function is
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
  the forces badly once already (the polarization-shell bug), so it is
  asserted by the test suite;
* the filter commutes with the angular part, so a filtered table drops into
  :class:`~mandacaru.basis.multizeta.TabulatedOrbital` unchanged and everything
  downstream -- grid sampling, the PAW-LCAO atom-centered projection quadrature, the
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
    own support radius and 0 at 1.5 times it (``SWITCH_MARGIN`` in the Rust
    filter).
``"mask"``
    Divide by a window that starts decaying *inside* the support, filter,
    multiply back -- the classic mask trick, which pre-compensates the
    multiplication so the result stays accurate near a hard confinement edge.

**Measured, at** :math:`k_c = 7.5\,a_0^{-1}` (765 eV), as the residual weight
above :math:`k_c` before -> after, and the range the result occupies
(smaller residual is better)::

    function              before     plain    switch      mask   range(switch)
    PAW-LCAO H 1s   zeta 1    1.2e-06   4.9e-07   4.9e-07   5.3e-05     17.3 Bohr
    PAW-LCAO H 1s   zeta 3    1.0e-04   3.7e-06   1.6e-05   2.0e-05      2.2
    H polarization (p)   3.4e-07   ~1e-12    ~1e-12    4.5e-08     21.8
    PAW-LCAO O 2s   zeta 1    1.3e-05   7.4e-08   7.4e-08   6.5e-07      9.5
    PAW-LCAO O 2s   zeta 3    9.3e-03   7.5e-07   2.8e-04   3.2e-04      1.7
    PAW-LCAO O 2p   zeta 1    8.4e-05   ~2e-12    2.1e-11   9.8e-07     13.7
    PAW-LCAO O 2p   zeta 3    4.7e-03   ~1e-11    2.8e-04   3.6e-04      1.9
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
water / PAW-LCAO-SZ by 832 meV at ``grid_spacing`` = 0.25 and 422 meV at ``grid_spacing`` = 0.20.  That
number looks alarming and mostly is not a loss.  Holding the cutoff **fixed**
(602 eV) and refining the grid, the filtered-to-unfiltered gap shrinks with
``grid_spacing`` -- 530 / 422 / 294 / 237 meV at ``grid_spacing`` = 0.25 / 0.20 / 0.16 / 0.13, close
to linear in ``grid_spacing`` -- so most of it is the *unfiltered* basis's own grid
error, not a smaller function space (these numbers were measured with
``kinetic="fd"``).  The finite-difference Laplacian
under-estimates the kinetic energy of a function the grid does not resolve, so
an unfiltered orbital is reported too low; filtering removes exactly the
components that were being mis-integrated.  On H2, where both bases are
resolved by ``grid_spacing`` = 0.12, the gap is 2 meV and has the *opposite* sign (the
filtered function is slightly better).

So the default is per family (:attr:`~mandacaru.pseudopotentials.families.FamilySpec.default_options`):
**on for PAW-LCAO and UPAW-LCAO**, whose smooth partial waves are built band-limited
(:func:`~mandacaru.pseudopotentials.partial_waves.optimize_pseudo_waves` minimizes the
kinetic energy beyond ``q_cut``) so the filter has little to take.
``basis={"name": "PAW-LCAO", "filter_cutoff": False}`` restores the unfiltered
basis exactly.  The measured prices and benefits are in
``docs/source/guide/pseudopotentials.md``.
"""

from __future__ import annotations

import hashlib
import threading
import warnings
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from .. import _native
from ..units import EV_TO_HARTREE

#: ``filter_cutoff="auto"`` / ``True`` puts the cutoff at this fraction of the grid's
#: Nyquist wave-vector :math:`\pi/h` -- i.e. **exactly at** it.  The principle
#: is the whole point of the filter: remove what the grid cannot represent and
#: keep everything it can.  It is also what the measurement says.  Water /
#: PAW-LCAO-SZ, peak-to-peak energy over one grid period under a rigid shift along
#: (1,1,1), against the variational cost (the rise in :math:`E_\text{RHF}`):
#:
#: ::
#:
#:               grid_spacing = 0.25 A   grid_spacing = 0.20 A
#:     k_c/k_N   ripple(meV) cost(meV)   ripple(meV) cost(meV)
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

#: The filter methods, the default (``"switch"``) first: defined once, in Rust
#: (``mandacaru_core::basis::filter``), together with the numerical constants of
#: the transform (the roll-off, the quadrature oversampling, the support tail,
#: the switch margin and the mask's inner fraction) and the measurements that
#: chose them.
FILTER_METHODS = tuple(_native.RADIAL_FILTER_METHODS)

#: The default, chosen on the measurements tabulated in the module docstring.
DEFAULT_FILTER_METHOD = FILTER_METHODS[0]

#: Warn when the filter throws away more than this fraction of a function's
#: norm -- at that point the cutoff is not smoothing the basis, it is
#: replacing it.
FILTER_WEIGHT_WARN = 0.05


# --------------------------------------------------------------------------- #
# The option: what ``basis={"name": "PAW-LCAO", "filter_cutoff": ...}`` may say.
# --------------------------------------------------------------------------- #

def validate_filter_cutoff(spec):
    """Normalize a ``filter_cutoff`` basis option; raise ``ValueError`` on
    anything else.

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
            f"unknown filter_cutoff {spec!r}; use 'auto' or True to tie the "
            "cutoff to the grid spacing, a positive number for an explicit "
            "kinetic-energy cutoff in eV, or False to switch filtering off")
    if isinstance(spec, (int, float, np.integer, np.floating)):
        value = float(spec)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"filter_cutoff must be a positive, finite energy in eV, got "
                f"{spec!r}")
        return value
    raise ValueError(
        f"unknown filter_cutoff {spec!r} (type {type(spec).__name__}); use "
        "'auto' / True, a positive kinetic-energy cutoff in eV, or False")


def resolve_filter_cutoff(spec, spacing_bohr=None) -> float | None:
    """The cutoff wave-vector :math:`k_c` in Bohr\\ :sup:`-1`, or ``None`` when off.

    ``spacing_bohr`` is the **coarsest** grid spacing (the worst axis of an
    anisotropic grid is what limits the representable band), needed only for
    the ``True`` / ``"auto"`` form.
    """
    spec = validate_filter_cutoff(spec)
    if spec is False:
        return None
    if spec is True:
        if spacing_bohr is None:
            raise ValueError(
                "filter_cutoff='auto' needs the grid spacing; pass "
                "spacing_bohr (or "
                "give an explicit cutoff in eV)")
        spacing = float(spacing_bohr)
        if spacing <= 0.0:
            raise ValueError(f"the grid spacing must be positive, got {spacing!r}")
        return FILTER_NYQUIST_FRACTION * np.pi / spacing
    return float(np.sqrt(2.0 * float(spec) * EV_TO_HARTREE))


class FilterPin:
    """The ``filter_cutoff="auto"`` cutoff of one calculator, resolved once.

    ``"auto"`` puts :math:`k_c` at the Nyquist wave-vector :math:`\\pi/h` of the
    grid *as built*.  A crystal whose cell changes during one calculation (a
    relaxation with a cell filter, a finite-strain stress, a lattice scan on
    one calculator) rebuilds its grid every time, and a node count that
    jumps or a cell that stretches moves :math:`\\pi/h`; the Fourier-filtered
    basis functions then change from one cell to the next, which is a spurious
    force on the lattice (a negative pressure that biased the lattice
    constants of the transition metals 2-5 % low).

    The pin removes it: the **first** build of a calculator resolves
    ``"auto"`` from its own grid, exactly as before, and every later build of
    that calculator reuses that :math:`k_c`.  An explicit ``filter_cutoff``
    (a number in eV) or ``False`` is never pinned.  The pin is forgotten when
    the set of chemical symbols changes (another material needs its own
    cutoff) and by :meth:`release`.
    """

    def __init__(self):
        self.wavevector: float | None = None
        self._key: tuple | None = None

    def resolve(self, spec, spacing_bohr, symbols=()):
        """``(k_c, held)`` -- the cutoff in Bohr^-1 (``None`` when off) and
        whether it was held from an earlier build."""
        spec = validate_filter_cutoff(spec)
        if spec is not True:
            return resolve_filter_cutoff(spec, spacing_bohr), False
        key = tuple(sorted(set(symbols)))
        if self.wavevector is not None and key == self._key:
            return self.wavevector, True
        self.wavevector = resolve_filter_cutoff(True, spacing_bohr)
        self._key = key
        return self.wavevector, False

    def release(self):
        """Forget the cutoff: the next build resolves ``"auto"`` afresh."""
        self.wavevector, self._key = None, None

    @property
    def cutoff_ev(self) -> float | None:
        """The pinned cutoff in eV, or ``None`` before the first build."""
        return (None if self.wavevector is None
                else cutoff_energy_ev(self.wavevector))


def filter_label(spec) -> str:
    """One short phrase describing the option, for the dry run and the log."""
    spec = validate_filter_cutoff(spec)
    if spec is False:
        return "unfiltered"
    if spec is True:
        return (f"filtered (auto: {FILTER_NYQUIST_FRACTION:g} x Nyquist)")
    return f"filtered ({spec:g} eV)"


def cutoff_energy_ev(k_c: float) -> float:
    """The kinetic-energy cutoff in eV a wave-vector corresponds to."""
    return float(0.5 * k_c * k_c / EV_TO_HARTREE)


# --------------------------------------------------------------------------- #
# The filter: the numerics are native (``mandacaru_core::basis::filter``).
# --------------------------------------------------------------------------- #

def _table(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=float)


def _angular_momentum(l) -> int:
    l = int(l)
    if l < 0:
        raise ValueError(f"the angular momentum must be non-negative, got {l}")
    return l


def residual_weight(r, values, l: int, k_c: float, oversample=None, *,
                    context=None) -> float:
    """Fraction of :math:`\\int R^2r^2dr` carried above ``k_c`` -- what the grid
    cannot represent.  Computed natively from *below* the cutoff (never by
    integrating a truncated high-:math:`k` tail, where the quadrature of
    :math:`j_l(kr)` on the radial mesh degrades)."""
    context = _native.Context(threads=1) if context is None else context
    return 1.0 - _native.radial_band_weight(
        context, _table(r), _table(values), _angular_momentum(l), float(k_c),
        oversample)


#: ``key -> (filtered, info, warning)``: a radial table's filter is a pure
#: function of the table and the options, and every crystal build (a strained
#: cell for the stress, each relaxation step) asked for the same ones again.
#: A least-recently-used cache of at most :data:`FILTERED_CACHE_SIZE` entries.
#: An entry is one filtered table (8 bytes per radial node) plus a short
#: ``info`` dict: the longest mesh of the PAW-LCAO library, 29,625 nodes, is
#: 237 kB, so the cache never holds more than 256 x 237 kB = 61 MB, and about
#: 6 MB at the usual 3,000 nodes.
FILTERED_CACHE_SIZE = 256
_FILTERED: OrderedDict = OrderedDict()
_FILTERED_LOCK = threading.Lock()


def filter_radial(r, values, l: int, k_c: float, *, context=None, **options):
    """Remove the content above ``k_c`` from one radial table (memoized).

    Options: ``method`` (one of :data:`FILTER_METHODS`), ``rolloff``,
    ``oversample``, ``renormalize`` (rescale to the original norm, default
    ``True``), ``switch_margin`` and ``warn``; an omitted option takes the
    native default.  Returns ``(filtered_values, info)`` where ``info`` records
    the residual weight above ``k_c`` before and after, the support radius, the
    mask radius and the largest real-space change -- the numbers the guide
    quotes and the tests assert.  The returned array is a copy, free for the
    caller to modify.

    The result is rescaled to the **original** norm rather than to 1: the
    tables do not all arrive normalized (the PAW-LCAO hydrogen 1s carries
    0.979), and forcing them to 1 would make the filter change the function
    even in the ``k_c -> infinity`` limit where it must be the identity.
    """
    r, values, l, k_c = _table(r), _table(values), _angular_momentum(l), float(k_c)
    digest = hashlib.sha1(r.tobytes())
    digest.update(values.tobytes())
    key = (digest.hexdigest(), l, k_c,
           tuple(sorted((k, v) for k, v in options.items() if k != "warn")))
    with _FILTERED_LOCK:
        cached = _FILTERED.get(key)
        if cached is not None:
            _FILTERED.move_to_end(key)
    if cached is None:
        unknown = set(options) - {"method", "rolloff", "oversample",
                                  "renormalize", "switch_margin", "warn"}
        if unknown:
            raise TypeError(f"unknown filter option(s) {sorted(unknown)}")
        context = _native.Context(threads=1) if context is None else context
        filtered, report = _native.radial_filter(
            context, r, values, l, k_c,
            options.get("method"), options.get("rolloff"),
            options.get("oversample"), options.get("renormalize", True),
            options.get("switch_margin"))
        (before, after, support, outer, masked, kept, change) = report
        info = {
            "l": l, "cutoff": k_c,
            "method": options.get("method") or DEFAULT_FILTER_METHOD,
            "residual_before": float(before), "residual_after": float(after),
            "support_radius": float(support), "mask_radius": float(outer),
            "masked": bool(masked), "norm_kept": float(kept),
            "max_change": float(change),
        }
        # The warning is recorded with the entry, so that a cached call warns
        # exactly as the first one did.
        message = None
        if 1.0 - kept > FILTER_WEIGHT_WARN:
            message = (
                f"the basis filter at k_c = {k_c:.3g} Bohr^-1 "
                f"({cutoff_energy_ev(k_c):.0f} eV) removed "
                f"{100 * (1.0 - kept):.1f}% of the norm of an l = {l} radial "
                "function: at that cutoff the filter is replacing the basis, "
                "not smoothing it.  Raise the cutoff (or the grid resolution).")
        cached = (filtered, info, message)
        with _FILTERED_LOCK:
            _FILTERED[key] = cached
            while len(_FILTERED) > FILTERED_CACHE_SIZE:
                _FILTERED.popitem(last=False)
    if cached[2] is not None and options.get("warn", True):
        warnings.warn(cached[2], RuntimeWarning, stacklevel=2)
    return cached[0].copy(), dict(cached[1])


def filter_table(table, k_c: float, **kwargs):
    """Filter a :class:`~mandacaru.basis.multizeta.RadialTable` in place of itself.

    Returns a **new** table (the input is a shared, cached object -- the
    pseudopotential datasets are memoized per element, so mutating one would
    leak the filter into every later calculation in the process).
    """
    values, info = filter_radial(table.r, table.values, table.l, k_c, **kwargs)
    return replace(table, values=values), info
