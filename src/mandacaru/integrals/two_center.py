# -*- coding: utf-8 -*-
# file: integrals/two_center.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Analytic two-center integrals of atom-centered radial-times-harmonic functions.

The overlap, the kinetic energy and the projections on the PAW-LCAO projectors
involve two centers, and for :math:`\phi = f(r)\,S_{lm}(\hat r)` (real
harmonics) they need no grid.  With the Fourier-Bessel transform of each
radial function, :math:`F_l(k) = \int f\, j_l(kr)\, r^2 dr`, the integral
between :math:`f_1 S_{l_1 m_1}` and :math:`f_2 S_{l_2 m_2}` a displacement
:math:`\mathbf R` apart is

.. math::

    \sum_{LM} G^{LM}_{l_1 m_1, l_2 m_2}\; s_L(R)\; S_{LM}(\hat R) ,
    \qquad
    s_L(R) = 8\,(-1)^{(l_1 - l_2 - L)/2}\int_0^\infty
        w(k)\,F_1F_2\,j_L(kR)\,k^2\,dk ,

with :math:`G` the real Gaunt coefficients and :math:`w = 1` (overlap) or
:math:`k^2/2` (kinetic energy).  :math:`s_L(R)` is tabulated once per pair of
radial functions as a cubic spline, so an atom pair costs a lookup; the
gradient is the spline's derivative and :math:`\nabla S_{LM}`, the exact
derivative of the matrix.  Only atom pairs within the sum of the cutoffs are
visited.  The transformed function is the one the basis samples
(``radial(r)``, Fourier filter included).

The quadrature meshes and the table step below are fixed, not options.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from ..basis._angular import spherical_harmonic, spherical_harmonic_gradient

#: How the two-center matrices (overlap, kinetic energy, projections) are
#: integrated: on the real-space grid (default) or analytically.
TWO_CENTER_METHODS = ("grid", "analytic")
DEFAULT_TWO_CENTER = "grid"

#: The operators a table can hold.
KINDS = ("overlap", "kinetic")

#: Width (Bohr) and node count of the radial Gauss-Legendre panels.
R_PANEL = 0.04
R_NODES = 6

#: Panel of a function cut at a finite radius (a projector): not
#: band-limited, so the panels follow the table's spline knots (exact for each
#: cubic piece; :data:`R_PANEL` leaves 2e-7).
CUT_PANEL = 0.01

#: Panels (Bohr^-1) and upper limit of the wave-number integral; a filtered
#: function has no content near :data:`K_MAX`.
K_PANEL = 0.2
K_NODES = 10
K_MAX = 40.0

#: Spacing (Bohr) of the knots of the cubic splines of :math:`s_L(R)`.
TABLE_STEP = 0.02

#: Two centers closer than this (Bohr) are the same center.
SAME_CENTER = 1e-8

_CACHE_LIMIT = 4096

#: Why a crystal cannot use the analytic route (one message, three entries).
PERIODIC_REFUSAL = (
    "two_center='analytic' is molecular: a crystal's overlap and kinetic "
    "matrices are Bloch sums of the two-center tables, which are not "
    "implemented yet")


def validate_two_center(spec) -> str:
    """Normalize a ``two_center`` option to one of :data:`TWO_CENTER_METHODS`.

    ``None`` is the default, ``"grid"``.
    """
    if spec is None:
        return DEFAULT_TWO_CENTER
    key = str(spec).strip().lower()
    if key not in TWO_CENTER_METHODS:
        raise ValueError(f"two_center must be one of "
                         f"{list(TWO_CENTER_METHODS)}, got {spec!r}")
    return key


# --------------------------------------------------------------------------- #
# Angular coupling.
# --------------------------------------------------------------------------- #

_GAUNT: dict = {}


def real_gaunt(l1: int, l2: int, L: int) -> np.ndarray:
    r"""``G[m1, m2, M] = int S_{l1 m1} S_{l2 m2} S_{LM} dOmega`` of the real
    harmonics, shape ``(2 l1 + 1, 2 l2 + 1, 2 L + 1)`` with ``m`` running
    from ``-l`` to ``l``.

    A product quadrature that is exact for these integrands (Gauss-Legendre
    in :math:`\cos\theta`, uniform in :math:`\varphi`); zero unless the
    triangle rule and the parity of :math:`l_1 + l_2 + L` allow it.
    """
    key = (int(l1), int(l2), int(L))
    cached = _GAUNT.get(key)
    if cached is not None:
        return cached
    l1, l2, L = key
    G = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1))
    if abs(l1 - l2) <= L <= l1 + l2 and (l1 + l2 + L) % 2 == 0:
        total = l1 + l2 + L
        mu, w_mu = np.polynomial.legendre.leggauss(total // 2 + 2)
        n_phi = total + 2
        phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
        theta = np.repeat(np.arccos(mu), n_phi)
        phi = np.tile(phi, mu.size)
        weight = np.repeat(w_mu, n_phi) * (2.0 * np.pi / n_phi)

        def table(l):
            return np.stack([spherical_harmonic(l, m, theta, phi, real=True)
                             for m in range(-l, l + 1)])
        G = np.einsum("g,ag,bg,cg->abc", weight, table(l1), table(l2),
                      table(L), optimize=True)
        G[np.abs(G) < 1e-13] = 0.0
    _GAUNT[key] = G
    return G


def multipoles(l1: int, l2: int) -> list[int]:
    """The multipoles :math:`L` two harmonics of degrees ``l1``, ``l2`` couple to."""
    return list(range(abs(int(l1) - int(l2)), int(l1) + int(l2) + 1, 2))


# --------------------------------------------------------------------------- #
# Radial functions and their Fourier-Bessel transforms.
# --------------------------------------------------------------------------- #

def _wave_numbers():
    """Nodes and weights of the wave-number integral."""
    x, w = np.polynomial.legendre.leggauss(K_NODES)
    n_panels = int(np.ceil(K_MAX / K_PANEL))
    lo = K_PANEL * np.arange(n_panels)
    k = (lo[:, None] + 0.5 * K_PANEL * (x[None, :] + 1.0)).ravel()
    weights = np.tile(0.5 * K_PANEL * w, n_panels)
    return k, weights


_K, _KW = _wave_numbers()


_RULE = np.polynomial.legendre.leggauss(R_NODES)


def _panel_nodes(lo, hi):
    """Gauss-Legendre nodes and weights of the radial panels ``[lo, hi]``
    (scalars: one panel; arrays: the nodes of each, concatenated)."""
    x, w = _RULE
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    half = 0.5 * (hi - lo)
    return ((lo[:, None] + half[:, None] * (x[None, :] + 1.0)).ravel(),
            (half[:, None] * w[None, :]).ravel())


# One matrix per (l, mesh): l is at most a few and there are three meshes, so
# the dictionary cannot grow past a few dozen entries, each replaced (not
# added to) when a longer one is asked for.
_BESSEL: dict = {}


def _bessel_rows(l: int, mesh: float, n_rows: int) -> np.ndarray:
    """The first ``n_rows`` rows of the matrix ``j_l(k_j r_i)``.

    ``mesh`` is a panel width (``r_i`` the Gauss-Legendre nodes of the full
    radial panels of that width, ``R_NODES`` rows per panel) or ``0`` (``r_i``
    the knots of the :math:`s_L(R)` tables, ``i * TABLE_STEP``).  Both are
    prefix-consistent, so one matrix per ``(l, mesh)`` serves every cutoff
    and is only recomputed when a longer one is asked for.
    """
    cached = _BESSEL.get((l, mesh))
    if cached is not None and cached.shape[0] >= n_rows:
        return cached[:n_rows]
    if mesh:
        n_panels = -(-n_rows // R_NODES)
        edges = mesh * np.arange(n_panels + 1)
        r = _panel_nodes(edges[:-1], edges[1:])[0]
    else:
        r = TABLE_STEP * np.arange(n_rows)
    matrix = np.empty((r.size, _K.size))
    for start in range(0, r.size, 256):
        block = r[start:start + 256]
        matrix[start:start + 256] = spherical_jn(l, block[:, None] * _K[None, :])
    _BESSEL[(l, mesh)] = matrix
    return matrix


@dataclass(frozen=True)
class RadialFunction:
    """A radial function sampled for its Fourier-Bessel transform.

    ``values`` are ``f(r_i) w_i r_i^2`` on the full Gauss-Legendre panels
    below ``cut``; ``tail`` holds the same for a last partial panel when
    ``cut`` falls inside one (a projector truncated at its augmentation
    radius).  ``digest`` identifies the function by content, so the same
    function reached from another geometry, calculator or process step hits
    the caches.
    """

    l: int
    cut: float
    panel: float
    values: np.ndarray
    tail: tuple | None
    digest: str

    @classmethod
    def sample(cls, radial, l: int, cut: float, *, truncate: bool):
        """Sample ``radial`` (a callable of ``r``) out to ``cut`` Bohr.

        With ``truncate`` the function is cut exactly at ``cut`` (the last
        panel is a partial one); without it the panels cover ``cut`` and the
        function is whatever ``radial`` gives there (zero, to the 1e-14 of
        the support threshold, for a basis function).
        """
        l, cut = int(l), float(cut)
        panel = CUT_PANEL if truncate else R_PANEL
        n_full = int(cut / panel + 1e-9) if truncate else int(
            np.ceil(cut / panel))
        edges = panel * np.arange(n_full + 1)
        r, w = _panel_nodes(edges[:-1], edges[1:])
        values = np.asarray(radial(r), dtype=float) * w * r * r
        tail = None
        if truncate and cut - n_full * panel > 1e-9:
            rt, wt = _panel_nodes(n_full * panel, cut)
            tail = (rt, np.asarray(radial(rt), dtype=float) * wt * rt * rt)
        digest = hashlib.sha1(values.tobytes())
        digest.update(np.float64(cut).tobytes())
        digest.update(np.array([l, panel]).tobytes())
        if tail is not None:
            digest.update(tail[1].tobytes())
        return cls(l, cut, panel, values, tail, digest.hexdigest())

    def transform(self) -> np.ndarray:
        """:math:`F_l(k)` on the wave-number nodes (memoized by content)."""
        cached = _TRANSFORMS.get(self.digest)
        if cached is None:
            n_rows = self.values.size
            n_panels = -(-n_rows // R_NODES)
            edges = self.panel * np.arange(n_panels + 1)
            r = _panel_nodes(edges[:-1], edges[1:])[0][:n_rows]
            F = np.zeros(_K.size)
            for start in range(0, n_rows, 256):
                stop = min(start + 256, n_rows)
                block_r = r[start:stop]
                block_jn = spherical_jn(self.l, block_r[:, None] * _K[None, :])
                F += block_jn.T @ self.values[start:stop]
            if self.tail is not None:
                r_tail, wf = self.tail
                F = F + spherical_jn(self.l, _K[:, None] * r_tail[None, :]) @ wf
            if len(_TRANSFORMS) >= _CACHE_LIMIT:
                _TRANSFORMS.clear()
            cached = _TRANSFORMS[self.digest] = F
        return cached


_TRANSFORMS: dict = {}


@dataclass(frozen=True)
class PairTable:
    """:math:`s_L(R)` of one ordered pair of radial functions, a cubic spline
    in :math:`R` per multipole ``L``, zero beyond ``reach`` (the sum of the
    cutoffs)."""

    l1: int
    l2: int
    reach: float
    L: tuple
    splines: tuple
    slopes: tuple


_TABLES: dict = {}


def pair_table(f1: RadialFunction, f2: RadialFunction, kind: str) -> PairTable:
    """The :math:`s_L(R)` splines of ``f1`` (bra) and ``f2`` (ket)."""
    key = (f1.digest, f2.digest, kind)
    table = _TABLES.get(key)
    if table is not None:
        return table
    n = int(np.ceil((f1.cut + f2.cut) / TABLE_STEP)) + 1
    weight = _KW * _K ** 2 * f1.transform() * f2.transform()
    if kind == "kinetic":
        weight = 0.5 * weight * _K ** 2
    R = TABLE_STEP * np.arange(n)
    multipole, splines, slopes = [], [], []
    for L in multipoles(f1.l, f2.l):
        sign = (-1) ** ((f1.l - f2.l - L) // 2)
        s = np.zeros(n)
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            block_r = R[start:stop]
            block_jn = spherical_jn(L, block_r[:, None] * _K[None, :])
            s[start:stop] = 8.0 * sign * (block_jn @ weight)
        spline = CubicSpline(R, s)
        multipole.append(L)
        splines.append(spline)
        slopes.append(spline.derivative())
    table = PairTable(f1.l, f2.l, float(R[-1]), tuple(multipole),
                      tuple(splines), tuple(slopes))
    if len(_TABLES) >= _CACHE_LIMIT:
        _TABLES.clear()
    _TABLES[key] = table
    return table


# --------------------------------------------------------------------------- #
# Sets of functions.
# --------------------------------------------------------------------------- #

def _cut_of(function) -> tuple[float, bool]:
    """``(cut, truncate)`` of a function: a projector is cut at its
    augmentation radius, a basis function at its support."""
    r_cut = getattr(function, "r_cut", None)
    if r_cut is not None and np.isfinite(r_cut) and hasattr(function, "kb_energy"):
        return float(r_cut), True
    support = getattr(function, "support_radius", None)
    if support is None:
        raise ValueError(
            "two_center='analytic' needs confined radial functions: "
            f"{type(function).__name__} has no finite support")
    return float(support), False


class FunctionSet:
    """The shells of a list of ``radial(r) S_lm`` functions.

    A shell is the ``2 l + 1`` functions sharing a center, a radial function
    and ``l``.  ``centers`` are the distinct centers, ``radials`` the distinct
    radial functions, and ``groups[r] = (center, index)`` the shells of radial
    function ``r``: the center of each (an index into ``centers``) and the
    positions of its members, ``m = -l .. l``, in the original list.
    """

    def __init__(self, functions):
        for function in functions:
            if not getattr(function, "real", False):
                raise ValueError("two_center='analytic' is for real "
                                 "spherical harmonics (a molecule)")
            if not hasattr(function, "radial"):
                raise ValueError(
                    "two_center='analytic' needs radial(r) * Y_lm functions;"
                    f" {type(function).__name__} has no radial part")
        self.n = len(functions)
        centers: list = []
        self.radials: list[RadialFunction] = []
        radial_index: dict = {}
        members: dict = {}
        for position, function in enumerate(functions):
            center = np.asarray(function.center, dtype=float)
            for c, known in enumerate(centers):
                if np.linalg.norm(known - center) < SAME_CENTER:
                    break
            else:
                c = len(centers)
                centers.append(center)
            cut, truncate = _cut_of(function)
            f = RadialFunction.sample(function.radial, function.l, cut,
                                      truncate=truncate)
            r = radial_index.setdefault(f.digest, len(radial_index))
            if r == len(self.radials):
                self.radials.append(f)
            members.setdefault((c, r), {})[int(function.m)] = position
        self.centers = np.asarray(centers).reshape(-1, 3)
        by_radial: dict = {}
        for (c, r), by_m in members.items():
            l = self.radials[r].l
            if sorted(by_m) != list(range(-l, l + 1)):
                raise ValueError("two_center='analytic' needs complete shells "
                                 "(every m of an l)")
            by_radial.setdefault(r, []).append(
                (c, [by_m[m] for m in range(-l, l + 1)]))
        self.groups = {r: (np.array([c for c, _ in shells]),
                           np.array([index for _, index in shells]))
                       for r, shells in by_radial.items()}


class TwoCenter:
    """Two-center integrals between a bra set and a ket set of functions.

    ``matrix(kind)`` is :math:`\\langle\\phi_\\mu|\\hat O|\\chi_\\nu\\rangle`
    (``(n_bra, n_ket)``) and ``gradient(kind)`` its derivative with respect
    to the displacement :math:`\\mathbf R_\\chi - \\mathbf R_\\phi` of the two
    centers (``(3, n_bra, n_ket)``): minus the derivative with respect to the
    bra's own center, plus the one with respect to the ket's.  Both are
    computed together and kept.
    """

    def __init__(self, bra, ket=None):
        self.bra = FunctionSet(bra)
        self.ket = self.bra if ket is None else FunctionSet(ket)
        self._results: dict = {}
        self._geometry = None

    def matrix(self, kind: str = "overlap") -> np.ndarray:
        return self._evaluate(kind)[0]

    def gradient(self, kind: str = "overlap") -> np.ndarray:
        return self._evaluate(kind)[1]

    # -- geometry ------------------------------------------------------------ #

    def _pair_geometry(self):
        """The neighbor list and what the tables need of each pair: for every
        bra-center / ket-center pair within the longest reach of any table, its
        distance, direction and harmonics up to the highest multipole.

        ``pair[a, b]`` is the pair's position in those arrays, ``-1`` for a
        pair out of reach.
        """
        if self._geometry is not None:
            return self._geometry
        bra, ket = self.bra.centers, self.ket.centers
        reach = max(f.cut for f in self.bra.radials) + max(
            f.cut for f in self.ket.radials)
        distance = np.linalg.norm(ket[None, :, :] - bra[:, None, :], axis=2)
        a, b = np.nonzero(distance < reach + 2.0 * TABLE_STEP)
        pair = np.full(distance.shape, -1)
        pair[a, b] = np.arange(a.size)
        vector = ket[b] - bra[a]
        d = distance[a, b]
        same = d < SAME_CENTER
        unit = np.where(same[:, None], 0.0,
                        vector / np.where(same, 1.0, d)[:, None])
        # A coincident pair is evaluated at an arbitrary direction (z): only
        # the monopole survives in the value.  Its gradient is not zero for
        # the dipole: s_1(R) S_1M(R^) is linear in the displacement,
        # s_1'(0) sqrt(3/4pi) along the axis (y, z, x) of M = -1, 0, 1
        # (`origin`).
        probe = np.where(same[:, None], np.array([0.0, 0.0, 1.0]), vector)
        l_max = (max(f.l for f in self.bra.radials)
                 + max(f.l for f in self.ket.radials))
        Y, dY = {}, {}
        for L in range(l_max + 1):
            values, gradients = [], []
            for M in range(-L, L + 1):
                v, g = spherical_harmonic_gradient(
                    L, M, probe[:, 0], probe[:, 1], probe[:, 2], real=True)
                values.append(v)
                gradients.append(np.stack(g, axis=-1))
            Y[L] = np.stack(values, axis=-1)                  # (P, 2L+1)
            dY[L] = np.stack(gradients, axis=-2)              # (P, 2L+1, 3)
            if L:
                Y[L] = np.where(same[:, None], 0.0, Y[L])
            dY[L] = np.where(same[:, None, None], 0.0, dY[L])
        origin = np.zeros((a.size, 3, 3))
        for m_index, axis in enumerate((1, 2, 0)):
            origin[:, m_index, axis] = np.where(
                same, np.sqrt(3.0 / (4.0 * np.pi)), 0.0)
        self._geometry = (pair, d, unit, Y, dY, origin)
        return self._geometry

    # -- evaluation ---------------------------------------------------------- #

    def _evaluate(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"unknown two-center operator {kind!r}; use one "
                             f"of {list(KINDS)}")
        cached = self._results.get(kind)
        if cached is not None:
            return cached
        pair, distance, unit, Y, dY, origin = self._pair_geometry()
        matrix = np.zeros((self.bra.n, self.ket.n))
        gradient = np.zeros((3, self.bra.n, self.ket.n))
        for a, (bra_center, bra_index) in self.bra.groups.items():
            for b, (ket_center, ket_index) in self.ket.groups.items():
                table = pair_table(self.bra.radials[a], self.ket.radials[b],
                                   kind)
                ia, ib = np.nonzero(pair[bra_center[:, None],
                                         ket_center[None, :]] >= 0)
                pid = pair[bra_center[ia], ket_center[ib]]
                near = distance[pid] < table.reach
                if not np.any(near):
                    continue
                ia, ib, pid = ia[near], ib[near], pid[near]
                d, n_hat = distance[pid], unit[pid]
                block = 0.0
                slope = 0.0
                for L, spline, derivative in zip(table.L, table.splines,
                                                 table.slopes):
                    G = real_gaunt(table.l1, table.l2, L)
                    s, ds = spline(d), derivative(d)
                    angular = np.einsum("mnM,pM->pmn", G, Y[L][pid])
                    block = block + s[:, None, None] * angular
                    slope = slope + (ds[:, None, None, None]
                                     * angular[..., None] * n_hat[:, None, None, :]
                                     + s[:, None, None, None]
                                     * np.einsum("mnM,pMk->pmnk", G, dY[L][pid]))
                    if L == 1:
                        slope = slope + (ds[:, None, None, None]
                                         * np.einsum("mnM,pMk->pmnk", G,
                                                     origin[pid]))
                rows, cols = bra_index[ia][:, :, None], ket_index[ib][:, None, :]
                matrix[rows, cols] = block
                for k in range(3):
                    gradient[k][rows, cols] = slope[..., k]
        self._results[kind] = (matrix, gradient)
        return self._results[kind]
