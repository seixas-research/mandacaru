# -*- coding: utf-8 -*-
# file: pseudopotentials/periodic_paw.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""PAW-LCAO in a crystal: Bloch sums, k-dependent matrices, lattice electrostatics.

The molecular PAW-LCAO Hamiltonian (:class:`~.paw.PAWIntegrals`) is carried to
a periodic cell with a Brillouin-zone sampling.  Every atom-centered function
becomes a **Bloch sum**

.. math::

    \chi_{\mu\mathbf k}(\mathbf r) = \sum_{\mathbf R} e^{i\mathbf k\cdot\mathbf R}
      \phi_\mu(\mathbf r - \mathbf R),

and every matrix element becomes an integral over **one cell**, so no
two-center lattice sum :math:`H_{\mu\nu}(\mathbf R)` is ever formed:

* the overlap and every grid potential (Hartree, the long-range ion
  potential, exchange-correlation) are cell integrals of Bloch sums sampled on
  the cell grid, and the kinetic energy is spectral, :math:`\tfrac12|\mathbf
  G + \mathbf k|^2` acting on the periodic part of each sum;
* the short-range terms -- the projections :math:`C_{\mu p}(\mathbf k) =
  \langle\chi_{\mu\mathbf k}|\tilde p_p\rangle` and the short-range local
  potential -- are quadratures over spheres around the atoms of the home cell,
  evaluated on the Bloch sums, because a cell integral of
  :math:`\chi^*_{\mu}\,v^{\rm per}\,\chi_\nu` unfolds to an all-space integral
  of the home-cell term.

Electrostatics
--------------
The charges are the smooth density :math:`\tilde n`, the compensation
multipoles :math:`\hat n = \sum_{A,LM} q^A_{LM}\,g_L Y_{LM}` and the ions,
each represented by the Gaussian whose potential is the long-range half of its
local channel (:mod:`.local_split`).  Together they are neutral, so their
Coulomb energy is a sum over :math:`\mathbf G \neq 0` with nothing to
regularize -- no Madelung term and no neutralizing background: each block may
drop its own :math:`\mathbf G = 0` term because the finite parts those terms
leave behind sum to zero for a neutral total.  The point-ion energy is then
the Gaussian-ion energy minus the Gaussian self-energies plus the short-range
:math:`Z_AZ_B\,\mathrm{erfc}(d/2\sigma)/d` pair terms (Ewald's split, with
the split width of the local potential).

**The Fourier filter.**  On the grid every charge is represented by the
:math:`\mathbf G` the grid carries and nothing above them
(:mod:`~mandacaru.integrals.reciprocal`).  For the Gaussian ions and the smooth
density that loses nothing.  The compensation shapes are compact and do carry
weight above the grid's cutoff; their interaction with the grid charges is
filtered, and their interaction with each other -- where the missing weight
would show -- is evaluated on a dense reciprocal lattice up to
:data:`COMPENSATION_CUTOFF`, plus the analytic tail beyond it.

The conventions that make the energy the molecular one in a large cell
---------------------------------------------------------------------
* The compensation charge of atom ``A`` is attracted to the local potentials
  of every *other* atom and image, but not to its own: that on-site term is
  inside the dataset, which is calibrated on the isolated atom
  (:meth:`~.paw.PAWIntegrals.compensation_ionic_at`).  The long-range part of
  it comes with the reciprocal sum and is subtracted again in closed form.
* The nonlocal coupling is the dataset's frozen :math:`D^{ion}`, and the
  frozen one-center energies are carried as constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from ..integrals import reciprocal as rc
from ..integrals._backend import single_threaded_blas

#: Reciprocal-space cutoff (Bohr^-1) of the sums among the compact charges
#: (compensation multipoles and Gaussian ions).  The shapes vanish at r_g with
#: two continuous derivatives, so their transforms fall as q^-5 and the
#: self-energy beyond the cutoff as q_c^-9; that tail is added analytically
#: (`_tail`).
COMPENSATION_CUTOFF = 15.0

#: Reciprocal vectors per block of the dense sums (bounds their memory).
DENSE_BLOCK = 200000

#: Radial points of the compensation shapes' transforms.
SHAPE_POINTS = 1201
#: The transforms are tabulated once on ``[0, SHAPE_TABLE_MAX]`` with this
#: step (Bohr^-1) and interpolated; the tail integral runs to the table's end,
#: where the self-energy density has fallen by ~1e-12.
SHAPE_TABLE_STEP = 0.01
SHAPE_TABLE_MAX = 120.0

#: Memory (bytes) one block of k-points' Bloch sums may occupy.
KPOINT_BLOCK_BYTES = 1 << 30

#: Lattice-sum reach (in Gaussian widths) of the short-range ion pair term.
ION_PAIR_REACH = 12.0


# --------------------------------------------------------------------------- #
# Bloch sums.
# --------------------------------------------------------------------------- #

#: A tabulated radial function counts as zero beyond the last point where it
#: exceeds this fraction of its peak.
SUPPORT_TOLERANCE = 1e-12

_SUPPORTS: dict = {}


class BasisGradient:
    r"""The Cartesian derivative :math:`\partial_a(R\,Y_{lm})` of a
    tabulated basis function, as a basis function of its own.

    A crystal force needs :math:`\langle\partial\chi_\mu|O|\chi_\nu\rangle`
    for every matrix :math:`O` the Hamiltonian is built of; with these in the
    basis, the same builder produces them (the C Bloch kernel evaluates them
    analytically, :meth:`evaluate` is its NumPy reference).  ``center`` is
    the parent's, and moves with it.
    """

    def __init__(self, parent, axis: int):
        self.parent = parent
        self.axis = int(axis)
        self.l, self.m = int(parent.l), int(parent.m)
        self.center = np.asarray(parent.center, dtype=float).copy()

    def evaluate(self, x, y, z) -> np.ndarray:
        """Sample the derivative at Cartesian points (Bohr)."""
        from ..basis._angular import spherical_harmonic_gradient
        dx, dy, dz = (np.asarray(c, dtype=float) - o
                      for c, o in zip((x, y, z), self.center))
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        out = np.zeros(r.shape, dtype=complex)
        away = r >= 1e-12
        if np.any(away):
            Y, gradient = spherical_harmonic_gradient(
                self.l, self.m, dx[away], dy[away], dz[away])
            ra = r[away]
            offset = (dx, dy, dz)[self.axis][away]
            out[away] = (radial_slope(self.parent, ra) * offset / ra * Y
                         + self.parent.radial(ra) * gradient[self.axis])
        if np.any(~away) and self.l == 1:
            # At the nucleus only a p function has a gradient: R ~ R'(0) r.
            c0 = 1.0 / np.sqrt(4.0 * np.pi)
            if self.m == 0:
                value = np.sqrt(3.0) * c0 if self.axis == 2 else 0.0
            else:
                value = -np.sqrt(1.5) * c0 * (1.0, 1j, 0.0)[self.axis]
                if self.m < 0:
                    value = -np.conj(value)
            out[~away] = radial_slope(self.parent, np.zeros(1))[0] * value
        return out


def radial_slope(function, r) -> np.ndarray:
    """``dR/dr`` of a tabulated function's radial spline, with its
    continuation ``R(r_0)(r / r_0)^l`` below the first breakpoint and zero
    past its cutoff -- the derivative of ``function.radial``."""
    breaks, coeffs, rc_ = _radial_spline(function)
    r = np.asarray(r, dtype=float)
    x0, x_end = float(breaks[0]), float(breaks[-1])
    index = np.clip(np.searchsorted(breaks, r, side="right") - 1, 0,
                    len(breaks) - 2)
    t = r - breaks[index]
    c = coeffs[index]
    slope = (3.0 * c[:, 0] * t + 2.0 * c[:, 1]) * t + c[:, 2]
    l = int(function.l)
    if x0 > 0.0:
        edge = float(function.radial(np.array([x0]))[0])
        inner = (l * edge * (np.maximum(r, 0.0) / x0) ** (l - 1) / x0
                 if l > 0 else np.zeros_like(r))
        slope = np.where(r < x0, inner, slope)
    return np.where((r > x_end) | (r >= rc_), 0.0, slope)


def _radial_table(function):
    """``(r, values)`` of a tabulated radial function, or ``None``.

    Both layouts in use: a multiple-zeta :class:`~mandacaru.basis.multizeta.
    TabulatedOrbital` keeps a ``table``; a pseudo-atomic orbital keeps the
    arrays themselves (``_r``, ``_values``); a :class:`BasisGradient` has its
    parent's.
    """
    if isinstance(function, BasisGradient):
        return _radial_table(function.parent)
    table = getattr(function, "table", None)
    if table is not None:
        return table.r, table.values
    if getattr(function, "_values", None) is not None:
        return function._r, function._values
    return None


def _support(function) -> float:
    """Radius (Bohr) beyond which an atom-centered function vanishes.

    For a tabulated orbital, where its values do -- not the end of its table:
    a confined, filtered orbital is exactly zero well inside the 30-Bohr table
    it is stored on, and the table's end would make every Bloch sum visit
    hundreds of lattice images that contribute nothing.
    """
    table = _radial_table(function)
    if table is not None:
        r, values = table
        key = id(values)
        if key not in _SUPPORTS:
            magnitude = np.abs(np.asarray(values, dtype=float))
            significant = np.nonzero(magnitude > SUPPORT_TOLERANCE
                                     * float(magnitude.max()))[0]
            r = np.asarray(r, dtype=float)
            edge = (r[min(significant[-1] + 1, r.size - 1)]
                    if significant.size else r[0])
            _SUPPORTS[key] = (values, float(edge))
        return _SUPPORTS[key][1]
    for name in ("r_c", "r_cut"):
        value = getattr(function, name, None)
        if value is not None and np.isfinite(value):
            return float(value)
    raise ValueError(f"{type(function).__name__} has no finite support radius; "
                     "a Bloch sum needs one")


def _shell_key(function):
    """Functions that share a radial table and a center differ only in ``m``
    (and gradients also by their axis)."""
    if isinstance(function, BasisGradient):
        key = _shell_key(function.parent)
        return None if key is None else (key[0], tuple(
            np.round(np.asarray(function.center, float), 12)),
            "d", function.axis)
    table = _radial_table(function)
    if table is None or not hasattr(function, "radial"):
        return None
    return (id(table[1]),
            tuple(np.round(np.asarray(function.center, float), 12)))


def _radial_spline(function):
    """``(breaks, coeffs, rc)`` of a tabulated function's radial spline, as
    the C Bloch kernel evaluates it, or ``None`` for any other radial form.

    Both layouts continue below the first breakpoint as
    ``R(r_0) (r / r_0)^l`` and vanish past the last one; a multiple-zeta
    orbital is also cut at its confinement radius (``r >= r_c``).
    """
    if isinstance(function, BasisGradient):
        return _radial_spline(function.parent)
    spline = getattr(function, "_spline", None)
    if spline is None or _radial_table(function) is None \
            or not hasattr(spline, "c") or spline.c.shape[0] != 4:
        return None
    rc = float(getattr(function, "r_c", np.inf))
    return (np.asarray(spline.x, dtype=float),
            np.ascontiguousarray(np.asarray(spline.c, dtype=float).T),
            rc if np.isfinite(rc) else np.inf)


#: ``(r_g, L) -> (q, F_L(q))`` of a compensation shape (see
#: :meth:`PeriodicPAW._shape_table`).
_SHAPE_TABLES: dict = {}

#: Quadrature points per call when a sphere's Bloch sums are streamed: the
#: values at every point and k-point at once ran to gigabytes (a sphere's
#: tens of thousands of points times the 50 k-points of a band path).
SPHERE_CHUNK = 16384


def _sphere_chunks(points, *arrays):
    """Yield ``(points, *arrays)`` chunks of a quadrature, its points sorted
    into 1-Bohr bins (a sum over the points does not depend on their order,
    and :func:`bloch_values` then has nothing to un-permute)."""
    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
    order = np.lexsort((np.floor(z), np.floor(y), np.floor(x)))
    x, y, z = x[order], y[order], z[order]
    arrays = [np.asarray(a).ravel()[order] for a in arrays]
    for start in range(0, x.size, SPHERE_CHUNK):
        part = slice(start, start + SPHERE_CHUNK)
        yield ((x[part], y[part], z[part]), *[a[part] for a in arrays])


def reaching(functions, lattice, region_center, region_radius) -> np.ndarray:
    """Indices of the ``functions`` with a lattice image whose support
    reaches the ball of ``region_radius`` about ``region_center``.

    The others are exactly zero on every point of the ball, so a sphere
    quadrature needs only these rows: in a large cell a short-range or a
    projector sphere is reached by the functions of a few neighbors, not
    by all ``M``.
    """
    region_center = np.asarray(region_center, dtype=float)
    keep = np.zeros(len(functions), dtype=bool)
    verdicts: dict = {}
    for mu, function in enumerate(functions):
        center = np.asarray(function.center, dtype=float)
        support = _support(function)
        key = (tuple(np.round(center, 12)), round(float(support), 12))
        if key not in verdicts:
            reach = support + float(region_radius)
            offset = float(np.linalg.norm(center - region_center))
            verdicts[key] = any(
                np.linalg.norm(center + R - region_center) <= reach
                for R in rc.lattice_translations(lattice, reach + offset))
        keep[mu] = verdicts[key]
    return np.flatnonzero(keep)


def bloch_values(functions, lattice, kpoints, points, region_center,
                 region_radius) -> np.ndarray:
    r"""``(nk, M, npts)`` Bloch sums of ``functions`` at ``points``.

    ``points`` is an ``(x, y, z)`` triple of flat arrays (Bohr) lying within
    ``region_radius`` of ``region_center``; only the images that reach that
    region are evaluated, each once for every k-point.  Functions sharing a
    radial table and a center -- the ``m`` components of one shell -- share
    the geometry and the radial interpolation of every image.
    """
    from scipy.sparse import csr_matrix

    from ..basis._angular import spherical_harmonic
    from ..integrals import _backend as backend

    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
    # The points are visited in 1-Bohr bins: the C kernel culls lattice
    # images per block of consecutive points, which a quadrature sphere's
    # radial-major order would spread over the whole sphere.
    order = np.lexsort((np.floor(z), np.floor(y), np.floor(x)))
    x, y, z = x[order], y[order], z[order]
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    out = np.zeros((len(kpoints), len(functions), x.size), dtype=complex)
    region_center = np.asarray(region_center, dtype=float)
    groups: dict = {}
    for mu, function in enumerate(functions):
        key = _shell_key(function)
        groups.setdefault(key if key is not None else ("single", mu),
                          []).append(mu)
    for members in groups.values():
        first = functions[members[0]]
        center = np.asarray(first.center, dtype=float)
        support = _support(first)
        reach = support + float(region_radius)
        offset = float(np.linalg.norm(center - region_center))
        gradient = isinstance(first, BasisGradient)
        shared = _shell_key(first) is not None and not gradient
        spline = _radial_spline(first) if (shared or gradient) else None
        if spline is not None and backend.HAS_C_BACKEND:
            # The C kernel tests every candidate image against every point
            # itself, and evaluates and phases what lies inside its support.
            candidates = [R for R in rc.lattice_translations(
                lattice, reach + offset)
                if np.linalg.norm(center + R - region_center) <= reach]
            if not candidates:
                continue
            translations = np.array(candidates, dtype=float)
            phases = np.exp(1j * (kpoints @ translations.T))  # (nk, n_R)
            if backend.bloch_shell(
                    (x, y, z), center, translations, phases, support,
                    spline, first.l, [functions[mu].m for mu in members],
                    members, out,
                    derivative=first.axis if gradient else -1):
                continue
        # Every image's values are independent of k: collect them as one
        # sparse (image x point) matrix per function and apply all the
        # k-points' phases in a single product, instead of scattering
        # nk x points per image.
        translations, rows, cols = [], [], []
        values = {mu: [] for mu in members}
        for R in rc.lattice_translations(lattice, reach + offset):
            origin = center + R
            if np.linalg.norm(origin - region_center) > reach:
                continue
            dx, dy, dz = x - origin[0], y - origin[1], z - origin[2]
            near = ((np.abs(dx) < support) & (np.abs(dy) < support)
                    & (np.abs(dz) < support))
            if not np.any(near):
                continue
            idx = np.nonzero(near)[0]
            r = np.sqrt(dx[idx] ** 2 + dy[idx] ** 2 + dz[idx] ** 2)
            inside = r < support
            if not np.any(inside):
                continue
            idx, r = idx[inside], r[inside]
            ddx, ddy, ddz = dx[idx], dy[idx], dz[idx]
            rows.append(np.full(idx.size, len(translations)))
            cols.append(idx)
            translations.append(R)
            if shared:
                theta = np.arccos(np.clip(ddz / np.maximum(r, 1e-300),
                                          -1.0, 1.0))
                phi = np.arctan2(ddy, ddx)
                radial = first.radial(r)
                for mu in members:
                    values[mu].append(radial * spherical_harmonic(
                        functions[mu].l, functions[mu].m, theta, phi))
            else:
                for mu in members:
                    values[mu].append(functions[mu].evaluate(
                        x[idx] - R[0], y[idx] - R[1], z[idx] - R[2]))
        if not translations:
            continue
        phases = np.exp(1j * (kpoints @ np.array(translations).T))  # (nk, n_R)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        for mu in members:
            images = csr_matrix((np.concatenate(values[mu]), (rows, cols)),
                                shape=(len(translations), x.size))
            out[:, mu, :] += (images.T @ phases.T).T
    if np.array_equal(order, np.arange(order.size)):
        return out
    restored = np.empty_like(out)
    restored[:, :, order] = out
    return restored


def time_reversal_reduce(kpoints_frac, tolerance: float = 1e-8):
    r"""``(representatives, weights)`` of a mesh under :math:`\mathbf k \to -\mathbf k`.

    The density of :math:`\psi_{-\mathbf k} = \psi^*_{\mathbf k}` equals that of
    :math:`\psi_{\mathbf k}`, and so does its compensation charge; every
    real-space quantity of the pair is twice the representative's.  Weights
    sum to one.
    """
    kpoints_frac = np.asarray(kpoints_frac, dtype=float)
    n = len(kpoints_frac)
    used = np.zeros(n, dtype=bool)
    keep, weights = [], []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        count = 1
        partner = -kpoints_frac[i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            delta = kpoints_frac[j] - partner
            if np.all(np.abs(delta - np.round(delta)) < tolerance):
                used[j] = True
                count += 1
                break
        keep.append(kpoints_frac[i])
        weights.append(count)
    weights = np.asarray(weights, dtype=float)
    return np.asarray(keep), weights / weights.sum()


# --------------------------------------------------------------------------- #
# Symmetry: the operations the grid carries.
# --------------------------------------------------------------------------- #

#: Directions used to fit the action of a rotation on the Y_LM of one L.
ROTATION_SAMPLES = 64
#: Largest residual / departure from unitarity accepted for those fits.
ROTATION_TOLERANCE = 1e-10


@dataclass
class GridSymmetry:
    r"""Space-group operations that map the grid onto itself.

    For every kept operation :math:`x \to Wx + t` (fractional coordinates):
    ``permutations[s]`` sends grid node ``n`` to the node at :math:`S x_n`,
    ``atom_maps[s][A]`` is the atom at :math:`S x_A`, and ``rotations[s][L]``
    the matrix :math:`T^L` with :math:`Y_{LM}(R\hat u) = \sum_{M'}
    Y_{LM'}(\hat u)\,T^L_{M'M}`, :math:`R` the Cartesian rotation.

    A symmetric quantity is the average of its images, so a density built
    from the irreducible wedge of the k-mesh becomes the full mesh's:
    :meth:`field` for the smooth density and the kinetic-energy density,
    :meth:`moments` for the compensation multipoles.
    """

    info: object                         # the SymmetryInfo actually used
    n_space_group: int                   # operations before the grid filter
    permutations: list
    atom_maps: list
    rotations: list

    @property
    def n_operations(self) -> int:
        return len(self.permutations)

    def keeping_mesh(self, mesh) -> "GridSymmetry | None":
        """The operations that also map the k-point ``mesh`` onto itself.

        A 2x1x1 mesh of a cubic crystal is not invariant under the cube's
        rotations, and neither is the density summed over it: reducing or
        symmetrizing with an operation the mesh does not share is wrong
        (1.4 mHa per cell on silicon's 2x1x1 mesh).  ``None`` when nothing
        beyond the identity survives.
        """
        from dataclasses import replace as dc_replace

        mesh = np.asarray(mesh, dtype=float)

        def on_mesh(points):
            delta = points[:, None, :] - mesh[None, :, :]
            delta -= np.round(delta)
            return np.all(np.any(np.all(np.abs(delta) < 1e-8, axis=2),
                                 axis=1))

        keep = [s for s, W in enumerate(self.info.rotations)
                if on_mesh(mesh @ np.asarray(W, dtype=float))]
        if len(keep) <= 1:
            return None
        info = dc_replace(self.info,
                          rotations=np.asarray(self.info.rotations)[keep],
                          translations=np.asarray(self.info.translations)[keep])
        return GridSymmetry(
            info=info, n_space_group=self.n_space_group,
            permutations=[self.permutations[s] for s in keep],
            atom_maps=[self.atom_maps[s] for s in keep],
            rotations=[self.rotations[s] for s in keep])

    def field(self, values) -> np.ndarray:
        """The average of a flat grid field over the operations."""
        values = np.asarray(values)
        return sum(values[perm] for perm in self.permutations) \
            / self.n_operations

    def moments(self, q: dict) -> dict:
        """The average of the multipoles ``{(atom, L, M): q}``."""
        out = {key: 0.0j for key in q}
        for atom_map, rotation in zip(self.atom_maps, self.rotations):
            for (atom, L, M) in q:
                T = rotation[L]
                source = atom_map[atom]
                total = 0.0j
                for Mp in range(-L, L + 1):
                    value = q.get((source, L, Mp))
                    if value is not None:
                        total += T[M + L, Mp + L] * value
                out[(atom, L, M)] += total
        return {key: value / self.n_operations for key, value in out.items()}


def _harmonic_rotation(R, L: int) -> np.ndarray:
    """``T`` with ``Y_L(R u) = Y_L(u) T`` for the complex ``Y_LM``."""
    from ..basis._angular import spherical_harmonic
    rng = np.random.default_rng(L + 7)
    u = rng.normal(size=(ROTATION_SAMPLES, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]

    def harmonics(v):
        theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
        phi = np.arctan2(v[:, 1], v[:, 0])
        return np.stack([spherical_harmonic(L, M, theta, phi)
                         for M in range(-L, L + 1)], axis=1)

    Y = harmonics(u)
    YR = harmonics(u @ np.asarray(R).T)
    T, *_ = np.linalg.lstsq(Y, YR, rcond=None)
    residual = float(np.max(np.abs(Y @ T - YR)))
    unitary = float(np.max(np.abs(T.conj().T @ T - np.eye(2 * L + 1))))
    if residual > ROTATION_TOLERANCE or unitary > ROTATION_TOLERANCE:
        raise RuntimeError(f"rotation of the L = {L} harmonics did not fit "
                           f"(residual {residual:.1e}, unitarity {unitary:.1e})")
    return T


def grid_symmetry(atoms, grid, max_L: int = 4, symprec: float | None = None):
    r"""The :class:`GridSymmetry` of a periodic ``atoms`` on ``grid``, or
    ``None`` when no operation beyond the identity survives.

    An operation is kept when it maps grid nodes onto grid nodes --
    :math:`W_{ij}N_i/N_j` integral wherever :math:`W_{ij} \neq 0` and
    :math:`N_it_i` integral -- and every atom onto an atom of the same
    species and initial magnetic moment.  Those operations form a subgroup
    (a composition of two that land on nodes lands on nodes), and the k-mesh
    must be reduced with that subgroup and no larger one: the grid decides
    it, through ``h``.
    """
    from dataclasses import replace as dc_replace

    from ..core.symmetry import DEFAULT_SYMPREC, crystal_symmetry

    info = crystal_symmetry(atoms, DEFAULT_SYMPREC if symprec is None
                            else symprec)
    N = np.asarray(grid.shape, dtype=int)
    lattice = rc.lattice_vectors(grid)
    inverse = np.linalg.inv(lattice)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    # An atom maps onto one of the same element *and* initial moment: an
    # operation swapping two opposite moments (an antiferromagnet) would
    # otherwise average the magnetization away.
    keys = sorted({(int(z), round(float(mag), 6)) for z, mag in
                   zip(atoms.get_atomic_numbers(),
                       atoms.get_initial_magnetic_moments())})
    numbers = np.array([keys.index((int(z), round(float(mag), 6)))
                        for z, mag in zip(atoms.get_atomic_numbers(),
                                          atoms.get_initial_magnetic_moments())])
    n1, n2, n3 = np.meshgrid(*[np.arange(n) for n in N], indexing="ij")
    nodes = np.stack([n1.ravel(), n2.ravel(), n3.ravel()])
    keep, permutations, atom_maps, rotations = [], [], [], []
    for s, (W, t) in enumerate(zip(info.rotations, info.translations)):
        W = np.asarray(W, dtype=float)
        scale = W * N[:, None] / N[None, :]
        shift = t * N
        if (np.any(np.abs(scale - np.round(scale)) > 1e-9)
                or np.any(np.abs(shift - np.round(shift)) > 1e-6)):
            continue
        image = (np.round(scale).astype(int) @ nodes
                 + np.round(shift).astype(int)[:, None]) % N[:, None]
        perm = np.ravel_multi_index(image, tuple(N))
        mapped = (frac @ W.T + t) % 1.0
        atom_map = np.full(len(frac), -1)
        for a, x in enumerate(mapped):
            delta = frac - x
            delta -= np.round(delta)
            match = np.nonzero((np.linalg.norm(delta @ lattice.T, axis=1)
                                < 1e-4) & (numbers == numbers[a]))[0]
            if match.size != 1:
                break
            atom_map[a] = match[0]
        if np.any(atom_map < 0):
            continue
        R = lattice @ W @ inverse
        keep.append(s)
        permutations.append(perm)
        atom_maps.append(atom_map)
        rotations.append({L: _harmonic_rotation(R, L)
                          for L in range(max_L + 1)})
    if len(keep) <= 1:
        return None
    used = dc_replace(info, rotations=np.asarray(info.rotations)[keep],
                      translations=np.asarray(info.translations)[keep])
    return GridSymmetry(info=used, n_space_group=info.n_operations,
                        permutations=permutations, atom_maps=atom_maps,
                        rotations=rotations)


# --------------------------------------------------------------------------- #
# Sphere quadratures.
# --------------------------------------------------------------------------- #

def _sphere_rule(center, radius: float, radial: int, polar: int,
                 azimuthal: int, panel: float | None = None):
    """Points ``(x, y, z)`` and weights of a product rule on a ball."""
    from .local_split import _angular_rule, _radial_rule
    r, w_r = _radial_rule(radius, panel if panel is not None else 0.0, radial)
    directions, w_ang = _angular_rule(polar, azimuthal)
    center = np.asarray(center, dtype=float)
    points = tuple((center[i] + r[:, None] * directions[i][None, :]).ravel()
                   for i in range(3))
    weights = (w_r[:, None] * w_ang[None, :]).ravel()
    return points, weights, r, directions


# --------------------------------------------------------------------------- #
# The crystal.
# --------------------------------------------------------------------------- #

@dataclass
class KPointMatrices:
    """Everything of one k-point that does not depend on the density."""

    k: np.ndarray                       # Cartesian, Bohr^-1
    weight: float
    psi: np.ndarray                     # (M, ngrid) Bloch sums on the grid
    overlap: np.ndarray                 # augmented S(k)
    fixed: np.ndarray                   # T + V_sr + C D^ion C^dagger
    projections: np.ndarray             # C(k), (M, P)


class PeriodicPAW:
    r"""A PAW-LCAO crystal: basis, projectors and datasets of one cell.

    Parameters
    ----------
    basis, atom_of_orbital
        The cell's basis functions (centers in Bohr) and their atoms.
    projectors : list
        :class:`~.orbitals.KBProjector` of the cell.
    datasets : list
        One :class:`~.paw.PAWDataset` per atom.
    centers : list
        Atomic positions in Bohr.
    grid : Grid
        The cell grid (``periodic=True``).
    kpoints : (nk, 3)
        Cartesian k-points (Bohr^-1), already reduced.
    weights : (nk,)
        Their weights, summing to one.
    coupling, overlap_blocks, multipole_blocks
        The dataset blocks of the molecular builder
        (:func:`~.paw.paw_coupling_blocks`, :func:`~.paw.paw_overlap_blocks`,
        :func:`~.paw.paw_multipole_blocks`).
    filter_cutoff : float, optional
        The basis' Fourier-filter cutoff (Bohr^-1), recorded so a strained
        cell rebuilds the same basis.  The partial core is *not* filtered
        with it (:meth:`core_density`).
    """

    def __init__(self, basis, atom_of_orbital, projectors, datasets, centers,
                 grid, kpoints, weights, coupling, overlap_blocks,
                 multipole_blocks, filter_cutoff=None, symmetry=None):
        from ..core.hamiltonian import assemble_block_matrix
        from .local_split import split_width

        self.basis = list(basis)
        self.atom_of_orbital = list(atom_of_orbital)
        self.projectors = list(projectors)
        self.datasets = list(datasets)
        self.centers = [np.asarray(c, dtype=float) for c in centers]
        self.grid = grid
        self.lattice = rc.lattice_vectors(grid)
        self.volume = rc.cell_volume(grid)
        self.kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        self.weights = np.asarray(weights, dtype=float)
        self.sigma = split_width(grid)
        self.filter_cutoff = filter_cutoff
        #: :class:`GridSymmetry` the k-points were reduced with, or ``None``
        #: (time reversal only).  Every density assembled from the wedge is
        #: symmetrized with it.
        self.symmetry = symmetry
        self.M = len(self.basis)
        self.charges = np.array([float(d.valence_charge)
                                 for d in self.datasets])
        self.D_ion = assemble_block_matrix(
            self.projectors, coupling,
            diagonal=[p.kb_energy for p in self.projectors])
        self.q_overlap = assemble_block_matrix(self.projectors, overlap_blocks)
        self._positions = {}
        for position, projector in enumerate(self.projectors):
            self._positions.setdefault(projector.atom_index, []).append(position)
        self.multipole_blocks = dict(multipole_blocks)
        self.channels = sorted(self.multipole_blocks)
        self.G = rc.wavevectors(grid)
        self.kernel = rc.coulomb_kernel(self.G)
        self._compensation_grid = None
        self._compensation_coulomb = None
        self.kpoint_data = self._kpoints()

    # -- per k-point -------------------------------------------------------- #

    def _grid_points(self):
        g = self.grid
        return (g.X.ravel(), g.Y.ravel(), g.Z.ravel())

    def _cell_region(self):
        """Center and bounding radius (Bohr) of the cell the grid spans.

        An atom and a grid point can each lie a radius away from the center,
        so an image list for a function of support ``s`` centered on an atom
        must reach ``s + 2 radius``.
        """
        center = 0.5 * self.lattice.sum(axis=1) + rc.grid_origin(self.grid)
        corners = [0.5 * (s1 * self.lattice[:, 0] + s2 * self.lattice[:, 1]
                          + s3 * self.lattice[:, 2])
                   for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
        radius = max(float(np.linalg.norm(c)) for c in corners)
        return center, radius

    def _kpoints(self) -> list:
        """:class:`KPointMatrices` of the crystal's own (reduced) k-points."""
        return self.kpoint_matrices(self.kpoints, self.weights)

    def cartesian_kpoints(self, fractional) -> np.ndarray:
        """Fractional k-points (reciprocal-lattice units) in Bohr^-1."""
        B = rc.reciprocal_vectors(self.lattice)
        return np.atleast_2d(np.asarray(fractional, dtype=float)) @ B.T

    @single_threaded_blas
    def kpoint_matrices(self, kpoints, weights=None) -> list:
        """:class:`KPointMatrices` at arbitrary Cartesian ``kpoints`` (Bohr^-1).

        ``weights`` defaults to zero: points off the SCF mesh carry none.
        Each region's Bloch sums -- the cell grid, every projector sphere and
        every short-range sphere -- are evaluated for a block of k-points at
        a time: the images and their radial values do not depend on k, so a
        block shares them, and the block size bounds the memory
        (:data:`KPOINT_BLOCK_BYTES`).
        """
        all_kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        all_weights = (np.zeros(len(all_kpoints)) if weights is None
                       else np.asarray(weights, dtype=float))
        center, radius = self._cell_region()
        block = self.kpoint_block()
        dV = self.grid.dV
        out = []
        for start in range(0, len(all_kpoints), block):
            kpoints = all_kpoints[start:start + block]
            psi_all = bloch_values(self.basis, self.lattice, kpoints,
                                   self._grid_points(), center, radius)
            C_all = self._projections(kpoints)
            V_all = self._short_range_local(kpoints)
            for i, k in enumerate(kpoints):
                weight = all_weights[start + i]
                psi, C = psi_all[i], C_all[i]
                S = (psi.conj() @ psi.T) * dV + C @ self.q_overlap @ C.conj().T
                fixed = (self._kinetic(psi, k) + V_all[i]
                         + C @ self.D_ion @ C.conj().T)
                out.append(KPointMatrices(
                    k=np.asarray(k, float), weight=float(weight), psi=psi,
                    overlap=0.5 * (S + S.conj().T),
                    fixed=0.5 * (fixed + fixed.conj().T), projections=C))
        return out

    def kpoint_block(self) -> int:
        """How many k-points' Bloch sums fit :data:`KPOINT_BLOCK_BYTES`."""
        largest = max(self.grid.size, 131072)
        return max(1, int(KPOINT_BLOCK_BYTES // (16 * self.M * largest)))

    def _periodic_part(self, psi, k):
        """``u = e^{-i k.r} chi`` on the grid, transformed: ``(M, n1, n2, n3)``."""
        g = self.grid
        phase = np.exp(-1j * (k[0] * g.X + k[1] * g.Y + k[2] * g.Z)).ravel()
        u = (psi * phase[None, :]).reshape(len(psi), *g.shape)
        return np.fft.fftn(u, axes=(1, 2, 3)) * g.dV

    def _kinetic(self, psi, k) -> np.ndarray:
        r""":math:`\tfrac12\int|\nabla\chi|^2`, spectral on the periodic part."""
        transform = self._periodic_part(psi, k).reshape(len(psi), -1)
        Gk = self.G.reshape(3, -1) + np.asarray(k)[:, None]
        weight = 0.5 * np.sum(Gk * Gk, axis=0) / self.volume
        return (transform.conj() * weight) @ transform.T

    def bloch_gradients(self, psi, k) -> np.ndarray:
        r""":math:`\nabla\chi` on the grid, ``(3, M, ngrid)``."""
        g = self.grid
        transform = self._periodic_part(psi, k)
        phase = np.exp(1j * (k[0] * g.X + k[1] * g.Y + k[2] * g.Z)).ravel()
        N = int(np.prod(g.shape))
        out = []
        for c in range(3):
            factor = 1j * (self.G[c] + k[c])
            u = np.fft.ifftn(transform * factor[None], axes=(1, 2, 3)) \
                * N / (g.dV * N)
            out.append(u.reshape(len(psi), -1) * phase[None, :])
        return np.stack(out)

    def _projections(self, kpoints, only=None) -> np.ndarray:
        r"""``C[k, mu, p]`` :math:`= \langle\chi_{\mu\mathbf k}|\tilde p_p\rangle`.

        Projectors of one atom with the same cutoff share one sphere rule, so
        the Bloch sums are evaluated once per distinct sphere.
        """
        from .paw import _projector_sphere
        C = np.zeros((len(kpoints), self.M, len(self.projectors)),
                     dtype=complex)
        spheres: dict = {}
        for p, projector in enumerate(self.projectors):
            if only is not None and p not in only:
                continue             # left zero: the caller wants these alone
            key = (projector.atom_index, round(float(projector.r_cut), 12))
            spheres.setdefault(key, []).append(p)
        for members in spheres.values():
            first = self.projectors[members[0]]
            rows = reaching(self.basis, self.lattice, first.center,
                            float(first.r_cut))
            if rows.size == 0:
                continue
            near = [self.basis[mu] for mu in rows]
            points, weights = _projector_sphere(first)
            values = np.stack([self.projectors[p].evaluate(*points).ravel()
                               * weights.ravel() for p in members])
            block = np.zeros((len(kpoints), rows.size, len(members)),
                             dtype=complex)
            for part, *columns in _sphere_chunks(points, *values):
                chi = bloch_values(near, self.lattice, kpoints, part,
                                   first.center, float(first.r_cut))
                block += np.einsum("kmx,px->kmp", chi.conj(),
                                   np.stack(columns))
            C[:, rows[:, None], np.asarray(members)[None, :]] += block
        return C

    def _short_range_local(self, kpoints) -> np.ndarray:
        """``V_sr[k]`` -- the short-range local term at ``kpoints``."""
        return sum(self.short_range_atom(atom, kpoints)
                   for atom in range(len(self.datasets)))

    def short_range_atom(self, atom, kpoints, center=None) -> np.ndarray:
        """``(nk, M, M)``: atom ``atom``'s short-range local potential between
        the Bloch sums, its sphere centered at ``center`` (default: the atom).
        The force differentiates it by moving the sphere alone."""
        from .local_split import (CRYSTAL_AZIMUTHAL_POINTS,
                                  CRYSTAL_POLAR_POINTS,
                                  CRYSTAL_RADIAL_POINTS, local_cutoff,
                                  short_range_potential, short_range_radius)
        dataset = self.datasets[atom]
        center = self.centers[atom] if center is None else np.asarray(center,
                                                                      float)
        radius = short_range_radius(dataset, self.sigma)
        points, weights, r, _dirs = _sphere_rule(
            center, radius, CRYSTAL_RADIAL_POINTS, CRYSTAL_POLAR_POINTS,
            CRYSTAL_AZIMUTHAL_POINTS, panel=local_cutoff(dataset))
        potential = np.repeat(short_range_potential(dataset, self.sigma, r),
                              weights.size // r.size)
        out = np.zeros((len(kpoints), self.M, self.M), dtype=complex)
        # Only the functions that reach the sphere: the rest are zero on it.
        rows = reaching(self.basis, self.lattice, center, radius)
        if rows.size == 0:
            return out
        near = [self.basis[mu] for mu in rows]
        block = np.zeros((len(kpoints), rows.size, rows.size), dtype=complex)
        for part, weighted in _sphere_chunks(points, weights * potential):
            chi = bloch_values(near, self.lattice, kpoints, part, center,
                               radius)
            for i in range(len(kpoints)):
                block[i] += (chi[i].conj() * weighted) @ chi[i].T
        out[:, rows[:, None], rows[None, :]] = block
        return out

    # -- charges ------------------------------------------------------------ #

    def density(self, matrices, kpoint_data=None) -> tuple[np.ndarray, dict]:
        r"""``(n~, q)`` of k-resolved density matrices ``P_k``.

        :math:`\tilde n = \sum_k w_k \sum_{\mu\nu} P^k_{\mu\nu}\chi_\mu\chi_\nu^*`
        and :math:`q^A_{LM} = \sum_k w_k \operatorname{tr}(P^k Q^A_{LM}(k))`,
        the moments through :meth:`moment_traces`.  ``kpoint_data`` names
        the k-points ``matrices`` belong to (default: the crystal's own).
        """
        kpoint_data = self.kpoint_data if kpoint_data is None else kpoint_data
        if len(kpoint_data) != len(matrices):
            raise ValueError(f"{len(matrices)} density matrices for "
                             f"{len(kpoint_data)} k-points")
        rho = np.zeros(self.grid.size)
        R = np.zeros((len(self.projectors),) * 2, dtype=complex)
        for data, P in zip(kpoint_data, matrices):
            rho += data.weight * np.real(np.sum(data.psi * (P @ data.psi.conj()),
                                                axis=0))
            C = data.projections
            R += data.weight * (C.conj().T @ P @ C)
        q = self.moment_traces(R)
        if self.symmetry is not None:
            rho, q = self.symmetry.field(rho), self.symmetry.moments(q)
        return rho, q

    # -- compensation moments ----------------------------------------------- #
    #
    # Each moment operator Q^A_LM(k) = C_A blk C_A^dagger has the rank of atom
    # A's projector block, so neither the Hamiltonian nor the density needs
    # it as a dense M x M matrix: both go through the (P, P) projector space.
    # Dense, they took 72 M^2 complex numbers per k-point for 8 silicon atoms
    # and 6.4 GB per k-point for 64.

    def moment_operator(self, w) -> np.ndarray:
        r"""``(P, P)``: :math:`\sum_{A,LM} w^A_{LM}\,\mathrm{blk}^A_{LM}` on
        each atom's projector block, so that
        :math:`\sum w^A_{LM} Q^A_{LM}(k) = C(k)\,D\,C(k)^\dagger`."""
        D = np.zeros((len(self.projectors),) * 2, dtype=complex)
        for channel, blk in self.multipole_blocks.items():
            block = np.ix_(self._positions[channel[0]],
                           self._positions[channel[0]])
            D[block] += w[channel] * blk
        return D

    def moment_traces(self, R) -> dict:
        r"""``{(A, L, M): q}`` of the projected density matrix
        :math:`R = \sum_k w_k C(k)^\dagger P^k C(k)` (``(P, P)``):
        :math:`q^A_{LM} = \operatorname{tr}(R_{AA}\,\mathrm{blk}^A_{LM})`."""
        q = {}
        for channel, blk in self.multipole_blocks.items():
            own = self._positions[channel[0]]
            q[channel] = np.sum(R[np.ix_(own, own)] * blk.T)
        return q

    def kinetic_energy_density(self, matrices, gradients=None) -> np.ndarray:
        r""":math:`\tau = \tfrac12\sum_k w_k \sum P^k_{\mu\nu}
        \nabla\chi_\mu\cdot\nabla\chi^*_\nu`, symmetrized like the density.

        ``gradients`` (per k-point, :meth:`bloch_gradients`) reuses ones a
        caller already holds -- the SCF evaluates tau every iteration.
        """
        if gradients is None:
            gradients = [self.bloch_gradients(d.psi, d.k)
                         for d in self.kpoint_data]
        tau = np.zeros(self.grid.size)
        for data, P, grads in zip(self.kpoint_data, matrices, gradients):
            for d in grads:
                tau += 0.5 * data.weight * np.real(np.sum(d * (P @ d.conj()),
                                                          axis=0))
        return tau if self.symmetry is None else self.symmetry.field(tau)

    def compensation_grid(self) -> dict:
        r"""``{channel: g_hat(G)}`` on the grid's reciprocal set (filtered)."""
        if self._compensation_grid is None:
            transforms = self.compensation_transforms(self.G)
            self._compensation_grid = dict(zip(self.channels, transforms))
        return self._compensation_grid

    def compensation_transforms(self, vectors) -> np.ndarray:
        r"""``(n_ch, ...)``: every channel's compensation shape,
        :math:`\int e^{-i\mathbf g\cdot\mathbf r}\hat g_{LM}(\mathbf r -
        \mathbf R_A)\,d\mathbf r`, at the vectors ``(3, ...)``."""
        vectors = np.asarray(vectors, dtype=float)
        norm = rc.spherical(vectors)[0]
        if not self.channels:
            return np.zeros((0,) + vectors.shape[1:], dtype=complex)
        return np.stack([rc.multipole_transform(
            vectors, self.centers[atom], self._shape_transform(atom, L, norm),
            L, M) for atom, L, M in self.channels])

    def projector_columns(self, atom) -> list:
        """The columns of the projections ``C[:, p]`` that hold ``atom``'s
        projectors."""
        return list(self._positions.get(atom, []))

    def bloch_sums(self, kpoints) -> np.ndarray:
        """``(nk, M, ngrid)``: the basis' Bloch sums on the grid at the
        Cartesian ``kpoints``."""
        center, radius = self._cell_region()
        return bloch_values(self.basis, self.lattice,
                            np.atleast_2d(np.asarray(kpoints, dtype=float)),
                            self._grid_points(), center, radius)

    def projections(self, kpoints) -> np.ndarray:
        r"""``(nk, M, P)``: :math:`\langle\chi_{\mu\mathbf k}|\tilde
        p_p\rangle` at the Cartesian ``kpoints``."""
        return self._projections(np.atleast_2d(np.asarray(kpoints,
                                                          dtype=float)))

    def _dense_set(self, shift) -> np.ndarray:
        r"""``(3, n)``: the dense set's vectors :math:`\mathbf G + \mathbf p`
        within :data:`COMPENSATION_CUTOFF`."""
        shift = np.asarray(shift, dtype=float)
        B = rc.reciprocal_vectors(self.lattice)
        Gs = rc.lattice_translations(
            B, COMPENSATION_CUTOFF + float(np.linalg.norm(shift))).T
        Gs = Gs + shift[:, None]
        return Gs[:, np.sum(Gs * Gs, axis=0) <= COMPENSATION_CUTOFF ** 2]

    def dense_compensation_gradient(self, shift, kernel) -> np.ndarray:
        r"""``(3, n_ch, n_ch)``: :meth:`dense_compensation`'s sum with each
        term weighted by :math:`i(\mathbf G + \mathbf p)` (no tail) -- its
        derivative when atom A moves is this times
        :math:`\delta_{aA} - \delta_{bA}`."""
        Gs = self._dense_set(shift)
        n = len(self.channels)
        out = np.zeros((3, n, n), dtype=complex)
        for start in range(0, Gs.shape[1], DENSE_BLOCK):
            G = Gs[:, start:start + DENSE_BLOCK]
            F = self.compensation_transforms(G)
            v = kernel(np.sum(G * G, axis=0))
            for d in range(3):
                out[d] += (F.conj() * (v * 1j * G[d])) @ F.T
        return out / self.volume

    def dense_compensation(self, shift, kernel) -> np.ndarray:
        r"""``(n_ch, n_ch)``: :math:`\tfrac1\Omega\sum_{\mathbf G}
        v(|\mathbf G + \mathbf p|)\,\hat g_a^*\hat g_b` over the dense
        set (:data:`COMPENSATION_CUTOFF`) at the wave vector ``shift``
        :math:`\mathbf p`, with the on-site tail of :meth:`dense_coulomb`.

        The compensation charges of Bloch pair densities between themselves
        -- a screened hybrid's exchange; ``kernel`` maps :math:`|\mathbf G
        + \mathbf p|^2` to the kernel, which also weights the tail.
        """
        Gs = self._dense_set(shift)
        n = len(self.channels)
        U = np.zeros((n, n), dtype=complex)
        for start in range(0, Gs.shape[1], DENSE_BLOCK):
            G = Gs[:, start:start + DENSE_BLOCK]
            F = self.compensation_transforms(G)
            U += (F.conj() * kernel(np.sum(G * G, axis=0))) @ F.T
        U /= self.volume
        for a, (atom, L, _M) in enumerate(self.channels):
            U[a, a] += self._tail(atom, L, kernel)
        return U

    def _shape_transform(self, atom, L, q) -> np.ndarray:
        r""":math:`F_L(q)` of atom ``atom``'s compensation shape, interpolated
        from a table built once per (shape, ``L``)."""
        q = np.asarray(q, dtype=float)
        table_q, table_F = self._shape_table(atom, L, float(np.max(q,
                                                                    initial=0.0)))
        return np.interp(q, table_q, table_F)

    def _shape_table(self, atom, L, q_max: float):
        from .multipoles import shape_function
        r_g = float(self.datasets[atom].compensation_radius)
        key = (round(r_g, 12), int(L))
        # Shared by every crystal: a strained cell or a relaxation step
        # rebuilt the same transforms.
        cached = _SHAPE_TABLES.get(key)
        if cached is not None and cached[0][-1] >= q_max:
            return cached
        top = max(q_max, COMPENSATION_CUTOFF, SHAPE_TABLE_MAX)
        q = np.linspace(0.0, top, int(top / SHAPE_TABLE_STEP) + 1)
        r = np.linspace(0.0, r_g, SHAPE_POINTS)
        table = (q, rc.radial_transform(r, shape_function(r, r_g, L), L, q))
        _SHAPE_TABLES[key] = table
        return table

    def ion_charge(self) -> np.ndarray:
        r"""The Gaussian ions on the grid's reciprocal set (electron sign)."""
        return self._ion_transform(self.G)

    def dense_coulomb(self):
        r"""``(U, U_ion, E_ion)`` of the compact charges, dense in G.

        ``U[a, b]`` :math:`= \int\int \hat g_a^*\hat g_b/r` over the lattice,
        ``U_ion[a]`` :math:`= \int\int \hat g_a^*\,\rho_{\rm ion}/r` and
        ``E_ion`` the Gaussian ions' own reciprocal energy, each with
        :math:`\mathbf G = 0` dropped.  The on-site self-energy beyond the
        cutoff is added to the diagonal of ``U`` (:meth:`_tail`).
        """
        if self._compensation_coulomb is not None:
            return self._compensation_coulomb
        B = rc.reciprocal_vectors(self.lattice)
        Gs = rc.lattice_translations(B, COMPENSATION_CUTOFF).T
        Gs = Gs[:, np.sum(Gs * Gs, axis=0) > 0.0]
        n = len(self.channels)
        U = np.zeros((n, n), dtype=complex)
        U_ion = np.zeros(n, dtype=complex)
        E_ion = 0.0
        for start in range(0, Gs.shape[1], DENSE_BLOCK):
            G = Gs[:, start:start + DENSE_BLOCK]
            norm = rc.spherical(G)[0]
            kernel = 4.0 * np.pi / norm ** 2
            F = np.stack([rc.multipole_transform(
                G, self.centers[atom], self._shape_transform(atom, L, norm),
                L, M) for atom, L, M in self.channels]) if n else \
                np.zeros((0, G.shape[1]), dtype=complex)
            ion = self._ion_transform(G)
            U += (F.conj() * kernel) @ F.T
            U_ion += (F.conj() * kernel) @ ion
            E_ion += 0.5 * float(np.sum(kernel * np.abs(ion) ** 2))
        U /= self.volume
        U_ion /= self.volume
        E_ion /= self.volume
        for a, (atom, L, _M) in enumerate(self.channels):
            U[a, a] += self._tail(atom, L)
        self._compensation_coulomb = (U, U_ion, E_ion)
        return self._compensation_coulomb

    def _tail(self, atom, L, kernel=None) -> float:
        r""":math:`8\int_{q_c}^\infty F_L(q)^2\,dq` -- the on-site
        self-energy of a unit :math:`g_LY_{LM}` beyond
        :data:`COMPENSATION_CUTOFF` (the images' share there averages out)
        -- under the bare kernel, or under ``kernel`` (a function of
        :math:`q^2`) weighted by its ratio to the bare one."""
        table_q, table_F = self._shape_table(atom, L, SHAPE_TABLE_MAX)
        keep = table_q >= COMPENSATION_CUTOFF
        q = table_q[keep]
        ratio = (1.0 if kernel is None
                 else kernel(q * q) * q * q / (4.0 * np.pi))
        return float(8.0 * np.trapezoid(table_F[keep] ** 2 * ratio, q))

    def _ion_transform(self, G) -> np.ndarray:
        """The Gaussian ions at the reciprocal vectors ``G`` (electron sign)."""
        G2 = np.sum(G * G, axis=0)
        total = np.zeros(G.shape[1:], dtype=complex)
        for Z, center in zip(self.charges, self.centers):
            total += -Z * np.exp(-0.5 * self.sigma ** 2 * G2) \
                * rc.structure_factor(G, center)
        return total

    # -- the constants ------------------------------------------------------ #

    def ion_constants(self, centers=None) -> float:
        r"""Point-ion energy minus the Gaussian ions' reciprocal energy.

        :math:`-\sum_A Z_A^2/(2\sqrt\pi\sigma) + \tfrac12\sum'_{A,B,\mathbf R}
        Z_AZ_B\,\mathrm{erfc}(d/2\sigma)/d`, at ``centers`` (default: the
        atoms').
        """
        sigma = self.sigma
        centers = self.centers if centers is None else centers
        energy = -float(np.sum(self.charges ** 2)) / (2.0 * np.sqrt(np.pi)
                                                     * sigma)
        reach = ION_PAIR_REACH * sigma
        for a, (Za, Ra) in enumerate(zip(self.charges, centers)):
            for b, (Zb, Rb) in enumerate(zip(self.charges, centers)):
                d0 = Rb - Ra
                for R in rc.lattice_translations(self.lattice,
                                                 reach + np.linalg.norm(d0)):
                    d = float(np.linalg.norm(d0 + R))
                    if d < 1e-10 or d > reach:
                        continue
                    energy += 0.5 * Za * Zb * erfc(d / (2.0 * sigma)) / d
        return energy

    def onsite_ion_compensation(self) -> dict:
        r"""``{channel: int g_A,00 v^lr_A}`` -- the on-site long-range term the
        reciprocal sum includes and the dataset convention excludes."""
        from .local_split import long_range_potential
        from .multipoles import shape_function
        out = {}
        for atom, L, M in self.channels:
            if L != 0:
                continue
            dataset = self.datasets[atom]
            r_g = float(dataset.compensation_radius)
            r = np.linspace(0.0, r_g, SHAPE_POINTS)
            integrand = (shape_function(r, r_g, 0)
                         * long_range_potential(dataset.valence_charge,
                                                self.sigma, r) * r * r)
            # g_0 Y_00 integrated over angles: sqrt(4 pi) g_0.
            out[(atom, 0, 0)] = float(np.sqrt(4.0 * np.pi)
                                      * np.trapezoid(integrand, r))
        return out

    def short_range_ion_compensation(self, centers=None, atom=None) -> dict:
        r"""``{channel: int g_A,LM sum' v^sr_B}`` over every other atom and
        image (the short-range half of :meth:`~.paw.PAWIntegrals.compensation_ionic_at`),
        at ``centers`` (default: the atoms').  With ``atom`` only the terms
        that atom takes part in -- its own shapes, and every shape against its
        potential -- which is all a displacement of it changes."""
        from ..basis._angular import spherical_harmonic
        from .local_split import short_range_potential, short_range_radius
        from .multipoles import shape_function
        from .paw import (COMPENSATION_RADIAL_POINTS,
                          PROJECTION_AZIMUTHAL_POINTS, PROJECTION_POLAR_POINTS)

        centers = self.centers if centers is None else centers
        out = {}
        for owner, L, M in self.channels:
            r_g = float(self.datasets[owner].compensation_radius)
            points, weights, r, dirs = _sphere_rule(
                centers[owner], r_g, COMPENSATION_RADIAL_POINTS,
                PROJECTION_POLAR_POINTS, PROJECTION_AZIMUTHAL_POINTS)
            theta = np.arccos(np.clip(dirs[2], -1.0, 1.0))
            phi = np.arctan2(dirs[1], dirs[0])
            density = (shape_function(r, r_g, L)[:, None]
                       * spherical_harmonic(L, M, theta, phi)[None, :]).ravel()
            x = np.stack(points, axis=1)
            total = 0.0j
            for other, dataset in enumerate(self.datasets):
                if atom is not None and atom not in (other, owner):
                    continue
                reach = short_range_radius(dataset, self.sigma) + r_g
                d0 = centers[owner] - centers[other]
                for R in rc.lattice_translations(self.lattice,
                                                 reach + np.linalg.norm(d0)):
                    if other == owner and np.linalg.norm(R) < 1e-10:
                        continue
                    if np.linalg.norm(d0 - R) > reach:
                        continue
                    distance = np.linalg.norm(
                        x - (centers[other] + R)[None, :], axis=1)
                    total += np.sum(weights * density
                                    * short_range_potential(dataset,
                                                            self.sigma,
                                                            distance))
            out[(owner, L, M)] = total
        return out

    def core_density(self) -> np.ndarray | None:
        r"""The datasets' smooth cores on the grid, with their images.

        Sampled from the radial table as it is, as the molecular path does
        (:func:`~mandacaru.integrals.exchange_correlation.
        core_density_on_grid`) and as the dataset was unscreened with.  It
        used to be Fourier-filtered at the basis cutoff, which put the crystal
        in another functional than the molecule and the dataset: Ne in a box
        sat 2.9 mHa below the same atom solved as a molecule with a 300 eV
        basis filter, at every grid spacing and box size (HISTORY, "K20").
        """
        from ..integrals.exchange_correlation import xc_core_density
        return self._place_cores(xc_core_density)

    def core_tau(self) -> np.ndarray | None:
        r"""The cores' kinetic-energy densities on the grid, with images.

        Each atom's :math:`\tilde\rho_c'^2/8\tilde\rho_c` evaluated radially
        (:func:`~mandacaru.integrals.exchange_correlation.core_tau_function`)
        and placed like the core density.  The ratio taken on the grid would
        divide by the core's vanishing tail.
        """
        from ..integrals.exchange_correlation import core_tau_function

        def table(dataset):
            function = core_tau_function(dataset)
            return (None if function is None
                    else function(np.asarray(dataset.r, dtype=float)))
        return self._place_cores(table)

    def _place_cores(self, table) -> np.ndarray | None:
        """Sum of ``table(dataset)`` (a radial array or ``None``) over atoms
        and images."""
        total = None
        for atom in range(len(self.datasets)):
            values = self.place_core(atom, table)
            if values is not None:
                total = values if total is None else total + values
        return total

    def place_core(self, atom, table, center=None) -> np.ndarray | None:
        """Atom ``atom``'s share of :meth:`_place_cores`, centered at
        ``center`` (default: the atom) -- what a force moves."""
        from scipy.interpolate import CubicSpline

        g = self.grid
        x = np.stack(self._grid_points(), axis=1)
        dataset = self.datasets[atom]
        center = self.centers[atom] if center is None else np.asarray(center,
                                                                      float)
        values_r = table(dataset)
        if values_r is None:
            return None
        r = np.asarray(dataset.r, dtype=float)
        values_r = np.asarray(values_r, dtype=float)
        peak = float(np.max(np.abs(values_r)))
        significant = np.nonzero(np.abs(values_r) > 1e-14 * peak)[0]
        support = float(r[significant[-1]]) if significant.size else 0.0
        spline = CubicSpline(r, values_r)
        values = np.zeros(g.size)
        for R in rc.lattice_translations(
                self.lattice, support + 2.0 * self._cell_region()[1]):
            distance = np.linalg.norm(x - (center + R)[None, :], axis=1)
            inside = distance <= support
            values[inside] += spline(distance[inside])
        return values

    def initial_density(self) -> tuple[np.ndarray, dict]:
        """Superposed smooth valence densities and reference monopoles."""
        from scipy.interpolate import CubicSpline
        x = np.stack(self._grid_points(), axis=1)
        rho = np.zeros(self.grid.size)
        for dataset, center in zip(self.datasets, self.centers):
            r = np.asarray(dataset.r, dtype=float)
            values = np.asarray(dataset.valence_density, dtype=float)
            peak = float(np.max(np.abs(values)))
            significant = np.nonzero(np.abs(values) > 1e-12 * peak)[0]
            support = float(r[significant[-1]])
            spline = CubicSpline(r, values)
            for R in rc.lattice_translations(self.lattice,
                                             support + 2.0 * self._cell_region()[1]):
                distance = np.linalg.norm(x - (center + R)[None, :], axis=1)
                inside = distance <= support
                rho[inside] += spline(distance[inside])
        q = {c: 0.0j for c in self.channels}
        for atom, dataset in enumerate(self.datasets):
            if (atom, 0, 0) in q:
                q[(atom, 0, 0)] = dataset.compensation_charge / np.sqrt(4.0
                                                                       * np.pi)
        return rho, q

    def initial_magnetization(self, moments) -> np.ndarray:
        r"""The starting magnetization :math:`n_\uparrow - n_\downarrow`:
        each atom's superposed valence density weighted by its initial moment
        over its valence charge (clamped to that charge, so neither spin
        density starts negative)."""
        from scipy.interpolate import CubicSpline
        x = np.stack(self._grid_points(), axis=1)
        m = np.zeros(self.grid.size)
        moments = np.zeros(len(self.datasets)) if moments is None else \
            np.asarray(moments, dtype=float)
        for dataset, center, moment in zip(self.datasets, self.centers,
                                           moments):
            charge = float(dataset.valence_charge)
            if moment == 0.0 or charge <= 0.0:
                continue
            fraction = float(np.clip(moment / charge, -1.0, 1.0))
            r = np.asarray(dataset.r, dtype=float)
            values = np.asarray(dataset.valence_density, dtype=float)
            peak = float(np.max(np.abs(values)))
            significant = np.nonzero(np.abs(values) > 1e-12 * peak)[0]
            support = float(r[significant[-1]])
            spline = CubicSpline(r, values)
            for R in rc.lattice_translations(
                    self.lattice, support + 2.0 * self._cell_region()[1]):
                distance = np.linalg.norm(x - (center + R)[None, :], axis=1)
                inside = distance <= support
                m[inside] += fraction * spline(distance[inside])
        return m

    @property
    def n_electrons_neutral(self) -> float:
        return float(np.sum(self.charges))

    # What the run log and the citations read from an integrals object.
    split_local_potential = True

    def local_split_width(self) -> float:
        """Gaussian width ``sigma`` (Bohr) of the local-potential split."""
        return self.sigma

    @property
    def pseudopotentials(self) -> list:
        return self.datasets



# --------------------------------------------------------------------------- #
# Building a crystal from a geometry.
# --------------------------------------------------------------------------- #

def reduce_mesh(mesh, operations=None):
    """``(points, weights, operations)`` of a fractional k-mesh's wedge.

    ``operations`` (a :class:`GridSymmetry`) is first restricted to the
    subgroup that maps ``mesh`` onto itself; with none left, or none given,
    the mesh is reduced by time reversal alone and ``operations`` comes back
    ``None``.  ``weights`` sum to one.
    """
    if operations is not None:
        operations = operations.keeping_mesh(mesh)
    if operations is None:
        reduced, weights = time_reversal_reduce(mesh)
        return reduced, weights, None
    from ..core.symmetry import irreducible_kpoints
    zone = irreducible_kpoints(mesh, operations.info, time_reversal=True)
    return zone.points, zone.weights / zone.weights.sum(), operations


@single_threaded_blas
def build_crystal(atoms, h: float, options: dict | None = None, kpts=None,
                  family: str = "paw-lcao", grid=None, ghosts=(),
                  symmetry: bool = True, full_mesh: bool = False):
    r"""``(crystal, context)`` for a periodic ``atoms`` with a PAW-LCAO basis.

    The basis, projectors and dataset blocks are the molecular builder's
    (:func:`~.families.build_valence_hamiltonian`), so a crystal and a molecule
    share every radial function; the grid is the cell itself
    (``periodic=True``), the k-points a Monkhorst-Pack mesh reduced to its
    irreducible wedge by the space-group operations that map both the grid
    and the mesh onto themselves (:func:`grid_symmetry`,
    :meth:`GridSymmetry.keeping_mesh`), or by time reversal alone with
    ``symmetry=False`` or when no such operation survives.  No two-body tensor is built: the Kohn-Sham problem needs only
    the density's potential.

    ``ghosts`` are atom indices that keep their basis functions but carry no
    dataset, projectors, core or electrons (the counterpoise correction of a
    layered or molecular crystal); with any, the k-mesh is reduced by time
    reversal alone -- the space group of the real atoms is not the cell's.
    ``full_mesh`` keeps every k-point, unreduced and without symmetry: a
    finite electric field couples neighboring k-points along strings of the
    whole mesh and breaks the symmetry the reduction relies on.
    """
    from ..algorithms._hamiltonian_from_atoms import monkhorst_pack_kpts
    from ..basis.filtering import filter_cutoff
    from ..integrals import Grid
    from ..units import to_bohr
    from .families import pseudo_basis_arguments, resolve_family
    from .orbitals import pseudo_basis, valence_electrons
    from .paw import (DEFAULT_PROJECTOR_BASIS, get_paw, get_upaw,
                      paw_coupling_blocks, paw_multipole_blocks,
                      paw_overlap_blocks, paw_projectors)

    # The Bloch sums and the Bader ascent run in the C library: compile it
    # (or a stale copy missing their symbols, as a tree synced from another
    # machine has) before anything is sampled, as the molecular engine does.
    from ..integrals import _backend
    _backend.warn_fallback(_backend.ensure_backend())

    options = resolve_family(family).resolved_options(dict(options or {}))
    load = get_upaw if family == "upaw-lcao" else get_paw
    symbols = atoms.get_chemical_symbols()
    atoms = atoms.copy()
    atoms.wrap()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    potentials = {s: load(s, options.get("directory")) for s in set(symbols)}
    if grid is None:
        g = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, h=h,
                 units="angstrom", cell=cell, periodic=True)
    else:
        g = grid
        # A grid of another cell would be used without complaint and give
        # another crystal's answer: it must be periodic and span this cell.
        from ..units import ANGSTROM_TO_BOHR
        spanned = (np.asarray(g.step) @ np.diag(g.shape)).T
        if not (getattr(g, "periodic", False) and np.allclose(
                spanned, cell * ANGSTROM_TO_BOHR, atol=1e-6)):
            raise ValueError(
                "an explicit grid for a crystal must be periodic and span "
                "its cell (Grid(..., cell=atoms.cell, periodic=True))")
    k_c = filter_cutoff(options.get("filter"), max(g.dx, g.dy, g.dz))
    confinement: dict = {}
    polarization: dict = {}
    # Ghosts lend their basis functions; everything physical is the real
    # atoms'.
    from ..algorithms._hamiltonian_from_atoms import validate_ghosts
    ghosts = validate_ghosts(ghosts, len(symbols))
    real = [i for i in range(len(symbols)) if i not in ghosts]
    real_symbols = [symbols[i] for i in real]
    basis, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, filter_cutoff=k_c,
        **pseudo_basis_arguments(family, options, confinement=confinement,
                                 polarization=polarization))
    projectors = paw_projectors(
        real_symbols, positions[real], potentials,
        projector_basis=options.get("projector_basis",
                                    DEFAULT_PROJECTOR_BASIS))
    datasets = [potentials[s] for s in real_symbols]
    size, gamma, mesh = monkhorst_pack_kpts(kpts)
    if full_mesh:
        reduced = np.asarray(mesh, dtype=float)
        weights = np.full(len(reduced), 1.0 / len(reduced))
        operations = None
    else:
        operations = grid_symmetry(atoms, g) if symmetry and not ghosts \
            else None
        reduced, weights, operations = reduce_mesh(mesh, operations)
    B = rc.reciprocal_vectors(rc.lattice_vectors(g))
    kpoints = reduced @ B.T
    crystal = PeriodicPAW(
        basis, atom_of_orbital, projectors, datasets,
        [to_bohr(p, "angstrom") for p in positions[real]], g, kpoints, weights,
        paw_coupling_blocks(projectors, real_symbols, potentials),
        paw_overlap_blocks(projectors, real_symbols, potentials),
        paw_multipole_blocks(projectors, datasets), filter_cutoff=k_c,
        symmetry=operations)
    context = {"crystal": crystal, "atom_of_orbital": atom_of_orbital,
               "n_electrons": float(valence_electrons(real_symbols,
                                                      potentials)),
               "ghosts": tuple(sorted(ghosts)),
               "pseudopotentials": potentials, "family": family,
               "filter_cutoff": k_c, "options": dict(options),
               "confinement": confinement, "polarization": polarization,
               "kpts_size": size, "kpts_gamma": gamma,
               "kpoints_fractional": reduced}
    return crystal, context
