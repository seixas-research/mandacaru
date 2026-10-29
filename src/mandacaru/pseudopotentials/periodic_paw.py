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
  G + \mathbf k|^2` acting on the periodic part of each sum (averaged over
  the aliases the grid lattice's operations pick, so it commutes with them);
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

from ..utils.profiling import profile_stage

from dataclasses import dataclass

import numpy as np
from scipy.linalg.blas import zgemm
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

#: The remainders of the augmentation densities beyond their multipoles
#: (:meth:`PeriodicPAW.remainder_transforms`) are tabulated on
#: ``[0, REMAINDER_TABLE_MAX]`` (Bohr^-1) with this step and interpolated
#: linearly; a vector past the end reads the last entry (zero by then: the
#: transforms of a smooth sphere density have fallen by 1e-4 of their peak).
REMAINDER_TABLE_STEP = 0.025
REMAINDER_TABLE_MAX = 12.0

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
                self.l, self.m, dx[away], dy[away], dz[away], real=False)
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
    past its cutoff -- the derivative of ``function.radial``, which the
    function itself provides (``radial_slope``)."""
    if isinstance(function, BasisGradient):
        return radial_slope(function.parent, r)
    return function.radial_slope(r)


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
    from ..integrals.sampling import radial_spline
    return radial_spline(function)



#: ``(r_g, L) -> (q, F_L(q))`` of a compensation shape (see
#: :meth:`PeriodicPAW._shape_table`).
_SHAPE_TABLES: dict = {}

#: Quadrature points per call when a sphere's Bloch sums are streamed: the
#: values at every point and k-point at once ran to gigabytes (a sphere's
#: tens of thousands of points times the 50 k-points of a band path).
SPHERE_CHUNK = 16384
#: Bytes of Bloch sums one sphere chunk may hold (:func:`_sphere_chunk_size`).
SPHERE_BYTES = 1 << 29


def _sphere_chunk_size(n_kpoints: int, n_functions: int) -> int:
    """Points per chunk of a sphere quadrature: :data:`SPHERE_CHUNK`, fewer
    when ``n_kpoints`` x ``n_functions`` Bloch sums (and their conjugates)
    of that many points would pass :data:`SPHERE_BYTES`.  A function of the
    sizes alone, never of the thread count."""
    fit = SPHERE_BYTES // (32 * max(1, n_kpoints) * max(1, n_functions))
    return int(max(4096, min(SPHERE_CHUNK, fit)))


def _sphere_chunks(points, *arrays, size: int = SPHERE_CHUNK):
    """Yield ``(points, *arrays)`` chunks of ``size`` points of a quadrature,
    its points sorted into 1-Bohr bins (a sum over the points does not depend
    on their order, and :func:`bloch_values` then has nothing to un-permute)."""
    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
    order = np.lexsort((np.floor(z), np.floor(y), np.floor(x)))
    x, y, z = x[order], y[order], z[order]
    arrays = [np.asarray(a).ravel()[order] for a in arrays]
    for start in range(0, x.size, size):
        part = slice(start, start + size)
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


@profile_stage('integration: crystal basis sampling')
def bloch_values(functions, lattice, kpoints, points, region_center,
                 region_radius, *, presorted=False) -> np.ndarray:
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
    order = None if presorted else np.lexsort((np.floor(z), np.floor(y), np.floor(x)))
    if order is not None:
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
        if spline is not None and backend.HAS_NATIVE_BACKEND:
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
                        functions[mu].l, functions[mu].m, theta, phi,
                        real=False))
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
    if order is None or np.array_equal(order, np.arange(order.size)):
        return out
    restored = np.empty_like(out)
    restored[:, :, order] = out
    return restored


@profile_stage('integration: crystal basis projection')
def bloch_projections(functions, lattice, kpoints, points, columns,
                      region_center, region_radius) -> np.ndarray:
    r"""``(nk, M, P)``: :math:`\sum_x \chi^*_{\mu\mathbf k}(x)\,
    \mathrm{columns}[p, x]` -- the Bloch sums of ``functions`` projected on
    ``columns`` (``(P, npts)``, e.g. projectors times quadrature weights)
    over the quadrature ``points``, which lie within ``region_radius`` of
    ``region_center``.

    The same numbers as :func:`bloch_values` contracted with ``columns``, but
    the contraction is done image by image, before the k-points' phases:
    :math:`\sum_R e^{-i\mathbf k\cdot\mathbf R}\sum_x
    \phi^*(x - \mathbf R)\,\mathrm{columns}[p, x]`.  The work on the
    points is then independent of the number of k-points.  ``points`` should
    be spatially ordered (:func:`_sphere_chunks`): the C kernel culls
    lattice images per block of consecutive points.
    """
    from ..integrals import _backend as backend

    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    columns = np.atleast_2d(np.asarray(columns, dtype=complex))
    out = np.zeros((len(kpoints), len(functions), len(columns)),
                   dtype=complex)
    region_center = np.asarray(region_center, dtype=float)
    groups: dict = {}
    for mu, function in enumerate(functions):
        key = _shell_key(function)
        groups.setdefault(key if key is not None else ("single", mu),
                          []).append(mu)
    rest: list = []
    for members in groups.values():
        first = functions[members[0]]
        center = np.asarray(first.center, dtype=float)
        support = _support(first)
        reach = support + float(region_radius)
        offset = float(np.linalg.norm(center - region_center))
        gradient = isinstance(first, BasisGradient)
        shared = _shell_key(first) is not None and not gradient
        spline = _radial_spline(first) if (shared or gradient) else None
        if spline is None or not backend.HAS_NATIVE_BACKEND:
            rest.extend(members)
            continue
        candidates = [R for R in rc.lattice_translations(lattice,
                                                         reach + offset)
                      if np.linalg.norm(center + R - region_center) <= reach]
        if not candidates:
            continue
        translations = np.array(candidates, dtype=float)
        blocks = backend.bloch_project(
            (x, y, z), center, translations, support, spline, first.l,
            [functions[mu].m for mu in members], columns,
            derivative=first.axis if gradient else -1)
        if blocks is None:
            rest.extend(members)
            continue
        # Blocks are added in order: the sum does not depend on how many
        # threads evaluated them.
        images = blocks if blocks.ndim == 3 else blocks.sum(axis=0)
        phases = np.exp(-1j * (kpoints @ translations.T))        # (nk, n_R)
        out[:, members, :] += np.einsum("kr,rmp->kmp", phases, images)
    if rest:
        chi = bloch_values([functions[mu] for mu in rest], lattice, kpoints,
                           (x, y, z), region_center, region_radius)
        out[:, rest, :] += np.matmul(chi, columns.conj().T[None]).conj()
    return out


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
        return np.stack([spherical_harmonic(L, M, theta, phi, real=False)
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
    it, through ``grid_spacing``.
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
    #: The spin-orbit term :math:`h^{SO}(\mathbf k)`, ``(2M, 2M)`` over the
    #: spinor Bloch basis (alpha block first), or ``None`` for a crystal
    #: built without it (:meth:`PeriodicPAW.spin_orbit_matrix`).
    spin_orbit: np.ndarray | None = None


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
    filter_wavevector : float, optional
        The basis' Fourier-filter cutoff wave-vector (Bohr^-1), recorded so a strained
        cell rebuilds the same basis.  The partial core is *not* filtered
        with it (:meth:`core_density`).
    spin_orbit_projectors, spin_orbit_blocks : optional
        The spin-orbit term of a spin-orbit set (``lda-fr/``, ``pbe-fr/``):
        the projectors it acts through
        (:func:`~.paw.paw_spin_orbit_projectors`, complex harmonics) and
        its ``{(atom, l): D_SO}`` couplings
        (:func:`~.paw.paw_spin_orbit_blocks`) -- the molecule's term, Bloch
        summed (:meth:`spin_orbit_matrix`).  ``None`` for a scalar crystal.
    """

    def __init__(self, basis, atom_of_orbital, projectors, datasets, centers,
                 grid, kpoints, weights, coupling, overlap_blocks,
                 multipole_blocks, filter_wavevector=None, symmetry=None,
                 spin_orbit_projectors=None, spin_orbit_blocks=None,
                 native_context=None, precision="double"):
        from ..models.paw_channels import assemble_block_matrix
        from .local_split import split_width

        self.basis = list(basis)
        self._native_context = native_context
        if precision not in ("double", "single"):
            raise ValueError("precision must be double or single")
        self.precision = precision
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
        self.filter_wavevector = filter_wavevector
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
        #: The Coulomb kernel and the grid charges (:meth:`compensation_grid`,
        #: :meth:`ion_charge`) are means over the aliases the grid lattice's
        #: operations pick (:func:`~mandacaru.integrals.reciprocal.alias_mean`),
        #: so they commute with the operations as the kinetic matrix does.
        self.kernel = rc.alias_mean(grid, rc.coulomb_kernel)
        self._compensation_grid = None
        self._compensation_coulomb = None
        #: The spin-orbit projectors and couplings (empty without the term).
        self.spin_orbit_projectors = list(spin_orbit_projectors or [])
        self.spin_orbit_blocks = dict(spin_orbit_blocks or {})
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
    @profile_stage('integration: crystal k-point assembly')
    def kpoint_matrices(self, kpoints, weights=None) -> list:
        """:class:`KPointMatrices` at arbitrary Cartesian ``kpoints`` (Bohr^-1).

        ``weights`` defaults to zero: points off the SCF mesh carry none.
        The cell grid's Bloch sums are evaluated for a block of k-points at a
        time, the block size bounding the memory (:data:`KPOINT_BLOCK_BYTES`).
        The projector and short-range spheres are small and their images and
        radial values do not depend on k, so they are evaluated for as many
        k-points at once as their matrices fit in the same budget -- a block
        of grid sums no longer repeats them.
        """
        all_kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        all_weights = (np.zeros(len(all_kpoints)) if weights is None
                       else np.asarray(weights, dtype=float))
        center, radius = self._cell_region()
        block = self.kpoint_block()
        # The spheres' matrices are small next to the grid Bloch sums, and
        # their images and radial values are k-independent: evaluate them
        # for as many k-points as fit KPOINT_BLOCK_BYTES, not once per block
        # of grid sums.
        per_k = 16 * self.M * (self.M + len(self.projectors)
                               + (4 * self.M if self.has_spin_orbit else 0))
        wide = max(block, int(KPOINT_BLOCK_BYTES // per_k))
        dV = self.grid.dV
        out = []
        for outer in range(0, len(all_kpoints), wide):
            wide_k = all_kpoints[outer:outer + wide]
            C_wide = self._projections(wide_k)
            V_wide = self._short_range_local(wide_k)
            SO_wide = (self.spin_orbit_matrix(wide_k) if self.has_spin_orbit
                       else [None] * len(wide_k))
            for start in range(outer, outer + len(wide_k), block):
                kpoints = all_kpoints[start:start + block]
                psi_all = bloch_values(self.basis, self.lattice, kpoints,
                                       self._grid_points(), center, radius)
                shift = start - outer
                for i, k in enumerate(kpoints):
                    weight = all_weights[start + i]
                    psi_raw, C = psi_all[i], C_wide[shift + i]
                    psi = psi_raw
                    S = (psi.conj() @ psi.T) * dV + C @ self.q_overlap \
                        @ C.conj().T
                    fixed = (self._kinetic(psi, k) + V_wide[shift + i]
                             + C @ self.D_ion @ C.conj().T)
                    out.append(KPointMatrices(
                        k=np.asarray(k, float), weight=float(weight),
                        psi=psi.astype(np.complex64) if self.precision == "single" else psi,
                        overlap=0.5 * (S + S.conj().T),
                        fixed=0.5 * (fixed + fixed.conj().T), projections=C,
                        spin_orbit=SO_wide[shift + i]))
        return out

    @property
    def has_spin_orbit(self) -> bool:
        """Whether the crystal carries a spin-orbit term (a spin-orbit set)."""
        return bool(self.spin_orbit_blocks)

    def spin_orbit_matrix(self, kpoints) -> np.ndarray:
        r"""``(nk, 2M, 2M)``: the spin-orbit term between the spinor Bloch
        sums at the Cartesian ``kpoints``, alpha block first.

        The molecule's operator
        (:func:`~mandacaru.core.spin_orbit.spin_orbit_one_body`), with the
        projections of the Bloch sums on the home cell's spin-orbit
        projectors in place of the molecular ones -- the same unfolding as the
        scalar nonlocal term :math:`C(\mathbf k)\,D\,C(\mathbf k)^\dagger`:

        .. math::

            h^{SO}_{\mu\sigma,\nu\tau}(\mathbf k) = \sum_{a,l}
              \sum_{ij,mm'} C^{SO}_{\mu,lmi}(\mathbf k)\,D^{SO,a,l}_{ij}\,
              \langle lm\sigma|\mathbf L\cdot\mathbf S|lm'\tau\rangle\,
              C^{SO}_{\nu,lm'j}(\mathbf k)^* .

        Zero (and still ``(nk, 2M, 2M)``) without the term.
        """
        from ..core.spin_orbit import spin_orbit_one_body

        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        if not self.has_spin_orbit:
            return np.zeros((len(kpoints), 2 * self.M, 2 * self.M),
                            dtype=complex)
        C = self._projections(kpoints, projectors=self.spin_orbit_projectors)
        context = self._native_context
        if context is None:
            from .. import _native
            context = _native.Context(threads=1)
        return np.stack([spin_orbit_one_body(C[i], self.spin_orbit_projectors,
                                             self.spin_orbit_blocks,
                                             context=context)
                         for i in range(len(kpoints))])

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

    @profile_stage('integration: crystal kinetic matrices')
    def _kinetic(self, psi, k) -> np.ndarray:
        r""":math:`\tfrac12\int|\nabla\chi|^2`, spectral on the periodic part.

        The multiplier of each frequency is the mean of
        :math:`|\mathbf G + \mathbf k|^2` over its aliases under the grid
        lattice's operations (:func:`~mandacaru.integrals.reciprocal.bloch_aliases`),
        not the FFT box's alias: with the box, the k-points of one star had
        different kinetic spectra wherever an operation mixes the grid's
        axes (hcp, fcc), and a wedge symmetrized to the full mesh missed it
        by 1.6e-4 Ha on semicore hcp Mg (HISTORY 2026-10-11, K13b).
        """
        transform = self._periodic_part(psi, k).reshape(len(psi), -1)
        _vectors, squares = rc.bloch_aliases(self.grid, k)
        weight = 0.5 * squares.ravel() / self.volume
        return (transform.conj() * weight) @ transform.T

    def bloch_gradients(self, psi, k) -> np.ndarray:
        r""":math:`\nabla\chi` on the grid, ``(3, M, ngrid)``: the periodic
        part multiplied by the alias-averaged :math:`i(\mathbf G + \mathbf
        k)` of :func:`~mandacaru.integrals.reciprocal.bloch_aliases`, so a
        kinetic-energy density summed over the wedge and symmetrized is the
        full mesh's."""
        g = self.grid
        transform = self._periodic_part(psi, k)
        vectors, _squares = rc.bloch_aliases(g, k)
        phase = np.exp(1j * (k[0] * g.X + k[1] * g.Y + k[2] * g.Z)).ravel()
        out = []
        for c in range(3):
            u = np.fft.ifftn(transform * (1j * vectors[c])[None],
                             axes=(1, 2, 3)) / g.dV
            out.append(u.reshape(len(psi), -1) * phase[None, :])
        return np.stack(out)

    @profile_stage('integration: crystal projector quadrature')
    def _projections(self, kpoints, only=None, projectors=None) -> np.ndarray:
        r"""``C[k, mu, p]`` :math:`= \langle\chi_{\mu\mathbf k}|\tilde p_p\rangle`.

        Projectors of one atom with the same cutoff share one sphere rule, so
        the Bloch sums are evaluated once per distinct sphere.  ``projectors``
        defaults to the scalar nonlocal ones (the spin-orbit term passes its
        own set).
        """
        from .paw import _projector_sphere
        projectors = self.projectors if projectors is None else projectors
        C = np.zeros((len(kpoints), self.M, len(projectors)),
                     dtype=complex)
        spheres: dict = {}
        for p, projector in enumerate(projectors):
            if only is not None and p not in only:
                continue             # left zero: the caller wants these alone
            key = (projector.atom_index, round(float(projector.r_cut), 12))
            spheres.setdefault(key, []).append(p)
        for members in spheres.values():
            first = projectors[members[0]]
            rows = reaching(self.basis, self.lattice, first.center,
                            float(first.r_cut))
            if rows.size == 0:
                continue
            near = [self.basis[mu] for mu in rows]
            points, weights = _projector_sphere(first)
            values = np.stack([projectors[p].evaluate(*points).ravel()
                               * weights.ravel() for p in members])
            block = np.zeros((len(kpoints), rows.size, len(members)),
                             dtype=complex)
            for part, *columns in _sphere_chunks(points, *values):
                block += bloch_projections(
                    near, self.lattice, kpoints, part, np.stack(columns),
                    first.center, float(first.r_cut))
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
        center = self.centers[atom] if center is None else np.asarray(center,
                                                                      float)
        rows, block = self._short_range_block(atom, kpoints, center,
                                              self.basis, self.basis)
        out = np.zeros((len(kpoints), self.M, self.M), dtype=complex)
        if rows[0].size:
            out[:, rows[0][:, None], rows[1][None, :]] = block
        return out

    def short_range_cross(self, atom, kpoints, bra, ket) -> np.ndarray:
        """``(nk, len(bra), len(ket))``: :meth:`short_range_atom`'s matrix
        between two sets of functions -- ``bra`` rows, ``ket`` columns --
        without the blocks nobody asked for (a force needs the derivative
        functions against the basis, not against each other)."""
        rows, block = self._short_range_block(atom, kpoints,
                                              self.centers[atom], bra, ket)
        out = np.zeros((len(kpoints), len(bra), len(ket)), dtype=complex)
        if rows[0].size and rows[1].size:
            out[:, rows[0][:, None], rows[1][None, :]] = block
        return out

    @profile_stage("integration: PAW short-range local quadrature")
    def _short_range_block(self, atom, kpoints, center, bra, ket):
        """``((bra rows, ket rows), block)``: the short-range sphere's matrix
        between the functions of ``bra`` and ``ket`` that reach it (the same
        list twice evaluates its Bloch sums once)."""
        from .local_split import (CRYSTAL_AZIMUTHAL_POINTS,
                                  CRYSTAL_POLAR_POINTS,
                                  CRYSTAL_RADIAL_POINTS, local_cutoff,
                                  short_range_potential, short_range_radius)
        dataset = self.datasets[atom]
        radius = short_range_radius(dataset, self.sigma)
        points, weights, r, _dirs = _sphere_rule(
            center, radius, CRYSTAL_RADIAL_POINTS, CRYSTAL_POLAR_POINTS,
            CRYSTAL_AZIMUTHAL_POINTS, panel=local_cutoff(dataset))
        potential = np.repeat(short_range_potential(dataset, self.sigma, r),
                              weights.size // r.size)
        # Only the functions that reach the sphere: the rest are zero on it.
        same = ket is bra
        bra_rows = reaching(bra, self.lattice, center, radius)
        ket_rows = bra_rows if same else reaching(ket, self.lattice, center,
                                                  radius)
        block = np.zeros((len(kpoints), bra_rows.size, ket_rows.size),
                         dtype=complex)
        if bra_rows.size == 0 or ket_rows.size == 0:
            return (bra_rows, ket_rows), block
        near_bra = [bra[mu] for mu in bra_rows]
        near_ket = near_bra if same else [ket[mu] for mu in ket_rows]
        size = _sphere_chunk_size(len(kpoints),
                                  bra_rows.size + (0 if same
                                                   else ket_rows.size))
        for part, weighted in _sphere_chunks(points, weights * potential,
                                             size=size):
            chi = bloch_values(near_bra, self.lattice, kpoints, part, center,
                               radius, presorted=True)
            chi_ket = chi if same else bloch_values(
                near_ket, self.lattice, kpoints, part, center, radius, presorted=True)
            for i in range(len(kpoints)):
                # conj(chi) w chi^T, with the conjugation left to BLAS (the
                # Fortran views of the C-ordered rows, ``A^H B``): no
                # conjugated copy of the Bloch sums.
                block[i] += zgemm(1.0, (chi[i] * weighted).T, chi_ket[i].T,
                                  trans_a=2)
        return (bra_rows, ket_rows), block

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

    def magnetization(self, matrices, kpoint_data=None) -> np.ndarray:
        r"""``(3, ngrid)``: the smooth magnetization of spinor density
        matrices ``P_k`` (``(2M, 2M)``, alpha block first).

        :math:`\rho_{\sigma\tau} = \sum_k w_k\sum_{\mu\nu}
        P^k_{\mu\sigma,\nu\tau}\chi_\mu\chi_\nu^*`, and
        :math:`m_x = 2\,\mathrm{Re}\,\rho_{\uparrow\downarrow}`,
        :math:`m_y = -2\,\mathrm{Im}\,\rho_{\uparrow\downarrow}`,
        :math:`m_z = \rho_{\uparrow\uparrow} - \rho_{\downarrow\downarrow}`.
        A magnetization is an axial vector, which the scalar averages of
        :class:`GridSymmetry` do not rotate: it is assembled from the whole
        mesh only, and refused on a reduced one.
        """
        if self.symmetry is not None:
            raise NotImplementedError(
                "a magnetization is assembled from the full k-point mesh: "
                "the crystal's operations average scalar fields and do not "
                "rotate an axial vector")
        kpoint_data = self.kpoint_data if kpoint_data is None else kpoint_data
        if len(kpoint_data) != len(matrices):
            raise ValueError(f"{len(matrices)} density matrices for "
                             f"{len(kpoint_data)} k-points")
        M = self.M
        m = np.zeros((3, self.grid.size))
        for data, P in zip(kpoint_data, matrices):
            psi = data.psi
            z = np.real(np.sum(psi * ((P[:M, :M] - P[M:, M:]) @ psi.conj()),
                               axis=0))
            ud = np.sum(psi * (P[:M, M:] @ psi.conj()), axis=0)
            m += data.weight * np.stack([2.0 * np.real(ud),
                                         -2.0 * np.imag(ud), z])
        return m

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

    def spinor_kinetic_energy_density(self, matrices, gradients,
                                      kpoint_data=None):
        r"""``(tau, tau_m)``: the kinetic-energy density matrix of spinor
        density matrices (``(2M, 2M)``, alpha block first),
        :math:`\tau_{\sigma\tau} = \tfrac12\sum_k w_k\sum
        P^k_{\mu\sigma,\nu\tau}\nabla\chi_\mu\cdot\nabla\chi^*_\nu`, as
        its trace and the ``(3, ngrid)`` vector
        :math:`(2\,\mathrm{Re}\,\tau_{\uparrow\downarrow},
        -2\,\mathrm{Im}\,\tau_{\uparrow\downarrow},
        \tau_{\uparrow\uparrow} - \tau_{\downarrow\downarrow})` (the
        magnetization's construction).  The whole mesh only, as
        :meth:`magnetization`."""
        if self.symmetry is not None:
            raise NotImplementedError(
                "a kinetic-energy density matrix is assembled from the full "
                "k-point mesh: its vector part is axial")
        kpoint_data = self.kpoint_data if kpoint_data is None else kpoint_data
        M = self.M
        uu = np.zeros(self.grid.size)
        dd = np.zeros(self.grid.size)
        ud = np.zeros(self.grid.size, dtype=complex)
        for data, P, grads in zip(kpoint_data, matrices, gradients):
            for d in grads:
                uu += 0.5 * data.weight * np.real(np.sum(
                    d * (P[:M, :M] @ d.conj()), axis=0))
                dd += 0.5 * data.weight * np.real(np.sum(
                    d * (P[M:, M:] @ d.conj()), axis=0))
                ud += 0.5 * data.weight * np.sum(d * (P[:M, M:] @ d.conj()),
                                                 axis=0)
        return uu + dd, np.stack([2.0 * np.real(ud), -2.0 * np.imag(ud),
                                  uu - dd])

    def compensation_grid(self) -> dict:
        r"""``{channel: g_hat(G)}`` on the grid's reciprocal set (filtered),
        each frequency the mean over its aliases
        (:func:`~mandacaru.integrals.reciprocal.alias_mean`)."""
        if self._compensation_grid is None:
            transforms = rc.alias_mean(self.grid, self.compensation_transforms)
            self._compensation_grid = dict(zip(self.channels, transforms))
        return self._compensation_grid

    def compensation_transforms(self, vectors) -> np.ndarray:
        r"""``(n_ch, ...)``: every channel's compensation shape,
        :math:`\int e^{-i\mathbf g\cdot\mathbf r}\hat g_{LM}(\mathbf r -
        \mathbf R_A)\,d\mathbf r`, at the vectors ``(3, ...)``.

        Share spherical coordinates, atom phases, harmonics and radial
        transforms within this call.  No vector-dependent cache survives a
        reciprocal block or a displaced geometry.
        """
        from ..basis._angular import spherical_harmonic

        vectors = np.asarray(vectors, dtype=float)
        if not self.channels:
            return np.zeros((0,) + vectors.shape[1:], dtype=complex)
        norm, theta, phi = rc.spherical(vectors)
        phases, harmonics, radial = {}, {}, {}
        out = np.empty((len(self.channels),) + vectors.shape[1:], dtype=complex)
        for row, (atom, L, M) in enumerate(self.channels):
            if atom not in phases:
                phases[atom] = rc.structure_factor(vectors, self.centers[atom])
            if (L, M) not in harmonics:
                harmonics[L, M] = spherical_harmonic(L, M, theta, phi, real=False)
            key = (float(self.datasets[atom].compensation_radius), L)
            if key not in radial:
                radial[key] = self._shape_transform(atom, L, norm)
            out[row] = (4.0 * np.pi * (-1j) ** int(L) * harmonics[L, M]
                        * radial[key] * phases[atom])
        return out

    def remainder_transforms(self, atom, vectors) -> np.ndarray:
        r"""``(n, n, ...)``: what the compensation charges leave of the
        augmentation density of atom ``atom``'s projector pairs,
        :math:`\int e^{-i\mathbf g\cdot\mathbf r}\,[\rho^1_{ab} -
        \tilde\rho^1_{ab} - \hat n_{ab}](\mathbf r - \mathbf R_A)\,
        d\mathbf r` at the vectors ``(3, ...)``, for the pair density
        :math:`\phi_a^*\phi_b` of unit projector coefficients.

        The all-electron minus smooth pair density has the multipole moments
        the compensation charge restores, but not its shape: the difference
        is :math:`\sum_{LM}Y_{LM}\,E^{(L)}_{ij}(r)`, whose transform
        (:func:`~mandacaru.pseudopotentials.multipoles.radial_remainder`)
        :math:`4\pi(-i)^L Y_{LM}(\hat g)E^{(L)}_{ij}(|g|)` times the
        angular coupling, summed over :math:`LM`, is returned.  It is what
        :meth:`compensation_transforms` together with the multipole blocks
        misses; at the wavevectors of a cell's few lowest reciprocal vectors
        it is small, at 1-3 Bohr^-1 a few percent to tens of percent of the
        pair density of a d shell.
        """
        from ..basis._angular import spherical_harmonic

        vectors = np.asarray(vectors, dtype=float)
        shape = vectors.shape[1:]
        plan = self._remainder_plan(atom)
        norm, theta, phi = rc.spherical(vectors)
        scale = np.minimum(norm / REMAINDER_TABLE_STEP,
                           plan["points"] - 1.000001)
        low = np.floor(scale).astype(int)
        frac = scale - low
        phase = rc.structure_factor(vectors, self.centers[atom])
        n = plan["n"]
        out = np.zeros((n, n) + shape, dtype=complex)
        for L, (table, couplings) in plan["levels"].items():
            # (n, n, ...) radial values by linear interpolation
            radial = (table[low] * (1.0 - frac)[..., None, None]
                      + table[low + 1] * frac[..., None, None])
            radial = np.moveaxis(radial, (-2, -1), (0, 1))
            for M, angular in couplings:
                harmonic = (4.0 * np.pi * (-1j) ** L * phase
                            * spherical_harmonic(L, M, theta, phi,
                                                 real=False))
                out += angular[(slice(None), slice(None))
                               + (None,) * len(shape)] * radial * harmonic
        return out

    def _remainder_plan(self, atom) -> dict:
        """The tables :meth:`remainder_transforms` reads, per atom, built
        once: ``levels[L] = (table (points, n, n), [(M, angular (n, n))])``
        with the radial remainders on the projector pairs (in the basis of
        the projectors) and the angular couplings."""
        from .multipoles import gaunt, multipole_range, radial_remainder
        from .paw import DEFAULT_PROJECTOR_BASIS

        cache = self.__dict__.setdefault("_remainder_cache", {})
        if atom in cache:
            return cache[atom]
        own = self.projector_columns(atom)
        projectors = [self.projectors[p] for p in own]
        dataset = self.datasets[atom]
        basis = str(getattr(projectors[0], "projector_basis",
                            DEFAULT_PROJECTOR_BASIS))
        n = len(projectors)
        q = np.arange(int(REMAINDER_TABLE_MAX / REMAINDER_TABLE_STEP) + 2
                      ) * REMAINDER_TABLE_STEP
        levels = {}
        for L in sorted({L for a in projectors for b in projectors
                         for L in multipole_range(a.l, b.l)}):
            g = self._shape_transform(atom, L, q)
            radial = {}
            for l1 in sorted({p.l for p in projectors}):
                for l2 in sorted({p.l for p in projectors}):
                    if L not in multipole_range(l1, l2):
                        continue
                    E = radial_remainder(dataset, l1, l2, L, q, g)
                    if basis == "raw":
                        b1 = np.linalg.inv(np.asarray(
                            dataset.channels[l1].vanderbilt, dtype=float))
                        b2 = np.linalg.inv(np.asarray(
                            dataset.channels[l2].vanderbilt, dtype=float))
                        E = np.einsum("ai,qij,bj->qab", b1, E, b2)
                    radial[(l1, l2)] = E
            table = np.zeros((q.size, n, n))
            for a, pa in enumerate(projectors):
                for b, pb in enumerate(projectors):
                    if (pa.l, pb.l) in radial:
                        table[:, a, b] = radial[(pa.l, pb.l)][
                            :, pa.index, pb.index]
            couplings = []
            for M in range(-L, L + 1):
                angular = np.zeros((n, n), dtype=complex)
                for a, pa in enumerate(projectors):
                    for b, pb in enumerate(projectors):
                        if L in multipole_range(pa.l, pb.l):
                            angular[a, b] = gaunt(L, M, pa.l, pa.m, pb.l,
                                                  pb.m, real=pa.real)
                if np.any(angular != 0):
                    couplings.append((M, angular))
            levels[L] = (table, couplings)
        cache[atom] = {"n": n, "points": q.size, "levels": levels}
        return cache[atom]

    def onsite_remainder(self, atom) -> np.ndarray:
        r"""``(n, n, n, n)`` (Hartree): :math:`\Delta W_{abcd} =
        (\varphi_a^*\varphi_b|\varphi_c^*\varphi_d) -
        (\tilde\varphi_a^*\tilde\varphi_b + \hat n_{ab}|
        \tilde\varphi_c^*\tilde\varphi_d + \hat n_{cd})` over atom
        ``atom``'s projectors: the Coulomb energy inside its augmentation
        sphere that the smooth pair densities with their compensation
        charges do not carry
        (:func:`~mandacaru.pseudopotentials.onecenter.one_center_coulomb`)."""
        from .onecenter import one_center_coulomb
        from .paw import DEFAULT_PROJECTOR_BASIS

        cache = self.__dict__.setdefault("_onsite_cache", {})
        if atom not in cache:
            projectors = [self.projectors[p]
                          for p in self.projector_columns(atom)]
            basis = str(getattr(projectors[0], "projector_basis",
                                DEFAULT_PROJECTOR_BASIS))
            cache[atom] = one_center_coulomb(self.datasets[atom], projectors,
                                             basis)
        return cache[atom]

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
        r"""The Gaussian ions on the grid's reciprocal set (electron sign),
        each frequency the mean over its aliases."""
        return rc.alias_mean(self.grid, self._ion_transform)

    @profile_stage('integration: crystal multipole coupling')
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
            norm = np.sqrt(np.sum(G * G, axis=0))
            kernel = 4.0 * np.pi / norm ** 2
            F = self.compensation_transforms(G)
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

    def ion_constants_gradient(self) -> np.ndarray:
        r"""``(n_atoms, 3)``: :math:`\partial E/\partial\mathbf R_A` of
        :meth:`ion_constants`, analytic.  Each pair term
        :math:`\tfrac12 Z_AZ_B f(d)`, :math:`f = \mathrm{erfc}(d/2\sigma)/d`,
        moves :math:`A` by :math:`-f'\hat d` and :math:`B` by :math:`+f'\hat d`
        (:math:`\mathbf d = \mathbf R_B + \mathbf R - \mathbf R_A`); the
        pairs inside one atom's images cancel."""
        sigma = self.sigma
        reach = ION_PAIR_REACH * sigma
        out = np.zeros((len(self.centers), 3))
        for a, (Za, Ra) in enumerate(zip(self.charges, self.centers)):
            for b, (Zb, Rb) in enumerate(zip(self.charges, self.centers)):
                if a == b:
                    continue
                d0 = np.asarray(Rb, float) - np.asarray(Ra, float)
                vectors = d0[None, :] + rc.lattice_translations(
                    self.lattice, reach + np.linalg.norm(d0))
                d = np.linalg.norm(vectors, axis=1)
                keep = (d >= 1e-10) & (d <= reach)
                vectors, d = vectors[keep], d[keep]
                slope = -(erfc(d / (2.0 * sigma)) / d ** 2
                          + np.exp(-(d / (2.0 * sigma)) ** 2)
                          / (np.sqrt(np.pi) * sigma * d))
                pull = 0.5 * Za * Zb * np.sum(
                    (slope / d)[:, None] * vectors, axis=0)
                out[a] -= pull
                out[b] += pull
        return out

    def short_range_ion_compensation_gradient(self) -> dict:
        r"""``{channel: (n_atoms, 3)}``: the derivative of
        :meth:`short_range_ion_compensation` with respect to every atom
        (complex, like the sums), analytic.  A shape moves with its owner
        (the quadrature rigidly) and sees every other atom's image through
        :math:`v^{sr\prime}`: the owner gets :math:`+\sum w g\,v'\hat x`,
        the atom it looks at the opposite; an atom's own images cancel."""
        from ..basis._angular import spherical_harmonic
        from .local_split import (short_range_potential_derivative,
                                  short_range_radius)
        from .multipoles import shape_function
        from .paw import (COMPENSATION_RADIAL_POINTS,
                          PROJECTION_AZIMUTHAL_POINTS, PROJECTION_POLAR_POINTS)

        centers = self.centers
        out = {}
        by_owner = {}
        for owner, L, M in self.channels:
            by_owner.setdefault(owner, []).append((L, M))
        for owner, lms in by_owner.items():
            r_g = float(self.datasets[owner].compensation_radius)
            points, weights, r, dirs = _sphere_rule(
                centers[owner], r_g, COMPENSATION_RADIAL_POINTS,
                PROJECTION_POLAR_POINTS, PROJECTION_AZIMUTHAL_POINTS)
            theta = np.arccos(np.clip(dirs[2], -1.0, 1.0))
            phi = np.arctan2(dirs[1], dirs[0])
            shapes = {}
            weighted_all = []
            for L, M in lms:
                if L not in shapes:
                    shapes[L] = shape_function(r, r_g, L)[:, None]
                density = (shapes[L]
                           * spherical_harmonic(L, M, theta, phi,
                                                real=False)[None, :]).ravel()
                weighted_all.append(weights * density)
            weighted_mat = np.stack(weighted_all, axis=0)
            x = np.stack(points, axis=1)
            grads = np.zeros((len(lms), len(centers), 3), dtype=complex)
            for other, dataset in enumerate(self.datasets):
                if other == owner:
                    continue
                reach = short_range_radius(dataset, self.sigma) + r_g
                d0 = centers[owner] - centers[other]
                for R in rc.lattice_translations(self.lattice,
                                                 reach + np.linalg.norm(d0)):
                    if np.linalg.norm(d0 - R) > reach:
                        continue
                    vector = x - (centers[other] + R)[None, :]
                    distance = np.linalg.norm(vector, axis=1)
                    slope = short_range_potential_derivative(
                        dataset, self.sigma, distance)
                    scale = (slope / np.maximum(distance, 1e-300))[:, None] * vector
                    pull = weighted_mat @ scale
                    grads[:, owner] += pull
                    grads[:, other] -= pull
            for idx, (L, M) in enumerate(lms):
                out[(owner, L, M)] = grads[idx]
        return out

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
        by_owner = {}
        for owner, L, M in self.channels:
            by_owner.setdefault(owner, []).append((L, M))
        for owner, lms in by_owner.items():
            r_g = float(self.datasets[owner].compensation_radius)
            points, weights, r, dirs = _sphere_rule(
                centers[owner], r_g, COMPENSATION_RADIAL_POINTS,
                PROJECTION_POLAR_POINTS, PROJECTION_AZIMUTHAL_POINTS)
            theta = np.arccos(np.clip(dirs[2], -1.0, 1.0))
            phi = np.arctan2(dirs[1], dirs[0])
            shapes = {}
            weighted_all = []
            for L, M in lms:
                if L not in shapes:
                    shapes[L] = shape_function(r, r_g, L)[:, None]
                density = (shapes[L]
                           * spherical_harmonic(L, M, theta, phi,
                                                real=False)[None, :]).ravel()
                weighted_all.append(weights * density)
            weighted_mat = np.stack(weighted_all, axis=0)
            x = np.stack(points, axis=1)
            pot_sum = np.zeros(x.shape[0], dtype=float)
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
                    pot_sum += short_range_potential(dataset, self.sigma,
                                                     distance)
            totals = weighted_mat @ pot_sum
            for idx, (L, M) in enumerate(lms):
                out[(owner, L, M)] = complex(totals[idx])
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
        """Periodic smooth-core kinetic density evaluated and placed in Rust."""
        from ..integrals.exchange_correlation import xc_core_density

        return self._place_cores(xc_core_density, kinetic_density=True)

    def _native_radial_fields(self, tables, *, kinetic_density=False):
        from .. import _native

        if not tables:
            return None
        context = getattr(self, "_native_context", None)
        if context is None:
            context = getattr(self, "_field_context", None)
            if context is None:
                context = _native.Context(threads=1)
                self._field_context = context
        grid = self.__dict__.get("_field_grid")
        if grid is None:
            grid = _native.SpatialGrid(
                self.grid.shape, rc.grid_origin(self.grid),
                np.ascontiguousarray(self.grid.step), periodic=True,
            )
            self._field_grid = grid
        return context.radial_fields(grid, tables, kinetic_density=kinetic_density)

    def _place_cores(self, table, *, kinetic_density=False) -> np.ndarray | None:
        """Transfer radial species tables and atom centers for native placement."""
        groups = {}
        for atom, dataset in enumerate(self.datasets):
            key = id(dataset)
            if key not in groups:
                values = table(dataset)
                groups[key] = None if values is None else [
                    np.ascontiguousarray(dataset.r, dtype=float),
                    np.ascontiguousarray(values, dtype=float), [],
                ]
            if groups[key] is not None:
                groups[key][2].append(self.centers[atom])
        tables = []
        for entry in groups.values():
            if entry is not None:
                r, values, centers = entry
                tables.append((r, values, np.ascontiguousarray(centers, dtype=float)))
        return self._native_radial_fields(tables, kinetic_density=kinetic_density)

    def place_core(self, atom, table, center=None) -> np.ndarray | None:
        """One atom's radial field, placed with the native periodic kernel."""
        dataset = self.datasets[atom]
        values = table(dataset)
        if values is None:
            return None
        center = self.centers[atom] if center is None else center
        return self._native_radial_fields([(
            np.ascontiguousarray(dataset.r, dtype=float),
            np.ascontiguousarray(values, dtype=float),
            np.ascontiguousarray([center], dtype=float),
        )])

    def _superposed_valence(self, dataset, center) -> np.ndarray:
        """One atom's smooth valence density summed over its images on the
        cell grid (flat)."""
        from scipy.interpolate import CubicSpline
        gx, gy, gz = self._grid_points()
        out = np.zeros(self.grid.size)
        r = np.asarray(dataset.r, dtype=float)
        values = np.asarray(dataset.valence_density, dtype=float)
        peak = float(np.max(np.abs(values)))
        significant = np.nonzero(np.abs(values) > 1e-12 * peak)[0]
        support = float(r[significant[-1]])
        support2 = support * support
        spline = CubicSpline(r, values)
        cell_center, cell_radius = self._cell_region()
        reach = support + cell_radius
        for R in rc.lattice_translations(self.lattice,
                                         support + 2.0 * cell_radius):
            pos = center + R
            if np.linalg.norm(pos - cell_center) > reach:
                continue
            cand = np.flatnonzero(np.abs(gx - pos[0]) <= support)
            if not cand.size:
                continue
            cand = cand[np.abs(gy[cand] - pos[1]) <= support]
            if not cand.size:
                continue
            cand = cand[np.abs(gz[cand] - pos[2]) <= support]
            if not cand.size:
                continue
            dx = gx[cand] - pos[0]
            dy = gy[cand] - pos[1]
            dz = gz[cand] - pos[2]
            r2 = dx * dx + dy * dy + dz * dz
            in_ball = r2 <= support2
            if np.any(in_ball):
                idx = cand[in_ball]
                out[idx] += spline(np.sqrt(r2[in_ball]))
        return out

    def initial_density(self) -> tuple[np.ndarray, dict]:
        """Superposed smooth valence densities and reference monopoles."""
        rho = np.zeros(self.grid.size)
        for dataset, center in zip(self.datasets, self.centers):
            rho += self._superposed_valence(dataset, center)
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
        m = np.zeros(self.grid.size)
        moments = np.zeros(len(self.datasets)) if moments is None else \
            np.asarray(moments, dtype=float)
        for dataset, center, moment in zip(self.datasets, self.centers,
                                           moments):
            charge = float(dataset.valence_charge)
            if moment == 0.0 or charge <= 0.0:
                continue
            fraction = float(np.clip(moment / charge, -1.0, 1.0))
            m += fraction * self._superposed_valence(dataset, center)
        return m

    def valence_shares(self) -> np.ndarray:
        r"""``(N, ngrid)``: each atom's share of the superposed valence
        density, :math:`w_A = n_A/\sum_B n_B` (zero where the sum is), the
        Hirshfeld weights of the free atoms' smooth valence densities."""
        pieces = np.stack([self._superposed_valence(d, c)
                           for d, c in zip(self.datasets, self.centers)])
        total = np.sum(pieces, axis=0)
        return np.divide(pieces, total, out=np.zeros_like(pieces),
                         where=total > 0.0)

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

def reduce_mesh(mesh, operations=None, time_reversal: bool = True):
    """``(points, weights, operations)`` of a fractional k-mesh's wedge.

    ``operations`` (a :class:`GridSymmetry`) is first restricted to the
    subgroup that maps ``mesh`` onto itself; with none left, or none given,
    the mesh is reduced by time reversal alone and ``operations`` comes back
    ``None``.  ``weights`` sum to one.  ``time_reversal=False`` (a magnetic
    two-component crystal, whose bands are not even in k) keeps the whole
    mesh, and takes no operations.
    """
    if not time_reversal:
        if operations is not None:
            raise ValueError("a mesh without time reversal is not reduced "
                             "by the crystal's operations either")
        mesh = np.asarray(mesh, dtype=float)
        return mesh, np.full(len(mesh), 1.0 / len(mesh)), None
    if operations is not None:
        operations = operations.keeping_mesh(mesh)
    if operations is None:
        reduced, weights = time_reversal_reduce(mesh)
        return reduced, weights, None
    from ..core.symmetry import irreducible_kpoints
    zone = irreducible_kpoints(mesh, operations.info, time_reversal=True)
    return zone.points, zone.weights / zone.weights.sum(), operations


def mesh_wedge(mesh, size, operations=None, time_reversal: bool = True):
    """``(points, mapping, operations)`` to unfold a quantity from the wedge.

    :func:`reduce_mesh` with the map back: ``mapping[j]`` is the index into
    ``points`` of the representative of ``mesh[j]``, so ``wedge[mapping]``
    is the quantity on the whole mesh -- exact for anything every operation
    and time reversal leave unchanged (a band energy, a Fermi surface).
    ``mesh`` is uniform, ``size`` ``(n1, n2, n3)``: Gamma-centered or a
    Monkhorst-Pack mesh, whose coordinates are multiples of ``1/(2 n)``.
    ``time_reversal=False`` returns the whole mesh and the identity map.
    """
    from ..core.symmetry import irreducible_kpoints

    mesh = np.asarray(mesh, dtype=float)
    if not time_reversal:
        if operations is not None:
            raise ValueError("a mesh without time reversal is not reduced "
                             "by the crystal's operations either")
        return mesh, np.arange(len(mesh)), None
    if operations is not None:
        operations = operations.keeping_mesh(mesh)
    if operations is not None:
        zone = irreducible_kpoints(mesh, operations.info, time_reversal=True)
        return zone.points, zone.mapping, operations
    # Time reversal alone: on the integer lattice 2 n k, -k is a mesh point.
    double = 2 * np.asarray(size, dtype=int)
    key = np.rint(mesh * double).astype(int) % double
    flat = np.ravel_multi_index(key.T, tuple(double))
    where = np.full(int(np.prod(double)), -1)
    where[flat] = np.arange(len(mesh))
    partner = where[np.ravel_multi_index(((-key) % double).T, tuple(double))]
    if np.any(partner < 0):
        raise ValueError("the k-point mesh is not symmetric under k -> -k")
    keep = np.flatnonzero(np.arange(len(mesh)) <= partner)
    slot = np.empty(len(mesh), dtype=int)
    slot[keep] = np.arange(len(keep))
    slot[partner[keep]] = slot[keep]
    return mesh[keep], slot, None


def spin_orbit_symbols(atoms, options: dict | None = None,
                       family: str = "paw-lcao") -> list:
    """The chemical symbols of ``atoms`` whose datasets (of ``options``'
    ``paw_dataset``) carry a spin-orbit term, sorted; empty on a scalar set.
    The datasets are cached, so asking costs one load per element."""
    from .families import resolve_family
    from .paw import get_paw, get_upaw

    options = resolve_family(family).resolved_options(dict(options or {}))
    load = get_upaw if family == "upaw-lcao" else get_paw
    return sorted(s for s in set(atoms.get_chemical_symbols())
                  if getattr(load(s, options.get("paw_dataset")),
                             "has_spin_orbit", False))


@single_threaded_blas
def build_crystal(atoms, grid_spacing: float, options: dict | None = None, kpts=None,
                  family: str = "paw-lcao", grid=None, ghosts=(),
                  symmetry: bool = True, full_mesh: bool = False,
                  spin_orbit: bool = False, filter_pin=None, native_context=None,
                  precision="double"):
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
    whole mesh and breaks the symmetry the reduction relies on, and so does
    a magnetic two-component crystal (time reversal flips its
    magnetization).

    ``spin_orbit`` says the caller solves two-component (spinor) Bloch
    states: the spin-orbit term of a spin-orbit set (``lda-fr/``,
    ``pbe-fr/``) is attached (:meth:`PeriodicPAW.spin_orbit_matrix`).
    Without it such a set is refused -- dropping the term would report a
    scalar-relativistic crystal under the spin-orbit set's name.

    ``filter_pin`` (a :class:`~mandacaru.basis.filtering.FilterPin`) is the
    calculator's holder of the ``filter_cutoff="auto"`` cutoff: the first
    build fills it, later builds of other cells reuse it, and
    ``context["filter_held"]`` says which happened.  Without one, ``"auto"``
    is resolved from this grid alone.
    """
    from ..models.setup import monkhorst_pack_kpts
    from ..basis.filtering import resolve_filter_cutoff
    from ..integrals import Grid
    from ..units import to_bohr
    from .families import pseudo_basis_arguments, resolve_family
    from .orbitals import pseudo_basis, valence_electrons
    from .paw import (DEFAULT_PROJECTOR_BASIS, get_paw, get_upaw,
                      paw_coupling_blocks, paw_multipole_blocks,
                      paw_overlap_blocks, paw_projectors,
                      paw_spin_orbit_blocks, paw_spin_orbit_projectors)

    # Sampling uses the packaged Rust kernels; calculation startup never compiles code.
    from ..integrals import _backend
    _backend.warn_fallback(_backend.ensure_backend())

    options = resolve_family(family).resolved_options(dict(options or {}))
    from .diffuse import refuse_crystal
    refuse_crystal(options.get("diffuse"))
    load = get_upaw if family == "upaw-lcao" else get_paw
    symbols = atoms.get_chemical_symbols()
    atoms = atoms.copy()
    atoms.wrap()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    potentials = {s: load(s, options.get("paw_dataset")) for s in set(symbols)}
    with_spin_orbit = sorted(s for s, p in potentials.items()
                             if getattr(p, "has_spin_orbit", False))
    if with_spin_orbit and not spin_orbit:
        # Only the two-component Kohn-Sham solver has spinor Bloch states;
        # dropping the term would report a scalar-relativistic crystal under
        # a spin-orbit dataset's name.
        raise NotImplementedError(
            f"the {', '.join(with_spin_orbit)} dataset(s) carry a spin-orbit "
            "term, which only two-component (spinor) Bloch states can hold: "
            "method='dft' (LDA or PBE) solves a crystal on them; this "
            "consumer of the crystal has no spinor form.  Use a "
            "scalar-relativistic set (paw_dataset='lda-sr' or 'pbe-sr') "
            "for it")
    if grid is None:
        g = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, grid_spacing=grid_spacing,
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
    # ``filter_pin`` holds the "auto" cutoff of the first build across the
    # cells of one calculator (see :class:`~mandacaru.basis.filtering.FilterPin`).
    spacing = max(g.dx, g.dy, g.dz)
    k_c, held = (filter_pin.resolve(options.get("filter_cutoff"), spacing,
                                    symbols)
                 if filter_pin is not None else
                 (resolve_filter_cutoff(options.get("filter_cutoff"), spacing),
                  False))
    confinement: dict = {}
    polarization: dict = {}
    # Ghosts lend their basis functions; everything physical is the real
    # atoms'.
    from ..models.setup import validate_ghosts
    ghosts = validate_ghosts(ghosts, len(symbols))
    real = [i for i in range(len(symbols)) if i not in ghosts]
    real_symbols = [symbols[i] for i in real]
    # A crystal's Bloch sums, its C kernel and its Wannier code index the
    # harmonics by m: complex ones, where a molecule uses real ones.
    basis, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, filter_wavevector=k_c,
        **pseudo_basis_arguments(family, options, confinement=confinement,
                                 polarization=polarization), real=False)
    projectors = paw_projectors(
        real_symbols, positions[real], potentials,
        projector_basis=options.get("projector_basis",
                                    DEFAULT_PROJECTOR_BASIS), real=False)
    datasets = [potentials[s] for s in real_symbols]
    size, gamma, mesh = monkhorst_pack_kpts(kpts)
    if full_mesh:
        reduced = np.asarray(mesh, dtype=float)
        weights = np.full(len(reduced), 1.0 / len(reduced))
        operations = None
    else:
        # The compensation multipoles reach L = 2 l_max: 6 for a dataset
        # with an f channel (W-Hg, the lanthanides), whose crystal had no
        # rotation table for L = 5 and stopped in the first SCF step.
        max_L = max(4, 2 * max(max(d.channels) for d in datasets))
        operations = grid_symmetry(atoms, g, max_L=max_L) \
            if symmetry and not ghosts else None
        reduced, weights, operations = reduce_mesh(mesh, operations)
    B = rc.reciprocal_vectors(rc.lattice_vectors(g))
    kpoints = reduced @ B.T
    so_projectors, so_blocks = None, None
    if with_spin_orbit:
        # Complex harmonics, as every crystal function (the L.S matrix
        # follows the projectors' own harmonics).
        so_projectors = paw_spin_orbit_projectors(
            real_symbols, positions[real], potentials, real=False)
        so_blocks = paw_spin_orbit_blocks(so_projectors, real_symbols,
                                          potentials)
    crystal = PeriodicPAW(
        basis, atom_of_orbital, projectors, datasets,
        [to_bohr(p, "angstrom") for p in positions[real]], g, kpoints, weights,
        paw_coupling_blocks(projectors, real_symbols, potentials),
        paw_overlap_blocks(projectors, real_symbols, potentials),
        paw_multipole_blocks(projectors, datasets), filter_wavevector=k_c,
        symmetry=operations, spin_orbit_projectors=so_projectors,
        spin_orbit_blocks=so_blocks, native_context=native_context, precision=precision)
    context = {"crystal": crystal, "atom_of_orbital": atom_of_orbital,
               "n_electrons": float(valence_electrons(real_symbols,
                                                      potentials)),
               "ghosts": tuple(sorted(ghosts)),
               "pseudopotentials": potentials, "family": family,
               "filter_wavevector": k_c, "filter_held": held,
               "options": dict(options),
               "confinement": confinement, "polarization": polarization,
               "kpts_size": size, "kpts_gamma": gamma,
               "kpoints_fractional": reduced, "full_mesh": bool(full_mesh)}
    return crystal, context
