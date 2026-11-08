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

import weakref
from dataclasses import dataclass

import numpy as np

from ..integrals import reciprocal as rc
from ..integrals._backend import single_threaded_blas
from .native_context import context as _default_native_ctx

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
# Keep the shared backing allocation bounded so transferred k-point arrays can
# be released in groups instead of retaining an entire mesh until handoff ends.
KPOINT_BLOCK_MAX = 4

#: Lattice-sum reach (in Gaussian widths) of the short-range ion pair term.
ION_PAIR_REACH = 12.0


# --------------------------------------------------------------------------- #
# Bloch sums.
# --------------------------------------------------------------------------- #

#: A tabulated radial function counts as zero beyond the last point where it
#: exceeds this fraction of its peak.
SUPPORT_TOLERANCE = 1e-12

#: ``id(values) -> (weak reference to values, support radius)``.  The entry holds the
#: radial array weakly and leaves with it: a filtered table is rebuilt for every geometry, and a
#: strong reference here kept every one of them alive across a scan.
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
        spline = _radial_spline(self.parent)
        if spline is None:
            raise NotImplementedError("BasisGradient requires a tabulated orbital")
        ctx = _default_native_ctx()
        breaks, coeffs, rc_val = spline
        x_arr = np.ascontiguousarray(x, dtype=float)
        y_arr = np.ascontiguousarray(y, dtype=float)
        z_arr = np.ascontiguousarray(z, dtype=float)
        out = ctx.periodic_basis_gradient(
            x_arr.ravel(),
            y_arr.ravel(),
            z_arr.ravel(),
            (float(self.center[0]), float(self.center[1]), float(self.center[2])),
            breaks,
            coeffs,
            float(rc_val),
            int(self.l),
            int(self.m),
            int(self.axis),
        )
        return out.reshape(x_arr.shape)


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
    """Radius (Bohr) beyond which an atom-centered function vanishes."""
    import math
    table = _radial_table(function)
    if table is not None:
        r, values = table
        held = _SUPPORTS.get(id(values))
        if held is None or held[0]() is not values:
            context = _default_native_ctx()
            edge = context.periodic_radial_support(r, values, SUPPORT_TOLERANCE)
            held = _SUPPORTS[id(values)] = (weakref.ref(
                values, lambda _, key=id(values): _SUPPORTS.pop(key, None)), edge)
        return held[1]
    for name in ("r_c", "r_cut"):
        value = getattr(function, name, None)
        if value is not None and math.isfinite(value):
            return float(value)
    raise ValueError(f"{type(function).__name__} has no finite support radius; "
                     "a Bloch sum needs one")


def _shell_key(function):
    """Functions that share a radial table and a center differ only in ``m``
    (and gradients also by their axis)."""
    context = _default_native_ctx()
    center = context.periodic_round_coords(tuple(function.center))
    if isinstance(function, BasisGradient):
        key = _shell_key(function.parent)
        return None if key is None else (key[0], center, "d", function.axis)
    table = _radial_table(function)
    if table is None or not hasattr(function, "radial"):
        return None
    return (id(table[1]), center)


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
#: tens of thousands of points times the 50 k-points of a band path).  A chunk
#: of 2048 holds an eighth of the sums one of 16384 did; the chunks add up to
#: the same quadrature.
SPHERE_CHUNK = 2048
#: Bytes of Bloch sums one sphere chunk may hold (:func:`_sphere_chunk_size`).
SPHERE_BYTES = 1 << 29


def _sphere_chunk_size(n_kpoints: int, n_functions: int) -> int:
    """Points per chunk of a sphere quadrature: :data:`SPHERE_CHUNK`."""
    context = _default_native_ctx()
    return context.periodic_sphere_chunk_size(
        int(n_kpoints), int(n_functions), SPHERE_BYTES, SPHERE_CHUNK,
    )


def _sphere_chunks(points, *arrays, size: int = SPHERE_CHUNK):
    """Yield ``(points, *arrays)`` chunks of ``size`` points of a quadrature,
    its points sorted into 1-Bohr bins (a sum over the points does not depend
    on their order, and :func:`bloch_values` then has nothing to un-permute)."""
    context = _default_native_ctx()
    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
    order = context.periodic_spatial_bins_order(x, y, z)
    x, y, z = x[order], y[order], z[order]
    arrays = [np.asarray(a).ravel()[order] for a in arrays]
    for start in range(0, x.size, size):
        part = slice(start, start + size)
        yield ((x[part], y[part], z[part]), *[a[part] for a in arrays])


def reaching(functions, lattice, region_center, region_radius) -> np.ndarray:
    """Indices of the ``functions`` with a lattice image whose support
    reaches the ball of ``region_radius`` about ``region_center``."""
    if not functions:
        return np.zeros(0, dtype=int)
    context = getattr(functions[0], "_native_context", None) or _default_native_ctx()
    centers = [list(f.center) for f in functions]
    supports = [_support(f) for f in functions]
    lat = lattice.tolist() if hasattr(lattice, "tolist") else list(lattice)
    rc_pt = region_center.tolist() if hasattr(region_center, "tolist") else list(region_center)
    indices = context.periodic_reaching(lat, centers, supports, rc_pt, float(region_radius))
    return np.asarray(indices, dtype=int)


@profile_stage('integration: crystal basis sampling')
def bloch_values(functions, lattice, kpoints, points, region_center,
                 region_radius, *, presorted=False, context=None) -> np.ndarray:
    r"""``(nk, M, npts)`` Bloch sums of ``functions`` at ``points``.

    ``points`` is an ``(x, y, z)`` triple of flat arrays (Bohr) lying within
    ``region_radius`` of ``region_center``; only the images that reach that
    region are evaluated, each once for every k-point.  Functions sharing a
    radial table and a center -- the ``m`` components of one shell -- share
    the geometry and the radial interpolation of every image.
    """
    if context is None:
        context = _default_native_ctx()

    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
    # The points are visited in 1-Bohr bins: the C kernel culls lattice
    # images per block of consecutive points, which a quadrature sphere's
    # radial-major order would spread over the whole sphere.
    if not presorted:
        order = context.periodic_spatial_bins_order(x, y, z)
        x, y, z = x[order], y[order], z[order]
    else:
        order = None
    kpoints_arr = np.ascontiguousarray(np.atleast_2d(np.asarray(kpoints, dtype=float)), dtype=float)
    lattice_arr = np.ascontiguousarray(lattice, dtype=float)
    out = np.zeros((len(kpoints_arr), len(functions), x.size), dtype=complex)
    reg_center = (float(region_center[0]), float(region_center[1]), float(region_center[2]))
    x_arr = np.ascontiguousarray(x, dtype=float)
    y_arr = np.ascontiguousarray(y, dtype=float)
    z_arr = np.ascontiguousarray(z, dtype=float)
    groups: dict = {}
    for mu, function in enumerate(functions):
        key = _shell_key(function)
        groups.setdefault(key if key is not None else ("single", mu),
                          []).append(mu)
    for members in groups.values():
        first = functions[members[0]]
        center = np.asarray(first.center, dtype=float)
        center_arr = (float(center[0]), float(center[1]), float(center[2]))
        support = float(_support(first))
        gradient = isinstance(first, BasisGradient)
        shared = _shell_key(first) is not None and not gradient
        spline = _radial_spline(first) if (shared or gradient) else None
        if spline is None:
            raise NotImplementedError(
                f"{type(first).__name__} without radial spline is not supported in periodic Bloch sums"
            )
        breaks, coeffs, rc_val = spline
        ms = [int(functions[mu].m) for mu in members]
        rows = [int(mu) for mu in members]
        breaks_arr = np.ascontiguousarray(breaks, dtype=float)
        coeffs_arr = np.ascontiguousarray(coeffs, dtype=float).ravel()
        context.periodic_bloch_shell(
            (x_arr, y_arr, z_arr), center_arr, lattice_arr, kpoints_arr,
            reg_center, float(region_radius), support,
            breaks_arr, coeffs_arr, float(rc_val), int(first.l),
            ms, rows, out,
            int(first.axis) if gradient else -1,
        )
    if order is not None:
        context.periodic_unpermute_rows(out, order)
    return out


@profile_stage('integration: crystal basis projection')
def bloch_projections(functions, lattice, kpoints, points, columns,
                      region_center, region_radius, *, context=None) -> np.ndarray:
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
    for members in groups.values():
        first = functions[members[0]]
        center = np.asarray(first.center, dtype=float)
        support = float(_support(first))
        gradient = isinstance(first, BasisGradient)
        shared = _shell_key(first) is not None and not gradient
        spline = _radial_spline(first) if (shared or gradient) else None
        if spline is None:
            raise NotImplementedError(
                f"{type(first).__name__} without radial spline is not supported in periodic Bloch projections"
            )
        if context is None:
            context = _default_native_ctx()
        breaks, coeffs, rc_val = spline
        lattice_arr = np.ascontiguousarray(lattice, dtype=float)
        kpoints_arr = np.ascontiguousarray(kpoints, dtype=float)
        center_arr = (float(center[0]), float(center[1]), float(center[2]))
        reg_center = (float(region_center[0]), float(region_center[1]), float(region_center[2]))
        ms = [int(functions[mu].m) for mu in members]
        rows = [int(mu) for mu in members]
        breaks_arr = np.ascontiguousarray(breaks, dtype=float)
        coeffs_arr = np.ascontiguousarray(coeffs, dtype=float).ravel()
        cols_arr = np.ascontiguousarray(columns, dtype=complex)
        x_arr = np.ascontiguousarray(x, dtype=float)
        y_arr = np.ascontiguousarray(y, dtype=float)
        z_arr = np.ascontiguousarray(z, dtype=float)
        context.periodic_bloch_project(
            (x_arr, y_arr, z_arr), center_arr, lattice_arr, kpoints_arr,
            reg_center, float(region_radius), support,
            breaks_arr, coeffs_arr, float(rc_val), int(first.l),
            ms, rows, cols_arr, out,
            int(first.axis) if gradient else -1,
        )
    return out


def time_reversal_reduce(kpoints_frac, tolerance: float = 1e-8):
    r"""``(representatives, weights)`` of a mesh under :math:`\mathbf k \to -\mathbf k`."""
    context = _default_native_ctx()
    k_arr = np.ascontiguousarray(kpoints_frac, dtype=float)
    return context.periodic_time_reversal_reduce(k_arr, float(tolerance))


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
        """The operations that also map the k-point ``mesh`` onto itself."""
        from dataclasses import replace as dc_replace
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        mesh_arr = np.ascontiguousarray(mesh, dtype=float)
        rotations = [np.asarray(W, dtype=float).tolist() for W in self.info.rotations]
        keep = context.periodic_filter_mesh_rotations(rotations, mesh_arr, 1e-8)
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
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        val_arr = np.ascontiguousarray(values, dtype=float).ravel()
        return context.periodic_symmetrize_field(self.permutations, val_arr)

    def moments(self, q: dict) -> dict:
        """The average of the multipoles ``{(atom, L, M): q}``."""
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        keys = list(q.keys())
        vals = np.ascontiguousarray([q[k] for k in keys], dtype=complex)
        out_vals = context.periodic_symmetrize_moments(self.atom_maps, self.rotations, keys, vals)
        return dict(zip(keys, out_vals))


def _harmonic_rotation(R, L: int) -> np.ndarray:
    """``T`` with ``Y_L(R u) = Y_L(u) T`` for the complex ``Y_LM``."""
    context = _default_native_ctx()
    mat = [[float(R[i][j]) for j in range(3)] for i in range(3)]
    return context.periodic_harmonic_rotation(mat, int(L))


def grid_symmetry(atoms, grid, max_L: int = 4, symprec: float | None = None):
    r"""The :class:`GridSymmetry` of a periodic ``atoms`` on ``grid``, or
    ``None`` when no operation beyond the identity survives."""
    from dataclasses import replace as dc_replace
    from .. import _native
    from ..core.symmetry import DEFAULT_SYMPREC, crystal_symmetry

    context = _default_native_ctx()
    info = crystal_symmetry(atoms, DEFAULT_SYMPREC if symprec is None else symprec)
    lattice = rc.lattice_vectors(grid)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    pairs = [(int(z), f"{float(mag):.6f}") for z, mag in
             zip(atoms.get_atomic_numbers(),
                 atoms.get_initial_magnetic_moments())]
    keys = sorted(set(pairs))
    numbers = [keys.index(p) for p in pairs]
    res = context.periodic_grid_symmetry(
        list(grid.shape), lattice.tolist(), frac.tolist(), numbers,
        [np.asarray(W, dtype=float).tolist() for W in info.rotations],
        [list(t) for t in info.translations], int(max_L),
    )
    if res is None:
        return None
    keep, permutations, atom_maps, rot_matrices = res
    rotations = [{L: rot_matrices[s][L] for L in range(max_L + 1)} for s in range(len(keep))]
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
    from .. import _native
    return _native.paw_sphere_rule(
        tuple(center), float(radius), int(radial), int(polar), int(azimuthal),
        0.0 if panel is None else float(panel),
    )


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
                 native_context=None, precision="float64"):
        from ..models.paw_channels import assemble_block_matrix
        from .local_split import split_width

        self.basis = list(basis)
        self._native_context = native_context
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")
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
        #: The Coulomb kernel and the grid charges (:meth:`compensation_grid`,
        #: :meth:`ion_charge`) are means over the aliases the grid lattice's
        #: operations pick (:func:`~mandacaru.integrals.reciprocal.alias_mean`),
        #: so they commute with the operations as the kinetic matrix does.
        context, field_grid = self._field_backend()
        self.kernel = context.periodic_coulomb_kernel(field_grid).reshape(grid.shape)
        self._compensation_grid = None
        self._compensation_values = None
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
        """Center and bounding radius (Bohr) of the cell the grid spans."""
        context = self._field_backend()[0]
        return context.periodic_cell_region(self.lattice.tolist(), rc.grid_origin(self.grid))

    def _kpoints(self) -> list:
        """:class:`KPointMatrices` of the crystal's own (reduced) k-points."""
        return self.kpoint_matrices(self.kpoints, self.weights)

    def cartesian_kpoints(self, fractional) -> np.ndarray:
        """Fractional k-points (reciprocal-lattice units) in Bohr^-1."""
        context = self._field_backend()[0]
        frac = np.ascontiguousarray(fractional, dtype=float)
        return context.periodic_cartesian_kpoints(self.lattice.tolist(), frac)

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
        context = self._field_backend()[0]
        wide = context.periodic_wide_kpoint_block(
            self.M, len(self.projectors), self.has_spin_orbit, block, KPOINT_BLOCK_BYTES,
        )
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
                                       self._grid_points(), center, radius,
                                       context=getattr(self, "_native_context", None))
                shift = start - outer
                for i, k in enumerate(kpoints):
                    weight = all_weights[start + i]
                    psi_raw, C = psi_all[i], C_wide[shift + i]
                    psi = psi_raw
                    context = self._field_backend()[0]
                    S, fixed = context.periodic_paw_matrices(
                        C, self.q_overlap, self.D_ion, context.periodic_overlap(psi, dV),
                        self._kinetic(psi, k), V_wide[shift + i])
                    out.append(KPointMatrices(
                        k=np.asarray(k, float), weight=float(weight),
                        psi=psi.astype(np.complex64) if self.precision == "float32" else psi,
                        overlap=S, fixed=fixed, projections=C,
                        spin_orbit=SO_wide[shift + i]))
        return out

    @property
    def has_spin_orbit(self) -> bool:
        """Whether the crystal carries a spin-orbit term (a spin-orbit set)."""
        return bool(self.spin_orbit_blocks)

    def spin_orbit_matrix(self, kpoints) -> np.ndarray:
        r"""``(nk, 2M, 2M)``: the spin-orbit term between the spinor Bloch
        sums at the Cartesian ``kpoints``, alpha block first."""
        kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        context = self._field_backend()[0]
        if not self.has_spin_orbit:
            return context.periodic_empty_spin_orbit(len(kpoints), self.M)
        from ..core.spin_orbit import _channels
        C = self._projections(kpoints, projectors=self.spin_orbit_projectors)
        channels = _channels(self.spin_orbit_projectors, self.spin_orbit_blocks)
        return context.periodic_spin_orbit_matrices(C, channels)

    def kpoint_block(self) -> int:
        """How many k-points' Bloch sums fit :data:`KPOINT_BLOCK_BYTES`."""
        context = self._field_backend()[0]
        return context.periodic_kpoint_block(self.grid.size, self.M, KPOINT_BLOCK_MAX, KPOINT_BLOCK_BYTES)

    def _orbital_samples(self, psi):
        samples = np.asarray(psi)
        if samples.dtype.kind in "fc" and samples.dtype.itemsize > {"f": 8, "c": 16}[samples.dtype.kind]:
            raise ValueError("orbital samples above float64 precision are not supported")
        dtype = {np.dtype(np.complex64): np.complex64}.get(samples.dtype, np.complex128)
        return np.ascontiguousarray(samples, dtype=dtype), dtype

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
        context, grid = self._field_backend()
        samples, dtype = self._orbital_samples(psi)
        method = {np.complex64: context.periodic_kinetic_f32, np.complex128: context.periodic_kinetic}[dtype]
        return method(grid, k, samples)

    def bloch_gradients(self, psi, k) -> np.ndarray:
        r""":math:`\nabla\chi` on the grid, ``(3, M, ngrid)``: the periodic
        part multiplied by the alias-averaged :math:`i(\mathbf G + \mathbf
        k)` of :func:`~mandacaru.integrals.reciprocal.bloch_aliases`, so a
        kinetic-energy density summed over the wedge and symmetrized is the
        full mesh's."""
        context, grid = self._field_backend()
        samples, dtype = self._orbital_samples(psi)
        method = {np.complex64: context.periodic_bloch_gradients_f32,
                  np.complex128: context.periodic_bloch_gradients}[dtype]
        return method(grid, k, samples)

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
            key = (projector.atom_index, f"{float(projector.r_cut):.12f}")
            spheres.setdefault(key, []).append(p)
        for members in spheres.values():
            first = projectors[members[0]]
            rows = reaching(self.basis, self.lattice, first.center,
                            float(first.r_cut))
            if rows.size == 0:
                continue
            near = [self.basis[mu] for mu in rows]
            points, weights = _projector_sphere(first)
            native_ctx = getattr(self, "_native_context", None) or _default_native_ctx()
            raw_columns = [np.ascontiguousarray(projectors[p].evaluate(*points).ravel(), dtype=complex)
                           for p in members]
            values = native_ctx.periodic_weighted_columns(raw_columns, np.ascontiguousarray(weights.ravel(), dtype=float))
            block = np.zeros((len(kpoints), rows.size, len(members)),
                             dtype=complex)
            for part, *columns in _sphere_chunks(points, *values):
                block += bloch_projections(
                    near, self.lattice, kpoints, part, np.stack(columns),
                    first.center, float(first.r_cut),
                    context=native_ctx)
            native_ctx.periodic_accumulate_projection_block(C, rows.tolist(), members, block)
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
        native_ctx = getattr(self, "_native_context", None) or _default_native_ctx()
        sr_weights = native_ctx.periodic_short_range_weights(
            weights, short_range_potential(dataset, self.sigma, r)
        )
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
        for part, weighted in _sphere_chunks(points, sr_weights,
                                             size=size):
            chi = bloch_values(near_bra, self.lattice, kpoints, part, center,
                               radius, presorted=True,
                               context=native_ctx)
            chi_ket = chi if same else bloch_values(
                near_ket, self.lattice, kpoints, part, center, radius, presorted=True,
                context=native_ctx)
            native_ctx.periodic_accumulate_zgemm(block, chi, chi_ket, weighted)
            # Unbound before the next chunk is evaluated: two chunks are never alive at once.
            del chi, chi_ket
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
        context = self._field_backend()[0]
        weights = np.asarray([d.weight for d in kpoint_data], dtype=float)
        psi_list = [np.ascontiguousarray(d.psi, dtype=np.complex128) for d in kpoint_data]
        mat_list = [np.ascontiguousarray(P, dtype=np.complex128) for P in matrices]
        c_list = [np.ascontiguousarray(d.projections, dtype=np.complex128) for d in kpoint_data]
        rho, R = context.periodic_density_and_projected(
            self.M, len(self.projectors), weights, mat_list, psi_list, c_list,
        )
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
        context = self._field_backend()[0]
        weights = np.asarray([d.weight for d in kpoint_data], dtype=float)
        psi_list = [np.ascontiguousarray(d.psi, dtype=np.complex128) for d in kpoint_data]
        mat_list = [np.ascontiguousarray(P, dtype=np.complex128) for P in matrices]
        return context.periodic_spinor_magnetization(
            self.M, weights, mat_list, psi_list,
        )

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
        context = self._field_backend()[0]
        channel_atoms = [ch[0] for ch in self.channels]
        weights = np.asarray([complex(w[ch]) for ch in self.channels], dtype=np.complex128)
        blocks = [self.multipole_blocks[ch] for ch in self.channels]
        pos_list = [self._positions.get(a, []) for a in range(len(self.centers))]
        return context.periodic_moment_operator(
            len(self.projectors), pos_list, weights, channel_atoms, blocks,
        )

    def moment_traces(self, R) -> dict:
        r"""``{(A, L, M): q}`` of the projected density matrix
        :math:`R = \sum_k w_k C(k)^\dagger P^k C(k)` (``(P, P)``):
        :math:`q^A_{LM} = \operatorname{tr}(R_{AA}\,\mathrm{blk}^A_{LM})`."""
        context = self._field_backend()[0]
        blocks = [self.multipole_blocks[ch] for ch in self.channels]
        pos_list = [self._positions.get(a, []) for a in range(len(self.centers))]
        r_arr = np.ascontiguousarray(R, dtype=np.complex128)
        q_vals = context.periodic_moment_traces(r_arr, pos_list, self.channels, blocks)
        return dict(zip(self.channels, q_vals))

    def kinetic_energy_density(self, matrices, gradients=None) -> np.ndarray:
        r""":math:`\tau = \tfrac12\sum_k w_k \sum P^k_{\mu\nu}
        \nabla\chi_\mu\cdot\nabla\chi^*_\nu`, symmetrized like the density.

        ``gradients`` (per k-point, :meth:`bloch_gradients`) reuses ones a
        caller already holds -- the SCF evaluates tau every iteration.
        """
        if gradients is None:
            gradients = [self.bloch_gradients(d.psi, d.k)
                         for d in self.kpoint_data]
        context = self._field_backend()[0]
        weights = np.asarray([d.weight for d in self.kpoint_data], dtype=float)
        mat_list = [np.ascontiguousarray(P, dtype=np.complex128) for P in matrices]
        grad_list = [np.ascontiguousarray(g, dtype=np.complex128) for g in gradients]
        tau = context.periodic_kinetic_density(
            self.M, weights, mat_list, grad_list,
        )
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
        context = self._field_backend()[0]
        weights = np.asarray([d.weight for d in kpoint_data], dtype=float)
        mat_list = [np.ascontiguousarray(P, dtype=np.complex128) for P in matrices]
        grad_list = [np.ascontiguousarray(g, dtype=np.complex128) for g in gradients]
        return context.periodic_spinor_kinetic_density(
            self.M, weights, mat_list, grad_list,
        )

    def compensation_grid(self) -> dict:
        r"""``{channel: g_hat(G)}`` on the grid's reciprocal set (filtered),
        each frequency the mean over its aliases
        (:func:`~mandacaru.integrals.reciprocal.alias_mean`)."""
        if self._compensation_grid is None:
            transforms = rc.alias_mean(self.grid, self.compensation_transforms)
            self._compensation_values = transforms
            self._compensation_grid = dict(zip(self.channels, transforms))
        return self._compensation_grid

    def compensation_grid_array(self) -> np.ndarray:
        """Compensation transforms in channel order, without caching intermediate dicts."""
        if self._compensation_values is not None:
            return self._compensation_values
        return rc.alias_mean(self.grid, self.compensation_transforms)

    def compensation_transforms(self, vectors) -> np.ndarray:
        r"""``(n_ch, ...)``: every channel's compensation shape,
        :math:`\int e^{-i\mathbf g\cdot\mathbf r}\hat g_{LM}(\mathbf r -
        \mathbf R_A)\,d\mathbf r`, at the vectors ``(3, ...)``.

        Share spherical coordinates, atom phases, harmonics and radial
        transforms within this call.  No vector-dependent cache survives a
        reciprocal block or a displaced geometry.
        """
        vectors = np.asarray(vectors, dtype=float)
        if not self.channels:
            return np.zeros((0,) + vectors.shape[1:], dtype=complex)
        if "_shape_transform" in self.__dict__:
            from ..basis._angular import spherical_harmonic
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
                out[row] = rc.multipole_transform(vectors, self.centers[atom], radial[key], L, M)
            return out
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        centers = [[float(c[0]), float(c[1]), float(c[2])] for c in self.centers]
        radii = [float(self.datasets[atom].compensation_radius) for atom in range(len(self.centers))]
        channels = [(int(atom), int(L), int(M)) for atom, L, M in self.channels]
        return context.periodic_compensation_transforms(centers, radii, channels, np.ascontiguousarray(vectors))

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
        from .. import _native

        context = _default_native_ctx()
        vectors = np.asarray(vectors, dtype=float)
        shape = vectors.shape[1:]
        flat_vectors = np.ascontiguousarray(vectors.reshape(3, -1).T, dtype=float)
        plan = self._remainder_plan(atom)
        levels = [(int(L), t, c) for L, (t, c) in plan["levels"].items()]
        out = context.periodic_remainder_transforms(
            levels, list(self.centers[atom]), flat_vectors, REMAINDER_TABLE_STEP
        )
        return out.reshape((plan["n"], plan["n"]) + shape)

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
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        q = context.periodic_remainder_q(REMAINDER_TABLE_MAX, REMAINDER_TABLE_STEP)
        proj_info = [(int(p.l), int(p.m), int(p.index), bool(p.real))
                     for p in projectors]
        p_ls = {p.l for p in projectors}
        all_l = [l for l in range(8) if l in p_ls]
        p_Ls = {L for a in projectors for b in projectors for L in multipole_range(a.l, b.l)}
        all_L = [L for L in range(16) if L in p_Ls]
        levels = {}
        for L in all_L:
            g = self._shape_transform(atom, L, q)
            radial = {}
            for l1 in all_l:
                for l2 in all_l:
                    if L not in multipole_range(l1, l2):
                        continue
                    E = radial_remainder(dataset, l1, l2, L, q, g)
                    if basis == "raw":
                        v1 = np.ascontiguousarray(dataset.channels[l1].vanderbilt, dtype=float)
                        v2 = np.ascontiguousarray(dataset.channels[l2].vanderbilt, dtype=float)
                        E = context.periodic_raw_vanderbilt_transform(v1, v2, E)
                    radial[(l1, l2)] = E
            table, couplings = context.periodic_remainder_plan_level(
                int(L), int(q.size), proj_info, radial
            )
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
                            self._grid_points(), center, radius,
                            context=getattr(self, "_native_context", None))

    def projections(self, kpoints) -> np.ndarray:
        r"""``(nk, M, P)``: :math:`\langle\chi_{\mu\mathbf k}|\tilde
        p_p\rangle` at the Cartesian ``kpoints``."""
        return self._projections(np.atleast_2d(np.asarray(kpoints,
                                                          dtype=float)))

    def dense_compensation_gradient(self, shift, kernel) -> np.ndarray:
        r"""``(3, n_ch, n_ch)``: :meth:`dense_compensation`'s sum with each
        term weighted by :math:`i(\mathbf G + \mathbf p)` (no tail)."""
        ctx = self._field_backend()[0]
        return ctx.periodic_dense_compensation_gradient(
            self.lattice.tolist(), [c.tolist() for c in self.centers],
            [float(z) for z in self.charges],
            [float(d.compensation_radius) for d in self.datasets],
            self.channels, float(self.sigma), COMPENSATION_CUTOFF,
            [float(s) for s in shift], kernel,
        )

    def dense_compensation(self, shift, kernel) -> np.ndarray:
        r"""``(n_ch, n_ch)``: :math:`\tfrac1\Omega\sum_{\mathbf G}
        v(|\mathbf G + \mathbf p|)\,\hat g_a^*\hat g_b` over the dense set."""
        ctx = self._field_backend()[0]
        return ctx.periodic_dense_compensation(
            self.lattice.tolist(), [c.tolist() for c in self.centers],
            [float(z) for z in self.charges],
            [float(d.compensation_radius) for d in self.datasets],
            self.channels, float(self.sigma), COMPENSATION_CUTOFF,
            [float(s) for s in shift], kernel,
        )

    def _shape_transform(self, atom, L, q) -> np.ndarray:
        r""":math:`F_L(q)` of atom ``atom``'s compensation shape, interpolated
        from a table built once per (shape, ``L``)."""
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        r_g = float(self.datasets[atom].compensation_radius)
        q_arr = np.ascontiguousarray(np.asarray(q, dtype=float))
        return context.periodic_shape_transform(r_g, int(L), q_arr)

    def _shape_table(self, atom, L, q_max: float):
        r_g = float(self.datasets[atom].compensation_radius)
        key = (round(r_g, 12), int(L))
        cached = _SHAPE_TABLES.get(key)
        if cached is not None and cached[0][-1] >= q_max:
            return cached
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        table = context.periodic_shape_table(r_g, int(L), float(q_max))
        _SHAPE_TABLES[key] = table
        return table

    def ion_charge(self) -> np.ndarray:
        r"""The Gaussian ions on the grid's reciprocal set (electron sign)."""
        ctx, grid = self._field_backend()
        out = ctx.periodic_ion_charge(
            grid, self.lattice.tolist(), [c.tolist() for c in self.centers],
            [float(z) for z in self.charges],
            [float(d.compensation_radius) for d in self.datasets], float(self.sigma),
        )
        return out.reshape(self.grid.shape)

    @profile_stage('integration: crystal multipole coupling')
    def dense_coulomb(self):
        r"""``(U, U_ion, E_ion)`` of the compact charges, dense in G."""
        if self._compensation_coulomb is not None:
            return self._compensation_coulomb
        context = getattr(self, "_native_context", None)
        if context is None:
            context = _default_native_ctx()
        radii = [float(dataset.compensation_radius) for dataset in self.datasets]
        charges = [float(z) for z in self.charges]
        self._compensation_coulomb = context.periodic_compensation_couplings(
            self.lattice.tolist(), [center.tolist() for center in self.centers],
            charges, radii, self.channels, None, float(self.sigma), COMPENSATION_CUTOFF)
        return self._compensation_coulomb

    def _tail(self, atom, L, kernel=None) -> float:
        r""":math:`8\int_{q_c}^\infty F_L(q)^2\,dq` -- on-site self-energy beyond cutoff."""
        if kernel is not None:
            raise NotImplementedError("custom kernel tail not supported")
        context = self._field_backend()[0]
        radius = float(self.datasets[atom].compensation_radius)
        return context.periodic_shape_tail(radius, int(L), COMPENSATION_CUTOFF)

    def _ion_transform(self, G) -> np.ndarray:
        """The Gaussian ions at the reciprocal vectors ``G`` (electron sign)."""
        ctx = self._field_backend()[0]
        return ctx.periodic_ion_transform(
            self.lattice.tolist(), [c.tolist() for c in self.centers],
            [float(z) for z in self.charges],
            [float(d.compensation_radius) for d in self.datasets],
            float(self.sigma), np.ascontiguousarray(G, dtype=float),
        )

    # -- the constants ------------------------------------------------------ #

    def ion_constants(self, centers=None) -> float:
        r"""Point-ion energy minus the Gaussian ions' reciprocal energy.

        :math:`-\sum_A Z_A^2/(2\sqrt\pi\sigma) + \tfrac12\sum'_{A,B,\mathbf R}
        Z_AZ_B\,\mathrm{erfc}(d/2\sigma)/d`, at ``centers`` (default: the
        atoms').
        """
        context = self._field_backend()[0]
        cents = self.centers if centers is None else centers
        radii = [float(d.compensation_radius) for d in self.datasets]
        charges = [float(z) for z in self.charges]
        return context.periodic_ion_constants(
            self.lattice.tolist(), [c.tolist() for c in cents],
            charges, radii, float(self.sigma),
        )

    def ion_constants_gradient(self) -> np.ndarray:
        r"""``(n_atoms, 3)``: :math:`\partial E/\partial\mathbf R_A` of
        :meth:`ion_constants`, analytic.  Each pair term
        :math:`\tfrac12 Z_AZ_B f(d)`, :math:`f = \mathrm{erfc}(d/2\sigma)/d`,
        moves :math:`A` by :math:`-f'\hat d` and :math:`B` by :math:`+f'\hat d`
        (:math:`\mathbf d = \mathbf R_B + \mathbf R - \mathbf R_A`); the
        pairs inside one atom's images cancel."""
        context = self._field_backend()[0]
        radii = [float(d.compensation_radius) for d in self.datasets]
        charges = [float(z) for z in self.charges]
        return context.periodic_ion_constants_gradient(
            self.lattice.tolist(), [c.tolist() for c in self.centers],
            charges, radii, float(self.sigma),
        )

    def short_range_ion_compensation_gradient(self) -> dict:
        r"""``{channel: (n_atoms, 3)}``: the derivative of
        :meth:`short_range_ion_compensation` with respect to every atom
        (complex, like the sums), analytic.  A shape moves with its owner
        (the quadrature rigidly) and sees every other atom's image through
        :math:`v^{sr\prime}`: the owner gets :math:`+\sum w g\,v'\hat x`,
        the atom it looks at the opposite; an atom's own images cancel."""
        from .local_split import local_cutoff

        context = self._field_backend()[0]
        radii = [float(d.compensation_radius) for d in self.datasets]
        charges = [float(z) for z in self.charges]
        r_tables = [d.r for d in self.datasets]
        v_tables = [d.v_local for d in self.datasets]
        cutoffs = [float(local_cutoff(d)) for d in self.datasets]
        raw = context.periodic_short_range_ion_compensation_gradient(
            self.lattice.tolist(),
            [c.tolist() if hasattr(c, "tolist") else list(c) for c in self.centers],
            charges, radii, self.channels, float(self.sigma),
            r_tables, v_tables, cutoffs,
        )
        return {ch: raw[i] for i, ch in enumerate(self.channels)}

    def onsite_ion_compensation(self) -> dict:
        r"""``{channel: int g_A,00 v^lr_A}`` -- the on-site long-range term the
        reciprocal sum includes and the dataset convention excludes."""
        context = self._field_backend()[0]
        radii = [float(d.compensation_radius) for d in self.datasets]
        charges = [float(z) for z in self.charges]
        vals = context.periodic_onsite_ion_compensation(
            self.lattice.tolist(), [c.tolist() for c in self.centers],
            charges, radii, self.channels, float(self.sigma),
        )
        return {(atom, 0, 0): float(val)
                for (atom, L, M), val in zip(self.channels, vals) if L == 0}

    def short_range_ion_compensation(self, centers=None, atom=None) -> dict:
        r"""``{channel: int g_A,LM sum' v^sr_B}`` over every other atom and
        image (the short-range half of :meth:`~.paw.PAWIntegrals.compensation_ionic_at`),
        at ``centers`` (default: the atoms').  With ``atom`` only the terms
        that atom takes part in -- its own shapes, and every shape against its
        potential -- which is all a displacement of it changes."""
        from .local_split import local_cutoff

        context = self._field_backend()[0]
        use_centers = self.centers if centers is None else centers
        radii = [float(d.compensation_radius) for d in self.datasets]
        charges = [float(z) for z in self.charges]
        r_tables = [d.r for d in self.datasets]
        v_tables = [d.v_local for d in self.datasets]
        cutoffs = [float(local_cutoff(d)) for d in self.datasets]
        vals = context.periodic_short_range_ion_compensation(
            self.lattice.tolist(),
            [c.tolist() if hasattr(c, "tolist") else list(c) for c in use_centers],
            charges, radii, self.channels, float(self.sigma),
            r_tables, v_tables, cutoffs,
        )
        return {ch: complex(val) for ch, val in zip(self.channels, vals)}

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

    def _field_backend(self):
        from .. import _native

        context = getattr(self, "_native_context", None) or getattr(self, "_field_context", None) or _default_native_ctx()
        self._field_context = context
        grid = self.__dict__.get("_field_grid")
        if grid is None and hasattr(self, "grid") and self.grid is not None:
            try:
                grid = _native.SpatialGrid(
                    self.grid.shape, rc.grid_origin(self.grid),
                    np.ascontiguousarray(self.grid.step), periodic=True,
                )
                self._field_grid = grid
            except Exception:
                grid = None
        return context, grid

    def _native_radial_fields(self, tables, *, kinetic_density=False):
        if not tables:
            return None
        context, grid = self._field_backend()
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
        context, grid = self._field_backend()
        return context.periodic_valence_field(
            grid, np.ascontiguousarray(dataset.r, dtype=float),
            np.ascontiguousarray(dataset.valence_density, dtype=float), center)

    def _valence_sources(self):
        return [(np.ascontiguousarray(d.r, dtype=float),
                 np.ascontiguousarray(d.valence_density, dtype=float), center)
                for d, center in zip(self.datasets, self.centers)]

    def initial_density(self) -> tuple[np.ndarray, dict]:
        """Superposed smooth valence densities and reference monopoles."""
        context, grid = self._field_backend()
        rho = context.periodic_initial_valence(grid, self._valence_sources())
        charges = [float(d.compensation_charge) for d in self.datasets]
        return rho, dict(zip(self.channels, context.periodic_initial_monopoles(self.channels, charges)))

    def initial_magnetization(self, moments) -> np.ndarray:
        r"""The starting magnetization :math:`n_\uparrow - n_\downarrow`:
        each atom's superposed valence density weighted by its initial moment
        over its valence charge (clamped to that charge, so neither spin
        density starts negative)."""
        context, grid = self._field_backend()
        charges = [float(d.valence_charge) for d in self.datasets]
        return context.periodic_initial_valence(
            grid, self._valence_sources(), charges=charges, moments=moments, magnetization=True)

    def valence_shares(self) -> np.ndarray:
        r"""``(N, ngrid)``: each atom's share of the superposed valence
        density, :math:`w_A = n_A/\sum_B n_B` (zero where the sum is), the
        Hirshfeld weights of the free atoms' smooth valence densities."""
        context, grid = self._field_backend()
        return context.periodic_valence_shares(grid, self._valence_sources())

    @property
    def n_electrons_neutral(self) -> float:
        return float(sum(self.charges.tolist()))

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
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        mesh = np.asarray(mesh, dtype=float)
        return mesh, context.periodic_uniform_weights(len(mesh)), None
    if operations is not None:
        operations = operations.keeping_mesh(mesh)
    if operations is None:
        reduced, weights = time_reversal_reduce(mesh)
        return reduced, weights, None
    from ..core.symmetry import irreducible_kpoints
    zone = irreducible_kpoints(mesh, operations.info, time_reversal=True)
    context = _default_native_ctx()
    return zone.points, context.periodic_normalize_weights(np.asarray(zone.weights, dtype=float)), operations


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
        context = getattr(self, "_native_context", None) or _default_native_ctx()
        return mesh, context.periodic_range_indices(len(mesh)), None
    if operations is not None:
        operations = operations.keeping_mesh(mesh)
    if operations is not None:
        zone = irreducible_kpoints(mesh, operations.info, time_reversal=True)
        return zone.points, zone.mapping, operations
    context = _default_native_ctx()
    points, slot = context.periodic_mesh_wedge_time_reversal(mesh, list(size))
    return points, slot, None


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
                  precision="float64"):
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
    _backend.ensure_backend()

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
    so_symbols = [s for s, p in potentials.items()
                  if getattr(p, "has_spin_orbit", False)]
    with_spin_orbit = sorted(so_symbols)
    if with_spin_orbit and not spin_orbit:
        # Only the two-component Kohn-Sham solver has spinor Bloch states;
        # dropping the term would report a scalar-relativistic crystal under
        # a spin-orbit dataset's name.
        raise NotImplementedError(
            f"the {', '.join(with_spin_orbit)} dataset(s) carry a spin-orbit term, which only two-component (spinor) Bloch states can hold: method='dft' (LDA or PBE) solves a crystal on them; this consumer of the crystal has no spinor form.  Use a scalar-relativistic set (paw_dataset='lda-sr' or 'pbe-sr') for it")
    if grid is None:
        g = Grid(center=0.5 * (cell[0] + cell[1] + cell[2]), box_size=0.0,
                 grid_spacing=grid_spacing, units="angstrom", cell=cell, periodic=True)
    else:
        g = grid
        # A grid of another cell would be used without complaint and give
        # another crystal's answer: it must be periodic and span this cell.
        from ..units import ANGSTROM_TO_BOHR
        _ctx = native_context or _default_native_ctx()
        valid = getattr(g, "periodic", False) and _ctx.periodic_grid_spans_cell(
            list(g.shape), np.ascontiguousarray(g.step, dtype=float), np.ascontiguousarray(cell, dtype=float), float(ANGSTROM_TO_BOHR)
        )
        if not valid:
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
        all_ch = [l for d in datasets for l in d.channels]
        max_L = max(4, 2 * max(all_ch)) if all_ch else 4
        operations = grid_symmetry(atoms, g, max_L=max_L) \
            if symmetry and not ghosts else None
        reduced, weights, operations = reduce_mesh(mesh, operations)
    _ctx = native_context or _default_native_ctx()
    kpoints = _ctx.periodic_cartesian_kpoints(
        rc.lattice_vectors(g).tolist(), np.ascontiguousarray(reduced, dtype=float)
    )
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
