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

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from ..integrals import reciprocal as rc

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


def _radial_table(function):
    """``(r, values)`` of a tabulated radial function, or ``None``.

    Both layouts in use: a multiple-zeta :class:`~mandacaru.basis.multizeta.
    TabulatedOrbital` keeps a ``table``; a pseudo-atomic orbital keeps the
    arrays themselves (``_r``, ``_values``).
    """
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
    """Functions that share a radial table and a center differ only in ``m``."""
    table = _radial_table(function)
    if table is None or not hasattr(function, "radial"):
        return None
    return (id(table[1]),
            tuple(np.round(np.asarray(function.center, float), 12)))


def bloch_values(functions, lattice, kpoints, points, region_center,
                 region_radius) -> np.ndarray:
    r"""``(nk, M, npts)`` Bloch sums of ``functions`` at ``points``.

    ``points`` is an ``(x, y, z)`` triple of flat arrays (Bohr) lying within
    ``region_radius`` of ``region_center``; only the images that reach that
    region are evaluated, each once for every k-point.  Functions sharing a
    radial table and a center -- the ``m`` components of one shell -- share
    the geometry and the radial interpolation of every image.
    """
    from ..basis._angular import spherical_harmonic

    x, y, z = (np.asarray(c, dtype=float).ravel() for c in points)
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
        shared = _shell_key(first) is not None
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
            phases = np.exp(1j * (kpoints @ R))
            if shared:
                theta = np.arccos(np.clip(dz[idx] / np.maximum(r, 1e-300),
                                          -1.0, 1.0))
                phi = np.arctan2(dy[idx], dx[idx])
                radial = first.radial(r)
                for mu in members:
                    values = radial * spherical_harmonic(
                        functions[mu].l, functions[mu].m, theta, phi)
                    out[:, mu, idx] += phases[:, None] * values[None, :]
            else:
                for mu in members:
                    values = functions[mu].evaluate(x[idx] - R[0],
                                                    y[idx] - R[1],
                                                    z[idx] - R[2])
                    out[:, mu, idx] += phases[:, None] * values[None, :]
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
    species.  Those operations form a subgroup (a composition of two that
    land on nodes lands on nodes), and the k-mesh must be reduced with that
    subgroup and no larger one: the grid decides it, through ``h``.
    """
    from dataclasses import replace as dc_replace

    from ..core.symmetry import DEFAULT_SYMPREC, crystal_symmetry

    info = crystal_symmetry(atoms, DEFAULT_SYMPREC if symprec is None
                            else symprec)
    N = np.asarray(grid.shape, dtype=int)
    lattice = rc.lattice_vectors(grid)
    inverse = np.linalg.inv(lattice)
    frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers())
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
    moments: dict = field(default_factory=dict)   # {(A, L, M): Q(k)}


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
        Fourier-filter cutoff (Bohr^-1) of the partial core density.
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
        self._shape_tables: dict = {}
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
        """:class:`KPointMatrices` of every k-point.

        Each region's Bloch sums -- the cell grid, every projector sphere and
        every short-range sphere -- are evaluated for a block of k-points at
        a time: the images and their radial values do not depend on k, so a
        block shares them, and the block size bounds the memory
        (:data:`KPOINT_BLOCK_BYTES`).
        """
        center, radius = self._cell_region()
        largest = max(self.grid.size, 131072)
        block = max(1, int(KPOINT_BLOCK_BYTES // (16 * self.M * largest)))
        dV = self.grid.dV
        out = []
        for start in range(0, len(self.kpoints), block):
            kpoints = self.kpoints[start:start + block]
            psi_all = bloch_values(self.basis, self.lattice, kpoints,
                                   self._grid_points(), center, radius)
            C_all = self._projections(kpoints)
            V_all = self._short_range_local(kpoints)
            for i, k in enumerate(kpoints):
                weight = self.weights[start + i]
                psi, C = psi_all[i], C_all[i]
                S = (psi.conj() @ psi.T) * dV + C @ self.q_overlap @ C.conj().T
                fixed = (self._kinetic(psi, k) + V_all[i]
                         + C @ self.D_ion @ C.conj().T)
                moments = {}
                for (atom, L, M), blk in self.multipole_blocks.items():
                    Ca = C[:, self._positions[atom]]
                    moments[(atom, L, M)] = Ca @ blk @ Ca.conj().T
                out.append(KPointMatrices(
                    k=np.asarray(k, float), weight=float(weight), psi=psi,
                    overlap=0.5 * (S + S.conj().T),
                    fixed=0.5 * (fixed + fixed.conj().T), projections=C,
                    moments=moments))
        return out

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

    def _projections(self, kpoints) -> np.ndarray:
        r"""``C[k, mu, p]`` :math:`= \langle\chi_{\mu\mathbf k}|\tilde p_p\rangle`.

        Projectors of one atom with the same cutoff share one sphere rule, so
        the Bloch sums are evaluated once per distinct sphere.
        """
        from .paw import _projector_sphere
        C = np.zeros((len(kpoints), self.M, len(self.projectors)),
                     dtype=complex)
        spheres: dict = {}
        for p, projector in enumerate(self.projectors):
            key = (projector.atom_index, round(float(projector.r_cut), 12))
            spheres.setdefault(key, []).append(p)
        for members in spheres.values():
            first = self.projectors[members[0]]
            points, weights = _projector_sphere(first)
            chi = bloch_values(self.basis, self.lattice, kpoints, points,
                               first.center, float(first.r_cut))
            for p in members:
                values = (self.projectors[p].evaluate(*points).ravel()
                          * weights.ravel())
                C[:, :, p] = np.einsum("kmx,x->km", chi.conj(), values)
        return C

    def _short_range_local(self, kpoints) -> np.ndarray:
        """``V_sr[k]`` -- the short-range local term at ``kpoints``."""
        from .local_split import (AZIMUTHAL_POINTS, POLAR_POINTS,
                                  RADIAL_POINTS, local_cutoff,
                                  short_range_potential, short_range_radius)
        out = np.zeros((len(kpoints), self.M, self.M), dtype=complex)
        for atom, dataset in enumerate(self.datasets):
            radius = short_range_radius(dataset, self.sigma)
            points, weights, r, _dirs = _sphere_rule(
                self.centers[atom], radius, RADIAL_POINTS, POLAR_POINTS,
                AZIMUTHAL_POINTS, panel=local_cutoff(dataset))
            potential = np.repeat(short_range_potential(dataset, self.sigma, r),
                                  weights.size // r.size)
            chi = bloch_values(self.basis, self.lattice, kpoints, points,
                               self.centers[atom], radius)
            weighted = weights * potential
            for i in range(len(kpoints)):
                out[i] += (chi[i].conj() * weighted) @ chi[i].T
        return out

    # -- charges ------------------------------------------------------------ #

    def density(self, matrices) -> tuple[np.ndarray, dict]:
        r"""``(n~, q)`` of k-resolved density matrices ``P_k``.

        :math:`\tilde n = \sum_k w_k \sum_{\mu\nu} P^k_{\mu\nu}\chi_\mu\chi_\nu^*`
        and :math:`q^A_{LM} = \sum_k w_k \operatorname{tr}(P^k Q^A_{LM}(k))`.
        """
        rho = np.zeros(self.grid.size)
        q = {c: 0.0j for c in self.channels}
        for data, P in zip(self.kpoint_data, matrices):
            rho += data.weight * np.real(np.sum(data.psi * (P @ data.psi.conj()),
                                                axis=0))
            for channel in self.channels:
                q[channel] += data.weight * np.sum(P * data.moments[channel].T)
        if self.symmetry is not None:
            rho, q = self.symmetry.field(rho), self.symmetry.moments(q)
        return rho, q

    def kinetic_energy_density(self, matrices) -> np.ndarray:
        r""":math:`\tau = \tfrac12\sum_k w_k \sum P^k_{\mu\nu}
        \nabla\chi_\mu\cdot\nabla\chi^*_\nu`."""
        tau = np.zeros(self.grid.size)
        for data, P in zip(self.kpoint_data, matrices):
            for d in self.bloch_gradients(data.psi, data.k):
                tau += 0.5 * data.weight * np.real(np.sum(d * (P @ d.conj()),
                                                          axis=0))
        return tau if self.symmetry is None else self.symmetry.field(tau)

    def compensation_grid(self) -> dict:
        r"""``{channel: g_hat(G)}`` on the grid's reciprocal set (filtered)."""
        if self._compensation_grid is None:
            norm = rc.spherical(self.G)[0]
            out = {}
            for atom, L, M in self.channels:
                out[(atom, L, M)] = rc.multipole_transform(
                    self.G, self.centers[atom], self._shape_transform(atom, L,
                                                                      norm),
                    L, M)
            self._compensation_grid = out
        return self._compensation_grid

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
        cached = self._shape_tables.get(key)
        if cached is not None and cached[0][-1] >= q_max:
            return cached
        top = max(q_max, COMPENSATION_CUTOFF, SHAPE_TABLE_MAX)
        q = np.linspace(0.0, top, int(top / SHAPE_TABLE_STEP) + 1)
        r = np.linspace(0.0, r_g, SHAPE_POINTS)
        table = (q, rc.radial_transform(r, shape_function(r, r_g, L), L, q))
        self._shape_tables[key] = table
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

    def _tail(self, atom, L) -> float:
        r""":math:`8\int_{q_c}^\infty F_L(q)^2\,dq` -- the on-site
        self-energy of a unit :math:`g_LY_{LM}` beyond
        :data:`COMPENSATION_CUTOFF` (the images' share there averages out)."""
        table_q, table_F = self._shape_table(atom, L, SHAPE_TABLE_MAX)
        keep = table_q >= COMPENSATION_CUTOFF
        return float(8.0 * np.trapezoid(table_F[keep] ** 2, table_q[keep]))

    def _ion_transform(self, G) -> np.ndarray:
        """The Gaussian ions at the reciprocal vectors ``G`` (electron sign)."""
        G2 = np.sum(G * G, axis=0)
        total = np.zeros(G.shape[1:], dtype=complex)
        for Z, center in zip(self.charges, self.centers):
            total += -Z * np.exp(-0.5 * self.sigma ** 2 * G2) \
                * rc.structure_factor(G, center)
        return total

    # -- the constants ------------------------------------------------------ #

    def ion_constants(self) -> float:
        r"""Point-ion energy minus the Gaussian ions' reciprocal energy.

        :math:`-\sum_A Z_A^2/(2\sqrt\pi\sigma) + \tfrac12\sum'_{A,B,\mathbf R}
        Z_AZ_B\,\mathrm{erfc}(d/2\sigma)/d`.
        """
        sigma = self.sigma
        energy = -float(np.sum(self.charges ** 2)) / (2.0 * np.sqrt(np.pi)
                                                     * sigma)
        reach = ION_PAIR_REACH * sigma
        for a, (Za, Ra) in enumerate(zip(self.charges, self.centers)):
            for b, (Zb, Rb) in enumerate(zip(self.charges, self.centers)):
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

    def short_range_ion_compensation(self) -> dict:
        r"""``{channel: int g_A,LM sum' v^sr_B}`` over every other atom and
        image (the short-range half of :meth:`~.paw.PAWIntegrals.compensation_ionic_at`)."""
        from ..basis._angular import spherical_harmonic
        from .local_split import short_range_potential, short_range_radius
        from .multipoles import shape_function
        from .paw import (COMPENSATION_RADIAL_POINTS,
                          PROJECTION_AZIMUTHAL_POINTS, PROJECTION_POLAR_POINTS)

        out = {}
        for atom, L, M in self.channels:
            r_g = float(self.datasets[atom].compensation_radius)
            points, weights, r, dirs = _sphere_rule(
                self.centers[atom], r_g, COMPENSATION_RADIAL_POINTS,
                PROJECTION_POLAR_POINTS, PROJECTION_AZIMUTHAL_POINTS)
            theta = np.arccos(np.clip(dirs[2], -1.0, 1.0))
            phi = np.arctan2(dirs[1], dirs[0])
            density = (shape_function(r, r_g, L)[:, None]
                       * spherical_harmonic(L, M, theta, phi)[None, :]).ravel()
            x = np.stack(points, axis=1)
            total = 0.0j
            for other, dataset in enumerate(self.datasets):
                reach = short_range_radius(dataset, self.sigma) + r_g
                d0 = self.centers[atom] - self.centers[other]
                for R in rc.lattice_translations(self.lattice,
                                                 reach + np.linalg.norm(d0)):
                    if other == atom and np.linalg.norm(R) < 1e-10:
                        continue
                    if np.linalg.norm(d0 - R) > reach:
                        continue
                    distance = np.linalg.norm(
                        x - (self.centers[other] + R)[None, :], axis=1)
                    total += np.sum(weights * density
                                    * short_range_potential(dataset,
                                                            self.sigma,
                                                            distance))
            out[(atom, L, M)] = total
        return out

    def core_density(self) -> np.ndarray | None:
        r"""The datasets' smooth cores on the grid, with their images,
        Fourier-filtered at :attr:`filter_cutoff`."""
        from ..integrals.exchange_correlation import xc_core_density
        return self._place_cores(xc_core_density)

    def core_tau(self) -> np.ndarray | None:
        r"""The cores' kinetic-energy densities on the grid, with images.

        Each atom's :math:`\tilde\rho_c'^2/8\tilde\rho_c` evaluated radially
        on the *unfiltered* core (:func:`~mandacaru.integrals.
        exchange_correlation.core_tau_function`), then filtered like the core
        density.  The ratio taken after filtering would divide by the
        filter's ringing tail.
        """
        from ..integrals.exchange_correlation import core_tau_function

        def table(dataset):
            function = core_tau_function(dataset)
            return (None if function is None
                    else function(np.asarray(dataset.r, dtype=float)))
        return self._place_cores(table)

    def _place_cores(self, table) -> np.ndarray | None:
        """Sum of ``table(dataset)`` (a radial array or ``None``) over atoms
        and images, Fourier-filtered at :attr:`filter_cutoff`."""
        from ..basis.filtering import filter_radial
        from scipy.interpolate import CubicSpline

        g = self.grid
        x = np.stack(self._grid_points(), axis=1)
        total = None
        for dataset, center in zip(self.datasets, self.centers):
            values_r = table(dataset)
            if values_r is None:
                continue
            r = np.asarray(dataset.r, dtype=float)
            if self.filter_cutoff is not None:
                values_r, _info = filter_radial(r, values_r, 0,
                                                self.filter_cutoff, warn=False)
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
            total = values if total is None else total + values
        return total

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

def build_crystal(atoms, h: float, options: dict | None = None, kpts=None,
                  family: str = "paw-lcao", grid=None,
                  symmetry: bool = True):
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

    options = resolve_family(family).resolved_options(dict(options or {}))
    load = get_upaw if family == "upaw-lcao" else get_paw
    symbols = atoms.get_chemical_symbols()
    atoms = atoms.copy()
    atoms.wrap()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    potentials = {s: load(s, options.get("directory")) for s in set(symbols)}
    g = grid if grid is not None else Grid(
        center=0.5 * cell.sum(axis=0), box_size=0.0, h=h, units="angstrom",
        cell=cell, periodic=True)
    k_c = filter_cutoff(options.get("filter"), max(g.dx, g.dy, g.dz))
    confinement: dict = {}
    polarization: dict = {}
    basis, atom_of_orbital = pseudo_basis(
        symbols, positions, potentials, filter_cutoff=k_c,
        **pseudo_basis_arguments(family, options, confinement=confinement,
                                 polarization=polarization))
    projectors = paw_projectors(
        symbols, positions, potentials,
        projector_basis=options.get("projector_basis",
                                    DEFAULT_PROJECTOR_BASIS))
    datasets = [potentials[s] for s in symbols]
    size, gamma, mesh = monkhorst_pack_kpts(kpts)
    operations = grid_symmetry(atoms, g) if symmetry else None
    if operations is not None:
        operations = operations.keeping_mesh(mesh)
    if operations is not None:
        from ..core.symmetry import irreducible_kpoints
        zone = irreducible_kpoints(mesh, operations.info, time_reversal=True)
        reduced = zone.points
        weights = zone.weights / zone.weights.sum()
    else:
        reduced, weights = time_reversal_reduce(mesh)
    B = rc.reciprocal_vectors(rc.lattice_vectors(g))
    kpoints = reduced @ B.T
    crystal = PeriodicPAW(
        basis, atom_of_orbital, projectors, datasets,
        [to_bohr(p, "angstrom") for p in positions], g, kpoints, weights,
        paw_coupling_blocks(projectors, symbols, potentials),
        paw_overlap_blocks(projectors, symbols, potentials),
        paw_multipole_blocks(projectors, datasets), filter_cutoff=k_c,
        symmetry=operations)
    context = {"crystal": crystal, "atom_of_orbital": atom_of_orbital,
               "n_electrons": float(valence_electrons(symbols, potentials)),
               "pseudopotentials": potentials, "family": family,
               "filter_cutoff": k_c, "options": dict(options),
               "confinement": confinement, "polarization": polarization,
               "kpts_size": size, "kpts_gamma": gamma,
               "kpoints_fractional": reduced}
    return crystal, context
