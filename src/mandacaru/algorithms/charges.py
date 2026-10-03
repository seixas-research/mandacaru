# -*- coding: utf-8 -*-
# file: algorithms/charges.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Splitting a converged density between the atoms that share it.

An atom in a molecule has no boundary.  The electron density is one continuous
function, and **every** partial charge is therefore a *convention* -- a choice
of weight function :math:`w_A(\mathbf r)` with :math:`\sum_A w_A = 1`, giving

.. math::

    N_A = \int w_A(\mathbf r)\, n(\mathbf r)\, d^3r , \qquad
    q_A = Z_A - N_A .

Three conventions are implemented, and they disagree by design:

``hirshfeld``
    :math:`w_A = n_A^0 / \sum_B n_B^0`, the *stockholder* split: each atom
    takes the share of the density its own free-atom density contributes to
    the superposition of free atoms (the promolecule).  The reference atoms
    are solved here, spherically and self-consistently, by the same LDA radial
    solver the bases are generated with
    (:mod:`mandacaru.basis.atomic_solver`), so a pseudopotential run is
    compared against a **valence** promolecule and an all-electron run against
    the full one -- matching what the grid density actually holds.  Smooth,
    basis-insensitive, and famously *small*: Hirshfeld charges understate
    ionicity because the reference is always a neutral atom.
``voronoi``
    :math:`w_A = 1` where :math:`A` is the nearest nucleus.  Purely geometric:
    it knows nothing about the density it is cutting, which makes it a useful
    control -- a charge that changes a lot between Voronoi and Hirshfeld is a
    charge the partition is deciding, not the physics.
``bader``
    The zero-flux partition: every point climbs the density along its
    steepest gradient to a maximum, and the points reaching the same nucleus
    are one basin (Bader's atoms in molecules).  The ascent is *continuous*
    -- on a cubic-spline interpolation of the density, from every point of a
    refined grid (:func:`bader_populations`) -- not a walk between grid
    nodes: an on-grid ascent steers along the grid's own directions and
    underestimated cation charges by 0.04-0.2 e at h = 0.18-0.30 Angstrom,
    moving by 0.01 e when the atoms were shifted against the grid
    (HISTORY.md, 2026-10-03).  A pseudopotential's valence density need not
    peak at the nuclei (covalent silicon's maxima sit at the bond centers), so
    for a valence density the ascent runs on valence plus the free atoms'
    frozen cores, and only the valence density is integrated.  It follows the
    density rather than a reference and gives the largest charges of the
    three.

The **total** charge and the **total** magnetic moment need none of this --
they are integrals of the whole density, fixed by the state:
:math:`\sum_A q_A` is the system's charge and
:math:`\int (n_\alpha - n_\beta) = N_\alpha - N_\beta` whatever the weights.
Only the split between atoms is a convention, and
:meth:`~mandacaru.algorithms.calculator.Mandacaru.get_total_magnetic_moment`
therefore does not take a ``method``.

Pseudopotentials
----------------
``Z_A`` is the charge the *Hamiltonian* carries, which for a pseudopotential
run is the valence charge (O is 6, not 8), so ``q_A`` is still the physical
partial charge.  For PAW-LCAO the grid holds the *smooth* density, and the charge
inside the augmentation spheres is added back per atom from
:math:`C_A q_A C_A^\dagger` -- the same on-site correction the overlap carries,
which is block-diagonal per atom, so the decomposition is exact rather than a
sharing rule.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

#: The partitions :func:`atomic_weights` knows.
PARTITION_METHODS = ("hirshfeld", "voronoi", "bader")

#: The partitions that are a weight on the grid nodes (:func:`atomic_weights`).
WEIGHT_METHODS = ("hirshfeld", "voronoi")

#: Radial grid of the free-atom references: nodes out to this radius (Bohr).
#: The promolecule only needs the density where the molecular grid has weight,
#: and a neutral atom's density is below 1e-10 e/Bohr^3 well inside 20 Bohr.
REFERENCE_POINTS = 1500
REFERENCE_RADIUS = 20.0

#: Floor of the promolecule denominator (e/Bohr^3).  Far from every nucleus
#: the reference densities underflow and the stockholder fractions become
#: 0/0; there is no density there to share, so the weights fall back to the
#: geometric (Voronoi) assignment rather than to a NaN.
PROMOLECULE_FLOOR = 1.0e-12

#: Cache of solved reference atoms, keyed by ``(Z, valence subshells)``.
_REFERENCE_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Free-atom references.
# --------------------------------------------------------------------------- #

def free_atom_density(atomic_number: int, subshells=None):
    """``(r, rho)`` of a neutral free atom: the Hirshfeld reference.

    ``rho`` is the spherical density in e/Bohr^3, so
    :math:`\\int 4\\pi r^2 \\rho\\,dr` is the electron count.  ``subshells``
    restricts the sum to those ``(n, l)`` -- how a **valence** reference is
    built for a pseudopotential run, whose grid density holds the valence
    electrons only.  ``None`` keeps every occupied subshell.

    The atom is solved once per ``(Z, subshells)`` and cached: a promolecule
    of a hundred carbons pays for one carbon.
    """
    from ..basis.atomic_solver import solve_atom

    Z = int(atomic_number)
    key = (Z, None if subshells is None else tuple(sorted(subshells)))
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    atom = solve_atom(Z, points=REFERENCE_POINTS, r_max=REFERENCE_RADIUS)
    if subshells is None:
        rho = np.asarray(atom.density, dtype=float)
    else:
        wanted = {tuple(int(v) for v in nl) for nl in subshells}
        rho = np.zeros_like(atom.r)
        for (n, l), occupancy in atom.occupations.items():
            if (int(n), int(l)) in wanted and occupancy > 0:
                # rho = sum_nl occ |R_nl|^2 / (4 pi); u = r R.
                u = atom.orbitals[(n, l)]
                rho += occupancy * (u / atom.r) ** 2 / (4.0 * np.pi)
    _REFERENCE_CACHE[key] = (np.asarray(atom.r, dtype=float), rho)
    return _REFERENCE_CACHE[key]


def reference_subshells(atomic_number: int, valence: bool):
    """The ``(n, l)`` set a reference atom is summed over.

    ``valence=True`` returns the same valence set the pseudopotentials and the
    minimal bases are built from (:func:`mandacaru.basis._config.
    valence_subshells`), so the promolecule holds exactly the electrons a
    pseudopotential run put on the grid.  It is taken in the configuration
    :func:`free_atom_density` solves the atom in, which is not always aufbau
    (lanthanum's reference atom carries 5d, not 4f).
    """
    if not valence:
        return None
    from ..basis._config import valence_subshells
    from ..basis.atomic_solver import relaxed_configuration
    Z = int(atomic_number)
    return valence_subshells(
        Z, configuration=relaxed_configuration(Z, r_max=REFERENCE_RADIUS))


# --------------------------------------------------------------------------- #
# Weight functions.
# --------------------------------------------------------------------------- #

def _distances(grid, position):
    """Distance (Bohr) from every grid node to ``position``."""
    return np.sqrt((grid.X - position[0]) ** 2
                   + (grid.Y - position[1]) ** 2
                   + (grid.Z - position[2]) ** 2)


def _grid_sphere(grid):
    """Center and radius (Bohr) of a sphere holding every grid node."""
    nodes = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)],
                     axis=1)
    center = nodes.mean(axis=0)
    return center, float(np.max(np.linalg.norm(nodes - center, axis=1)))


def _images(grid, position, lattice, cutoff: float) -> np.ndarray:
    """Translations ``R`` (rows, Bohr) that bring ``position + R`` within
    ``cutoff`` of some grid node -- the only images a quantity of range
    ``cutoff`` sees.  Independent of how skewed the cell is or where the atom
    was wrapped."""
    from ..integrals.reciprocal import lattice_translations
    center, radius = _grid_sphere(grid)
    offset = np.asarray(position, dtype=float) - center
    reach = radius + float(cutoff)
    R = lattice_translations(lattice, float(np.linalg.norm(offset)) + reach)
    return R[np.linalg.norm(offset + R, axis=1) <= reach]


def _nearest_reach(grid) -> float:
    """A range that contains every node's nearest image of any atom: the
    grid's diameter (an atom inside the node sphere is within it of every
    node, and some image of every atom is)."""
    return 2.0 * _grid_sphere(grid)[1]


def _nearest_distances(grid, position, lattice=None):
    """Distance from every node to ``position`` or, in a crystal (``lattice``
    as columns, Bohr), to its nearest periodic image."""
    if lattice is None:
        return _distances(grid, position)
    best = None
    for R in _images(grid, position, lattice, _nearest_reach(grid)):
        d = _distances(grid, np.asarray(position) + R)
        best = d if best is None else np.minimum(best, d)
    return best


#: Relative tolerance on a distance tie in the Voronoi partition.  A symmetric
#: molecule on a symmetric grid puts whole node *layers* exactly on the
#: dividing plane, and giving them all to the lower atom index charges a
#: homonuclear dimer by 0.16 e at some grid spacings and not at others.
VORONOI_TIE = 1.0e-9


def voronoi_weights(grid, positions, lattice=None) -> np.ndarray:
    """``(A, nx, ny, nz)`` indicator of the nearest nucleus.

    A node equidistant from several nuclei is **split evenly** between them
    (:data:`VORONOI_TIE`), which a symmetric molecule needs: its dividing
    plane falls on a node layer whenever the grid has an odd number of them
    between the atoms, and handing that layer to one side is a pure
    discretization charge.  With the split, H2 comes out neutral at every
    ``h``.  In a crystal (``lattice``) the distance is to the nearest image.
    """
    positions = np.asarray(positions, dtype=float)
    distances = np.stack([_nearest_distances(grid, position, lattice)
                          for position in positions])
    best = distances.min(axis=0)
    tied = distances <= best * (1.0 + VORONOI_TIE) + VORONOI_TIE
    return tied / tied.sum(axis=0)


def hirshfeld_weights(grid, positions, numbers, valence=False,
                      lattice=None) -> np.ndarray:
    """``(A, nx, ny, nz)`` stockholder weights from a free-atom promolecule.

    Where the promolecule underflows -- far outside every atom, below
    :data:`PROMOLECULE_FLOOR` -- there is no density to share and the
    stockholder fraction is 0/0; those nodes take the geometric assignment
    instead, which keeps :math:`\\sum_A w_A = 1` everywhere without inventing
    a share.  In a crystal (``lattice``) each atom's reference is summed over
    every image within :data:`REFERENCE_RADIUS` of the cell: the promolecule
    is the periodic superposition of free atoms.
    """
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    references = np.zeros((len(positions),) + tuple(grid.shape), dtype=float)
    for index, (Z, position) in enumerate(zip(numbers, positions)):
        r, rho = free_atom_density(Z, reference_subshells(Z, valence))
        images = (np.zeros((1, 3)) if lattice is None else
                  _images(grid, position, lattice, REFERENCE_RADIUS))
        for R in images:
            distance = _distances(grid, position + R)
            # Outside the reference grid the density is zero, not the last
            # node's value, so the promolecule does not grow a plateau at
            # large r.
            references[index] += np.interp(distance, r, rho, left=rho[0],
                                           right=0.0)
    total = references.sum(axis=0)
    empty = total < PROMOLECULE_FLOOR
    weights = np.divide(references, np.where(empty, 1.0, total))
    if np.any(empty):
        weights[:, empty] = voronoi_weights(grid, positions,
                                            lattice)[:, empty]
    return weights


def atomic_weights(method: str, grid, positions, numbers=None,
                   valence: bool = False, lattice=None) -> np.ndarray:
    """``(A, nx, ny, nz)`` weights of one of :data:`WEIGHT_METHODS`;
    periodic with ``lattice`` (columns, Bohr).  Bader's basins are not a
    weight on the grid nodes: :func:`bader_populations`."""
    key = str(method).strip().lower()
    if key == "voronoi":
        return voronoi_weights(grid, positions, lattice)
    if key == "hirshfeld":
        if numbers is None:
            raise ValueError("the Hirshfeld promolecule needs atomic numbers")
        return hirshfeld_weights(grid, positions, numbers, valence=valence,
                                 lattice=lattice)
    raise ValueError(f"unknown weight partition {method!r}; use one of "
                     f"{WEIGHT_METHODS} (Bader: bader_populations)")


# --------------------------------------------------------------------------- #
# Bader: continuous steepest ascent.
# --------------------------------------------------------------------------- #

#: Integration points per grid spacing along each axis.  Refining 1 -> 2
#: moved the converged charges by ~0.01 e; 2 -> 3 less.
BADER_REFINE = 2

#: Ascent step, as a fraction of the shortest grid step.
BADER_STEP = 0.2

#: A trajectory within this distance (Bohr) of a nucleus belongs to it.
BADER_CAPTURE = 0.25

#: Steps after which a trajectory still free (a flat tail) is assigned to
#: the nucleus nearest where it stopped.
BADER_MAX_STEPS = 400

#: Every this many steps a trajectory that has moved less than two steps is
#: stalled -- oscillating about a saddle (silicon's bond centers, a dimer's
#: midpoint) -- and is assigned like one that ran out of steps.  Without it
#: those few trajectories ran all :data:`BADER_MAX_STEPS` and were most of
#: the cost (silicon, 12 nodes: 130 s).
BADER_STALL_WINDOW = 10

#: Below this density (e/Bohr^3) a point is not traced but goes to its
#: nearest nucleus.  A molecular box is mostly such tail -- millions of
#: refined points carrying, all together, ~1e-3 e in a 10 Angstrom box --
#: and tracing them was the whole cost of a molecular partition.
BADER_DENSITY_FLOOR = 1e-7

#: Relative tolerance on a distance tie between nuclei (a point stopped on a
#: symmetric saddle is split evenly, as a Voronoi tie is).
BADER_TIE = 1e-6


def _frozen_cores(numbers):
    """Per atom ``(r, core, d core/dr)`` of the free atom's core density
    (all-electron minus valence), or ``None`` without a core."""
    out = []
    for Z in np.asarray(numbers, dtype=int):
        r, full = free_atom_density(int(Z), reference_subshells(int(Z), False))
        r_valence, valence = free_atom_density(
            int(Z), reference_subshells(int(Z), True))
        core = np.clip(full - np.interp(r, r_valence, valence), 0.0, None)
        if not np.any(core > 0.0):
            out.append(None)
            continue
        support = float(r[np.nonzero(core > 1e-12 * core.max())[0][-1]])
        out.append((r, core, np.gradient(core, r), support))
    return out


def _image_sets(grid, positions, lattice, cutoff):
    """Per atom the image centers within ``cutoff`` (Bohr) of some grid node
    -- a single one without a lattice.  Trajectories are kept inside the
    cell, so nothing farther can matter to them."""
    positions = np.asarray(positions, dtype=float)
    if lattice is None:
        return [p[None, :] for p in positions]
    return [p[None, :] + _images(grid, p, lattice, cutoff)
            for p in positions]


def _nearest(points, images):
    """``(distance (n, A), )`` to each atom's nearest image."""
    out = np.empty((len(points), len(images)))
    for atom, centers in enumerate(images):
        d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        out[:, atom] = d.min(axis=1)
    return out


def bader_populations(grid, fields, positions, *, lattice=None, cores=None,
                      refine: int = BADER_REFINE) -> np.ndarray:
    r"""``(len(fields), n_atoms)``: each field integrated over the Bader
    basins of ``fields[0]`` (plus the analytic ``cores``, if given).

    ``fields`` are ``grid.shape`` arrays (the first one is the density whose
    basins are traced; a magnetization can follow).  Each is interpolated by
    a cubic spline -- periodic with ``lattice`` (columns, Bohr), clamped at a
    molecular box's faces -- on ``refine`` points per spacing and axis, and
    renormalized so its integral is the grid's.  From every point with
    density, the trajectory climbs :math:`\nabla\rho` in steps of
    :data:`BADER_STEP` grid spacings until it comes within
    :data:`BADER_CAPTURE` of a nucleus (its nearest image); a trajectory that
    stops elsewhere goes to the nucleus nearest its end, split evenly on a
    tie.  ``cores`` (from :func:`_frozen_cores`) are added analytically to
    the climbed density -- their gradient is exact -- and never integrated.
    """
    from scipy.ndimage import map_coordinates, spline_filter

    from ..integrals import _backend as backend

    shape = tuple(int(n) for n in grid.shape)
    A = np.asarray(grid.step, dtype=float)
    A_inv = np.linalg.inv(A)
    origin = np.array([np.ravel(grid.X)[0], np.ravel(grid.Y)[0],
                       np.ravel(grid.Z)[0]])
    mode = "grid-wrap" if lattice is not None else "nearest"
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    fields = [np.asarray(f, dtype=float).reshape(shape) for f in fields]

    def spline(field):
        return spline_filter(field, order=3, mode=mode)

    def evaluate(coefficients, index):
        return map_coordinates(coefficients, index.T, order=3, mode=mode,
                               prefilter=False)

    axes = [np.arange(n * refine) / refine for n in shape]
    if lattice is None:
        axes = [a[a <= n - 1] for a, n in zip(axes, shape)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    weight = abs(np.linalg.det(A)) / refine ** 3
    values = []
    for array in fields:
        sampled = evaluate(spline(array), points)
        exact = float(np.sum(array)) * abs(np.linalg.det(A))
        approximate = float(np.sum(sampled)) * weight
        if abs(approximate) > 1e-14 and abs(exact) > 1e-14:
            sampled = sampled * (exact / approximate)
        values.append(sampled)

    basin = spline(fields[0])
    h = float(np.min(np.linalg.norm(A, axis=0)))
    # Free trajectories are assigned to their nearest nucleus, so the capture
    # images must also cover the farthest point of the cell from an atom.
    capture_images = _image_sets(grid, positions, lattice,
                                 _nearest_reach(grid))
    has_cores = cores is not None and any(c is not None for c in cores)

    # The spline's gradient, taken once at the nodes and interpolated
    # trilinearly along the trajectories: the ascent needs a smooth direction
    # field, and six cubic-spline evaluations per point and step were the
    # whole cost of a partition.
    nodes_index = np.stack(np.meshgrid(
        *[np.arange(n, dtype=float) for n in shape], indexing="ij"),
        axis=-1).reshape(-1, 3)
    node_gradient = []
    eps = 0.02
    for d in range(3):
        shift = np.zeros(3)
        shift[d] = eps
        node_gradient.append(((evaluate(basin, nodes_index + shift)
                               - evaluate(basin, nodes_index - shift))
                              / (2.0 * eps)).reshape(shape))

    fractional_atoms = positions @ np.linalg.inv(lattice).T \
        if lattice is not None else None

    # A frozen core's gradient: the nearest image's -- the steep part,
    # too sharp to interpolate (LiH moved by 0.02 e) -- is analytic along
    # the trajectories; every other image lies beyond half a lattice-plane
    # spacing, where the core is small and smooth, and those join the node
    # field once.

    def offsets(x):
        """``(n, A, 3)``: from each atom's nearest image to every point (by
        rounding fractional coordinates)."""
        if lattice is None:
            return x[:, None, :] - positions[None, :, :]
        delta = (x @ np.linalg.inv(lattice).T)[:, None, :] \
            - fractional_atoms[None, :, :]
        delta -= np.round(delta)
        return delta @ lattice.T

    def core_slope(v, core):
        """Cartesian gradient of one core at offsets ``v`` (n, 3)."""
        r_table, _c, dcore, support = core
        r = np.linalg.norm(v, axis=1)
        g = np.zeros_like(v)
        near = (r < support) & (r > 1e-12)
        if np.any(near):
            g[near] = (np.interp(r[near], r_table, dcore)
                       / r[near])[:, None] * v[near]
        return g

    if has_cores and lattice is not None:
        x_nodes = origin + nodes_index @ A.T
        nearest_offsets = offsets(x_nodes)
        others = np.zeros_like(x_nodes)
        reach = max(c[3] for c in cores if c is not None)
        for atom, (centers, core) in enumerate(zip(
                _image_sets(grid, positions, lattice, reach), cores)):
            if core is None:
                continue
            for center in centers:
                others += core_slope(x_nodes - center, core)
            others -= core_slope(nearest_offsets[:, atom], core)
        others_index = others @ A                # d/d index = A^T grad
        for d in range(3):
            node_gradient[d] = node_gradient[d] + others_index[:, d].reshape(
                shape)

    def gradient(index):
        # Interpolated (trilinearly) from the node field, plus each core's
        # nearest image.
        g = np.stack([map_coordinates(component, index.T, order=1,
                                      mode=mode)
                      for component in node_gradient], axis=1) @ A_inv
        if has_cores:
            v = offsets(origin + index @ A.T)
            for atom, core in enumerate(cores):
                if core is not None:
                    g += core_slope(v[:, atom], core)
        return g

    def captured_by(x):
        """``(atom, distance)`` of the nucleus each point is closest to,
        among the images a capture can involve: the capture radius is far
        below any cell, so the nearest lattice vector by rounding is it."""
        d = np.linalg.norm(offsets(x), axis=2)
        nearest = np.argmin(d, axis=1)
        return nearest, d[np.arange(len(x)), nearest]

    upper = np.asarray(shape, float) - 1.0

    def trace(starts):
        """``(n, n_atoms)`` ownership of trajectories from index ``starts``:
        one-hot where captured, the nearest nucleus (split on a tie) where a
        trajectory is still free after :data:`BADER_MAX_STEPS`."""
        index = np.array(starts, dtype=float)
        result = np.zeros((len(index), n_atoms))
        traced = backend.bader_ascent(
            index, node_gradient, lattice is not None, A, A_inv, origin,
            lattice, positions,
            [None if c is None else (c[0], c[2], c[3])
             for c in (cores if has_cores else [None] * n_atoms)],
            BADER_STEP * h, BADER_CAPTURE, BADER_MAX_STEPS,
            BADER_STALL_WINDOW, 2.0 * BADER_STEP * h)
        if traced is not None:
            owner, final = traced
            captured = owner >= 0
            result[np.nonzero(captured)[0], owner[captured]] = 1.0
            if np.any(~captured):
                result[~captured] = _nearest_share(origin + final[~captured]
                                                   @ A.T)
            return result
        active = np.arange(len(index))
        anchor = index.copy()
        unwrapped = index.copy()                # for the stall test
        for count in range(1, BADER_MAX_STEPS + 1):
            if not active.size:
                break
            if count % BADER_STALL_WINDOW == 0:
                moved_by = np.linalg.norm(
                    (unwrapped[active] - anchor[active]) @ A.T, axis=1)
                stalled = moved_by < 2.0 * BADER_STEP * h
                if np.any(stalled):
                    rows = active[stalled]
                    result[rows] = _nearest_share(origin + index[rows] @ A.T)
                    active = active[~stalled]
                    if not active.size:
                        break
                anchor[active] = unwrapped[active]
            x = origin + index[active] @ A.T
            nearest, distance = captured_by(x)
            captured = distance < BADER_CAPTURE
            result[active[captured], nearest[captured]] = 1.0
            active = active[~captured]
            if not active.size:
                break
            g = gradient(index[active])
            norm = np.linalg.norm(g, axis=1, keepdims=True)
            step = BADER_STEP * h * g / np.where(norm > 0.0, norm, 1.0)
            moved = index[active] + step @ A_inv.T
            unwrapped[active] += step @ A_inv.T
            # A crystal's trajectory is folded back into the cell (the
            # density is periodic); a molecule's stays inside its box.
            moved = (np.mod(moved, np.asarray(shape, float))
                     if lattice is not None else np.clip(moved, 0.0, upper))
            index[active] = moved
        if active.size:
            result[active] = _nearest_share(origin + index[active] @ A.T)
        return result

    def _nearest_share(cartesian):
        distances = _nearest(cartesian, capture_images)
        best = distances.min(axis=1, keepdims=True)
        tied = distances <= best * (1.0 + BADER_TIE) + BADER_TIE
        return tied / tied.sum(axis=1, keepdims=True)

    # 1. The grid nodes: traced where they carry density.
    nodes = np.stack(np.meshgrid(*[np.arange(n, dtype=float) for n in shape],
                                 indexing="ij"), axis=-1).reshape(-1, 3)
    node_density = fields[0].reshape(-1)
    node_owner = np.zeros((len(nodes), n_atoms))
    live = node_density > BADER_DENSITY_FLOOR
    node_owner[live] = trace(nodes[live])
    if np.any(~live):
        node_owner[~live] = _nearest_share(origin + nodes[~live] @ A.T)

    # 2. Boundary nodes: a neighbor went to another basin.
    labels = np.argmax(node_owner, axis=1).reshape(shape)
    boundary = np.zeros(shape, dtype=bool)
    for axis in range(3):
        for shift in (-1, 1):
            differs = np.roll(labels, shift, axis=axis) != labels
            if lattice is None:                 # a box face has no neighbor
                edge = [slice(None)] * 3
                edge[axis] = 0 if shift == 1 else -1
                differs[tuple(edge)] = False
            boundary |= differs

    # 3. The integration points: inside a basin -- every node around a point
    # agrees -- a point takes its nodes' basin; next to the zero-flux surface
    # it is traced from the eight corners of a quarter-spacing cube and the
    # ownership averaged.  A symmetric grid puts whole planes of points
    # exactly *on* that surface (a dimer's bond midplane), where round-off
    # would pick the side (H2 charged by 0.08 e); the corners split them
    # evenly, and resolve the surface below the grid spacing.
    lower = np.floor(points).astype(int)
    near = np.zeros(len(points), dtype=bool)
    for corner in np.ndindex(2, 2, 2):
        around = lower + np.asarray(corner)
        if lattice is not None:
            around %= np.asarray(shape)
        else:
            around = np.minimum(around, np.asarray(shape) - 1)
        flat = np.ravel_multi_index(tuple(around.T), shape)
        near |= boundary.reshape(-1)[flat]
    nearest_node = np.rint(points).astype(int)
    if lattice is not None:
        nearest_node %= np.asarray(shape)
    else:
        nearest_node = np.minimum(nearest_node, np.asarray(shape) - 1)
    owner = node_owner[np.ravel_multi_index(tuple(nearest_node.T), shape)]
    refine_these = near & (values[0] > BADER_DENSITY_FLOOR)
    if np.any(refine_these):
        quarter = 0.25 / refine
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)], dtype=float) * quarter
        starts = points[refine_these][:, None, :] + corners[None, :, :]
        if lattice is None:
            starts = np.clip(starts, 0.0, upper)
        owner[refine_these] = trace(starts.reshape(-1, 3)).reshape(
            -1, 8, n_atoms).mean(axis=1)

    out = np.array([owner.T @ v * weight for v in values])
    if np.any(owner.sum(axis=0) == 0.0):
        starved = [int(a) for a in np.nonzero(owner.sum(axis=0) == 0.0)[0]]
        warnings.warn(
            f"the Bader partition gave atom(s) {starved} no basin: no "
            "trajectory reached them.  Refine h.", RuntimeWarning,
            stacklevel=2)
    return out


# --------------------------------------------------------------------------- #
# The result.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AtomicPartition:
    """Per-atom charges and moments of one converged state.

    ``charges`` is ``reference_charges - populations``: positive means the
    atom gave electrons away.  ``augmentation`` is the part of each
    population that came from inside a PAW sphere rather than from the grid
    (zero for every other basis), and is already included in ``populations``.
    """

    method: str
    charges: np.ndarray
    populations: np.ndarray
    reference_charges: np.ndarray
    magnetic_moments: np.ndarray
    total_charge: float
    total_magnetic_moment: float
    augmentation: np.ndarray
    notes: tuple = field(default_factory=tuple)

    @property
    def grid_electrons(self) -> float:
        """Electrons the partition accounted for, augmentation included."""
        return float(np.sum(self.populations))

    def summary(self) -> str:
        """Multi-line report, one row per atom, for a script to print."""
        lines = [f"{self.method} partition",
                 f"{'atom':>5}{'Z_eff':>8}{'electrons':>12}{'charge':>10}"
                 f"{'moment':>10}"]
        for i, (Z, N, q, m) in enumerate(zip(self.reference_charges,
                                             self.populations, self.charges,
                                             self.magnetic_moments)):
            lines.append(f"{i:>5}{Z:>8.2f}{N:>12.6f}{q:>+10.6f}{m:>+10.6f}")
        lines.append(f"{'total':>5}{'':>8}{self.grid_electrons:>12.6f}"
                     f"{self.total_charge:>+10.6f}"
                     f"{self.total_magnetic_moment:>+10.6f}")
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# From a converged state.
# --------------------------------------------------------------------------- #

def _augmentation_by_atom(integrals, orbitals, n_atoms: int) -> np.ndarray:
    """Charge each PAW sphere holds off the grid, per atom.

    :math:`S - \\tilde S = C q C^\\dagger` is block-diagonal over the projector
    channels ``(atom, l, m)``, so restricting ``C`` to one atom's columns
    splits it exactly -- no sharing rule is involved.  Zero for every
    norm-conserving and all-electron basis, which carry no such term.
    """
    out = np.zeros(n_atoms, dtype=float)
    if getattr(integrals, "nonlocal_overlap", None) is None:
        return out
    from ..core.hamiltonian import projector_blocks

    Q = integrals.nonlocal_overlap_matrix()
    C = integrals.projections()
    if Q is None or C.shape[1] == 0:
        return out
    columns: dict = {}
    for (atom, _l, _m), positions in projector_blocks(
            integrals.kb_projectors).items():
        columns.setdefault(int(atom), []).extend(int(p) for p in positions)
    coefficients = orbitals.coefficients
    for atom, positions in columns.items():
        index = np.asarray(sorted(positions), dtype=int)
        block = C[:, index] @ Q[np.ix_(index, index)] @ C[:, index].conj().T
        per_orbital = np.real(np.einsum("mi,mn,ni->i", np.conj(coefficients),
                                        block, coefficients))
        out[atom] = float(np.sum(orbitals.occupations * per_orbital))
    return out


def partition_state(integrals, gamma, *, method: str = "hirshfeld",
                    frozen=(), n_spatial_orbitals=None, numbers=None,
                    grid=None, active=None) -> AtomicPartition:
    """Split a converged state's density and spin density between the atoms.

    This is the solver-free entry point, the counterpart of
    :func:`~mandacaru.algorithms.volumetric.volumetric_field`:
    :meth:`~mandacaru.algorithms.calculator.Mandacaru.get_charges` is the
    user-facing wrapper that supplies ``integrals`` and ``gamma`` from a
    finished run.

    Parameters
    ----------
    integrals : MolecularIntegrals
        The live integral object of the run -- its basis, grid, nuclei and
        molecular orbitals.
    gamma : (2*M_act, 2*M_act) array
        The active-space spin-orbital one-RDM, alpha block first.
    method : str
        One of :data:`PARTITION_METHODS`.
    frozen : sequence of int
        Frozen spatial orbitals, refilled into both spin channels.
    active : sequence of int, optional
        Spatial orbitals the register carried; needed only when the virtual
        space was truncated (see
        :func:`~mandacaru.algorithms.volumetric.spin_resolved_rdm`).
    numbers : sequence of int, optional
        True atomic numbers, used only to pick the Hirshfeld reference atoms.
        Defaults to the Hamiltonian's charges, which for a pseudopotential run
        are the valence charges -- pass ``atoms.get_atomic_numbers()`` so the
        reference is the right element.
    grid : Grid, optional
        Partition on this grid instead of the calculation's own.
    """
    from .volumetric import OrbitalExpansion, _spinors, spin_resolved_rdm

    key = str(method).strip().lower()
    if key not in PARTITION_METHODS:
        raise ValueError(f"unknown partition {method!r}; use one of "
                         f"{PARTITION_METHODS}")

    expansion = OrbitalExpansion(integrals, grid=grid)
    used = expansion.grid
    M = int(len(integrals.basis) if n_spatial_orbitals is None
            else n_spatial_orbitals)
    D_alpha, D_beta = spin_resolved_rdm(gamma, M, frozen, active,
                                        spinors=_spinors(integrals))

    # `_potentials.nuclei` is the Bohr frame the grid and the basis functions
    # live in; `integrals.nuclei` is in the integrals' own `units` (Angstrom by
    # default) and putting it on the grid displaces every nucleus by 1.889.
    nuclei = integrals._potentials.nuclei
    charges = np.array([Z for Z, _R in nuclei], dtype=float)
    positions = np.array([R for _Z, R in nuclei], dtype=float)
    n_atoms = len(charges)
    elements = (np.rint(charges).astype(int) if numbers is None
                else np.asarray(numbers, dtype=int).reshape(-1))
    if len(elements) != n_atoms:
        raise ValueError(f"expected {n_atoms} atomic numbers, got "
                         f"{len(elements)}")
    # A pseudopotential run put only the valence electrons on the grid, so the
    # promolecule must hold the same ones or the stockholder fractions would
    # be weighted by core density that is not there to share.  Asked of the
    # potentials rather than inferred from the charges, which agree for
    # hydrogen and helium whose valence *is* the whole atom.
    valence = getattr(integrals._potentials, "pseudopotentials", None) is not None

    alpha_flat, _ = expansion.density(D_alpha)
    beta_flat, _ = expansion.density(D_beta)
    alpha = np.real(alpha_flat).reshape(used.shape)
    beta = np.real(beta_flat).reshape(used.shape)
    density = alpha + beta

    if key == "bader":
        populations, moments = bader_populations(
            used, [density, alpha - beta], positions,
            cores=_frozen_cores(elements) if valence else None)
    else:
        weights = atomic_weights(key, used, positions, numbers=elements,
                                 valence=valence)
        dV = used.dV
        populations = np.einsum("axyz,xyz->a", weights, density) * dV
        moments = np.einsum("axyz,xyz->a", weights, alpha - beta) * dV

    augmentation = (
        _augmentation_by_atom(integrals,
                              expansion.natural_orbitals(D_alpha + D_beta),
                              n_atoms))
    populations = populations + augmentation
    spin_augmentation = (
        _augmentation_by_atom(integrals,
                              expansion.natural_orbitals(D_alpha), n_atoms)
        - _augmentation_by_atom(integrals,
                                expansion.natural_orbitals(D_beta), n_atoms))
    moments = moments + spin_augmentation

    notes = []
    if valence:
        notes.append("pseudopotential run: Z is the valence charge and the "
                     "Hirshfeld reference is the valence free atom")
    if np.any(augmentation):
        notes.append(f"PAW augmentation {np.sum(augmentation):+.6f} e added "
                     f"per atom from inside the spheres")
    return AtomicPartition(
        method=key,
        charges=charges - populations,
        populations=populations,
        reference_charges=charges,
        magnetic_moments=moments,
        total_charge=float(np.sum(charges) - np.sum(populations)),
        total_magnetic_moment=float(np.real(np.trace(D_alpha)
                                            - np.trace(D_beta))),
        augmentation=augmentation,
        notes=tuple(notes))


# --------------------------------------------------------------------------- #
# From a converged crystal.
# --------------------------------------------------------------------------- #

def partition_crystal(crystal, matrices, *, method: str = "hirshfeld",
                      numbers=None) -> AtomicPartition:
    r"""Split a periodic Kohn-Sham density between the atoms of the cell.

    ``crystal`` is the run's
    :class:`~mandacaru.pseudopotentials.periodic_paw.PeriodicPAW` and
    ``matrices`` its converged k-point density matrices :math:`P_{\mathbf k}` (occupations included).  The smooth
    density is the crystal's own, symmetrized like the SCF's; the weights are
    periodic (nearest image, periodic promolecule, a periodic continuous
    ascent on valence plus frozen core, :func:`bader_populations`); and the
    charge inside each atom's augmentation sphere,
    :math:`\sum_k w_k\,\mathrm{tr}\,P_k C_A q_A C_A^\dagger`, is added to that
    atom exactly, as for a molecule.  ``matrices`` may be a pair
    ``(P_up, P_down)`` of a spin-polarized crystal: the charges are those of
    their sum, and each atom's moment is the same weights applied to the
    magnetization (its sphere's share included); otherwise every moment is
    zero.
    """
    key = str(method).strip().lower()
    if key not in PARTITION_METHODS:
        raise ValueError(f"unknown partition {method!r}; use one of "
                         f"{PARTITION_METHODS}")
    grid = crystal.grid
    spin = isinstance(matrices, tuple)
    channels = list(matrices) if spin else [matrices]
    densities = [np.real(np.asarray(crystal.density(P)[0])).reshape(grid.shape)
                 for P in channels]
    density = sum(densities)
    positions = np.asarray(crystal.centers, dtype=float)
    charges = np.asarray(crystal.charges, dtype=float)
    n_atoms = len(charges)
    elements = (np.rint(charges).astype(int) if numbers is None
                else np.asarray(numbers, dtype=int).reshape(-1))
    if len(elements) != n_atoms:
        raise ValueError(f"expected {n_atoms} atomic numbers, got "
                         f"{len(elements)}")
    magnetization = (densities[0] - densities[1]) if spin else \
        np.zeros_like(density)
    if key == "bader":
        populations, grid_moments = bader_populations(
            grid, [density, magnetization], positions,
            lattice=crystal.lattice, cores=_frozen_cores(elements))
    else:
        weights = atomic_weights(key, grid, positions, numbers=elements,
                                 valence=True, lattice=crystal.lattice)
        populations = np.einsum("axyz,xyz->a", weights, density) * grid.dV
        grid_moments = np.einsum("axyz,xyz->a", weights,
                                 magnetization) * grid.dV

    def sphere_charges(matrix_list):
        out = np.zeros(n_atoms, dtype=float)
        for atom, columns in crystal._positions.items():
            index = np.asarray(columns, dtype=int)
            Q = crystal.q_overlap[np.ix_(index, index)]
            for data, P in zip(crystal.kpoint_data, matrix_list):
                C = data.projections[:, index]
                out[atom] += data.weight * float(np.real(
                    np.sum(P * (C @ Q @ C.conj().T).T)))
        symmetry = getattr(crystal, "symmetry", None)
        if symmetry is not None:
            # Summed over the wedge, each atom's sphere charge is the full
            # mesh's only once averaged over the atoms the operations map it
            # to -- as the density and the multipoles are.
            maps = symmetry.atom_maps
            out = np.array([np.mean([out[int(m[atom])] for m in maps])
                            for atom in range(n_atoms)])
        return out

    spheres = [sphere_charges(P) for P in channels]
    augmentation = sum(spheres)
    populations = populations + augmentation
    moments = np.zeros(n_atoms)
    total_moment = 0.0
    if spin:
        moments = grid_moments + spheres[0] - spheres[1]
        total_moment = float(np.sum(magnetization) * grid.dV
                             + np.sum(spheres[0] - spheres[1]))

    notes = ["pseudopotential run: Z is the valence charge and the Hirshfeld "
             "reference is the valence free atom",
             "periodic: nearest-image distances and a periodic promolecule"]
    if key == "bader":
        notes.append("Bader basins by continuous ascent on valence + "
                     "frozen-core density; only the valence is integrated")
    if np.any(augmentation):
        notes.append(f"PAW augmentation {np.sum(augmentation):+.6f} e added "
                     f"per atom from inside the spheres")
    return AtomicPartition(
        method=key, charges=charges - populations, populations=populations,
        reference_charges=charges, magnetic_moments=moments,
        total_charge=float(np.sum(charges) - np.sum(populations)),
        total_magnetic_moment=total_moment, augmentation=augmentation,
        notes=tuple(notes))
