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
    The zero-flux partition, on the grid: every point walks uphill along the
    steepest density gradient until it stops at a maximum, and the points that
    reach the same maximum are one basin (Henkelman, Arnaldsson and Jonsson,
    *Comput. Mater. Sci.* **36**, 354, 2006).  Each basin is then attached to
    the nucleus nearest its maximum.  It follows the density rather than a
    reference and gives the largest charges of the three, at the price of
    being the one that needs a fine grid: the basin boundary is resolved to
    one grid spacing.

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
    pseudopotential run put on the grid.
    """
    if not valence:
        return None
    from ..basis._config import valence_subshells
    return valence_subshells(int(atomic_number))


# --------------------------------------------------------------------------- #
# Weight functions.
# --------------------------------------------------------------------------- #

def _distances(grid, position):
    """Distance (Bohr) from every grid node to ``position``."""
    return np.sqrt((grid.X - position[0]) ** 2
                   + (grid.Y - position[1]) ** 2
                   + (grid.Z - position[2]) ** 2)


#: Relative tolerance on a distance tie in the Voronoi partition.  A symmetric
#: molecule on a symmetric grid puts whole node *layers* exactly on the
#: dividing plane, and giving them all to the lower atom index charges a
#: homonuclear dimer by 0.16 e at some grid spacings and not at others.
VORONOI_TIE = 1.0e-9


def voronoi_weights(grid, positions) -> np.ndarray:
    """``(A, nx, ny, nz)`` indicator of the nearest nucleus.

    A node equidistant from several nuclei is **split evenly** between them
    (:data:`VORONOI_TIE`), which a symmetric molecule needs: its dividing
    plane falls on a node layer whenever the grid has an odd number of them
    between the atoms, and handing that layer to one side is a pure
    discretization charge.  With the split, H2 comes out neutral at every
    ``h``.
    """
    positions = np.asarray(positions, dtype=float)
    distances = np.stack([_distances(grid, position)
                          for position in positions])
    best = distances.min(axis=0)
    tied = distances <= best * (1.0 + VORONOI_TIE) + VORONOI_TIE
    return tied / tied.sum(axis=0)


def hirshfeld_weights(grid, positions, numbers, valence=False) -> np.ndarray:
    """``(A, nx, ny, nz)`` stockholder weights from a free-atom promolecule.

    Where the promolecule underflows -- far outside every atom, below
    :data:`PROMOLECULE_FLOOR` -- there is no density to share and the
    stockholder fraction is 0/0; those nodes take the geometric assignment
    instead, which keeps :math:`\\sum_A w_A = 1` everywhere without inventing
    a share.
    """
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    references = np.zeros((len(positions),) + tuple(grid.shape), dtype=float)
    for index, (Z, position) in enumerate(zip(numbers, positions)):
        r, rho = free_atom_density(Z, reference_subshells(Z, valence))
        distance = _distances(grid, position)
        # Outside the reference grid the density is zero, not the last node's
        # value, so the promolecule does not grow a plateau at large r.
        references[index] = np.interp(distance, r, rho, left=rho[0], right=0.0)
    total = references.sum(axis=0)
    empty = total < PROMOLECULE_FLOOR
    weights = np.divide(references, np.where(empty, 1.0, total))
    if np.any(empty):
        weights[:, empty] = voronoi_weights(grid, positions)[:, empty]
    return weights


#: Offsets of the 26 neighbours of a grid node.
_NEIGHBORS = tuple((i, j, k)
                   for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                   if (i, j, k) != (0, 0, 0))


#: Relative tolerance on a steepest-ascent tie.  The separatrix between two
#: basins is where the uphill directions are equal, and a symmetric molecule
#: puts a whole node layer on it.
BADER_TIE = 1.0e-9


def _ascent(grid, density):
    """``(best_slope, pointer)``: the steepest uphill neighbour of every node.

    ``pointer`` is the flat index of that neighbour, or the node itself when
    nothing around it is higher.  ``best_slope`` is the rise per unit distance
    it achieves, which :func:`bader_weights` reuses to find the ties.
    """
    density = np.asarray(density, dtype=float)
    shape = density.shape
    flat = np.arange(density.size, dtype=np.intp).reshape(shape)
    step = np.asarray(grid.step, dtype=float)

    best_slope = np.zeros(shape, dtype=float)
    pointer = flat.copy()
    for offset in _NEIGHBORS:
        # np.roll is periodic; the molecular box has ~zero density at its
        # faces, and a wrapped neighbour there cannot out-climb the interior.
        shifted = np.roll(density, shift=[-o for o in offset], axis=(0, 1, 2))
        neighbor = np.roll(flat, shift=[-o for o in offset], axis=(0, 1, 2))
        distance = float(np.linalg.norm(step @ np.asarray(offset, dtype=float)))
        slope = (shifted - density) / distance
        uphill = slope > best_slope
        best_slope = np.where(uphill, slope, best_slope)
        pointer = np.where(uphill, neighbor, pointer)
    return best_slope, pointer


def bader_basins(grid, density) -> np.ndarray:
    """Basin label of every grid node, by on-grid steepest ascent.

    Each node points at the neighbour with the largest density *rise per unit
    distance* -- the discrete gradient, so a diagonal neighbour is not
    favoured just for being further away -- and a node with no uphill
    neighbour points at itself and is a maximum.  Following the pointers by
    repeated squaring (:math:`p \\leftarrow p[p]`) resolves every node to its
    maximum in :math:`O(\\log N)` passes rather than by walking each path.

    Returns the flat index of the maximum each node ascends to; nodes sharing
    one are one basin.
    """
    shape = np.asarray(density).shape
    _slope, pointer = _ascent(grid, density)
    pointer = pointer.reshape(-1)
    # Pointer doubling: after k passes each node has climbed 2^k steps.
    while True:
        hopped = pointer[pointer]
        if np.array_equal(hopped, pointer):
            break
        pointer = hopped
    return pointer.reshape(shape)


def bader_weights(grid, density, positions) -> np.ndarray:
    """``(A, nx, ny, nz)`` indicator of the Bader basin of each atom.

    Every basin is attached to the nucleus nearest its maximum, and basins
    whose maximum is not near any nucleus (a bond critical point the grid
    turned into a shallow maximum, or noise in the tail) go to the nearest
    nucleus all the same -- so no charge is lost.
    """
    positions = np.asarray(positions, dtype=float)
    basins = bader_basins(grid, density)
    maxima = np.unique(basins)

    coordinates = np.stack([grid.X.reshape(-1), grid.Y.reshape(-1),
                            grid.Z.reshape(-1)], axis=1)
    # One nucleus per maximum, then one label per node through the basin map.
    separation = np.linalg.norm(coordinates[maxima][:, None, :]
                                - positions[None, :, :], axis=2)
    owner_of_maximum = np.argmin(separation, axis=1)
    lookup = np.zeros(int(basins.max()) + 1, dtype=np.intp)
    lookup[maxima] = owner_of_maximum
    owner = lookup[basins]

    # Fractional weights on the separatrix.  A node whose steepest ascent is a
    # tie sits on the boundary between the basins its tied neighbours belong
    # to, and giving it wholly to the first of them is the same discretization
    # charge the Voronoi tie-break had: it charges H2 by 0.16 e at some grid
    # spacings.  A node with one best neighbour is unaffected -- the split
    # reproduces the hard assignment exactly.
    density = np.asarray(density, dtype=float)
    best_slope, _pointer = _ascent(grid, density)
    threshold = best_slope * (1.0 - BADER_TIE) - BADER_TIE
    flat_owner = owner.reshape(-1)
    flat = np.arange(density.size).reshape(density.shape)
    step = np.asarray(grid.step, dtype=float)

    ties = []
    shares = np.zeros(density.shape, dtype=float)
    for offset in _NEIGHBORS:
        shifted = np.roll(density, shift=[-o for o in offset], axis=(0, 1, 2))
        neighbor = np.roll(flat, shift=[-o for o in offset], axis=(0, 1, 2))
        distance = float(np.linalg.norm(step @ np.asarray(offset, dtype=float)))
        tie = ((shifted - density) / distance >= threshold) & (best_slope > 0.0)
        shares += tie
        ties.append((tie, flat_owner[neighbor]))

    n_atoms = len(positions)
    weights = np.zeros((n_atoms,) + tuple(grid.shape), dtype=float)
    fraction = np.where(shares > 0, 1.0 / np.maximum(shares, 1.0), 0.0)
    # A local maximum has no uphill neighbour at all and keeps itself.
    alone = shares == 0
    for index in range(n_atoms):
        weights[index] += alone & (owner == index)
    for tie, neighbor_owner in ties:
        share = np.where(tie, fraction, 0.0)
        for index in range(n_atoms):
            weights[index] += np.where(neighbor_owner == index, share, 0.0)

    # A grid too coarse to separate two nuclei merges their basins, and the
    # atom that loses its maximum is handed *zero* electrons -- a silent,
    # spectacular wrong answer (H2 at h = 0.30 A: one basin, charges +-1).
    # Say so; there is no tolerance that makes it right.
    starved = [int(a) for a in range(len(positions)) if not weights[a].any()]
    if starved:
        warnings.warn(
            f"the Bader partition found {len(maxima)} density maxima for "
            f"{len(positions)} atoms, so atom(s) {starved} were left with no "
            f"basin and no electrons: the grid does not separate them.  Refine "
            f"h, or use method='hirshfeld', whose weights need no basin.",
            RuntimeWarning, stacklevel=2)
    return weights


def atomic_weights(method: str, grid, positions, numbers=None, density=None,
                   valence: bool = False) -> np.ndarray:
    """``(A, nx, ny, nz)`` weights of one of :data:`PARTITION_METHODS`."""
    key = str(method).strip().lower()
    if key == "voronoi":
        return voronoi_weights(grid, positions)
    if key == "hirshfeld":
        if numbers is None:
            raise ValueError("the Hirshfeld promolecule needs atomic numbers")
        return hirshfeld_weights(grid, positions, numbers, valence=valence)
    if key == "bader":
        if density is None:
            raise ValueError("the Bader partition needs the density it cuts")
        return bader_weights(grid, density, positions)
    raise ValueError(f"unknown partition {method!r}; use one of "
                     f"{PARTITION_METHODS}")


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
    from .volumetric import OrbitalExpansion, spin_resolved_rdm

    key = str(method).strip().lower()
    if key not in PARTITION_METHODS:
        raise ValueError(f"unknown partition {method!r}; use one of "
                         f"{PARTITION_METHODS}")

    expansion = OrbitalExpansion(integrals, grid=grid)
    used = expansion.grid
    M = int(len(integrals.basis) if n_spatial_orbitals is None
            else n_spatial_orbitals)
    D_alpha, D_beta = spin_resolved_rdm(gamma, M, frozen, active)

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

    weights = atomic_weights(key, used, positions, numbers=elements,
                             density=density, valence=valence)

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
