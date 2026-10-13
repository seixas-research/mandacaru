# -*- coding: utf-8 -*-
# file: algorithms/periodic_volumetric.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""One-particle pictures of a periodic Kohn-Sham crystal, on its cell grid.

The crystal counterpart of :mod:`mandacaru.algorithms.volumetric`, reached
through the same calculator methods (``write_cube``, ``volumetric_field``,
``natural_orbitals``).  Everything here is a one-particle quantity of the
converged Kohn-Sham state, sampled on the cell grid the SCF ran on: a
density, or one Bloch orbital.

Quantities
----------
``density``
    The smooth valence density :math:`\tilde n`, symmetrized like the SCF's
    (e/Bohr^3).  As for a PAW-LCAO molecule it holds the valence electrons
    minus the augmentation charge, which is reported
    (:attr:`~mandacaru.algorithms.volumetric.VolumetricField.
    augmentation_charge`): ``integral + augmentation_charge`` is the valence
    count exactly.
``alpha_density`` / ``beta_density`` / ``spin_density``
    The channels of a spin-polarized crystal and their difference, whose
    smooth integral plus its augmentation is the moment per cell.
``reconstructed_density``
    The all-electron valence density: the smooth density plus, inside each
    PAW sphere, the spherical part of the one-center correction
    :math:`n^1_A - \tilde n^1_A` -- the density the Bader partition climbs
    (:func:`~mandacaru.algorithms.charges._crystal_reconstruction`), without
    the frozen core.  The correction peaks within a fraction of a Bohr of
    the nucleus, far below a grid's spacing, so it is written as its
    **average over each voxel** (a Gauss-Legendre rule of
    :data:`VOXEL_POINTS` per axis): the file then integrates to the valence
    count, which point samples of those peaks would not (silicon at
    grid_spacing = 0.2 Angstrom: +0.21 e).
``natural_orbital``
    Natural orbital ``index`` (descending occupation) of the spin-summed
    density matrix at an SCF k-point (:func:`bloch_natural_orbitals`).
``bloch_orbital``
    Kohn-Sham orbital ``index`` (ascending energy) at any k-point and spin
    channel, from the converged potential (non-self-consistently away
    from the mesh).

A Bloch orbital :math:`\psi_{n\mathbf k}` is complex.  At a k-point that
time reversal maps onto itself (:math:`2\mathbf k` a reciprocal-lattice
vector, Gamma included) each degenerate set is rotated so its members are
real up to a global phase (:func:`real_gauge`); elsewhere the real part,
the imaginary part or the modulus has to be asked for (``component=``).
Every operator commutes with :math:`\mathbf k\to-\mathbf k` (the kinetic
multiplier of each frequency is a mean over the aliases of
:math:`\mathbf G + \mathbf k`), so that holds on the zone boundary too:
silicon's X point 2e-10, Gamma 1e-14.  ``component="auto"``
still checks the samples and refuses an orbital that is not real.
Repeated cells carry the Bloch phase,
:math:`\psi(\mathbf r + \mathbf R) = e^{i\mathbf k\cdot\mathbf R}\psi(\mathbf r)`;
a density tiles unchanged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..integrals import reciprocal as rc
from ..units import HARTREE_TO_EV

#: What :func:`crystal_field` builds.
CRYSTAL_QUANTITIES = ("density", "alpha_density", "beta_density",
                      "spin_density", "reconstructed_density",
                      "natural_orbital", "bloch_orbital")

#: Gauss-Legendre points per axis of the voxel average of the one-center
#: reconstruction.  Silicon (SZ, h 0.34 and 0.20 Angstrom): the averages
#: integrate to the radial value within 5e-5 e at 8 points, and a 48-point
#: rule on the voxels at the nuclei moves them by less than 3e-5 e.
VOXEL_POINTS = 8

#: Nodes per block of the voxel average (bounds the working set).
VOXEL_BLOCK = 2048

#: Two eigenvalues (Hartree) or occupations closer than this form one
#: degenerate set for the real gauge.
DEGENERACY_TOLERANCE = 1e-6

#: How far ``2k`` may be from a reciprocal-lattice vector (fractional) for k
#: to be time-reversal invariant.
TRIM_TOLERANCE = 1e-8

#: Largest ``|K K^* - 1|`` of a basis closed under complex conjugation.
CONJUGATION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class BlochNaturalOrbitals:
    """Natural orbitals of a crystal's Kohn-Sham density matrix, per k-point.

    The eigen-decomposition of the spin-summed density matrix
    :math:`P_\\mathbf k = P^\\uparrow_\\mathbf k + P^\\downarrow_\\mathbf k`
    at each k-point of the SCF (the reduced set), in the overlap
    :math:`S(\\mathbf k)` of the Bloch sums: :math:`S P S\\,c = n\\,S c`.
    For a spin-restricted crystal they are the Kohn-Sham orbitals and the
    occupations the smeared ones; a spin-polarized crystal's are its
    natural orbitals proper (triplet O2: two orbitals at 1).

    Attributes
    ----------
    kpoints : (nk, 3) ndarray
        The SCF k-points, fractional (reciprocal-lattice units).
    weights : (nk,) ndarray
        Their weights, summing to one.
    occupations : (nk, M) ndarray
        Per k-point, in ``[0, 2]`` and descending.
    coefficients : (nk, M, M) ndarray
        Column ``i`` of ``coefficients[k]`` expands natural orbital ``i`` in
        the Bloch sums of the basis, ``S(k)``-orthonormal.
    n_spins : int
        2 for a spin-polarized crystal.
    """

    kpoints: np.ndarray
    weights: np.ndarray
    occupations: np.ndarray
    coefficients: np.ndarray
    n_spins: int = 1

    @property
    def n_electrons(self) -> float:
        """The valence electrons per cell: the weighted sum of the
        occupations."""
        return float(np.sum(self.weights[:, None] * self.occupations))


# --------------------------------------------------------------------------- #
# k-points and the converged state.
# --------------------------------------------------------------------------- #

def fractional_kpoints(crystal, cartesian) -> np.ndarray:
    """Cartesian k-points (Bohr^-1) in reciprocal-lattice units."""
    B = rc.reciprocal_vectors(crystal.lattice)
    return np.atleast_2d(np.asarray(cartesian, dtype=float)) @ np.linalg.inv(
        B.T)


def is_time_reversal_invariant(fractional) -> bool:
    """Whether ``-k`` is ``k`` up to a reciprocal-lattice vector."""
    twice = 2.0 * np.asarray(fractional, dtype=float)
    return bool(np.all(np.abs(twice - np.rint(twice)) < TRIM_TOLERANCE))


def density_channels(solver) -> list:
    """The converged k-point density matrices, one list per spin channel."""
    matrices = solver.density_matrices
    if matrices is None:
        raise RuntimeError("the crystal has no converged density: run() first")
    return list(matrices) if isinstance(matrices, tuple) else [matrices]


def scf_kpoint(crystal, kpoint) -> int:
    """The index of an SCF k-point: ``None`` (the first, Gamma on a
    Gamma-centered mesh), an index, or fractional coordinates that must be
    one of the reduced points up to a reciprocal-lattice vector."""
    n = len(crystal.kpoint_data)
    if kpoint is None:
        return 0
    if isinstance(kpoint, (int, np.integer)) and not isinstance(kpoint, bool):
        if not 0 <= int(kpoint) < n:
            raise ValueError(f"k-point {int(kpoint)} is out of range: the SCF "
                             f"has {n} (calc.get_ibz_k_points())")
        return int(kpoint)
    wanted = np.asarray(kpoint, dtype=float).reshape(3)
    mesh = fractional_kpoints(crystal, crystal.kpoints)
    for index, k in enumerate(mesh):
        delta = wanted - k
        if np.all(np.abs(delta - np.rint(delta)) < 1e-8):
            return index
    raise ValueError(
        f"k = {tuple(wanted)} is not an SCF k-point: the density matrix "
        f"exists on the reduced mesh only (calc.get_ibz_k_points()).  A Kohn-"
        f"Sham orbital at any k-point is quantity='bloch_orbital'")


def requested_kpoint(crystal, kpoint) -> np.ndarray:
    """Fractional coordinates of a ``kpoint`` option: ``None`` and an index
    name an SCF k-point, three numbers any point."""
    if kpoint is None or (isinstance(kpoint, (int, np.integer))
                          and not isinstance(kpoint, bool)):
        index = scf_kpoint(crystal, kpoint)
        return fractional_kpoints(crystal, crystal.kpoints[index])[0]
    k = np.asarray(kpoint, dtype=float).reshape(-1)
    if k.size != 3 or not np.all(np.isfinite(k)):
        raise ValueError(f"kpoint must be an SCF k-point index or three "
                         f"fractional coordinates, got {kpoint!r}")
    return k


# --------------------------------------------------------------------------- #
# A real gauge where time reversal allows one.
# --------------------------------------------------------------------------- #

def conjugation_matrix(psi, dV: float):
    r"""``K`` with :math:`\chi^*_{\mathbf k} = \chi_{\mathbf k} K` for the
    Bloch sums ``psi`` (``(M, ngrid)``) at a time-reversal invariant k-point,
    or ``None`` when the basis is not closed under conjugation.

    Complex harmonics conjugate into their :math:`-m` partners
    (:math:`Y^*_{lm} = (-1)^m Y_{l,-m}`), and at such a k-point the Bloch
    phases are real, so the sums conjugate among themselves; ``K`` is
    solved from the samples as the molecular
    :meth:`~mandacaru.core.hamiltonian.MolecularIntegrals.conjugation_matrix`
    does.
    """
    psi = np.asarray(psi)
    S0 = (psi.conj() @ psi.T) * dV
    N = (psi.conj() @ psi.conj().T) * dV
    K = np.linalg.solve(S0, N)
    if np.abs(K @ K.conj() - np.eye(len(K))).max() > CONJUGATION_TOLERANCE:
        return None
    return K


def real_gauge(vectors, values, psi, overlap, dV: float,
               tolerance: float = DEGENERACY_TOLERANCE) -> np.ndarray:
    r"""``vectors`` (``S``-orthonormal columns) rotated within each set of
    equal ``values`` so that every orbital is real up to a sign.

    At a time-reversal invariant k-point each set spans a space closed
    under conjugation: :math:`K V^* = V B` with
    :math:`B = V^\dagger S K V^*` unitary and symmetric, its principal
    square root :math:`W` satisfies :math:`W W^T = B`, and :math:`V W` is
    real (:func:`~mandacaru.core.hamiltonian.conjugation_real_orbitals`, in
    the overlap of the Bloch sums).  A set where the construction does not
    hold is left as it is.
    """
    from scipy.linalg import sqrtm

    vectors = np.array(vectors, dtype=complex)
    K = conjugation_matrix(psi, dV)
    if K is None:
        return vectors
    S = np.asarray(overlap)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    groups, current = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] < tolerance:
            current.append(b)
        else:
            groups.append(current)
            current = [b]
    groups.append(current)
    for columns in groups:
        V = vectors[:, columns]
        B = V.conj().T @ S @ K @ V.conj()
        n = len(columns)
        if np.abs(B @ B.conj().T - np.eye(n)).max() > 1e-6:
            continue
        W = sqrtm(0.5 * (B + B.T))
        real = V @ W
        if np.abs(K @ real.conj() - real).max() > 1e-6:
            continue
        vectors[:, columns] = real
    return vectors


# --------------------------------------------------------------------------- #
# Natural orbitals.
# --------------------------------------------------------------------------- #

def _natural_at(crystal, channels, index):
    """``(occupations, coefficients)`` at SCF k-point ``index``, descending."""
    from scipy.linalg import eigh

    data = crystal.kpoint_data[index]
    S = data.overlap
    P = sum(np.asarray(matrices[index]) for matrices in channels)
    SPS = S @ P @ S
    n, c = eigh(0.5 * (SPS + SPS.conj().T), S)
    order = np.argsort(n)[::-1]
    n, c = n[order], c[:, order]
    k = fractional_kpoints(crystal, data.k)[0]
    if is_time_reversal_invariant(k):
        c = real_gauge(c, n, data.psi, S, crystal.grid.dV)
    return np.clip(n, 0.0, None), c


def bloch_natural_orbitals(solver) -> BlochNaturalOrbitals:
    """:class:`BlochNaturalOrbitals` of a converged
    :class:`~mandacaru.algorithms.periodic_dft.PeriodicKohnSham`."""
    crystal = solver.crystal
    channels = density_channels(solver)
    occupations, coefficients = [], []
    for index in range(len(crystal.kpoint_data)):
        n, c = _natural_at(crystal, channels, index)
        occupations.append(n)
        coefficients.append(c)
    return BlochNaturalOrbitals(
        kpoints=fractional_kpoints(crystal, crystal.kpoints),
        weights=np.asarray(crystal.weights, dtype=float),
        occupations=np.array(occupations), coefficients=np.array(coefficients),
        n_spins=len(channels))


# --------------------------------------------------------------------------- #
# Densities.
# --------------------------------------------------------------------------- #

def _augmentation(crystal, matrices) -> float:
    r""":math:`\sum_k w_k\,\mathrm{tr}\,P_k C_k q C_k^\dagger`: the charge
    inside the spheres that the smooth density does not carry."""
    total = 0.0
    for data, P in zip(crystal.kpoint_data, matrices):
        C = data.projections
        total += data.weight * float(np.real(np.sum(
            np.asarray(P) * (C @ crystal.q_overlap @ C.conj().T).T)))
    return total


def _electrons(crystal, matrices) -> float:
    r""":math:`\sum_k w_k\,\mathrm{tr}\,P_k S_k`."""
    return float(sum(data.weight * np.real(np.sum(np.asarray(P)
                                                  * data.overlap.T))
                     for data, P in zip(crystal.kpoint_data, matrices)))


def _smooth(crystal, matrices) -> np.ndarray:
    return np.real(np.asarray(crystal.density(matrices)[0]))


def _support(r, values) -> float:
    """The radius beyond which a radial table is negligible."""
    large = np.nonzero(np.abs(values) > 1e-14 * np.abs(values).max())[0]
    return float(r[large[-1]]) if large.size else 0.0


def reconstruction_on_grid(crystal, channels,
                           points: int = VOXEL_POINTS) -> np.ndarray:
    r"""The spherical one-center correction :math:`\sum_A (n^1_A - \tilde
    n^1_A)` of every atom and image, averaged over each voxel of the cell
    grid (a node's voxel is the parallelepiped of the step vectors centered
    on it), flattened like the grid."""
    from .charges import _crystal_reconstruction, _images

    grid = crystal.grid
    step = np.asarray(grid.step, dtype=float)
    nodes = np.stack([np.ravel(grid.X), np.ravel(grid.Y), np.ravel(grid.Z)],
                     axis=1)
    x, w = np.polynomial.legendre.leggauss(int(points))
    x, w = 0.5 * x, 0.5 * w
    offsets = np.stack(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T \
        @ step.T
    weights = np.einsum("i,j,k->ijk", w, w, w).ravel()
    half = max(float(np.linalg.norm(step @ np.array(s, dtype=float)))
               for s in itertools.product((-0.5, 0.5), repeat=3))
    out = np.zeros(len(nodes))
    for atom, table in enumerate(_crystal_reconstruction(crystal, channels)):
        if table is None:
            continue
        r, values = (np.asarray(t, dtype=float) for t in table)
        reach = _support(r, values) + half
        center = np.asarray(crystal.centers[atom], dtype=float)
        for R in _images(grid, center, crystal.lattice, reach):
            near = np.nonzero(np.linalg.norm(nodes - (center + R), axis=1)
                              < reach)[0]
            for start in range(0, near.size, VOXEL_BLOCK):
                block = near[start:start + VOXEL_BLOCK]
                points_ = nodes[block][:, None, :] + offsets[None]
                distance = np.linalg.norm(points_ - (center + R), axis=2)
                out[block] += np.interp(distance, r, values, right=0.0) \
                    @ weights
    return out


# --------------------------------------------------------------------------- #
# Orbitals.
# --------------------------------------------------------------------------- #

def bloch_orbital(solver, band: int, fractional, spin: int = 0):
    """``(values, energy)``: Kohn-Sham orbital ``band`` (ascending energy)
    of channel ``spin`` at the fractional k-point, as ``(ngrid,)`` complex
    samples of its smooth part, and its energy (Hartree, on the result's
    reference).  At a time-reversal invariant k-point the degenerate sets
    are put in the real gauge (:func:`real_gauge`)."""
    crystal = solver.crystal
    k = crystal.cartesian_kpoints(fractional)
    V, v_tau, w, channel = solver.channel_potentials()[spin]
    try:
        # An SCF k-point keeps its matrices: only H(k) is rebuilt.
        own = [crystal.kpoint_data[scf_kpoint(crystal, fractional)]]
        k = own[0].k[None, :]
    except ValueError:
        own = None
    data, eps, vectors = next(solver._diagonalized(k, V, v_tau, w, channel,
                                                   data=own))
    data, eps, vectors = data[0], np.asarray(eps[0]), np.asarray(vectors[0])
    if is_time_reversal_invariant(fractional):
        vectors = real_gauge(vectors, eps, data.psi, data.overlap,
                             crystal.grid.dV)
    energy = float(eps[band] + solver.eigenvalue_reference())
    return vectors[:, band] @ data.psi, energy


# --------------------------------------------------------------------------- #
# The field.
# --------------------------------------------------------------------------- #

def imaginary_ratio(values) -> float:
    """``max |Im| / max |Re|`` of ``values`` once its global phase (that of
    its largest sample) is removed: zero for a function real up to a
    phase."""
    values = np.asarray(values)
    largest = values.flat[int(np.argmax(np.abs(values)))]
    if abs(largest) > 0:
        values = values * np.exp(-1j * np.angle(largest))
    return float(np.abs(values.imag).max()
                 / (np.abs(values.real).max() or 1.0))


def _repeat(repeat) -> tuple:
    if repeat is None:
        return (1, 1, 1)
    if isinstance(repeat, (int, np.integer)) and not isinstance(repeat, bool):
        repeat = (repeat,) * 3
    out = tuple(int(n) for n in repeat)
    if len(out) != 3 or min(out) < 1 or any(
            float(a) != float(b) for a, b in zip(out, repeat)):
        raise ValueError(f"repeat must be three positive integers, got "
                         f"{repeat!r}")
    return out


def tile(values, lattice, repeat, k=None) -> np.ndarray:
    """A cell field ``values`` (``(n1, n2, n3)``) over ``repeat`` cells: a
    density unchanged, a Bloch orbital at the Cartesian ``k`` times
    :math:`e^{i\\mathbf k\\cdot\\mathbf R}` in the cell at ``R``."""
    values = np.asarray(values)
    if repeat == (1, 1, 1):
        return values
    n = values.shape
    out = np.empty(tuple(r * m for r, m in zip(repeat, n)),
                   dtype=complex if k is not None else values.dtype)
    for t in itertools.product(*[range(r) for r in repeat]):
        block = tuple(slice(i * m, (i + 1) * m) for i, m in zip(t, n))
        if k is None:
            out[block] = values
        else:
            R = np.asarray(lattice, dtype=float) @ np.asarray(t, dtype=float)
            out[block] = values * np.exp(1j * float(np.dot(k, R)))
    return out


def cell_atoms(crystal, repeat, numbers=None):
    """``(numbers, valence charges, positions)`` of the atoms in the written
    box: each atom wrapped into the cell the grid spans, then repeated."""
    from ase.data import atomic_numbers

    lattice = np.asarray(crystal.lattice, dtype=float)       # columns
    origin = rc.grid_origin(crystal.grid)
    if numbers is None:
        numbers = [atomic_numbers[d.symbol] for d in crystal.datasets]
    numbers = np.asarray(numbers, dtype=int).reshape(-1)
    if len(numbers) != len(crystal.centers):
        raise ValueError(f"{len(numbers)} atomic numbers were given for "
                         f"{len(crystal.centers)} atoms")
    charges = np.asarray(crystal.charges, dtype=float)
    fractional = np.linalg.solve(
        lattice, (np.asarray(crystal.centers, dtype=float) - origin).T).T
    fractional = fractional - np.floor(fractional + 1e-9)
    out_numbers, out_charges, out_positions = [], [], []
    for t in itertools.product(*[range(r) for r in repeat]):
        for Z, q, f in zip(numbers, charges, fractional):
            out_numbers.append(int(Z))
            out_charges.append(float(q))
            out_positions.append(origin + lattice @ (f + np.asarray(t)))
    return (np.array(out_numbers), np.array(out_charges),
            np.reshape(out_positions, (-1, 3)))


def crystal_field(solver, quantity: str = "density", index: int = 0, *,
                  kpoint=None, spin=None, component: str = "auto",
                  repeat=None, numbers=None):
    """One real-space quantity of a converged periodic Kohn-Sham crystal
    (:data:`CRYSTAL_QUANTITIES`), as a periodic
    :class:`~mandacaru.algorithms.volumetric.VolumetricField`.

    ``solver`` is the converged
    :class:`~mandacaru.algorithms.periodic_dft.PeriodicKohnSham`.  ``index``
    is the band (``bloch_orbital``, ascending energy) or the natural orbital
    (descending occupation); ``kpoint`` an SCF k-point index or fractional
    coordinates (default the first SCF k-point, Gamma on a Gamma-centered
    mesh); ``spin`` the channel of a spin-polarized crystal's
    ``bloch_orbital``; ``repeat`` the number of cells along each lattice
    vector; ``numbers`` the elements drawn (default the datasets').
    """
    from .volumetric import (DENSITY_UNITS, ORBITAL_UNITS,
                             REAL_ORBITAL_TOLERANCE, VolumetricField,
                             _real_part)

    if quantity not in CRYSTAL_QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r} for a crystal; use "
                         f"one of {CRYSTAL_QUANTITIES}")
    crystal = solver.crystal
    grid = crystal.grid
    repeat = _repeat(repeat)
    channels = density_channels(solver)
    n_spins = len(channels)
    spin_quantities = ("alpha_density", "beta_density", "spin_density")
    if quantity in spin_quantities and n_spins == 1:
        raise ValueError(f"{quantity} needs a spin-polarized crystal (initial "
                         f"magnetic moments); this one is spin-restricted")
    if spin is not None and quantity != "bloch_orbital":
        raise ValueError(f"spin= selects the channel of a bloch_orbital; for "
                         f"{quantity} use alpha_density / beta_density / "
                         f"spin_density")
    if kpoint is not None and quantity not in ("bloch_orbital",
                                               "natural_orbital"):
        raise ValueError(f"kpoint= belongs to an orbital; a {quantity} is "
                         f"summed over the whole mesh")
    lattice = np.asarray(crystal.lattice, dtype=float)
    notes: list[str] = []
    occupation = n_electrons = k_fractional = None
    orbital_index = None
    augmentation = 0.0
    if quantity in ("bloch_orbital", "natural_orbital"):
        index = int(index)
        if not 0 <= index < crystal.M:
            raise ValueError(f"orbital index {index} is out of range: the "
                             f"basis has {crystal.M} functions per cell")
        orbital_index = index
        if quantity == "natural_orbital":
            at = scf_kpoint(crystal, kpoint)
            k_fractional = fractional_kpoints(crystal, crystal.kpoints[at])[0]
            n, c = _natural_at(crystal, channels, at)
            occupation = float(n[index])
            values = c[:, index] @ crystal.kpoint_data[at].psi
            notes.append("natural orbitals of the spin-summed density "
                         "matrix, by descending occupation")
        else:
            if n_spins == 2 and spin not in (0, 1):
                raise ValueError("a spin-polarized crystal has orbitals per "
                                 "channel: pass spin=0 or 1")
            if n_spins == 1 and spin not in (None, 0):
                raise ValueError("a spin-restricted crystal has one channel: "
                                 "spin=0 or None")
            k_fractional = requested_kpoint(crystal, kpoint)
            values, energy = bloch_orbital(solver, index, k_fractional,
                                           0 if spin is None else int(spin))
            notes.append(f"Kohn-Sham band {index} at "
                         f"{energy * HARTREE_TO_EV:+.6f} eV")
            if n_spins == 2:
                notes.append(f"spin {'up' if int(spin) == 0 else 'down'}")
        notes.append("k = (" + ", ".join(f"{x:.6f}" for x in k_fractional)
                     + ") reciprocal-lattice units")
        k_cartesian = crystal.cartesian_kpoints(k_fractional)[0]
        cell_values = np.asarray(values).reshape(grid.shape)
        if component == "auto" and not is_time_reversal_invariant(
                k_fractional):
            raise ValueError(
                "a Bloch orbital away from a time-reversal invariant "
                "k-point (2k a reciprocal-lattice vector) is genuinely "
                "complex; pass component='real', 'imag' or 'modulus'")
        tiled = tile(cell_values, lattice, repeat, k_cartesian)
        if component == "auto":
            residual = imaginary_ratio(tiled)
            if residual > REAL_ORBITAL_TOLERANCE:
                raise ValueError(
                    f"this Bloch orbital is real only to max |Im| / max |Re| "
                    f"= {residual:.2e} after removing its global phase.  "
                    f"Pass component='real', 'imag' or 'modulus'")
        data, component = _real_part(tiled, component,
                                     f"{quantity} {index}")
        cell_part = data[:grid.shape[0], :grid.shape[1], :grid.shape[2]]
        units = ORBITAL_UNITS
        if component == "modulus":
            notes.append("modulus of a complex orbital")
        elif component == "imag":
            notes.append("imaginary part of a complex orbital")
        notes.append("smooth part of the orbital (inside the PAW spheres the "
                     "all-electron one differs)")
    else:
        if quantity == "spin_density":
            values = _smooth(crystal, channels[0]) - _smooth(crystal,
                                                             channels[1])
            augmentation = (_augmentation(crystal, channels[0])
                            - _augmentation(crystal, channels[1]))
            n_electrons = (_electrons(crystal, channels[0])
                           - _electrons(crystal, channels[1]))
        elif quantity in ("alpha_density", "beta_density"):
            matrices = channels[0 if quantity == "alpha_density" else 1]
            values = _smooth(crystal, matrices)
            augmentation = _augmentation(crystal, matrices)
            n_electrons = _electrons(crystal, matrices)
        else:
            values = sum(_smooth(crystal, m) for m in channels)
            n_electrons = sum(_electrons(crystal, m) for m in channels)
            augmentation = sum(_augmentation(crystal, m) for m in channels)
            if quantity == "reconstructed_density":
                values = values + reconstruction_on_grid(crystal, channels)
                augmentation = 0.0
                notes.append("all-electron valence density: smooth density "
                             "plus the spherical one-center correction, "
                             "averaged over each voxel; no core")
        cell_part = np.ascontiguousarray(values.reshape(grid.shape))
        data = tile(cell_part, lattice, repeat)
        units = DENSITY_UNITS
        component = "real"
        if augmentation:
            notes.append("smooth pseudo valence density: the augmentation "
                         "charge above is NOT in this file")
    if repeat != (1, 1, 1):
        notes.append("{}x{}x{} cells; integral and norm of one".format(*repeat))
    elements, charges, positions = cell_atoms(crystal, repeat, numbers)
    dV = grid.dV
    return VolumetricField(
        quantity=quantity, data=np.ascontiguousarray(data), grid=grid,
        numbers=elements, positions=positions, charges=charges, units=units,
        integral=float(np.sum(cell_part) * dV),
        norm=float(np.sum(cell_part * cell_part) * dV),
        index=orbital_index, occupation=occupation, n_electrons=n_electrons,
        augmentation_charge=float(augmentation), component=component,
        notes=tuple(notes), cell=(lattice @ np.diag(repeat)).T,
        repeat=repeat, kpoint=k_fractional,
        spin=None if spin is None else int(spin))
