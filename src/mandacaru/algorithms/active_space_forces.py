# -*- coding: utf-8 -*-
# file: algorithms/active_space_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Nuclear response of a geometry-dependent reduced active Hamiltonian.

The usual AO derivative holds the molecular orbital coefficients fixed.  When
virtual orbitals are deleted, the active projector itself moves with the nuclei
(and an MP2 or natural selector rotates it).  A central difference of the
*reduced* Hamiltonian includes this response and the frozen-core constant.
Only the Hamiltonian is rebuilt at each displacement; the optimized state and
its RDMs are reused by the variational Hellmann-Feynman theorem.  A polar
alignment removes arbitrary SCF eigenvector phases and active-space rotations
before a displaced Hamiltonian is contracted with the reference RDMs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..units import HARTREE_TO_EV

if TYPE_CHECKING:
    from ase import Atoms
    from ..core.hamiltonian import MolecularIntegrals
    from .base import VariationalDriver

ACTIVE_SPACE_STEP_ANGSTROM = 0.005
"""Default central-difference displacement, in Angstrom."""


def reduced_energy(integrals: MolecularIntegrals, gamma: np.ndarray,
                   gamma2: np.ndarray, rotation: np.ndarray | None = None
                   ) -> float:
    r"""Contract a reduced Hamiltonian with fixed spin-orbital RDMs, in Ha.

    ``rotation`` maps the displaced active orbitals into the reference orbital
    gauge.  The integral arrays already contain the core effective potential;
    ``active_constant`` includes core, ionic, and one-center energies.
    """
    h = np.asarray(integrals.active_h_so)
    g = np.asarray(integrals.active_g_so)
    if rotation is not None:
        n = rotation.shape[0]
        u = np.zeros((2 * n, 2 * n), dtype=complex)
        u[:n, :n] = rotation
        u[n:, n:] = rotation
        h = u.conj().T @ h @ u
        g = np.einsum("ap,bq,cr,ds,abcd->pqrs", u.conj(), u.conj(),
                      u, u, g, optimize=True)
    energy = (np.einsum("pq,pq->", h, gamma, optimize=True)
              + 0.5 * np.einsum("pqrs,pqrs->", g, gamma2, optimize=True))
    return float(np.real(energy) + integrals.active_constant)


def align_active_orbitals(reference: MolecularIntegrals,
                          displaced: MolecularIntegrals,
                          reference_active: tuple[int, ...],
                          displaced_active: tuple[int, ...]) -> np.ndarray:
    r"""Unitary bringing displaced active MOs into the reference MO gauge.

    The Löwdin AO coordinates have the same atom and function ordering for a
    small nuclear step.  Their active-orbital overlap is unitized by its polar
    factor.  A small singular value signals that the selector switched to a
    different active subspace, where a smooth local derivative is undefined.
    """
    a = np.asarray(reference.mo_coefficients)[:, reference_active]
    b = np.asarray(displaced.mo_coefficients)[:, displaced_active]
    if a.shape != b.shape:
        raise RuntimeError("the displaced active space changed dimension")
    left, singular, right = np.linalg.svd(b.conj().T @ a)
    if singular.min(initial=1.0) < 0.5:
        raise RuntimeError(
            "the active orbital selection changed discontinuously under a "
            "nuclear displacement; reduce the force step or give the "
            "active_space an explicit 'orbitals' list")
    return left @ right


def internal_directions(positions: np.ndarray,
                        tolerance: float = 1e-8) -> np.ndarray:
    r"""Orthonormal displacement directions that are not rigid motions.

    The energy of an isolated molecule does not change under a rigid
    translation or rotation, so its gradient is orthogonal to those six
    motions (five for a linear molecule, three for a single atom) and lives in
    the ``3N - 6`` dimensional complement.  Returns that complement as the
    columns of a ``(3N, k)`` matrix with orthonormal columns: the translations
    are :math:`\hat e_a` on every atom, the rotations
    :math:`\hat e_a \times (\mathbf R_i - \mathbf R_c)` about the
    centroid, and their rank -- five for collinear atoms -- is read off an
    SVD, so a linear molecule needs no special case.
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    relative = positions - positions.mean(axis=0)
    rigid = []
    for axis in range(3):
        translation = np.zeros((n, 3))
        translation[:, axis] = 1.0
        rigid.append(translation.ravel())
        unit = np.zeros(3)
        unit[axis] = 1.0
        rigid.append(np.cross(unit, relative).ravel())
    rigid = np.array(rigid)                                  # (6, 3N)
    _u, singular, vt = np.linalg.svd(rigid, full_matrices=True)
    rank = int((singular > tolerance * max(singular.max(), 1.0)).sum())
    return vt[rank:].T                                       # (3N, 3N - rank)


def active_space_gradient(atoms: Atoms, solver: VariationalDriver,
                          gamma: np.ndarray,
                          gamma2: np.ndarray,
                          step: float = ACTIVE_SPACE_STEP_ANGSTROM
                          ) -> np.ndarray:
    r"""Central-difference reduced-Hamiltonian gradient in eV/Angstrom.

    Each displaced build repeats the PAW integral, SCF, frozen-core, and active
    selector pipeline on the force calculation's fixed grid.  The RDMs belong
    to the *active* register before any frozen orbitals are refilled.  No VQE or
    hardware job is run at the displaced geometries.

    With a pseudopotential family only the internal directions are displaced
    (:func:`internal_directions`): the gradient of an isolated molecule has
    no component along a rigid translation or rotation, so ``3N - 6`` central
    differences (``3N - 5`` for a linear molecule) determine it instead of
    ``3N`` -- one, two builds, for a diatomic instead of twelve.  Each direction moves several atoms at
    once, by at most ``step / 2`` each, so no interatomic distance changes by
    more than ``step`` -- the bound a Cartesian step of one atom has, and so
    the same central-difference error: on O2 the two agree to 3e-6
    eV/Angstrom at equal bond change, while moving both atoms by the full
    ``step`` doubled the error (1e-2 eV/Angstrom).

    The shortcut rests on the invariance, which the fixed real-space grid
    breaks (the egg-box effect).  With a smooth pseudopotential and a
    filtered basis the break is negligible -- 1.7e-5 eV/Angstrom for O2 in
    PAW-LCAO at h = 0.15 -- and dropping it is what the calculator's
    translation projection does anyway.  A bare all-electron nucleus on the
    grid breaks it badly (all-electron LiH/6-31G at h = 0.35: moving Li put
    125 eV/Angstrom into the hydrogen's force), so an all-electron basis keeps
    the full ``3N`` Cartesian differences, each atom moved alone.
    """
    from ._hamiltonian_from_atoms import build_basis_hamiltonian

    if step <= 0:
        raise ValueError("the active-space force step must be positive")
    context = solver._gradient_context
    reference = context["integrals"]
    active = tuple(context["active"])
    positions = np.asarray(atoms.get_positions(), dtype=float)

    def energy(displacement: np.ndarray) -> float:
        shifted = atoms.copy()
        shifted.set_positions(positions + displacement)
        _hamiltonian, particles, n_orbitals, _profile, displaced = \
            build_basis_hamiltonian(
                shifted, solver.basis, reference.grid, solver.h,
                solver.charge, solver.n_electrons, spin=solver.spin,
                kinetic=solver.kinetic,
                active_space=solver._active_space_request(),
                ghosts=getattr(solver, "ghosts", ()))
        if (tuple(particles) != tuple(solver.num_particles)
                or n_orbitals != len(active)
                or tuple(displaced["frozen"]) != tuple(context["frozen"])):
            raise RuntimeError(
                "the displaced active problem changed its electron count, "
                "frozen orbitals, or register width")
        rotation = align_active_orbitals(
            reference, displaced["integrals"], active,
            tuple(displaced["active"]))
        return reduced_energy(displaced["integrals"], gamma, gamma2, rotation)

    directions = (internal_directions(positions)
                  if context.get("family") is not None
                  else np.eye(positions.size))
    gradient = np.zeros(positions.size)
    for column in directions.T:
        # No atom moves farther than step / 2, so no distance changes by more
        # than `step` (see the docstring).
        scale = 0.5 * step / np.linalg.norm(column.reshape(-1, 3), axis=1).max()
        displacement = (scale * column).reshape(positions.shape)
        slope = (energy(displacement) - energy(-displacement)) / (2 * scale)
        gradient += slope * column
    return (gradient * HARTREE_TO_EV).reshape(positions.shape)
