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
            "nuclear displacement; reduce the force step or use an explicit "
            "active_orbitals list")
    return left @ right


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
    """
    from ._hamiltonian_from_atoms import build_basis_hamiltonian

    if step <= 0:
        raise ValueError("the active-space force step must be positive")
    context = solver._gradient_context
    reference = context["integrals"]
    active = tuple(context["active"])
    positions = np.asarray(atoms.get_positions(), dtype=float)
    gradient = np.zeros_like(positions)

    def energy(atom: int, axis: int, sign: int) -> float:
        shifted = atoms.copy()
        coords = positions.copy()
        coords[atom, axis] += sign * step
        shifted.set_positions(coords)
        _hamiltonian, particles, n_orbitals, _profile, displaced = \
            build_basis_hamiltonian(
                shifted, solver.basis, reference.grid, solver.h,
                solver.charge, solver.n_electrons, spin=solver.spin,
                frozen_core=solver.frozen_core,
                frozen_orbitals=solver.frozen_orbitals,
                kinetic=solver.kinetic,
                active_orbitals=solver.active_orbitals,
                active_selection=solver.active_selection,
                active_threshold=solver.active_threshold)
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

    for atom in range(len(positions)):
        for axis in range(3):
            gradient[atom, axis] = (
                energy(atom, axis, +1) - energy(atom, axis, -1)) \
                * HARTREE_TO_EV / (2 * step)
    return gradient
