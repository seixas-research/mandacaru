# -*- coding: utf-8 -*-
# file: algorithms/periodic_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""Hellmann-Feynman, Pulay and stress for a crystal.

The molecular gradient of :mod:`mandacaru.algorithms.pseudo_forces` cannot be
pointed at a periodic cell, because two of its terms are different objects
there: the electron feels the **Ewald potential of the whole ion lattice**
rather than a sum of :math:`-Z/r`, and the ion-ion energy is the **Ewald
energy** rather than a finite pair sum.  Everything else -- the algebraic
contraction of the RDMs against ``(S, h, g)``, the displaced-sampling
derivatives, the orbital-response residual -- is shared and reused from there.

What the energy is
------------------

.. math::

    E = \sum_{pq} D_{pq} h^{\text{MO}}_{pq}
      + \tfrac12 \sum_{pqrs} \Gamma_{pqrs} g^{\text{MO}}_{pqrs}
      + E_{\text{Ewald}} + \tfrac12 N_e v_{\text{M}} ,

and the force is its derivative, term by term:

**Hellmann-Feynman** -- the operators move while the basis stays put.  Two
pieces: the external potential's ion lattice (its Ewald potential is rebuilt
with one ion displaced) and the ion-ion Ewald energy, whose gradient is
analytic (:func:`~mandacaru.core.ewald.ewald_forces`).

**Pulay** -- the basis functions move while the operators stay put.  The
functions of the displaced atom are re-sampled, *including their periodic
images* (moving an atom moves every image of it), and ``S``, ``T``, ``V`` and
the two-electron tensor are rebuilt from the displaced stack.

**The Madelung constant contributes nothing to the force.**  It is
:math:`\tfrac12 N_e v_{\text{M}}`, a function of the cell alone, so it drops
out of a fixed-cell derivative -- and reappears in the stress, where the cell
is what is being varied.

The stress
----------

:meth:`stress` differentiates the same total energy with respect to a
**symmetric strain applied to the cell, the atoms and the grid together**.
Straining the grid with the cell is what keeps the derivative meaningful: the
alternative -- straining the cell over a frozen grid -- changes the
discretization as well as the physics, and the two are not separable.  The
grid's node *count* is held fixed so the strained grid stays commensurate with
the strained cell, which is the condition
:class:`~mandacaru.core.periodic.PeriodicIntegrals` checks and Bloch's theorem
needs.

This makes the stress a finite difference of full Hamiltonian builds -- six of
them for the six independent strain components -- which is expensive but is the
derivative of the energy that was actually reported.  The Ewald part is
available analytically (:func:`~mandacaru.core.ewald.ewald_stress`), and is
shear-free to ``3e-20`` for a cubic-symmetric system, so it can be used to
isolate the electronic remainder.

**The shear components used to depend on the grid's parity**, and no longer
do.  Measured on a single hydrogen in a cubic cell, where cubic symmetry forces
the shear to vanish exactly, the stress came out at ``6.4e-4`` eV/Angstrom^3 on
a ``10x10x10`` grid and ``2.6e-4`` on ``12x12x12``, against ``1e-13`` on the
odd ``9x9x9`` and ``15x15x15``.  The cause was the Nyquist mode of the FFT
mesh, not the real-space sampling: see
:func:`~mandacaru.integrals.poisson.fft_g_squared`, which symmetrizes it.
Every grid now gives ``1e-13``, and because the symmetrization is a no-op for
an orthogonal cell, no energy moved.
"""

from __future__ import annotations

import numpy as np

from ..core.ewald import ewald_forces
from ..units import from_hartree
from .forces import ForceResult
from .pseudo_forces import (DEFAULT_ALGEBRAIC_STEP, AlgebraicEnergy,
                            spatial_rdms)

__all__ = ["periodic_nuclear_gradient", "periodic_stress",
           "strained_energy", "DEFAULT_ORBITAL_DELTA", "DEFAULT_STRAIN"]

#: Displacement of the sampled basis functions, in Bohr.  The same value the
#: molecular path uses; small enough that the central difference is in its
#: quadratic regime and large enough to stay above grid round-off.
DEFAULT_ORBITAL_DELTA = 1e-3

#: Symmetric strain used for the stress finite difference.  Each component
#: costs two full Hamiltonian builds, so this is a genuine cost knob.
DEFAULT_STRAIN = 1e-3

#: Voigt order ASE expects from ``get_stress``.
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _sample_stack(integrals, positions):
    """The basis stack with the atoms at ``positions`` (Bohr).

    Only the *functions* move: the grid, the cell and the image sum are the
    engine's own.  Periodic sampling is used when the engine is periodic, so a
    displaced atom drags its whole lattice of images with it -- which is the
    physical statement that there is one atom per cell, not one atom plus a
    frozen halo.
    """
    engine = integrals._engine
    moved = []
    for function, center in zip(integrals.basis, positions):
        shifted = _shift_function(function, center)
        moved.append(engine._sample_periodic(shifted) if engine.periodic
                     else shifted.sample(engine.grid))
    return np.ascontiguousarray(np.stack(moved), dtype=np.complex128)


def _shift_function(function, center):
    """A shallow copy of ``function`` centered at ``center`` (Bohr)."""
    import copy

    shifted = copy.copy(function)
    shifted.center = np.asarray(center, dtype=float)
    return shifted


def _function_centers(integrals):
    """Where each basis function sits, in Bohr."""
    return np.array([np.asarray(function.center, dtype=float)
                     for function in integrals.basis])


def _matrices(integrals, stack, external):
    """``(S, h, g)`` from a sampled stack and an external potential."""
    engine = integrals._engine
    original = engine._psi
    try:
        engine._psi = stack
        T, V = engine.one_body(external, energy_units="Ha",
                              kinetic=integrals.kinetic)
        overlap = (np.conj(stack) @ stack.T) * integrals.grid.dV
        overlap = 0.5 * (overlap + overlap.conj().T)
        g = engine.two_body(method="fft", energy_units="Ha")
    finally:
        engine._psi = original
    return overlap, 0.5 * ((T + V) + (T + V).conj().T), g


def _ion_potential(integrals, sites):
    """The engine-facing Ewald potential of ions at ``sites`` (Bohr)."""
    from ..core.ewald import ewald_potential

    charges = np.array([-float(charge) for charge, _p in integrals.nuclei])

    def potential(x, y, z):
        points = np.stack([np.asarray(x, dtype=float).ravel(),
                           np.asarray(y, dtype=float).ravel(),
                           np.asarray(z, dtype=float).ravel()], axis=1)
        values = ewald_potential(sites, charges, integrals.cell, points,
                                 softening=integrals._potentials.softening)
        return values.reshape(np.asarray(x).shape)

    return potential


def periodic_nuclear_gradient(integrals, gamma, gamma2, *, atom_of_orbital,
                              orbital_delta: float | None = None,
                              include_pulay: bool = True,
                              algebraic_step: float = DEFAULT_ALGEBRAIC_STEP,
                              orbital_gradient: bool = True) -> ForceResult:
    r"""Hellmann-Feynman + Pulay gradient of a periodic calculation.

    Parameters
    ----------
    integrals : PeriodicIntegrals
        The engine the Hamiltonian was built from, after
        ``molecular_hamiltonian(mo_basis=True)``.
    gamma, gamma2 : ndarray
        Spin-orbital RDMs of the converged state, over all orbitals.
    atom_of_orbital : sequence of int
        Which atom each basis function belongs to.
    orbital_delta : float, optional
        Displacement of the sampled functions, in Bohr; default
        :data:`DEFAULT_ORBITAL_DELTA`.
    include_pulay : bool
        Compute the Pulay half.  ``False`` leaves the Hellmann-Feynman
        gradient alone, which is not the derivative of anything and is for
        diagnosis only.
    algebraic_step : float
        Step of the directional derivative of the algebraic energy.
    orbital_gradient : bool
        Measure the orbital-rotation residual.

    Returns
    -------
    ForceResult
        Forces in eV/Angstrom, with the two halves and the rebuilt energy in
        ``details``.
    """
    atom_of_orbital = np.asarray(atom_of_orbital)
    n_orbitals = len(integrals.basis)
    D, Gamma = spatial_rdms(gamma, gamma2, n_orbitals)
    energy = AlgebraicEnergy(integrals.mo_coefficients, D, Gamma)

    from ..units import to_bohr

    sites = np.array([to_bohr(position, integrals.units)
                      for _charge, position in integrals.nuclei])
    charges = np.array([float(charge) for charge, _p in integrals.nuclei])
    centers = _function_centers(integrals)

    stack0 = integrals._engine._psi
    S0, h0, g0 = _matrices(integrals, stack0, _ion_potential(integrals, sites))
    electronic = energy(S0, h0, g0)
    total = electronic + integrals.nuclear_repulsion + integrals.constant_energy

    n_atoms = len(sites)
    hf = np.zeros((n_atoms, 3))
    pulay = np.zeros((n_atoms, 3))
    # `orbital_delta` is None on the driver unless a user set it.
    delta = float(DEFAULT_ORBITAL_DELTA if orbital_delta is None
                  else orbital_delta)
    zero_g = np.zeros_like(g0)

    for atom in range(n_atoms):
        own = atom_of_orbital == atom
        for k in range(3):
            step = np.zeros(3)
            step[k] = delta

            # -- Hellmann-Feynman: the ion lattice moves, the basis does not.
            plus = sites.copy()
            plus[atom] += step
            minus = sites.copy()
            minus[atom] -= step
            _S, h_plus, _g = _matrices(integrals, stack0,
                                       _ion_potential(integrals, plus))
            _S, h_minus, _g = _matrices(integrals, stack0,
                                        _ion_potential(integrals, minus))
            dh = (h_plus - h_minus) / (2.0 * delta)
            hf[atom, k] = energy.directional(S0, h0, g0,
                                             np.zeros_like(S0), dh, zero_g,
                                             step=algebraic_step)

            if not include_pulay:
                continue

            # -- Pulay: the basis functions move, the operators do not.
            moved_plus = centers.copy()
            moved_plus[own] += step
            moved_minus = centers.copy()
            moved_minus[own] -= step
            potential = _ion_potential(integrals, sites)
            S_p, h_p, g_p = _matrices(
                integrals, _sample_stack(integrals, moved_plus), potential)
            S_m, h_m, g_m = _matrices(
                integrals, _sample_stack(integrals, moved_minus), potential)
            pulay[atom, k] = energy.directional(
                S0, h0, g0,
                (S_p - S_m) / (2.0 * delta),
                (h_p - h_m) / (2.0 * delta),
                (g_p - g_m) / (2.0 * delta),
                step=algebraic_step)

    # The ion-ion Ewald energy is analytic; the Madelung constant depends only
    # on the cell and so contributes nothing at fixed cell.
    ion_gradient = -ewald_forces(sites, charges, integrals.cell)
    hf = hf + ion_gradient

    gradient = hf + pulay
    forces_hartree_bohr = -gradient
    conversion = from_hartree(1.0, "eV") / 0.52917721092
    forces = forces_hartree_bohr * conversion

    residual = None
    if orbital_gradient:
        residual = energy.orbital_gradient(S0, h0, g0)

    # ForceResult keeps the *gradient* convention (+dE/dR) for the component
    # breakdown and the *force* convention (-dE/dR) for `forces`.
    details = {
        "energy_hartree": float(np.real(total)),
        "electronic_energy": float(np.real(electronic)),
        "ewald_energy": float(integrals.nuclear_repulsion),
        "madelung_energy": float(integrals.constant_energy),
        "ion_gradient": ion_gradient * conversion,
        "orbital_gradient": residual,
        "forces_unprojected": forces,
        "translational_residual": float(np.abs(forces.sum(axis=0)).max()),
        "periodic": True,
    }
    return ForceResult(forces=forces,
                       hellmann_feynman=hf * conversion,
                       pulay=pulay * conversion,
                       gradient=gradient * conversion,
                       n_electrons=float(np.real(np.trace(D))),
                       details=details)


def strained_energy(integrals, D, Gamma, mo_coefficients, strain_matrix,
                    *, n_electrons):
    r"""The frozen-state total energy with cell, atoms and grid all strained.

    Rebuilds the integrals on a grid whose **node counts are unchanged** -- the
    per-axis spacing is set to ``|a_m| / n_m`` of the strained cell, so
    ``round(L/h)`` cannot flip between the plus and minus builds and turn the
    finite difference into noise.  The state is held fixed: the RDMs and the
    molecular orbitals are the converged ones, exactly as the force does, and
    the orbital-response residual measured there covers both.
    """
    from ..core.periodic import PeriodicIntegrals
    from ..integrals import Grid
    from ..units import to_bohr

    strain_matrix = np.asarray(strain_matrix, dtype=float)
    cell = np.asarray(integrals.cell, dtype=float) @ strain_matrix.T
    shape = tuple(integrals.grid.shape)
    lengths = np.linalg.norm(cell, axis=1)
    spacing = lengths / np.array(shape, dtype=float)

    nuclei = []
    for charge, position in integrals.nuclei:
        moved = to_bohr(position, integrals.units) @ strain_matrix.T
        nuclei.append((charge, moved))
    basis = [_shift_function(function,
                             np.asarray(function.center, float) @ strain_matrix.T)
             for function in integrals.basis]

    grid = Grid(center=0.5 * cell.sum(axis=0), box_size=0.0, h=spacing,
                units="bohr", cell=cell, periodic=True)
    strained = PeriodicIntegrals(
        nuclei, basis, grid, cell, n_electrons=n_electrons, units="bohr",
        softening=integrals._potentials.softening, kinetic=integrals.kinetic)

    sites = np.array([np.asarray(position, float) for _z, position in nuclei])
    S, h, g = _matrices(strained, strained._engine._psi,
                        _ion_potential(strained, sites))
    energy = AlgebraicEnergy(mo_coefficients, D, Gamma)
    return (float(np.real(energy(S, h, g)))
            + float(strained.nuclear_repulsion)
            + float(strained.constant_energy))


def periodic_stress(build, cell, *, strain: float = DEFAULT_STRAIN):
    r"""Stress tensor by symmetric strain of cell, atoms and grid together.

    Parameters
    ----------
    build : callable
        ``build(strain_matrix) -> energy in Hartree``, where ``strain_matrix``
        is ``1 + epsilon``.  It must rebuild the whole Hamiltonian on a grid
        commensurate with the strained cell and return the **supercell**
        energy; the caller owns that, because only it knows the driver's
        options.  :func:`strained_energy` is the implementation the periodic
        driver uses.
    cell : (3, 3) array_like
        The unstrained lattice vectors as rows, in Bohr.
    strain : float
        Half-amplitude of the symmetric strain.

    Returns
    -------
    (3, 3) ndarray
        :math:`\sigma_{\alpha\beta} = \frac{1}{\Omega}
        \partial E / \partial \varepsilon_{\alpha\beta}`, Hartree/Bohr^3,
        symmetrized.

    Notes
    -----
    Six builds either side of zero, one per independent component.  The strain
    is applied as :math:`(1 + \varepsilon)` with :math:`\varepsilon`
    symmetric, so the diagonal components are volume-changing and the
    off-diagonal ones are shears.
    """
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    volume = abs(float(np.linalg.det(cell)))
    stress = np.zeros((3, 3))
    for alpha, beta in VOIGT:
        epsilon = np.zeros((3, 3))
        epsilon[alpha, beta] += strain
        epsilon[beta, alpha] += strain
        plus = build(np.eye(3) + 0.5 * epsilon)
        minus = build(np.eye(3) - 0.5 * epsilon)
        value = (plus - minus) / (2.0 * strain) / volume
        stress[alpha, beta] = value
        stress[beta, alpha] = value
    return stress
