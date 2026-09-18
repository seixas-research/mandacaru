# -*- coding: utf-8 -*-
# file: test/test_all_electron_forces.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""All-electron forces are the gradient of the energy the solver reported.

The default ``force_method="rdm"`` differentiates the same energy expression
for every atom-centered basis: the reduced density matrices and molecular
orbitals are held fixed, the atom's local potential is the bare ``-Z/r``, and a
frozen core is put back into the density matrices before contracting.  Each
case here is checked against a central difference of that *same* energy on the
same frozen grid -- the legacy differentiated-SCF path was off by ~8x on a
basis with p orbitals.
"""

import numpy as np
import pytest
from ase import Atoms

from carcara import Carcara
from carcara.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
from carcara.algorithms.pseudo_forces import AlgebraicEnergy, spatial_rdms
from carcara.algorithms.rdm import (expand_frozen_core, one_rdm, two_rdm)
from carcara.integrals import Grid
from carcara.units import HARTREE_TO_EV


def fixed_state_energy(atoms, basis, grid, h, algebra, **options):
    """``E(S, h, g)`` at this geometry with the RDMs and orbitals frozen."""
    integrals = build_basis_hamiltonian(atoms, basis, grid, h, 0, None,
                                        **options)[4]["integrals"]
    external = integrals.external_potential()
    kinetic, potential = integrals._engine.one_body(external, energy_units="Ha",
                                                    kinetic="fd")
    one = kinetic + potential + integrals.kb_nonlocal()
    one_body = integrals.one_body_augmentation()
    if one_body is not None:
        one = one + np.asarray(one_body)
    two = integrals._engine.two_body(method="fft", energy_units="Ha")
    augmentation = integrals.two_body_augmentation()
    if augmentation is not None:
        two = two + np.asarray(augmentation)
    return (algebra(integrals.overlap(), 0.5 * (one + one.conj().T), two)
            + integrals.nuclear_repulsion + integrals.constant_energy)


def analytic_and_numerical(atoms, basis, grid, h, step=0.002, **options):
    """``(analytic forces, finite-difference forces)`` in eV/Angstrom."""
    atoms = atoms.copy()
    atoms.calc = Carcara(method="adapt-vqe", basis=basis, grid=grid, h=h,
                         verbose=False, profile=False,
                         gradient_tolerance=1e-6, **options)
    forces = atoms.get_forces()
    solver = atoms.calc.solver
    context = solver._gradient_context
    integrals = context["integrals"]
    psi = atoms.calc._converged_state(solver)
    sector = getattr(solver, "_sector", None)
    gamma = one_rdm(psi, solver.n_qubits, solver.mapping, sector=sector)
    gamma2 = two_rdm(psi, solver.n_qubits, solver.mapping, sector=sector)
    if context.get("frozen"):
        gamma, gamma2 = expand_frozen_core(gamma, gamma2, context["frozen"],
                                           len(integrals.basis))
    algebra = AlgebraicEnergy(integrals.mo_coefficients,
                              *spatial_rdms(gamma, gamma2,
                                            integrals.n_orbitals))
    build = {k: v for k, v in options.items() if k == "frozen_core"}

    numerical = np.zeros_like(forces)
    for atom in range(len(atoms)):
        for axis in range(3):
            shifted = []
            for sign in (+1, -1):
                moved = atoms.copy()
                moved.positions[atom, axis] += sign * step
                shifted.append(fixed_state_energy(moved, basis, grid, h,
                                                  algebra, **build))
            numerical[atom, axis] = (-(shifted[0] - shifted[1]) / (2 * step)
                                     * HARTREE_TO_EV)
    return forces, numerical


class TestAllElectronGradient:
    def test_h2_minimal_basis(self):
        grid = Grid(center=[4.0, 4.0, 4.0], box_size=4.0, h=0.25,
                    units="angstrom")
        atoms = Atoms("H2", positions=[[4.013, 3.979, 3.607],
                                       [4.013, 3.979, 4.407]], cell=[8.0] * 3)
        forces, numerical = analytic_and_numerical(atoms, "FAO", grid, 0.25)
        assert np.abs(forces - numerical).max() < 1e-2
        assert np.abs(numerical).max() > 1.0        # a real force, not noise

    def test_lih_with_a_frozen_core(self):
        """The inert core is refilled before the density matrices are used."""
        grid = Grid(center=[4.0, 4.0, 4.0], box_size=4.0, h=0.45,
                    units="angstrom")
        atoms = Atoms("LiH", positions=[[4.013, 3.979, 3.207],
                                        [4.013, 3.979, 4.807]], cell=[8.0] * 3)
        forces, numerical = analytic_and_numerical(atoms, "FAO", grid, 0.45,
                                                    frozen_core=True,
                                                    max_iterations=6)
        scale = np.abs(numerical).max()
        assert np.abs(forces - numerical).max() < 1e-3 * scale

    def test_frozen_core_expansion_matches_an_explicit_determinant(self):
        """Core-core, core-active and active-active blocks, exactly."""
        def determinant(n_modes, occupied):
            psi = np.zeros(2 ** n_modes, dtype=complex)
            index = 0
            for qubit in occupied:
                index |= 1 << (n_modes - 1 - qubit)
            psi[index] = 1.0
            return psi

        full = determinant(6, [0, 1, 3, 4])          # spatial 0, 1 filled
        active = determinant(4, [0, 2])              # spatial 1 filled
        expanded = expand_frozen_core(
            one_rdm(active, 4, "jordan_wigner"),
            two_rdm(active, 4, "jordan_wigner"), (0,), 3)
        assert np.allclose(expanded[0], one_rdm(full, 6, "jordan_wigner"),
                           atol=1e-12)
        assert np.allclose(expanded[1], two_rdm(full, 6, "jordan_wigner"),
                           atol=1e-12)

    def test_an_unfrozen_run_is_untouched(self):
        gamma = np.eye(4, dtype=complex)
        gamma2 = np.zeros((4,) * 4, dtype=complex)
        assert expand_frozen_core(gamma, gamma2, (), 2)[0] is gamma

    def test_a_wrong_active_size_is_refused(self):
        """Four spin-orbitals cannot be the active block of 4 - 1 spatials."""
        with pytest.raises(ValueError, match="active RDMs over 6"):
            expand_frozen_core(np.eye(4, dtype=complex),
                               np.zeros((4,) * 4, dtype=complex), (0,), 4)
