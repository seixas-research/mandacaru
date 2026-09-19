# -*- coding: utf-8 -*-
# file: test/test_all_electron_forces.py

# This code is part of Mandacaru.
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

from mandacaru import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
from mandacaru.algorithms.pseudo_forces import AlgebraicEnergy, spatial_rdms
from mandacaru.algorithms.rdm import (expand_frozen_core, one_rdm, two_rdm)
from mandacaru.integrals import Grid
from mandacaru.units import HARTREE_TO_EV


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
    """``(analytic forces, finite-difference forces)`` in eV/Angstrom.

    The analytic array is the **unprojected** one: the identity being checked is
    ``forces == -dE/dR`` of the *discretized* energy, and that energy is not
    translation invariant, so the projection ``project_translation`` applies by
    default deliberately breaks it (see ``ForceResult.unprojected``).
    """
    atoms = atoms.copy()
    atoms.calc = Mandacaru(method="adapt-vqe", basis=basis, grid=grid, h=h, profile=False,
                           gradient_tolerance=1e-6, **options)
    atoms.get_forces()
    forces = atoms.calc.force_result.unprojected
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


class TestOrbitalResponseResidual:
    """What ``details["orbital_gradient"]`` measures, and what moves it.

    The residual is ``max |dE/dkappa|`` with the density matrices frozen, and it
    vanishes at *both* ends of the correlation range: the Hartree-Fock
    determinant is orbital-stationary, and the exact ground state of an orbital
    space is invariant under any rotation of those orbitals.  So a nonzero value
    does not report an unconverged optimizer -- it reports an ansatz that
    stopped short of that exact state.  Measured on H2O / PAW-SZ: 1.2e-9 Ha for
    the HF determinant, 1.8e-9 for the sector FCI, 1.0e-2 for an ADAPT state
    5.4e-4 Ha above it, where tightening ``gradient_tolerance`` tenfold moved
    the energy by 1e-6 eV and the residual by 0.6 %.
    """

    @pytest.fixture(scope="class")
    def problem(self):
        """Integrals, sector and the HF / exact states of a 3-orbital system.

        A scalene H3+ triangle, deliberately: H2 in a minimal basis has two
        orbitals of opposite g/u parity, so its energy is *even* in the rotation
        angle and the first-order term vanishes by symmetry however unconverged
        the state is -- the test would pass on a residual that is identically
        zero for the wrong reason.
        """
        from scipy.sparse.linalg import eigsh

        from mandacaru.core.sector import ParticleSector

        grid = Grid(center=[4.0, 4.0, 4.0], box_size=4.5, h=0.3,
                    units="angstrom")
        atoms = Atoms("H3", positions=[[4.0, 4.0, 4.0], [4.9, 4.0, 4.0],
                                       [4.3, 4.8, 4.0]], cell=[8.0, 8.0, 8.0])
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", grid=grid, h=0.3,
                               charge=1, pool="fermionic", profile=False,
                               trace=False, gradient_tolerance=1e-8)
        atoms.get_potential_energy()
        forces = atoms.get_forces()
        solver = atoms.calc.solver
        sector = ParticleSector(solver.n_qubits, solver.num_particles,
                                solver.mapping)
        values, vectors = eigsh(sector.restrict(solver.hamiltonian), k=1,
                               which="SA")
        exact = sector.embed(vectors[:, 0])
        reference = solver.pool.occupied_orbitals
        hf = np.zeros(2 ** solver.n_qubits, dtype=complex)
        index = sum(1 << (solver.n_qubits - 1 - q)
                    for q in solver.ansatz.reference_qubits())
        hf[index] = 1.0
        assert reference is not None
        return atoms.calc, solver, exact, hf, float(values[0]), forces

    #: What "zero" means here: the residual of a state that is exactly at one
    #: of the two ends, limited by the finite-difference step of the rotation.
    ZERO = 1e-6

    @staticmethod
    def _residual(calc, solver, psi):
        from mandacaru.algorithms.pseudo_forces import pseudo_nuclear_gradient

        context = solver._gradient_context
        gamma = one_rdm(psi, solver.n_qubits, solver.mapping)
        gamma2 = two_rdm(psi, solver.n_qubits, solver.mapping)
        result = pseudo_nuclear_gradient(
            context["integrals"], gamma, gamma2,
            atom_of_orbital=context["atom_of_orbital"])
        return (result.details["orbital_gradient"],
                result.details["energy_hartree"], result.forces)

    def test_zero_for_the_hartree_fock_determinant(self, problem):
        calc, solver, _exact, hf, _e, _f = problem
        residual, _energy, _forces = self._residual(calc, solver, hf)
        assert residual < self.ZERO, residual

    def test_zero_for_the_exact_state_of_the_orbital_space(self, problem):
        calc, solver, exact, _hf, energy, _f = problem
        residual, rebuilt, _forces = self._residual(calc, solver, exact)
        assert rebuilt == pytest.approx(energy, abs=1e-9)
        # A full CI in this space is invariant under rotating its orbitals, so
        # the frozen-orbital gradient is the whole derivative there.
        assert residual < self.ZERO, residual

    def test_nonzero_between_the_two_ends(self, problem):
        calc, solver, exact, hf, energy, _f = problem
        # Neither end: the exact state contaminated with its own reference.
        mixed = exact + 0.1 * hf
        mixed /= np.linalg.norm(mixed)
        residual, rebuilt, _forces = self._residual(calc, solver, mixed)
        assert rebuilt > energy                      # variational, so above
        assert residual > 20 * self.ZERO             # and far off stationary

    def test_first_order_in_the_state_error_where_the_energy_is_second(
            self, problem):
        """Why the residual is the sharper diagnostic -- and why tightening the
        optimizer does so little for it.

        Contaminating the exact state by ``eps`` raises the energy by
        ``O(eps^2)`` (it is variational) and the orbital gradient by
        ``O(eps)``.  So an energy that has stopped moving to 1e-6 eV can sit
        beside a residual that is still 1e-2 Ha, which is exactly what the
        water run shows.
        """
        _calc, solver, exact, hf, energy, _f = problem
        calc = _calc
        residuals, errors = [], []
        for eps in (0.02, 0.04, 0.08):
            mixed = exact + eps * hf
            mixed /= np.linalg.norm(mixed)
            residual, rebuilt, _forces = self._residual(calc, solver, mixed)
            residuals.append(residual)
            errors.append(rebuilt - energy)

        # Quadrupling eps quadruples the residual and multiplies the energy
        # error by sixteen.
        assert residuals[2] / residuals[0] == pytest.approx(4.0, rel=0.25)
        assert errors[2] / errors[0] == pytest.approx(16.0, rel=0.35)

    def test_a_converged_run_stays_under_the_tolerance(self, problem):
        from mandacaru.algorithms.calculator import ORBITAL_RESPONSE_TOLERANCE

        calc, _solver, _exact, _hf, _e, _forces = problem
        # H2 in this basis *is* reachable, so the run warns about nothing.
        residual = calc.force_result.details["orbital_gradient"]
        assert residual < ORBITAL_RESPONSE_TOLERANCE
