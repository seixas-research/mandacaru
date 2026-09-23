# -*- coding: utf-8 -*-
# file: test/algorithms/test_periodic_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Forces and stress for a crystal: :mod:`mandacaru.algorithms.periodic_forces`.

The governing rule of every force in this repository is that it must be the
derivative of the energy that was actually reported, so the tests are almost
all comparisons against a finite difference of that same energy.

Two of them are exact identities rather than tolerances:

* **The Ewald forces sum to zero.**  Translational invariance of a lattice sum
  is exact, unlike the grid terms, so this holds to round-off.
* **A cubic-symmetric system has no shear stress.**  Which is how the even-node
  sampling artifact was found: the grid's nodes span ``[-n/2, n/2)`` modulo the
  lattice, a range that omits ``+n/2`` for even ``n``, so the sampling is not
  symmetric under ``r -> -r`` and a shear picks up a term odd in the strain
  that a central difference cannot cancel.  Odd node counts are clean.
"""

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms.periodic_forces import (_function_centers,
                                                  _ion_potential, _matrices,
                                                  _sample_stack,
                                                  periodic_nuclear_gradient)
from mandacaru.algorithms.pseudo_forces import AlgebraicEnergy, spatial_rdms
from mandacaru.core.ewald import ewald_energy, ewald_forces, ewald_stress
from mandacaru.optimizers import Optimizer
from mandacaru.units import to_bohr

SLSQP = Optimizer(method="SLSQP", maxiter=1000, tol=1e-12)

#: eV/Angstrom per Hartree/Bohr.
FORCE_CONVERSION = 27.211386245988 / 0.52917721092

CELLS = {
    "cubic": np.diag([5.4, 5.4, 5.4]),
    "hexagonal": np.array([[5.0, 0.0, 0.0], [-2.5, 4.330127, 0.0],
                           [0.0, 0.0, 6.0]]),
    "triclinic": np.array([[5.0, 0.0, 0.0], [1.2, 4.8, 0.0], [0.7, 0.5, 5.5]]),
}


class TestTheEwaldGradients:
    """The analytic half: no grid, so these can be exact."""

    @pytest.mark.parametrize("name", sorted(CELLS))
    def test_forces_match_finite_differences(self, name):
        cell = CELLS[name]
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.8, 1.2]])
        charges = np.array([3.0, -3.0])
        analytic = ewald_forces(positions, charges, cell, accuracy=1e-12)

        step = 1e-5
        numeric = np.zeros_like(positions)
        for atom in range(len(positions)):
            for k in range(3):
                plus = positions.copy()
                plus[atom, k] += step
                minus = positions.copy()
                minus[atom, k] -= step
                numeric[atom, k] = -(
                    ewald_energy(plus, charges, cell, accuracy=1e-12)
                    - ewald_energy(minus, charges, cell, accuracy=1e-12)
                ) / (2 * step)
        assert analytic == pytest.approx(numeric, abs=1e-8)

    def test_the_forces_sum_to_zero(self):
        """Translational invariance of a lattice sum, exactly."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5],
                              [1.0, 3.0, 2.0]])
        charges = np.array([1.0, 1.0, -2.0])
        forces = ewald_forces(positions, charges, np.diag([6.0] * 3))
        assert np.abs(forces.sum(axis=0)).max() < 1e-14

    @pytest.mark.parametrize("name", sorted(CELLS))
    def test_stress_matches_finite_differences(self, name):
        """The strain derivative, with ``eta`` held fixed on both sides."""
        cell = CELLS[name]
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.8, 1.2]])
        charges = np.array([2.0, -2.0])
        eta = 0.35
        analytic = ewald_stress(positions, charges, cell, accuracy=1e-13,
                                eta=eta)

        step = 1e-6
        volume = abs(float(np.linalg.det(cell)))
        numeric = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                epsilon = np.zeros((3, 3))
                epsilon[a, b] += step
                epsilon[b, a] += step
                plus = np.eye(3) + 0.5 * epsilon
                minus = np.eye(3) - 0.5 * epsilon
                numeric[a, b] = (
                    ewald_energy(positions @ plus.T, charges, cell @ plus.T,
                                 accuracy=1e-13, eta=eta)
                    - ewald_energy(positions @ minus.T, charges,
                                   cell @ minus.T, accuracy=1e-13, eta=eta)
                ) / (2 * step) / volume
        assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-11)

    def test_the_stress_is_symmetric(self):
        stress = ewald_stress(np.array([[0.0, 0.0, 0.0], [1.5, 0.8, 1.2]]),
                              np.array([2.0, -2.0]), CELLS["triclinic"])
        assert stress == pytest.approx(stress.T, abs=1e-18)

    def test_a_cubic_symmetric_system_has_no_ionic_shear(self):
        """One charge in a cube: the shear must vanish by symmetry."""
        stress = ewald_stress(np.array([[0.0, 0.0, 0.0]]), np.array([1.0]),
                              np.diag([5.66918] * 3), accuracy=1e-13)
        off = stress - np.diag(np.diag(stress))
        assert np.abs(off).max() < 1e-18
        diagonal = np.diag(stress)
        assert diagonal.max() - diagonal.min() < 1e-18


@pytest.fixture(scope="module")
def solved_chain():
    """A periodic H2 chain, solved once: 4 qubits, an asymmetric geometry."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.15, 0.0]],
                  cell=np.diag([3.0, 5.0, 5.0]), pbc=[True, False, False])
    # h = 0.35 and a 5 Angstrom transverse cell rather than 0.30 and 7: the
    # comparison is analytic-against-finite-difference of the *same* energy on
    # the *same* grid, so neither a coarser grid nor a tighter box weakens it,
    # and every one of the twelve displaced rebuilds gets cheaper.
    atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                           kpts={"size": (1, 1, 1), "gamma": True},
                           basis="HAO", h=0.35, trace=False, optimizer=SLSQP)
    atoms.get_potential_energy()
    return atoms


@pytest.fixture(scope="module")
def chain_gradient(solved_chain):
    """The periodic gradient, computed once: 12 displaced rebuilds each time."""
    solver = solved_chain.calc.solver
    context = solver._gradient_context
    gamma, gamma2 = solved_chain.calc._state_rdms(solver, two_body=True)
    return periodic_nuclear_gradient(
        context["integrals"], gamma, gamma2,
        atom_of_orbital=context["atom_of_orbital"])


class TestThePeriodicGradient:
    def test_it_is_the_derivative_of_the_reported_energy(self, solved_chain,
                                                         chain_gradient):
        """Against a central difference of the same fixed-state energy.

        Both the basis functions and the ion lattice are displaced together,
        which is the sum of the Pulay and Hellmann-Feynman halves.
        """
        solver = solved_chain.calc.solver
        context = solver._gradient_context
        integrals = context["integrals"]
        gamma, gamma2 = solved_chain.calc._state_rdms(solver, two_body=True)
        result = chain_gradient

        D, Gamma = spatial_rdms(gamma, gamma2, len(integrals.basis))
        energy = AlgebraicEnergy(integrals.mo_coefficients, D, Gamma)
        sites = np.array([to_bohr(p, integrals.units)
                          for _z, p in integrals.nuclei])
        charges = np.array([float(z) for z, _p in integrals.nuclei])
        centers = _function_centers(integrals)
        atom_of = np.asarray(context["atom_of_orbital"])

        def total(sites_now, centers_now):
            S, h, g = _matrices(integrals,
                                _sample_stack(integrals, centers_now),
                                _ion_potential(integrals, sites_now))
            return (float(np.real(energy(S, h, g)))
                    + float(ewald_energy(sites_now, charges, integrals.cell))
                    + integrals.constant_energy)

        step = 1e-4
        numeric = np.zeros((len(sites), 3))
        for atom in range(len(sites)):
            for k in range(3):
                shift = np.zeros(3)
                shift[k] = step
                plus_sites = sites.copy()
                plus_sites[atom] += shift
                plus_centers = centers.copy()
                plus_centers[atom_of == atom] += shift
                minus_sites = sites.copy()
                minus_sites[atom] -= shift
                minus_centers = centers.copy()
                minus_centers[atom_of == atom] -= shift
                numeric[atom, k] = -(total(plus_sites, plus_centers)
                                     - total(minus_sites, minus_centers)
                                     ) / (2 * step)
        assert result.forces == pytest.approx(numeric * FORCE_CONVERSION,
                                              abs=5e-4)

    def test_the_halves_add_to_the_total(self, chain_gradient):
        result = chain_gradient
        assert result.gradient == pytest.approx(
            result.hellmann_feynman + result.pulay, rel=1e-12)
        assert result.forces == pytest.approx(-result.gradient, rel=1e-12)

    def test_the_rebuilt_energy_matches_the_solver(self, solved_chain,
                                                   chain_gradient):
        """The check that the gradient differentiates the reported energy."""
        solver = solved_chain.calc.solver
        result = chain_gradient
        reported = float(solver._from_energy_units(
            solver.result.optimal_energy, "Ha"))
        assert result.details["energy_hartree"] == pytest.approx(reported,
                                                                 abs=1e-6)

    def test_the_madelung_term_carries_no_force(self, solved_chain,
                                               chain_gradient):
        """It depends on the cell alone, so it drops at fixed cell."""
        integrals = solved_chain.calc.solver._gradient_context["integrals"]
        assert integrals.constant_energy != 0.0        # it is in the energy
        # ... and it appears in the details, so its absence from the gradient
        # is a statement rather than an oversight.
        assert chain_gradient.details["madelung_energy"] == pytest.approx(
            integrals.constant_energy)


class TestForcesThroughASE:
    def test_the_periodic_driver_routes_to_the_periodic_gradient(self):
        """``periodic_hamiltonian`` is what sends ``_forces`` down the Ewald
        branch, so a driver that lost it would silently get the molecular
        gradient -- which differentiates -Z/r potentials this Hamiltonian was
        never built from."""
        from mandacaru.algorithms.bloch import _bloch_drivers

        for cls in _bloch_drivers().values():
            assert cls.periodic_hamiltonian is True

    def test_the_gradient_has_a_row_per_atom(self, chain_gradient):
        """The ASE round trip itself is pinned in
        ``test_bloch.TestForcesAreImplemented``, cheaply; calling
        ``get_forces()`` here would buy the same facts for a second full force
        run."""
        assert chain_gradient.forces.shape == (2, 3)
        assert np.all(np.isfinite(chain_gradient.forces))

    def test_the_electron_count_is_checked(self, chain_gradient):
        assert chain_gradient.n_electrons == pytest.approx(2.0)

    def test_the_periodic_grid_is_not_replaced_by_a_bounding_box(self,
                                                                 solved_chain):
        """A frozen molecular grid would not even span the cell.

        The periodic grid is the cell, built commensurate with it, so it is
        already the same at every geometry and must not be swapped for the
        bounding-box grid the molecular force path freezes.
        """
        integrals = solved_chain.calc.solver._gradient_context["integrals"]
        grid = integrals.grid
        spanned = np.asarray(grid.step) @ np.diag(grid.shape)
        assert spanned == pytest.approx(integrals.cell.T, abs=1e-10)


class TestTheStress:
    @pytest.fixture(scope="class")
    def cubic_atom(self):
        """One hydrogen in a cube on an ODD grid: cubic symmetry, clean shear."""
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=np.diag([3.0] * 3), pbc=True)
        atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                               kpts={"size": (1, 1, 1), "gamma": True},
                               basis="HAO", h=0.35, trace=False,
                               optimizer=SLSQP)
        atoms.get_potential_energy()
        return atoms

    @pytest.fixture(scope="class")
    def cubic_tensor(self, cubic_atom):
        """Twelve Hamiltonian rebuilds; once is enough for every assertion."""
        return cubic_atom.calc.get_stress(voigt=False)

    def test_a_cubic_system_is_isotropic_and_shear_free(self, cubic_atom,
                                                        cubic_tensor):
        """Cubic symmetry forces it, so this is an identity, not a tolerance."""
        assert cubic_atom.calc.solver._gradient_context[
            "integrals"].grid.shape == (9, 9, 9)
        stress = cubic_tensor
        diagonal = np.diag(stress)
        assert diagonal.max() - diagonal.min() < 1e-10
        assert np.abs(stress - np.diag(diagonal)).max() < 1e-10

    def test_it_is_symmetric(self, cubic_tensor):
        stress = cubic_tensor
        assert stress == pytest.approx(stress.T, abs=1e-14)

    def test_the_voigt_order_is_ase_s(self, cubic_atom, cubic_tensor):
        tensor = cubic_tensor
        voigt = cubic_atom.calc.get_stress()
        assert voigt.shape == (6,)
        expected = [tensor[0, 0], tensor[1, 1], tensor[2, 2],
                    tensor[1, 2], tensor[0, 2], tensor[0, 1]]
        assert voigt == pytest.approx(expected)

    def test_an_even_grid_is_shear_free_too(self):
        h, shape = 0.30, 10        # the grid with the largest original artifact
        """The regression: the shear used to depend on the grid's parity.

        Before ``fft_g_squared`` symmetrized the Nyquist plane, this system --
        where cubic symmetry forces the shear to vanish -- gave 6.4e-4
        eV/Angstrom^3 on a 10x10x10 grid and 2.6e-4 on 12x12x12, against 1e-13
        on the odd 9x9x9 and 15x15x15.  Both are now at round-off.
        """
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                      cell=np.diag([3.0] * 3), pbc=True)
        atoms.calc = Mandacaru(method="bloch-adapt-vqe",
                               kpts={"size": (1, 1, 1), "gamma": True},
                               basis="HAO", h=h, trace=False,
                               optimizer=SLSQP)
        atoms.get_potential_energy()
        assert atoms.calc.solver._gradient_context[
            "integrals"].grid.shape == (shape,) * 3
        stress = atoms.calc.get_stress(voigt=False)
        off = stress - np.diag(np.diag(stress))
        assert np.abs(off).max() < 1e-10

    def test_a_molecular_method_has_no_stress(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]],
                      cell=np.diag([8.0] * 3))
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", h=0.4,
                               trace=False, optimizer=SLSQP)
        atoms.get_potential_energy()
        with pytest.raises(NotImplementedError, match="not periodic"):
            atoms.calc.get_stress()
