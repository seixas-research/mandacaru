# -*- coding: utf-8 -*-
# file: test/test_open_shell.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Open-shell (odd-electron) systems in the UHF natural-orbital basis.

An odd electron count has no closed-shell RHF basis.  The molecular Hamiltonian
is then written in the natural orbitals of the UHF total density -- one spatial
basis for both spins -- with the doublet (or a higher spin state from the
magnetic moments) as the reference.  These tests pin the solver, the basis, the
geometry path (defaults, frozen core, quartets, the plane-wave refusal) and
that the variational drivers reach the sector's exact ground state.
"""

import numpy as np
from mandacaru.units import HARTREE_TO_EV
import pytest
from ase import Atoms

from mandacaru.algorithms import (estimate_qubits, Mandacaru, natural_orbitals,
                                  RHF, UHF)
from mandacaru.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
from mandacaru.core import MolecularIntegrals, minimal_fao_basis
from mandacaru.integrals import Grid
from mandacaru.optimizers import Optimizer

# The classical optimizers used below, with the iteration budget and
# the convergence tolerance written out rather than left to the
# library default: a test that pins an energy should say what it was
# optimized with.
LBFGSB = Optimizer(method="L-BFGS-B", maxiter=2000, tol=1e-12)


def _integrals(nuclei, h=0.30, box=6.0):
    grid = Grid(center=np.mean([p for _z, p in nuclei], axis=0), box_size=box,
                h=h)
    return MolecularIntegrals(nuclei, minimal_fao_basis(nuclei), grid,
                              softening=0.5 * min(grid.dx, grid.dy, grid.dz))


def _sector_ground_state(hamiltonian, num_particles, mapping="jordan_wigner"):
    """Lowest eigenvalue of the qubit Hamiltonian in the (n_alpha, n_beta) sector.

    Accepts a ``Fermion`` (mapped here) or an already mapped ``PauliSum``.
    """
    if hasattr(hamiltonian, "map_to_qubits"):
        hamiltonian = hamiltonian.map_to_qubits(mapping)
    h = hamiltonian.to_matrix()
    h = 0.5 * (h + h.conj().T)
    n = int(round(np.log2(h.shape[0])))
    M = n // 2
    keep = []
    for index in range(h.shape[0]):
        bits = [(index >> (n - 1 - q)) & 1 for q in range(n)]   # qubit 0 first
        if (sum(bits[:M]), sum(bits[M:])) == tuple(num_particles):
            keep.append(index)
    sub = h[np.ix_(keep, keep)]
    return float(np.linalg.eigvalsh(sub).min())


@pytest.fixture(scope="module")
def h_atom():
    return _integrals([(1.0, np.array([0.0, 0.0, 0.0]))], h=0.25, box=5.0)


@pytest.fixture(scope="module")
def h3():
    """Linear H3, three electrons: a doublet with 6 qubits in FAO."""
    R = 0.9
    return _integrals([(1.0, np.array([0.0, 0.0, -R])),
                       (1.0, np.array([0.0, 0.0, 0.0])),
                       (1.0, np.array([0.0, 0.0, R]))], h=0.30, box=6.5)


# --------------------------------------------------------------------------- #
# The UHF solver and its natural-orbital basis.
# --------------------------------------------------------------------------- #

class TestUHFSolve:
    def test_hydrogen_atom(self, h_atom):
        result = UHF(h_atom.one_body(), h_atom.two_body(), 1, 0).solve()
        assert result.converged and (result.n_alpha, result.n_beta) == (1, 0)
        assert result.electronic_energy == pytest.approx(-0.5, abs=0.03)
        assert result.natural_occupations[0] == pytest.approx(1.0)
        assert result.reference_energy == pytest.approx(result.electronic_energy,
                                                        abs=1e-8)
        assert result.spin_contamination == pytest.approx(0.0)

    def test_run_still_returns_the_energy(self, h_atom):
        solver = UHF(h_atom.one_body(), h_atom.two_body(), 1, 0)
        assert solver.run() == pytest.approx(solver.solve().electronic_energy)

    def test_natural_orbitals_are_orthonormal_and_ordered(self, h3):
        result = UHF(h3.one_body(), h3.two_body(), 2, 1).solve()
        C = result.natural_orbitals
        assert np.allclose(C.conj().T @ C, np.eye(C.shape[0]), atol=1e-10)
        occ = result.natural_occupations
        assert np.all(np.diff(occ) <= 1e-10)                   # descending
        assert occ.sum() == pytest.approx(3.0)
        assert np.all(occ >= -1e-10) and np.all(occ <= 2.0 + 1e-10)

    def test_natural_orbital_reference_is_above_uhf(self, h3):
        # The NO determinant is variational but not the UHF minimum.
        result = UHF(h3.one_body(), h3.two_body(), 2, 1).solve()
        assert result.reference_energy >= result.electronic_energy - 1e-10

    def test_closed_shell_reduces_to_rhf(self, h3):
        # Two electrons in H3+: UHF(1,1) and RHF(2) agree.
        uhf = UHF(h3.one_body(), h3.two_body(), 1, 1).solve()
        rhf = RHF(h3.one_body(), h3.two_body(), 2).run()
        assert uhf.electronic_energy == pytest.approx(rhf.electronic_energy,
                                                      abs=1e-6)

    def test_natural_orbitals_helper(self):
        D = np.diag([1.7, 0.3, 2.0])
        occ, C = natural_orbitals(D)
        assert np.allclose(occ, [2.0, 1.7, 0.3])
        assert np.allclose(np.abs(C[:, 0]), [0, 0, 1])


# --------------------------------------------------------------------------- #
# The open-shell molecular Hamiltonian.
# --------------------------------------------------------------------------- #

class TestOpenShellHamiltonian:
    def test_odd_count_builds_in_the_no_basis(self, h3):
        H = h3.molecular_hamiltonian(mo_basis=True, n_electrons=3)
        assert H.n_modes() == 6
        # The NO reference determinant energy is what the Hamiltonian gives
        # for the (2, 1) aufbau state.
        uhf = h3.open_shell_hartree_fock(2, 1)
        exact = _sector_ground_state(H, (2, 1))
        assert exact <= uhf.reference_energy + h3.nuclear_repulsion + 1e-9
        assert exact <= uhf.electronic_energy + h3.nuclear_repulsion + 1e-9

    def test_num_particles_must_match(self, h3):
        with pytest.raises(ValueError, match="does not sum"):
            h3.molecular_hamiltonian(mo_basis=True, n_electrons=3,
                                     num_particles=(2, 2))

    def test_rhf_can_be_forced_only_for_even_counts(self, h3):
        with pytest.raises(ValueError, match="even electron count"):
            h3.molecular_hamiltonian(mo_basis=True, n_electrons=3,
                                     open_shell=False)
        even = h3.molecular_hamiltonian(mo_basis=True, n_electrons=2,
                                        open_shell=True)
        assert even.n_modes() == 6

    def test_even_count_keeps_the_rhf_basis_by_default(self, h3):
        rhf = h3.molecular_hamiltonian(mo_basis=True, n_electrons=2)
        forced = h3.molecular_hamiltonian(mo_basis=True, n_electrons=2,
                                          open_shell=False)
        assert rhf.map_to_qubits().to_matrix() == pytest.approx(
            forced.map_to_qubits().to_matrix())

    def test_quartet_reference(self, h3):
        H = h3.molecular_hamiltonian(mo_basis=True, n_electrons=3,
                                     num_particles=(3, 0))
        assert H.n_modes() == 6
        assert _sector_ground_state(H, (3, 0)) > _sector_ground_state(H, (2, 1))


# --------------------------------------------------------------------------- #
# The geometry path.
# --------------------------------------------------------------------------- #

class TestGeometryPath:
    def test_hydrogen_atom_is_a_doublet(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5.0] * 3)
        H, particles, n_orb, _p, _c = build_basis_hamiltonian(
            atoms, "FAO", None, 0.3, 0, None)
        assert particles == (1, 0) and n_orb == 1 and H.n_modes() == 2

    def test_spin_flag_is_not_needed(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5.0] * 3)
        for spin in (False, True):
            _H, particles, _n, _p, _c = build_basis_hamiltonian(
                atoms, "FAO", None, 0.3, 0, None, spin=spin)
            assert particles == (1, 0)

    def test_magmoms_select_the_quartet(self):
        atoms = Atoms("H3", positions=[[0, 0, -0.9], [0, 0, 0], [0, 0, 0.9]],
                      cell=[6.5] * 3, magmoms=[1, 1, 1])
        _H, particles, _n, _p, _c = build_basis_hamiltonian(
            atoms, "FAO", None, 0.4, 0, None)
        assert particles == (3, 0)

    def test_wrong_parity_is_rejected(self):
        atoms = Atoms("H3", positions=[[0, 0, -0.9], [0, 0, 0], [0, 0, 0.9]],
                      cell=[6.5] * 3, magmoms=[1, 1, 0])
        with pytest.raises(ValueError, match="must be odd"):
            build_basis_hamiltonian(atoms, "FAO", None, 0.4, 0, None)

    def test_frozen_core_with_an_odd_count(self):
        # Li: 3 electrons; freezing the 1s leaves one active electron in 2s.
        atoms = Atoms("Li", positions=[[0, 0, 0]], cell=[8.0] * 3)
        H, particles, n_orb, _p, _c = build_basis_hamiltonian(
            atoms, "FAO", None, 0.4, 0, None, frozen_core=True)
        assert particles == (1, 0) and n_orb == 1 and H.n_modes() == 2
        with pytest.raises(ValueError, match="doubly"):
            build_basis_hamiltonian(atoms, "FAO", None, 0.4, 0, None,
                                    frozen_orbitals=[1])   # the singly occupied 2s

    def test_charge_makes_the_count_odd(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[6.0] * 3)
        _H, particles, _n, _p, _c = build_basis_hamiltonian(
            atoms, "FAO", None, 0.35, 1, None)             # H2+
        assert particles == (1, 0)

    def test_plane_waves_build_odd_counts(self):
        atoms = Atoms("H", positions=[[1.5, 1.5, 1.5]], cell=[3.0] * 3, pbc=True)
        H, particles, n_orb, _p, _c = build_basis_hamiltonian(
            atoms, {"name": "PW", "energy_cutoff": 50}, None, 0.3, 0, None)
        assert particles == (1, 0) and H.n_modes() == 2 * n_orb

    def test_dry_run_counts_odd_systems(self):
        atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]], cell=[7.0] * 3)
        est = estimate_qubits(atoms)                        # 9 electrons
        assert est.n_electrons == 9 and est.num_particles == (5, 4)
        assert est.n_qubits == 12


# --------------------------------------------------------------------------- #
# The solvers reach the sector ground state.
# --------------------------------------------------------------------------- #

class TestSolvers:
    def test_adapt_vqe_hydrogen_atom(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5.0] * 3)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="FAO", h=0.25, profile=False)
        atoms.get_potential_energy()
        calc = atoms.calc
        assert calc.num_particles == (1, 0) and calc.n_qubits == 2
        exact = _sector_ground_state(calc.hamiltonian, (1, 0))       # Hartree
        assert calc.result.in_units("Ha") == pytest.approx(exact, abs=1e-8)
        assert calc.result.in_units("Ha") == pytest.approx(-0.5, abs=0.03)
        assert calc.result.optimal_energy == pytest.approx(
            -0.5 * HARTREE_TO_EV, abs=0.03 * HARTREE_TO_EV)

    def test_adapt_vqe_h3_doublet_reaches_sector_fci(self, h3):
        H = h3.molecular_hamiltonian(mo_basis=True, n_electrons=3)
        exact = _sector_ground_state(H, (2, 1))
        driver = Mandacaru(method="adapt-vqe", hamiltonian=H, pool="fermionic",
                           num_particles=(2, 1), n_spatial_orbitals=3,
                           trace=False, profile=False, optimizer=LBFGSB,
                           gradient_tolerance=1e-6, max_iterations=30)
        result = driver.run()
        assert result.in_units("Ha") == pytest.approx(exact, abs=1e-5)
        # Singles carry gradient from a non-stationary reference: allowed.
        assert result.num_operators >= 1

    def test_vqe_h3_doublet_through_the_calculator(self):
        atoms = Atoms("H3", positions=[[0, 0, -0.9], [0, 0, 0], [0, 0, 0.9]],
                      cell=[6.5] * 3)
        atoms.calc = Mandacaru(method="vqe", basis="FAO", h=0.30,
                               optimizer=LBFGSB)
        atoms.get_potential_energy()
        calc = atoms.calc
        assert calc.num_particles == (2, 1) and calc.n_qubits == 6
        exact = _sector_ground_state(calc.hamiltonian, (2, 1))       # Hartree
        energy_ha = calc.result.in_units("Ha")
        assert energy_ha >= exact - 1e-9
        # UCCSD from the NO reference recovers the sector ground state.
        assert energy_ha == pytest.approx(exact, abs=1e-4)
        context = calc.solver._gradient_context
        uhf = context["integrals"].open_shell_hartree_fock(2, 1)
        assert energy_ha <= (
            uhf.electronic_energy + context["integrals"].nuclear_repulsion + 1e-6)


# --------------------------------------------------------------------------- #
# Plane waves.
# --------------------------------------------------------------------------- #

class TestPlaneWaves:
    @pytest.fixture(scope="class")
    def pw(self):
        from mandacaru.core import PlaneWaveIntegrals
        # The tiny anisotropic cell of test_planewave: 8 eV -> 3 PWs, so the
        # 6-qubit sector diagonalization below stays instant.
        cell = np.diag([16.0, 3.0, 3.0])                         # Bohr
        nuclei = [(1.0, np.array([8.0 - 0.7, 1.5, 1.5])),
                  (1.0, np.array([8.0 + 0.7, 1.5, 1.5]))]
        return PlaneWaveIntegrals(nuclei, cell, energy_cutoff=8, units="bohr")

    def test_open_shell_hartree_fock(self, pw):
        result = pw.open_shell_hartree_fock(1, 0)               # H2+
        assert result.converged
        C = result.natural_orbitals
        assert np.allclose(C.conj().T @ C, np.eye(C.shape[0]), atol=1e-10)
        assert result.natural_occupations[0] == pytest.approx(1.0)
        assert result.natural_occupations.sum() == pytest.approx(1.0)

    def test_odd_count_hamiltonian_is_variational(self, pw):
        H = pw.molecular_hamiltonian(mo_basis=True, n_electrons=1)
        assert H.n_modes() == 2 * pw.n_orbitals == 6
        exact = _sector_ground_state(H, (1, 0))
        uhf = pw.open_shell_hartree_fock(1, 0)
        assert exact <= uhf.reference_energy + pw.nuclear_repulsion + 1e-9
        with pytest.raises(ValueError, match="even electron count"):
            pw.molecular_hamiltonian(mo_basis=True, n_electrons=1,
                                     open_shell=False)

    def test_h2_plus_through_the_calculator(self):
        from ase.units import Bohr as B2A
        # The tiny anisotropic cell of test_planewave: 8 eV -> 3 PWs, 6 qubits.
        cell = np.diag([16 * B2A, 3 * B2A, 3 * B2A])
        atoms = Atoms("H2", positions=[[8 * B2A - 0.37, 1.5 * B2A, 1.5 * B2A],
                                       [8 * B2A + 0.37, 1.5 * B2A, 1.5 * B2A]],
                      cell=cell, pbc=True)
        atoms.calc = Mandacaru(method="adapt-vqe",
                                       basis={"name": "PW", "energy_cutoff": 8},
                                       charge=1, profile=False,
                                       optimizer=LBFGSB,
                                       gradient_tolerance=1e-6)
        atoms.get_potential_energy()
        calc = atoms.calc
        assert calc.num_particles == (1, 0) and calc.n_qubits == 6
        exact = _sector_ground_state(calc.hamiltonian, (1, 0))       # Hartree
        assert calc.result.in_units("Ha") == pytest.approx(exact, abs=1e-5)
