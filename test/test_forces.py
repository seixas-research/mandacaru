# -*- coding: utf-8 -*-
# file: test/test_forces.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Nuclear gradients: reduced density matrices, Hellmann-Feynman + Pulay forces.

The acceptance criterion throughout is the one that actually matters for a
geometry optimization: the analytic force must equal the negative finite
difference of the *same* energy the driver reports.  Everything else --- the RDM
contraction, the individual integral derivatives --- is checked as an
intermediate so that a failure localizes.

All finite-difference comparisons use a **fixed grid**.  The drivers regenerate
the grid per geometry by default, which makes it move with the molecule and adds
a spurious grid-drag term to the energy; the gradient is taken at fixed grid, so
that is the configuration in which the two must agree (see
:class:`~carcara.algorithms.QuantumCalculator`).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carcara.algorithms import (ADAPTVQE, QuantumCalculator, VQE, nuclear_gradient,
                                one_rdm, two_rdm)
from carcara.algorithms._hamiltonian_from_atoms import build_basis_hamiltonian
from carcara.algorithms._jax_energy import (energy_from_integrals,
                                            integral_gradients, jax_available)
from carcara.algorithms.rdm import electronic_energy, particle_number
from carcara.core.hamiltonian import spin_block_integrals
from carcara.integrals import Grid
from carcara.units import HARTREE_TO_EV

needs_jax = pytest.mark.skipif(not jax_available(), reason="jax not installed")

BOX, SPACING = 6.0, 0.20


@pytest.fixture(scope="module")
def fixed_grid():
    """One grid, shared by every geometry -- see the module docstring."""
    return Grid(center=[0.0, 0.0, 0.0], box_size=BOX, h=SPACING)


def h2(distance: float) -> Atoms:
    return Atoms("H2", positions=[[0, 0, -distance / 2], [0, 0, distance / 2]])


def run_vqe(atoms, grid):
    """Converge VQE on ``atoms`` and return ``(driver, state vector)``."""
    work = atoms.copy()
    work.calc = VQE(basis="FAO", grid=grid, verbose=False)
    work.get_potential_energy()
    driver = work.calc
    psi = driver.ansatz.state(driver.result.optimal_parameters)
    return driver, psi


# --------------------------------------------------------------------------- #
# Reduced density matrices.
# --------------------------------------------------------------------------- #

class TestReducedDensityMatrices:
    def test_one_rdm_counts_the_electrons(self, fixed_grid):
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        assert particle_number(gamma) == pytest.approx(2.0, abs=1e-10)

    def test_one_rdm_is_hermitian_and_idempotent_in_bounds(self, fixed_grid):
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        assert np.allclose(gamma, gamma.conj().T)
        # Occupation numbers of a fermionic 1-RDM lie in [0, 1].
        occupations = np.linalg.eigvalsh(gamma)
        assert occupations.min() > -1e-9
        assert occupations.max() < 1.0 + 1e-9

    def test_two_rdm_antisymmetry(self, fixed_grid):
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        assert np.allclose(gamma2, -gamma2.transpose(1, 0, 2, 3))
        assert np.allclose(gamma2, -gamma2.transpose(0, 1, 3, 2))

    def test_two_rdm_trace_gives_electron_pairs(self, fixed_grid):
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        # sum_pq Gamma_pqpq = N(N-1)
        pair_trace = np.real(np.einsum("pqpq->", gamma2))
        assert pair_trace == pytest.approx(2.0 * 1.0, abs=1e-9)

    def test_rdms_reproduce_the_driver_energy(self, fixed_grid):
        """The identity the whole gradient rests on."""
        atoms = h2(0.74)
        driver, psi = run_vqe(atoms, fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)

        integrals = driver._gradient_context["integrals"]
        rhf = integrals.hartree_fock(2)
        h_so, g_so = spin_block_integrals(rhf.h_mo, rhf.eri_mo)
        energy = (electronic_energy(gamma, gamma2, h_so, g_so)
                  + integrals.nuclear_repulsion)
        assert energy == pytest.approx(driver.result.optimal_energy,
                                       abs=1e-10)


# --------------------------------------------------------------------------- #
# The differentiable algebraic layer.
# --------------------------------------------------------------------------- #

@needs_jax
class TestJaxEnergyLayer:
    @staticmethod
    def _raw_integrals(integrals):
        S = np.real(integrals.overlap())
        T, V = integrals._engine.one_body(
            integrals._potentials.nuclear_potential, energy_units="Ha")
        h = np.real(0.5 * ((T + V) + (T + V).conj().T))
        eri = np.real(integrals._engine.two_body(method="fft",
                                                 energy_units="Ha"))
        return S, h, eri

    def test_reproduces_the_driver_energy(self, fixed_grid):
        """If the differentiated function were the wrong function, this fails."""
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        integrals = driver._gradient_context["integrals"]
        S, h, eri = self._raw_integrals(integrals)

        energy = energy_from_integrals(
            S, h, eri, gamma, gamma2, n_electrons=2,
            nuclear_repulsion=integrals.nuclear_repulsion)
        assert float(energy) == pytest.approx(
            driver.result.optimal_energy, abs=1e-10)

    def test_dE_dh_is_the_ao_density_matrix(self, fixed_grid):
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        integrals = driver._gradient_context["integrals"]
        S, h, eri = self._raw_integrals(integrals)

        _ds, dh, _dg = integral_gradients(S, h, eri, gamma, gamma2,
                                          n_electrons=2)
        # tr(D S) is the electron count for an AO-basis density matrix.
        assert np.trace(dh @ S) == pytest.approx(2.0, abs=1e-8)
        assert np.allclose(dh, dh.T)

    def test_overlap_response_is_non_zero(self, fixed_grid):
        """dE/dS is the orthonormality constraint -- the reason AD is used."""
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        integrals = driver._gradient_context["integrals"]
        S, h, eri = self._raw_integrals(integrals)

        ds, _dh, _dg = integral_gradients(S, h, eri, gamma, gamma2,
                                          n_electrons=2)
        assert np.max(np.abs(ds)) > 1e-3


# --------------------------------------------------------------------------- #
# Forces against finite differences.
# --------------------------------------------------------------------------- #

def analytic_forces(atoms, grid):
    """Converged VQE forces for ``atoms`` on ``grid``."""
    driver, psi = run_vqe(atoms, grid)
    context = driver._gradient_context
    gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
    gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
    return nuclear_gradient(
        context["integrals"], gamma, gamma2,
        n_electrons=context["n_electrons"],
        atom_of_orbital=context["atom_of_orbital"], frozen=context["frozen"])


def driver_energy(atoms, grid) -> float:
    work = atoms.copy()
    work.calc = VQE(basis="FAO", grid=grid, verbose=False)
    work.get_potential_energy()
    return work.calc.result.optimal_energy


@needs_jax
class TestForcesMatchFiniteDifference:
    def test_every_component(self, fixed_grid):
        atoms = h2(0.74)
        result = analytic_forces(atoms, fixed_grid)
        positions = atoms.get_positions()
        delta = 2e-3

        for atom in range(len(atoms)):
            for direction in range(3):
                plus, minus = positions.copy(), positions.copy()
                plus[atom, direction] += delta
                minus[atom, direction] -= delta
                fd = -(driver_energy(h2(0.74).__class__(
                            "H2", positions=plus), fixed_grid)
                       - driver_energy(h2(0.74).__class__(
                            "H2", positions=minus), fixed_grid)
                       ) / (2 * delta) * HARTREE_TO_EV
                analytic = result.forces[atom, direction]
                assert analytic == pytest.approx(fd, abs=5e-3), (
                    f"atom {atom} direction {direction}: "
                    f"{analytic} vs {fd}")

    def test_newtons_third_law(self, fixed_grid):
        """Forces on an isolated molecule must sum to zero.

        The residual (~1e-5 eV/A against forces of ~3 eV/A) is the
        finite-difference error of the orbital derivative plus the grid's own
        asymmetry, not a violation -- it is seven orders of magnitude below the
        force itself.
        """
        result = analytic_forces(h2(0.74), fixed_grid)
        residual = np.abs(result.forces.sum(axis=0)).max()
        assert residual < 1e-4 * result.max_force

    def test_force_points_along_the_bond(self, fixed_grid):
        """A stretched bond pulls the atoms together, a compressed one apart."""
        stretched = analytic_forces(h2(1.10), fixed_grid).forces
        assert stretched[0, 2] > 0 and stretched[1, 2] < 0   # restoring
        compressed = analytic_forces(h2(0.55), fixed_grid).forces
        assert compressed[0, 2] < 0 and compressed[1, 2] > 0  # repulsive


@needs_jax
class TestPulayContribution:
    def test_pulay_is_large_and_opposes_hellmann_feynman(self, fixed_grid):
        """The reason Hellmann-Feynman alone is not enough for this basis."""
        result = analytic_forces(h2(0.74), fixed_grid)
        hf = result.hellmann_feynman[1, 2]
        pulay = result.pulay[1, 2]

        assert np.sign(hf) != np.sign(pulay)          # it cancels, not adds
        assert abs(pulay) > 0.3 * abs(hf)             # and by a large fraction
        assert result.pulay_fraction > 0.5

    def test_hellmann_feynman_alone_is_wrong(self, fixed_grid):
        """Without Pulay the 'force' is not the derivative of the energy."""
        atoms = h2(0.74)
        full = analytic_forces(atoms, fixed_grid)

        driver, psi = run_vqe(atoms, fixed_grid)
        context = driver._gradient_context
        bare = nuclear_gradient(
            context["integrals"],
            one_rdm(psi, driver.n_qubits, driver.mapping),
            two_rdm(psi, driver.n_qubits, driver.mapping),
            n_electrons=context["n_electrons"],
            atom_of_orbital=context["atom_of_orbital"],
            frozen=context["frozen"], include_pulay=False)

        assert np.allclose(bare.pulay, 0.0)
        # The bare force differs from the true one by far more than the
        # finite-difference agreement tolerance.
        assert abs(bare.forces[1, 2] - full.forces[1, 2]) > 1.0


# --------------------------------------------------------------------------- #
# Isolated atoms: the force must vanish by symmetry.
# --------------------------------------------------------------------------- #

_ATOMIC_NUMBER = {"He": 2, "Be": 4, "C": 6, "O": 8}


def isolated_atom_force(symbol, n_electrons, spacing, box=5.0, offset=0.37,
                        use_analytic_density_gradient=False,
                        **force_kwargs) -> float:
    """Largest force on a lone atom, in eV/Angstrom.  Exact answer: zero.

    The atom is deliberately placed ``offset`` grid steps away from a node, so
    the test probes a generic sub-grid position rather than a symmetric one that
    would cancel the artifact by luck.
    """
    from carcara.algorithms.forces import (hellmann_feynman_gradient,
                                           orbital_gradients)
    from carcara.basis import BasisSet
    from carcara.core import MolecularIntegrals
    from carcara.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

    grid = Grid(center=[0.0, 0.0, 0.0], box_size=box, h=spacing)
    position = np.array([0.0, 0.0, offset * grid.dx * BOHR_TO_ANGSTROM])
    functions = BasisSet.build("FAO").atom(symbol, center=position,
                                           units="angstrom")
    charge = float(_ATOMIC_NUMBER[symbol])
    integrals = MolecularIntegrals(
        [(charge, position)], functions, grid,
        softening=0.5 * min(grid.dx, grid.dy, grid.dz))

    # Hartree-Fock density is enough: the symmetry argument does not care how
    # correlated the state is, only that it is spherical.
    rhf = integrals.hartree_fock(n_electrons)
    coefficients = integrals._lowdin_x() @ rhf.mo_coefficients
    occupied = coefficients[:, : n_electrons // 2]
    density = 2.0 * np.real(occupied @ occupied.conj().T)
    sampled = np.stack([f.evaluate(grid.X, grid.Y, grid.Z).ravel()
                        for f in functions])

    if use_analytic_density_gradient:
        force_kwargs["grad_psi"] = orbital_gradients(functions, grid)
    gradient = hellmann_feynman_gradient(
        density, sampled, grid, [position * ANGSTROM_TO_BOHR], [charge],
        integrals._potentials.softening, **force_kwargs)
    return float(np.linalg.norm(gradient)) * HARTREE_TO_EV / BOHR_TO_ANGSTROM


class TestIsolatedAtom:
    """A lone atom is spherically symmetric, so every force component is zero.

    This is the sharpest available probe of the electron-nucleus force term: any
    non-zero result is pure discretization error, with no physics to hide behind.
    It is what localized the heavy-atom problem, and what refuted the first
    attempt at a cure.
    """

    def test_light_atom_is_far_better_than_a_heavy_one(self):
        """The artifact grows steeply with Z -- the core is the culprit.

        Helium's 1s length scale (a0/Z = 0.26 A) is within reach of a practical
        grid; oxygen's (0.066 A) is not, and the spurious force is orders of
        magnitude larger.
        """
        helium = isolated_atom_force("He", 2, spacing=0.15)
        oxygen = isolated_atom_force("O", 8, spacing=0.15)
        assert oxygen > 10 * helium, (helium, oxygen)

    @pytest.mark.parametrize("symbol,n_electrons,spacing,ceiling", [
        ("He", 2, 0.15, 300.0),
        ("Be", 4, 0.15, 1500.0),
        ("O", 8, 0.15, 6000.0),
    ])
    def test_spurious_force_stays_within_the_known_bound(
            self, symbol, n_electrons, spacing, ceiling):
        """Regression guard.

        These ceilings record the *current* accuracy, which is poor -- the exact
        answer is zero.  They exist so that any change to the integral engine
        (pseudopotentials, a finer default grid, analytic integrals) shows up
        here as a dramatic improvement, and so that a regression cannot slip in
        unnoticed.
        """
        assert isolated_atom_force(symbol, n_electrons, spacing) < ceiling

    def test_analytic_form_does_not_converge_with_the_grid(self):
        """Pin the pathology: refining the grid does *not* help for Be.

        Until the spacing resolves the 1s scale a0/Z (0.13 A for beryllium), the
        nearest grid node moves into an unresolved cusp faster than the sampling
        improves, so the error grows.  If this ever starts failing, the engine
        has improved and the guidance in forces.py should be revisited.
        """
        coarse = isolated_atom_force("Be", 4, spacing=0.20)
        fine = isolated_atom_force("Be", 4, spacing=0.10)
        assert fine > coarse, (
            f"the analytic form now converges ({coarse:.0f} -> {fine:.0f}); "
            "the integral engine must have improved -- revisit forces.py")

    def test_by_parts_does_not_cure_it(self):
        """The finding that refuted integration by parts as a fix.

        With a *faithful* (analytic) density gradient the by-parts form is no
        better than the standard one.  Its apparent advantage when the density
        gradient is finite-differenced is a smoothing artifact across the nuclear
        cusp, not accuracy -- see `hellmann_feynman_gradient`.
        """
        from carcara.algorithms.forces import orbital_gradients

        standard = isolated_atom_force("Be", 4, spacing=0.10)
        faithful = isolated_atom_force("Be", 4, spacing=0.10, form="by-parts",
                                       use_analytic_density_gradient=True)
        assert faithful > 0.5 * standard, (
            "by-parts with an analytic density gradient unexpectedly fixed the "
            f"heavy-atom artifact ({standard:.0f} -> {faithful:.0f}); if real, "
            "update the guidance in forces.py and the LaTeX note")

    def test_unknown_form_rejected(self):
        with pytest.raises(ValueError, match="unknown Hellmann-Feynman form"):
            isolated_atom_force("He", 2, spacing=0.20, form="magic")


@needs_jax
class TestByPartsMode:
    """The by-parts mode is a diagnostic, not a better force -- pin that."""

    def test_it_disagrees_with_the_verified_force(self, fixed_grid):
        """For H2 the analytic form is verified to 0.04 %; by-parts is not.

        The disagreement is the point: two formulations that are identical in
        the continuum differ substantially on this grid, which is a direct
        measure of the discretization error near the nuclei.
        """
        atoms = h2(0.74)
        driver, psi = run_vqe(atoms, fixed_grid)
        context = driver._gradient_context
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        common = dict(n_electrons=context["n_electrons"],
                      atom_of_orbital=context["atom_of_orbital"],
                      frozen=context["frozen"])

        standard = nuclear_gradient(context["integrals"], gamma, gamma2,
                                    **common, hellmann_feynman="analytic")
        by_parts = nuclear_gradient(context["integrals"], gamma, gamma2,
                                    **common, hellmann_feynman="by-parts")
        assert np.isfinite(by_parts.forces).all()
        # They differ by tens of percent -- so by-parts must not be the default.
        relative = abs(by_parts.forces[1, 2] - standard.forces[1, 2]) \
            / abs(standard.forces[1, 2])
        assert relative > 0.1

    def test_recorded_in_the_result(self, fixed_grid):
        atoms = h2(0.74)
        driver, psi = run_vqe(atoms, fixed_grid)
        context = driver._gradient_context
        result = nuclear_gradient(
            context["integrals"],
            one_rdm(psi, driver.n_qubits, driver.mapping),
            two_rdm(psi, driver.n_qubits, driver.mapping),
            n_electrons=context["n_electrons"],
            atom_of_orbital=context["atom_of_orbital"],
            frozen=context["frozen"], hellmann_feynman="by-parts")
        assert result.details["hellmann_feynman"] == "by-parts"

    def test_default_is_the_verified_form(self):
        assert QuantumCalculator().hellmann_feynman == "analytic"

    def test_calculator_exposes_the_option(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = QuantumCalculator(method="vqe", basis="FAO",
                                       grid=fixed_grid, verbose=False,
                                       hellmann_feynman="by-parts")
        forces = atoms.get_forces()
        assert atoms.calc.force_result.details["hellmann_feynman"] == "by-parts"
        assert np.isfinite(forces).all()


# --------------------------------------------------------------------------- #
# The ASE calculator.
# --------------------------------------------------------------------------- #

@needs_jax
class TestQuantumCalculator:
    def test_implements_energy_and_forces(self):
        calc = QuantumCalculator(method="vqe", basis="FAO", h=SPACING,
                                 verbose=False)
        assert "energy" in calc.implemented_properties
        assert "forces" in calc.implemented_properties

    def test_get_forces_matches_the_driver(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = QuantumCalculator(method="vqe", basis="FAO",
                                       grid=fixed_grid, verbose=False)
        forces = atoms.get_forces()
        expected = analytic_forces(h2(0.74), fixed_grid).forces
        assert np.allclose(forces, expected, atol=1e-8)

    def test_energy_matches_the_bare_driver(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = QuantumCalculator(method="vqe", basis="FAO",
                                       grid=fixed_grid, verbose=False)
        energy = atoms.get_potential_energy()
        reference = driver_energy(h2(0.74), fixed_grid) * HARTREE_TO_EV
        assert energy == pytest.approx(reference, abs=1e-8)

    def test_grid_is_frozen_across_geometries(self):
        """The grid must not follow the atoms, or forces stop matching energies."""
        calc = QuantumCalculator(method="vqe", basis="FAO", h=SPACING,
                                 vacuum=2.5, verbose=False)
        atoms = h2(0.74)
        atoms.calc = calc
        atoms.get_forces()
        first = calc._grid
        assert first is not None

        atoms.set_positions(atoms.get_positions() + 0.3)
        atoms.get_forces()
        assert calc._grid is first

    def test_force_breakdown_is_exposed(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = QuantumCalculator(method="vqe", basis="FAO",
                                       grid=fixed_grid, verbose=False)
        atoms.get_forces()
        hf, pulay = atoms.calc.get_force_breakdown()
        assert hf.shape == (2, 3) and pulay.shape == (2, 3)
        assert np.max(np.abs(pulay)) > 0

    def test_adapt_vqe_driver_also_works(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = QuantumCalculator(method="adapt-vqe", basis="FAO",
                                       grid=fixed_grid, pool="qeb",
                                       verbose=False)
        forces = atoms.get_forces()
        # Same physics as the fixed-ansatz driver: H2 UCCSD and ADAPT both
        # reach FCI in this two-orbital space.
        expected = analytic_forces(h2(0.74), fixed_grid).forces
        assert np.allclose(forces, expected, atol=2e-3)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            QuantumCalculator(method="qaoa")

    def test_plane_wave_basis_is_rejected(self):
        """A plane-wave basis does not move with the nuclei: no Pulay machinery."""
        from carcara.units import BOHR_TO_ANGSTROM

        edge = 4 * BOHR_TO_ANGSTROM
        center = edge / 2
        atoms = Atoms("H2", positions=[[center, center, center - 0.37],
                                       [center, center, center + 0.37]],
                      cell=np.diag([edge, edge, edge]), pbc=True)
        # Rejected as soon as forces are requested, before any expensive
        # variational run (the energy path stays open for plane waves).
        atoms.calc = QuantumCalculator(method="vqe", verbose=False,
                                       basis={"name": "PW", "energy_cutoff": 60})
        with pytest.raises(NotImplementedError, match="atom-centered"):
            atoms.get_forces()
