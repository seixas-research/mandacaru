# -*- coding: utf-8 -*-
# file: test/test_forces.py

# This code is part of Mandacaru.
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
:class:`~mandacaru.algorithms.Mandacaru`).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mandacaru.algorithms import Mandacaru, nuclear_gradient, one_rdm, two_rdm
from mandacaru.algorithms._jax_energy import (energy_from_integrals,
                                              integral_gradients)
from mandacaru.algorithms.rdm import electronic_energy, particle_number
from mandacaru.core.hamiltonian import spin_block_integrals
from mandacaru.integrals import Grid
from mandacaru.units import HARTREE_TO_EV
import warnings
from mandacaru.utils import parse_output

BOX, SPACING = 6.0, 0.20


@pytest.fixture(scope="module")
def fixed_grid():
    """One grid, shared by every geometry -- see the module docstring."""
    return Grid(center=[0.0, 0.0, 0.0], box_size=BOX, grid_spacing=SPACING)


def h2(distance: float) -> Atoms:
    return Atoms("H2", positions=[[0, 0, -distance / 2], [0, 0, distance / 2]])


def run_vqe(atoms, grid):
    """Converge VQE on ``atoms`` and return ``(driver, state vector)``."""
    work = atoms.copy()
    work.calc = Mandacaru(method="vqe", basis="HAO", grid=grid, trace=False)
    work.get_potential_energy()
    driver = work.calc
    psi = driver.solver.ansatz.state(driver.result.optimal_parameters)
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

        integrals = driver.solver._gradient_context["integrals"]
        rhf = integrals.hartree_fock(2)
        h_so, g_so = spin_block_integrals(rhf.h_mo, rhf.eri_mo)
        energy = (electronic_energy(gamma, gamma2, h_so, g_so)
                  + integrals.nuclear_repulsion)               # Hartree
        assert energy == pytest.approx(driver.result.in_units("Ha"),
                                       abs=1e-10)


# --------------------------------------------------------------------------- #
# The differentiable algebraic layer.
# --------------------------------------------------------------------------- #

class TestJaxEnergyLayer:
    @staticmethod
    def _raw_integrals(integrals):
        S = np.real(integrals.overlap())
        T, V = integrals._engine.one_body(
            integrals._potentials.nuclear_potential, energy_units="Ha",
            kinetic=integrals.kinetic)
        h = np.real(0.5 * ((T + V) + (T + V).conj().T))
        eri = np.real(integrals._engine.two_body(energy_units="Ha"))
        return S, h, eri

    def test_reproduces_the_driver_energy(self, fixed_grid):
        """If the differentiated function were the wrong function, this fails."""
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        integrals = driver.solver._gradient_context["integrals"]
        S, h, eri = self._raw_integrals(integrals)

        energy = energy_from_integrals(
            S, h, eri, gamma, gamma2, n_electrons=2,
            nuclear_repulsion=integrals.nuclear_repulsion)
        # The differentiated function works in Hartree; the result is eV.
        assert float(energy) == pytest.approx(
            driver.result.in_units("Ha"), abs=1e-10)

    def test_dE_dh_is_the_ao_density_matrix(self, fixed_grid):
        driver, psi = run_vqe(h2(0.74), fixed_grid)
        gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
        gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
        integrals = driver.solver._gradient_context["integrals"]
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
        integrals = driver.solver._gradient_context["integrals"]
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
    context = driver.solver._gradient_context
    gamma = one_rdm(psi, driver.n_qubits, driver.mapping)
    gamma2 = two_rdm(psi, driver.n_qubits, driver.mapping)
    return nuclear_gradient(
        context["integrals"], gamma, gamma2,
        n_electrons=context["n_electrons"],
        atom_of_orbital=context["atom_of_orbital"], frozen=context["frozen"])


def driver_energy(atoms, grid) -> float:
    """VQE ground-state energy in eV (the unit of every driver result)."""
    work = atoms.copy()
    work.calc = Mandacaru(method="vqe", basis="HAO", grid=grid, trace=False)
    work.get_potential_energy()
    return work.calc.result.optimal_energy


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
                       ) / (2 * delta)                   # eV/Angstrom
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
        context = driver.solver._gradient_context
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
    from mandacaru.algorithms.forces import (hellmann_feynman_gradient,
                                             orbital_gradients)
    from mandacaru.basis import BasisSet
    from mandacaru.core import MolecularIntegrals
    from mandacaru.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

    grid = Grid(center=[0.0, 0.0, 0.0], box_size=box, grid_spacing=spacing)
    position = np.array([0.0, 0.0, offset * grid.dx * BOHR_TO_ANGSTROM])
    functions = BasisSet.build("HAO").atom(symbol, center=position,
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
        context = driver.solver._gradient_context
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
        context = driver.solver._gradient_context
        result = nuclear_gradient(
            context["integrals"],
            one_rdm(psi, driver.n_qubits, driver.mapping),
            two_rdm(psi, driver.n_qubits, driver.mapping),
            n_electrons=context["n_electrons"],
            atom_of_orbital=context["atom_of_orbital"],
            frozen=context["frozen"], hellmann_feynman="by-parts")
        assert result.details["hellmann_feynman"] == "by-parts"

    def test_default_is_the_verified_form(self):
        assert Mandacaru().hellmann_feynman == "analytic"

    def test_calculator_exposes_the_option(self, fixed_grid):
        # `hellmann_feynman` selects a form inside the legacy SCF-response
        # gradient; the default RDM gradient does not use it.
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="vqe", basis="HAO",
                                       grid=fixed_grid,
                                       force_method="scf-response",
                                       hellmann_feynman="by-parts")
        forces = atoms.get_forces()
        assert atoms.calc.force_result.details["hellmann_feynman"] == "by-parts"
        assert np.isfinite(forces).all()


# --------------------------------------------------------------------------- #
# The ASE calculator.
# --------------------------------------------------------------------------- #

class TestMandacaru:
    def test_implements_energy_and_forces(self):
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=SPACING)
        assert "energy" in calc.implemented_properties
        assert "forces" in calc.implemented_properties

    def test_get_forces_matches_the_driver(self, fixed_grid):
        """The legacy mode is exactly the driver function it wraps.

        Compared before the translational projection: the driver returns the raw
        gradient, and the projection the calculator applies by default is a
        separate, deliberate step on top of it.
        """
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid=fixed_grid,
                                       force_method="scf-response")
        atoms.get_forces()
        forces = atoms.calc.force_result.unprojected
        expected = analytic_forces(h2(0.74), fixed_grid).forces
        assert np.allclose(forces, expected, atol=1e-8)

    def test_the_two_force_methods_agree_on_a_closed_shell_s_basis(self,
                                                                   fixed_grid):
        """Where the legacy path is valid, the RDM gradient reproduces it.

        H2 in a minimal s basis is the one case the differentiated-SCF path
        models correctly (real closed-shell orbitals, no projectors), so the
        two independent implementations must agree.
        """
        def run(method):
            atoms = h2(0.74)
            atoms.calc = Mandacaru(method="vqe", basis="HAO", grid=fixed_grid, force_method=method)
            return atoms.get_forces()

        assert np.allclose(run("rdm"), run("scf-response"), atol=5e-3)

    def test_the_two_force_methods_agree_with_the_spectral_operator(
            self, fixed_grid):
        """The legacy path differentiates ``integrals.kinetic`` too: with
        the FFT Laplacian both gradients agree as they do on the stencil,
        and both are the derivative of the spectral energy (measured
        1.5e-6 eV/Angstrom for the legacy path, HISTORY Phase N)."""
        def run(method):
            atoms = h2(0.74)
            atoms.calc = Mandacaru(method="vqe", basis="HAO", grid=fixed_grid,
                                   force_method=method, kinetic="spectral",
                                   trace=False)
            return atoms.get_forces()

        legacy = run("scf-response")
        assert np.allclose(run("rdm"), legacy, atol=5e-3)
        stencil = h2(0.74)
        stencil.calc = Mandacaru(method="vqe", basis="HAO", grid=fixed_grid,
                                 force_method="scf-response", kinetic="fd",
                                 trace=False)
        assert np.abs(stencil.get_forces() - legacy).max() > 0.1

    def test_effective_potential_works_for_vqe(self, fixed_grid):
        """VQE can compute fast single-particle effective potential forces."""
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="vqe", basis="HAO", grid=fixed_grid,
                               force_method="effective-potential", trace=False)
        forces = atoms.get_forces()
        assert forces.shape == (2, 3)
        assert np.isfinite(forces).all()
        assert np.allclose(forces[0] + forces[1], 0.0, atol=1e-8)

    def test_effective_potential_rejected_for_unsupported_methods(self):
        """Effective-potential force method is rejected for unsupported solvers."""
        with pytest.raises(ValueError, match="supported for methods"):
            Mandacaru(method="subspace-vqe", force_method="effective-potential")

    def test_energy_matches_the_bare_driver(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="vqe", basis="HAO",
                                       grid=fixed_grid)
        energy = atoms.get_potential_energy()
        reference = driver_energy(h2(0.74), fixed_grid)          # eV already
        assert energy == pytest.approx(reference, abs=1e-8)

    def test_grid_is_frozen_across_geometries(self):
        """The grid must not follow the atoms, or forces stop matching energies."""
        calc = Mandacaru(method="vqe", basis="HAO", grid_spacing=SPACING)
        atoms = h2(0.74)
        atoms.center(vacuum=2.5)            # the cell is the frozen box
        atoms.calc = calc
        atoms.get_forces()
        first = calc._grid
        assert first is not None

        atoms.set_positions(atoms.get_positions() + 0.3)
        atoms.get_forces()
        assert calc._grid is first

    def test_force_breakdown_is_exposed(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="vqe", basis="HAO",
                                       grid=fixed_grid)
        atoms.get_forces()
        hf, pulay = atoms.calc.get_force_breakdown()
        assert hf.shape == (2, 3) and pulay.shape == (2, 3)
        assert np.max(np.abs(pulay)) > 0

    def test_adapt_vqe_driver_also_works(self, fixed_grid):
        atoms = h2(0.74)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO",
                                       grid=fixed_grid, pool="qeb")
        forces = atoms.get_forces()
        # Same physics as the fixed-ansatz driver: H2 UCCSD and ADAPT both
        # reach FCI in this two-orbital space.
        expected = analytic_forces(h2(0.74), fixed_grid).forces
        assert np.allclose(forces, expected, atol=2e-3)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            Mandacaru(method="qaoa")

    def test_plane_wave_basis_is_rejected(self):
        """A plane-wave basis does not move with the nuclei: no Pulay machinery."""
        from mandacaru.units import BOHR_TO_ANGSTROM

        edge = 4 * BOHR_TO_ANGSTROM
        center = edge / 2
        atoms = Atoms("H2", positions=[[center, center, center - 0.37],
                                       [center, center, center + 0.37]],
                      cell=np.diag([edge, edge, edge]), pbc=True)
        # Rejected as soon as forces are requested, before any expensive
        # variational run (the energy path stays open for plane waves).
        atoms.calc = Mandacaru(method="vqe",
                                       basis={"name": "PW", "energy_cutoff": 60})
        with pytest.raises(NotImplementedError, match="atom-centered"):
            atoms.get_forces()


class TestProjectedForcesAreQualified:
    @pytest.fixture(scope="class")
    def logged(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("f10") / "output.txt")
        # Asymmetric, so the projection removes something.
        atoms = Atoms("H3", positions=[[3, 3, 2.6], [3, 3, 3.4], [3, 3.7, 3.1]],
                      cell=[6.0] * 3)
        # profile=False: this file reads Parquet above, and building Qiskit
        # circuits in the same process then crashes in Qiskit's allocator (the
        # documented pandas/pyarrow interaction).  The forces need no circuits.
        atoms.calc = Mandacaru(method="adapt-vqe", basis="HAO", grid_spacing=0.45, charge=1,
                               pool="fermionic", max_iterations=3, profile=False,
                               convergence={"gradient": 1e-3}, txt=out)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms.get_forces()
        return parse_output(out)["forces"], atoms.calc.force_result

    def test_both_arrays_are_written(self, logged):
        block, result = logged
        assert np.allclose(block["forces"], result.forces, atol=1e-7)
        assert np.allclose(block["forces_unprojected"], result.unprojected,
                           atol=1e-7)

    def test_the_relation_between_them_is_recorded(self, logged):
        block, _result = logged
        raw = np.asarray(block["forces_unprojected"])
        assert np.allclose(block["forces"], raw - raw.mean(axis=0), atol=1e-7)
        assert "forces_unprojected - mean" in block["convention"]

    def test_the_unqualified_convention_is_not_claimed(self, logged):
        block, _result = logged
        # The projected force is not -(HF + Pulay); the block must not say it is.
        assert block["convention"] != \
            "forces = -dE/dR (ASE sign); hellmann_feynman and pulay are +dE/dR"
        assert "translation_removed" in block


# --------------------------------------------------------------------------- #
# The kinetic cross block, for either Laplacian.
# --------------------------------------------------------------------------- #

class TestTheKineticCrossBlock:
    """``one_body_cross`` is the Pulay derivative of ``T + V`` with the
    energy's own Laplacian: against a central difference of the matrix built
    with the moved function, for the stencil and the FFT Laplacian."""

    STEP = 1e-3                       # Bohr

    @staticmethod
    def _basis(shift=0.0):
        from mandacaru.basis import GaussianOrbital

        return [GaussianOrbital(0, 0, [0.9], [1.0], center=[0.2 + shift, 0.1,
                                                            -0.1],
                                units="bohr"),
                GaussianOrbital(1, 0, [0.5], [1.0], center=[-0.4, 0.0, 0.3],
                                units="bohr")]

    @pytest.mark.parametrize("kinetic", ["fd", "spectral"])
    def test_it_is_the_derivative_of_the_one_body_matrix(self, kinetic):
        from mandacaru.algorithms.forces import (one_body_cross,
                                                 orbital_derivatives)
        from mandacaru.integrals import IntegralEngine

        grid = Grid(center=[0.0, 0.0, 0.0], box_size=10.0, grid_spacing=0.4,
                    units="bohr")
        vext = (-1.0 / np.maximum(np.sqrt(grid.X ** 2 + grid.Y ** 2
                                          + grid.Z ** 2), 0.5)).ravel()

        def potential(x, y, z):
            return -1.0 / np.maximum(np.sqrt(x ** 2 + y ** 2 + z ** 2), 0.5)

        def matrix(shift):
            engine = IntegralEngine(self._basis(shift), grid)
            T, V = engine.one_body(potential, energy_units="Ha",
                                   kinetic=kinetic)
            return T + V

        numerical = (matrix(self.STEP) - matrix(-self.STEP)) / (2 * self.STEP)
        engine = IntegralEngine(self._basis(), grid)
        psi = engine._psi
        dpsi = orbital_derivatives(self._basis(), grid, [0, 1], 0)[0]
        laplacian = (engine.spectral_laplacian(psi) if kinetic == "spectral"
                     else None)
        cross = one_body_cross(psi, dpsi, vext, grid, kinetic, laplacian)
        analytic = cross + cross.conj().T
        assert np.abs(analytic - numerical).max() < 1e-6 * np.abs(
            numerical).max()

    def test_an_unknown_operator_or_a_missing_laplacian_is_refused(self):
        from mandacaru.algorithms.forces import one_body_cross

        grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.0, grid_spacing=0.5, units="bohr")
        psi = np.ones((1, grid.size), dtype=complex)
        vext = np.zeros(grid.size)
        with pytest.raises(ValueError, match="unknown kinetic"):
            one_body_cross(psi, psi, vext, grid, "fourier")
        with pytest.raises(ValueError, match="laplacian"):
            one_body_cross(psi, psi, vext, grid, "spectral")


class TestStreamedPairPotentials:
    """The two-body Pulay term streams the pair potentials a block at a time:
    the same numbers as the whole ``M^2 G`` stack, a peak set by the block."""

    @staticmethod
    def _orbitals(M=4, seed=0):
        grid = Grid(center=[0.0, 0.0, 0.0], box_size=4.0, grid_spacing=0.4, units="bohr")
        rng = np.random.default_rng(seed)
        r2 = (grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2).ravel()
        psi = np.stack([np.exp(-(0.6 + 0.3 * m) * r2)
                        * (rng.normal(size=r2.size) * 0.05 + 1.0
                           + 1j * rng.normal(size=r2.size) * 0.05)
                        for m in range(M)])
        return grid, psi

    def test_blocks_are_the_whole_stack(self, monkeypatch):
        from mandacaru.algorithms.forces import pair_potential_blocks
        from mandacaru.integrals.poisson import PoissonFFTSolver

        grid, psi = self._orbitals()
        M, G = psi.shape
        whole = PoissonFFTSolver(grid.shape, step=grid.step).solve_stack(
            (np.conj(psi)[:, None, :] * psi[None, :, :]).reshape(M * M, G))
        # A budget of three pairs per block: 16 G (2 + 2M) bytes each.
        monkeypatch.setenv("MANDACARU_ERI_MEMORY_MB",
                           str(3 * 16 * G * (2 + 2 * M) / 2 ** 20))
        sizes = []
        for start, stop, phi in pair_potential_blocks(psi, grid):
            sizes.append(stop - start)
            assert np.abs(phi - whole[start:stop]).max() < 1e-12
        assert sizes[:-1] == [3] * (len(sizes) - 1) and sum(sizes) == M * M

    def test_the_gradient_is_the_dense_contraction(self, monkeypatch):
        from mandacaru.algorithms.forces import _two_body_pulay
        from mandacaru.integrals.poisson import PoissonFFTSolver

        grid, psi = self._orbitals(seed=1)
        M, G = psi.shape
        rng = np.random.default_rng(2)
        derivatives = rng.normal(size=(3, M, G)) * psi[None]
        de_dg = rng.normal(size=(M,) * 4)
        owner = [0, 0, 1, 1]
        # What the gradient was before streaming: all pair potentials held.
        phi = PoissonFFTSolver(grid.shape, step=grid.step).solve_stack(
            (np.conj(psi)[:, None, :] * psi[None, :, :]).reshape(M * M, G)
        ).reshape(M, M, G)
        A = np.einsum("mnls,nsg->mlg", de_dg, phi)
        B = np.einsum("mnls,mlg->nsg", de_dg, phi)
        expected = np.zeros((2, 3))
        for atom in range(2):
            for k in range(3):
                dpsi = np.where((np.asarray(owner) == atom)[:, None],
                                derivatives[k], 0.0)
                drho = (dpsi.conj()[:, None, :] * psi[None, :, :]
                        + psi.conj()[:, None, :] * dpsi[None, :, :])
                expected[atom, k] = np.real(
                    np.einsum("mlg,mlg->", drho, A)
                    + np.einsum("nsg,nsg->", drho, B)) * grid.dV
        monkeypatch.setenv("MANDACARU_ERI_MEMORY_MB",
                           str(2 * 16 * G * (2 + 2 * M) / 2 ** 20))
        streamed = _two_body_pulay(psi, derivatives, grid, de_dg, owner, 2)
        assert np.abs(streamed - expected).max() < 1e-10 * np.abs(
            expected).max()

    def test_the_peak_follows_the_block(self, monkeypatch):
        import tracemalloc

        from mandacaru.algorithms.forces import _two_body_pulay

        from mandacaru.integrals.poisson import PoissonFFTSolver

        grid, psi = self._orbitals(M=12, seed=3)
        M, G = psi.shape
        rng = np.random.default_rng(4)
        derivatives = rng.normal(size=(3, M, G)) * psi[None]
        de_dg = rng.normal(size=(M,) * 4)
        owner = [0] * 6 + [1] * 6
        # One pair per block.
        monkeypatch.setenv("MANDACARU_ERI_MEMORY_MB",
                           str(16 * G * (2 + 2 * M) / 2 ** 20))
        # The kernel and the FFT workspace are cached per grid: build them
        # first, so the measurement is what the gradient allocates.
        PoissonFFTSolver(grid.shape, step=grid.step).solve(np.abs(psi[0]) ** 2)
        tracemalloc.start()
        try:
            _two_body_pulay(psi, derivatives, grid, de_dg, owner, 2)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The whole stack of pair potentials alone would be 16 M^2 G bytes;
        # a block is its pair's density, potential and two (M, G) products.
        assert peak < 16 * M * M * G / 2
