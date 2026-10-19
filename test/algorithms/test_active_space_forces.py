# -*- coding: utf-8 -*-
# file: test/algorithms/test_active_space_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Forces of a variational state in an active space.

Two routes give the gradient of the energy a displaced calculation reports.
For the canonical orbitals of a closed-shell RHF reference (the ``"energy"``
ranking, or no truncation) it is analytic: the z-vector orbital response
folded into the RDMs (:mod:`~mandacaru.algorithms.orbital_response`).  For a
selector that rotates the orbitals (``"mp2"``) or an open shell it is a central
difference of the reduced Hamiltonian along the internal directions, in the
canonical orbital gauge.  Both are compared here with a central difference of
the *re-optimized* energy -- the energy the calculator reports at the
displaced geometry -- extrapolated from two steps (Richardson), along the
internal directions: an isolated molecule's gradient has no component along a
rigid translation or rotation, and the grid breaks that invariance slightly,
so a Cartesian derivative of one atom is not comparable with a rigid-free one.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.algorithms._hamiltonian_from_atoms import grid_from_cell
from mandacaru.algorithms.active_space_forces import (
    align_active_orbitals, active_space_gradient, internal_directions)


def calculator_atoms(symbols, positions, h, **options):
    atoms = Atoms(symbols, positions=positions, cell=[8.0] * 3)
    atoms.center()
    base = dict(method="adapt-vqe", grid=grid_from_cell(atoms, h),
                grid_spacing=h, project_translation=False, profile=False,
                pool="fermionic", transfer=True,
                convergence={"gradient": 1e-7})
    base.update(options)
    atoms.calc = Mandacaru(**base)
    return atoms


def reoptimized_slope(atoms, column, step):
    """``-dE/ds`` of the re-optimized energy along a displacement direction."""
    column = column.reshape(-1, 3)
    scale = step / np.linalg.norm(column, axis=1).max()
    energies = []
    for sign in (+1, -1):
        moved = atoms.copy()
        moved.positions = moved.positions + sign * scale * column
        moved.calc = atoms.calc
        energies.append(moved.get_potential_energy())
    return -(energies[0] - energies[1]) / (2 * scale)


def worst_directional_error(atoms, steps=(0.01, 0.005)):
    """``(error, details)``: the largest |F . d - Richardson(re-optimized
    slope)| over the internal directions, in eV/Angstrom, and the force
    result's details (a later energy clears them from the calculator)."""
    forces = atoms.get_forces()
    details = atoms.calc.force_result.details
    worst = 0.0
    for column in internal_directions(atoms.get_positions()).T:
        coarse = reoptimized_slope(atoms, column, steps[0])
        fine = reoptimized_slope(atoms, column, steps[1])
        extrapolated = (4 * fine - coarse) / 3
        worst = max(worst, abs(forces.ravel() @ column - extrapolated))
    return worst, details


def rigid_motions(positions):
    """The six rigid-motion vectors (translations, rotations about the centroid)."""
    relative = positions - positions.mean(axis=0)
    out = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = 1.0
        unit = np.zeros(3)
        unit[axis] = 1.0
        out += [translation.ravel(), np.cross(unit, relative).ravel()]
    return np.array(out)


@pytest.mark.parametrize("positions, count", [
    ([[0, 0, 0], [0, 0, 1.2]], 1),                                  # diatomic
    ([[0, 0, 0], [0, 0, 1.1], [0, 0, 2.3]], 4),                     # linear
    ([[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]], 3),            # bent
    ([[0, 0, 0], [0.94, 0, -0.38], [-0.47, 0.81, -0.38],
      [-0.47, -0.81, -0.38]], 6),
    ([[1.0, 2.0, 3.0]], 0),                                         # one atom
])
def test_the_internal_directions_count_3n_minus_rigid_motions(positions,
                                                               count):
    positions = np.asarray(positions, dtype=float)
    directions = internal_directions(positions)
    assert directions.shape == (positions.size, count)
    assert np.allclose(directions.T @ directions, np.eye(count))
    assert np.allclose(rigid_motions(positions) @ directions, 0.0,
                       atol=1e-12)


def test_a_gradient_free_of_rigid_motions_is_recovered_exactly():
    # What the finite differences assemble: the directional derivatives
    # along the internal directions, projected back.
    rng = np.random.default_rng(5)
    positions = rng.normal(size=(4, 3))
    directions = internal_directions(positions)
    gradient = directions @ rng.normal(size=directions.shape[1])
    rebuilt = directions @ (directions.T @ gradient)
    assert np.allclose(rebuilt, gradient)


# --------------------------------------------------------------------------- #
# The canonical orbital gauge.
# --------------------------------------------------------------------------- #

def orbitals(columns):
    return SimpleNamespace(mo_coefficients=np.asarray(columns, dtype=float))


def test_the_canonical_gauge_keeps_the_rotation_a_polar_alignment_removes():
    # A displaced active pair rotated by 0.05 rad relative to the reference:
    # the canonical orbitals are what the displaced build made (identity up to
    # a phase), the polar factor undoes the rotation.
    reference = orbitals(np.eye(3))
    angle = 0.05
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    displaced = orbitals(rotation * np.array([1.0, -1.0, 1.0]))
    active = (0, 1, 2)
    canonical = align_active_orbitals(reference, displaced, active, active)
    polar = align_active_orbitals(reference, displaced, active, active,
                                  gauge="polar")
    assert np.allclose(np.abs(canonical), np.eye(3))
    assert canonical[1, 1] == pytest.approx(-1.0, abs=1e-12) \
        or canonical[1, 1] == pytest.approx(-np.cos(angle) / abs(np.cos(angle)))
    assert not np.allclose(np.abs(polar), np.eye(3), atol=1e-3)


def test_a_degenerate_pair_falls_back_to_the_polar_gauge():
    reference = orbitals(np.eye(2))
    angle = 0.7                      # a free rotation inside a degenerate pair
    displaced = orbitals([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
    with pytest.warns(RuntimeWarning, match="polar gauge"):
        unitary = align_active_orbitals(reference, displaced, (0, 1), (0, 1))
    assert np.allclose(unitary.T @ unitary, np.eye(2))
    assert not np.allclose(np.abs(unitary), np.eye(2), atol=1e-3)


def test_an_unknown_gauge_is_refused():
    with pytest.raises(ValueError, match="gauge"):
        align_active_orbitals(orbitals(np.eye(2)), orbitals(np.eye(2)),
                              (0, 1), (0, 1), gauge="random")


# --------------------------------------------------------------------------- #
# Analytic forces (z-vector) against the re-optimized energy.
# --------------------------------------------------------------------------- #

class TestAnalyticForces:
    def test_h2_truncated_ansatz_full_space(self):
        """Two ADAPT operators: not orbital-stationary (|dE/dkappa| ~ 1e-2)."""
        atoms = calculator_atoms(
            "H2", [[0, 0, 0], [0.15, 0.1, 0.85]], 0.35,
            basis={"name": "PAW-LCAO", "size": "DZ"}, max_iterations=2)
        error, details = worst_directional_error(atoms)
        assert error < 2e-4
        assert details["orbital_response_method"] == "z-vector"
        assert details["orbital_gradient"] > 1e-3

    def test_lih_with_deleted_virtual_orbitals(self):
        atoms = calculator_atoms(
            "LiH", [[0, 0, 0], [0.2, 0.1, 1.65]], 0.35,
            basis={"name": "PAW-LCAO", "size": "DZP"},
            active_space={"orbitals": 4, "method": "energy"},
            max_iterations=3)
        error, details = worst_directional_error(atoms)
        assert error < 2e-4
        context = atoms.calc.solver._gradient_context
        assert len(context["deleted"]) == 6
        assert details["orbital_response_method"] == "z-vector"

    def test_lih_all_electron_frozen_core(self):
        atoms = calculator_atoms(
            "LiH", [[0, 0, 0], [0.2, 0.1, 1.65]], 0.30, basis="6-31G",
            active_space={"orbitals": 4, "method": "energy", "frozen": [0]},
            max_iterations=3)
        error, _details = worst_directional_error(atoms)
        assert atoms.calc.solver._gradient_context["frozen"] == (0,)
        assert error < 1e-3

    def test_water_against_the_canonical_finite_difference(self):
        """The central difference of the reduced Hamiltonian in the canonical
        gauge, at two steps, extrapolated: the analytic force is its limit."""
        b, half = 1.0, np.radians(95) / 2
        atoms = calculator_atoms(
            "OH2", [[0, 0, 0], [b * np.sin(half), 0, b * np.cos(half)],
                    [-b * np.sin(half), 0, b * np.cos(half)]], 0.30,
            basis={"name": "PAW-LCAO", "size": "SZ"},
            active_space={"orbitals": 5, "method": "energy"},
            max_iterations=3)
        forces = atoms.get_forces()
        calc = atoms.calc
        gamma, gamma2 = calc._state_rdms(calc.solver)
        directions = internal_directions(atoms.get_positions())
        coarse = active_space_gradient(atoms, calc.solver, gamma, gamma2,
                                       step=0.01)
        fine = active_space_gradient(atoms, calc.solver, gamma, gamma2,
                                     step=0.005)
        limit = (4 * fine - coarse) / 3
        project = lambda f: (directions @ (directions.T @ f.ravel())
                             ).reshape(-1, 3)
        assert project(forces) == pytest.approx(-limit, abs=2e-4)
        # The step dependence of the finite difference is its own truncation
        # error (h^2), not a discrepancy with the analytic force.
        assert abs(np.abs(project(forces) + coarse).max()) > \
            abs(np.abs(project(forces) + fine).max())
        # The old polar alignment missed the first-order rotation of the
        # active orbitals: several percent of the force on this state.
        polar = active_space_gradient(atoms, calc.solver, gamma, gamma2,
                                      step=0.005, gauge="polar")
        assert np.abs(project(forces) + polar).max() > 0.05


class TestFiniteDifferenceRoute:
    def test_an_mp2_selector_keeps_the_finite_difference(self):
        atoms = calculator_atoms(
            "LiH", [[0, 0, 0], [0.2, 0.1, 1.65]], 0.35,
            basis={"name": "PAW-LCAO", "size": "DZP"},
            active_space={"orbitals": 4, "method": "mp2"}, max_iterations=3)
        # The fallback is announced, not silent.
        with pytest.warns(RuntimeWarning,
                          match="central difference of the reduced"):
            error, details = worst_directional_error(atoms)
        assert error < 5e-4
        assert "orbital_response_method" not in details
        assert "active_space_response" in details

    @pytest.mark.slow
    def test_an_open_shell_cation_keeps_the_finite_difference(self):
        atoms = calculator_atoms(
            "OH2", [[0, 0, 0], [0.9, 0.1, 0.55], [-0.8, 0.2, 0.6]], 0.30,
            basis={"name": "PAW-LCAO", "size": "SZ"}, charge=1,
            active_space={"orbitals": 5, "method": "energy"},
            max_iterations=4)
        error, details = worst_directional_error(atoms)
        assert error < 5e-3
        assert "orbital_response_method" not in details
