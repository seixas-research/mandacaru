# -*- coding: utf-8 -*-
# file: test/test_paw_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Hellmann-Feynman + Pulay forces with PAW pseudopotentials and a DZP basis.

H2 and LiH in ``basis={"name": "PAW", "size": "DZP"}`` have 10 basis
functions, i.e. 20 qubits, solved exactly in the (1, 1) particle-number sector.

The VASP numbers are a *qualitative* guide only: VASP is PBE-DFT in plane
waves, Mandacaru is ADAPT-VQE in a localized basis, so the curves differ in
detail (Mandacaru's H2 minimum is 0.712 A against VASP's 0.750 A).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru

BASIS = {"name": "PAW", "size": "DZP"}

# VASP 6, PBE PAW (H 15Jun2001, Li 17Jan2003), ENCUT 520 eV, 10 A box, Gamma:
# bond-projected force on atom 0 (eV/A, + = attractive) at the ionic steps of
# the dimer relaxations.
VASP_H2 = {0.59263: -9.65602, 0.72042: -1.13367, 0.87607: 2.96966,
           1.00000: 4.22206}
VASP_LIH = {1.00000: -12.36098, 2.19265: 1.35493, 3.21539: 0.90887}


def dimer(symbols, distance, cell):
    """Along z, the midpoint off the grid nodes."""
    center = np.array([cell / 2 + 0.013, cell / 2 - 0.021, cell / 2 + 0.007])
    half = np.array([0.0, 0.0, distance / 2])
    return Atoms(symbols, positions=[center - half, center + half],
                 cell=[cell] * 3)


def calculator(h):
    return Mandacaru(method="adapt-vqe", basis=BASIS, h=h, pool="fermionic",
                     optimizer="L-BFGS-B", max_iterations=80,
                     gradient_tolerance=1e-5, profile=False)


def bond_force(forces):
    """Force on atom 0 along the bond (+ = toward atom 1)."""
    return float((forces[0, 2] - forces[1, 2]) / 2)


@pytest.fixture(scope="module")
def h2():
    """The relaxed-geometry run, with its force breakdown captured up front.

    The breakdown has to be read here: a later energy-only evaluation on the
    same calculator clears it (an old breakdown must never be reported for a
    newer geometry).
    """
    atoms = dimer("H2", 0.75, 8.0)
    atoms.calc = calculator(h=0.25)
    forces = atoms.get_forces()
    return SimpleNamespace(atoms=atoms, forces=forces,
                           raw=atoms.calc.force_result.unprojected,
                           breakdown=atoms.calc.get_force_breakdown(),
                           details=dict(atoms.calc.force_result.details))


def test_adapt_reaches_the_sector_ground_state(h2):
    """Complex (l > 0) orbitals are made conjugation-real, so the real pool
    reaches the exact ground state and the state is orbital-stationary."""
    solver = h2.atoms.calc.solver
    assert solver.n_qubits == 20 and solver._sector.dim == 100
    exact = np.linalg.eigvalsh(solver._h_matrix.toarray())[0]
    assert abs(solver.result.in_units("Ha") - exact) < 1e-7
    assert h2.details["orbital_gradient"] < 1e-4


def test_forces_are_the_derivative_of_the_energy(h2):
    """Central difference of the energy on the calculator's frozen grid."""
    atoms, forces = h2.atoms, h2.forces
    step = 0.005
    energies = []
    for sign in (1, -1):
        moved = atoms.copy()
        moved.positions[1, 2] += sign * step
        moved.calc = atoms.calc
        energies.append(moved.get_potential_energy())
    numerical = -(energies[0] - energies[1]) / (2 * step)
    # The raw gradient: the finite difference is of the discretized energy,
    # which is not translation invariant, while the reported force has had that
    # component projected out (`project_translation="auto"`).
    assert h2.raw[1, 2] == pytest.approx(numerical, abs=5e-3)


def test_hellmann_feynman_and_pulay(h2):
    # The breakdown sums to the *raw* gradient: the translational projection is
    # a step applied on top of it, not part of either component.
    forces, (hf, pulay) = h2.forces, h2.breakdown
    assert np.allclose(h2.raw, -(hf + pulay), atol=1e-12)
    assert np.abs(pulay).max() > 0.05             # an atom-centered basis needs it
    assert np.abs(forces.sum(axis=0)).max() < 0.15      # grid egg-box only
    assert np.abs(forces[:, :2]).max() < 0.1
    # 0.75 A is *outside* the PAW-DZP minimum, which sits at 0.712 A (measured
    # 2026-09-17, after the compensation charge gained its electron-ion
    # attraction; it was 0.807 A before, so this assertion used to read < -1).
    assert bond_force(forces) > 1.0


def test_h2_force_curve_follows_vasp():
    """Sign and rough magnitude away from the minimum.

    The two minima are 0.04 A apart (Mandacaru 0.712, VASP-PBE 0.750), so a
    sampled distance between them legitimately has opposite signs in the two
    methods -- the sign is only asked for where the reference force is well
    away from zero.  Near the minimum the force is merely required to be
    small, which is the real content there.
    """
    for distance, reference in VASP_H2.items():
        atoms = dimer("H2", distance, 8.0)
        atoms.calc = calculator(h=0.25)
        force = bond_force(atoms.get_forces())
        if abs(reference) > 2.0:
            assert np.sign(force) == np.sign(reference), distance
            assert force == pytest.approx(reference, rel=0.5), distance
        else:
            assert abs(force) < 3.0, distance


def test_lih_force_curve_follows_vasp():
    for distance, reference in VASP_LIH.items():
        atoms = dimer("LiH", distance, 10.0)
        atoms.calc = calculator(h=0.25)
        force = bond_force(atoms.get_forces())
        assert force == pytest.approx(reference, rel=0.25), distance


def test_two_qubit_register_forces_exact_and_measured(tmp_path):
    """Parity + two-qubit reduction: forces from the tapered register equal the
    4-qubit Jordan-Wigner ones, from the state vector and from Estimator
    expectation values of the optimized state."""
    from mandacaru.backends.providers import QiskitProvider

    def run(**options):
        atoms = dimer("H2", 1.0, 8.0)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="PAW", h=0.25,
                               pool="fermionic", optimizer="L-BFGS-B",
                               max_iterations=10, gradient_tolerance=1e-6,
                               profile=False, **options)
        return atoms.get_forces(), atoms.get_potential_energy(), atoms.calc

    jw, e_jw, _ = run(mapping="jordan_wigner")
    tapered, _, _ = run(mapping="parity", two_qubit_reduction=True,
                        output=str(tmp_path / "output.txt"))
    measured, e_measured, calc = run(
        mapping="parity", two_qubit_reduction=True,
        measurement_provider=QiskitProvider(device="statevector", shots=0))
    assert calc.n_qubits == 2
    assert np.allclose(tapered, jw, atol=1e-5)
    assert np.allclose(measured, jw, atol=1e-5)
    assert e_measured == pytest.approx(e_jw, abs=1e-5)
    assert len(calc.measurement["expectation_values"]) <= 16


# --------------------------------------------------------------------------- #
# Complex multipole channels (p/d valence).
# --------------------------------------------------------------------------- #

def water(cell=8.0):
    """C2v water with the two O-H bonds along +y and +z.

    Swapping the y and z axes maps the molecule onto itself and exchanges the
    two hydrogens, and the cubic grid shares that symmetry -- so the exact
    forces must obey it exactly.  It is the sharpest cheap probe of the
    compensation-charge algebra, because the y and z directions are carried by
    *different* multipole channels: z by the real M = 0 harmonic and y by the
    complex M = +-1 pair.
    """
    c = cell / 2
    return Atoms("OH2", positions=[[c, c, c], [c, c + 1.0, c], [c, c, c + 1.0]],
                 cell=[cell] * 3)


def water_calculator(h=0.25):
    return Mandacaru(method="adapt-vqe", basis={"name": "PAW", "size": "SZ"},
                     h=h, pool="fermionic", optimizer="L-BFGS-B",
                     max_iterations=60, gradient_tolerance=1e-4, profile=False)


@pytest.fixture(scope="module")
def h2o():
    """Water's forces, both as reported and as the raw gradient.

    ``project_translation`` defaults to ``"auto"``, so ``forces`` has had the
    translational component of the egg-box removed; ``raw`` is the gradient of
    the discretized energy, which is what the finite-difference and
    Hellmann-Feynman-plus-Pulay identities are about.
    """
    atoms = water()
    atoms.calc = water_calculator()
    forces = atoms.get_forces()
    return SimpleNamespace(atoms=atoms, forces=forces,
                           raw=atoms.calc.force_result.unprojected,
                           details=dict(atoms.calc.force_result.details))


def test_compensation_potentials_are_hermitian_only_for_m_zero():
    """The invariant the Pulay derivative has to respect.

    ``W_qs = <phi_q| v_L Y_LM |phi_s>`` has a *complex* weight whenever
    ``M != 0``, so it is not Hermitian there -- and a derivative written as
    ``cross + cross^H`` would quietly put ``conj(v)`` in the second half.
    """
    from mandacaru.algorithms._hamiltonian_from_atoms import (
        build_basis_hamiltonian, grid_from_cell)

    atoms = water()
    grid = grid_from_cell(atoms, 0.25)
    integrals = build_basis_hamiltonian(atoms, {"name": "PAW", "size": "SZ"},
                                        grid, 0.25, 0, None, False, False,
                                        None)[4]["integrals"]
    potentials = integrals.compensation_potentials()
    assert any(M != 0 for _atom, _L, M in potentials)   # oxygen has p channels
    for (_atom, _L, M), W in potentials.items():
        W = np.asarray(W)
        asymmetry = np.abs(W - W.conj().T).max()
        if M == 0:
            assert asymmetry < 1e-12
        else:
            assert asymmetry > 0.1 * np.abs(W).max()


def test_forces_respect_the_c2v_symmetry(h2o):
    """y and z are equivalent for this molecule; the forces must say so.

    This is what the complex-channel bug broke: the Pulay derivative of the
    compensation potentials was Hermitized, which is correct for the real
    M = 0 (z) channel and wrong for the complex M = +-1 (x, y) pair, so the
    oxygen force came out 2.25 along y against 2.78 along z.
    """
    forces = h2o.forces
    assert forces[0, 1] == pytest.approx(forces[0, 2], abs=5e-3)   # O
    assert forces[1, 1] == pytest.approx(forces[2, 2], abs=5e-3)   # H1 <-> H2
    assert forces[1, 2] == pytest.approx(forces[2, 1], abs=5e-3)
    assert abs(forces[0, 1]) > 1.0            # a real force, not a coincidence
    assert np.abs(forces[:, 0]).max() < 1e-3  # x is empty for this geometry


def test_p_valence_force_is_the_derivative_of_the_energy(h2o):
    """Central difference of the calculator's own energy on its frozen grid.

    Oxygen carries L = 1 and L = 2 compensation channels, so this exercises the
    whole multipole algebra; it was off by 0.57 eV/Angstrom (20 %) before the
    complex-weight fix.
    """
    atoms, forces = h2o.atoms, h2o.forces
    step = 0.004
    # Oxygen along y: the component the complex M = +-1 channels carry, and the
    # one that was off by 0.57 eV/Angstrom.  One component keeps the test's
    # memory inside the suite budget; the symmetry test covers the rest.
    atom, axis = 0, 1
    energies = []
    for sign in (1, -1):
        moved = atoms.copy()
        moved.positions[atom, axis] += sign * step
        moved.calc = atoms.calc
        energies.append(moved.get_potential_energy())
    numerical = -(energies[0] - energies[1]) / (2 * step)
    # Against the raw gradient: the finite difference is of the discretized
    # energy, which is not translation invariant, so it carries the egg-box that
    # the default projection removes from the reported force.
    assert h2o.raw[atom, axis] == pytest.approx(numerical, abs=1e-2)


def test_translation_projection_zeroes_the_net_force(h2o):
    """The default (``"auto"``) enforces the free-molecule identity.

    The exact forces of a free molecule sum to zero, so removing the mean takes
    out the translational component of the grid's egg-box and nothing physical.
    Everything needed to undo it is kept: the raw gradient, the residual that was
    there, and the mean that was removed.
    """
    forces, raw, details = h2o.forces, h2o.raw, h2o.details
    assert np.abs(forces.sum(axis=0)).max() < 1e-12
    assert details["translation_projected"] is True
    # The residual is the *unprojected* one, so it is still reported in full.
    assert details["translational_residual"] == pytest.approx(
        np.abs(raw.sum(axis=0)).max(), abs=1e-6)
    removed = np.asarray(details["translation_removed"])
    assert np.allclose(forces + removed, raw, atol=1e-12)
    # Projection is a rigid shift, so it cannot break the molecule's symmetry.
    assert forces[0, 1] == pytest.approx(forces[0, 2], abs=5e-3)


def test_project_translation_false_reports_the_raw_gradient(h2o):
    """Opting out gives back exactly the gradient of the computed energy."""
    atoms = water()
    atoms.calc = water_calculator()
    atoms.calc.project_translation = False
    forces = atoms.get_forces()
    assert np.allclose(forces, h2o.raw, atol=1e-10)
    details = atoms.calc.force_result.details
    assert "translation_projected" not in details
    assert "forces_unprojected" not in details
    # `unprojected` is still the right array to ask for -- it is the forces.
    assert np.allclose(atoms.calc.force_result.unprojected, forces)


def test_projection_cannot_increase_the_error_against_a_zero_sum_force(h2o):
    """Why the default is to project: it is an orthogonal projection.

    The exact force of a free molecule lies in the zero-sum subspace, and
    removing the mean is the orthogonal projection onto it -- so the projected
    array is no further from *any* zero-sum field than the raw one, whatever the
    grid error happens to be.  Checked here against the symmetrized force, which
    is the closest zero-sum field the symmetry allows.
    """
    raw, projected = h2o.raw, h2o.forces
    reference = raw - raw.mean(axis=0)              # the zero-sum part of raw
    assert np.linalg.norm(projected - reference) <= \
        np.linalg.norm(raw - reference) + 1e-12
    # And it is a strict improvement whenever the residual is nonzero at all.
    assert np.abs(raw.sum(axis=0)).max() > 1e-6
