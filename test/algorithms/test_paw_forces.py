# -*- coding: utf-8 -*-
# file: test/test_paw_forces.py

# This code is part of Mandacaru.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Hellmann-Feynman + Pulay forces with PAW-LCAO pseudopotentials and a DZP basis.

H2 and LiH in ``basis={"name": "PAW-LCAO", "size": "DZP"}`` have 10 basis
functions, i.e. 20 qubits, solved exactly in the (1, 1) particle-number sector.

Forces come from ``atoms.get_forces()`` (eV/Angstrom).  The default
``project_translation="auto"`` reports the gradient projected onto the
internal molecular directions; ``project_translation=False`` (or
``calc.result.forces()``) is the raw derivative of the discretized energy,
which is what a finite difference of the energy on the calculator's frozen
grid measures.

The reference numbers are a *qualitative* guide only: they are PBE-DFT in
plane waves, Mandacaru is ADAPT-VQE in a localized basis, so the curves differ
in detail (Mandacaru's H2 minimum is 0.785 A against the reference's 0.750 A).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from mandacaru import Mandacaru
from mandacaru.integrals import Grid

# The classical optimizer used below, with its budget and tolerance written
# out.  `tol` is deliberately looser than the library default (1e-12): these
# tests compare an analytic gradient against a finite difference of the *same*
# state, so they need a converged state, not a converged screening gradient --
# and 1e-12 costs 15x the wall time here for no change in what is measured.
LBFGS = {"method": "L-BFGS", "maxiter": 2000, "tol": 1e-8}

BASIS = {"name": "PAW-LCAO", "size": "DZP"}

# A plane-wave PBE PAW reference (520 eV cutoff, 10 A box, Gamma point):
# bond-projected force on atom 0 (eV/A, + = attractive) at the ionic steps of
# the dimer relaxations.
PLANE_WAVE_H2 = {0.59263: -9.65602, 0.72042: -1.13367, 0.87607: 2.96966,
           1.00000: 4.22206}
PLANE_WAVE_LIH = {1.00000: -12.36098, 2.19265: 1.35493, 3.21539: 0.90887}


def dimer(symbols, distance, cell):
    """Along z, the midpoint off the grid nodes."""
    center = np.array([cell / 2 + 0.013, cell / 2 - 0.021, cell / 2 + 0.007])
    half = np.array([0.0, 0.0, distance / 2])
    return Atoms(symbols, positions=[center - half, center + half],
                 cell=[cell] * 3)


def calculator(grid_spacing, basis=BASIS):
    return Mandacaru(method="adapt-vqe", basis=basis, grid_spacing=grid_spacing, pool="fermionic",
                     optimizer=LBFGS, max_iterations=80,
                     convergence={"gradient": 1e-5}, profile=False)


def bond_force(forces):
    """Force on atom 0 along the bond (+ = toward atom 1)."""
    return float((forces[0, 2] - forces[1, 2]) / 2)


def sector_ground_energy(calc):
    """Lowest eigenvalue of the qubit Hamiltonian in the particle-number sector."""
    operator = calc.result.qubit_hamiltonian()
    indices = np.asarray(calc.problem.basis_indices, dtype=int)
    matrix = np.empty((len(indices), len(indices)), dtype=complex)
    vector = np.zeros(2 ** calc.n_qubits, dtype=complex)
    for column, index in enumerate(indices):
        vector[:] = 0.0
        vector[index] = 1.0
        matrix[:, column] = operator.apply(vector)[indices]
    return float(np.linalg.eigvalsh(matrix)[0])


def energy_derivative(atoms, atom, axis, step):
    """Central difference of the potential energy on the calculator's frozen grid."""
    energies = []
    for sign in (1, -1):
        moved = atoms.copy()
        moved.positions[atom, axis] += sign * step
        moved.calc = atoms.calc
        energies.append(moved.get_potential_energy())
    return -(energies[0] - energies[1]) / (2 * step)


def richardson_derivative(atoms, atom, axis, steps):
    """Richardson-extrapolated central difference for the two ``steps``."""
    coarse, fine = (energy_derivative(atoms, atom, axis, s) for s in steps)
    return fine + (fine - coarse) / 3.0


@pytest.fixture(scope="module")
def h2():
    """The relaxed-geometry run, with everything read from it captured up front.

    A later energy evaluation on the same calculator replaces its result, so
    the state, the details and the raw gradient are read here.
    """
    atoms = dimer("H2", 0.75, 8.0)
    atoms.calc = calculator(grid_spacing=0.25)
    forces = atoms.get_forces()
    calc = atoms.calc
    details = dict(calc.result.force_details)
    return SimpleNamespace(atoms=atoms, forces=forces, details=details,
                           raw=calc.result.forces(),
                           energy=calc.result.in_units("Ha"),
                           exact=sector_ground_energy(calc),
                           n_qubits=calc.n_qubits,
                           dimension=calc.problem.dimension)


def test_adapt_reaches_the_sector_ground_state(h2):
    """Complex (l > 0) orbitals are made conjugation-real, so the real pool
    reaches the exact ground state."""
    assert h2.n_qubits == 20 and h2.dimension == 100
    assert abs(h2.energy - h2.exact) < 1e-7


def test_forces_are_the_derivative_of_the_energy(h2):
    """Central difference of the energy on the calculator's frozen grid."""
    numerical = energy_derivative(h2.atoms, 1, 2, 0.005)
    # The raw gradient: the finite difference is of the discretized energy,
    # which is not translation invariant, while the reported force has had that
    # component projected out (`project_translation="auto"`).
    assert h2.raw[1, 2] == pytest.approx(numerical, abs=5e-3)


def test_the_force_is_the_energy_derivative_of_a_displaced_hamiltonian(h2):
    """Native details: the gradient is the central difference of the reduced
    Hamiltonian with the orbitals re-solved at each displacement."""
    assert h2.details["scheme"] == "Central-reduced-Hamiltonian"
    assert h2.details["orbital_response"] == "displaced-SCF-and-selector"
    assert h2.details["energy_ha"] == pytest.approx(h2.energy, abs=1e-10)


def test_h2_raw_force_is_a_bond_force(h2):
    forces, raw = h2.forces, h2.raw
    assert np.abs(raw.sum(axis=0)).max() < 0.15      # grid egg-box only
    assert np.abs(raw[:, :2]).max() < 0.1
    # 0.75 A is just *outside* the PAW-LCAO-DZP minimum, which sits at 0.733 A with
    # the family's default basis since 2026-09-20 -- confined (0.1 eV), the
    # first_zeta split-valence scheme, the Gaussian polarization -- against
    # 0.741 (experiment) and 0.750 (plane-wave PBE).  The history of this line
    # is the history of the basis: 0.807 A before the compensation charge
    # gained its electron-ion attraction (< -1), 0.712 A with the last_zeta
    # split and free-atom orbitals (> 1.0; kept reproducible below), 0.760 A
    # with the first_zeta split alone (< 0).  Since 2026-10-12 the spheres of
    # a wave-function Hamiltonian are exact, and the minimum moved out to
    # 0.772 A (grid_spacing = 0.25, FCI): the frozen LDA spheres had put H2 26 mHa below
    # the exact energy and the exact ones put it 16 mHa above, the excess
    # growing as the two hydrogen spheres (r_c 1.30 Bohr) overlap -- 1 mHa at
    # R = 2.6 Bohr, 16 at 1.4 (HISTORY.md "PAW-LCAO Hamiltonian: one-center
    # exact exchange").  0.75 A is now *inside* the minimum.  Since
    # 2026-10-13 the kinetic operator is spectral, which the stencil had
    # under-estimated: the minimum moved out again to 0.785 A (bond force
    # -1.352 / -0.173 / +0.495 eV/Angstrom at 0.75 / 0.78 / 0.80 A), and the
    # force here from -0.5 ... -1.2 to -1.35.  Since 2026-10-25 the hydrogen
    # dataset has a 1.0 Bohr sphere and an empty p channel, which brings the
    # overlap excess down (HISTORY.md "LiH bisect"): the minimum moves back in
    # and the force here is -0.95.
    assert -1.3 < bond_force(forces) < -0.7
    assert bond_force(raw) == pytest.approx(bond_force(forces), abs=0.15)


def test_the_last_zeta_split_is_still_available():
    atoms = dimer("H2", 0.75, 8.0)
    atoms.calc = calculator(grid_spacing=0.25,
                            basis={**BASIS, "zeta_split": "last_zeta",
                                   "energy_shift": None})
    # Outside that basis's minimum (0.711 A) with the frozen spheres, the
    # pre-2026-09-20 behavior (> 1.0); with the exact spheres (2026-10-12) the
    # minimum moved past 0.75 A (-0.22 eV/Angstrom), as for the default basis,
    # and with the spectral kinetic default (2026-10-13) to 0.776 A
    # (-0.954 / +0.141 eV/Angstrom at 0.75 / 0.78 A); with the 1.0 Bohr
    # hydrogen sphere (2026-10-25) the force at 0.75 A is -0.497.
    assert -0.8 < bond_force(atoms.get_forces()) < -0.3


@pytest.mark.parametrize("family", ["PAW-LCAO", "UPAW-LCAO"])
def test_spectral_kinetic_forces_are_the_derivative_of_the_energy(family):
    """``kinetic="spectral"`` through the displaced-Hamiltonian gradient.

    LiH, SZ, tilted off its axis: ADAPT-VQE reaches the sector ground state,
    so a Richardson-extrapolated central difference of the reported energy
    on the frozen grid is the reference (measured 4e-6 eV/Angstrom, as the
    stencil's)."""
    atoms = dimer("LiH", 1.6, 8.0)
    atoms.positions[1, 0] += 0.05
    atoms.calc = Mandacaru(method="adapt-vqe", grid_spacing=0.25, pool="fermionic",
                           basis={"name": family, "size": "SZ"},
                           kinetic="spectral", optimizer=LBFGS,
                           convergence={"gradient": 1e-6}, profile=False)
    atoms.get_forces()
    raw = atoms.calc.result.forces()

    numerical = richardson_derivative(atoms, 1, 2, (0.004, 0.002))
    assert abs(numerical) > 0.1
    assert raw[1, 2] == pytest.approx(numerical, abs=1e-4)


def test_h2_force_curve_follows_the_plane_wave_reference():
    """Sign and rough magnitude away from the minimum.

    The two minima are 0.04 A apart (Mandacaru 0.712, plane-wave PBE 0.750), so a
    sampled distance between them legitimately has opposite signs in the two
    methods -- the sign is only asked for where the reference force is well
    away from zero.  Near the minimum the force is merely required to be
    small, which is the real content there.
    """
    for distance, reference in PLANE_WAVE_H2.items():
        atoms = dimer("H2", distance, 8.0)
        atoms.calc = calculator(grid_spacing=0.25)
        force = bond_force(atoms.get_forces())
        if abs(reference) > 2.0:
            assert np.sign(force) == np.sign(reference), distance
            assert force == pytest.approx(reference, rel=0.5), distance
        else:
            assert abs(force) < 3.0, distance


def test_lih_force_curve_follows_the_plane_wave_reference():
    for distance, reference in PLANE_WAVE_LIH.items():
        atoms = dimer("LiH", distance, 10.0)
        atoms.calc = calculator(grid_spacing=0.25)
        force = bond_force(atoms.get_forces())
        assert force == pytest.approx(reference, rel=0.25), distance


def test_two_qubit_register_forces_match_the_jordan_wigner_ones():
    """Parity + two-qubit reduction: forces from the reduced register equal the
    4-qubit Jordan-Wigner ones."""
    def run(**options):
        atoms = dimer("H2", 1.0, 8.0)
        atoms.calc = Mandacaru(method="adapt-vqe", basis="PAW-LCAO", grid_spacing=0.25,
                               pool="fermionic", optimizer=LBFGS,
                               max_iterations=10, convergence={"gradient": 1e-6},
                               profile=False, **options)
        return atoms.get_forces(), atoms.get_potential_energy(), atoms.calc

    jw, e_jw, _ = run(mapping="jordan_wigner")
    reduced, e_reduced, calc = run(mapping="parity_reduced")
    assert calc.n_qubits == 2
    assert np.allclose(reduced, jw, atol=1e-5)
    assert e_reduced == pytest.approx(e_jw, abs=1e-5)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity_reduced"])
def test_reduced_active_space_force_tracks_optimized_energy(mapping):
    """The PAW force includes the moving selected orbitals and frozen grid."""
    atoms = dimer("H2", 0.92, 8.0)
    grid = Grid(center=atoms.positions.mean(axis=0), cell=np.eye(3) * 8.0,
                grid_spacing=0.35)
    atoms.calc = Mandacaru(
        method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "DZ"},
        grid=grid, grid_spacing=0.35, active_space={"orbitals": 2, "method": "mp2"},
        mapping=mapping, pool="fermionic", optimizer=LBFGS,
        max_iterations=20, convergence={"gradient": 1e-6},
        project_translation=False, profile=False)
    force = atoms.get_forces()[1, 2]
    assert atoms.calc.result.force_details["orbital_response"] == "displaced-SCF-and-selector"

    assert force == pytest.approx(energy_derivative(atoms, 1, 2, 0.005), abs=0.02)


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


def water_calculator(grid_spacing=0.25, **options):
    return Mandacaru(method="adapt-vqe", basis={"name": "PAW-LCAO", "size": "SZ"},
                     grid_spacing=grid_spacing, pool="fermionic", optimizer=LBFGS,
                     max_iterations=60, convergence={"gradient": 1e-4}, profile=False,
                     **options)


@pytest.fixture(scope="module")
def h2o():
    """Water's forces, both as reported and as the raw gradient.

    ``project_translation`` defaults to ``"auto"``, so ``forces`` is the
    gradient projected onto the internal molecular directions; ``raw`` is the
    gradient of the discretized energy, which is what the finite-difference
    checks are about.
    """
    atoms = water()
    atoms.calc = water_calculator()
    forces = atoms.get_forces()
    return SimpleNamespace(atoms=atoms, forces=forces,
                           raw=atoms.calc.result.forces())


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
    # A real force, not a coincidence: 3.44 eV/Angstrom with the spectral
    # kinetic default (2026-10-13; 0.71 on the stencil with the exact spheres
    # of 2026-10-12, > 1 before them).
    assert abs(forces[0, 1]) > 0.5
    assert np.abs(forces[:, 0]).max() < 1e-3  # x is empty for this geometry


def test_p_valence_force_is_the_derivative_of_the_energy(h2o):
    """Central difference of the calculator's own energy on its frozen grid.

    Oxygen carries L = 1 and L = 2 compensation channels, so this exercises the
    whole multipole algebra; it was off by 0.57 eV/Angstrom (20 %) before the
    complex-weight fix.
    """
    # Oxygen along y: the component the complex M = +-1 channels carry, and the
    # one that was off by 0.57 eV/Angstrom.  One component keeps the test's
    # memory inside the suite budget; the symmetry test covers the rest.
    numerical = energy_derivative(h2o.atoms, 0, 1, 0.004)
    # Against the raw gradient: the finite difference is of the discretized
    # energy, which is not translation invariant, so it carries the egg-box that
    # the default projection removes from the reported force.
    assert h2o.raw[0, 1] == pytest.approx(numerical, abs=1e-2)


def test_translation_projection_zeroes_the_net_force(h2o):
    """The default (``"auto"``) enforces the free-molecule identities.

    The exact forces of a free molecule sum to zero and exert no torque, so
    projecting onto the internal directions takes out the translational and
    rotational components of the grid's egg-box and nothing physical.
    """
    forces, raw = h2o.forces, h2o.raw
    assert np.abs(forces.sum(axis=0)).max() < 1e-12
    positions = h2o.atoms.positions - h2o.atoms.positions.mean(axis=0)
    assert np.abs(np.cross(positions, forces).sum(axis=0)).max() < 1e-8
    # Projection is a rigid shift, so it cannot break the molecule's symmetry.
    assert forces[0, 1] == pytest.approx(forces[0, 2], abs=5e-3)
    # The reported force is the raw one with only the rigid translation (the
    # mean force) removed: this molecule's raw torque is already zero.
    assert np.allclose(raw - forces, raw.mean(axis=0), atol=1e-4)


def test_project_translation_false_reports_the_raw_gradient(h2o):
    """Opting out gives back exactly the gradient of the computed energy."""
    atoms = water()
    atoms.calc = water_calculator(project_translation=False)
    forces = atoms.get_forces()
    assert np.allclose(forces, h2o.raw, atol=1e-10)


def test_projection_cannot_increase_the_error_against_a_zero_sum_force(h2o):
    """Why the default is to project: it is an orthogonal projection.

    The exact force of a free molecule lies in the subspace of internal
    directions, and the projection onto it is orthogonal -- the removed part is
    orthogonal to what is kept -- so the projected array is no further from the
    exact force than the raw one, whatever the grid error happens to be.
    """
    raw, projected = h2o.raw, h2o.forces
    removed = raw - projected
    assert abs(np.sum(removed * projected)) < 1e-6 * max(np.sum(raw * raw), 1.0)
    assert np.linalg.norm(projected) <= np.linalg.norm(raw) + 1e-12
    # And it is a strict improvement whenever the residual is nonzero at all.
    assert np.abs(raw.sum(axis=0)).max() > 1e-6


def test_exact_spheres_force_is_the_derivative_of_the_energy():
    """The one-center Coulomb tensor of a Hartree-Fock Hamiltonian moves with
    the projections; its derivative is 1.3 eV/Angstrom of this force.

    HF (hydrogen fluoride) in SZ, the RHF determinant: orbital-stationary, so
    the gradient at fixed orbitals is the derivative of the RHF energy on the
    frozen grid.  The 2.5e-4 eV/Angstrom left is the step and the grid's, and
    the frozen-sphere Hamiltonian shows the same.
    """
    atoms = Atoms("FH", positions=[[3.013, 2.979, 3.007],
                                   [3.323, 2.979, 3.877]], cell=[6.0] * 3)
    grid = Grid(center=[3.0] * 3, cell=np.eye(3) * 6.0, grid_spacing=0.25)
    atoms.calc = Mandacaru(method="rhf", basis={"name": "PAW-LCAO", "size": "SZ"},
                           grid=grid, grid_spacing=0.25, project_translation=False,
                           scf_options={"tolerance": 1e-11}, trace=False)
    force = atoms.get_forces()[0, 2]
    numerical = energy_derivative(atoms, 0, 2, 0.002)
    assert abs(force) > 2.0
    assert force == pytest.approx(numerical, abs=1e-3)


def test_a_diffuse_basis_force_is_the_derivative_of_the_energy():
    """Diffuse shells are tabulated radial functions on their own grid; their
    Pulay terms take the same path as every other function's.  H2 in SZ plus
    one diffuse s shell per atom, in a cell that holds the shells, against a
    central difference on the calculator's frozen grid."""
    basis = {"name": "PAW-LCAO", "diffuse": True}
    atoms = dimer("H2", 0.75, 15.4)
    atoms.calc = calculator(grid_spacing=0.3, basis=basis)
    atoms.get_forces()
    raw = atoms.calc.result.forces()
    numerical = energy_derivative(atoms, 1, 2, 0.005)
    assert raw[1, 2] == pytest.approx(numerical, abs=5e-3)
