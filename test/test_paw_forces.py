# -*- coding: utf-8 -*-
# file: test/test_paw_forces.py

# This code is part of Carcará.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Hellmann-Feynman + Pulay forces with PAW pseudopotentials and a DZP basis.

H2 and LiH in ``basis={"name": "PAW", "size": "DZP"}`` have 10 basis
functions, i.e. 20 qubits, solved exactly in the (1, 1) particle-number sector.

The VASP numbers are a *qualitative* guide only: VASP is PBE-DFT in plane
waves, Carcará is ADAPT-VQE in a localized basis, so the curves differ in
detail (Carcará's H2 minimum is ~0.79 A against VASP's 0.750 A).
"""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from carcara import Carcara

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
    return Carcara(method="adapt-vqe", basis=BASIS, h=h, pool="fermionic",
                   optimizer="L-BFGS-B", max_iterations=80,
                   gradient_tolerance=1e-5, profile=False, verbose=False)


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
    assert forces[1, 2] == pytest.approx(numerical, abs=5e-3)


def test_hellmann_feynman_and_pulay(h2):
    forces, (hf, pulay) = h2.forces, h2.breakdown
    assert np.allclose(forces, -(hf + pulay), atol=1e-12)
    assert np.abs(pulay).max() > 0.05             # an atom-centered basis needs it
    assert np.abs(forces.sum(axis=0)).max() < 0.15      # grid egg-box only
    assert np.abs(forces[:, :2]).max() < 0.1
    assert bond_force(forces) < -1.0               # 0.75 A is inside the minimum


def test_h2_force_curve_follows_vasp():
    for distance, reference in VASP_H2.items():
        atoms = dimer("H2", distance, 8.0)
        atoms.calc = calculator(h=0.25)
        force = bond_force(atoms.get_forces())
        assert np.sign(force) == np.sign(reference), distance
        if abs(reference) > 2.0:
            assert force == pytest.approx(reference, rel=0.5), distance


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
    from carcara.backends.providers import QiskitProvider

    def run(**options):
        atoms = dimer("H2", 1.0, 8.0)
        atoms.calc = Carcara(method="adapt-vqe", basis="PAW", h=0.25,
                             pool="fermionic", optimizer="L-BFGS-B",
                             max_iterations=10, gradient_tolerance=1e-6,
                             profile=False, verbose=False, **options)
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
