"""ADAPT-VQE followed by a weak electric dipole echo for H2 in three dimensions.

Run with ``conda run -n mandacaru python examples/quantum_echoes.py``.
The 0.74 Angstrom bond points along (1, 1, 1). Hydrogen 1s orbitals and the
electronic Coulomb integrals are evaluated on a nonperiodic 3D Cartesian grid.
The minimal basis gives two spatial orbitals and four qubits; it demonstrates
bond-polarized response, not a basis-converged optical spectrum. Geometry and
grid settings are in Angstrom; dipole integrals and evolution use atomic units.
The standard ASE calculator workflow writes the ADAPT report, including
``[BASIS]``, to ``output.txt``; a ``[QUANTUM ECHOES]`` block follows with the
aligned echo/peak tables and the RHF HOMO-LUMO gap. The complete Fourier
spectrum is exported to ``spectrum.csv`` beside the report. To select the
field, pass e.g. ``--field-direction 1 0 0 --field-strength 0.01`` (atomic units).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from numpy.typing import ArrayLike, NDArray

from mandacaru import Mandacaru
from mandacaru.algorithms import QuantumEchoes, time_evolve
from mandacaru.core import (MolecularIntegrals, PauliSum, electric_dipole_potential,
                            electric_field_vector)
from mandacaru.integrals import Grid
from mandacaru.units import ANGSTROM_TO_BOHR, HARTREE_TO_EV
from mandacaru.utils import append_quantum_echoes


def build_h2_problem(txt: str | None = None) -> tuple[
    Atoms, MolecularIntegrals, PauliSum, NDArray[np.complex128]
]:
    """Run H2 through the standard ASE calculator and construct its dipoles.

    Return the geometry with its converged calculator, integral engine,
    MO-basis qubit Hamiltonian, and
    position matrices of shape (3, 2, 2) in Bohr. Both operators use the same
    molecular orbitals, including the AO overlap's Loewdin transformation.
    ``txt`` selects the standard report destination; None prints the report.
    """
    bond_length = 0.74  # Angstrom
    direction = np.ones(3) / np.sqrt(3)
    atoms = Atoms("H2", positions=[-0.5*bond_length*direction,
                                    0.5*bond_length*direction], pbc=False)
    grid = Grid(center=[0, 0, 0], box_size=4.0, h=0.1)
    atoms.calc = Mandacaru(
        method="adapt-vqe", basis={"name": "HAO"}, pool="fermionic",
        mapping="jordan_wigner", device="AER_simulator",
        grid=grid, h=grid.h, max_iterations=8, gradient_tolerance=1e-6,
        optimizer={"method": "BFGS", "maxiter": 100, "tol": 1e-12,
                   "options": {"gtol": 1e-8}},
        txt=txt,
    )
    atoms.get_total_energy()
    # Reuse the exact integral engine and MO rotation retained by the standard
    # builder. Rebuilding them separately could change the orbital phases.
    integrals = atoms.calc.solver._gradient_context["integrals"]
    hamiltonian = atoms.calc.hamiltonian

    # Integrate <chi_p|r_axis|chi_q> d^3r, including off-diagonal AO terms.
    # Grid coordinates and dV are already in Bohr and Bohr^3 internally.
    ao = np.stack([orbital.evaluate(grid.X, grid.Y, grid.Z).ravel()
                   for orbital in integrals.basis])
    r_ao = np.stack([(ao.conj() * coordinate.ravel()) @ ao.T * grid.dV
                     for coordinate in (grid.X, grid.Y, grid.Z)])

    # mo_coefficients rotates the Loewdin basis, not the original AO basis:
    # the complete AO -> MO coefficients are C = S^(-1/2) @ C_Loewdin.
    eigenvalues, eigenvectors = np.linalg.eigh(integrals.overlap())
    lowdin = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    coefficients = lowdin @ integrals.mo_coefficients
    r_mo = np.stack([coefficients.conj().T @ component @ coefficients
                     for component in r_ao]).astype(complex)
    return atoms, integrals, hamiltonian, r_mo


def main(txt: str = "output.txt", *, field_direction: ArrayLike | None = None,
         field_strength: float = 0.01) -> None:
    """Run H2 with a configurable unit field direction and atomic-unit strength.

    The default unit direction (1,1,1)/sqrt(3) follows the bond. Invalid field
    inputs are rejected before building the molecule. The orbital gap is from
    canonical RHF energies of this same Hamiltonian, not the FFT peak spacing.
    """
    direction = np.ones(3)/np.sqrt(3) if field_direction is None else field_direction
    field = electric_field_vector(direction, field_strength)
    atoms, integrals, _, position_integrals = build_h2_problem(txt=txt)
    calc = atoms.calc
    ground = calc.result
    # The integral tensors are cached; this reference SCF does not rebuild
    # the real-space integrals or alter the ADAPT state/MO rotation.
    reference = integrals.hartree_fock(sum(calc.num_particles))
    # A checkpoint holds the actual prepared state; ADAPTVQEResult holds
    # optimizer results and does not contain a state-vector attribute.
    prepared = calc.solver.checkpoint
    h_qubit = prepared.hamiltonian
    psi_t = time_evolve(prepared, h_qubit, time=0.5, steps=20)

    # Contract the user-selected polarization with all three dipole components.
    nuclear_dipole = (atoms.get_atomic_numbers() @ atoms.positions) * ANGSTROM_TO_BOHR
    potential = electric_dipole_potential(
        position_integrals, field_direction=direction, field_strength=field_strength,
        nuclear_dipole=nuclear_dipole,
        mapping=prepared.mapping, num_particles=prepared.num_particles,
    )
    order, steps, kick_steps = 2, 40, 1
    echoes = QuantumEchoes(h_qubit, potential, order=order)
    trajectory = [echoes.run(prepared, float(time), tau_p=0.01, steps=steps,
                             kick_steps=kick_steps)
                  for time in np.linspace(0, 4, 9)]
    # A longer trace is needed to resolve excitation energies. Propagate in
    # small increments so Trotter accuracy does not deteriorate at late times.
    spectrum_steps = 10
    spectrum = echoes.spectrum(prepared, time_step=0.5, num_samples=1024,
                                steps_per_sample=spectrum_steps)
    spectrum_path = Path(txt).with_name("spectrum.csv")
    for target in calc.solver.log_targets:
        append_quantum_echoes(target, trajectory, order=order, steps=steps,
                              kick_steps=kick_steps, field=field,
                              max_perturbation=echoes.max_perturbation,
                              spectrum=spectrum,
                              spectrum_steps_per_sample=spectrum_steps,
                              field_direction=direction, orbital_reference=reference,
                              spectrum_path=spectrum_path)
    print(f"Report with [BASIS] and [QUANTUM ECHOES]: {txt}")
    print(f"H2  {integrals.n_orbitals} orbitals ({calc.n_qubits} qubits), "
          f"E = {ground.in_units('eV'):.8f} eV, converged={ground.converged}")
    print(f"     {ground.num_operators} operators, {ground.metrics.cnot_count} CNOTs")
    print(f"HOMO-LUMO gap (RHF): {reference.homo_lumo_gap:.8f} Ha = "
          f"{reference.homo_lumo_gap*HARTREE_TO_EV:.6f} eV")
    print(f"Propagated norm: {np.linalg.norm(psi_t):.12f}")
    print(f"Quantum Echoes: {len(trajectory)} samples, "
          f"final fidelity={trajectory[-1].fidelity:.10f}")
    print(f"Fourier spectrum: {spectrum.times.size} samples, "
          f"energy-bin spacing={spectrum.resolution*HARTREE_TO_EV:.6f} eV")
    print(f"Spectrum CSV: {spectrum_path}")
    print("Excitation peaks (Fourier magnitude, not an absorption cross section):")
    for i in spectrum.peaks():
        print(f"    {spectrum.energies[i]:.6f} Ha = {spectrum.energies_ev[i]:.6f} eV, "
              f"magnitude={spectrum.intensities[i]:.6e} Ha")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output.txt", help="Main report path")
    parser.add_argument("--field-direction", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Cartesian unit vector; default is along the H2 bond")
    parser.add_argument("--field-strength", type=float, default=0.01,
                        help="Nonnegative field magnitude in atomic units (default: 0.01)")
    args = parser.parse_args()
    main(txt=args.output, field_direction=args.field_direction,
         field_strength=args.field_strength)
