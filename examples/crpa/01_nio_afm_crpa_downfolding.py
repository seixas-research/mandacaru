"""Antiferromagnetic NiO downfolded to an active d-band / eg model with cRPA, then solved with ADAPT-VQE.

This clean workflow demonstrates:
1. Periodic spin-polarized DFT for an antiferromagnetic crystal.
2. Spin-resolved Maximally Localized Wannier Functions (MLWFs) with disentanglement.
3. Constrained RPA (cRPA) screening of on-site and inter-site Coulomb interactions.
4. Downfolding with double-counting subtraction into an effective Hamiltonian.
5. Variational quantum eigensolver (ADAPT-VQE) calculation of the correlated model.

The structure is modular and can be reused for other transition-metal oxides or correlated crystals.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
from mandacaru import Mandacaru
from mandacaru.units import HARTREE_TO_EV


def build_afm_nio(a: float = 4.17) -> Atoms:
    """Build a 4-atom cell of rocksalt NiO in the AFM-II magnetic configuration."""
    # Primitive FCC rocksalt NiO (1 Ni, 1 O)
    prim = bulk("NiO", "rocksalt", a=a)

    # Supercell matrix doubling along [111] (alternating ferromagnetic (111) sheets)
    M_afm2 = np.array([
        [1,  0, 0],
        [0,  1, 0],
        [-1, -1, 2],
    ])
    atoms = make_supercell(prim, M_afm2)

    # Alternating magnetic moments on Ni atoms (+2 on Ni0, -2 on Ni1)
    # Order of atoms in supercell: Ni, O, Ni, O
    atoms.set_initial_magnetic_moments([2.0, 0.0, -2.0, 0.0])
    return atoms


def run_dft(atoms: Atoms, kpts=(2, 2, 1), grid_spacing: float = 0.35) -> Mandacaru:
    """Perform self-consistent spin-polarized periodic DFT."""
    calc = Mandacaru(
        method="dft",
        xc="lda",
        basis={"name": "PAW-LCAO", "size": "SZ"},
        grid_spacing=grid_spacing,
        kpts={"size": kpts, "gamma": True},
        smearing={"method": "fermi-dirac", "width": 0.01},
        trace=False,
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    magmom = calc.get_total_magnetic_moment()
    print(f"DFT Total Energy: {energy:.4f} eV")
    print(f"Total Magnetic Moment: {magmom:.4f} mu_B")
    return calc


def compute_wannier(calc: Mandacaru, guess=None, outer_span: float = 5.0):
    """Compute spin-resolved Maximally Localized Wannier Functions."""
    if guess is None:
        guess = {"Ni": "d"}

    fermi = calc.get_fermi_level()
    windows = {"outer": (fermi - outer_span, fermi + outer_span)}

    print(f"\nComputing Wannier functions for {guess}...")
    wannier = calc.wannier(guess=guess, windows=windows)
    print(f"Total spread (Spin up):   {wannier.up.total_spread:.3f} A^2")
    print(f"Total spread (Spin down): {wannier.down.total_spread:.3f} A^2")
    return wannier


def downfold_and_screen(
    wannier,
    active_functions=None,
    num_particles=None,
    screening_cutoff: float = 2.0,
):
    """Compute bare and cRPA-screened Coulomb interactions and downfold the model."""
    print("\nEvaluating Coulomb interactions and cRPA screening...")
    # Bare interaction
    bare_model = wannier.downfold(
        functions=active_functions,
        num_particles=num_particles,
        coulomb="isolated",
    )

    # Constrained RPA screened interaction
    crpa_model = wannier.downfold(
        functions=active_functions,
        num_particles=num_particles,
        screening="crpa",
        screening_cutoff=screening_cutoff,
        coulomb="isolated",
    )

    # Print summary of screened parameters
    u_bare = bare_model.two_body[0, 1, 0, 0, 0, 0].real * HARTREE_TO_EV
    u_crpa = crpa_model.two_body[0, 1, 0, 0, 0, 0].real * HARTREE_TO_EV
    print(f"  Bare Hubbard U_0:  {u_bare:6.2f} eV")
    print(f"  cRPA Screened U:   {u_crpa:6.2f} eV  (Screening ratio: {u_crpa / u_bare:.3f})")
    if crpa_model.dielectric_head is not None:
        print(f"  Dielectric head eps^-1(q->0): {crpa_model.dielectric_head:.4f}")

    return bare_model, crpa_model


def solve_quantum_model(model, max_iterations: int = 5):
    """Solve the downfolded effective many-body problem with ADAPT-VQE."""
    print(f"\nSolving effective many-body model with ADAPT-VQE ({model.n_spatial_orbitals} spatial orbitals)...")
    options = model.as_quantum_problem()
    solver = Mandacaru(
        method="adapt-vqe",
        **options,
        max_iterations=max_iterations,
        trace=False,
    )
    result = solver.run()
    print(f"  Reference Hartree-Fock Energy: {result.reference_energy:.4f} eV")
    print(f"  ADAPT-VQE Ground State Energy: {result.optimal_energy:.4f} eV")
    print(f"  Correlation Energy:            {result.correlation_energy:.4f} eV")
    return result


def main():
    print("=== Mandacaru cRPA & Downfolding Workflow: AFM-II NiO ===")

    # 1. Geometry & Magnetic Configuration
    atoms = build_afm_nio()

    # 2. Self-consistent spin-polarized DFT
    calc = run_dft(atoms, kpts=(2, 2, 1), grid_spacing=0.35)

    # 3. Spin-resolved Wannier functions
    wannier = compute_wannier(calc, guess={"Ni": "d"}, outer_span=5.0)

    # 4. Downfolding & cRPA screening
    # For a low-energy 4-orbital model: select eg orbitals on both Ni sites (indices 2, 4 on Ni0 and 7, 9 on Ni1)
    eg_modes = [2, 4, 7, 9]
    _bare, crpa_model = downfold_and_screen(
        wannier,
        active_functions=eg_modes,
        num_particles=(2, 2),
        screening_cutoff=2.0,
    )

    # 5. Quantum simulation of the effective Hamiltonian
    solve_quantum_model(crpa_model, max_iterations=2)


if __name__ == "__main__":
    main()
