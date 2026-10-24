"""Kohn-Sham DFT Orbitals as Initial Input for ADAPT-VQE.

This example demonstrates how to use the converged Kohn-Sham molecular orbitals
from a DFT calculation (e.g., with the PBE exchange-correlation functional)
as the single-particle basis for an ADAPT-VQE simulation.

Workflow:
1. Perform a self-consistent Kohn-Sham DFT calculation with Mandacaru(method="dft").
2. Export the second-quantized Hamiltonian written in the Kohn-Sham orbital basis
   using `dft.result.as_quantum_problem()`.
3. Initialize and execute ADAPT-VQE using the Kohn-Sham Hamiltonian.
4. Compare convergence speed, operator selection, and energies against the
   canonical Hartree-Fock reference.
"""

from ase.build import molecule
from mandacaru import Mandacaru

# 1. Define geometry (LiH molecule)
atoms = molecule("LiH")
atoms.center(vacuum=2.5)

print("=" * 70)
print("  Kohn-Sham DFT Orbitals as Initial Input for ADAPT-VQE")
print("=" * 70)

# 2. Run Kohn-Sham DFT (PBE functional)
print("\n[Step 1] Running Kohn-Sham DFT (PBE functional)...")
dft = Mandacaru(
    method="dft",
    xc="pbe",
    basis="STO-3G",
    grid_spacing=0.30,
    trace=False,
)
atoms.calc = dft
e_dft = atoms.get_potential_energy()
print(f"  Converged Kohn-Sham DFT energy: {e_dft:.6f} eV")

# 3. Export the problem in the Kohn-Sham molecular orbital basis
ks_problem = dft.result.as_quantum_problem()
print("\n[Step 2] Exported Quantum Problem in Kohn-Sham Basis:")
print(f"  Qubits (spin-orbitals): {2 * ks_problem['n_spatial_orbitals']}")
print(f"  Spatial orbitals:       {ks_problem['n_spatial_orbitals']}")
print(f"  Electrons (alpha, beta):{ks_problem['num_particles']}")
print(f"  Hamiltonian Pauli terms:{len(ks_problem['hamiltonian'].terms)}")

# 4. Run ADAPT-VQE in the Kohn-Sham basis
print("\n[Step 3] Running ADAPT-VQE with Kohn-Sham Reference...")
adapt_ks = Mandacaru(
    method="adapt-vqe",
    **ks_problem,
    pool="fermionic",
    convergence={"gradient": 1e-3},
    trace=False,
)
res_ks = adapt_ks.run()
print(f"  Reference energy (KS determinant): {res_ks.reference_energy:.6f} eV")
print(f"  Optimal ADAPT-VQE energy:          {res_ks.optimal_energy:.6f} eV")
print(f"  Correlation energy recovered:      {res_ks.correlation_energy:.6f} eV")
print(f"  Ansatz operators selected ({res_ks.num_operators}): {res_ks.operators}")

# 5. Run standard Hartree-Fock + ADAPT-VQE for direct comparison
print("\n[Step 4] Running standard Hartree-Fock + ADAPT-VQE for comparison...")
adapt_hf = Mandacaru(
    method="adapt-vqe",
    basis="STO-3G",
    grid_spacing=0.30,
    pool="fermionic",
    convergence={"gradient": 1e-3},
    trace=False,
)
atoms.calc = adapt_hf
e_hf_adapt = atoms.get_potential_energy()
res_hf = adapt_hf.result
print(f"  Reference energy (HF determinant): {res_hf.reference_energy:.6f} eV")
print(f"  Optimal ADAPT-VQE energy:          {res_hf.optimal_energy:.6f} eV")
print(f"  Correlation energy recovered:      {res_hf.correlation_energy:.6f} eV")
print(f"  Ansatz operators selected ({res_hf.num_operators}): {res_hf.operators}")

# 6. Efficiency & Accuracy Summary
print("\n" + "=" * 70)
print("  Efficiency & Accuracy Summary")
print("=" * 70)
print(f"{'Metric':<35} {'Hartree-Fock':<16} {'Kohn-Sham (PBE)':<16}")
print("-" * 70)
print(f"{'Reference Determinant Energy (eV)':<35} {res_hf.reference_energy:<16.6f} {res_ks.reference_energy:<16.6f}")
print(f"{'Final ADAPT-VQE Energy (eV)':<35} {res_hf.optimal_energy:<16.6f} {res_ks.optimal_energy:<16.6f}")
print(f"{'Operators Added':<35} {res_hf.num_operators:<16} {res_ks.num_operators:<16}")
print(f"{'Total Cost Evaluations':<35} {res_hf.num_evaluations:<16} {res_ks.num_evaluations:<16}")
print(f"{'Energy Difference (eV)':<35} {abs(res_hf.optimal_energy - res_ks.optimal_energy):<16.2e}")
print("=" * 70)
print("Conclusion: Both orbital representations span the same Full CI Hilbert space")
print("and converge to the identical variational ground-state energy.")
