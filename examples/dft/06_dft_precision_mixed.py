"""Mixed precision (single vs double) in Kohn-Sham DFT.

Using precision="single" stores sampled molecular orbitals in float32, halving
their retained storage. Integral accumulation, Hamiltonian matrices and FFT
workspaces use double precision. This compares the resulting SCF energies.
"""
import time
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.5)

print("Kohn-Sham DFT (PBE/DZP) with precision comparison:")
print("-" * 65)
print(f"{'Precision':<10} {'Energy (eV)':<16} {'Time (s)':<10} {'SCF Iters':<10}")
print("-" * 65)

results = {}
for prec in ("double", "single"):
    t0 = time.perf_counter()
    atoms.calc = Mandacaru(
        method="dft",
        xc="pbe",
        basis={"name": "PAW-LCAO", "size": "DZP"},
        grid_spacing=0.2,
        precision=prec,
        trace=False,
    )
    energy = atoms.get_potential_energy()
    dt = time.perf_counter() - t0
    scf = atoms.calc.result.scf
    results[prec] = energy
    print(f"{prec:<10} {energy:<16.6f} {dt:<10.2f} {scf.n_iterations:<10}")

print("-" * 65)
delta_e = abs(results["single"] - results["double"])
print(f"Energy difference: {delta_e * 1e3:.4f} meV ({delta_e:.2e} eV)")
