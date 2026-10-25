"""Silicon diamond unit cell relaxation with Bloch-ADAPT-VQE and PAW-LCAO.

Relaxes the periodic unit cell (minimizing Voigt stress and forces) using
Bloch-ADAPT-VQE, PAW-LCAO with active space (10 qubits <= 18 qubits),
matrix-free Cholesky decomposition, and ASE's ExpCellFilter with BFGS.
"""
from ase.build import bulk
from ase.filters import ExpCellFilter
from ase.optimize import BFGS
import numpy as np
from mandacaru import Mandacaru

# 1. Silicon diamond structure (2 atoms in primitive cell)
# Initial lattice parameter a = 5.50 Angstrom (strained away from equilibrium)
si = bulk("Si", "diamond", a=5.50)
initial_volume = si.get_volume()
initial_cell = np.array(si.get_cell())
print(f"Initial lattice parameter: a = 5.5000 Angstrom | Volume = {initial_volume:.4f} Angstrom^3")

# 2. Attach Bloch-ADAPT-VQE calculator with PAW-LCAO and Cholesky
# Full valence space has 8 spatial orbitals = 16 qubits (<= 18 qubits).
calc = Mandacaru(
    method="bloch-adapt-vqe",
    basis="PAW-LCAO",
    two_body="cholesky",
    kpts=(1, 1, 1),
    max_iterations=2,
)
si.calc = calc

# Initial forces and stress
forces0 = si.get_forces()
stress0 = si.get_stress(voigt=True)
print(f"Initial max force: {np.max(np.abs(forces0)):.4e} eV/Angstrom")
print(f"Initial Voigt stress (xx, yy, zz, yz, xz, xy):")
print(f"  {stress0.round(4)} eV/Angstrom^3")

# 3. Unit cell relaxation minimizing forces and stress
# ExpCellFilter with hydrostatic_strain=True preserves diamond cubic symmetry.
ucf = ExpCellFilter(si, hydrostatic_strain=True)
dyn = BFGS(ucf, logfile="-")

print("\nStarting unit cell optimization with BFGS + ExpCellFilter...")
dyn.run(fmax=0.05, steps=3)

final_volume = si.get_volume()
final_cell = np.array(si.get_cell())
final_forces = si.get_forces()
final_stress = si.get_stress(voigt=True)

print("\n--- Relaxation Summary ---")
print(f"Initial Volume: {initial_volume:.4f} Angstrom^3 -> Final Volume: {final_volume:.4f} Angstrom^3")
print(f"Final max force: {np.max(np.abs(final_forces)):.4e} eV/Angstrom")
print(f"Final Voigt stress:")
print(f"  {final_stress.round(4)} eV/Angstrom^3")
print("Unit cell relaxation completed successfully.")
