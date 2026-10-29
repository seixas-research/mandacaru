"""Ethanol relaxed with ADAPT-VQE forces from a Gaussian-deformed geometry.

Every Cartesian coordinate of the equilibrium structure is displaced by a
normal deviate (standard deviation 0.1 Angstrom, fixed seed), and BFGS walks
the molecule back. With virtual orbitals deleted, the force is the central
difference of the reduced Hamiltonian along the 21 internal directions; the
two-electron integrals are Cholesky factors, and every displaced geometry is
factored on the reference's pivots. Those 42 rebuilds per step are why the
basis is DZP on a 0.30 Angstrom grid.
The deformed start, the trajectory and the run log (with the forces on every
atom at every step) are written to ``data/``.
"""
import os
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = Atoms("CCOHHHHHH",
              positions=[[8.114804, 5.636852, 5.886933],
                         [6.946623, 6.596696, 5.886933],
                         [5.756540, 5.809565, 5.886933],
                         [5.000000, 6.418759, 5.886933],
                         [6.989180, 7.244742, 6.773866],
                         [6.989180, 7.244742, 5.000000],
                         [9.062514, 6.182034, 5.886933],
                         [8.075222, 5.000000, 6.772814],
                         [8.075222, 5.000000, 5.001052]],
              cell=[14.062514, 12.244742, 11.773866])
atoms.rattle(stdev=0.1, seed=7)
write(os.path.join(DATA, "ethanol_deformed.xyz"), atoms)

atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZP"},
                       grid_spacing=0.30,
                       two_body={"method": "cholesky", "tolerance": 1e-5},
                       active_space={"orbitals": {"occupied": 4, "virtual": 4},
                                     "method": "mp2"},
                       pool="ceo",
                       max_iterations=60,
                       convergence={"gradient": 1e-3},
                       transfer=True,
                       txt=os.path.join(DATA, "ethanol_relaxation.txt"))

trajectory = os.path.join(DATA, "ethanol_relaxation.traj")
optimizer = BFGS(atoms, trajectory=trajectory)
optimizer.run(fmax=0.05)
atoms.calc.write_optimization_summary(optimizer=optimizer)
print(f"C-C: {atoms.get_distance(0, 1):.4f}  C-O: {atoms.get_distance(1, 2):.4f}  "
      f"O-H: {atoms.get_distance(2, 3):.4f} Angstrom, "
      f"C-O-H: {atoms.get_angle(1, 2, 3):.2f} degrees")
