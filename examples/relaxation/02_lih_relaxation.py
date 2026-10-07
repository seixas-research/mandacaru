"""LiH relaxed from 2.0 Angstrom with ADAPT-VQE forces, the trajectory saved."""
import os
from ase import Atoms
from ase.optimize import BFGS
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
atoms.center(vacuum=4.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.10,
                       pool="ceo",
                       optimizer={"method": "L-BFGS", "maxiter": 2000, "tol": 1e-12},
                       convergence={"gradient": 1e-5})

BFGS(atoms, trajectory=os.path.join(DATA, "lih_relaxation.traj")).run(fmax=0.02)
print(f"Li-H: {atoms.get_distance(0, 1):.4f} Angstrom, "
      f"E = {atoms.get_potential_energy():.6f} eV")
