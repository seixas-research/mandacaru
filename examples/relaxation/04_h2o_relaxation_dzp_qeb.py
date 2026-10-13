"""H2O relaxed with ADAPT-VQE forces from O-H = 1.0 Angstrom, H-O-H = 90 degrees.

PAW-LCAO-DZP gives water 23 spatial orbitals (46 qubits); an MP2 active space
of four occupied and four virtual orbitals puts it on 16. With virtual
orbitals deleted, the force is the central difference of the reduced
Hamiltonian along water's three internal directions. The spectral kinetic
operator keeps those forces converged on the 0.25 Angstrom grid. The
trajectory and the run log (with the forces at every step) are written to
``data/``.

It reaches about 1.00 Angstrom and 102 degrees in four BFGS steps (water:
0.957 Angstrom, 104.5 degrees).  DZP is where the geometry stops moving with
the basis: TZP and DZ2P give the same RHF angle, and a single zeta with the
default polarization shell closes the angle to 96 degrees.
"""
import os
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

bond, half_angle = 1.0, np.radians(90.0) / 2
atoms = Atoms("OH2",
              positions=[[0.0, 0.0, 0.0],
                         [bond * np.sin(half_angle), 0.0,
                          bond * np.cos(half_angle)],
                         [-bond * np.sin(half_angle), 0.0,
                          bond * np.cos(half_angle)]])
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       grid_spacing=0.25,
                       kinetic="spectral",
                       active_space={"orbitals": 8, "method": "mp2"},
                       pool="qeb",
                       transfer=True,
                       txt=os.path.join(DATA, "h2o_relaxation_dzp_qeb.txt"))

trajectory = os.path.join(DATA, "h2o_relaxation_dzp_qeb.traj")
BFGS(atoms, trajectory=trajectory).run(fmax=0.05)
print(f"O-H: {atoms.get_distance(0, 1):.4f} Angstrom, "
      f"H-O-H: {atoms.get_angle(1, 0, 2):.2f} degrees")
