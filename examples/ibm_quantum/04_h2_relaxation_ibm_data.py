"""Replay the H2 relaxation whose energies and forces were measured on IBM hardware.

The trajectory in data/ is a paid hardware result: this script only reads it.
"""
import os
import numpy as np
from ase.io import read

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

for step, atoms in enumerate(read(os.path.join(DATA, "h2_relax_ibm.traj"), index=":")):
    fmax = np.abs(atoms.get_forces()).max()
    print(f"step {step}  d = {atoms.get_distance(0, 1):.4f} A  "
          f"E = {atoms.get_potential_energy():.6f} eV  fmax = {fmax:.4f} eV/A")
