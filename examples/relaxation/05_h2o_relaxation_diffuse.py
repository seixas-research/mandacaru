"""H2O relaxed with diffuse PAW-LCAO-DZP orbitals: a weaker confinement.

The same relaxation as ``04_h2o_relaxation_dzp_qeb.py`` with ``energy_shift``
lowered from the default 0.1 eV to 0.01 eV.  The confinement radius grows as
``1/sqrt(energy_shift)``, so every orbital reaches further (hydrogen 1s from
6.7 to 9.0 Bohr) and the box gets 5 Angstrom of vacuum instead of 3 to hold
the tails.  The trajectory and the run log, whose ``[BASIS]`` block lists
each orbital's cutoff radius, are written to ``data/``.
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
atoms.center(vacuum=5.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP",
                              "energy_shift": 0.01},
                       grid_spacing=0.25,
                       active_space={"orbitals": 8, "method": "mp2"},
                       pool="qeb",
                       transfer=True,
                       txt=os.path.join(DATA, "h2o_relaxation_diffuse.txt"))

trajectory = os.path.join(DATA, "h2o_relaxation_diffuse.traj")
BFGS(atoms, trajectory=trajectory).run(fmax=0.05)
print(f"O-H: {atoms.get_distance(0, 1):.4f} Angstrom, "
      f"H-O-H: {atoms.get_angle(1, 0, 2):.2f} degrees")
