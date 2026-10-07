"""H2 potential energy curve with ADAPT-VQE, saved as a PNG."""
import os
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

distances = np.arange(0.5, 3.01, 0.25)
energies = []
for distance in distances:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    atoms.center(vacuum=3.0)
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.25,
                           pool="qeb")
    energies.append(atoms.get_potential_energy())
    print(f"d = {distance:.2f} A  E = {energies[-1]:.6f} eV")

plt.plot(distances, energies, "o-")
plt.xlabel("H-H distance (Angstrom)")
plt.ylabel("Energy (eV)")
plt.savefig(os.path.join(DATA, "h2_dissociation.png"), dpi=150)
