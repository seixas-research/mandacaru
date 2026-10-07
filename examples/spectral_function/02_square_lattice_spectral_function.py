"""A square lattice of H: symmetry, a band path and A(k, E) along it, saved as a PNG."""
import os
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

lattice = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                cell=[2.6, 2.6, 9.0], pbc=[True, True, False])
lattice.calc = Mandacaru(method="bloch-adapt-vqe",
                         basis="HAO",
                         h=0.35,
                         kpts={"size": (2, 2, 1), "gamma": True},
                         optimizer={"method": "SLSQP", "maxiter": 1000, "tol": 1e-12})
print(lattice.calc.symmetry().summary())
print(lattice.calc.irreducible_zone().summary())
print(f"Energy per cell: {lattice.get_potential_energy():.6f} eV")

spectral = lattice.calc.get_spectral_function(energies=np.linspace(-20.0, 20.0, 700),
                                              eta=0.3, path="GXMG")
plt.pcolormesh(spectral.kpath.distances, spectral.energies, spectral.weights.T,
               cmap="Blues", shading="nearest")
plt.axhline(spectral.chemical_potential, color="tab:red", ls="--")
plt.xticks(spectral.kpath.label_distances,
           [r"$\Gamma$" if label == "G" else label for label in spectral.kpath.labels])
plt.ylabel("E - E0 (eV)")
plt.savefig(os.path.join(DATA, "square_lattice_spectral_function.png"), dpi=150)
