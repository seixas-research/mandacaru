"""Spectral function A(k, E) of a periodic hydrogen chain, saved as a PNG."""
import os
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

chain = Atoms("H", positions=[[0.0, 0.0, 0.0]],
              cell=[1.0, 10.0, 10.0], pbc=[True, False, False])
chain.calc = Mandacaru(method="bloch-adapt-vqe",
                       basis="HAO",
                       h=0.25,
                       kpts={"size": (4, 1, 1), "gamma": True})
chain.get_potential_energy()

spectral = chain.calc.get_spectral_function(energies=np.linspace(-35.0, 35.0, 2000),
                                            eta=0.15)
print(f"Chemical potential: {spectral.chemical_potential:+.4f} eV")

order = np.argsort(spectral.kpoints[:, 0])
k = spectral.kpoints[order, 0]
dk = k[1] - k[0]
plt.imshow(spectral.weights[order].T, origin="lower", aspect="auto", cmap="Blues",
           extent=[k[0] - dk / 2, k[-1] + dk / 2, spectral.energies[0], spectral.energies[-1]])
plt.axhline(spectral.chemical_potential, color="tab:red", ls="--")
plt.xlabel("k (fractional)")
plt.ylabel("E - E0 (eV)")
plt.savefig(os.path.join(DATA, "h_chain_spectral_function.png"), dpi=150)
