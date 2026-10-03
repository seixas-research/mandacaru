"""Density of states of H2O from its Kohn-Sham eigenvalues, saved as a PNG."""
import os
import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.build import molecule
from mandacaru import Mandacaru

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="dft",
                       xc="lda",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.2)
atoms.get_potential_energy()

scf = atoms.calc.result.scf
levels = scf.mo_energies * units.Hartree
homo, lumo = levels[scf.n_occupied - 1], levels[scf.n_occupied]
print(f"HOMO {homo:.3f} eV, LUMO {lumo:.3f} eV, gap {lumo - homo:.3f} eV")

sigma = 0.3
energy = np.linspace(levels[0] - 2.0, homo + 20.0, 2000)
dos = 2.0 * np.exp(-0.5 * ((energy[:, None] - levels) / sigma) ** 2).sum(axis=1) \
    / (sigma * np.sqrt(2.0 * np.pi))

plt.plot(energy, dos)
plt.axvline(homo, color="gray", ls="--")
plt.xlabel("Energy (eV)")
plt.ylabel("DOS (states/eV)")
plt.savefig(os.path.join(DATA, "h2o_dos.png"), dpi=150)
