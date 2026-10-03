"""How the basis confinement (energy_shift) changes the energy of H2O."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)

for energy_shift in (0.3, 0.1, 0.03, 0.01):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ",
                                  "energy_shift": energy_shift},
                           h=0.25)
    energy = atoms.get_potential_energy()
    print(f"energy_shift = {energy_shift:5.2f} eV  E = {energy:.6f} eV")
