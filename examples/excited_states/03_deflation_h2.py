"""Energy levels of H2 one after another with variational quantum deflation."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25)
atoms.get_potential_energy()

levels = atoms.calc.energy_levels(num_states=2, restarts=4)
for i, energy in enumerate(levels.energies):
    print(f"Level {i}: {energy:.6f} eV")
print(f"Excitation energy: {levels.excitation_energies[1]:.4f} eV")
