"""Ground and first excited state of H2 in one subspace-search VQE run."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="subspace-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25,
                       num_states=2)

atoms.get_potential_energy()
for i, energy in enumerate(atoms.calc.result.energies):
    print(f"State {i}: {energy:.6f} eV")
