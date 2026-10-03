"""H2O in a large basis on a small register: an MP2 natural-orbital active space."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.25,
                       active_space={"orbitals": 6, "method": "mp2",
                                     "symmetry": True})

energy = atoms.get_potential_energy()
print(f"Energy: {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
