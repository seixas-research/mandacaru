"""H2O active space from local (DLPNO) MP2, without the two-electron tensor."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "DZP"},
                       h=0.25,
                       active_space={"orbitals": {"occupied": 3, "virtual": 3},
                                     "method": "dlpno-mp2"})

energy = atoms.get_potential_energy()
print(f"Energy: {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
