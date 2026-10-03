"""Ground state of LiH with VQE and the UCCSD ansatz."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25)

energy = atoms.get_potential_energy()
print(f"VQE energy: {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
