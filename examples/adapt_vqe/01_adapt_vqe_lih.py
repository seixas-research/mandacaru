"""Ground state of LiH with ADAPT-VQE: the ansatz grows one operator at a time."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25,
                       pool="qeb",
                       convergence={"gradient": 1e-3})

energy = atoms.get_potential_energy()
result = atoms.calc.result
print(f"ADAPT-VQE energy: {energy:.6f} eV")
print(f"Operators ({len(result.operators)}): {result.operators}")
