"""ADAPT-VQE of LiH with each operator pool."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)

for pool in ("fermionic", "qubit", "qeb", "ceo", "ceo-ovp", "spin-orbit"):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "DZ"},
                           h=0.25,
                           active_space={"orbitals": 4, "method": "mp2"},
                           pool=pool)
    energy = atoms.get_potential_energy()
    print(f"{pool:10s} E = {energy:.6f} eV  "
          f"({atoms.calc.result.num_operators} operators)")
