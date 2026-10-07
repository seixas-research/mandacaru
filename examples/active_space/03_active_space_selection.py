"""LiH in PAW-LCAO TZP: which virtual orbitals to keep, by energy, by MP2 or by threshold."""
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("LiH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.60]])
atoms.center(vacuum=4.0)

for active_space in ({"orbitals": 4, "method": "energy"},
                     {"orbitals": 4, "method": "mp2"},
                     {"threshold": 1e-4, "method": "mp2"}):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "TZP"},
                           h=0.25,
                           active_space=active_space)
    energy = atoms.get_potential_energy()
    print(f"{str(active_space):40s} {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
