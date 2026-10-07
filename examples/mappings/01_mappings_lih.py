"""LiH with ADAPT-VQE under each fermion-to-qubit mapping: same energy, different circuits."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("LiH")
atoms.center(vacuum=3.0)

for mapping in ("jordan_wigner", "parity", "parity_reduced", "bravyi_kitaev"):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.25,
                           pool="fermionic",
                           mapping=mapping)
    energy = atoms.get_potential_energy()
    print(f"{mapping:15s} E = {energy:.6f} eV on {atoms.calc.n_qubits} qubits  "
          f"({atoms.calc.result.metrics.cnot_count} CNOTs)")
