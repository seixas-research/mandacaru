"""H2 in all-electron numerical atomic orbitals (NAO-AE): each tier adds functions."""
from ase.build import molecule
from mandacaru import Mandacaru
from mandacaru.algorithms import estimate_qubits

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for tier in (0, 1, 2):
    basis = {"name": "NAO-AE", "tier": tier, "onset": 2.5, "width": 1.0}
    qubits = estimate_qubits(atoms, basis=basis).n_qubits
    atoms.calc = Mandacaru(method="rhf", basis=basis, h=0.20)
    energy = atoms.get_potential_energy()
    print(f"tier {tier}  E_RHF = {energy:.4f} eV  ({qubits} qubits for a quantum solver)")
