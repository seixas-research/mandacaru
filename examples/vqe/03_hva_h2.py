"""H2 with the Hamiltonian variational ansatz: exact groups and a Trotter circuit."""
from ase import Atoms
from mandacaru import Mandacaru

atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="rhf", basis="HAO", h=0.35)
print(f"RHF          {atoms.get_potential_energy():.8f} eV")
problem = atoms.calc.result.as_quantum_problem()

for label, ansatz in (("exact groups", {"name": "hva", "layers": 2}),
                      ("Trotter", {"name": "hva", "layers": 2,
                                   "evolution": "trotter", "order": 2, "steps": 2})):
    result = Mandacaru(method="vqe", ansatz=ansatz, **problem).run()
    print(f"{label:12s} {result.optimal_energy:.8f} eV")
