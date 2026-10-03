"""H2 with ADAPT-VQE, the ansatz executed as circuits on three SDKs' simulators."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)

for provider in ("qiskit", "braket", "cirq"):
    atoms.calc = Mandacaru(method="adapt-vqe",
                           basis={"name": "PAW-LCAO", "size": "SZ"},
                           h=0.25,
                           backend_provider=provider,
                           execute_circuits=True)
    energy = atoms.get_potential_energy()
    print(f"{provider:7s} {energy:.8f} eV")
