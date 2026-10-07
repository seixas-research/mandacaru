"""H2O with ADAPT-VQE in a frozen-core active space: O 1s frozen, 12 qubits."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2O")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="adapt-vqe",
                       basis="HAO",
                       h=0.30,
                       pool="fermionic",
                       active_space={"frozen": "auto"},
                       max_iterations=20,
                       convergence={"gradient": 1e-3})

energy = atoms.get_potential_energy()
result = atoms.calc.result
print(f"ADAPT-VQE energy: {energy:.6f} eV on {atoms.calc.n_qubits} qubits")
print(f"Correlation energy: {result.correlation_energy:.4f} eV "
      f"({result.num_operators} operators)")
