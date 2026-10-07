"""Ground and first excited state of H2 with subspace-search ADAPT-VQE."""
from ase.build import molecule
from mandacaru import Mandacaru

atoms = molecule("H2")
atoms.center(vacuum=3.0)
atoms.calc = Mandacaru(method="subspace-adapt-vqe",
                       basis={"name": "PAW-LCAO", "size": "SZ"},
                       h=0.25,
                       pool="fermionic",
                       num_states=2,
                       convergence={"gradient": 1e-4})

atoms.get_potential_energy()
result = atoms.calc.result
for i, energy in enumerate(result.energies):
    print(f"State {i}: {energy:.6f} eV")
print(f"Excitation energy: {result.excitation_energies[1]:.4f} eV "
      f"({result.num_operators} operators)")
